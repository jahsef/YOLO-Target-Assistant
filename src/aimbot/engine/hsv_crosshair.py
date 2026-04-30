import cupy as cp
import numpy as np


_RED_MASK_KERNEL = cp.RawKernel(r"""
extern "C" __global__
void rgb_red_mask(const unsigned char* __restrict__ rgb,
                  unsigned char* __restrict__ mask,
                  const int n_pixels,
                  const int color_range_half,
                  const int s_min,
                  const int v_min,
                  const int s_max,
                  const int v_max) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_pixels) return;

    int base = i * 3;
    unsigned char r = rgb[base + 0];
    unsigned char g = rgb[base + 1];
    unsigned char b = rgb[base + 2];

    unsigned char mx = r > g ? r : g; if (b > mx) mx = b;
    unsigned char mn = r < g ? r : g; if (b < mn) mn = b;

    int v = mx;
    if (v < v_min || v > v_max) { mask[i] = 0; return; }

    int diff = mx - mn;
    int s = (mx == 0) ? 0 : (diff * 255) / mx;
    if (s < s_min || s > s_max) { mask[i] = 0; return; }
    if (diff == 0)               { mask[i] = 0; return; }

    // OpenCV hue scale: degrees / 2 -> [0, 180).
    // 30 == 60/2 (factor that absorbs the /2 into the numerator).
    int h;
    if (mx == r)       h = (30 * (g - b)) / diff;
    else if (mx == g)  h = 60 + (30 * (b - r)) / diff;
    else               h = 120 + (30 * (r - g)) / diff;
    if (h < 0) h += 180;

    bool red = (h <= color_range_half) || (h >= 179 - color_range_half);
    mask[i] = red ? 1 : 0;
}
""", "rgb_red_mask")


def cupy_red_mask(rgb_gpu: cp.ndarray, color_range: int,
                  s_min: int, v_min: int,
                  s_max: int = 255, v_max: int = 255) -> cp.ndarray:
    """rgb_gpu: (H, W, 3) cp.uint8. Returns (H, W) cp.bool_."""
    assert rgb_gpu.dtype == cp.uint8 and rgb_gpu.ndim == 3 and rgb_gpu.shape[2] == 3
    h, w, _ = rgb_gpu.shape
    n = h * w
    mask_u8 = cp.empty((h, w), dtype=cp.uint8)

    rgb_contig = cp.ascontiguousarray(rgb_gpu)

    threads = 256
    blocks = (n + threads - 1) // threads
    _RED_MASK_KERNEL(
        (blocks,), (threads,),
        (rgb_contig, mask_u8, np.int32(n),
         np.int32(color_range // 2),
         np.int32(s_min), np.int32(v_min),
         np.int32(s_max), np.int32(v_max))
    )
    return mask_u8.view(cp.bool_)


# Hardcoded HSV thresholds tuned empirically for this game's red crosshair
# via src/hsv_testing/fart.py visual sweep.
_HSV_COLOR_RANGE = 9
_HSV_S_MIN = 120
_HSV_S_MAX = 245
_HSV_V_MIN = 120
_HSV_V_MAX = 245
_HSV_BOX_SIZE = 64  # synthetic detection box side, in base-region pixels
# 64x64 gives better iou for jittering crosshair detections so better tracker stickiness.

# Sigma chosen so corner weight ≈ 0.25 * center weight:
# exp(-d²/(2σ²)) = 0.25 at d = roi_half → σ = roi_half / sqrt(2*ln(4)) ≈ roi_half / 1.665
_GAUSS_EDGE_FACTOR = 1.665  # sqrt(2 * ln(4))


class HSVCrosshairDetector:
    """
    Red-crosshair detector backed by the fused HSV kernel + a configurable
    voting scheme. Owns any pre-allocated buffers the chosen scheme needs so
    the hot path never allocates.

    RED CROSSHAIRS ONLY. Other colors / shapes are not supported yet --
    intentionally narrow because the only thing we need from this path right
    now is locating the game's red reticle. Generalize the kernel rather than
    special-casing here if you want green/blue/cyan later.

    Voting schemes:
      "simple":          plain mean of mask-true coords. cheapest, can drift if
                         a few stray red pixels exist outside the actual crosshair.
      "weighted_center": Gaussian-center-weighted mean. center pixels pull harder
                         than edge pixels (corner weight ≈ 0.25 * center). cheap,
                         robust to peripheral red, slight bias toward ROI center.
      "connected":       largest connected red component centroid. most robust
                         (e.g. survives red enemy uniforms leaking into ROI), but
                         pays for cupyx.scipy.ndimage.label which is the slowest path.
    """

    VOTING_SCHEMES = ("simple", "weighted_center", "connected")

    def __init__(
        self,
        voting_scheme: str,
        crosshair_cls_id: int,
        frame_hw: tuple[int, int],
        center_crop_hw: tuple[int, int] | None = None,
    ):
        if voting_scheme not in self.VOTING_SCHEMES:
            raise ValueError(
                f"Invalid voting_scheme: {voting_scheme!r}. Must be one of {list(self.VOTING_SCHEMES)}."
            )
        self.voting_scheme = voting_scheme
        self.crosshair_cls_id = int(crosshair_cls_id)
        self.frame_h, self.frame_w = int(frame_hw[0]), int(frame_hw[1])
        self.center_crop_hw = (int(center_crop_hw[0]), int(center_crop_hw[1])) if center_crop_hw else None

        # ROI shape (post crop) determines any per-shape buffer sizes.
        self.roi_h, self.roi_w = self.center_crop_hw if self.center_crop_hw else (self.frame_h, self.frame_w)

        # Cache of crop origin so detect() doesn't recompute it every frame.
        if self.center_crop_hw:
            self._y0 = (self.frame_h - self.roi_h) // 2
            self._x0 = (self.frame_w - self.roi_w) // 2
        else:
            self._y0 = 0
            self._x0 = 0

        # Bind the chosen vote method directly so detect() doesn't dispatch on a string each frame.
        vote_fn_map = {
            "simple": self._vote_simple,
            "weighted_center": self._vote_weighted_center,
            "connected": self._vote_connected,
        }
        self._vote = vote_fn_map[voting_scheme]

        # Scheme-specific buffer init.
        if voting_scheme == "weighted_center":
            self._init_weighted_center_buffers()

    # --- buffer init ----------------------------------------------------------

    def _init_weighted_center_buffers(self):
        h, w = self.roi_h, self.roi_w
        ys = cp.arange(h, dtype=cp.float32).reshape(h, 1)
        xs = cp.arange(w, dtype=cp.float32).reshape(1, w)
        cy = (h - 1) * 0.5
        cx = (w - 1) * 0.5
        sigma_y = (h * 0.5) / _GAUSS_EDGE_FACTOR
        sigma_x = (w * 0.5) / _GAUSS_EDGE_FACTOR
        weights = cp.exp(
            -(((ys - cy) ** 2) / (2.0 * sigma_y * sigma_y)
              + ((xs - cx) ** 2) / (2.0 * sigma_x * sigma_x))
        ).astype(cp.float32)
        self._w_ys = ys
        self._w_xs = xs
        self._w_weights = weights
        self._w_wm = cp.empty((h, w), dtype=cp.float32)  # scratch for weights*mask

    # --- voting ---------------------------------------------------------------

    def _vote_simple(self, mask: cp.ndarray):
        ys, xs = cp.where(mask)
        if ys.size == 0:
            return None
        return float(ys.mean()), float(xs.mean())

    def _vote_weighted_center(self, mask: cp.ndarray):
        cp.multiply(self._w_weights, mask, out=self._w_wm)  # mask broadcasts bool->0/1
        total = float(self._w_wm.sum())
        if total <= 0.0:
            return None
        cy = float((self._w_wm * self._w_ys).sum()) / total
        cx = float((self._w_wm * self._w_xs).sum()) / total
        return cy, cx

    def _vote_connected(self, mask: cp.ndarray):
        from cupyx.scipy.ndimage import label  # lazy import; only loaded if scheme used
        labels, n = label(mask)
        if n == 0:
            return None
        counts = cp.bincount(labels.ravel())
        counts[0] = 0  # background
        largest = int(cp.argmax(counts))
        ys, xs = cp.where(labels == largest)
        if ys.size == 0:
            return None
        return float(ys.mean()), float(xs.mean())

    # --- public entry point ---------------------------------------------------

    def detect(self, rgb_frame_gpu: cp.ndarray) -> np.ndarray:
        """
        rgb_frame_gpu: (H, W, 3) cp.uint8 RGB (raw frame from betterercam),
            shape must match `frame_hw` from __init__.

        Returns (1, 6) np.float32 [x1, y1, x2, y2, conf, cls] in base-region xyxy
        coords if a centroid is found, else (0, 6) empty array.
        """
        if self.center_crop_hw:
            roi = rgb_frame_gpu[self._y0:self._y0 + self.roi_h, self._x0:self._x0 + self.roi_w]
        else:
            roi = rgb_frame_gpu

        mask = cupy_red_mask(roi, _HSV_COLOR_RANGE, s_min=_HSV_S_MIN, s_max=_HSV_S_MAX,
                             v_min=_HSV_V_MIN, v_max=_HSV_V_MAX)
        vote = self._vote(mask)
        if vote is None:
            return np.empty((0, 6), dtype=np.float32)

        cy_local, cx_local = vote
        cy = cy_local + self._y0
        cx = cx_local + self._x0
        half = _HSV_BOX_SIZE / 2
        return np.array([[
            cx - half, cy - half, cx + half, cy + half,
            1.0, float(self.crosshair_cls_id)
        ]], dtype=np.float32)
