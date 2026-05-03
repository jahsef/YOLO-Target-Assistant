import cupy as cp
import numpy as np
import torch
import torch.nn as nn

_RED_MASK_KERNEL = cp.RawKernel(r"""
extern "C" __global__
void rgb_red_mask(const unsigned char* __restrict__ rgb,
                  unsigned char* __restrict__ mask,
                  const int n_pixels,
                  const int color_center,
                  const int color_range,
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

    int d = h - color_center;
    if (d < 0) d = -d;
    if (d > 90) d = 180 - d;
    mask[i] = (2 * d <= color_range) ? 1 : 0;
}
""", "rgb_red_mask")


def cupy_red_mask(rgb_gpu: cp.ndarray, color_center: int, color_range: int,
                  s_min: int, v_min: int,
                  s_max: int = 255, v_max: int = 255) -> cp.ndarray:
    """rgb_gpu: (B, H, W, 3) cp.uint8. Returns (B, H, W) cp.bool_.

    if batch dimension  doesnt exist, we convert input to have batch dimension , but return without a batch dimension
    
    Hue band is color_center +/- color_range/2 on OpenCV's [0, 180) scale,
    with wraparound.
    """
    is_chw = False
    if rgb_gpu.ndim == 3:
        rgb_gpu = rgb_gpu[cp.newaxis, ...]
        is_chw = True
    
    assert rgb_gpu.dtype == cp.uint8 and rgb_gpu.ndim == 4 and rgb_gpu.shape[3] == 3
    b, h, w, _ = rgb_gpu.shape
    n = b * h * w
    mask_u8 = cp.empty((b, h, w), dtype=cp.uint8)

    rgb_contig = cp.ascontiguousarray(rgb_gpu)

    threads = 256
    blocks = (n + threads - 1) // threads
    _RED_MASK_KERNEL(
        (blocks,), (threads,),
        (rgb_contig, mask_u8, np.int32(n),
         np.int32(color_center), np.int32(color_range),
         np.int32(s_min), np.int32(v_min),
         np.int32(s_max), np.int32(v_max))
    )
    
    return mask_u8.view(cp.bool_)[0, ...] if is_chw else mask_u8.view(cp.bool_)

# Hardcoded HSV thresholds tuned empirically for this game's red crosshair
# via src/hsv_testing/fart.py visual sweep.
_HSV_COLOR_CENTER = 6
_HSV_COLOR_RANGE = 12
_HSV_S_MIN = 170
_HSV_S_MAX = 255
_HSV_V_MIN = 150
_HSV_V_MAX = 245
_HSV_BOX_SIZE = 64  # synthetic detection box side, in base-region pixels
# 64x64 gives better iou for jittering crosshair detections so better tracker stickiness.

# Sigma chosen so corner weight ≈ 0.25 * center weight:
# exp(-d²/(2σ²)) = 0.25 at d = roi_half → σ = roi_half / sqrt(2*ln(4)) ≈ roi_half / 1.665
_GAUSS_EDGE_FACTOR = 1.665  # sqrt(2 * ln(4))

# heuristic_spam args (prototype-tuned in src/tests/fart_avgpool.py).
# row-suppression: kill mask pixels where row_sum[y] > _HS_RATIO_THRESHOLD * col_sum[x],
# applied pre-opening so opening sees nametag-free input.
_HS_RATIO_THRESHOLD = 1.5
# top-K argmax patches considered for cluster-spread analysis.
_HS_TOP_K = 48
# scatter threshold in sqrt(input pixel) units; mean sqrt-distance of valid top-K
# members from their centroid above this -> fall through to multi-element mode.
_HS_SCATTER_THRESHOLD = 5.0
# small-dot ROI side length (input pixels) around argmax patch for centroid.
_HS_SMALL_ROI_HW = 24


class _MinPool2d(nn.Module):
    """min(x) = -max(-x). pytorch has no min-pool primitive."""
    def __init__(self, kernel_size, stride=1, padding=0):
        super().__init__()
        self.mp = nn.MaxPool2d(kernel_size, stride, padding)

    def forward(self, x):
        return -self.mp(-x)

# croshair rgba(222, 40, 14) -> 8, 0.94, 0.87
# names(1)    rgba(184, 75, 65) -> 5, 0.65, 0.72
# names(2)    rgba(208, 84, 73) -> 5, 0.65, 0.82
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
      "heuristic_spam":  row-suppression + morphological opening + avgpool density +
                         top-K scatter analysis -> small-dot ROI centroid OR multi-
                         element weighted_center. handles dot reticles, fat dots, and
                         ring sights via a dual-mode switch. priciest scheme; ~2.5ms
                         on 320x320 in synthetic bench. shares weighted_center buffers
                         for the multi-element fallback.
    """

    VOTING_SCHEMES = ("simple", "weighted_center", "connected", "heuristic_spam")

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
            "heuristic_spam": self._vote_heuristic_spam,
        }
        self._vote = vote_fn_map[voting_scheme]

        # Scheme-specific buffer init. heuristic_spam reuses weighted_center buffers
        # for its multi-element fallback path.
        if voting_scheme in ("weighted_center", "heuristic_spam"):
            self._init_weighted_center_buffers()
        if voting_scheme == "heuristic_spam":
            self._init_heuristic_spam_buffers()

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

    def _init_heuristic_spam_buffers(self):
        # opening: erosion (k=2 minpool with asymmetric pad to keep size) -> dilation
        # (k=3 maxpool same-pad). zero-pad shifts the asymmetric k=2 window so output
        # shape matches input.
        self._opening = nn.Sequential(
            nn.ZeroPad2d((1, 0, 1, 0)),
            _MinPool2d(kernel_size=2, stride=1),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        ).cuda().eval()
        # density: 2 stride-2 avgpool layers -> ~4x downsample with same-pad.
        # output stride per dim = 4 for 320x320 -> 80x80 grid.
        self._density = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
        ).cuda().eval()

    # --- voting ---------------------------------------------------------------

    def _vote_simple(self, mask: cp.ndarray):
        ys, xs = cp.where(mask)
        if ys.size == 0:
            return None
        return float(ys.mean()), float(xs.mean())

    def _weighted_center_inner(self, mask: cp.ndarray):
        """Gaussian-weighted center over the input mask. Mask must be (roi_h, roi_w)
        bool or {0,1}-valued float. Reused by _vote_weighted_center and by the multi-
        element fallback in _vote_heuristic_spam."""
        cp.multiply(self._w_weights, mask, out=self._w_wm)  # mask broadcasts bool->0/1
        total = float(self._w_wm.sum())
        if total <= 0.0:
            return None
        cy = float((self._w_wm * self._w_ys).sum()) / total
        cx = float((self._w_wm * self._w_xs).sum()) / total
        return cy, cx

    def _vote_weighted_center(self, mask: cp.ndarray):
        return self._weighted_center_inner(mask)

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

    def _vote_heuristic_spam(self, mask: cp.ndarray):
        """Multi-stage pipeline: row-suppress -> opening -> avgpool density ->
        top-K argmax -> scatter check -> small-dot ROI centroid OR multi-element
        weighted_center fallback. See src/tests/fart_avgpool.py for the visual
        prototype each stage was tuned in."""
        H, W = self.roi_h, self.roi_w

        # row-suppression (kill horizontal stripes e.g. red nametags) on raw mask
        # before opening so opening doesn't dilate nametag remnants back.
        mask_u8 = mask.astype(cp.uint8)
        row_sum = mask_u8.sum(axis=1, keepdims=True)              # (H, 1)
        col_sum = mask_u8.sum(axis=0, keepdims=True)              # (1, W)
        suppress = row_sum > _HS_RATIO_THRESHOLD * col_sum        # (H, W) via broadcast
        mask_filtered = mask & ~suppress                          # (H, W) bool

        # opening + density via torch on a dlpack-bridged float32 view of the mask.
        mask_f32 = mask_filtered.astype(cp.float32)
        mask_t = torch.from_dlpack(mask_f32)[None, None, ...]
        with torch.no_grad():
            opened = self._opening(mask_t)                         # (1,1,H,W) {0,1}
            pooled = self._density(opened)[0, 0]                   # (Hp, Wp)

        Hp, Wp = pooled.shape
        stride_y = H / Hp
        stride_x = W / Wp
        roi_half = _HS_SMALL_ROI_HW // 2

        # top-K argmax positions in pooled grid.
        flat = pooled.flatten()
        K = min(_HS_TOP_K, flat.numel())
        vals, idxs = flat.topk(K)
        ys_top = (idxs // Wp).float()
        xs_top = (idxs % Wp).float()
        top1_val = float(vals[0])
        if top1_val <= 0.0:
            return None

        # mean sqrt-of-L2 dist (input-pixel units) of valid top-K from their centroid.
        # sqrt compresses outliers — a couple of far-away patches don't dominate.
        valid = vals > 0
        n_valid = int(valid.sum().item())
        if n_valid > 1:
            valid_ys = ys_top[valid]
            valid_xs = xs_top[valid]
            cy_topk = valid_ys.mean()
            cx_topk = valid_xs.mean()
            dy = (valid_ys - cy_topk) * stride_y
            dx = (valid_xs - cx_topk) * stride_x
            l2 = torch.sqrt(dy * dy + dx * dx)
            mean_dist = float(torch.sqrt(l2).mean())
        else:
            mean_dist = 0.0

        if mean_dist <= _HS_SCATTER_THRESHOLD:
            # SMALL-DOT MODE: argmax patch + ROI centroid on opened mask. argmax was
            # computed over density(opened) so opened in the patch's RF is non-empty
            # by construction whenever top1_val > 0.
            py0 = int(ys_top[0].item())
            px0 = int(xs_top[0].item())
            cy_op = py0 * stride_y                                # corner-aligned
            cx_op = px0 * stride_x
            y_lo = max(0, int(cy_op - roi_half))
            y_hi = min(H, int(cy_op + roi_half))
            x_lo = max(0, int(cx_op - roi_half))
            x_hi = min(W, int(cx_op + roi_half))
            opened_cp = cp.from_dlpack(opened.detach())[0, 0]
            roi_mask = opened_cp[y_lo:y_hi, x_lo:x_hi] > 0
            cys, cxs = cp.where(roi_mask)
            if cys.size == 0:
                return cy_op, cx_op  # fallback to patch center (shouldn't normally hit)
            return float(cys.mean()) + y_lo, float(cxs.mean()) + x_lo

        # MULTI-ELEMENT MODE: weighted_center on the opened mask. shares buffers
        # with _vote_weighted_center via _weighted_center_inner.
        opened_cp = cp.from_dlpack(opened.detach())[0, 0]
        return self._weighted_center_inner(opened_cp)

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

        mask = cupy_red_mask(roi, color_center= _HSV_COLOR_CENTER, color_range= _HSV_COLOR_RANGE, s_min=_HSV_S_MIN, s_max=_HSV_S_MAX,
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
