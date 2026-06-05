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
_HSV_COLOR_CENTER = 0
_HSV_COLOR_RANGE = 10
_HSV_S_MIN = 160
_HSV_S_MAX = 255
_HSV_V_MIN = 140
_HSV_V_MAX = 255
_HSV_BOX_SIZE = 64  # synthetic detection box side, in base-region pixels
# 64x64 gives better iou for jittering crosshair detections so better tracker stickiness.

# Sigma chosen so corner weight ≈ 0.25 * center weight:
# exp(-d²/(2σ²)) = 0.25 at d = roi_half → σ = roi_half / sqrt(2*ln(4)) ≈ roi_half / 1.665
_GAUSS_EDGE_FACTOR = 1.665  # sqrt(2 * ln(4))

# heuristic_spam args.
# row-suppression: kill mask pixels where row_sum[y] > _HS_RATIO_THRESHOLD * col_sum[x],
# applied pre-opening so opening sees nametag-free input.
_HS_RATIO_THRESHOLD = 1.5


class _MinPool2d(nn.Module):
    """min(x) = -max(-x). pytorch has no min-pool primitive."""
    def __init__(self, kernel_size, stride=1, padding=0):
        super().__init__()
        self.mp = nn.MaxPool2d(kernel_size, stride, padding)

    def forward(self, x):
        return -self.mp(-x)


# Fused row-suppression kernel.
# Replaces the multi-op cupy chain that was previously:
#   mask_u8 = mask.astype(cp.uint8)                            # cast kernel
#   row_sum = mask_u8.sum(axis=1, keepdims=True)               # reduction kernel
#   col_sum = mask_u8.sum(axis=0, keepdims=True)               # reduction kernel
#   suppress = row_sum > _HS_RATIO_THRESHOLD * col_sum         # broadcast compare kernel
#   mask_filtered = mask & ~suppress                           # boolean AND kernel
# 5 kernel launches + 5 intermediate allocations. Bench showed ~0.33 ms.
# The two reductions can't be fused trivially (need full row/col sums before per-pixel apply),
# but the broadcast-compare + AND CAN — that's what this kernel does. Reductions stay as
# native cp.sum (CUB-backed, already optimal); kernel does the per-pixel decision in one pass.
_ROW_SUPPRESS_KERNEL = cp.RawKernel(r"""
extern "C" __global__
void row_suppress_apply(const unsigned char* __restrict__ mask,
                        const long long*    __restrict__ row_sum,
                        const long long*    __restrict__ col_sum,
                        unsigned char*      __restrict__ out,
                        const int H,
                        const int W,
                        const float ratio) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = H * W;
    if (idx >= total) return;
    int y = idx / W;
    int x = idx % W;
    // suppress when row is more than ratio*col times wider than column is tall.
    bool suppress = ((float)row_sum[y]) > ratio * ((float)col_sum[x]);
    out[idx] = (mask[idx] && !suppress) ? 1 : 0;
}
""", "row_suppress_apply")

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
      "heuristic_spam":  row-suppression (fused kernel) + morphological opening +
                         density downsample (2x stride-2 avgpools) + weighted_center
                         on the density output (coarse). small FPs (nametags, tracers,
                         random red gear) get smoothed into near-zero density by
                         downsampling, while the reticle's concentrated mass dominates.
                         bench_fine_vs_coarse_fp.py shows ~40% lower centroid error vs
                         running weighted_center directly on the opened mask in FP-heavy
                         scenes. uses its own buffers at density resolution — distinct
                         from "weighted_center" scheme's full-resolution buffers.
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

        # Scheme-specific buffer init. heuristic_spam runs weighted_center at the
        # density-output resolution (coarse), so it owns its own buffers separately.
        if voting_scheme == "weighted_center":
            self._w_ys, self._w_xs, self._w_weights, self._w_wm = (
                self._make_wc_buffers(self.roi_h, self.roi_w)
            )
        if voting_scheme == "heuristic_spam":
            self._init_heuristic_spam_buffers()

    # --- buffer init ----------------------------------------------------------

    def _make_wc_buffers(self, h: int, w: int):
        """Build Gaussian weighted-center buffers at (h, w). Returns
        (ys, xs, weights, wm). Used by both 'weighted_center' (full-res) and
        'heuristic_spam' (density-res) schemes."""
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
        wm = cp.empty((h, w), dtype=cp.float32)  # scratch for weights*mask
        return ys, xs, weights, wm

    def _init_heuristic_spam_buffers(self):
        # opening: erosion (k=2 minpool with asymmetric pad to keep size) -> dilation
        # (k=3 maxpool same-pad). zero-pad shifts the asymmetric k=2 window so output
        # shape matches input.
        self._opening = nn.Sequential(
            nn.ZeroPad2d((1, 0, 1, 0)),
            _MinPool2d(kernel_size=2, stride=1),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        ).cuda().eval()
        # density: 2 stride-2 avgpool layers -> 4x downsample with same-pad. exact output
        # shape depends on roi dims (computed via dummy forward below).
        self._density = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
        ).cuda().eval()
        # probe density output shape so weighted_center buffers can match.
        with torch.no_grad():
            dummy = torch.zeros((1, 1, self.roi_h, self.roi_w), device='cuda')
            dh, dw = self._density(dummy).shape[2:]
        self._hs_dh, self._hs_dw = int(dh), int(dw)
        # corner-aligned stride for mapping density-coords back to roi-pixel-coords.
        self._hs_stride_y = self.roi_h / self._hs_dh
        self._hs_stride_x = self.roi_w / self._hs_dw
        # coarse weighted_center buffers at density resolution.
        self._hs_ys, self._hs_xs, self._hs_weights, self._hs_wm = (
            self._make_wc_buffers(self._hs_dh, self._hs_dw)
        )

    # --- voting ---------------------------------------------------------------

    def _vote_simple(self, mask: cp.ndarray):
        ys, xs = cp.where(mask)
        if ys.size == 0:
            return None
        return float(ys.mean()), float(xs.mean())

    def _weighted_center_inner(self, mask: cp.ndarray, ys: cp.ndarray, xs: cp.ndarray,
                               weights: cp.ndarray, wm: cp.ndarray):
        """Gaussian-weighted center over `mask` using precomputed buffers. Mask shape
        must match the buffers' shape. Reused at both full-res and density-res."""
        cp.multiply(weights, mask, out=wm)  # mask broadcasts bool->0/1
        total = float(wm.sum())
        if total <= 0.0:
            return None
        cy = float((wm * ys).sum()) / total
        cx = float((wm * xs).sum()) / total
        return cy, cx

    def _vote_weighted_center(self, mask: cp.ndarray):
        return self._weighted_center_inner(
            mask, self._w_ys, self._w_xs, self._w_weights, self._w_wm
        )

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

    def _row_suppress(self, mask: cp.ndarray) -> cp.ndarray:
        """Fused row-suppression: returns mask AND NOT (row_sum[y] > ratio * col_sum[x]).
        Reductions stay native (CUB-backed); the broadcast-compare + AND fuses into
        one custom kernel — see _ROW_SUPPRESS_KERNEL for rationale."""
        H, W = mask.shape
        mask_u8 = mask.view(cp.uint8)  # bool and uint8 share memory layout
        row_sum = mask_u8.sum(axis=1, dtype=cp.int64)  # (H,)
        col_sum = mask_u8.sum(axis=0, dtype=cp.int64)  # (W,)
        out = cp.empty_like(mask_u8)
        n = H * W
        threads = 256
        blocks = (n + threads - 1) // threads
        _ROW_SUPPRESS_KERNEL(
            (blocks,), (threads,),
            (mask_u8, row_sum, col_sum, out,
             np.int32(H), np.int32(W), np.float32(_HS_RATIO_THRESHOLD)),
        )
        return out.view(cp.bool_)

    def _vote_heuristic_spam(self, mask: cp.ndarray):
        """row-suppress -> opening -> density -> weighted_center on density (coarse).
        Density downsampling smooths small FPs into near-zero contribution. Coordinates
        come back in density-grid space; multiply by the stride to map to ROI pixels."""
        mask_filtered = self._row_suppress(mask)
        mask_f32 = mask_filtered.astype(cp.float32)
        mask_t = torch.from_dlpack(mask_f32)[None, None, ...]
        with torch.no_grad():
            opened = self._opening(mask_t)
            pooled = self._density(opened)
        pooled_cp = cp.from_dlpack(pooled.detach())[0, 0]
        pt = self._weighted_center_inner(
            pooled_cp, self._hs_ys, self._hs_xs, self._hs_weights, self._hs_wm
        )
        if pt is None:
            return None
        cy_d, cx_d = pt
        return cy_d * self._hs_stride_y, cx_d * self._hs_stride_x

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
