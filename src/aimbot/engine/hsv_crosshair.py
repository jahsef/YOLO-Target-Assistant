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
_HSV_S_MIN = 160
_HSV_S_MAX = 245
_HSV_V_MIN = 160
_HSV_V_MAX = 245
_HSV_BOX_SIZE = 64  # synthetic detection box side, in base-region pixels
# 64x64 gives better iou for jittering crosshair detections so better tracker detections

# Centroid voting: weight contributions by a 2D Gaussian centered on the ROI so
# pixels near the center pull the centroid more than pixels at the edges. Sigma
# chosen so the corner weight ≈ 0.25 of the center weight (exp(-d²/(2σ²)) = 0.25
# at d = roi_half → σ = roi_half / sqrt(2*ln(4)) ≈ roi_half / 1.665).
_GAUSS_EDGE_FACTOR = 1.665  # sqrt(2 * ln(4))

# Cache of (ys, xs, weights, wm_buf) per ROI shape. Built lazily on first call;
# reused every frame after that. ys is (h,1), xs is (1,w), weights and wm_buf
# are (h,w). wm_buf is the scratch buffer for `weights * mask` so the hot path
# doesn't allocate.
_CENTROID_CACHE: dict[tuple[int, int], tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]] = {}


def _get_centroid_buffers(h: int, w: int):
    key = (h, w)
    cached = _CENTROID_CACHE.get(key)
    if cached is not None:
        return cached
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
    wm_buf = cp.empty((h, w), dtype=cp.float32)
    _CENTROID_CACHE[key] = (ys, xs, weights, wm_buf)
    return _CENTROID_CACHE[key]


def hsv_crosshair_detection(
    rgb_frame_gpu: cp.ndarray,
    crosshair_cls_id: int,
    center_crop_hw: tuple[int, int] | None = None,
) -> np.ndarray:
    """
    Red-crosshair detector backed by the fused HSV kernel. Returns a (1, 6)
    np.float32 row [x1, y1, x2, y2, conf, cls] in base-region xyxy coords,
    ready to concat with model detections before xyxy->xywh + tracker.update.
    Returns (0, 6) empty array if no red pixels match.

    RED CROSSHAIRS ONLY. Other colors / shapes are not supported yet -- this
    is intentionally narrow because the only thing we need from this path
    right now is locating the game's red reticle. Generalize the kernel
    rather than special-casing here if you want green/blue/cyan later.

    rgb_frame_gpu: (H, W, 3) cp.uint8 RGB (raw frame from betterercam).
    center_crop_hw: optional (crop_h, crop_w) -- only look in the center crop.
        Useful when the crosshair is roughly screen-centered and you want
        to skip 90%+ of the pixels.
    """
    H, W, _ = rgb_frame_gpu.shape

    if center_crop_hw is not None:
        crop_h, crop_w = center_crop_hw
        y0 = (H - crop_h) // 2
        x0 = (W - crop_w) // 2
        roi = rgb_frame_gpu[y0:y0 + crop_h, x0:x0 + crop_w]
    else:
        y0, x0 = 0, 0
        roi = rgb_frame_gpu

    mask = cupy_red_mask(roi, _HSV_COLOR_RANGE, s_min = _HSV_S_MIN, s_max = _HSV_S_MAX, v_min = _HSV_V_MIN, v_max = _HSV_V_MAX)

    # Gaussian-weighted centroid: pre-cached weights * mask gives per-pixel
    # vote strength; then weighted mean over the (cached) y/x index grids.
    # wm is a pre-allocated scratch buffer reused frame-to-frame (no per-frame alloc).
    ys, xs, weights, wm = _get_centroid_buffers(int(roi.shape[0]), int(roi.shape[1]))
    cp.multiply(weights, mask, out=wm)  # mask broadcasts bool->0/1
    total = float(wm.sum())
    if total <= 0.0:
        return np.empty((0, 6), dtype=np.float32)

    cy_local = float((wm * ys).sum()) / total
    cx_local = float((wm * xs).sum()) / total
    cy = cy_local + y0
    cx = cx_local + x0
    half = _HSV_BOX_SIZE / 2
    return np.array([[
        cx - half, cy - half, cx + half, cy + half,
        1.0, float(crosshair_cls_id)
    ]], dtype=np.float32)
