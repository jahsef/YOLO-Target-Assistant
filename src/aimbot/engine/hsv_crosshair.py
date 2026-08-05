import cupy as cp
import numpy as np

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


# Single-block fused kernel: HSV red-mask + row/col sums + row-suppression in ONE
# launch. Row/col sums are a global reduction over the mask, so a single launch needs
# a barrier between "compute mask + sums" and "apply suppression"; with gridDim.x == 1
# a block-wide __syncthreads() is that barrier and no cooperative-launch support is
# required. One block grid-strides the whole ROI — single-SM only, fine for the fixed
# small ROI (240x240); init asserts the size cap. HSV math is identical to
# _RED_MASK_KERNEL; sums are over the PRE-suppression mask and the
# (float) row_sum > ratio * col_sum compare is exact for counts < 2^24.
_FUSED_MASK_SUPPRESS_KERNEL = cp.RawKernel(r"""
extern "C" __global__
void fused_red_mask_row_suppress(const unsigned char* __restrict__ rgb,
                                 unsigned char* __restrict__ mask,
                                 const int H,
                                 const int W,
                                 const int src_pitch_px,
                                 const int color_center,
                                 const int color_range,
                                 const int s_min,
                                 const int v_min,
                                 const int s_max,
                                 const int v_max,
                                 const float ratio) {
    extern __shared__ int smem[];
    int* row_sum = smem;       // H ints
    int* col_sum = smem + H;   // W ints

    const int n = H * W;

    for (int i = threadIdx.x; i < H + W; i += blockDim.x) smem[i] = 0;
    __syncthreads();

    // phase A: per-pixel HSV mask + shared-memory sums of the pre-suppression mask.
    // rgb may be a strided view of the full frame: row y starts at y * src_pitch_px px.
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        int y = i / W;
        int x = i % W;
        int base = (y * src_pitch_px + x) * 3;
        unsigned char r = rgb[base + 0];
        unsigned char g = rgb[base + 1];
        unsigned char b = rgb[base + 2];

        unsigned char mx = r > g ? r : g; if (b > mx) mx = b;
        unsigned char mn = r < g ? r : g; if (b < mn) mn = b;

        unsigned char m = 0;
        int v = mx;
        if (v >= v_min && v <= v_max) {
            int diff = mx - mn;
            int s = (mx == 0) ? 0 : (diff * 255) / mx;
            if (s >= s_min && s <= s_max && diff != 0) {
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
                m = (2 * d <= color_range) ? 1 : 0;
            }
        }
        mask[i] = m;
        if (m) {
            atomicAdd(&row_sum[y], 1);
            atomicAdd(&col_sum[x], 1);
        }
    }
    __syncthreads();

    // phase B: zero pixels whose row is more than ratio*col wider than the column is tall.
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        if (mask[i]) {
            int y = i / W;
            int x = i % W;
            if ((float)row_sum[y] > ratio * (float)col_sum[x]) mask[i] = 0;
        }
    }
}
""", "fused_red_mask_row_suppress")

_FUSED_BLOCK_THREADS = 1024


# Whole heuristic_spam path in ONE launch: HSV mask -> row-suppress -> opening
# (erode 2x2 + dilate 3x3) -> 2x stride-2 avgpool -> gaussian-weighted centroid.
# Only (wsum, wy, wx) crosses back to the host, so the path costs one 24-byte D2H —
# the centroid divide is done host-side on 3 scalars.
#
# Same single-block trick as _FUSED_MASK_SUPPRESS_KERNEL: every stage is a global
# reduction or stencil over the previous one, so it needs a device-wide barrier
# between stages; with gridDim.x == 1 a __syncthreads() IS that barrier and no
# cooperative launch is required. One block = one SM, fine for a fixed 240x240 ROI.
#
# The morphology/pooling stages follow torch's exact conventions, since
# tests/unit/test_hsv_detector.py checks this against a torch reference:
#   erosion  = ZeroPad2d((1,0,1,0)) + MinPool(k=2,s=1): min over the 2x2 window
#              anchored bottom-right, out-of-range reads as 0.
#   dilation = MaxPool2d(k=3,s=1,p=1): torch pads with -inf, and the mask is
#              non-negative, so out-of-range simply doesn't participate.
#   avgpool  = AvgPool2d(k=3,s=2,p=1) with count_include_pad=True: sum of the
#              in-range neighbours divided by 9 REGARDLESS of padding.
# Centroid accumulates in double so 3600 terms in a fixed tree order stay stable
# against a pairwise float32 sum.
_FUSED_PIPELINE_KERNEL = cp.RawKernel(r"""
extern "C" __global__
void hsv_crosshair_pipeline(const unsigned char* __restrict__ rgb,
                            unsigned char* __restrict__ mask,
                            unsigned char* __restrict__ eroded,
                            unsigned char* __restrict__ opened,
                            float* __restrict__ d1,
                            float* __restrict__ d2,
                            const float* __restrict__ weights,
                            double* __restrict__ out,
                            const int H, const int W,
                            const int H1, const int W1,
                            const int H2, const int W2,
                            const int src_pitch_px,
                            const int color_center,
                            const int color_range,
                            const int s_min,
                            const int v_min,
                            const int s_max,
                            const int v_max,
                            const float ratio) {
    // doubles first so the int arrays that follow can't break 8-byte alignment
    extern __shared__ double smem[];
    double* warp_acc = smem;          // 3 * 32 doubles
    int* row_sum = (int*)(smem + 96); // H ints
    int* col_sum = row_sum + H;       // W ints

    const int n = H * W;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    for (int i = tid; i < H + W; i += nthreads) row_sum[i] = 0;
    __syncthreads();

    // --- A: per-pixel HSV mask + row/col sums of the pre-suppression mask ---
    for (int i = tid; i < n; i += nthreads) {
        int y = i / W;
        int x = i - y * W;
        int base = (y * src_pitch_px + x) * 3;
        unsigned char r = rgb[base + 0];
        unsigned char g = rgb[base + 1];
        unsigned char b = rgb[base + 2];

        unsigned char mx = r > g ? r : g; if (b > mx) mx = b;
        unsigned char mn = r < g ? r : g; if (b < mn) mn = b;

        unsigned char m = 0;
        int v = mx;
        if (v >= v_min && v <= v_max) {
            int diff = mx - mn;
            int s = (mx == 0) ? 0 : (diff * 255) / mx;
            if (s >= s_min && s <= s_max && diff != 0) {
                int h;
                if (mx == r)       h = (30 * (g - b)) / diff;
                else if (mx == g)  h = 60 + (30 * (b - r)) / diff;
                else               h = 120 + (30 * (r - g)) / diff;
                if (h < 0) h += 180;
                int d = h - color_center;
                if (d < 0) d = -d;
                if (d > 90) d = 180 - d;
                m = (2 * d <= color_range) ? 1 : 0;
            }
        }
        mask[i] = m;
        if (m) { atomicAdd(&row_sum[y], 1); atomicAdd(&col_sum[x], 1); }
    }
    __syncthreads();

    // --- B: row suppression (kills wide-thin nametag bars) ---
    for (int i = tid; i < n; i += nthreads) {
        if (mask[i]) {
            int y = i / W;
            int x = i - y * W;
            if ((float)row_sum[y] > ratio * (float)col_sum[x]) mask[i] = 0;
        }
    }
    __syncthreads();

    // --- C: erosion, 2x2 anchored bottom-right, zero-padded ---
    for (int i = tid; i < n; i += nthreads) {
        int y = i / W;
        int x = i - y * W;
        unsigned char m = mask[i];
        if (y == 0 || x == 0) {
            m = 0;  // the zero pad guarantees a 0 inside the window
        } else {
            if (!mask[(y - 1) * W + (x - 1)]) m = 0;
            if (!mask[(y - 1) * W + x])       m = 0;
            if (!mask[y * W + (x - 1)])       m = 0;
        }
        eroded[i] = m;
    }
    __syncthreads();

    // --- D: dilation, 3x3 same-pad ---
    for (int i = tid; i < n; i += nthreads) {
        int y = i / W;
        int x = i - y * W;
        unsigned char m = 0;
        for (int dy = -1; dy <= 1 && !m; ++dy) {
            int yy = y + dy;
            if (yy < 0 || yy >= H) continue;
            for (int dx = -1; dx <= 1; ++dx) {
                int xx = x + dx;
                if (xx < 0 || xx >= W) continue;
                if (eroded[yy * W + xx]) { m = 1; break; }
            }
        }
        opened[i] = m;
    }
    __syncthreads();

    // --- E: avgpool k=3 s=2 p=1 -> (H1, W1) ---
    for (int i = tid; i < H1 * W1; i += nthreads) {
        int oy = i / W1;
        int ox = i - oy * W1;
        int acc = 0;
        for (int ky = 0; ky < 3; ++ky) {
            int yy = oy * 2 - 1 + ky;
            if (yy < 0 || yy >= H) continue;
            for (int kx = 0; kx < 3; ++kx) {
                int xx = ox * 2 - 1 + kx;
                if (xx < 0 || xx >= W) continue;
                acc += opened[yy * W + xx];
            }
        }
        d1[i] = (float)acc / 9.0f;
    }
    __syncthreads();

    // --- F: avgpool again -> (H2, W2) ---
    for (int i = tid; i < H2 * W2; i += nthreads) {
        int oy = i / W2;
        int ox = i - oy * W2;
        float acc = 0.0f;
        for (int ky = 0; ky < 3; ++ky) {
            int yy = oy * 2 - 1 + ky;
            if (yy < 0 || yy >= H1) continue;
            for (int kx = 0; kx < 3; ++kx) {
                int xx = ox * 2 - 1 + kx;
                if (xx < 0 || xx >= W1) continue;
                acc += d1[yy * W1 + xx];
            }
        }
        d2[i] = acc / 9.0f;
    }
    __syncthreads();

    // --- G: gaussian-weighted centroid over the density map ---
    double a0 = 0.0, a1 = 0.0, a2 = 0.0;
    for (int i = tid; i < H2 * W2; i += nthreads) {
        int y = i / W2;
        int x = i - y * W2;
        double v = (double)weights[i] * (double)d2[i];
        a0 += v;
        a1 += v * (double)y;
        a2 += v * (double)x;
    }
    for (int off = 16; off > 0; off >>= 1) {
        a0 += __shfl_down_sync(0xffffffff, a0, off);
        a1 += __shfl_down_sync(0xffffffff, a1, off);
        a2 += __shfl_down_sync(0xffffffff, a2, off);
    }
    int warp = tid >> 5;
    int lane = tid & 31;
    if (lane == 0) {
        warp_acc[warp] = a0;
        warp_acc[32 + warp] = a1;
        warp_acc[64 + warp] = a2;
    }
    __syncthreads();

    if (tid < 32) {
        int nwarps = nthreads >> 5;
        double b0 = (tid < nwarps) ? warp_acc[tid] : 0.0;
        double b1 = (tid < nwarps) ? warp_acc[32 + tid] : 0.0;
        double b2 = (tid < nwarps) ? warp_acc[64 + tid] : 0.0;
        for (int off = 16; off > 0; off >>= 1) {
            b0 += __shfl_down_sync(0xffffffff, b0, off);
            b1 += __shfl_down_sync(0xffffffff, b1, off);
            b2 += __shfl_down_sync(0xffffffff, b2, off);
        }
        if (tid == 0) { out[0] = b0; out[1] = b1; out[2] = b2; }
    }
}
""", "hsv_crosshair_pipeline")

_WARP_ACC_DOUBLES = 96  # 3 accumulators x 32 warps


def fused_red_mask_suppress(rgb_gpu: cp.ndarray, color_center: int, color_range: int,
                            s_min: int, v_min: int, s_max: int = 255, v_max: int = 255,
                            ratio: float = _HS_RATIO_THRESHOLD,
                            out: cp.ndarray | None = None,
                            validate_device: bool = True) -> cp.ndarray:
    """rgb_gpu: (H, W, 3) cp.uint8 RGB. A basic row/col slice of a larger frame is
    consumed in place via its row pitch (no copy); any other layout (e.g. a
    negative-stride BGR->RGB flip) falls back to one ascontiguousarray copy.

    Returns (H, W) cp.bool_: red mask with row-suppression applied, in ONE kernel
    launch (see _FUSED_MASK_SUPPRESS_KERNEL). `out` is an optional pre-allocated
    (H, W) uint8 buffer so hot paths never allocate. validate_device=False skips
    the per-call shared-mem/block-size asserts for callers that checked at init.
    """
    assert rgb_gpu.dtype == cp.uint8 and rgb_gpu.ndim == 3 and rgb_gpu.shape[2] == 3
    h, w = int(rgb_gpu.shape[0]), int(rgb_gpu.shape[1])
    shared_bytes = (h + w) * 4
    if validate_device:
        attrs = cp.cuda.Device().attributes
        assert shared_bytes <= attrs['MaxSharedMemoryPerBlock'], (
            f"ROI {h}x{w} needs {shared_bytes} B shared mem, "
            f"device block limit is {attrs['MaxSharedMemoryPerBlock']} B"
        )
        assert attrs['MaxThreadsPerBlock'] >= _FUSED_BLOCK_THREADS

    s0, s1, s2 = rgb_gpu.strides
    if s1 == 3 and s2 == 1 and s0 > 0 and s0 % 3 == 0:
        src = rgb_gpu
        pitch = s0 // 3
    else:
        src = cp.ascontiguousarray(rgb_gpu)
        pitch = w
    if out is None:
        out = cp.empty((h, w), dtype=cp.uint8)

    _FUSED_MASK_SUPPRESS_KERNEL(
        (1,), (_FUSED_BLOCK_THREADS,),
        (src, out, np.int32(h), np.int32(w), np.int32(pitch),
         np.int32(color_center), np.int32(color_range),
         np.int32(s_min), np.int32(v_min),
         np.int32(s_max), np.int32(v_max),
         np.float32(ratio)),
        shared_mem=shared_bytes,
    )
    return out.view(cp.bool_)

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
      "heuristic_spam":  mask + row-suppression + morphological opening + density
                         downsample (2x stride-2 avgpools) + weighted_center on the
                         density output (coarse), all in ONE kernel launch ending in a
                         single 24-byte D2H — see _FUSED_PIPELINE_KERNEL. small FPs
                         (nametags, tracers, random red gear) get smoothed into
                         near-zero density by downsampling, while the reticle's
                         concentrated mass dominates. bench_fine_vs_coarse_fp.py shows
                         ~40% lower centroid error vs running weighted_center directly
                         on the opened mask in FP-heavy scenes. uses its own buffers at
                         density resolution — distinct from "weighted_center" scheme's
                         full-resolution buffers.
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

        # Bind the chosen scheme directly so detect() doesn't dispatch on a string each
        # frame. Every _process_* takes the RGB ROI and returns (cy, cx) in ROI-local
        # coords or None — heuristic_spam fuses masking into its own kernel, the other
        # schemes wrap cupy_red_mask + their vote.
        process_fn_map = {
            "simple": self._process_simple,
            "weighted_center": self._process_weighted_center,
            "connected": self._process_connected,
            "heuristic_spam": self._process_heuristic_spam,
        }
        self._process = process_fn_map[voting_scheme]

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

    @staticmethod
    def _avgpool_out(n: int) -> int:
        """Output length of AvgPool2d(kernel_size=3, stride=2, padding=1)."""
        return (n + 2 * 1 - 3) // 2 + 1

    def _init_heuristic_spam_buffers(self):
        # density resolution after the two stride-2 avgpools (4x downsample).
        h1, w1 = self._avgpool_out(self.roi_h), self._avgpool_out(self.roi_w)
        self._hs_h1, self._hs_w1 = h1, w1
        self._hs_dh, self._hs_dw = self._avgpool_out(h1), self._avgpool_out(w1)
        # corner-aligned stride for mapping density-coords back to roi-pixel-coords.
        self._hs_stride_y = self.roi_h / self._hs_dh
        self._hs_stride_x = self.roi_w / self._hs_dw
        # coarse weighted_center buffers at density resolution. only `weights` is used
        # by the fused kernel; ys/xs/wm stay for _weighted_center_inner comparisons.
        self._hs_ys, self._hs_xs, self._hs_weights, self._hs_wm = (
            self._make_wc_buffers(self._hs_dh, self._hs_dw)
        )
        self._hs_weights = cp.ascontiguousarray(self._hs_weights)

        # single-block kernel: shared row/col sums + the reduction accumulators must fit
        # one block's smem, and the launch needs a full 1024-thread block. trivially true
        # at 240x240 (2.7 KB); the asserts catch a future full-frame ROI before it
        # silently breaks.
        dev_attrs = cp.cuda.Device().attributes
        self._hs_shared_bytes = _WARP_ACC_DOUBLES * 8 + (self.roi_h + self.roi_w) * 4
        assert self._hs_shared_bytes <= dev_attrs['MaxSharedMemoryPerBlock'], (
            f"ROI {self.roi_h}x{self.roi_w} needs {self._hs_shared_bytes} B shared mem, "
            f"device block limit is {dev_attrs['MaxSharedMemoryPerBlock']} B"
        )
        assert dev_attrs['MaxThreadsPerBlock'] >= _FUSED_BLOCK_THREADS

        # every intermediate is pre-allocated and fully consumed within one launch.
        self._hs_mask_buf = cp.empty((self.roi_h, self.roi_w), dtype=cp.uint8)
        self._hs_eroded = cp.empty((self.roi_h, self.roi_w), dtype=cp.uint8)
        self._hs_opened = cp.empty((self.roi_h, self.roi_w), dtype=cp.uint8)
        self._hs_d1 = cp.empty((h1, w1), dtype=cp.float32)
        self._hs_d2 = cp.empty((self._hs_dh, self._hs_dw), dtype=cp.float32)
        # (weighted mass, weighted sum of y, weighted sum of x) — the only thing that
        # crosses back to the host, in one 24-byte copy.
        self._hs_out = cp.empty(3, dtype=cp.float64)

    # --- per-scheme processing (RGB ROI in, ROI-local (cy, cx) or None out) ----

    @staticmethod
    def _red_mask(roi: cp.ndarray) -> cp.ndarray:
        return cupy_red_mask(roi, color_center=_HSV_COLOR_CENTER, color_range=_HSV_COLOR_RANGE,
                             s_min=_HSV_S_MIN, s_max=_HSV_S_MAX,
                             v_min=_HSV_V_MIN, v_max=_HSV_V_MAX)

    def _process_simple(self, roi: cp.ndarray):
        return self._vote_simple(self._red_mask(roi))

    def _process_weighted_center(self, roi: cp.ndarray):
        return self._vote_weighted_center(self._red_mask(roi))

    def _process_connected(self, roi: cp.ndarray):
        return self._vote_connected(self._red_mask(roi))

    def _process_heuristic_spam(self, roi: cp.ndarray):
        """Whole chain in one launch (see _FUSED_PIPELINE_KERNEL): mask + row-suppress
        -> opening -> density downsample -> gaussian-weighted centroid. Density
        downsampling smooths small FPs into near-zero contribution. The kernel returns
        weighted sums; the centroid divide happens host-side on 3 scalars."""
        self._launch_pipeline(roi)
        wsum, wy, wx = self._hs_out.get()  # the single host sync of the whole path
        if wsum <= 0.0:
            return None
        return (wy / wsum) * self._hs_stride_y, (wx / wsum) * self._hs_stride_x

    def _launch_pipeline(self, roi: cp.ndarray) -> None:
        """Queue the fused pipeline. Basic row/col slices of a larger frame are read in
        place via their row pitch; any other layout (e.g. a negative-stride channel
        flip) costs one ascontiguousarray copy."""
        s0, s1, s2 = roi.strides
        if s1 == 3 and s2 == 1 and s0 > 0 and s0 % 3 == 0:
            src, pitch = roi, s0 // 3
        else:
            src = cp.ascontiguousarray(roi)
            pitch = int(roi.shape[1])

        _FUSED_PIPELINE_KERNEL(
            (1,), (_FUSED_BLOCK_THREADS,),
            (src, self._hs_mask_buf, self._hs_eroded, self._hs_opened,
             self._hs_d1, self._hs_d2, self._hs_weights, self._hs_out,
             np.int32(self.roi_h), np.int32(self.roi_w),
             np.int32(self._hs_h1), np.int32(self._hs_w1),
             np.int32(self._hs_dh), np.int32(self._hs_dw),
             np.int32(pitch),
             np.int32(_HSV_COLOR_CENTER), np.int32(_HSV_COLOR_RANGE),
             np.int32(_HSV_S_MIN), np.int32(_HSV_V_MIN),
             np.int32(_HSV_S_MAX), np.int32(_HSV_V_MAX),
             np.float32(_HS_RATIO_THRESHOLD)),
            shared_mem=self._hs_shared_bytes,
        )

    def _fused_mask_suppress(self, roi: cp.ndarray) -> cp.ndarray:
        """One-launch fused HSV red-mask + row-suppression into the pre-allocated
        buffer. Device limits were asserted once at init, so skip per-call checks."""
        return fused_red_mask_suppress(
            roi, _HSV_COLOR_CENTER, _HSV_COLOR_RANGE,
            s_min=_HSV_S_MIN, v_min=_HSV_V_MIN, s_max=_HSV_S_MAX, v_max=_HSV_V_MAX,
            ratio=_HS_RATIO_THRESHOLD, out=self._hs_mask_buf, validate_device=False,
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

        vote = self._process(roi)
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
