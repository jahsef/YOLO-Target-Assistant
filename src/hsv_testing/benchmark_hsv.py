"""
Benchmark: OpenCV BGR HSV mask vs CuPy native RGB fused mask.

Two paths produce a boolean (H, W) red mask using OpenCV's HSV convention
(H in [0,180), S/V in [0,256)):

1. opencv_bgr_cpu  — cv2.cvtColor(BGR2HSV) + cv2.inRange (x2) + bitwise_or.
                      Frame is a CPU np.uint8 BGR array.
2. cupy_rgb_gpu    — fused RawKernel over a GPU cp.uint8 RGB array.
                      No full HSV materialization; computes per-pixel
                      max/min/saturation/hue inline and emits the mask bit.

The aimbot pipeline already has frames as RGB on GPU (betterercam nvidia_gpu),
so the CuPy path is what we'd actually use; the OpenCV path is the baseline.

Validation: both paths run on the same source pixels (RGB->BGR for opencv)
and must produce identical masks within a tiny tolerance (rounding diffs at
hue boundaries are allowed via --tol-pixels).
"""

import argparse
import time

import cv2
import cupy as cp
import numpy as np


# ---------------------------------------------------------------------------
# CuPy fused kernel: RGB uint8 -> bool mask, OpenCV HSV semantics, red wrap.
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# CuPy ops-only path (no custom kernel). Same OpenCV HSV semantics, red wrap.
# ---------------------------------------------------------------------------

def cupy_red_mask_ops(rgb_gpu: cp.ndarray, color_range: int,
                      s_min: int, v_min: int,
                      s_max: int = 255, v_max: int = 255) -> cp.ndarray:
    """Same output as cupy_red_mask but built from stock CuPy elementwise ops."""
    assert rgb_gpu.dtype == cp.uint8 and rgb_gpu.ndim == 3 and rgb_gpu.shape[2] == 3

    r = rgb_gpu[..., 0].astype(cp.int32)
    g = rgb_gpu[..., 1].astype(cp.int32)
    b = rgb_gpu[..., 2].astype(cp.int32)

    mx = cp.maximum(cp.maximum(r, g), b)
    mn = cp.minimum(cp.minimum(r, g), b)
    diff = mx - mn

    v = mx
    # Avoid div-by-zero; pixels with mx==0 get s=0 below.
    # int32 needed: diff*255 can reach 65025 which overflows int16.
    s = cp.where(mx == 0, cp.int32(0), (diff * 255) // cp.maximum(mx, 1))

    # Conditional hue (OpenCV scale: degrees / 2, in [0, 180)).
    # 30 == 60/2.
    h_r = (30 * (g - b)) // cp.maximum(diff, 1)
    h_g = 60  + (30 * (b - r)) // cp.maximum(diff, 1)
    h_b = 120 + (30 * (r - g)) // cp.maximum(diff, 1)
    h = cp.where(mx == r, h_r, cp.where(mx == g, h_g, h_b))
    h = cp.where(h < 0, h + 180, h)

    half = color_range // 2
    sv_ok = (v >= v_min) & (v <= v_max) & (s >= s_min) & (s <= s_max) & (diff > 0)
    red = (h <= half) | (h >= 179 - half)
    return sv_ok & red


# ---------------------------------------------------------------------------
# OpenCV reference (baseline) — BGR uint8 on CPU.
# ---------------------------------------------------------------------------

def opencv_red_mask(bgr_cpu: np.ndarray, color_range: int,
                    s_min: int, v_min: int,
                    s_max: int = 255, v_max: int = 255) -> np.ndarray:
    hsv = cv2.cvtColor(bgr_cpu, cv2.COLOR_BGR2HSV)
    half = color_range // 2
    lo_mask = cv2.inRange(hsv, (0, s_min, v_min),         (half, s_max, v_max))
    hi_mask = cv2.inRange(hsv, (179 - half, s_min, v_min), (179, s_max, v_max))
    return cv2.bitwise_or(lo_mask, hi_mask).astype(bool)


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def percentiles(samples_ms: np.ndarray) -> dict:
    return {
        "mean": float(samples_ms.mean()),
        "p50":  float(np.percentile(samples_ms, 50)),
        "p95":  float(np.percentile(samples_ms, 95)),
        "p99":  float(np.percentile(samples_ms, 99)),
        "fps_mean": 1000.0 / float(samples_ms.mean()),
    }


def fmt_row(label: str, stats: dict) -> str:
    return (f"{label:<28} mean={stats['mean']:7.3f} ms  "
            f"p50={stats['p50']:7.3f}  p95={stats['p95']:7.3f}  "
            f"p99={stats['p99']:7.3f}  fps≈{stats['fps_mean']:7.1f}")


def bench_opencv(frame_bgr: np.ndarray, iters: int, warmup: int,
                 color_range: int, s_min: int, v_min: int) -> np.ndarray:
    for _ in range(warmup):
        opencv_red_mask(frame_bgr, color_range, s_min, v_min)
    samples = np.empty(iters, dtype=np.float64)
    for i in range(iters):
        t0 = time.perf_counter()
        opencv_red_mask(frame_bgr, color_range, s_min, v_min)
        samples[i] = (time.perf_counter() - t0) * 1000.0
    return samples


def bench_cupy(fn, frame_rgb_gpu: cp.ndarray, iters: int, warmup: int,
               color_range: int, s_min: int, v_min: int) -> np.ndarray:
    """Time `fn(frame_rgb_gpu, ...)` on-device, with stream sync around each call."""
    for _ in range(warmup):
        _ = fn(frame_rgb_gpu, color_range, s_min, v_min)
    cp.cuda.Stream.null.synchronize()

    samples = np.empty(iters, dtype=np.float64)
    for i in range(iters):
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        _ = fn(frame_rgb_gpu, color_range, s_min, v_min)
        cp.cuda.Stream.null.synchronize()
        samples[i] = (time.perf_counter() - t0) * 1000.0
    return samples


def make_synthetic_frame(h: int, w: int, seed: int = 0) -> np.ndarray:
    """RGB uint8 with a mix of pixels including some that should match red mask."""
    rng = np.random.default_rng(seed)
    frame = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    # Splat some pure-ish red blobs so the mask isn't ~empty.
    for _ in range(20):
        cy = rng.integers(20, h - 20)
        cx = rng.integers(20, w - 20)
        rr = rng.integers(5, 25)
        ys = slice(max(0, cy - rr), min(h, cy + rr))
        xs = slice(max(0, cx - rr), min(w, cx + rr))
        frame[ys, xs, 0] = 255
        frame[ys, xs, 1] = rng.integers(0, 30)
        frame[ys, xs, 2] = rng.integers(0, 30)
    return frame


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=int, default=640)
    ap.add_argument("--w", type=int, default=640)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--color-range", type=int, default=4)
    ap.add_argument("--s-min", type=int, default=90)
    ap.add_argument("--v-min", type=int, default=90)
    ap.add_argument("--tol-pixels", type=int, default=8,
                    help="Allowed mask pixel disagreement (boundary rounding).")
    ap.add_argument("--image", type=str, default=None,
                    help="Optional path to an image (BGR via cv2.imread). Else synthetic.")
    args = ap.parse_args()
    
    if args.image:
        bgr = cv2.imread(args.image)
        if bgr is None:
            raise FileNotFoundError(args.image)
        bgr = cv2.resize(bgr, (args.w, args.h))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    else:
        rgb = make_synthetic_frame(args.h, args.w)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # Correctness check.
    mask_cv   = opencv_red_mask(bgr, args.color_range, args.s_min, args.v_min)
    rgb_gpu   = cp.asarray(rgb)
    mask_kern = cp.asnumpy(cupy_red_mask(rgb_gpu,     args.color_range, args.s_min, args.v_min))
    mask_ops  = cp.asnumpy(cupy_red_mask_ops(rgb_gpu, args.color_range, args.s_min, args.v_min))
    total = mask_cv.size
    diff_kern = int(np.count_nonzero(mask_cv != mask_kern))
    diff_ops  = int(np.count_nonzero(mask_cv != mask_ops))
    print(f"[validate] frame={args.h}x{args.w}  opencv_pos={int(mask_cv.sum())}  "
          f"kernel_pos={int(mask_kern.sum())}  ops_pos={int(mask_ops.sum())}")
    print(f"[validate] disagreement vs opencv  kernel={diff_kern}/{total} "
          f"({100.0*diff_kern/total:.4f}%)  ops={diff_ops}/{total} "
          f"({100.0*diff_ops/total:.4f}%)")
    if max(diff_kern, diff_ops) > args.tol_pixels:
        print(f"[validate] WARNING: disagreement above tolerance ({args.tol_pixels})")

    # Benchmarks.
    print(f"\n[bench] iters={args.iters} warmup={args.warmup}")
    cv_stats   = percentiles(bench_opencv(bgr, args.iters, args.warmup,
                                          args.color_range, args.s_min, args.v_min))
    kern_stats = percentiles(bench_cupy(cupy_red_mask, rgb_gpu, args.iters, args.warmup,
                                        args.color_range, args.s_min, args.v_min))
    ops_stats  = percentiles(bench_cupy(cupy_red_mask_ops, rgb_gpu, args.iters, args.warmup,
                                        args.color_range, args.s_min, args.v_min))

    print()
    print(fmt_row("opencv_bgr (CPU)",            cv_stats))
    print(fmt_row("cupy_rgb kernel (GPU)",       kern_stats))
    print(fmt_row("cupy_rgb ops-only (GPU)",     ops_stats))


if __name__ == "__main__":
    main()
