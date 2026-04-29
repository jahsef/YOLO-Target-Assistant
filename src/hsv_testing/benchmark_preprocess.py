"""
Benchmark: stock CuPy _preprocess_cp vs fused RawKernel.

Both paths take a (H, W, 3) cp.uint8 frame and produce a (1, 3, H, W)
contiguous cp.float32 tensor with values in [0, 1] (i.e. HWC uint8 ->
NCHW float32 / 255).

1. cupy_pipeline_preprocess  -- the current model.py path:
       frame.transpose(2, 0, 1)[cp.newaxis, ...]
       cp.ascontiguousarray(..., dtype=cp.float32)
       cp.true_divide(frame, 255.0, out=frame, dtype=cp.float32)
   Three separate kernels: a transpose+copy-cast, then a divide.

2. fused_preprocess  -- single RawKernel: reads HWC uint8 directly, writes
   NCHW float32 with /255 baked in. One kernel launch, one pass over memory.
"""

import argparse
import time

import cupy as cp
import numpy as np


# ---------------------------------------------------------------------------
# Fused kernel: HWC uint8 -> NCHW float32 / 255 in a single pass.
# ---------------------------------------------------------------------------

_PREPROCESS_KERNEL = cp.RawKernel(r"""
extern "C" __global__
void hwc_u8_to_nchw_f32_div255(const unsigned char* __restrict__ src,
                                float* __restrict__ dst,
                                const int H, const int W) {
    // dst layout: (1, 3, H, W) contiguous -> dst[c*H*W + y*W + x]
    // src layout: (H, W, 3)    contiguous -> src[y*W*3 + x*3 + c]
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z;  // 0,1,2
    if (x >= W || y >= H) return;

    unsigned char v = src[(y * W + x) * 3 + c];
    dst[c * H * W + y * W + x] = (float)v * (1.0f / 255.0f);
}
""", "hwc_u8_to_nchw_f32_div255")


def fused_preprocess(frame: cp.ndarray) -> cp.ndarray:
    """frame: (H, W, 3) cp.uint8. Returns (1, 3, H, W) cp.float32."""
    assert frame.dtype == cp.uint8 and frame.ndim == 3 and frame.shape[2] == 3
    H, W, _ = frame.shape
    src = cp.ascontiguousarray(frame)  # no-op if already contiguous
    out = cp.empty((1, 3, H, W), dtype=cp.float32)

    block = (32, 8, 1)
    grid = ((W + block[0] - 1) // block[0],
            (H + block[1] - 1) // block[1],
            3)
    _PREPROCESS_KERNEL(grid, block, (src, out, np.int32(H), np.int32(W)))
    return out


# ---------------------------------------------------------------------------
# Stock CuPy pipeline (matches model.py:_preprocess_cp).
# ---------------------------------------------------------------------------

def cupy_pipeline_preprocess(frame: cp.ndarray) -> cp.ndarray:
    frame = frame.transpose(2, 0, 1)[cp.newaxis, ...]
    frame = cp.ascontiguousarray(frame, dtype=cp.float32)
    cp.true_divide(frame, 255.0, out=frame, dtype=cp.float32)
    return frame


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
    return (f"{label:<32} mean={stats['mean']:7.4f} ms  "
            f"p50={stats['p50']:7.4f}  p95={stats['p95']:7.4f}  "
            f"p99={stats['p99']:7.4f}  fps≈{stats['fps_mean']:8.1f}")


def bench(fn, frame: cp.ndarray, iters: int, warmup: int) -> np.ndarray:
    for _ in range(warmup):
        _ = fn(frame)
    cp.cuda.Stream.null.synchronize()

    samples = np.empty(iters, dtype=np.float64)
    for i in range(iters):
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        _ = fn(frame)
        cp.cuda.Stream.null.synchronize()
        samples[i] = (time.perf_counter() - t0) * 1000.0
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=int, default=640)
    ap.add_argument("--w", type=int, default=640)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--warmup", type=int, default=200)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    frame_cpu = rng.integers(0, 256, size=(args.h, args.w, 3), dtype=np.uint8)
    frame_gpu = cp.asarray(frame_cpu)

    # Validate equivalence.
    out_pipe = cupy_pipeline_preprocess(frame_gpu)
    out_fused = fused_preprocess(frame_gpu)
    assert out_pipe.shape == out_fused.shape == (1, 3, args.h, args.w)
    max_abs = float(cp.abs(out_pipe - out_fused).max())
    print(f"[validate] frame={args.h}x{args.w}  shape={out_pipe.shape}  "
          f"max_abs_diff={max_abs:.3e}")
    if max_abs > 1e-6:
        print(f"[validate] WARNING: outputs disagree above 1e-6")

    # Benchmarks.
    print(f"\n[bench] iters={args.iters} warmup={args.warmup}")
    pipe_stats  = percentiles(bench(cupy_pipeline_preprocess, frame_gpu, args.iters, args.warmup))
    fused_stats = percentiles(bench(fused_preprocess,         frame_gpu, args.iters, args.warmup))

    print()
    print(fmt_row("cupy pipeline (transpose/cast/div)", pipe_stats))
    print(fmt_row("fused RawKernel (HWC->NCHW/255)",    fused_stats))


if __name__ == "__main__":
    main()
