"""
Benchmark: div_ vs /= 255.0 on pre-allocated tensors.

Run:
    python -m src.tests.bench_div
"""

import time
import numpy as np
import torch


WARMUP = 200
N_SAMPLES = 1024
N_FRAMES = 512


def timer(fn, warmup, n_samples, iters, gpu_sync=True):
    for _ in range(warmup):
        fn()
    if gpu_sync:
        torch.cuda.synchronize()

    sprint_means = np.empty(n_samples)
    for s in range(n_samples):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        if gpu_sync:
            torch.cuda.synchronize()
        sprint_means[s] = (time.perf_counter() - t0) / iters * 1e6

    mean = sprint_means.mean()
    std = sprint_means.std(ddof=1)
    ci95 = 1.96 * std / np.sqrt(n_samples)
    return mean, std, ci95


def bench(label, fn, iters, n_samples, gpu_sync, baseline_mean=None):
    mean, std, ci95 = timer(fn, WARMUP, n_samples, iters, gpu_sync)
    speedup = f"  {baseline_mean/mean:.2f}x" if baseline_mean is not None else ""
    print(f"  {label:<20}: {mean:7.2f} ±{std:6.2f} µs  CI95=[{mean-ci95:.2f}, {mean+ci95:.2f}]{speedup}")
    return mean


if __name__ == "__main__":
    print("Benchmark: div_ vs /= 255.0")
    print(f"CUDA: {torch.cuda.get_device_name(0)}")

    for device_name in ["cuda", "cpu"]:
        gpu = device_name == "cuda"
        iters = N_FRAMES if gpu else max(1, int(N_FRAMES * 0.5))
        for dtype_name, dtype in [("fp32", torch.float32), ("fp16", torch.float16)]:
            device = torch.device(device_name)
            # pre-allocate source tensor once, copy each iter to avoid measuring alloc
            src = torch.randint(0, 256, (1, 3, 640, 640), device=device, dtype=dtype)

            print(f"\n{'='*72}")
            print(f"  {device_name}/{dtype_name}  |  warmup={WARMUP}  samples={N_SAMPLES}  iters/sample={iters}")
            print(f"{'='*72}")

            buf = src.clone()

            def run_div_(b=buf, s=src):
                b.copy_(s)
                b.div_(255.0)

            def run_idiv(b=buf, s=src):
                b.copy_(s)
                b /= 255.0

            base = bench("div_", run_div_, iters, N_SAMPLES, gpu)
            bench("/= 255.0", run_idiv, iters, N_SAMPLES, gpu, base)
