"""Timing helpers for the perf suite.

Two sampling styles, matching what the existing src/tests/bench_*.py scripts do:
  sprint_bench  -> mean of N inner iters per sample. kills timer overhead for
                   sub-microsecond calls, but hides the tail.
  sample_bench  -> one measurement per sample. keeps the tail, which is what
                   latency numbers actually care about.
"""

import time

import numpy as np


def summarize(samples_ms) -> dict:
    s = np.asarray(samples_ms, dtype=np.float64)
    n = s.size
    mean = float(s.mean())
    std = float(s.std(ddof=1)) if n > 1 else 0.0
    return {
        "n": int(n),
        "mean": mean,
        "std": std,
        "ci95": float(1.96 * std / np.sqrt(n)) if n > 1 else 0.0,
        "min": float(s.min()),
        "p50": float(np.percentile(s, 50)),
        "p95": float(np.percentile(s, 95)),
        "p99": float(np.percentile(s, 99)),
        "max": float(s.max()),
    }


def sprint_bench(fn, n_samples=64, n_iters=64, warmup=8, sync=None) -> np.ndarray:
    """Returns (n_samples,) array of per-call times in ms."""
    for _ in range(warmup):
        fn()
    if sync:
        sync()
    out = np.empty(n_samples, dtype=np.float64)
    for i in range(n_samples):
        t0 = time.perf_counter_ns()
        for _ in range(n_iters):
            fn()
        if sync:
            sync()
        out[i] = (time.perf_counter_ns() - t0) / n_iters / 1e6
    return out


def sample_bench(fn, n_samples=256, warmup=16, sync=None) -> np.ndarray:
    """Returns (n_samples,) array of per-call times in ms, one timer per call."""
    for _ in range(warmup):
        fn()
    if sync:
        sync()
    out = np.empty(n_samples, dtype=np.float64)
    for i in range(n_samples):
        if sync:
            sync()
        t0 = time.perf_counter_ns()
        fn()
        if sync:
            sync()
        out[i] = (time.perf_counter_ns() - t0) / 1e6
    return out


def cuda_sync():
    """Sync both runtimes — cupy kernels and torch ops can be on different streams."""
    import cupy as cp
    import torch

    cp.cuda.runtime.deviceSynchronize()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def budgeted(fn, budget_s: float, n_iters: int = 1, sync=None) -> np.ndarray:
    """sprint_bench but capped by wall-clock budget instead of a fixed sample count.
    Keeps GPU benches to short bursts. Returns per-call times in ms."""
    fn()
    if sync:
        sync()
    out = []
    t_end = time.perf_counter() + budget_s
    while time.perf_counter() < t_end:
        t0 = time.perf_counter_ns()
        for _ in range(n_iters):
            fn()
        if sync:
            sync()
        out.append((time.perf_counter_ns() - t0) / n_iters / 1e6)
    return np.asarray(out, dtype=np.float64)
