"""
TRT inference benchmark: yours (execute_async_v3 + CuPy) vs Ultralytics (execute_v2 + torch)

Uses actual production code — no reimplementation:
  - Ultralytics: AutoBackend.forward() called directly
  - Yours:       TensorRT_Engine.inference_cp() called directly

Edit ENGINE_PATH if needed.

Run from repo root:
    python -m src.tests.bench_trt
"""

import sys
import time
from pathlib import Path

import cupy as cp
import numpy as np
import tensorrt as trt
import torch

ULTRALYTICS_ENGINE_PATH = "data/models/combined_test2/weights/640x640.engine"
ENGINE_PATH             = "data/models/combined_test2/weights/640x640_stripped.engine"
WARMUP                  = 200
N_SAMPLES               = 512
N_FRAMES                = 64
INPUT_DTYPE             = cp.float32   # cp.float32 or cp.float16

# ── path setup ────────────────────────────────────────────────────────────────
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from src.aimbot.engine.tensorrt_engine import TensorRT_Engine
from ultralytics.nn.autobackend import AutoBackend


# ── timer ─────────────────────────────────────────────────────────────────────

def timer(fn, warmup, n_samples, iters):
    for _ in range(warmup):
        fn()
    cp.cuda.runtime.deviceSynchronize()
    torch.cuda.synchronize()

    sprint_means = np.empty(n_samples)
    for s in range(n_samples):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        cp.cuda.runtime.deviceSynchronize()
        torch.cuda.synchronize()
        sprint_means[s] = (time.perf_counter() - t0) / iters * 1e6

    mean = sprint_means.mean()
    std  = sprint_means.std(ddof=1)
    ci95 = 1.96 * std / np.sqrt(n_samples)
    return mean, std, ci95


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    engine_path = Path(ENGINE_PATH)
    if not engine_path.exists():
        print(f"Engine not found: {engine_path}")
        sys.exit(1)

    device = torch.device("cuda")

    print("TRT inference benchmark")
    print(f"CUDA device : {torch.cuda.get_device_name(0)}")
    print(f"Engine      : {engine_path}")
    print(f"Warmup: {WARMUP}  |  Samples: {N_SAMPLES}  |  Iters/sample: {N_FRAMES}\n")

    # ── Ultralytics ───────────────────────────────────────────────────────────
    print("Loading Ultralytics AutoBackend...")
    ult = AutoBackend(str(Path(ULTRALYTICS_ENGINE_PATH)), device=device, verbose=False)
    ult.eval()
    print(f"  ult.fp16 = {ult.fp16}")

    in_shape = ult.bindings["images"].shape  # (1, 3, H, W)
    # match whatever dtype AutoBackend will actually use internally
    ult_im   = torch.zeros(in_shape, dtype=torch.float16 if ult.fp16 else torch.float32, device=device)

    def ult_infer():
        ult(ult_im)

    # ── yours ─────────────────────────────────────────────────────────────────
    print("Loading TensorRT_Engine...")
    mine = TensorRT_Engine(str(engine_path), conf_threshold=0.25, verbose=False)
    print(f"  mine input_dtype = {mine.input_dtype}")

    _, _, H, W = in_shape
    mine_im = cp.zeros((1, 3, H, W), dtype=INPUT_DTYPE)

    def mine_infer():
        mine.inference_cp(mine_im)

    # execute_async_v3 with device sync instead of CuPy stream sync
    def mine_infer_devicesync():
        mine.context.set_tensor_address(mine.input_tensor_name, mine_im.data.ptr)
        mine.context.execute_async_v3(mine.stream.ptr)
        cp.cuda.runtime.deviceSynchronize()

    # execute_async_v3 + deviceSync + fp16 input
    mine_im_fp16 = mine_im.astype(cp.float16)
    def mine_infer_fp16():
        mine.context.set_tensor_address(mine.input_tensor_name, mine_im_fp16.data.ptr)
        mine.context.execute_async_v3(mine.stream.ptr)
        cp.cuda.runtime.deviceSynchronize()

    def bench(label, fn, baseline_mean=None):
        mean, std, ci95 = timer(fn, WARMUP, N_SAMPLES, N_FRAMES)
        speedup = f"  {baseline_mean/mean:.2f}x  ({(1 - mean/baseline_mean)*100:.1f}%)" if baseline_mean is not None else ""
        print(f"  {label:<40}: {mean:7.2f} ±{std:6.2f} µs  CI95=[{mean-ci95:.2f}, {mean+ci95:.2f}]{speedup}")
        return mean
    
    # ── bench ─────────────────────────────────────────────────────────────────
    print("\nRunning...\n")
    print(f"{'='*90}")
    ult_mean = bench("execute_v2  + torch   (Ultralytics)", ult_infer)
    # bench("execute_v3  + stream.sync (yours)",      mine_infer,            ult_mean)
    bench("execute_v3  + deviceSync",               mine_infer_devicesync, ult_mean)
    bench("execute_v3  + deviceSync + fp16 input",  mine_infer_fp16,       ult_mean)
    print(f"{'='*90}")
