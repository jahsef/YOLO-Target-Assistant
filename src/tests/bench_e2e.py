"""
End-to-end inference benchmark: yours (cupy_nobgr5 + TRT) vs YOLO.predict()

Sequential design: each pipeline runs all its sprints before the next,
with a sleep between pipelines to let GPU/CPU settle.
No warmup.

Run from repo root:
    python -m src.tests.bench_e2e
"""

import sys
import time
from pathlib import Path

import cupy as cp
import numpy as np
import torch

ULTRALYTICS_ENGINE_PATH = "data/models/combined_test2/weights/640x640.engine"
ENGINE_PATH             = "data/models/combined_test2/weights/640x640_stripped.engine"
SLEEP_BETWEEN_S         = 3.0   # seconds to sleep between pipelines
N_SAMPLES               = 512
N_FRAMES                = 64

repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from src.aimbot.engine.model import Model
from ultralytics import YOLO


def sync():
    cp.cuda.runtime.deviceSynchronize()
    torch.cuda.synchronize()


def bench(fn, n_samples, n_frames):
    """No warmup. Returns (mean, std, ci95) in µs."""
    sprint_means = np.empty(n_samples)
    for s in range(n_samples):
        t_total = 0
        for _ in range(n_frames):
            sync()
            t0 = time.perf_counter_ns()
            fn()
            sync()
            t_total += time.perf_counter_ns() - t0
        sprint_means[s] = t_total / n_frames / 1e3
    mean = sprint_means.mean()
    std  = sprint_means.std(ddof=1)
    ci95 = 1.96 * std / np.sqrt(n_samples)
    return mean, std, ci95


def print_results(results, baseline_label):
    baseline_mean = results[baseline_label][0]
    w = max(len(l) for l in results)
    for label, (mean, std, ci95) in results.items():
        speedup = f"  {baseline_mean/mean:.2f}x  ({(1 - mean/baseline_mean)*100:.1f}%)" if label != baseline_label else ""
        print(f"  {label:<{w}}: {mean:7.2f} ±{std:6.2f} µs  CI95=[{mean-ci95:.2f}, {mean+ci95:.2f}]{speedup}")


if __name__ == "__main__":
    device = torch.device("cuda")

    print("End-to-end inference benchmark (sequential, no warmup)")
    print(f"CUDA device : {torch.cuda.get_device_name(0)}")
    print(f"Samples: {N_SAMPLES}  |  Iters/sample: {N_FRAMES}  |  Sleep between: {SLEEP_BETWEEN_S}s\n")

    frame_np = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    frame_cp = cp.asarray(frame_np)

    # ── load stock YOLO ───────────────────────────────────────────────────────
    print("Loading YOLO...")
    yolo = YOLO(str(Path(ULTRALYTICS_ENGINE_PATH)), task="detect")

    # ── load mine ─────────────────────────────────────────────────────────────
    print("Loading TensorRT_Engine...")
    mine = Model(model_path=Path(ENGINE_PATH), hw_capture=(640, 640))

    # ── define pipelines ──────────────────────────────────────────────────────
    def yolo_predict():
        yolo.predict(frame_np, verbose=False)

    def mine_e2e():
        mine.inference(frame_cp)

    def fork_preprocess_mine():
        # fork preprocess: transfer first, layout + BGR->RGB on GPU
        im = torch.from_numpy(frame_np).to(device, non_blocking=True).unsqueeze(0)
        im = im.permute(0, 3, 1, 2).flip(1).contiguous().float().div_(255.0)
        mine.model.inference_cp(cp.asarray(im))

    def cupy_preprocess_only():
        mine._preprocess_cp(frame_cp)

    def fork_preprocess_only():
        im = torch.from_numpy(frame_np).to(device, non_blocking=True).unsqueeze(0)
        im = im.permute(0, 3, 1, 2).flip(1).contiguous().float().div_(255.0)

    # ── bench ─────────────────────────────────────────────────────────────────
    print("\nRunning...\n")
    fns = [
        ("YOLO.predict() e2e",                    yolo_predict),
        ("Yours (cupy_nobgr5 + TRT) e2e",          mine_e2e),
        ("Yours (fork preprocess + TRT) e2e",      fork_preprocess_mine),
        ("ccupy_preprocess_only",            cupy_preprocess_only),
        ("fork preprocess only",                   fork_preprocess_only),
    ]

    results = {}
    for label, fn in fns:
        print(f"  benching: {label}...")
        results[label] = bench(fn, N_SAMPLES, N_FRAMES)
        time.sleep(SLEEP_BETWEEN_S)

    print(f"\n{'='*90}")
    print_results(results, "YOLO.predict() e2e")
    print(f"{'='*90}")

    # ── takeaways ─────────────────────────────────────────────────────────────
    # comparing: cupy preprocess (frame already on GPU, no transfer) vs
    #            fork preprocess (CPU numpy frame → PCIe transfer → GPU ops)
    #
    # 1. cupy preprocess is faster in isolation but e2e means are identical — bench artifact:
    #    persistent GPU frame may be cache-cold by TRT time, eating the preprocess saving.
    #
    # 2. cupy e2e std is much higher than fork despite identical means, yet preprocess-only
    #    stds are nearly the same — extra variance comes from TRT seeing a cache-cold input;
    #    fork's PCIe DMA lands the frame hot in L2 right before TRT runs.
    #
    # 3. real usage (GPU screen capture) delivers a freshly DMAed CuPy frame each call —
    #    same cache-warm condition as fork, so cupy gets both the faster preprocess and
    #    lower variance without paying the PCIe cost.
