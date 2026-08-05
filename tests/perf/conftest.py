"""Perf-suite fixtures.

Everything here is marked `perf`, so `pytest` alone never runs it — use `pytest -m perf`.
GPU work is kept to short bursts (a few hundred ms per metric).

Measurement note: this box shares its GPU with the desktop compositor and whatever
browser is open, so single-shot timings drift more than the effects being measured.
Every bench takes the FASTEST of several uninterrupted trials as the steady-state
number, which is what a sustained aimbot loop actually sees.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.perf


def hot_trials(fn, trials=6, iters=100, warm=60, sync=None):
    """Per-call ms for each trial. Fastest trial = steady state; the spread tells you
    how contended the machine was."""
    for _ in range(warm):
        fn()
    if sync:
        sync()
    out = np.empty(trials, dtype=np.float64)
    for i in range(trials):
        import time
        t0 = time.perf_counter_ns()
        for _ in range(iters):
            fn()
        if sync:
            sync()
        out[i] = (time.perf_counter_ns() - t0) / iters / 1e6
    return out


@pytest.fixture(scope="session")
def cuda_sync():
    import cupy as cp
    import torch

    def _sync():
        cp.cuda.runtime.deviceSynchronize()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    return _sync


@pytest.fixture(scope="session")
def gpu_frame():
    import cupy as cp
    from tests.support.fakes import crosshair_frame, nametag_bar
    frame = crosshair_frame(640, 640, 320, 320, seed=1)
    nametag_bar(frame, 290, 320, half_w=60)
    return cp.asarray(frame)


@pytest.fixture(scope="session")
def pipeline():
    """Real DetectionPipeline against the local engines. Skipped if they're absent."""
    from src.aimbot.engine.detection_pipeline import DetectionPipeline
    from tests.support.cfg import default_cfg
    return DetectionPipeline(default_cfg())
