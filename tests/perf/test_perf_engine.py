"""Full-pipeline throughput and screenshot->mouse latency, against the real engines."""

import time

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from src.aimbot import bootstrap  # noqa: E402
from src.aimbot.aimbot import Aimbot  # noqa: E402
from src.aimbot.data_parsing import targetselector  # noqa: E402
from tests.perf.conftest import hot_trials  # noqa: E402
from tests.support.fakes import FakeFPS, RecordingMouse  # noqa: E402
from tests.support.loop_harness import FakeInput  # noqa: E402

pytestmark = [pytest.mark.perf, pytest.mark.gpu, pytest.mark.engine]


@pytest.fixture(scope="module")
def locked():
    return np.array([300, 300, 320, 320, 1, 0.9, 0, 0, 0, 30], dtype=np.float32)


class TestPipeline:
    def test_scan_path(self, perf, pipeline, gpu_frame, cuda_sync):
        perf.record("pipeline.run [base + hsv]",
                    hot_trials(lambda: pipeline.run(gpu_frame, False, None, 0),
                               iters=100, warm=60, sync=cuda_sync),
                    group="pipeline", note="no ADS / no lock")

    def test_sr_path(self, perf, pipeline, gpu_frame, locked, cuda_sync):
        perf.record("pipeline.run [sr + hsv]",
                    hot_trials(lambda: pipeline.run(gpu_frame, True, locked, 0),
                               iters=100, warm=60, sync=cuda_sync),
                    group="pipeline", note="ADS + small lock")

    def test_base_inference_alone(self, perf, pipeline, gpu_frame, cuda_sync):
        pp = pipeline.base_model._preprocess_cp(gpu_frame)
        perf.record("trt base inference_cp",
                    hot_trials(lambda: pipeline.base_model.model.inference_cp(pp),
                               iters=100, warm=60, sync=cuda_sync),
                    group="pipeline", note="detector only, no hsv")

    def test_hsv_marginal_cost(self, perf, pipeline, gpu_frame, cuda_sync):
        """What HSV adds on top of the detector — the number the kernel fusion moved.
        Paired inside one trial loop so GPU drift hits both arms equally."""
        pp = pipeline.base_model._preprocess_cp(gpu_frame)

        def trt_only():
            cp.asnumpy(pipeline.base_model.model.inference_cp(pp))

        def trt_plus_hsv():
            cp.asnumpy(pipeline.base_model.model.inference_cp(pp))
            pipeline.hsv_detector.detect(gpu_frame)

        a = hot_trials(trt_only, iters=60, warm=40, sync=cuda_sync)
        b = hot_trials(trt_plus_hsv, iters=60, warm=40, sync=cuda_sync)
        perf.record("hsv marginal cost on top of trt", np.maximum(b - a, 0.0),
                    group="pipeline", note="paired difference")


class TestEndToEndLatency:
    def test_frame_in_to_mouse_out(self, perf, pipeline, gpu_frame, cfg, monkeypatch):
        """Screenshot -> mouse movement, minus the capture itself: hand a frame to the
        pipeline and stop the clock when the mouse move returns. This is THE latency
        number; treat every other metric as an explanation of it."""
        from src.aimbot.input import mousemover
        monkeypatch.setattr(mousemover.win32api, "mouse_event", lambda *a: None)

        bot = object.__new__(Aimbot)
        bot.cfg = cfg
        bot.pipeline = pipeline
        bot.tracker = bootstrap.create_tracker(cfg, ".engine")
        bot.fps_tracker = FakeFPS(144.0)
        bot.mousemover = bootstrap.create_mousemover(cfg)
        bot.inputdetector = FakeInput(rmb=True)
        bot.gui_manager = None
        bot.target_selector = targetselector.TargetSelector(
            cfg=cfg, detection_window_dim=pipeline.base_hw_capture,
            screen_hw=(1440, 2560), fps_tracker=bot.fps_tracker)
        import threading
        bot._frame_count = 0
        bot._route_lock = threading.Lock()
        bot._starve_ema = 0.0
        bot._lmb_was_down = False
        bot._stop = threading.Event()
        bot._worker_error = None

        def one_frame():
            results, bypass = bot._stage_detect(gpu_frame)
            tracked = bot._stage_track(results, bypass)
            bot._stage_aim(tracked, aimbot_active=True)

        for _ in range(60):  # warm engines + age a track past min_frames_to_target
            one_frame()

        samples = np.empty(200, dtype=np.float64)
        for i in range(samples.size):
            t0 = time.perf_counter_ns()
            one_frame()
            samples[i] = (time.perf_counter_ns() - t0) / 1e6
        perf.record("e2e: frame -> mouse move (serial)", samples,
                    group="latency", note="excludes camera.grab; per-frame, not batched")
        perf.record_value("e2e fps (steady state)", 1000.0 / float(samples.min()),
                          unit="fps", group="latency",
                          note="1000 / fastest frame; the comparable one")
        perf.record_value("e2e fps (median, as-observed)",
                          1000.0 / float(np.median(samples)), unit="fps", group="latency",
                          note="what this run actually did, desktop contention included")
