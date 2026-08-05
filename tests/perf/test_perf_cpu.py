"""CPU-side hot-loop costs. No GPU, no models."""

import numpy as np
import pytest

from src.aimbot import bootstrap
from src.aimbot.aimbot import _Slot
from src.aimbot.data_parsing import targetselector
from src.aimbot.engine.tracker_adapter import crosshair_rows_to_tracked
from tests.perf.conftest import hot_trials
from tests.support.fakes import FakeFPS, det_rows, enemy, tracked_rows

pytestmark = pytest.mark.perf
GROUP = "cpu tail"


@pytest.fixture(scope="module")
def dets():
    return np.array([[320, 320, 40, 80, 0.9, 0],
                     [200, 200, 40, 80, 0.8, 0]], dtype=np.float32)


@pytest.fixture(scope="module")
def tracked():
    return tracked_rows([enemy(300, 280, 340, 360, 1), enemy(180, 180, 220, 260, 2)])


def record(perf, name, fn, **kw):
    perf.record(name, hot_trials(fn, **kw), group=GROUP,
                note="fastest of 6 trials")


class TestTracker:
    @pytest.mark.parametrize("impl", ["ultralytics", "ultralytics_vectorized", "cpp"])
    def test_tracker_step(self, perf, cfg, dets, impl):
        cfg["tracker_settings"]["tracker_impl"] = impl
        tracker = bootstrap.create_tracker(cfg, ".engine")

        def step():
            tracker.update(dets)
            tracker.multi_predict(tracks=None)
            tracker.get_active_tracks_with_lifetime()

        record(perf, f"tracker.step[{impl}]", step, iters=500, warm=200)


class TestTargetSelector:
    @pytest.fixture
    def ts(self, cfg):
        t = targetselector.TargetSelector(cfg=cfg, detection_window_dim=(640, 640),
                                          screen_hw=(1440, 2560), fps_tracker=FakeFPS(144.0))
        for _ in range(64):
            t.update_movement_buffer((3, -2))
        return t

    def test_update_prev_detection(self, perf, ts, tracked):
        record(perf, "targetselector.update_prev_detection",
               lambda: ts.update_prev_detection(tracked), iters=500, warm=200)

    def test_get_deltas(self, perf, ts, tracked):
        record(perf, "targetselector.get_deltas",
               lambda: ts.get_deltas(tracked), iters=300, warm=100)

    def test_lead_target(self, perf, ts):
        record(perf, "targetselector.lead_target",
               lambda: ts.lead_target(0.05, (3, -2), target_id=1, track_age=64),
               iters=500, warm=200)

    def test_ring_buffer_ordered(self, perf, ts):
        record(perf, "targetselector.buffer.ordered", ts.buffer.ordered,
               iters=2000, warm=500)


class TestPlumbing:
    def test_crosshair_rows_to_tracked(self, perf):
        rows = det_rows([[100, 100, 164, 164, 1.0, 2]])
        record(perf, "crosshair_rows_to_tracked",
               lambda: crosshair_rows_to_tracked(rows), iters=2000, warm=500)

    def test_mousemover_move(self, perf, cfg, monkeypatch):
        from src.aimbot.input import mousemover
        monkeypatch.setattr(mousemover.win32api, "mouse_event", lambda *a: None)
        mm = bootstrap.create_mousemover(cfg)
        record(perf, "mousemover.move_mouse_humanized (win32 stubbed)",
               lambda: mm.move_mouse_humanized(12.0, -7.0), iters=2000, warm=500)

    def test_slot_handoff_roundtrip(self, perf):
        """Cost the async pipeline pays per stage boundary."""
        s = _Slot()

        def roundtrip():
            s.put(1)
            s.get()

        record(perf, "_Slot.put+get roundtrip", roundtrip, iters=5000, warm=1000)


class TestInput:
    def test_get_async_key_state(self, perf):
        """Polled per frame now that pynput's hook threads are gone."""
        from src.aimbot.input.inputdetector import InputDetector
        d = InputDetector()
        record(perf, "inputdetector.is_rmb_pressed (GetAsyncKeyState)",
               lambda: d.is_rmb_pressed, iters=2000, warm=500)
