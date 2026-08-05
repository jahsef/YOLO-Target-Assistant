"""The 3-stage threaded loop: handoff semantics, frame flow, shutdown, error paths.

Runs the real _capture_loop / _detect_loop / _aim_loop against a fake camera and fake
pipeline, so the threading is exercised without a GPU.
"""

import threading
import time

import numpy as np
import pytest

from src.aimbot.aimbot import Aimbot, _Slot
from src.aimbot import bootstrap
from src.aimbot.data_parsing import targetselector
from tests.support.fakes import FakeFPS, RecordingMouse
from tests.support.loop_harness import FakeInput

CENTER = 320


class FakeCamera:
    """Produces a new frame every `period` seconds, None in between — same contract as
    betterercam's grab()."""

    def __init__(self, period=0.001, fail_after=None):
        self.period = period
        self.next_at = 0.0
        self.grabs = 0
        self.produced = 0
        self.fail_after = fail_after

    def grab(self):
        self.grabs += 1
        if self.fail_after is not None and self.grabs > self.fail_after:
            raise RuntimeError("camera exploded")
        now = time.perf_counter()
        if now < self.next_at:
            return None
        self.next_at = now + self.period
        self.produced += 1
        return np.full((8, 8, 3), self.produced % 256, dtype=np.uint8)

    def release(self):
        pass


class FakePipeline:
    def __init__(self, dets, work_s=0.0, raise_after=None):
        self.dets = np.asarray(dets, dtype=np.float32).reshape(-1, 6)
        self.work_s = work_s
        self.calls = 0
        self.raise_after = raise_after
        self.seen_locked = []

    def run(self, frame, ads, locked, locked_lifetime):
        self.calls += 1
        if self.raise_after is not None and self.calls > self.raise_after:
            raise RuntimeError("pipeline exploded")
        self.seen_locked.append((locked is not None, locked_lifetime))
        if self.work_s:
            time.sleep(self.work_s)
        return self.dets.copy(), np.empty((0, 6), dtype=np.float32)


def build_bot(cfg, *, camera, pipeline):
    bot = object.__new__(Aimbot)
    bot.cfg = cfg
    bot.camera = camera
    bot.pipeline = pipeline
    bot.tracker = bootstrap.create_tracker(cfg, ".engine")
    bot.fps_tracker = FakeFPS(144.0)
    bot.mousemover = RecordingMouse()
    bot.inputdetector = FakeInput()
    bot.gui_manager = None
    bot.target_selector = targetselector.TargetSelector(
        cfg=cfg, detection_window_dim=(640, 640), screen_hw=(1440, 2560),
        fps_tracker=bot.fps_tracker)
    bot._frame_count = 0
    bot._route_lock = threading.Lock()
    bot._stop = threading.Event()
    bot._worker_error = None
    bot._detect_ema = 0.0
    bot._grab_ema = 0.0
    bot._starve_ema = 0.0
    bot._lmb_was_down = False
    return bot


class RunningPipeline:
    """Starts the three stage loops and stops them cleanly."""

    def __init__(self, bot):
        self.bot = bot
        self.frames = _Slot()
        self.detections = _Slot()
        self.threads = [
            threading.Thread(target=bot._capture_loop, args=(self.frames,), daemon=True),
            threading.Thread(target=bot._detect_loop, args=(self.frames, self.detections), daemon=True),
            threading.Thread(target=bot._aim_loop, args=(self.detections,), daemon=True),
        ]

    def __enter__(self):
        for t in self.threads:
            t.start()
        return self

    def wait_for_frames(self, n, timeout=5.0):
        end = time.perf_counter() + timeout
        while self.bot._frame_count < n and time.perf_counter() < end:
            if self.bot._worker_error:
                return
            time.sleep(0.002)

    def wait_for_stop(self, timeout=5.0):
        end = time.perf_counter() + timeout
        while not self.bot._stop.is_set() and time.perf_counter() < end:
            time.sleep(0.002)

    def __exit__(self, *exc):
        self.bot._stop.set()
        self.frames.close()
        self.detections.close()
        for t in self.threads:
            t.join(timeout=2.0)
        assert not any(t.is_alive() for t in self.threads), "stage threads did not shut down"


@pytest.fixture
def enemy_rows():
    return [[CENTER + 60, CENTER - 40, CENTER + 100, CENTER + 40, 0.9, 0]]


class TestSlot:
    def test_get_returns_what_was_put(self):
        s = _Slot()
        s.put("a")
        assert s.get() == "a"

    def test_latest_wins_and_counts_drops(self):
        """A backlog would only let you aim where the target used to be."""
        s = _Slot()
        s.put("old")
        s.put("new")
        assert s.get() == "new"
        assert s.dropped == 1

    def test_get_drains_the_slot(self):
        s = _Slot()
        s.put("a")
        s.get()
        assert s.get(timeout=0.01) is None

    def test_get_times_out_when_empty(self):
        t0 = time.perf_counter()
        assert _Slot().get(timeout=0.05) is None
        assert time.perf_counter() - t0 >= 0.04

    def test_close_wakes_a_blocked_consumer(self):
        s = _Slot()
        result = []

        def consume():
            result.append(s.get(timeout=5.0))

        t = threading.Thread(target=consume, daemon=True)
        t.start()
        time.sleep(0.05)
        s.close()
        t.join(timeout=2.0)
        assert not t.is_alive(), "close() must unblock a waiting consumer"
        assert result == [None]

    def test_producer_never_blocks(self):
        s = _Slot()
        for i in range(1000):
            s.put(i)
        assert s.get() == 999
        assert s.dropped == 999


class TestFrameFlow:
    def test_frames_reach_the_aim_stage(self, cfg, enemy_rows):
        bot = build_bot(cfg, camera=FakeCamera(period=0.001),
                        pipeline=FakePipeline(enemy_rows))
        with RunningPipeline(bot) as run:
            run.wait_for_frames(20)
        assert bot._frame_count >= 20
        assert bot._worker_error is None

    def test_mouse_moves_toward_the_target(self, cfg, enemy_rows):
        bot = build_bot(cfg, camera=FakeCamera(period=0.001),
                        pipeline=FakePipeline(enemy_rows))
        with RunningPipeline(bot) as run:
            run.wait_for_frames(20)
        moves = [m for m, _ in bot.mousemover.moves]
        assert moves, "no mouse movement produced"
        assert any(dx > 0 for dx, _ in moves), "target is right of center"

    def test_routing_state_reaches_the_detect_stage(self, cfg, enemy_rows):
        """_prev_detection is written on the aim thread and read on the detect thread;
        the lock has to actually let the value through."""
        pipeline = FakePipeline(enemy_rows)
        bot = build_bot(cfg, camera=FakeCamera(period=0.001), pipeline=pipeline)
        with RunningPipeline(bot) as run:
            run.wait_for_frames(30)
        assert any(had_lock for had_lock, _ in pipeline.seen_locked), \
            "detect stage never observed a locked target"

    def test_capture_does_not_waste_grabs_on_a_slow_consumer(self, cfg, enemy_rows):
        """Capture at ~1 kHz into a 20 ms pipeline. A free-running capture thread would
        grab ~20x per processed frame, and each wasted grab costs GPU and GIL time that
        the detect stage wanted — measurably worse than not threading at all."""
        cam = FakeCamera(period=0.001)
        bot = build_bot(cfg, camera=cam,
                        pipeline=FakePipeline(enemy_rows, work_s=0.020))
        with RunningPipeline(bot) as run:
            run.wait_for_frames(5)
            dropped = run.frames.dropped
        assert bot._frame_count >= 5
        assert cam.produced <= bot._frame_count + 2, (
            f"{cam.produced} grabs for {bot._frame_count} processed frames")
        assert dropped == 0, "a grabbed frame should never go unread"

    def test_capture_waits_for_the_consumer(self, cfg, enemy_rows):
        """The slot is a handoff, not a queue: capture blocks while a frame sits unread."""
        bot = build_bot(cfg, camera=FakeCamera(period=0.0),
                        pipeline=FakePipeline(enemy_rows, work_s=0.020))
        frames = _Slot()
        frames.put("occupied")
        t = threading.Thread(target=bot._capture_loop, args=(frames,), daemon=True)
        t.start()
        try:
            time.sleep(0.1)
            assert frames._item == "occupied", "capture overwrote an unread frame"
        finally:
            bot._stop.set()
            frames.close()
            t.join(timeout=2.0)

    def test_no_detections_still_advances(self, cfg):
        bot = build_bot(cfg, camera=FakeCamera(period=0.001), pipeline=FakePipeline([]))
        with RunningPipeline(bot) as run:
            run.wait_for_frames(10)
        assert bot._frame_count >= 10
        assert bot.mousemover.moves == []


class TestFailureHandling:
    def test_pipeline_error_stops_the_world(self, cfg, enemy_rows):
        bot = build_bot(cfg, camera=FakeCamera(period=0.0005),
                        pipeline=FakePipeline(enemy_rows, raise_after=3))
        with RunningPipeline(bot) as run:
            run.wait_for_stop()
        assert bot._stop.is_set()
        assert "pipeline exploded" in (bot._worker_error or "")

    def test_camera_error_stops_the_world(self, cfg, enemy_rows):
        bot = build_bot(cfg, camera=FakeCamera(period=0.0005, fail_after=5),
                        pipeline=FakePipeline(enemy_rows))
        with RunningPipeline(bot) as run:
            run.wait_for_stop()
        assert bot._stop.is_set()
        assert "camera exploded" in (bot._worker_error or "")


class TestCaptureOwnership:
    def test_async_capture_copies_the_frame(self, cfg):
        """betterercam recycles a 2-deep buffer ring; without a copy the capture thread
        would overwrite pixels the detect stage is still reading."""
        cam = FakeCamera(period=0.0)
        bot = build_bot(cfg, camera=cam, pipeline=FakePipeline([]))
        original = cam.grab()
        cam.next_at = 0.0

        class OneShot:
            def grab(self_inner):
                return original
        bot.camera = OneShot()

        owned = bot._stage_capture(own_pixels=True)
        shared = bot._stage_capture(own_pixels=False)
        assert owned is not original
        assert shared is original
        np.testing.assert_array_equal(owned, original)

    def test_serial_capture_skips_the_copy(self, cfg):
        bot = build_bot(cfg, camera=FakeCamera(period=0.0), pipeline=FakePipeline([]))
        assert bot._stage_capture(own_pixels=False) is not None

    def test_none_frame_is_passed_through(self, cfg):
        class NoFrames:
            def grab(self):
                return None
        bot = build_bot(cfg, camera=NoFrames(), pipeline=FakePipeline([]))
        assert bot._stage_capture(own_pixels=True) is None


class TestLoopSelection:
    def test_config_flag_picks_the_async_loop(self, cfg, monkeypatch):
        bot = build_bot(cfg, camera=FakeCamera(), pipeline=FakePipeline([]))
        called = []
        monkeypatch.setattr(Aimbot, "main_serial", lambda self: called.append("serial"))
        monkeypatch.setattr(Aimbot, "main_async", lambda self: called.append("async"))

        cfg["other"]["async_pipeline"] = False
        bot.main()
        cfg["other"]["async_pipeline"] = True
        bot.main()
        assert called == ["serial", "async"]

    def test_defaults_to_serial_when_key_is_absent(self, cfg, monkeypatch):
        cfg["other"].pop("async_pipeline", None)
        bot = build_bot(cfg, camera=FakeCamera(), pipeline=FakePipeline([]))
        called = []
        monkeypatch.setattr(Aimbot, "main_serial", lambda self: called.append("serial"))
        bot.main()
        assert called == ["serial"]
