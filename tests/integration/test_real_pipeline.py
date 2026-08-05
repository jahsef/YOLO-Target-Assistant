"""Smoke tests against the REAL camera and REAL engines.

Everything else stubs the camera and the pipeline, which is why a 3x async throughput
regression could sit in the suite unnoticed. These actually start the thing up, run it
for a second or two, and check it does work at a sane rate. Mouse output is stubbed —
nothing here moves the cursor.
"""

import threading
import time

import numpy as np
import pytest

from src.aimbot.aimbot import Aimbot, _Slot

pytestmark = [pytest.mark.gpu, pytest.mark.engine, pytest.mark.slow]

RUN_S = 1.5


class ForcedInput:
    """Both loops must see the same input state or the inactive throttle fires in one
    and not the other, which silently invalidates any comparison between them."""

    def __init__(self, rmb=True):
        self.is_rmb_pressed = rmb
        self.is_toggled = True
        self.is_lmb_pressed = False

    def stop(self):
        pass


@pytest.fixture(scope="module")
def bot(repo_root):
    from src.aimbot.input import mousemover
    real_move = mousemover.win32api.mouse_event
    mousemover.win32api.mouse_event = lambda *a: None
    b = Aimbot(str(repo_root / "config" / "cfg.json"))
    b.cfg["logging"]["print_fps"] = False
    b.inputdetector = ForcedInput()
    yield b
    b.cleanup()
    mousemover.win32api.mouse_event = real_move


@pytest.fixture(scope="module", autouse=True)
def live_desktop(bot):
    """DXGI desktop duplication only emits a frame when screen content actually
    changes. On an idle desktop the camera returns None forever, so these tests would
    measure nothing and fail for reasons that have nothing to do with the code. Skip
    instead, and say why — run them while the game (or anything animating) is on screen.
    """
    got = 0
    end = time.perf_counter() + 3.0
    while time.perf_counter() < end and got < 5:
        if bot._stage_capture(own_pixels=False) is not None:
            got += 1
    if got < 5:
        pytest.skip(f"desktop is static (camera gave {got} frames in 3s) — "
                    "run these with something animating on screen")


def run_serial(bot, duration):
    n = 0
    end = time.perf_counter() + duration
    while time.perf_counter() < end:
        frame = bot._stage_capture(own_pixels=False)
        if frame is None:
            continue
        results, bypass = bot._stage_detect(frame)
        tracked = bot._stage_track(results, bypass)
        bot._stage_aim(tracked, True)
        n += 1
    return n


def run_async(bot, duration):
    frames, dets = _Slot(), _Slot()
    bot._stop.clear()
    start_count = bot._frame_count
    bot._detect_ema = bot._grab_ema = bot._starve_ema = 0.0
    threads = [
        threading.Thread(target=bot._capture_loop, args=(frames,), daemon=True),
        threading.Thread(target=bot._detect_loop, args=(frames, dets), daemon=True),
        threading.Thread(target=bot._aim_loop, args=(dets,), daemon=True),
    ]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    time.sleep(duration)
    elapsed = time.perf_counter() - t0
    processed = bot._frame_count - start_count
    bot._stop.set()
    frames.close()
    dets.close()
    for t in threads:
        t.join(timeout=2.0)
    return processed, elapsed, threads, frames


class TestRealCamera:
    def test_camera_yields_a_frame_of_the_right_shape(self, bot):
        frame = None
        deadline = time.perf_counter() + 3.0
        while frame is None and time.perf_counter() < deadline:
            frame = bot._stage_capture(own_pixels=False)
        assert frame is not None, "camera never produced a frame"
        assert frame.shape == (*bot.base_hw_capture, 3)

    def test_capture_copy_is_a_distinct_buffer(self, bot):
        owned = None
        deadline = time.perf_counter() + 3.0
        while owned is None and time.perf_counter() < deadline:
            owned = bot._stage_capture(own_pixels=True)
        assert owned is not None
        shared = bot._stage_capture(own_pixels=False)
        if shared is not None:
            assert owned.data.ptr != shared.data.ptr


class TestRealPipeline:
    def test_one_real_frame_through_every_stage(self, bot):
        frame = None
        deadline = time.perf_counter() + 3.0
        while frame is None and time.perf_counter() < deadline:
            frame = bot._stage_capture(own_pixels=False)
        results, bypass = bot._stage_detect(frame)
        assert results.ndim == 2 and results.shape[1] == 6
        assert bypass.ndim == 2 and bypass.shape[1] == 6
        tracked = bot._stage_track(results, bypass)
        assert tracked.ndim == 2 and tracked.shape[1] == 10
        raw, scaled = bot._stage_aim(tracked, aimbot_active=True)
        assert len(raw) == 2 and len(scaled) == 2

    def test_serial_loop_keeps_running(self, bot):
        """No absolute fps assertion: DXGI only emits frames when screen content
        changes, so on an idle desktop the loop is camera-starved and its rate says
        nothing about the code. Only catastrophic breakage is worth catching here; the
        serial-vs-async comparison below is the meaningful one."""
        run_serial(bot, 0.4)  # warm
        n = run_serial(bot, RUN_S)
        assert n > 0, "serial loop produced no frames at all"
        assert n / RUN_S > 5, f"serial loop managed only {n / RUN_S:.1f} fps"


class TestRealAsyncPipeline:
    def test_async_loop_runs_and_shuts_down(self, bot):
        run_async(bot, 0.4)  # warm
        n, elapsed, threads, frames = run_async(bot, RUN_S)
        assert bot._worker_error is None, bot._worker_error
        assert n > 0, "async pipeline produced no frames"
        assert not any(t.is_alive() for t in threads), "stage threads did not stop"

    def test_async_does_not_waste_grabs(self, bot):
        """JIT capture should grab about once per processed frame. Free-running would
        grab several times per frame and hand the waste to the GPU."""
        n, elapsed, _, frames = run_async(bot, RUN_S)
        assert frames.dropped <= max(2, n * 0.1), (
            f"{frames.dropped} grabbed frames went unread out of {n} processed")

    def test_async_is_not_slower_than_serial(self, bot):
        """The regression that stubbed tests could not see: async was 3x slower than
        serial under real conditions while every mocked test passed."""
        run_serial(bot, 0.4)
        serial_n = run_serial(bot, RUN_S)
        run_async(bot, 0.4)
        n, elapsed, _, _ = run_async(bot, RUN_S)
        if min(serial_n, n) < 20:
            pytest.skip(f"too few frames to compare (serial {serial_n}, async {n})")
        serial_fps, async_fps = serial_n / RUN_S, n / elapsed
        assert async_fps > serial_fps * 0.7, (
            f"async {async_fps:.1f} fps vs serial {serial_fps:.1f} fps")

    def test_async_estimates_converge(self, bot):
        """JIT timing is only as good as its stage estimates."""
        run_async(bot, RUN_S)
        assert 0 < bot._detect_ema < 0.5, f"detect_ema {bot._detect_ema*1e3:.1f}ms"
        assert 0 < bot._grab_ema < 0.5, f"grab_ema {bot._grab_ema*1e3:.1f}ms"
