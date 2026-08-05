"""DeltaDrain / ThreadedMouseMover.

The two properties worth guarding hardest are the ones that silently eat movement:
the fractional carry (truncating loses ~45% of a drain at these step sizes) and the
vector-magnitude floor (a per-axis floor bends the path).
"""

import math
import time

import pytest

from src.aimbot import bootstrap
from src.aimbot.input import mousemover
from src.aimbot.input.mousemover import DeltaDrain, MouseMover, ThreadedMouseMover



def drain_all(d, ticks=400):
    """Run to completion, return (total_x, total_y, steps)."""
    tx = ty = 0
    steps = []
    for _ in range(ticks):
        sx, sy = d.tick()
        tx += sx
        ty += sy
        if sx or sy:
            steps.append((sx, sy))
    return tx, ty, steps


class TestDrainMath:
    def test_delivers_what_was_added(self):
        d = DeltaDrain(drain_alpha=0.175, min_step_px=2.0)
        d.add(20.0, 0.0)
        tx, ty, _ = drain_all(d)
        assert abs(20 - tx) <= 1, f"delivered {tx} of 20"
        assert ty == 0

    def test_never_teleports(self):
        """The entire point: a 20px frame must not go out as one 20px event."""
        d = DeltaDrain(drain_alpha=0.175, min_step_px=2.0)
        d.add(20.0, 0.0)
        _, _, steps = drain_all(d)
        assert len(steps) >= 5, f"only {len(steps)} events for 20px"
        assert max(abs(sx) for sx, _ in steps) <= 5

    def test_carries_the_fraction(self):
        """Sub-pixel steps must accumulate, not truncate away. Truncating loses about
        half the movement at these step sizes."""
        d = DeltaDrain(drain_alpha=0.03, min_step_px=0.0)  # forces tiny sub-pixel steps
        d.add(10.0, 0.0)
        tx, _, _ = drain_all(d, ticks=2000)
        assert abs(10 - tx) <= 1, f"lost movement to truncation: delivered {tx} of 10"

    def test_floor_preserves_direction(self):
        """A per-axis floor would drain (10, 5) at 45 degrees instead of 63."""
        d = DeltaDrain(drain_alpha=0.175, min_step_px=2.0)
        d.add(10.0, 5.0)
        tx, ty, _ = drain_all(d)
        assert abs(tx / ty - 2.0) < 0.35, f"path bent: ({tx}, {ty}) should be ~2:1"

    def test_direction_holds_per_step(self):
        d = DeltaDrain(drain_alpha=0.175, min_step_px=2.0)
        d.add(30.0, 10.0)
        cum_x = cum_y = 0
        for _ in range(200):
            sx, sy = d.tick()
            cum_x += sx
            cum_y += sy
            if cum_y >= 3:  # once there is enough signal to judge the ratio
                assert abs(cum_x / cum_y - 3.0) < 1.2, f"drifted at ({cum_x},{cum_y})"

    def test_negative_and_diagonal(self):
        d = DeltaDrain(drain_alpha=0.175, min_step_px=2.0)
        d.add(-12.0, 7.0)
        tx, ty, _ = drain_all(d)
        assert abs(-12 - tx) <= 1 and abs(7 - ty) <= 1

    def test_floor_bounds_the_drain_time(self):
        """min_step_px guarantees a completion time, which is what kills backlog."""
        d = DeltaDrain(drain_alpha=0.0015, min_step_px=2.0)  # alpha so low only the floor acts
        d.add(20.0, 0.0)
        ticks_used = 0
        while abs(d.pending[0]) > 0.01 and ticks_used < 500:
            d.tick()
            ticks_used += 1
        assert ticks_used <= 12, f"took {ticks_used} ticks, floor not applied"

    def test_zero_floor_leaves_a_tail(self):
        """Pinning why the floor exists: pure exponential never finishes."""
        d = DeltaDrain(drain_alpha=0.175, min_step_px=0.0)
        d.add(20.0, 0.0)
        for _ in range(30):
            d.tick()
        assert abs(d.pending[0]) > 0.0

    def test_empty_drain_emits_nothing(self):
        d = DeltaDrain(drain_alpha=0.175, min_step_px=2.0)
        assert all(d.tick() == (0, 0) for _ in range(50))

    def test_accumulates_concurrent_adds(self):
        d = DeltaDrain(drain_alpha=0.175, min_step_px=2.0)
        for _ in range(5):
            d.add(4.0, 0.0)
        tx, _, _ = drain_all(d)
        assert abs(20 - tx) <= 1

    def test_backlog_does_not_build_under_steady_input(self):
        """A delta every 7ms, drained at 650Hz. The floor must outrun production."""
        d = DeltaDrain(drain_alpha=0.175, min_step_px=2.0)
        for i in range(2000):
            if i % 5 == 0:      # a delta every 5 ticks, like a frame every ~7ms
                d.add(5.0, 0.0)
            d.tick()
        assert abs(d.pending[0]) < 5.0, f"backlog grew to {d.pending[0]:.2f}px"

    @pytest.mark.parametrize("alpha,floor", [(0, 2.0), (-1, 2.0), (1.5, 2.0), (0.2, -1)])
    def test_rejects_nonsense_params(self, alpha, floor):
        with pytest.raises(ValueError):
            DeltaDrain(drain_alpha=alpha, min_step_px=floor)


class TestThreadedMouseMover:
    @pytest.fixture
    def emitted(self, monkeypatch):
        out = []
        monkeypatch.setattr(mousemover.win32api, "mouse_event",
                            lambda flags, dx, dy, *a: out.append((dx, dy)))
        return out

    def make(self, **kw):
        kw.setdefault("poll_hz", 1000)
        return ThreadedMouseMover(1.0, 1.0, 256, 0.0, 0.0, 0.0, **kw)

    def test_queues_instead_of_emitting(self, emitted):
        mm = self.make()
        mm.move_mouse_humanized(20.0, 0.0)
        assert emitted == [], "should not emit synchronously"
        assert mm.drain.pending[0] == pytest.approx(20.0)

    def test_returns_the_queued_delta(self, emitted):
        """Callers (GUI, lead buffer) must still see the movement they asked for."""
        mm = self.make()
        assert mm.move_mouse_humanized(20.0, -6.0) == (20, -6)

    def test_thread_delivers_it(self, emitted):
        mm = self.make()
        mm.start()
        try:
            mm.move_mouse_humanized(20.0, 0.0)
            deadline = time.perf_counter() + 2.0
            while sum(dx for dx, _ in emitted) < 19 and time.perf_counter() < deadline:
                time.sleep(0.005)
        finally:
            mm.stop()
        total = sum(dx for dx, _ in emitted)
        assert total >= 19, f"only {total} of 20px delivered"
        assert len(emitted) >= 5, f"delivered in {len(emitted)} events, too chunky"
        assert max(abs(dx) for dx, _ in emitted) <= 6

    def test_actually_ticks_at_high_rate(self, emitted):
        """Regression: the loop first used Event.wait(period), which Windows quantises
        to its ~15.6ms timer tick. The thread ran at 65Hz and delivered a 20px move in
        3 fat steps. time.sleep uses a high-resolution timer and measures 1.53ms for
        the same 1ms request."""
        mm = self.make(poll_hz=1000)
        mm.start()
        try:
            time.sleep(0.30)
            rate = mm.ticks / 0.30
        finally:
            mm.stop()
        assert rate > 300, f"only {rate:.0f} ticks/s — coarse timer, expect ~650"

    def test_start_is_idempotent_and_stop_joins(self, emitted):
        mm = self.make()
        mm.start()
        first = mm._thread
        mm.start()
        assert mm._thread is first
        mm.stop()
        assert mm._thread is None
        assert not first.is_alive()

    def test_thread_is_daemon(self, emitted):
        mm = self.make()
        mm.start()
        try:
            assert mm._thread.daemon, "must not keep the process alive"
        finally:
            mm.stop()

    def test_idle_thread_emits_nothing(self, emitted):
        mm = self.make()
        mm.start()
        time.sleep(0.05)
        mm.stop()
        assert emitted == []

    def test_stop_without_start_is_safe(self, emitted):
        self.make().stop()

    def test_rejects_bad_poll_hz(self):
        with pytest.raises(ValueError):
            self.make(poll_hz=0)


class TestWiring:
    def test_config_off_gives_the_direct_mover(self, cfg):
        cfg["input_settings"]["separate_mouse_thread"] = False
        mover = bootstrap.create_mousemover(cfg)
        assert type(mover) is MouseMover

    def test_config_on_gives_a_started_thread(self, cfg, monkeypatch):
        monkeypatch.setattr(mousemover.win32api, "mouse_event", lambda *a: None)
        cfg["input_settings"]["separate_mouse_thread"] = True
        mover = bootstrap.create_mousemover(cfg)
        try:
            assert isinstance(mover, ThreadedMouseMover)
            assert mover._thread is not None and mover._thread.is_alive()
            mt = cfg["input_settings"]["mouse_thread_config"]
            assert mover.poll_hz == mt["poll_hz"]
            assert mover.drain.min_step_px == mt["min_step_px"]
            assert mover.drain.drain_alpha == mt["drain_alpha"]
            assert mover.drain.output_alpha == mt["output_alpha"]
        finally:
            mover.stop()

    def test_both_movers_scale_identically(self, cfg, monkeypatch):
        """The threaded one must only change WHEN movement goes out, not how much."""
        monkeypatch.setattr(mousemover.win32api, "mouse_event", lambda *a: None)
        direct = bootstrap.create_mousemover(cfg)
        cfg["input_settings"]["separate_mouse_thread"] = True
        threaded = bootstrap.create_mousemover(cfg)
        try:
            for d in (3.0, 12.0, 40.0, -25.0):
                assert direct.move_mouse_humanized(d, d / 2) == \
                    threaded.move_mouse_humanized(d, d / 2)
        finally:
            threaded.stop()


class TestPull:
    """Recoil pull: raw screen px, no sensitivity curve, no humanization."""

    @pytest.fixture
    def emitted(self, monkeypatch):
        out = []
        monkeypatch.setattr(mousemover.win32api, "mouse_event",
                            lambda flags, dx, dy, *a: out.append((dx, dy)))
        return out

    def test_direct_mover_emits_immediately(self, emitted):
        MouseMover(0.5, 1.0, 256, 0.0, 0.0, 0.0).pull(0.0, 20.0)
        assert emitted == [(0, 20)]

    def test_pull_ignores_the_sensitivity_curve(self, emitted):
        """20px of recoil must move 20px, not 20 * overall_sens."""
        MouseMover(0.1, 0.0, 256, 0.0, 0.0, 0.0).pull(0.0, 20.0)
        assert emitted == [(0, 20)]

    def test_threaded_mover_queues_it(self, emitted):
        mm = ThreadedMouseMover(1.0, 1.0, 256, 0.0, 0.0, 0.0)
        mm.pull(0.0, 20.0)
        assert emitted == []
        assert mm.drain.pending[1] == pytest.approx(20.0)

    def test_pull_and_aim_share_one_buffer(self, emitted):
        """Both feed the same drain, so they sum instead of fighting."""
        mm = ThreadedMouseMover(1.0, 1.0, 256, 0.0, 0.0, 0.0)
        mm.move_mouse_humanized(0.0, -6.0)
        mm.pull(0.0, 20.0)
        assert mm.drain.pending[1] == pytest.approx(14.0)

    def test_zero_pull_emits_nothing(self, emitted):
        MouseMover(1.0, 1.0, 256, 0.0, 0.0, 0.0).pull(0.0, 0.0)
        assert emitted == []


class TestOutputEma:
    """output_alpha: a second EMA over the emitted step, so speed ramps up."""

    def test_ramps_instead_of_starting_at_full_speed(self):
        plain = DeltaDrain(drain_alpha=0.175, min_step_px=0.0)
        ramped = DeltaDrain(drain_alpha=0.175, min_step_px=0.0, output_alpha=0.5)
        plain.add(40.0, 0.0)
        ramped.add(40.0, 0.0)
        first_plain = plain.tick()[0] + plain._carry_x
        first_ramped = ramped.tick()[0] + ramped._carry_x
        assert first_ramped < first_plain * 0.75, (
            f"no ramp: {first_ramped:.2f} vs {first_plain:.2f}")

    def test_speed_builds_over_successive_ticks(self):
        d = DeltaDrain(drain_alpha=0.038, min_step_px=0.0, output_alpha=0.32)
        d.add(60.0, 0.0)
        speeds = []
        for _ in range(4):
            before = d.pending[0]
            d.tick()
            speeds.append(before - d.pending[0])
        assert speeds == sorted(speeds), f"speed did not build: {speeds}"

    def test_alpha_is_literally_the_fraction(self):
        """output_alpha 0.5 from rest gives exactly half the drain step."""
        d = DeltaDrain(drain_alpha=0.2, min_step_px=0.0, output_alpha=0.5)
        d.add(100.0, 0.0)
        d.tick()
        assert 100.0 - d.pending[0] == pytest.approx(100.0 * 0.2 * 0.5)

    def test_floor_overrides_the_ema(self):
        """The linear phase wins, otherwise the ramp would stall the tail."""
        d = DeltaDrain(drain_alpha=0.175, min_step_px=2.0, output_alpha=0.03)
        d.add(20.0, 0.0)
        moved = 20.0 - d.pending[0]
        d.tick()
        moved = 20.0 - d.pending[0]
        assert moved >= 2.0 - 1e-9, f"ema starved the floor: only {moved:.3f}px"

    def test_still_delivers_everything(self):
        d = DeltaDrain(drain_alpha=0.175, min_step_px=2.0, output_alpha=0.5)
        d.add(20.0, 0.0)
        tx, _, _ = drain_all(d)
        assert abs(20 - tx) <= 1, f"delivered {tx} of 20"

    def test_ema_rests_when_idle_so_the_next_burst_ramps_again(self):
        d = DeltaDrain(drain_alpha=0.175, min_step_px=0.0, output_alpha=0.32)
        d.add(40.0, 0.0)
        for _ in range(20):
            d.tick()
        d._px = d._py = 0.0
        for _ in range(5):
            d.tick()
        assert d._ema_x == 0.0 and d._ema_y == 0.0

        d.add(40.0, 0.0)
        before = d.pending[0]
        d.tick()
        first = before - d.pending[0]
        assert first < 3.0, f"resumed at old speed: {first:.2f}px"

    def test_alpha_one_is_no_smoothing(self):
        off = DeltaDrain(drain_alpha=0.175, min_step_px=2.0, output_alpha=1.0)
        ref = DeltaDrain(drain_alpha=0.175, min_step_px=2.0)
        off.add(17.0, 9.0)
        ref.add(17.0, 9.0)
        assert drain_all(off) == drain_all(ref)

    def test_direction_survives_the_ramp(self):
        d = DeltaDrain(drain_alpha=0.175, min_step_px=2.0, output_alpha=0.5)
        d.add(20.0, 10.0)
        tx, ty, _ = drain_all(d)
        assert abs(tx / ty - 2.0) < 0.35, f"path bent: ({tx}, {ty})"

    def test_rejects_out_of_range_alpha(self):
        with pytest.raises(ValueError):
            DeltaDrain(drain_alpha=0.175, min_step_px=2.0, output_alpha=-1.0)
