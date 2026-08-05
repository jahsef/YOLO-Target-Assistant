"""MouseMover scaling / humanization. win32api is stubbed so nothing actually moves."""

import math

import pytest

from src.aimbot.input import mousemover
from src.aimbot.input.mousemover import MouseMover


@pytest.fixture
def no_real_mouse(monkeypatch):
    calls = []
    monkeypatch.setattr(mousemover.win32api, "mouse_event",
                        lambda flags, dx, dy, *a: calls.append((flags, dx, dy)))
    return calls


def make(sens=0.64, scaling=0.0, max_deltas=256, jitter=0.0, os_strength=0.0, os_chance=0.0):
    return MouseMover(sens, scaling, max_deltas, jitter, os_strength, os_chance)


class TestScaleDelta:
    def test_zero_stays_zero(self):
        assert make()._scale_delta(0) == 0.0

    def test_unit_scaling_is_pure_sensitivity(self):
        """sens_scaling == 1 collapses the blend term, leaving sens * delta."""
        mm = make(sens=0.5, scaling=1.0)
        assert mm._scale_delta(100) == pytest.approx(50.0)
        assert mm._scale_delta(-100) == pytest.approx(-50.0)

    def test_matches_the_documented_curve(self):
        mm = make(sens=0.64, scaling=0.0, max_deltas=256)
        for d in (1, 7, 32, 100, 255):
            blend = 1 - math.exp(-(d / 256) * 8)
            assert mm._scale_delta(d) == pytest.approx(0.64 * d * (1 - blend))

    def test_sign_is_preserved(self):
        mm = make()
        for d in (1, 5, 50, 200):
            assert mm._scale_delta(d) > 0
            assert mm._scale_delta(-d) < 0
            assert mm._scale_delta(-d) == pytest.approx(-mm._scale_delta(d))

    def test_small_deltas_keep_near_full_sensitivity(self):
        """The whole point of the steep curve: fine corrections aren't thrown away."""
        mm = make(sens=1.0, scaling=0.0, max_deltas=256)
        assert mm._scale_delta(2) / 2 > 0.9

    def test_large_deltas_are_damped_below_the_peak(self):
        """Non-monotonic on purpose — output peaks near max_deltas/8 and falls off,
        which is what stops wild flicks."""
        mm = make(sens=1.0, scaling=0.0, max_deltas=256)
        assert mm._scale_delta(32) > mm._scale_delta(200)
        assert mm._scale_delta(32) > mm._scale_delta(5)


class TestHumanize:
    def test_disabled_knobs_are_identity(self):
        assert make()._humanize_movement(12.0, -8.0) == (12.0, -8.0)

    def test_jitter_stays_within_its_bound(self):
        mm = make(jitter=0.1)
        for _ in range(200):
            dx, dy = mm._humanize_movement(10.0, -20.0)
            bound = 0.1 * 30.0  # jitter_strength * (|dx| + |dy|)
            assert abs(dx - 10.0) <= bound and abs(dy + 20.0) <= bound

    def test_overshoot_always_fires_at_chance_one(self):
        mm = make(os_strength=2.0, os_chance=1.0)
        assert mm._humanize_movement(10.0, -5.0) == (20.0, -10.0)

    def test_overshoot_never_fires_at_chance_zero(self):
        mm = make(os_strength=2.0, os_chance=0.0)
        assert mm._humanize_movement(10.0, -5.0) == (10.0, -5.0)


class TestMoveMouse:
    def test_emits_one_relative_move_and_returns_it(self, no_real_mouse):
        mm = make(sens=1.0, scaling=1.0)
        out = mm.move_mouse_humanized(10.4, -3.6)
        assert out == (10, -4)
        assert len(no_real_mouse) == 1
        flags, dx, dy = no_real_mouse[0]
        assert (dx, dy) == (10, -4)
        assert flags == mousemover.win32con.MOUSEEVENTF_MOVE

    def test_emits_ints_not_floats(self, no_real_mouse):
        make().move_mouse_humanized(3.7, 9.2)
        _, dx, dy = no_real_mouse[0]
        assert isinstance(dx, int) and isinstance(dy, int)

    def test_subpixel_move_rounds_to_zero(self, no_real_mouse):
        """Sub-pixel deltas collapse to a no-op move — the aimbot still 'moves',
        it just sends (0, 0). Pinned because it's the floor on fine correction."""
        mm = make(sens=0.01, scaling=1.0)
        assert mm.move_mouse_humanized(1.0, 1.0) == (0, 0)
