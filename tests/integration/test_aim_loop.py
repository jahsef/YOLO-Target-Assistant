"""Tracker + TargetSelector + Aimbot.aimbot + MouseMover wired together.

Exercises the aiming half of the loop as a unit, across all three tracker backends,
with no GPU and no models. This is the behavioral baseline the async refactor must
not disturb.
"""

import numpy as np
import pytest

from src.aimbot import bootstrap
from tests.support.loop_harness import LoopHarness

TRACKERS = ["ultralytics", "ultralytics_vectorized", "cpp"]
CENTER = 320


def enemy_det(cx, cy, w=40, h=80, conf=0.9, cls=0):
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, conf, cls]


@pytest.fixture(params=TRACKERS)
def make_harness(request, cfg):
    if request.param != "ultralytics":
        pytest.importorskip("c_bytetracker")
    cfg["tracker_settings"]["tracker_impl"] = request.param

    def _make(**targeting_overrides):
        cfg["targeting_settings"].update(targeting_overrides)
        return LoopHarness(cfg, bootstrap.create_tracker(cfg, ".engine"))
    return _make


@pytest.fixture
def harness(make_harness):
    return make_harness()


def settle(harness, det, frames=6, **kw):
    """Feed the same detection for N frames so the track ages past min_frames_to_target."""
    out = None
    for _ in range(frames):
        out = harness.step([det], **kw)
    return out


class TestAimDirection:
    def test_moves_toward_a_target_right_of_center(self, harness):
        raw, scaled = settle(harness, enemy_det(CENTER + 100, CENTER))
        assert raw[0] > 0, "target to the right -> positive dx"
        assert scaled[0] > 0

    def test_moves_toward_a_target_left_of_center(self, harness):
        raw, _ = settle(harness, enemy_det(CENTER - 100, CENTER))
        assert raw[0] < 0

    def test_aims_above_center_of_mass(self, harness):
        """head offset + bullet drop both push the aim point up."""
        raw, _ = settle(harness, enemy_det(CENTER, CENTER))
        assert raw[1] < 0

    def test_no_movement_without_detections(self, harness):
        for _ in range(4):
            raw, scaled = harness.step([])
        assert raw == (0, 0) and scaled == (0, 0)
        assert harness.moves == []

    def test_no_movement_while_inactive(self, harness):
        settle(harness, enemy_det(CENTER + 100, CENTER), active=False)
        assert harness.moves == []


class TestTrackAgeGate:
    def test_first_frame_does_not_shoot(self, harness):
        """min_frames_to_target=2 — a brand new track is not aimed at yet."""
        raw, _ = harness.step([enemy_det(CENTER + 100, CENTER)])
        assert raw == (0, 0)
        assert harness.moves == []

    def test_track_eventually_matures(self, harness):
        moved = False
        for _ in range(8):
            raw, _ = harness.step([enemy_det(CENTER + 100, CENTER)])
            moved = moved or raw != (0, 0)
        assert moved, "track never aged past min_frames_to_target"


class TestCrosshairReference:
    def test_bypass_reticle_survives_the_min_age_gate(self, make_harness):
        """A bypass reticle carries no lifetime (it never went through the tracker), so
        the min_frames_to_target filter must not judge it as a young track — otherwise
        _get_crosshair never sees it and aim silently measures from the window centre."""
        harness = make_harness(lead_target=False)
        target = enemy_det(CENTER + 100, CENTER)
        for _ in range(6):
            raw_center, _ = harness.step([target])
        assert raw_center[0] == 100, "sanity: dx measured from the window center"

        reticle = [CENTER + 50 - 32, CENTER - 32, CENTER + 50 + 32, CENTER + 32, 1.0, 2]
        for _ in range(3):
            raw_shifted, _ = harness.step([target], bypass_rows=[reticle])
        assert raw_shifted[0] == 50, "reticle 50px right of centre should halve dx"

    def test_reticle_is_the_aim_origin_with_the_gate_wide_open(self, make_harness):
        """Same result with min_frames_to_target=0, i.e. the gate is not what makes
        the reticle work."""
        harness = make_harness(lead_target=False, min_frames_to_target=0)
        target = enemy_det(CENTER + 100, CENTER)
        for _ in range(6):
            raw_center, _ = harness.step([target])
        assert raw_center[0] == 100

        reticle = [CENTER + 50 - 32, CENTER - 32, CENTER + 50 + 32, CENTER + 32, 1.0, 2]
        for _ in range(3):
            raw_shifted, _ = harness.step([target], bypass_rows=[reticle])
        assert raw_shifted[0] == 50

    def test_bypass_row_is_not_shot_at(self, harness):
        """A reticle row has zero lifetime, so the min-age filter drops it from
        targeting even though _get_crosshair still uses it."""
        reticle = [CENTER - 32, CENTER - 32, CENTER + 32, CENTER + 32, 1.0, 2]
        for _ in range(6):
            raw, _ = harness.step([], bypass_rows=[reticle])
        assert raw == (0, 0)


class TestMultiTarget:
    def test_picks_the_closest_enemy(self, harness):
        near = enemy_det(CENTER + 40, CENTER)
        far = enemy_det(CENTER + 200, CENTER)
        for _ in range(6):
            raw, _ = harness.step([near, far])
        assert 0 < raw[0] < 100, f"should chase the near target, got dx={raw[0]}"

    def test_follows_a_target_across_the_screen(self, harness):
        # 8px/frame on a 40px-wide box keeps IoU at ~0.67, above match_thresh 0.6.
        # Bigger steps break association every frame and the track never matures.
        xs = list(range(CENTER - 120, CENTER + 121, 8))
        seen = []
        for x in xs:
            raw, _ = harness.step([enemy_det(x, CENTER)])
            seen.append(raw[0])
        tail = [d for d in seen[4:] if d != 0]
        assert tail, "never locked on during the sweep"
        # dx should flip sign as the target crosses the crosshair
        assert min(tail) < 0 < max(tail)


class TestStateHygiene:
    def test_movement_buffer_tracks_raw_deltas(self, harness):
        settle(harness, enemy_det(CENTER + 100, CENTER))
        assert np.any(harness.target_selector.buffer.buf != 0)

    def test_zoom_ramps_only_while_ads(self, harness):
        settle(harness, enemy_det(CENTER, CENTER), frames=10, ads=True)
        zoomed = harness.target_selector.zoom
        assert zoomed > 1.0
        harness.step([enemy_det(CENTER, CENTER)], ads=False)
        assert harness.target_selector.zoom == 1.0

    def test_prev_detection_survives_a_dropped_frame(self, harness):
        settle(harness, enemy_det(CENTER + 60, CENTER))
        assert harness.target_selector._prev_detection is not None
        harness.step([])
        assert harness.target_selector._prev_detection is not None
        assert harness.target_selector._prev_detection_lifetime >= 1


class TestRecoilPull:
    """on_click_pull_down_px fires on the LMB rising edge."""

    def press(self, harness, down):
        harness.inputdetector.is_lmb_pressed = down

    def test_pulls_once_per_press(self, make_harness):
        h = make_harness()
        h.cfg["input_settings"]["on_click_pull_down_px"] = 20
        pulls = []
        h.bot.mousemover.pull = lambda dx, dy: pulls.append((dx, dy))
        self.press(h, True)
        for _ in range(5):
            h.step([])
        assert pulls == [(0.0, 20.0)], "held button must not pull every frame"

    def test_release_then_press_pulls_again(self, make_harness):
        h = make_harness()
        h.cfg["input_settings"]["on_click_pull_down_px"] = 15
        pulls = []
        h.bot.mousemover.pull = lambda dx, dy: pulls.append((dx, dy))
        for down in (True, True, False, True, True):
            self.press(h, down)
            h.step([])
        assert pulls == [(0.0, 15.0), (0.0, 15.0)]

    def test_zero_disables_it(self, make_harness):
        h = make_harness()
        h.cfg["input_settings"]["on_click_pull_down_px"] = 0
        pulls = []
        h.bot.mousemover.pull = lambda dx, dy: pulls.append((dx, dy))
        self.press(h, True)
        h.step([])
        assert pulls == []

    def test_master_toggle_gates_it(self, make_harness):
        h = make_harness()
        h.cfg["input_settings"]["on_click_pull_down_px"] = 20
        h.inputdetector.is_toggled = False
        pulls = []
        h.bot.mousemover.pull = lambda dx, dy: pulls.append((dx, dy))
        self.press(h, True)
        h.step([])
        assert pulls == []

    def test_requires_ads(self, make_harness):
        """Firing from the hip must not pull."""
        h = make_harness()
        h.cfg["input_settings"]["on_click_pull_down_px"] = 20
        pulls = []
        h.bot.mousemover.pull = lambda dx, dy: pulls.append((dx, dy))
        self.press(h, True)
        h.step([], ads=False)
        assert pulls == []

    def test_pulls_while_ads_without_a_target(self, make_harness):
        """Recoil control is about the gun, not the target — no detections needed."""
        h = make_harness()
        h.cfg["input_settings"]["on_click_pull_down_px"] = 20
        pulls = []
        h.bot.mousemover.pull = lambda dx, dy: pulls.append((dx, dy))
        self.press(h, True)
        h.step([], ads=True)
        assert pulls == [(0.0, 20.0)]

    def test_click_without_ads_does_not_arm_a_later_pull(self, make_harness):
        """Click from the hip, then ADS while still holding: no new click, no pull."""
        h = make_harness()
        h.cfg["input_settings"]["on_click_pull_down_px"] = 20
        pulls = []
        h.bot.mousemover.pull = lambda dx, dy: pulls.append((dx, dy))
        self.press(h, True)
        h.step([], ads=False)
        for _ in range(3):
            h.step([], ads=True)
        assert pulls == []
