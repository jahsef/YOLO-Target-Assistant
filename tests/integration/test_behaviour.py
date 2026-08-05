"""Whole-system behaviour over time, with the aim->view feedback loop closed.

These ask coarse questions that per-function tests cannot: does it converge, does it
stay on a mover for 30 frames, does it recover when the detector goes blind. Thresholds
are loose on purpose — they exist to catch a system that aims the wrong way or locks
onto nothing, not to pin pixel values.

Scenario sizing is dictated by the shipped sensitivity curve, which peaks around a 40px
raw delta and collapses past it (200px -> 0.25px of mouse movement per frame). That is
the deliberate anti-flick shape, so offsets here stay inside the range the aimbot is
actually built to close.
"""

import numpy as np
import pytest

from tests.support.scenario import (
    CENTER,
    ScriptedPipeline,
    Target,
    geometry_cfg,
    linear,
    run_scenario,
    still,
)

TRACKERS = ["ultralytics_vectorized", "cpp"]


@pytest.fixture(params=TRACKERS)
def scen_cfg(request, cfg):
    cfg["tracker_settings"]["tracker_impl"] = request.param
    return cfg


@pytest.fixture
def geom_cfg(scen_cfg):
    return geometry_cfg(scen_cfg)


class TestConvergence:
    def test_converges_on_a_stationary_target(self, geom_cfg):
        pipe = ScriptedPipeline([Target(still(CENTER + 60, CENTER))])
        tr = run_scenario(geom_cfg, pipe, frames=40)
        err = tr.error()
        assert err[0] > 50, "sanity: should start far away"
        assert err[-1] < 12, f"never converged, final error {err[-1]:.1f}px"

    @pytest.mark.parametrize("dx,dy", [(60, 0), (-60, 0), (0, 60), (0, -60),
                                       (45, 45), (-45, -45), (45, -45), (-45, 45)])
    def test_converges_from_every_direction(self, geom_cfg, dx, dy):
        """Catches sign errors and x/y transposition, which look fine on one axis."""
        pipe = ScriptedPipeline([Target(still(CENTER + dx, CENTER + dy))])
        tr = run_scenario(geom_cfg, pipe, frames=40)
        assert tr.error()[-1] < 15, f"({dx},{dy}) ended {tr.error()[-1]:.1f}px out"

    def test_moves_the_right_way_on_the_first_correction(self, geom_cfg):
        """A sign flip anywhere in the chain shows up here immediately."""
        for dx, dy in [(60, 0), (-60, 0), (0, 60), (0, -60)]:
            pipe = ScriptedPipeline([Target(still(CENTER + dx, CENTER + dy))])
            tr = run_scenario(geom_cfg, pipe, frames=6)
            moves = [s for s in tr.col("scaled") if s != (0, 0)]
            assert moves, f"({dx},{dy}) never moved"
            assert np.sign(moves[0][0]) == np.sign(dx) or dx == 0
            assert np.sign(moves[0][1]) == np.sign(dy) or dy == 0

    def test_does_not_overshoot_wildly(self, geom_cfg):
        pipe = ScriptedPipeline([Target(still(CENTER + 60, CENTER))])
        tr = run_scenario(geom_cfg, pipe, frames=40)
        assert tr.error().max() <= tr.error()[0] + 5, "aim diverged before settling"

    def test_stays_put_on_a_centred_target(self, geom_cfg):
        pipe = ScriptedPipeline([Target(still(CENTER, CENTER))])
        tr = run_scenario(geom_cfg, pipe, frames=30)
        assert tr.error().max() < 15, "jittered off an already-centred target"

    def test_head_offset_biases_the_aim_upward(self, scen_cfg):
        """With the shipped config the aim point sits above centre on purpose."""
        scen_cfg["targeting_settings"].update(predict_drop=False, lead_target=False)
        pipe = ScriptedPipeline([Target(still(CENTER, CENTER))])
        tr = run_scenario(scen_cfg, pipe, frames=30)
        assert tr.rows[-1]["err_y"] > 15, "head offset not applied"


class TestChasing:
    def test_chases_a_moving_target_for_30_frames(self, geom_cfg):
        """The headline heuristic: a target sliding 4px/frame should be tracked, not
        trailed further and further behind."""
        pipe = ScriptedPipeline([Target(linear(CENTER + 30, CENTER, dx=4.0))])
        tr = run_scenario(geom_cfg, pipe, frames=30)
        settled = tr.settled_error(12)
        assert np.median(settled) < 30, f"median chase error {np.median(settled):.1f}px"

    def test_chase_error_does_not_grow(self, geom_cfg):
        """A systematic lag would show up as error climbing frame over frame."""
        pipe = ScriptedPipeline([Target(linear(CENTER + 20, CENTER, dx=3.0))])
        tr = run_scenario(geom_cfg, pipe, frames=45)
        first, second = tr.error()[15:25], tr.error()[35:45]
        assert np.median(second) <= np.median(first) + 12

    def test_follows_a_direction_change(self, geom_cfg):
        def zigzag(f):
            return (CENTER + (4.0 * f if f < 15 else 4.0 * 15 - 4.0 * (f - 15)), CENTER)
        pipe = ScriptedPipeline([Target(zigzag)])
        tr = run_scenario(geom_cfg, pipe, frames=34)
        assert np.median(tr.settled_error(24)) < 35

    def test_vertical_chase(self, geom_cfg):
        pipe = ScriptedPipeline([Target(linear(CENTER, CENTER - 30, dy=4.0))])
        tr = run_scenario(geom_cfg, pipe, frames=30)
        assert np.median(tr.settled_error(12)) < 30


class TestRecovery:
    def test_recovers_after_the_target_blinks_out(self, geom_cfg):
        gone = Target(still(CENTER + 50, CENTER), visible_to=10)
        back = Target(still(CENTER + 50, CENTER), visible_from=16)
        pipe = ScriptedPipeline([gone, back])
        tr = run_scenario(geom_cfg, pipe, frames=45, primary=1)
        assert tr.error()[-1] < 20, "never re-acquired after the gap"

    def test_no_movement_when_nothing_is_visible(self, geom_cfg):
        pipe = ScriptedPipeline([Target(still(CENTER + 50, CENTER), visible_to=0)])
        tr = run_scenario(geom_cfg, pipe, frames=12)
        assert all(s == (0, 0) for s in tr.col("scaled")), "moved with no detections"


class TestSRRouting:
    """the sr path skips the base model, so anything outside the 80px crop is invisible
    that frame. These check the system can still escape that state."""

    def test_sr_path_engages_for_a_small_target(self, geom_cfg):
        pipe = ScriptedPipeline([Target(still(CENTER + 10, CENTER), w=20, h=20)])
        tr = run_scenario(geom_cfg, pipe, frames=20, ads=True)
        assert "sr" in tr.col("route"), "never routed to the sr path"

    def test_recovers_when_the_target_leaves_the_crop(self, geom_cfg):
        """Locks small and centred, then the target steps outside the crop. The lock
        must go stale and the scan path take back over."""
        def step_out(f):
            return (CENTER + 6, CENTER) if f < 8 else (CENTER + 70, CENTER + 30)
        pipe = ScriptedPipeline([Target(step_out, w=20, h=20)])
        tr = run_scenario(geom_cfg, pipe, frames=45, ads=True)
        assert "sr" in tr.col("route")[:8], "sanity: should lock sr first"
        assert "base" in tr.col("route")[10:], "never bailed back to the scan path"
        assert tr.error()[-1] < 25, f"never re-acquired, ended {tr.error()[-1]:.1f}px out"

    def test_releasing_ads_restores_the_scan_path(self, geom_cfg):
        def step_out(f):
            return (CENTER + 6, CENTER) if f < 8 else (CENTER + 70, CENTER + 30)
        pipe = ScriptedPipeline([Target(step_out, w=20, h=20)])
        tr = run_scenario(geom_cfg, pipe, frames=45, ads=False)
        assert set(tr.col("route")) == {"base"}
        assert tr.error()[-1] < 25


class TestCrosshairReference:
    def test_reticle_offset_shifts_where_it_aims(self, geom_cfg):
        """With HSV bypass on, aim is measured from the reticle, not the window centre."""
        pipe = ScriptedPipeline([Target(still(CENTER + 60, CENTER))],
                                reticle=(CENTER + 20, CENTER))
        tr = run_scenario(geom_cfg, pipe, frames=40)
        assert tr.error()[-1] < 15
