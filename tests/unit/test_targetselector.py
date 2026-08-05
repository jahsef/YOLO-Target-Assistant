"""Correctness of the targeting math. Pure CPU.

A few of these pin behavior that is surprising but deliberate (the lead EMA seeded at
BASE_LEAD_SENS, the first lead_target call counting as a target switch). If one fails
after a refactor, check whether the change was intended before "fixing" the test.
"""

import math

import numpy as np
import pytest

from src.aimbot.data_parsing.targetselector import (
    RSIDampener,
    RingBuffer2D,
    TargetSelector,
    _RSIChannel,
)
from tests.support.fakes import FakeFPS, enemy, tracked_rows


@pytest.fixture
def ts(cfg):
    return TargetSelector(cfg=cfg, detection_window_dim=(640, 640),
                          screen_hw=(1440, 2560), fps_tracker=FakeFPS(144.0))


# --- RingBuffer2D -------------------------------------------------------------

class TestRingBuffer2D:
    def test_ordered_is_most_recent_first(self):
        buf = RingBuffer2D(4)
        for i, (x, y) in enumerate([(1, 10), (2, 20), (3, 30)]):
            buf.push(x, y)
        got = buf.ordered()
        assert got.shape == (4, 2)
        np.testing.assert_array_equal(got[:3], [[3, 30], [2, 20], [1, 10]])
        np.testing.assert_array_equal(got[3], [0, 0])  # unwritten slot

    def test_wraps_and_overwrites_oldest(self):
        buf = RingBuffer2D(3)
        for i in range(5):
            buf.push(i, -i)
        np.testing.assert_array_equal(buf.ordered(), [[4, -4], [3, -3], [2, -2]])

    def test_newest(self):
        buf = RingBuffer2D(3)
        buf.push(7, 8)
        np.testing.assert_array_equal(buf.newest, [7, 8])
        buf.push(9, 10)
        np.testing.assert_array_equal(buf.newest, [9, 10])

    def test_newest_wraps_to_last_slot(self):
        buf = RingBuffer2D(2)
        buf.push(1, 1)
        buf.push(2, 2)  # idx back to 0
        np.testing.assert_array_equal(buf.newest, [2, 2])

    def test_decay_scales_everything(self):
        buf = RingBuffer2D(3)
        buf.push(10, -20)
        buf.decay(0.3)
        np.testing.assert_allclose(buf.newest, [3.0, -6.0])

    def test_push_does_not_allocate_new_buffer(self):
        buf = RingBuffer2D(4)
        before = buf.buf
        buf.push(1, 2)
        assert buf.buf is before


# --- RSI ----------------------------------------------------------------------

class TestRSI:
    def test_channel_first_update_is_none(self):
        assert _RSIChannel(4).update(5.0) is None

    def test_channel_monotonic_rise_never_produces_rsi(self):
        """avg_loss stays 0 so the channel returns None forever — by design, the
        divide would blow up. Callers treat None as 'no opinion'."""
        ch = _RSIChannel(4)
        assert all(ch.update(float(i)) is None for i in range(20))

    def test_channel_oscillation_lands_in_range(self):
        ch = _RSIChannel(4)
        vals = [v for v in (ch.update(10.0 + 5 * math.sin(i)) for i in range(40)) if v is not None]
        assert vals, "oscillating input should eventually produce RSI"
        assert all(0.0 <= v <= 100.0 for v in vals)

    def test_dampener_first_call_is_unity(self):
        assert RSIDampener().update(3.0, 4.0) == 1.0

    def test_dampener_constant_magnitude_is_unity(self):
        d = RSIDampener()
        for _ in range(50):
            assert d.update(3.0, 4.0) == 1.0

    def test_dampener_output_is_bounded(self):
        d = RSIDampener(periods=(4, 8), k=12)
        rng = np.random.default_rng(0)
        for _ in range(300):
            f = d.update(float(rng.normal(0, 5)), float(rng.normal(0, 5)))
            assert 0.33 <= f <= 1.0

    def test_dampener_uses_magnitude_not_sign(self):
        a, b = RSIDampener(), RSIDampener()
        for i in range(30):
            m = 1.0 + i % 3
            fa = a.update(m, 0.0)
            fb = b.update(0.0, -m)
        assert fa == pytest.approx(fb)


# --- distance / drop ----------------------------------------------------------

class TestDistance:
    def test_degenerate_box_returns_floor(self, ts):
        # 0 is falsy for both args -> no estimates at all -> MIN_DISTANCE, not NaN
        assert ts._calculate_distance(0, 0) == ts.MIN_DISTANCE

    def test_bigger_target_is_closer(self, ts):
        near = ts._calculate_distance(target_height_pixels=200, target_width_pixels=140)
        far = ts._calculate_distance(target_height_pixels=20, target_width_pixels=14)
        assert near < far

    def test_zoom_pushes_distance_out(self, ts):
        ts.zoom = 1.0
        d1 = ts._calculate_distance(target_height_pixels=60, target_width_pixels=42)
        ts.zoom = 2.0
        d2 = ts._calculate_distance(target_height_pixels=60, target_width_pixels=42)
        assert d2 > d1

    def test_single_axis_estimates_are_usable(self, ts):
        assert ts._calculate_distance(target_height_pixels=60, target_width_pixels=None) > 0
        assert ts._calculate_distance(target_height_pixels=None, target_width_pixels=42) > 0

    @staticmethod
    def _blend_position(ts, h, w):
        """Where the blend sits between the height-only and width-only estimates.
        0 = entirely height, 1 = entirely width."""
        blended = ts._calculate_distance(target_height_pixels=h, target_width_pixels=w)
        d_h = ts._calculate_distance(target_height_pixels=h, target_width_pixels=None)
        d_w = ts._calculate_distance(target_height_pixels=None, target_width_pixels=w)
        return (blended - d_h) / (d_w - d_h)

    def test_occluded_box_leans_on_the_width_estimate(self, ts):
        """A squat box means the legs are cut off, so its height-derived distance is
        the less trustworthy half of the blend."""
        assert self._blend_position(ts, h=60, w=100) > 0.5

    @staticmethod
    def _reference_blend(ts, h, w, height_trust):
        """The blend written out longhand, with trust as a free parameter so the
        occlusion term can be switched off independently of the box shape."""
        eff_v = 2 * np.atan(np.tan(ts.vfov_rad / 2) / ts.zoom)
        eff_h = 2 * np.atan(np.tan(ts.hfov_rad / 2) / ts.zoom)
        th = h / (ts.screen_height / eff_v)
        tw = w / (ts.screen_width / eff_h)
        d_h = ts.TARGET_REAL_HEIGHT / (2 * math.tan(th / 2))
        d_w = ts.TARGET_REAL_WIDTH / (2 * math.tan(tw / 2))
        wt_h, wt_w = h ** 2 * height_trust, w ** 2
        blended = (d_h * wt_h + d_w * wt_w) / (wt_h + wt_w)
        return max(ts.MIN_DISTANCE, blended * ts.DISTANCE_CALIBRATION_FACTOR)

    @staticmethod
    def _trust(ts, h, w):
        expected = ts.TARGET_REAL_HEIGHT / ts.TARGET_REAL_WIDTH
        return max(0.3, min(1.0, (h / w) / expected)) ** 2

    def test_matches_the_inverse_variance_blend(self, ts):
        for h, w in [(60, 100), (143, 100), (200, 60)]:
            assert ts._calculate_distance(h, w) == pytest.approx(
                self._reference_blend(ts, h, w, self._trust(ts, h, w)))

    def test_small_targets_are_weighted_less(self, ts):
        """Weight goes as pixels**2, so the axis with more pixels on it drives the
        blend. A tall thin box should lean on its height estimate."""
        h, w = 200, 40
        blended = ts._calculate_distance(target_height_pixels=h, target_width_pixels=w)
        d_h = ts._calculate_distance(target_height_pixels=h, target_width_pixels=None)
        d_w = ts._calculate_distance(target_height_pixels=None, target_width_pixels=w)
        assert abs(blended - d_h) < abs(blended - d_w)

    def test_occlusion_term_pulls_the_blend_toward_width(self, ts):
        """The occlusion adjustment in isolation: same box, trust on vs off."""
        h, w = 60, 100
        d_w = ts._calculate_distance(target_height_pixels=None, target_width_pixels=w)
        adjusted = self._reference_blend(ts, h, w, self._trust(ts, h, w))
        unadjusted = self._reference_blend(ts, h, w, 1.0)
        assert abs(adjusted - d_w) < abs(unadjusted - d_w)

    def test_squatter_boxes_are_trusted_less(self, ts):
        """Trust falls monotonically as the box gets wider relative to its height."""
        trusts = [self._trust(ts, h=h, w=100) for h in (60, 80, 110, 143)]
        assert trusts == sorted(trusts)
        assert trusts[-1] == 1.0, "a correctly proportioned box keeps full trust"

    def test_taller_than_expected_is_not_rewarded(self, ts):
        """Trust clamps at 1.0, so an unusually tall box can't out-weight the width."""
        assert self._trust(ts, h=300, w=100) == 1.0

    def test_drop_grows_with_flight_time(self, ts):
        assert ts._calculate_bullet_drop(0.2) > ts._calculate_bullet_drop(0.1)
        assert ts._calculate_bullet_drop(0.0) == 0.0

    def test_travel_time_scales_with_distance(self, ts):
        assert ts._calculate_travel_time(2870.0) == pytest.approx(1.0)


# --- selection ----------------------------------------------------------------

class TestSelection:
    def test_l1_distances(self):
        dets = tracked_rows([enemy(0, 0, 10, 10), enemy(100, 100, 110, 110)])
        got = TargetSelector._l1_distances(dets, (5.0, 5.0))
        np.testing.assert_allclose(got, [0.0, 200.0])

    def test_closest_detection(self, ts):
        dets = tracked_rows([enemy(300, 300, 340, 380, 1), enemy(0, 0, 40, 80, 2)])
        pick, dist = ts._get_closest_detection(dets, (320, 340))
        assert pick[4] == 1
        assert dist == pytest.approx(0.0)

    def test_crosshair_from_crosshair_class_row(self, ts):
        dets = tracked_rows([
            enemy(0, 0, 40, 80, 1),
            enemy(300, 300, 340, 340, 9, cls=2),  # crosshair class
        ])
        assert ts._get_crosshair(dets) == (320, 320)

    def test_crosshair_falls_back_to_window_center(self, ts):
        dets = tracked_rows([enemy(0, 0, 40, 80, 1)])
        assert ts._get_crosshair(dets) == (320, 320)

    def test_detection_window_center_is_xy(self, cfg):
        """detection_window_dim arrives as (h, w) and the centre is consumed as an
        (x, y) point, so a non-square capture region must not come out transposed."""
        t = TargetSelector(cfg=cfg, detection_window_dim=(480, 640),
                           screen_hw=(1440, 2560), fps_tracker=FakeFPS())
        assert t.detection_window_center == (320, 240)  # (w//2, h//2)

    def test_select_enemy_returns_none_without_enemy_class(self, ts):
        dets = tracked_rows([enemy(300, 300, 340, 340, 9, cls=2)])
        assert ts._select_enemy(dets, prioritize_oldest=False) is None

    def test_select_enemy_closest_mode_ignores_lru(self, ts):
        dets = tracked_rows([enemy(0, 0, 40, 80, 1), enemy(300, 300, 340, 380, 2)])
        pick, crosshair = ts._select_enemy(dets, prioritize_oldest=False)
        assert pick[4] == 2
        assert ts.target_lru == {}  # closest mode must not touch the LRU

    def test_priority_target_seeds_lru_on_first_call(self, ts):
        dets = tracked_rows([enemy(0, 0, 40, 80, 1), enemy(300, 300, 340, 380, 2)])
        pick = ts._get_highest_priority_target(dets, (320, 340))
        assert pick[4] == 2
        assert list(ts.target_lru) == [2.0]

    def test_priority_sticks_to_known_target_within_hysteresis(self, ts):
        crosshair = (320, 340)
        known = enemy(320, 300, 360, 380, 1)  # center (340, 340) -> l1 20
        near = enemy(315, 300, 355, 380, 2)   # center (335, 340) -> l1 15
        ts._get_highest_priority_target(tracked_rows([known]), crosshair)  # seed id 1
        pick = ts._get_highest_priority_target(tracked_rows([known, near]), crosshair)
        assert pick[4] == 1, "20 < 15*1.5, so the known target keeps the lock"

    def test_priority_drops_known_target_outside_hysteresis(self, ts):
        crosshair = (320, 340)
        far = enemy(0, 0, 40, 80, 1)
        near = enemy(300, 300, 340, 380, 2)
        ts._get_highest_priority_target(tracked_rows([far]), crosshair)
        pick = ts._get_highest_priority_target(tracked_rows([far, near]), crosshair)
        assert pick[4] == 2

    def test_lru_is_capped(self, ts):
        crosshair = (320, 320)
        for tid in range(80):
            ts._get_highest_priority_target(tracked_rows([enemy(0, 0, 10, 10, tid)]), crosshair)
        assert len(ts.target_lru) <= 64


# --- deltas -------------------------------------------------------------------

class TestDeltas:
    def test_rounds_and_signs(self, ts):
        assert ts._get_deltas((110.4, 89.6), (100, 100)) == (10, -10)

    def test_zeroes_when_at_or_past_max(self, ts):
        m = ts.max_deltas
        assert ts._get_deltas((m, 0), (0, 0)) == (0, 0)
        assert ts._get_deltas((m - 1, 0), (0, 0)) == (m - 1, 0)
        assert ts._get_deltas((0, -m), (0, 0)) == (0, 0)

    def test_get_deltas_no_enemies(self, ts):
        assert ts.get_deltas(tracked_rows([enemy(0, 0, 10, 10, 1, cls=2)])) == (0, 0)

    def test_head_offset_aims_above_center(self, cfg):
        cfg["targeting_settings"]["predict_drop"] = False
        cfg["targeting_settings"]["lead_target"] = False
        box = enemy(300, 200, 340, 300, 1)  # center y = 250, h = 100
        with_head = TargetSelector(cfg=cfg, detection_window_dim=(640, 640),
                                   screen_hw=(1440, 2560), fps_tracker=FakeFPS())
        dy_head = with_head.get_deltas(tracked_rows([box]))[1]

        cfg2 = dict(cfg)
        cfg2["targeting_settings"] = dict(cfg["targeting_settings"], head_toggle=False)
        no_head = TargetSelector(cfg=cfg2, detection_window_dim=(640, 640),
                                 screen_hw=(1440, 2560), fps_tracker=FakeFPS())
        dy_plain = no_head.get_deltas(tracked_rows([box]))[1]

        assert dy_head == dy_plain - round(100 * cfg["targeting_settings"]["base_head_offset"])

    def test_drop_compensation_aims_higher_still(self, cfg):
        cfg["targeting_settings"]["lead_target"] = False
        # small box = distant target; drop scales with distance, so a near target's
        # compensation rounds away below a pixel
        box = enemy(300, 200, 316, 240, 1)
        cfg_nodrop = dict(cfg, targeting_settings=dict(cfg["targeting_settings"], predict_drop=False))
        drop = TargetSelector(cfg, (640, 640), (1440, 2560), FakeFPS()).get_deltas(tracked_rows([box]))[1]
        plain = TargetSelector(cfg_nodrop, (640, 640), (1440, 2560), FakeFPS()).get_deltas(tracked_rows([box]))[1]
        assert drop < plain, "bullet drop should raise the aim point (smaller dy)"


# --- lead ---------------------------------------------------------------------

class TestLead:
    def test_no_lead_before_fps_warmup(self, cfg):
        t = TargetSelector(cfg, (640, 640), (1440, 2560), FakeFPS(0.0))
        for _ in range(32):
            t.update_movement_buffer((5, 3))
        assert t.lead_target(0.05, (5, 3), target_id=1, track_age=64) == (0.0, 0.0)

    def test_age_warmup_drains_lead_but_not_instantly(self, ts):
        """age_factor is 0 at track_age 0, yet _lead_sens_ema is seeded at
        BASE_LEAD_SENS — so frame one still leads a little and only decays from
        there. Pinned because 'age 0 means no lead' is the intuitive-but-wrong read."""
        for _ in range(64):
            ts.update_movement_buffer((5, 3))
        first = ts.lead_target(0.05, (5, 3), target_id=1, track_age=0)
        assert first[0] > 0
        for _ in range(40):
            later = ts.lead_target(0.05, (5, 3), target_id=1, track_age=0)
        assert abs(later[0]) < abs(first[0]) * 0.1

    def test_mature_track_leads_more_than_young_one(self, ts):
        for _ in range(64):
            ts.update_movement_buffer((5, 3))
        young = ts.lead_target(0.05, (5, 3), target_id=1, track_age=1)
        ts._lead_sens_ema = ts.BASE_LEAD_SENS  # reset the EMA so age is the only variable
        old = ts.lead_target(0.05, (5, 3), target_id=1, track_age=ts.LEAD_AGE_WARMUP_FRAMES)
        assert old[0] > young[0]

    def test_lead_follows_movement_direction(self, ts):
        for _ in range(64):
            ts.update_movement_buffer((6, -4))
        for _ in range(40):  # let the EMA spin up
            lead = ts.lead_target(0.05, (6, -4), target_id=1, track_age=64)
        assert lead[0] > 0 and lead[1] < 0

    def test_target_switch_decays_buffer(self, ts):
        for _ in range(64):
            ts.update_movement_buffer((10, 10))
        # first call counts as a switch too (last_target_id starts None), so settle first
        ts.lead_target(0.05, (10, 10), target_id=1, track_age=10)
        before = ts.buffer.newest.copy()
        ts.lead_target(0.05, (10, 10), target_id=2, track_age=10)  # switch
        np.testing.assert_allclose(ts.buffer.newest, before * ts.TARGET_SWITCH_DECAY)

    def test_first_lead_call_counts_as_a_switch(self, ts):
        """PINNING: last_target_id starts None, so the very first lead_target call
        decays a buffer that was just filled by update_movement_buffer."""
        for _ in range(64):
            ts.update_movement_buffer((10, 10))
        before = ts.buffer.newest.copy()
        ts.lead_target(0.05, (10, 10), target_id=1, track_age=10)
        np.testing.assert_allclose(ts.buffer.newest, before * ts.TARGET_SWITCH_DECAY)

    def test_no_decay_when_target_unchanged(self, ts):
        for _ in range(8):
            ts.update_movement_buffer((10, 10))
        ts.lead_target(0.05, (10, 10), target_id=7, track_age=10)
        before = ts.buffer.newest.copy()
        ts.lead_target(0.05, (10, 10), target_id=7, track_age=10)
        np.testing.assert_allclose(ts.buffer.newest, before)

    def test_movement_buffer_records_raw_deltas(self, ts):
        ts.update_movement_buffer((13, -7))
        np.testing.assert_array_equal(ts.buffer.newest, [13, -7])


# --- zoom + routing state -----------------------------------------------------

class TestZoomAndRouting:
    def test_zoom_ramps_to_final_and_clamps(self, ts):
        for _ in range(ts.zoom_interpolation_frames + 10):
            ts.update_zoom_interpolation()
        assert ts.zoom == pytest.approx(ts.final_zoom)
        assert ts.zoom_progress == 1.0

    def test_zoom_is_monotonic_while_ramping(self, ts):
        prev = ts.zoom
        for _ in range(ts.zoom_interpolation_frames):
            ts.update_zoom_interpolation()
            assert ts.zoom >= prev
            prev = ts.zoom

    def test_reset_zoom(self, ts):
        for _ in range(20):
            ts.update_zoom_interpolation()
        ts.reset_zoom()
        assert ts.zoom == ts.base_zoom and ts.zoom_progress == 0.0

    def test_prev_detection_refreshes_on_enemy(self, ts):
        ts.update_prev_detection(tracked_rows([enemy(0, 0, 40, 80, 1)]))
        assert ts._prev_detection_lifetime == 0
        assert ts._prev_detection[4] == 1

    def test_prev_detection_retained_and_aged_on_empty_frame(self, ts):
        ts.update_prev_detection(tracked_rows([enemy(0, 0, 40, 80, 1)]))
        ts.update_prev_detection(tracked_rows([]))
        ts.update_prev_detection(tracked_rows([]))
        assert ts._prev_detection[4] == 1, "stale lock must survive for the hysteresis path"
        assert ts._prev_detection_lifetime == 2

    def test_prev_detection_ages_on_crosshair_only_frame(self, ts):
        ts.update_prev_detection(tracked_rows([enemy(0, 0, 40, 80, 1)]))
        ts.update_prev_detection(tracked_rows([enemy(300, 300, 340, 340, 9, cls=2)]))
        assert ts._prev_detection_lifetime == 1

    def test_prev_detection_swaps_to_new_top_priority(self, ts):
        ts.update_prev_detection(tracked_rows([enemy(0, 0, 40, 80, 1)]))
        ts.update_prev_detection(tracked_rows([enemy(310, 310, 330, 330, 2)]))
        assert ts._prev_detection[4] == 2
        assert ts._prev_detection_lifetime == 0
