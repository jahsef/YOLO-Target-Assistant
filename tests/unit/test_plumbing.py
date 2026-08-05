"""Config validation, tracker adapter shaping, fps math, and live-config drift."""

import json

import numpy as np
import pytest

from src.aimbot.engine.tracker_adapter import crosshair_rows_to_tracked
from src.aimbot.utils.fpstracker import FPSTracker
from tests.support.cfg import key_paths
from tests.support.fakes import det_rows


class TestValidateTargetingConfig:
    @pytest.fixture(autouse=True)
    def _import(self):
        from src.aimbot import bootstrap
        self.validate = bootstrap.validate_targeting_config

    def test_default_config_is_valid(self, cfg):
        self.validate(cfg)

    def test_lead_without_drop_is_rejected(self, cfg):
        cfg["targeting_settings"]["predict_drop"] = False
        with pytest.raises(ValueError, match="predict_drop"):
            self.validate(cfg)

    def test_drop_without_lead_is_fine(self, cfg):
        cfg["targeting_settings"]["lead_target"] = False
        self.validate(cfg)

    def test_two_crosshair_sources_rejected(self, cfg):
        cfg["targeting_settings"]["model_predict_crosshair"] = True
        cfg["targeting_settings"]["hsv_settings"]["enabled"] = True
        with pytest.raises(ValueError, match="mutually exclusive"):
            self.validate(cfg)

    def test_no_crosshair_source_is_allowed(self, cfg):
        cfg["targeting_settings"]["model_predict_crosshair"] = False
        cfg["targeting_settings"]["hsv_settings"]["enabled"] = False
        self.validate(cfg)

    def test_unknown_voting_scheme_rejected(self, cfg):
        cfg["targeting_settings"]["hsv_settings"]["voting_scheme"] = "vibes"
        with pytest.raises(ValueError, match="voting_scheme"):
            self.validate(cfg)

    def test_voting_scheme_unchecked_when_hsv_disabled(self, cfg):
        cfg["targeting_settings"]["hsv_settings"]["enabled"] = False
        cfg["targeting_settings"]["hsv_settings"]["voting_scheme"] = "vibes"
        self.validate(cfg)


class TestCrosshairRowsToTracked:
    def test_column_mapping(self):
        rows = det_rows([[10, 20, 30, 40, 0.9, 2]])
        out = crosshair_rows_to_tracked(rows)
        assert out.shape == (1, 10)
        np.testing.assert_array_equal(out[0, 0:4], [10, 20, 30, 40])
        assert out[0, 4] == -1        # track_id: untracked
        assert out[0, 5] == pytest.approx(0.9)
        assert out[0, 6] == 2
        assert out[0, 7] == -1
        np.testing.assert_array_equal(out[0, 8:10], [0, 0])

    def test_zero_lifetime_would_fail_min_frames_gate(self):
        """start_frame == last_frame == 0, so lifetime is 0. That's fine only because
        bypass rows are concatenated AFTER the tracker and the min_frames_to_target
        filter in Aimbot.aimbot() drops them from targeting — they exist to be found
        by _get_crosshair, not to be shot at."""
        out = crosshair_rows_to_tracked(det_rows([[10, 20, 30, 40, 0.9, 2]]))
        assert out[0, 9] - out[0, 8] == 0

    def test_empty_input(self):
        out = crosshair_rows_to_tracked(det_rows([]))
        assert out.shape == (0, 10)

    def test_multiple_rows(self):
        out = crosshair_rows_to_tracked(det_rows([[0, 0, 1, 1, 0.5, 2], [2, 2, 3, 3, 0.6, 2]]))
        assert out.shape == (2, 10)
        assert out.dtype == np.float32


class TestFPSTracker:
    def test_cold_tracker_reports_zero(self):
        t = FPSTracker()
        assert t.get_fps() == 0.0
        t.update()
        assert t.get_fps() == 0.0, "one timestamp spans no interval"

    def test_known_timestamps(self):
        t = FPSTracker()
        # appendleft, so buffer[0] is newest. 11 stamps 10ms apart -> 100 fps
        for i in range(11):
            t.buffer.appendleft(i * 0.01)
        assert t.get_fps() == pytest.approx(100.0)

    def test_buffer_is_capped(self):
        t = FPSTracker()
        for i in range(500):
            t.update()
        assert len(t.buffer) == t.fps_buffer_len

    def test_non_advancing_clock_is_safe(self):
        t = FPSTracker()
        for _ in range(5):
            t.buffer.appendleft(1.0)
        assert t.get_fps() == 0.0, "zero elapsed must not divide by zero"


class TestLiveConfigDrift:
    """config/cfg.json is the user's live tuning surface; tests use their own dict.
    This guards the one thing that actually matters — that the live file still
    carries every key the code reads."""

    def test_live_cfg_has_every_key_the_tests_assume(self, repo_root, cfg):
        live = json.loads((repo_root / "config" / "cfg.json").read_text())
        missing = key_paths(cfg) - key_paths(live)
        assert not missing, f"config/cfg.json is missing keys the code reads: {sorted(missing)}"
