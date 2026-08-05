"""DetectionPipeline routing + crosshair splitting, with stubbed models.

The four-way routing (base+scan / base-only / sr-crop / hysteresis) is the most
stateful decision in the codebase and is invisible from the outside — it only shows
up as a frame that silently detected nothing. Driven here directly through
_run_engine_path so every branch is pinned. Needs a GPU only because the sr path
really slices the frame; no .engine files required.
"""

import numpy as np
import pytest
import torch

cp = pytest.importorskip("cupy")

from src.aimbot.engine.detection_pipeline import DetectionPipeline  # noqa: E402
from tests.support.fakes import det_rows  # noqa: E402

pytestmark = pytest.mark.gpu

BASE = "base"
SCAN = "scan_sr"
SR = "sr"


class StubModel:
    """Stands in for Model / the deprecated SRBundleEngine. Records which arm ran.

    Output shape convention matches the real thing: base returns (M, 6); the SR
    bundles return (1, M, 6) because callers strip the batch dim.
    """

    def __init__(self, log, tag, out=None, source_size=80):
        if out is None:
            out = (cp.empty((0, 6), dtype=cp.float32) if tag == BASE
                   else cp.empty((1, 0, 6), dtype=cp.float32))
        self.log = log
        self.tag = tag
        self.out = out
        self.source_size = source_size
        self.last_input = None

    def inference_cp(self, src):
        self.log.append(self.tag)
        self.last_input = src
        return self.out


class StubSR:
    """Stands in for SRModel. Callable, and owns its own crop/threshold geometry."""

    def __init__(self, log, out=None, crop_size=80, bb_side=48, sr_scale=1):
        self.log = log
        self.crop_size = crop_size
        self.bb_side = bb_side
        self.sr_scale = sr_scale
        self.out = torch.zeros((1, 0, 6)) if out is None else out
        self.last_input = None

    def __call__(self, crop):
        self.log.append(SR)
        self.last_input = crop
        return self.out


class StubBase:
    def __init__(self, log, out=None):
        self.model = StubModel(log, BASE, out)
        self.model_ext = ".engine"
        self.log = log

    def _preprocess_cp(self, frame):
        self.log.append("preprocess")
        return frame


def make_pipeline(log, *, scan=None, sr=None, hysteresis=2, base_out=None):
    p = object.__new__(DetectionPipeline)
    p.base_model = StubBase(log, base_out)
    p.scan_sr = scan
    p.sr = sr
    p.sr_hysteresis_frames = hysteresis
    p.union_nms_iou = 0.25
    p.conf_threshold = 0.25
    p.hsv_detector = None
    p.hsv_bypass_tracker = True
    p.cfg = {"targeting_settings": {"crosshair_cls_id": 2, "model_predict_crosshair": False}}
    return p


def locked_box(side):
    """10-col tracker row whose bbox has the given max side."""
    return np.array([100, 100, 100 + side, 100 + side, 1, 0.9, 0, 0, 0, 30], dtype=np.float32)


@pytest.fixture(scope="module")
def frame():
    """Preprocessed-frame stand-in. Real array because the sr path slices it."""
    return cp.zeros((1, 3, 640, 640), dtype=cp.float32)


class TestRouting:
    def test_no_ads_runs_base_only_when_scan_disabled(self, frame):
        log = []
        p = make_pipeline(log)
        p._run_engine_path(frame, ads=False, locked=None, locked_lifetime=0)
        assert log == ["preprocess", BASE]

    def test_no_ads_runs_base_plus_scan(self, frame):
        log = []
        p = make_pipeline(log, scan=StubModel(log, SCAN, np.empty((1, 0, 6), np.float32)))
        p._run_engine_path(frame, ads=False, locked=None, locked_lifetime=0)
        assert log == ["preprocess", BASE, SCAN]

    def test_ads_without_lock_uses_scan_path(self, frame):
        log = []
        p = make_pipeline(log, sr=StubSR(log))
        p._run_engine_path(frame, ads=True, locked=None, locked_lifetime=0)
        assert SR not in log

    def test_ads_small_fresh_lock_runs_sr_only(self, frame):
        log = []
        p = make_pipeline(log, sr=StubSR(log))
        p._run_engine_path(frame, ads=True, locked=locked_box(20), locked_lifetime=0)
        assert log == ["preprocess", SR]
        assert BASE not in log, "base is deliberately skipped on the sr path"

    def test_ads_large_fresh_lock_runs_base_only(self, frame):
        log = []
        p = make_pipeline(log, sr=StubSR(log), scan=StubModel(log, SCAN))
        p._run_engine_path(frame, ads=True, locked=locked_box(200), locked_lifetime=0)
        assert log == ["preprocess", BASE], "big target is easy; skip SR and scan"

    @pytest.mark.parametrize("lifetime", [1, 2])
    def test_stale_small_lock_inside_budget_keeps_cropping(self, lifetime, frame):
        log = []
        p = make_pipeline(log, hysteresis=2, sr=StubSR(log))
        p._run_engine_path(frame, ads=True, locked=locked_box(20), locked_lifetime=lifetime)
        assert log == ["preprocess", SR]

    def test_stale_small_lock_past_budget_bails_to_scan(self, frame):
        log = []
        p = make_pipeline(log, hysteresis=2, sr=StubSR(log),
                          scan=StubModel(log, SCAN, np.empty((1, 0, 6), np.float32)))
        p._run_engine_path(frame, ads=True, locked=locked_box(20), locked_lifetime=3)
        assert log == ["preprocess", BASE, SCAN]

    def test_stale_large_lock_bails_to_scan_immediately(self, frame):
        log = []
        p = make_pipeline(log, sr=StubSR(log))
        p._run_engine_path(frame, ads=True, locked=locked_box(200), locked_lifetime=1)
        assert log == ["preprocess", BASE]

    def test_hysteresis_zero_is_fresh_only(self, frame):
        log = []
        p = make_pipeline(log, hysteresis=0, sr=StubSR(log))
        p._run_engine_path(frame, ads=True, locked=locked_box(20), locked_lifetime=0)
        assert log == ["preprocess", SR]
        log.clear()
        p._run_engine_path(frame, ads=True, locked=locked_box(20), locked_lifetime=1)
        assert SR not in log

    def test_threshold_boundary_is_strict(self, frame):
        """bb_max_side < threshold, so a box exactly at the threshold is 'large'."""
        log = []
        p = make_pipeline(log, sr=StubSR(log, bb_side=48))
        p._run_engine_path(frame, ads=True, locked=locked_box(48), locked_lifetime=0)
        assert log == ["preprocess", BASE]
        log.clear()
        p._run_engine_path(frame, ads=True, locked=locked_box(47), locked_lifetime=0)
        assert log == ["preprocess", SR]

    def test_threshold_comes_from_the_model(self, frame):
        """The model owns bb_side; there is no config override any more."""
        log = []
        p = make_pipeline(log, sr=StubSR(log, bb_side=200))
        p._run_engine_path(frame, ads=True, locked=locked_box(100), locked_lifetime=0)
        assert log == ["preprocess", SR], "100px lock is 'small' when the model says 200"

    def test_non_square_lock_uses_longest_side(self, frame):
        log = []
        p = make_pipeline(log, sr=StubSR(log))
        locked = np.array([0, 0, 10, 200, 1, 0.9, 0, 0, 0, 30], dtype=np.float32)
        p._run_engine_path(frame, ads=True, locked=locked, locked_lifetime=0)
        assert SR not in log, "200px tall counts as large even though it's 10px wide"

    def test_sr_disabled_falls_back_to_base(self, frame):
        log = []
        p = make_pipeline(log, sr=None)
        p._run_engine_path(frame, ads=True, locked=locked_box(20), locked_lifetime=0)
        assert log == ["preprocess", BASE]


class TestCrosshairRouting:
    def _pipeline(self, hsv_row, *, bypass, model_predicts=False):
        p = make_pipeline([])
        p.hsv_bypass_tracker = bypass
        p.cfg["targeting_settings"]["model_predict_crosshair"] = model_predicts
        p.hsv_detector = type("D", (), {"detect": staticmethod(lambda f: hsv_row)})()
        return p

    def test_model_crosshair_rows_dropped_when_disabled(self):
        p = self._pipeline(det_rows([]), bypass=True, model_predicts=False)
        dets = det_rows([[0, 0, 10, 10, 0.9, 0], [5, 5, 15, 15, 0.9, 2]])
        out, bypass = p._apply_crosshair_routing(dets, None)
        assert out.shape[0] == 1 and out[0, 5] == 0
        assert bypass.shape == (0, 6)

    def test_model_crosshair_rows_kept_when_enabled(self):
        p = self._pipeline(det_rows([]), bypass=True, model_predicts=True)
        dets = det_rows([[0, 0, 10, 10, 0.9, 0], [5, 5, 15, 15, 0.9, 2]])
        out, _ = p._apply_crosshair_routing(dets, None)
        assert out.shape[0] == 2

    def test_hsv_row_bypasses_tracker(self):
        hsv = det_rows([[100, 100, 164, 164, 1.0, 2]])
        p = self._pipeline(hsv, bypass=True)
        out, bypass = p._apply_crosshair_routing(det_rows([[0, 0, 10, 10, 0.9, 0]]), None)
        assert out.shape[0] == 1, "hsv row must not reach the tracker"
        np.testing.assert_array_equal(bypass, hsv)

    def test_hsv_row_joins_tracker_input_when_not_bypassing(self):
        hsv = det_rows([[100, 100, 164, 164, 1.0, 2]])
        p = self._pipeline(hsv, bypass=False)
        out, bypass = p._apply_crosshair_routing(det_rows([[0, 0, 10, 10, 0.9, 0]]), None)
        assert out.shape[0] == 2
        assert bypass.shape == (0, 6)

    def test_empty_hsv_result_is_a_noop(self):
        p = self._pipeline(det_rows([]), bypass=True)
        dets = det_rows([[0, 0, 10, 10, 0.9, 0]])
        out, bypass = p._apply_crosshair_routing(dets, None)
        assert out.shape[0] == 1 and bypass.shape == (0, 6)

    def test_no_hsv_detector_is_a_noop(self):
        p = make_pipeline([])
        dets = det_rows([[0, 0, 10, 10, 0.9, 0]])
        out, bypass = p._apply_crosshair_routing(dets, None)
        assert out.shape[0] == 1 and bypass.shape == (0, 6)
