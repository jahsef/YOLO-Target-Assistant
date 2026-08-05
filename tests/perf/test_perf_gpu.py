"""GPU hot-loop costs. Needs CUDA; the engine-backed ones also need local model files."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from src.aimbot.engine.hsv_crosshair import (  # noqa: E402
    _HSV_COLOR_CENTER, _HSV_COLOR_RANGE, _HSV_S_MAX, _HSV_S_MIN, _HSV_V_MAX,
    _HSV_V_MIN, _HS_RATIO_THRESHOLD, HSVCrosshairDetector, fused_red_mask_suppress,
)
from src.aimbot.engine.model import Model  # noqa: E402
from tests.perf.conftest import hot_trials  # noqa: E402

pytestmark = [pytest.mark.perf, pytest.mark.gpu]


@pytest.fixture(scope="module")
def hsv():
    return HSVCrosshairDetector("heuristic_spam", crosshair_cls_id=2,
                                frame_hw=(640, 640), center_crop_hw=(240, 240))


class TestHsv:
    def test_detect_all_schemes(self, perf, gpu_frame, cuda_sync):
        for scheme in HSVCrosshairDetector.VOTING_SCHEMES:
            d = HSVCrosshairDetector(scheme, 2, (640, 640), (240, 240))
            perf.record(f"hsv.detect[{scheme}]",
                        hot_trials(lambda d=d: d.detect(gpu_frame), iters=200,
                                   warm=100, sync=cuda_sync),
                        group="hsv", note="240x240 roi, fastest of 6")

    def test_fused_pipeline_launch_only(self, perf, hsv, gpu_frame, cuda_sync):
        """The whole heuristic_spam chain minus the single host sync."""
        roi = gpu_frame[hsv._y0:hsv._y0 + hsv.roi_h, hsv._x0:hsv._x0 + hsv.roi_w]
        perf.record("hsv.fused_pipeline (launch, no sync)",
                    hot_trials(lambda: hsv._launch_pipeline(roi), iters=400,
                               warm=200, sync=cuda_sync),
                    group="hsv", note="one kernel, no D2H")

    def test_mask_kernel_alone(self, perf, hsv, gpu_frame, cuda_sync):
        roi = gpu_frame[hsv._y0:hsv._y0 + hsv.roi_h, hsv._x0:hsv._x0 + hsv.roi_w]
        perf.record("hsv.fused_red_mask_suppress",
                    hot_trials(lambda: fused_red_mask_suppress(
                        roi, _HSV_COLOR_CENTER, _HSV_COLOR_RANGE, s_min=_HSV_S_MIN,
                        v_min=_HSV_V_MIN, s_max=_HSV_S_MAX, v_max=_HSV_V_MAX,
                        ratio=_HS_RATIO_THRESHOLD, out=hsv._hs_mask_buf,
                        validate_device=False),
                        iters=400, warm=200, sync=cuda_sync),
                    group="hsv", note="mask + row-suppress only")

    def test_single_device_sync_cost(self, perf, cuda_sync):
        """Reference point: what one host<-device sync costs. The old HSV vote paid
        three of these per frame; the fused path pays one."""
        scalar = cp.zeros((), dtype=cp.float32)
        perf.record("one host<-device sync (float())",
                    hot_trials(lambda: float(scalar), iters=400, warm=200),
                    group="hsv", note="baseline for sync-count changes")


class TestPreprocess:
    def test_preprocess_cp(self, perf, gpu_frame, cuda_sync):
        pre = Model._preprocess_cp.__get__(object.__new__(Model))
        perf.record("model._preprocess_cp",
                    hot_trials(lambda: pre(gpu_frame), iters=400, warm=200, sync=cuda_sync),
                    group="gpu misc", note="640x640 hwc u8 -> nchw f32")


class TestMemory:
    def test_frame_copy(self, perf, gpu_frame, cuda_sync):
        """Cost the async capture stage pays to own its pixels across the handoff."""
        perf.record("frame.copy() (640x640x3 u8)",
                    hot_trials(lambda: gpu_frame.copy(), iters=400, warm=200, sync=cuda_sync),
                    group="gpu misc", note="async capture ownership")

    def test_d2h_small(self, perf, cuda_sync):
        buf = cp.empty(3, dtype=cp.float64)
        perf.record("D2H 3 float64 (.get())",
                    hot_trials(buf.get, iters=400, warm=200),
                    group="gpu misc", note="hsv result transfer")
