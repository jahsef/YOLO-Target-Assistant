"""HSVCrosshairDetector end-to-end behavior.

_ReferenceHeuristicSpam re-implements the CURRENT heuristic_spam chain (fused mask ->
torch opening -> torch density -> gaussian weighted center) independently of src. It
is the equivalence oracle: if the production path is ever rewritten (e.g. fused into
one kernel), it must still agree with this.
"""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from src.aimbot.engine.hsv_crosshair import (  # noqa: E402
    _GAUSS_EDGE_FACTOR,
    _HSV_BOX_SIZE,
    _HS_RATIO_THRESHOLD,
    _HSV_COLOR_CENTER,
    _HSV_COLOR_RANGE,
    _HSV_S_MAX,
    _HSV_S_MIN,
    _HSV_V_MAX,
    _HSV_V_MIN,
    HSVCrosshairDetector,
    fused_red_mask_suppress,
)
from tests.support.fakes import crosshair_frame, nametag_bar  # noqa: E402

pytestmark = pytest.mark.gpu

FRAME_HW = (640, 640)
CROP_HW = (240, 240)
CLS_ID = 2


def make_detector(scheme="heuristic_spam", crop=CROP_HW):
    return HSVCrosshairDetector(voting_scheme=scheme, crosshair_cls_id=CLS_ID,
                                frame_hw=FRAME_HW, center_crop_hw=crop)


def frame_with_reticle_at(cy, cx, **kw):
    return cp.asarray(crosshair_frame(FRAME_HW[0], FRAME_HW[1], cy, cx, **kw))


class _ReferenceHeuristicSpam:
    """Independent re-implementation of today's heuristic_spam chain."""

    def __init__(self, roi_h, roi_w):
        self.opening = nn.Sequential(
            nn.ZeroPad2d((1, 0, 1, 0)),
            _MinPool(2, 1),
            nn.MaxPool2d(3, 1, 1),
        ).cuda().eval()
        self.density = nn.Sequential(
            nn.AvgPool2d(3, 2, 1), nn.AvgPool2d(3, 2, 1)
        ).cuda().eval()
        with torch.no_grad():
            dh, dw = self.density(torch.zeros((1, 1, roi_h, roi_w), device="cuda")).shape[2:]
        self.dh, self.dw = int(dh), int(dw)
        self.stride_y, self.stride_x = roi_h / self.dh, roi_w / self.dw

    def __call__(self, roi_cp):
        mask = fused_red_mask_suppress(
            roi_cp, _HSV_COLOR_CENTER, _HSV_COLOR_RANGE, s_min=_HSV_S_MIN,
            v_min=_HSV_V_MIN, s_max=_HSV_S_MAX, v_max=_HSV_V_MAX,
            ratio=_HS_RATIO_THRESHOLD)
        t = torch.from_dlpack(mask.astype(cp.float32))[None, None]
        with torch.no_grad():
            pooled = self.density(self.opening(t))[0, 0]
        return _gaussian_center(pooled)


class _MinPool(nn.Module):
    def __init__(self, k, s):
        super().__init__()
        self.mp = nn.MaxPool2d(k, s)

    def forward(self, x):
        return -self.mp(-x)


def _gaussian_center(dense: torch.Tensor):
    """Weighted centroid in density-grid coords, or None if the map is empty."""
    h, w = dense.shape
    ys = torch.arange(h, device=dense.device, dtype=torch.float32).reshape(h, 1)
    xs = torch.arange(w, device=dense.device, dtype=torch.float32).reshape(1, w)
    sy = (h * 0.5) / _GAUSS_EDGE_FACTOR
    sx = (w * 0.5) / _GAUSS_EDGE_FACTOR
    weights = torch.exp(-(((ys - (h - 1) * 0.5) ** 2) / (2 * sy * sy)
                          + ((xs - (w - 1) * 0.5) ** 2) / (2 * sx * sx)))
    wm = weights * dense
    total = float(wm.sum())
    if total <= 0.0:
        return None
    return float((wm * ys).sum()) / total, float((wm * xs).sum()) / total


def box_center(row):
    return (row[0, 1] + row[0, 3]) / 2, (row[0, 0] + row[0, 2]) / 2  # (cy, cx)


# --- output contract ----------------------------------------------------------

class TestDetectContract:
    @pytest.mark.parametrize("scheme", HSVCrosshairDetector.VOTING_SCHEMES)
    def test_finds_a_centered_reticle(self, scheme):
        det = make_detector(scheme)
        row = det.detect(frame_with_reticle_at(320, 320))
        assert row.shape == (1, 6)
        cy, cx = box_center(row)
        assert abs(cy - 320) < 8 and abs(cx - 320) < 8, f"{scheme} drifted to {(cy, cx)}"

    @pytest.mark.parametrize("scheme", HSVCrosshairDetector.VOTING_SCHEMES)
    def test_empty_frame_returns_no_rows(self, scheme):
        blank = cp.zeros((*FRAME_HW, 3), dtype=cp.uint8)
        out = make_detector(scheme).detect(blank)
        assert out.shape == (0, 6)
        assert out.dtype == np.float32

    def test_row_format(self):
        row = make_detector().detect(frame_with_reticle_at(320, 320))
        x1, y1, x2, y2, conf, cls = row[0]
        assert conf == 1.0
        assert cls == CLS_ID
        assert x2 - x1 == pytest.approx(_HSV_BOX_SIZE)
        assert y2 - y1 == pytest.approx(_HSV_BOX_SIZE)
        assert row.dtype == np.float32

    def test_coords_are_full_frame_not_crop_local(self):
        """Reticle 40px up-left of screen center must come back near (280, 280) in
        base-region coords, i.e. the crop origin was added back."""
        det = make_detector()
        cy, cx = box_center(det.detect(frame_with_reticle_at(280, 280)))
        assert abs(cy - 280) < 8 and abs(cx - 280) < 8

    def test_reticle_outside_the_crop_is_not_reported(self):
        """Crop is 240x240 centered on a 640x640 frame -> only y,x in [200, 440)."""
        det = make_detector()
        out = det.detect(frame_with_reticle_at(60, 60))
        if out.shape[0]:
            cy, cx = box_center(out)
            assert 200 <= cy < 440 and 200 <= cx < 440

    def test_no_crop_uses_whole_frame(self):
        det = make_detector(crop=None)
        assert (det.roi_h, det.roi_w) == FRAME_HW
        assert (det._y0, det._x0) == (0, 0)
        cy, cx = box_center(det.detect(frame_with_reticle_at(400, 300)))
        assert abs(cy - 400) < 10 and abs(cx - 300) < 10

    def test_rejects_unknown_scheme(self):
        with pytest.raises(ValueError, match="voting_scheme"):
            HSVCrosshairDetector("vibes", CLS_ID, FRAME_HW, CROP_HW)


# --- heuristic_spam equivalence + robustness ----------------------------------

class TestHeuristicSpam:
    def test_matches_the_reference_chain(self):
        det = make_detector()
        ref = _ReferenceHeuristicSpam(det.roi_h, det.roi_w)
        for cy, cx in [(320, 320), (300, 340), (250, 260), (420, 410)]:
            frame = frame_with_reticle_at(cy, cx)
            roi = frame[det._y0:det._y0 + det.roi_h, det._x0:det._x0 + det.roi_w]
            got = det._process(roi)
            want = ref(roi)
            assert (got is None) == (want is None)
            if got is not None:
                np.testing.assert_allclose(
                    [got[0], got[1]],
                    [want[0] * ref.stride_y, want[1] * ref.stride_x],
                    rtol=1e-4, atol=1e-3)

    def test_density_grid_is_a_4x_downsample(self):
        det = make_detector()
        assert (det._hs_dh, det._hs_dw) == (60, 60)
        assert det._hs_stride_y == pytest.approx(4.0)

    def test_reuses_its_mask_buffer(self):
        det = make_detector()
        before = det._hs_mask_buf.data.ptr
        for _ in range(3):
            det.detect(frame_with_reticle_at(320, 320))
        assert det._hs_mask_buf.data.ptr == before

    def test_survives_a_nametag_false_positive(self):
        """A wide thin nametag bar 30px above the reticle must not drag the centroid
        onto it — row-suppression plus density smoothing is the whole point."""
        det = make_detector()
        frame = crosshair_frame(*FRAME_HW, 320, 320)
        nametag_bar(frame, 290, 320, half_w=60)
        cy, cx = box_center(det.detect(cp.asarray(frame)))
        assert abs(cy - 320) < 10, f"nametag pulled the centroid to y={cy}"
        assert abs(cx - 320) < 10

    def test_beats_raw_weighted_center_under_false_positives(self):
        """The docstring's claim: coarse density voting is more FP-robust than running
        weighted_center straight on the mask."""
        frame = crosshair_frame(*FRAME_HW, 320, 320)
        for dy in (-40, -30, 34, 44):
            nametag_bar(frame, 320 + dy, 300, half_w=55)
        frame_gpu = cp.asarray(frame)
        spam_cy, spam_cx = box_center(make_detector("heuristic_spam").detect(frame_gpu))
        wc_cy, wc_cx = box_center(make_detector("weighted_center").detect(frame_gpu))
        spam_err = abs(spam_cy - 320) + abs(spam_cx - 320)
        wc_err = abs(wc_cy - 320) + abs(wc_cx - 320)
        assert spam_err <= wc_err, f"heuristic_spam {spam_err:.1f} vs weighted_center {wc_err:.1f}"
