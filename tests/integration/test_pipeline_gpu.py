"""Cross-model NMS and the sr-crop coordinate math. GPU, no engines."""

import numpy as np
import pytest
import torch

cp = pytest.importorskip("cupy")

from src.aimbot.engine.detection_pipeline import DetectionPipeline  # noqa: E402

pytestmark = pytest.mark.gpu


def bare_pipeline(**attrs):
    p = object.__new__(DetectionPipeline)
    p.union_nms_iou = 0.25
    p.conf_threshold = 0.25
    for k, v in attrs.items():
        setattr(p, k, v)
    return p


class TestUnionNms:
    def test_empty_passthrough(self):
        p = bare_pipeline()
        out = p._union_nms(cp.empty((0, 6), dtype=cp.float32))
        assert out.shape == (0, 6)

    def test_dedupes_between_models(self):
        """Same target found by base and scan_sr with slightly different boxes."""
        p = bare_pipeline()
        dets = cp.asarray([[100, 100, 140, 180, 0.80, 0],
                           [102, 101, 141, 179, 0.91, 0]], dtype=cp.float32)
        out = cp.asnumpy(p._union_nms(dets))
        assert out.shape[0] == 1
        assert out[0, 4] == pytest.approx(0.91), "higher-confidence box wins"

    def test_keeps_distinct_targets(self):
        p = bare_pipeline()
        dets = cp.asarray([[0, 0, 40, 80, 0.9, 0],
                           [300, 300, 340, 380, 0.8, 0]], dtype=cp.float32)
        assert p._union_nms(dets).shape[0] == 2

    def test_is_class_aware(self):
        """An enemy box and a crosshair box on the same pixels must both survive."""
        p = bare_pipeline()
        dets = cp.asarray([[100, 100, 140, 140, 0.9, 0],
                           [100, 100, 140, 140, 0.9, 2]], dtype=cp.float32)
        assert p._union_nms(dets).shape[0] == 2

    def test_output_is_cupy_float32(self):
        p = bare_pipeline()
        out = p._union_nms(cp.asarray([[0, 0, 10, 10, 0.9, 0]], dtype=cp.float32))
        assert isinstance(out, cp.ndarray) and out.dtype == cp.float32


class SRSpy:
    """SRModel stand-in that records the crop it was handed.

    Returns baked-NMS format (B, max_det, 6), boxes in upscaled px — what the real
    engine emits, since the model upscales internally.
    """

    def __init__(self, crop_size=80, sr_scale=1, out=None):
        self.crop_size = crop_size
        self.sr_scale = sr_scale
        self.bb_side = 48
        self.crops = []
        self.out = torch.zeros((1, 0, 6)) if out is None else out

    def __call__(self, crop):
        self.crops.append(crop)
        return self.out


def baked(rows):
    return cp.asarray([rows], dtype=cp.float32)


class TestSRCrop:
    @pytest.fixture
    def frame(self):
        return cp.zeros((1, 3, 640, 640), dtype=cp.float32)

    def locked(self, cx, cy, side=20):
        h = side / 2
        return np.array([cx - h, cy - h, cx + h, cy + h, 1, 0.9, 0, 0, 0, 30], dtype=np.float32)

    def test_crop_is_centered_on_the_lock(self, frame):
        spy = SRSpy(80)
        p = bare_pipeline(sr=spy)
        p._run_sr_crop(frame, self.locked(320, 240))
        assert spy.crops[0].shape == (1, 3, 80, 80)

    def test_crop_origin_clamps_at_the_top_left(self, frame):
        # lock at (5, 5) would want origin (-35, -35)
        spy = SRSpy(80, out=baked([[0, 0, 10, 10, 0.9, 0]]))
        p = bare_pipeline(sr=spy)
        out = cp.asnumpy(p._run_sr_crop(frame, self.locked(5, 5)))
        np.testing.assert_allclose(out[0, :4], [0, 0, 10, 10]), "origin clamped to 0"

    def test_crop_origin_clamps_at_the_bottom_right(self, frame):
        spy = SRSpy(80, out=baked([[0, 0, 10, 10, 0.9, 0]]))
        p = bare_pipeline(sr=spy)
        out = cp.asnumpy(p._run_sr_crop(frame, self.locked(639, 639)))
        # max origin is 640 - 80 = 560
        np.testing.assert_allclose(out[0, :4], [560, 560, 570, 570])

    def test_detections_translate_back_to_base_coords(self, frame):
        spy = SRSpy(80, out=baked([[10, 20, 30, 40, 0.9, 0]]))
        p = bare_pipeline(sr=spy)
        out = cp.asnumpy(p._run_sr_crop(frame, self.locked(320, 240)))
        # origin = (320-40, 240-40) = (280, 200)
        np.testing.assert_allclose(out[0, :4], [290, 220, 310, 240])
        assert out[0, 4] == pytest.approx(0.9) and out[0, 5] == 0

    def test_boxes_are_divided_down_by_sr_scale(self, frame):
        """The model upscales internally, so its boxes are in crop_size * sr_scale px.
        Skipping the divide would place every detection 4x too far from the origin."""
        spy = SRSpy(80, sr_scale=4, out=baked([[40, 80, 120, 160, 0.9, 0]]))
        p = bare_pipeline(sr=spy)
        out = cp.asnumpy(p._run_sr_crop(frame, self.locked(320, 240)))
        # /4 -> [10, 20, 30, 40], then + origin (280, 200)
        np.testing.assert_allclose(out[0, :4], [290, 220, 310, 240])

    def test_padded_rows_are_dropped(self, frame):
        """Baked-NMS engines zero-pad to max_det; those rows must not become boxes."""
        spy = SRSpy(80, out=baked([[10, 20, 30, 40, 0.9, 0], [0, 0, 0, 0, 0, 0]]))
        p = bare_pipeline(sr=spy)
        assert p._run_sr_crop(frame, self.locked(320, 240)).shape == (1, 6)

    def test_below_conf_threshold_is_dropped(self, frame):
        spy = SRSpy(80, out=baked([[10, 20, 30, 40, 0.1, 0]]))
        p = bare_pipeline(sr=spy)
        assert p._run_sr_crop(frame, self.locked(320, 240)).shape == (0, 6)

    def test_accepts_a_torch_return(self, frame):
        """The model may hand back torch or cupy; both are the same GPU memory."""
        spy = SRSpy(80, out=torch.tensor([[[10, 20, 30, 40, 0.9, 0]]],
                                         dtype=torch.float32, device="cuda"))
        p = bare_pipeline(sr=spy)
        out = cp.asnumpy(p._run_sr_crop(frame, self.locked(320, 240)))
        np.testing.assert_allclose(out[0, :4], [290, 220, 310, 240])

    def test_empty_detection_set_skips_translation(self, frame):
        p = bare_pipeline(sr=SRSpy(80))
        out = p._run_sr_crop(frame, self.locked(320, 240))
        assert out.shape == (0, 6)

    def test_crop_is_contiguous_for_trt(self, frame):
        spy = SRSpy(80)
        p = bare_pipeline(sr=spy)
        p._run_sr_crop(frame, self.locked(320, 240))
        assert spy.crops[0].is_contiguous(), "TRT bindings need a contiguous input"

    def test_crop_is_handed_over_as_torch(self, frame):
        """SRModel.__call__ does x.to(...) — a cupy array would AttributeError."""
        spy = SRSpy(80)
        p = bare_pipeline(sr=spy)
        p._run_sr_crop(frame, self.locked(320, 240))
        assert isinstance(spy.crops[0], torch.Tensor)

    def test_fp16_engine_output_comes_back_float32(self, frame):
        """An fp16 export resolves 0.5px near a 560px origin; downstream wants f32."""
        spy = SRSpy(80, out=cp.asarray([[[10, 20, 30, 40, 0.9, 0]]], dtype=cp.float16))
        p = bare_pipeline(sr=spy)
        out = p._run_sr_crop(frame, self.locked(320, 240))
        assert out.dtype == cp.float32
        np.testing.assert_allclose(cp.asnumpy(out)[0, :4], [290, 220, 310, 240])
