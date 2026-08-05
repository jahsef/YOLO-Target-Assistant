"""Preprocess kernel + YOLO decode/NMS. GPU, no .engine files needed."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
torch = pytest.importorskip("torch")

from src.aimbot.engine.model import Model  # noqa: E402
from src.aimbot.engine.yolo_decode import decode_baked, decode_raw, nms_pack  # noqa: E402

pytestmark = pytest.mark.gpu


@pytest.fixture
def preprocess():
    """_preprocess_cp never touches self, so skip __init__ and the model load."""
    return Model._preprocess_cp.__get__(object.__new__(Model))


class TestPreprocess:
    def test_layout_and_scaling(self, preprocess):
        rng = np.random.default_rng(0)
        frame = rng.integers(0, 256, size=(64, 96, 3), dtype=np.uint8)
        out = cp.asnumpy(preprocess(cp.asarray(frame)))
        assert out.shape == (1, 3, 64, 96)
        assert out.dtype == np.float32
        # kernel multiplies by the float32 reciprocal; match that exactly
        want = frame.transpose(2, 0, 1)[None].astype(np.float32) * np.float32(1.0 / 255.0)
        np.testing.assert_array_equal(out, want)

    def test_channel_order_is_preserved(self, preprocess):
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        frame[..., 0] = 255  # R
        out = cp.asnumpy(preprocess(cp.asarray(frame)))
        assert out[0, 0].max() == pytest.approx(1.0)
        assert out[0, 1].max() == 0.0 and out[0, 2].max() == 0.0

    def test_extremes(self, preprocess):
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        frame[0, 0] = 255
        out = cp.asnumpy(preprocess(cp.asarray(frame)))
        assert out.min() == 0.0
        assert out.max() == pytest.approx(1.0)

    def test_accepts_a_strided_view(self, preprocess):
        rng = np.random.default_rng(1)
        full = cp.asarray(rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8))
        view = full[8:40, 8:40]
        out = cp.asnumpy(preprocess(view))
        want = cp.asnumpy(view).transpose(2, 0, 1)[None].astype(np.float32) * np.float32(1 / 255)
        np.testing.assert_array_equal(out, want)

    def test_rejects_wrong_dtype(self, preprocess):
        with pytest.raises(AssertionError):
            preprocess(cp.zeros((8, 8, 3), dtype=cp.float32))


class TestDecodeRaw:
    def _pred(self, boxes_xywh, class_scores):
        """(1, 4+nc, A) raw head output."""
        a = len(boxes_xywh)
        nc = len(class_scores[0])
        out = np.zeros((1, 4 + nc, a), dtype=np.float32)
        out[0, :4, :] = np.asarray(boxes_xywh, dtype=np.float32).T
        out[0, 4:, :] = np.asarray(class_scores, dtype=np.float32).T
        return torch.as_tensor(out, device="cuda")

    def test_xywh_to_xyxy(self):
        t = self._pred([[100, 200, 40, 60]], [[0.9, 0.1]])
        boxes, scores, classes, batch_idx = decode_raw(t, 0.25)
        np.testing.assert_allclose(boxes.cpu().numpy(), [[80, 170, 120, 230]])
        assert scores.item() == pytest.approx(0.9)
        assert classes.item() == 0
        assert batch_idx.item() == 0

    def test_threshold_is_strictly_greater(self):
        t = self._pred([[10, 10, 4, 4]], [[0.25, 0.0]])
        boxes, _, _, _ = decode_raw(t, 0.25)
        assert boxes.shape[0] == 0

    def test_takes_argmax_class(self):
        t = self._pred([[10, 10, 4, 4]], [[0.3, 0.8, 0.1]])
        _, scores, classes, _ = decode_raw(t, 0.25)
        assert classes.item() == 1 and scores.item() == pytest.approx(0.8)

    def test_batch_index_tracks_the_source_image(self):
        t = torch.zeros((3, 5, 2), device="cuda")
        t[2, 4, 1] = 0.9  # image 2, anchor 1
        _, _, _, batch_idx = decode_raw(t, 0.25)
        assert batch_idx.tolist() == [2]

    def test_empty_when_all_below_threshold(self):
        t = self._pred([[10, 10, 4, 4]], [[0.01, 0.02]])
        boxes, scores, classes, _ = decode_raw(t, 0.25)
        assert boxes.numel() == 0 and scores.numel() == 0 and classes.numel() == 0


class TestDecodeBaked:
    def test_drops_zero_padded_rows(self):
        out = torch.zeros((1, 4, 6), device="cuda")
        out[0, 0] = torch.tensor([10, 20, 30, 40, 0.9, 1.0], device="cuda")
        boxes, scores, classes, _ = decode_baked(out, 0.25)
        assert boxes.shape == (1, 4)
        np.testing.assert_allclose(boxes.cpu().numpy(), [[10, 20, 30, 40]])
        assert classes.item() == 1 and classes.dtype == torch.int64

    def test_boxes_pass_through_as_xyxy(self):
        out = torch.zeros((1, 1, 6), device="cuda")
        out[0, 0] = torch.tensor([1, 2, 3, 4, 0.5, 0.0], device="cuda")
        boxes, _, _, _ = decode_baked(out, 0.25)
        np.testing.assert_allclose(boxes.cpu().numpy(), [[1, 2, 3, 4]])


class TestNmsPack:
    def _t(self, arr):
        return torch.as_tensor(np.asarray(arr, dtype=np.float32), device="cuda")

    def test_packs_six_columns_score_sorted(self):
        boxes = self._t([[0, 0, 10, 10], [100, 100, 110, 110]])
        scores = self._t([0.4, 0.9])
        classes = torch.as_tensor([0, 0], device="cuda")
        out = nms_pack(boxes, scores, classes, 0.5).cpu().numpy()
        assert out.shape == (2, 6)
        assert out[0, 4] > out[1, 4], "highest score first"
        np.testing.assert_allclose(out[0, :4], [100, 100, 110, 110])

    def test_suppresses_overlap_within_a_class(self):
        boxes = self._t([[0, 0, 10, 10], [1, 1, 11, 11]])
        scores = self._t([0.9, 0.8])
        classes = torch.as_tensor([0, 0], device="cuda")
        assert nms_pack(boxes, scores, classes, 0.5).shape[0] == 1

    def test_keeps_overlap_across_classes(self):
        boxes = self._t([[0, 0, 10, 10], [1, 1, 11, 11]])
        scores = self._t([0.9, 0.8])
        classes = torch.as_tensor([0, 2], device="cuda")
        assert nms_pack(boxes, scores, classes, 0.5).shape[0] == 2

    def test_max_det_caps_output(self):
        n = 20
        boxes = self._t([[i * 50, 0, i * 50 + 10, 10] for i in range(n)])
        scores = self._t(np.linspace(0.1, 0.9, n))
        classes = torch.zeros(n, dtype=torch.int64, device="cuda")
        assert nms_pack(boxes, scores, classes, 0.5, max_det=5).shape[0] == 5
        assert nms_pack(boxes, scores, classes, 0.5, max_det=None).shape[0] == n

    def test_class_column_is_float(self):
        boxes = self._t([[0, 0, 10, 10]])
        scores = self._t([0.9])
        classes = torch.as_tensor([3], device="cuda")
        out = nms_pack(boxes, scores, classes, 0.5)
        assert out.dtype == torch.float32
        assert out[0, 5].item() == 3.0
