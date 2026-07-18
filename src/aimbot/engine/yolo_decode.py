"""Shared YOLO output decoding + NMS packing.

Three call sites previously each carried a copy of this logic:
TensorRT_Engine._external_nms_cp (single-image raw output),
SRBundleEngine._patchified_nms (batched raw OR baked output), and
DetectionPipeline._union_nms (already-decoded rows). Pure torch — cupy<->torch
dlpack bridging stays at the call sites.

Output format conventions:
  raw:   (B, 4+nc, A) xywh anchor predictions from a YOLO head without NMS baked in.
  baked: (B, max_det, 6) [x1,y1,x2,y2,conf,cls] zero-padded rows (conf==0) from an
         engine exported with NMS.

Both decoders conf-threshold with strict `>` and return flat
(boxes_xyxy (N,4), scores (N,), classes int64 (N,), batch_idx int64 (N,)).
Callers handle the empty case (shapes differ per site) and any coordinate
translation (e.g. SR patch origins) between decode and nms_pack.
"""

import torch
from torchvision.ops import batched_nms


def decode_raw(out_t: torch.Tensor, conf_threshold: float):
    """(B, 4+nc, A) xywh -> conf-masked flat (boxes_xyxy, scores, classes, batch_idx)."""
    B, _, A = out_t.shape
    pred = out_t.transpose(1, 2)  # (B, A, 4+nc)
    boxes_xywh = pred[..., :4].reshape(-1, 4)
    scores, classes = pred[..., 4:].max(dim=-1)
    scores = scores.reshape(-1)
    classes = classes.reshape(-1)
    batch_idx = torch.arange(B, device=out_t.device).repeat_interleave(A)

    mask = scores > conf_threshold
    boxes_xywh = boxes_xywh[mask]

    cx, cy, w, h = boxes_xywh.unbind(-1)
    boxes_xyxy = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)
    return boxes_xyxy, scores[mask], classes[mask], batch_idx[mask]


def decode_baked(out_t: torch.Tensor, conf_threshold: float):
    """(B, max_det, 6) xyxy zero-padded -> conf-masked flat (boxes_xyxy, scores, classes, batch_idx)."""
    B, M, _ = out_t.shape
    boxes_xyxy = out_t[..., :4].reshape(-1, 4)
    scores = out_t[..., 4].reshape(-1)
    classes = out_t[..., 5].reshape(-1).long()
    batch_idx = torch.arange(B, device=out_t.device).repeat_interleave(M)

    mask = scores > conf_threshold
    return boxes_xyxy[mask], scores[mask], classes[mask], batch_idx[mask]


def nms_pack(boxes_xyxy: torch.Tensor, scores: torch.Tensor, classes: torch.Tensor,
             iou_threshold: float, max_det: int | None = None) -> torch.Tensor:
    """Class-aware NMS, packed as (M, 6) [x1,y1,x2,y2,conf,cls] rows (score-sorted).
    classes must be int64 (batched_nms idxs). max_det=None means uncapped."""
    keep = batched_nms(boxes_xyxy, scores, classes, iou_threshold)
    if max_det is not None:
        keep = keep[:max_det]
    return torch.cat([
        boxes_xyxy[keep],
        scores[keep, None],
        classes[keep, None].to(boxes_xyxy.dtype),
    ], dim=-1)
