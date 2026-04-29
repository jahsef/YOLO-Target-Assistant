"""
SR + YOLO bundle engine. Square non-overlapping patchify only, single image in.

Flow:
    1. patchify (1, 3, H, W) -> (B, 3, p, p) where B = grid_dim^2
    2. SR engine .forward()  -> (B, 3, p*scale, p*scale)
    3. YOLO engine .forward() -> raw (B, 4+nc, anchors) OR baked (B, max_det, 6)
    4. cross-patch class-aware NMS in global source-image coords -> (1, M, 6)

Works whether YOLO has NMS baked in or not — both paths end with a single
cross-patch NMS over the flattened, source-coord-translated boxes.
"""

import cupy as cp
import numpy as np
import tensorrt as trt
import torch
from torchvision.ops import batched_nms

from .tensorrt_engine import TensorRT_Engine
from ..utils.utils import log


class SRBundleEngine:
    def __init__(
        self,
        bundle_path: str,
        conf_threshold: float,
        verbose: bool = False,
        bb_largest_side_threshold_override: int = -1,
    ):
        bundle = torch.load(bundle_path, weights_only=False)
        self.sr = TensorRT_Engine(engine_bytes=bundle["sr_engine"], conf_threshold=0.0, verbose=verbose, nms=False)
        self.yolo = TensorRT_Engine(engine_bytes=bundle["yolo_engine"], conf_threshold=conf_threshold, verbose=verbose, nms=True)
        bundle_thresh = bundle["bb_largest_side_threshold"]
        if bb_largest_side_threshold_override == -1:
            self.bb_largest_side_threshold = bundle_thresh
        else:
            log(
                f"overriding bb_largest_side_threshold: bundle={bundle_thresh} -> override={bb_largest_side_threshold_override}",
                "INFO",
            )
            self.bb_largest_side_threshold = bb_largest_side_threshold_override
        self.class_names = bundle["class_names"]
        self.sr_model_name = bundle["sr_model_name"]
        self.yolo_model_name = bundle["yolo_model_name"]

        if int(self.sr.input_shape[0]) != int(self.yolo.input_shape[0]):
            raise ValueError(
                f"batch mismatch: SR={self.sr.input_shape[0]} YOLO={self.yolo.input_shape[0]}"
            )
        self.batch = int(self.sr.input_shape[0])

        sr_h, sr_w = int(self.sr.input_shape[2]), int(self.sr.input_shape[3])
        if sr_h != sr_w:
            raise ValueError(f"SR input must be square, got {(sr_h, sr_w)}")
        self.patch_size = sr_h

        grid_dim = int(self.batch ** 0.5)
        if grid_dim * grid_dim != self.batch:
            raise ValueError(f"batch={self.batch} is not a perfect square (need NxN grid)")
        self.grid_dim = grid_dim
        self.source_size = self.patch_size * grid_dim  # H = W

        sr_out_side = int(self.sr.output_shape[2])
        if sr_out_side % self.patch_size != 0:
            raise ValueError(f"SR output {sr_out_side} not a clean multiple of input {self.patch_size}")
        self.sr_scale = sr_out_side // self.patch_size

        # patch origins in source-image pixel coords, row-major (B, 2) as (x, y)
        ys, xs = cp.meshgrid(cp.arange(grid_dim), cp.arange(grid_dim), indexing="ij")
        self.patch_origins = (cp.stack([xs.ravel(), ys.ravel()], axis=-1) * self.patch_size).astype(cp.float32)

    def patchify(self, img: cp.ndarray) -> cp.ndarray:
        """(1, 3, H, W) -> (B, 3, p, p), row-major non-overlapping square tiles."""
        if img.shape[0] != 1:
            raise ValueError(f"single image only, got batch={img.shape[0]}")
        if img.shape[2] != self.source_size or img.shape[3] != self.source_size:
            raise ValueError(
                f"expected source {self.source_size}x{self.source_size}, got {img.shape[2:]}"
            )
        c = img.shape[1]
        x = img.reshape(c, self.grid_dim, self.patch_size, self.grid_dim, self.patch_size)
        x = x.transpose(1, 3, 0, 2, 4)  # (gd, gd, c, p, p)
        return cp.ascontiguousarray(x.reshape(self.batch, c, self.patch_size, self.patch_size))

    def inference_cp(self, src: cp.ndarray) -> cp.ndarray:
        patches = self.patchify(src)
        sr_out = self.sr.forward(patches)
        self.yolo.forward(sr_out)
        return self._patchified_nms()

    def _patchified_nms(self):
        """Cross-patch class-aware NMS, returning (1, num_dets, 6) in source-image global coords.

        Two YOLO output formats supported:
          - raw   (NMS not baked): (B, 4+nc, anchors), boxes xywh in SR-local px
          - baked (NMS baked):     (B, max_det, 6),    boxes xyxy in SR-local px (zero-padded)

        Both paths produce (boxes_xyxy_local, scores, classes, batch_idx), then translate
        to source coords by dividing by sr_scale and adding the patch origin, then
        run a single cross-patch NMS.
        """
        out_t = torch.from_dlpack(self.yolo.output_buffer)

        if not self.yolo.baked_nms:
            # raw: (B, 4+nc, A) xywh
            B, _, A = out_t.shape
            pred = out_t.transpose(1, 2)  # (B, A, 4+nc)
            boxes_xywh = pred[..., :4].reshape(-1, 4)
            scores, classes = pred[..., 4:].max(dim=-1)
            scores_flat = scores.reshape(-1)
            classes_flat = classes.reshape(-1)
            batch_idx = torch.arange(B, device=out_t.device).repeat_interleave(A)

            mask = scores_flat > self.yolo.conf_threshold
            boxes_xywh = boxes_xywh[mask]
            scores_flat = scores_flat[mask]
            classes_flat = classes_flat[mask]
            batch_idx = batch_idx[mask]

            if boxes_xywh.numel() == 0:
                return cp.empty((1, 0, 6), dtype=cp.float32)

            cx, cy, w, h = boxes_xywh.unbind(-1)
            boxes_xyxy_local = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)
        else:
            # baked NMS: (B, max_det, 6) xyxy + conf + cls, zero-padded rows when conf=0
            B, M, _ = out_t.shape
            boxes_xyxy_local = out_t[..., :4].reshape(-1, 4)
            scores_flat = out_t[..., 4].reshape(-1)
            classes_flat = out_t[..., 5].reshape(-1).long()
            batch_idx = torch.arange(B, device=out_t.device).repeat_interleave(M)

            mask = scores_flat > self.yolo.conf_threshold
            boxes_xyxy_local = boxes_xyxy_local[mask]
            scores_flat = scores_flat[mask]
            classes_flat = classes_flat[mask]
            batch_idx = batch_idx[mask]

            if boxes_xyxy_local.numel() == 0:
                return cp.empty((1, 0, 6), dtype=cp.float32)

        origins_t = torch.from_dlpack(self.patch_origins)  # (B, 2) (x, y)
        off_xyxy = origins_t[batch_idx].repeat(1, 2)  # (N, 4)
        boxes_global = boxes_xyxy_local / self.sr_scale + off_xyxy

        # idxs = classes only -> cross-patch suppression IS desired (boxes spanning seams)
        keep = batched_nms(boxes_global, scores_flat, classes_flat, self.yolo.iou_threshold)[: self.yolo.max_det]

        out = torch.cat([
            boxes_global[keep],
            scores_flat[keep, None],
            classes_flat[keep, None].to(boxes_global.dtype),
        ], dim=-1).unsqueeze(0)  # (1, num_dets, 6)
        return cp.from_dlpack(out)
