from pathlib import Path

import cupy as cp
import numpy as np
import torch
from sr import SRModel

from . import model
from .hsv_crosshair import HSVCrosshairDetector
from .yolo_decode import decode_baked, decode_raw, nms_pack
from ..utils.utils import log

# shared no-detections sentinel; callers only read it, never mutate
_EMPTY_DETS = np.empty((0, 6), dtype=np.float32)

# within-crop NMS, for SR engines without it baked in. TensorRT_Engine's defaults.
SR_NMS_IOU = 0.7
SR_MAX_DET = 300


class DetectionPipeline:
    """
    Owns the full per-frame detection stack: base detector, optional SR model on a crop
    around the lock, optional HSV red-crosshair detector, and cross-model NMS.

    Routing (mutually exclusive, decided per call):
      - ADS + locked SMALL target  -> sr only, on a crop centered on the lock. base is
                                      skipped: the crop doesn't cover anything else.
      - ADS + lock missing this frame but inside hysteresis budget -> sr at last known
                                      location. survives 1-2 frames of detector flicker
                                      without flipping back to the scan path.
      - ADS + locked LARGE target  -> base only (target is already easy; skip SR cost).
      - else (no ADS / no lock / hysteresis expired) -> base [+ scan_sr] union'd then NMS'd.

    scan_sr is deprecated for now — the branch and _union_nms are kept for its return.

    `run(frame, ads, locked, locked_lifetime)` returns
    (dets_for_tracker, bypass_crosshair_rows), both (N, 6) np.float32
    [x1,y1,x2,y2,conf,cls] in base-region xyxy coords. Caller converts the first
    array xyxy->xywh for the tracker; bypass rows stay xyxy and are injected
    post-tracker (see hsv_settings.bypass_tracker).
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg

        # base detector
        m = cfg['model']
        base_path = Path.cwd() / m['base_dir'] / m['base_filename']
        if not base_path.exists():
            raise FileNotFoundError(f"base model missing: {base_path}")

        pt_hw_capture = tuple(m['pt_hw_capture'])
        self.conf_threshold = m['conf_threshold']

        self.base_model = model.Model(base_path, hw_capture=pt_hw_capture, conf_threshold=self.conf_threshold)
        self.base_hw_capture = self.base_model.hw_capture

        self.scan_sr = None  # deprecated, planned to return
        self.sr = self._load_sr_model(m['sr_model'])

        if self.sr is not None and self.sr.crop_size > min(self.base_hw_capture):
            log(f"sr crop_size={self.sr.crop_size} exceeds base capture {self.base_hw_capture}", "WARNING")

        self.union_nms_iou = float(m['union_nms_iou'])

        # sr hysteresis: # of stale-lock frames we tolerate before bailing back to the
        # scan path. 0 = bail as soon as the lock goes stale.
        self.sr_hysteresis_frames = int(m['sr_hysteresis_frames'])

        # HSV crosshair detector (separate from main detection but lives here so the
        # pipeline owns all per-frame frame-consumers).
        self.hsv_detector = self._build_hsv_detector(cfg)
        # hsv rows can skip the tracker entirely (no track latency on the reticle);
        # the split happens in _apply_crosshair_routing so the loop never re-derives it.
        self.hsv_bypass_tracker = cfg['targeting_settings']['hsv_settings']['bypass_tracker']

    # --- init helpers ---------------------------------------------------------

    def _load_sr_model(self, cfg_path) -> SRModel | None:
        """The SR artifact from src.sr.export. It owns its own crop size, scale and
        box-routing threshold, so nothing here re-specifies them."""
        if not cfg_path:
            log("sr model disabled (no path configured)", "INFO")
            return None
        sr_path = Path.cwd() / cfg_path
        if not sr_path.exists():
            log(f"sr model missing at {sr_path} — disabling", "WARNING")
            return None
        sr = SRModel.load(sr_path)
        if sr.kind != "engine":
            log(f"sr model at {sr_path} is {sr.kind!r}, needs 'engine' — disabling", "WARNING")
            return None
        log(f"loaded sr model: {sr.describe()}", "INFO")
        return sr

    def _build_hsv_detector(self, cfg) -> HSVCrosshairDetector | None:
        ts = cfg['targeting_settings']
        hsv_cfg = ts['hsv_settings']
        if not hsv_cfg['enabled']:
            return None
        crop = hsv_cfg['center_crop']
        return HSVCrosshairDetector(
            voting_scheme=hsv_cfg['voting_scheme'],
            crosshair_cls_id=ts['crosshair_cls_id'],
            frame_hw=self.base_hw_capture,
            center_crop_hw=tuple(crop) if crop else None,
        )

    # --- main entry point -----------------------------------------------------

    def run(self, frame: cp.ndarray, ads: bool, locked: np.ndarray | None,
            locked_lifetime: int) -> tuple[np.ndarray, np.ndarray]:
        """
        frame: (H, W, 3) uint8 RGB cupy.
        ads: is RMB held this frame.
        locked: latest top-priority enemy row (10-col tracker output) or None.
                may be stale (not refreshed this frame) — locked_lifetime tells you how stale.
        locked_lifetime: 0 if `locked` was refreshed this frame, N if N frames stale.
                         the sr hysteresis path consults this to decide whether
                         to keep cropping at the cached location.

        Returns (dets_for_tracker, bypass_crosshair_rows) — see class docstring.
        """
        # SR needs cupy/TRT. fall back to wrapper API for .pt base.
        if self.base_model.model_ext != ".engine":
            model_dets = self.base_model.inference(src=frame)
        else:
            model_dets = cp.asnumpy(self._run_engine_path(frame, ads, locked, locked_lifetime))

        return self._apply_crosshair_routing(model_dets, frame)

    # --- engine-path branches -------------------------------------------------

    def _run_engine_path(self, frame: cp.ndarray, ads: bool, locked: np.ndarray | None,
                         locked_lifetime: int) -> cp.ndarray:
        preprocessed = self.base_model._preprocess_cp(frame)  # (1, 3, H, W) cp.float32

        if ads and locked is not None:
            bb_max_side = max(float(locked[2] - locked[0]), float(locked[3] - locked[1]))
            small_lock = self.sr is not None and bb_max_side < self.sr.bb_side
            if small_lock and locked_lifetime <= self.sr_hysteresis_frames:
                # fresh small lock OR stale-but-within-budget. crop at the cached location.
                tag = "fresh" if locked_lifetime == 0 else f"hysteresis {locked_lifetime}/{self.sr_hysteresis_frames}"
                log(f"sr ({tag})", level="DEBUG")
                return self._run_sr_crop(preprocessed, locked)
            if locked_lifetime == 0:
                # fresh large lock (or sr disabled). target's already easy; skip SR.
                log("base only", level="DEBUG")
                return self.base_model.model.inference_cp(preprocessed)
            # large stale lock or sr hysteresis expired -> fall through to scan path.

        # default: base [+ scan_sr] union NMS. both arms NMS internally; this catches
        # between-model dups. cheap.
        log("base + scan_sr", level="DEBUG")
        base_res = self.base_model.model.inference_cp(preprocessed)
        if self.scan_sr is not None:  # deprecated, planned to return
            sr_res = self.scan_sr.inference_cp(preprocessed)[0]  # strip batch dim
            if sr_res.shape[0]:
                base_res = self._union_nms(cp.concatenate([base_res, sr_res], axis=0))
        return base_res

    # --- crosshair routing ----------------------------------------------------

    def _apply_crosshair_routing(self, model_dets: np.ndarray, frame_rgb_gpu: cp.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Filter model crosshair-class dets per cfg, run the HSV detector if enabled,
        and split its row out when it should bypass the tracker.
        model_dets: (M, 6) np.float32 [x1, y1, x2, y2, conf, cls] in base-region coords.
        Returns (dets_for_tracker, bypass_crosshair_rows). Model-predicted crosshair
        rows always go through the tracker; only HSV rows can bypass.
        """
        ts = self.cfg['targeting_settings']
        crosshair_cls_id = ts['crosshair_cls_id']

        if not ts['model_predict_crosshair']:
            model_dets = model_dets[model_dets[:, 5] != crosshair_cls_id]

        bypass_rows = _EMPTY_DETS
        if self.hsv_detector is not None:
            hsv_row = self.hsv_detector.detect(frame_rgb_gpu)
            if hsv_row.shape[0]:
                if self.hsv_bypass_tracker:
                    bypass_rows = hsv_row
                else:
                    model_dets = np.concatenate([model_dets, hsv_row], axis=0)

        return model_dets, bypass_rows

    # --- helpers --------------------------------------------------------------

    def _union_nms(self, dets: cp.ndarray) -> cp.ndarray:
        """Class-aware NMS over a concatenation of detections from multiple models.
        Input: cupy (N, 6) [x1,y1,x2,y2,conf,cls]. Output: cupy (M, 6).
        """
        if dets.shape[0] == 0:
            return dets
        t = torch.from_dlpack(dets)
        # uncapped on purpose: union of per-model outputs is already small
        out = nms_pack(t[:, :4], t[:, 4], t[:, 5].long(), self.union_nms_iou)
        return cp.from_dlpack(out)

    def _run_sr_crop(self, preprocessed: cp.ndarray, locked: np.ndarray) -> cp.ndarray:
        """Crop centered on the locked target, run it through the SR model, translate
        detections back to base-region coords. Returns cupy (M, 6)."""
        c = self.sr.crop_size
        H, W = int(preprocessed.shape[2]), int(preprocessed.shape[3])
        cx = float((locked[0] + locked[2]) * 0.5)
        cy = float((locked[1] + locked[3]) * 0.5)
        x0 = int(max(0, min(W - c, round(cx - c / 2))))
        y0 = int(max(0, min(H - c, round(cy - c / 2))))
        crop = cp.ascontiguousarray(preprocessed[:, :, y0:y0 + c, x0:x0 + c])
        return self._decode_sr(self.sr(torch.from_dlpack(crop)), x0, y0)  # SRModel takes torch

    def _decode_sr(self, out, x0: int, y0: int) -> cp.ndarray:
        """SR engine output -> cupy (M, 6) in base-region coords.

        The model upscales internally, so its boxes come back in crop_size * sr_scale
        px and have to be divided down before the crop origin means anything.
        """
        t = out if isinstance(out, torch.Tensor) else torch.from_dlpack(out)
        if t.ndim != 3:
            raise ValueError(f"sr model returned {tuple(t.shape)}; expected raw engine "
                             f"output (B, max_det, 6) or (B, 4+nc, anchors)")
        baked = t.shape[2] == 6  # (B, max_det, 6) vs raw (B, 4+nc, anchors)
        decode = decode_baked if baked else decode_raw
        # .float() before the offset: an fp16 engine resolves 0.5px at a 560px origin
        boxes, scores, classes, _ = decode(t.float(), self.conf_threshold)
        if boxes.numel() == 0:
            return cp.empty((0, 6), dtype=cp.float32)

        boxes = boxes / self.sr.sr_scale
        boxes[:, 0::2] += x0
        boxes[:, 1::2] += y0
        if baked:
            return cp.from_dlpack(torch.cat(
                [boxes, scores[:, None], classes[:, None].to(boxes.dtype)], dim=-1))
        return cp.from_dlpack(nms_pack(boxes, scores, classes, SR_NMS_IOU, SR_MAX_DET))
