from pathlib import Path

import cupy as cp
import numpy as np
import torch
from torchvision.ops import batched_nms

from . import model
from .sr_bundle_engine import SRBundleEngine
from .hsv_crosshair import HSVCrosshairDetector
from ..utils.utils import log


class DetectionPipeline:
    """
    Owns the full per-frame detection stack: base detector, optional scan_sr +
    precision_sr SR bundles, optional HSV red-crosshair detector, cross-model NMS,
    and the precision-crop hysteresis state machine.

    Routing (mutually exclusive, decided per call):
      - ADS + locked SMALL target  -> precision_sr only on a small crop centered on the lock.
                                      base is skipped: the precision crop ROI doesn't cover
                                      anything else, and the locked target is what matters.
      - ADS + lock missing this frame but inside hysteresis budget -> precision_sr at last
                                      known location. lets us survive 1-2 frames of detector
                                      flicker without flipping back to scan path.
      - ADS + locked LARGE target  -> base only (target is already easy; skip SR cost).
      - else (no ADS / no lock / hysteresis expired) -> base [+ scan_sr] union'd then NMS'd.

    `run(frame, ads, locked, locked_lifetime)` returns (M, 6) np.float32
    [x1,y1,x2,y2,conf,cls] in base-region xyxy coords. Caller is responsible for
    xyxy->xywh and feeding to the tracker.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg

        # base detector
        m = cfg['model']
        base_path = Path.cwd() / m['base_dir'] / m['base_filename']
        if not base_path.exists():
            raise FileNotFoundError(f"base model missing: {base_path}")

        pt_hw_capture = tuple(m['pt_hw_capture'])
        conf_threshold = m['conf_threshold']

        self.base_model = model.Model(base_path, hw_capture=pt_hw_capture)
        self.base_hw_capture = self.base_model.hw_capture

        # SR bundles
        bb_thresh_override = m['bb_largest_side_threshold_override']
        self.scan_sr = self._load_sr_bundle(m['scan_sr_bundle'], conf_threshold, "scan_sr", bb_thresh_override)
        self.precision_sr = self._load_sr_bundle(m['precision_sr_bundle'], conf_threshold, "precision_sr", bb_thresh_override)

        if self.scan_sr is not None and self.scan_sr.source_size != int(self.base_hw_capture[0]):
            log(
                f"scan_sr source_size={self.scan_sr.source_size} mismatched with base capture {self.base_hw_capture}",
                "WARNING",
            )

        self.union_nms_iou = float(m['union_nms_iou'])

        # precision_sr hysteresis: # of stale-lock frames we tolerate before bailing
        # back to the scan path. 0 = no hysteresis (legacy behavior).
        self.precision_sr_hysteresis_frames = int(m['precision_sr_hysteresis_frames'])

        # HSV crosshair detector (separate from main detection but lives here so the
        # pipeline owns all per-frame frame-consumers).
        self.hsv_detector = self._build_hsv_detector(cfg)

    # --- init helpers ---------------------------------------------------------

    def _load_sr_bundle(self, cfg_path, conf_threshold, label, bb_largest_side_threshold_override):
        if not cfg_path:
            log(f"{label} bundle disabled (no path configured)", "INFO")
            return None
        bundle_path = Path.cwd() / cfg_path
        if not bundle_path.exists():
            log(f"{label} bundle missing at {bundle_path} — disabling", "WARNING")
            return None
        log(f"loading {label} bundle from {bundle_path}", "INFO")
        return SRBundleEngine(
            str(bundle_path),
            conf_threshold=conf_threshold,
            bb_largest_side_threshold_override=bb_largest_side_threshold_override,
        )

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
            locked_lifetime: int) -> np.ndarray:
        """
        frame: (H, W, 3) uint8 RGB cupy.
        ads: is RMB held this frame.
        locked: latest top-priority enemy row (10-col tracker output) or None.
                may be stale (not refreshed this frame) — locked_lifetime tells you how stale.
        locked_lifetime: 0 if `locked` was refreshed this frame, N if N frames stale.
                         the precision_sr hysteresis path consults this to decide whether
                         to keep cropping at the cached location.
        """
        # SR bundles need cupy/TRT. fall back to wrapper API for .pt base.
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
            small_lock = (
                self.precision_sr is not None
                and bb_max_side < self.precision_sr.bb_largest_side_threshold
            )
            if small_lock and locked_lifetime <= self.precision_sr_hysteresis_frames:
                # fresh small lock OR stale-but-within-budget. crop at the cached location.
                tag = "fresh" if locked_lifetime == 0 else f"hysteresis {locked_lifetime}/{self.precision_sr_hysteresis_frames}"
                log(f"precision sr ({tag})", level="DEBUG")
                return self._run_precision_crop(preprocessed, locked)
            if locked_lifetime == 0:
                # fresh large lock (or precision_sr disabled). target's already easy; skip SR.
                log("base only", level="DEBUG")
                return self.base_model.model.inference_cp(preprocessed)
            # large stale lock or precision hysteresis expired -> fall through to scan path.

        # default: base [+ scan_sr] union NMS. both arms NMS internally; this catches
        # between-model dups. cheap.
        log("base + scan_sr", level="DEBUG")
        base_res = self.base_model.model.inference_cp(preprocessed)
        if self.scan_sr is not None:
            sr_res = self.scan_sr.inference_cp(preprocessed)[0]  # strip batch dim
            if sr_res.shape[0]:
                base_res = self._union_nms(cp.concatenate([base_res, sr_res], axis=0))
        return base_res

    # --- crosshair routing ----------------------------------------------------

    def _apply_crosshair_routing(self, model_dets: np.ndarray, frame_rgb_gpu: cp.ndarray) -> np.ndarray:
        """Filter model crosshair-class dets per cfg, then append HSV-derived crosshair row if enabled.
        model_dets: (M, 6) np.float32 [x1, y1, x2, y2, conf, cls] in base-region coords.
        """
        ts = self.cfg['targeting_settings']
        crosshair_cls_id = ts['crosshair_cls_id']

        if not ts['model_predict_crosshair']:
            model_dets = model_dets[model_dets[:, 5] != crosshair_cls_id]

        if self.hsv_detector is not None:
            hsv_row = self.hsv_detector.detect(frame_rgb_gpu)
            if hsv_row.shape[0]:
                model_dets = np.concatenate([model_dets, hsv_row], axis=0)

        return model_dets

    # --- helpers --------------------------------------------------------------

    def _union_nms(self, dets: cp.ndarray) -> cp.ndarray:
        """Class-aware NMS over a concatenation of detections from multiple models.
        Input: cupy (N, 6) [x1,y1,x2,y2,conf,cls]. Output: cupy (M, 6).
        """
        if dets.shape[0] == 0:
            return dets
        t = torch.from_dlpack(dets)
        boxes = t[:, :4]
        scores = t[:, 4]
        classes = t[:, 5].long()
        keep = batched_nms(boxes, scores, classes, self.union_nms_iou)
        return cp.from_dlpack(t[keep])

    def _run_precision_crop(self, preprocessed: cp.ndarray, locked: np.ndarray) -> cp.ndarray:
        """Crop centered on the locked target, run through precision_sr,
        translate detections back to base-region coords. Returns cupy (M, 6)."""
        p = self.precision_sr.source_size
        H, W = int(preprocessed.shape[2]), int(preprocessed.shape[3])
        cx = float((locked[0] + locked[2]) * 0.5)
        cy = float((locked[1] + locked[3]) * 0.5)
        x0 = int(max(0, min(W - p, round(cx - p / 2))))
        y0 = int(max(0, min(H - p, round(cy - p / 2))))
        crop = cp.ascontiguousarray(preprocessed[:, :, y0:y0 + p, x0:x0 + p])
        out = self.precision_sr.inference_cp(crop)[0]  # cupy (M, 6) in local crop coords
        if out.shape[0]:
            out[:, 0] += x0
            out[:, 2] += x0
            out[:, 1] += y0
            out[:, 3] += y0
        return out

    def cleanup(self):
        del self.base_model
        del self.scan_sr
        del self.precision_sr
        del self.hsv_detector
