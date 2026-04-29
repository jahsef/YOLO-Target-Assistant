from BetterBYTETracker.trackers.byte_tracker import BYTETracker
import time
import torch
import sys
from pathlib import Path
from ultralytics.utils.ops import xyxy2xywh
import numpy as np
import cupy as cp
import json

from torchvision.ops import batched_nms

# can replace with bettercam just no cupy support, when init camera object set nvidia_gpu = False for bettercam
import betterercam

from .data_parsing import targetselector
from .engine import model
from .engine.sr_bundle_engine import SRBundleEngine
from .engine.hsv_crosshair import hsv_crosshair_detection
from .input import mousemover, inputdetector
from .gui import gui_manager
from .utils import fpstracker

from argparse import Namespace
from screeninfo import get_monitors
import traceback
import argparse
from .utils.utils import log
import logging

class Aimbot:
    def __init__(self, config_path):
        with open(config_path) as f:
            self.cfg = json.load(f)
        logging.basicConfig(
            level=logging.INFO,
            format='%(levelname)s: %(message)s',
            force=True
        )
        # set our app logger to the configured level (root stays INFO to suppress betterercam debug spam)
        logging.getLogger('aimbot').setLevel(logging.getLevelNamesMapping()[self.cfg['logging']['logging_level']])
        log("Aimbot: Initializing...", "INFO")
        self._validate_targeting_config()
        self.init_model()
        log("init_model complete", "INFO")
        self.init_monitor()
        log("init_monitor complete", "INFO")
        #i think these have to be after monitor/model init
        self.init_input()
        log("init_input complete", "INFO")

        self.gui_manager = gui_manager.GUI_Manager(config = self.cfg,hw_capture = self.base_hw_capture)

        log("GUI_Manager initialized", "INFO")


        self.fps_tracker = fpstracker.FPSTracker()
        self.init_camera()
        log("init_camera complete", "INFO")

        self.target_selector = targetselector.TargetSelector(
            cfg=self.cfg,
            detection_window_dim=self.base_hw_capture,
            screen_hw=(self.screen_y, self.screen_x),
            fps_tracker = self.fps_tracker
        )
        log("target_selector initialized", "INFO")
        self.setup_bytetracker()
        log("setup_bytetracker complete", "INFO")

        self._frame_count: int = 0

    def _validate_targeting_config(self):
        """
        Validates targeting configuration settings.
        Raises ValueError if configuration is invalid.
        """
        ts = self.cfg['targeting_settings']
        lead_target = ts['lead_target']
        predict_drop = ts['predict_drop']
        model_xh = ts['model_predict_crosshair']
        hsv_xh = ts['hsv_predict_crosshair']

        if lead_target and not predict_drop:
            raise ValueError(
                "Invalid targeting configuration: 'lead_target' requires 'predict_drop' to be enabled. "
                "Please enable 'predict_drop' in your config or disable 'lead_target'."
            )

        if model_xh and hsv_xh:
            raise ValueError(
                "Invalid targeting configuration: 'model_predict_crosshair' and 'hsv_predict_crosshair' "
                "are mutually exclusive. Enable exactly one (or neither) in your config."
            )

    def init_model(self):
        # base_model: always-on detector (TRT .engine or .pt).
        # scan_sr / precision_sr: optional SR bundles. config path may be null/empty to disable.
        base_path = Path.cwd() / self.cfg['model']['base_dir'] / self.cfg['model']['base_filename']
        if not base_path.exists():
            raise FileNotFoundError(f"base model missing: {base_path}")

        pt_hw_capture = tuple(self.cfg['model']['pt_hw_capture'])
        conf_threshold = self.cfg['model']['conf_threshold']

        self.base_model = model.Model(base_path, hw_capture=pt_hw_capture)
        self.base_hw_capture = self.base_model.hw_capture

        scan_sr_cfg = self.cfg['model']['scan_sr_bundle']
        precision_sr_cfg = self.cfg['model']['precision_sr_bundle']
        bb_thresh_override = self.cfg['model']['bb_largest_side_threshold_override']

        self.scan_sr = self._load_sr_bundle(scan_sr_cfg, conf_threshold, "scan_sr", bb_thresh_override)
        self.precision_sr = self._load_sr_bundle(precision_sr_cfg, conf_threshold, "precision_sr", bb_thresh_override)

        # iou for the final cross-model NMS pass (base ∪ SR), see _union_nms.
        self.union_nms_iou = float(self.cfg['model']['union_nms_iou'])

        # source-size sanity check: scan_sr must match capture region
        if self.scan_sr is not None and self.scan_sr.source_size != int(self.base_hw_capture[0]):
            log(
                f"scan_sr source_size={self.scan_sr.source_size} mismatched with base capture {self.base_hw_capture}",
                "WARNING",
            )

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

    def init_monitor(self):
        #dynamic monitor settings
        monitor_idx = self.cfg['display_settings']['monitor_idx']
        monitor = get_monitors()[monitor_idx]
        log(f'LOOKING AT MONITOR: {monitor_idx}', "INFO")
        self.screen_x = monitor.width
        self.screen_y = monitor.height
        log(f'MONITOR DIMS: {monitor.width} x {monitor.height}', "INFO")
        self.screen_center = (self.screen_x // 2, self.screen_y // 2)


    def init_input(self):
        sens_cfg = self.cfg['sensitivity_settings']
        input_cfg = self.cfg['input_settings']

        self.mousemover = mousemover.MouseMover(
            sens_cfg['overall_sens'],
            sens_cfg['sens_scaling'],
            sens_cfg['max_deltas'],
            sens_cfg['jitter_strength'],
            sens_cfg['overshoot_strength'],
            sens_cfg['overshoot_chance']
        )
        self.inputdetector = inputdetector.InputDetector(input_cfg['toggle_hotkey'])
        self.inputdetector.start_input_detection()

    def init_camera(self):
        # single fixed region centered on screen at base_hw_capture size.
        # base_model + scan_sr both consume the full frame; precision_sr consumes an 80x80 sub-slice.
        base_x_offset = (self.screen_x - self.base_hw_capture[1]) // 2
        base_y_offset = (self.screen_y - self.base_hw_capture[0]) // 2
        log(f'screen_x: {self.screen_x}', "DEBUG")
        log(f'base_hw_capture: {self.base_hw_capture}', "DEBUG")

        self.base_region = (
            base_x_offset,
            base_y_offset,
            self.screen_x - base_x_offset,
            self.screen_y - base_y_offset,
        )

        self.camera = betterercam.create(region=self.base_region, output_color='RGB', max_buffer_len=2, nvidia_gpu=True)

        #if using yolo inference need to use BGR since they assume your input is BGR



    def main(self):
        log("Entering main loop", "INFO")
        try:
            while True:
                aimbot_active = (self.inputdetector.is_toggled and (self.inputdetector.is_rmb_pressed)) or not self.cfg['input_settings']['right_click_toggle']

                if not aimbot_active:
                    #throttling so when scanning we dont use all resources
                    time.sleep(self.cfg['other']['inactive_throttle_ms'] / 1000)

                frame = self.camera.grab()
                #capture lib sometimes may return none
                if frame is None:
                    continue

                results = self._run_detection_pipeline(frame)
                results[:,0:4] = xyxy2xywh(results[:,0:4])
                self.tracker.update(results) # expects (N, 6) [x, y, w, h, conf, cls]
                self.tracker.multi_predict(tracks = None) # ultralytics expects stracks, our custom impl uses internal state (tracks arg unused)
                tracked_detections = self.tracker.get_active_tracks_with_lifetime() # returns (M, 10) [x1,y1,x2,y2,track_id,score,cls,idx,start_frame,last_frame]
                # refresh routing state every frame from the freshest tracker output, independent of
                # whether aimbot is firing. without this, precision_sr can get stuck if the small target
                # is lost or replaced — see TargetSelector.update_prev_detection for the full reason.
                self.target_selector.update_prev_detection(tracked_detections)
                self._frame_count += 1

                # update tracker max_time_lost with real fps every 60 frames
                if self._frame_count % 60 == 0 and len(self.fps_tracker.buffer) == self.fps_tracker.fps_buffer_len:
                    real_fps = self.fps_tracker.get_fps()
                    self.tracker.max_time_lost = int(real_fps / 30.0 * self.tracker.args.track_buffer)

                raw_deltas = (0,0)
                scaled_deltas = (0,0)
                if self.inputdetector.is_rmb_pressed:
                    self.target_selector.update_zoom_interpolation()#while right clicking we interpolate zoom till final zoom level
                else:
                    self.target_selector.reset_zoom()

                if aimbot_active and len(tracked_detections) > 0:
                    raw_deltas, scaled_deltas = self.aimbot(tracked_detections)

                if self.cfg['targeting_settings']['lead_target']:
                    self.target_selector.update_movement_buffer(raw_deltas)

                self.fps_tracker.update()
                if self.cfg['logging']['print_fps']:
                    self.fps_tracker.print_fps()

                if self.gui_manager:
                    self.gui_manager.render(frame = frame,
                                            tracked_detections = tracked_detections,
                                            is_rmb_pressed= self.inputdetector.is_rmb_pressed,
                                            raw_deltas = raw_deltas,
                                            scaled_deltas = scaled_deltas)

        except KeyboardInterrupt:
            log("\nShutting down...", "INFO")
            self.cleanup()
            sys.exit(0)  # Clean exit

        except Exception as e:
            log(f"Fatal error: {traceback.print_exc()}", "ERROR")
            self.cleanup()
            sys.exit(1)

    def _run_detection_pipeline(self, frame: cp.ndarray) -> np.ndarray:
        """
        Branch selection (mutually exclusive):
          - ADS + locked SMALL target -> precision_sr only on 80x80 crop centered on the lock.
                                         base is skipped: precision_sr's ROI doesn't cover anything
                                         else, and the locked target is the only thing that matters.
          - ADS + locked LARGE target -> base only (target is already easy; skip SR cost).
          - else (scanning, or ADS w/ no lock) -> base + scan_sr, union'd then cross-model NMS'd.

        Returns (M, 6) np.float32 in base-region xyxy coords.
        """
        ads = self.inputdetector.is_rmb_pressed
        locked = self.target_selector._prev_detection

        # SR bundles need cupy/TRT. fall back to wrapper API for .pt base.
        if self.base_model.model_ext != ".engine":
            model_dets = self.base_model.inference(src=frame)
        else:
            preprocessed = self.base_model._preprocess_cp(frame)  # (1, 3, H, W) cp.float32

            if ads and locked is not None:
                bb_max_side = max(float(locked[2] - locked[0]), float(locked[3] - locked[1]))
                if self.precision_sr is not None and bb_max_side < self.precision_sr.bb_largest_side_threshold:
                    log("precision sr only", level="DEBUG")
                    model_dets_cp = self._run_precision_crop(preprocessed, locked)
                else:
                    log("base only", level="DEBUG")
                    model_dets_cp = self.base_model.model.inference_cp(preprocessed)
            else:
                # scanning path: base + scan_sr concatenated, then cross-model NMS to dedupe overlap.
                # both arms already NMS internally (baked NMS for base; cross-patch NMS in SRBundleEngine);
                # this final pass only catches the *between-model* duplicates. wasteful but cheap.
                log("base + scan_sr", level="DEBUG")
                base_res = self.base_model.model.inference_cp(preprocessed)
                if self.scan_sr is not None:
                    sr_res = self.scan_sr.inference_cp(preprocessed)[0]  # strip batch dim
                    if sr_res.shape[0]:
                        base_res = self._union_nms(cp.concatenate([base_res, sr_res], axis=0))
                model_dets_cp = base_res

            model_dets = cp.asnumpy(model_dets_cp)

        return self._apply_crosshair_routing(model_dets, frame)

    def _apply_crosshair_routing(self, model_dets: np.ndarray, frame_rgb_gpu: cp.ndarray) -> np.ndarray:
        """Filter model crosshair-class dets and/or append HSV-derived crosshair dets per cfg.

        model_dets: (M, 6) np.float32 [x1, y1, x2, y2, conf, cls] in base-region coords.
        frame_rgb_gpu: raw uint8 (H, W, 3) RGB cupy frame -- HSV path needs the un-preprocessed pixels.
        """
        ts = self.cfg['targeting_settings']
        crosshair_cls_id = ts['crosshair_cls_id']

        if not ts['model_predict_crosshair']:
            model_dets = model_dets[model_dets[:, 5] != crosshair_cls_id]

        if ts['hsv_predict_crosshair']:
            crop = ts['hsv_center_crop']
            crop_hw = tuple(crop) if crop else None
            hsv_row = hsv_crosshair_detection(frame_rgb_gpu, crosshair_cls_id, center_crop_hw=crop_hw)
            if hsv_row.shape[0]:
                model_dets = np.concatenate([model_dets, hsv_row], axis=0)

        return model_dets

    def _union_nms(self, dets: cp.ndarray) -> cp.ndarray:
        """Class-aware NMS over a concatenation of detections from multiple models.

        Input: cupy (N, 6) [x1,y1,x2,y2,conf,cls].
        Output: cupy (M, 6), M <= N, deduped within each class.
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
        """80x80 crop centered on the locked target, run through precision_sr,
        translate detections back to base-region coords. Returns cupy (M, 6)."""
        p = self.precision_sr.source_size
        H, W = int(preprocessed.shape[2]), int(preprocessed.shape[3])
        cx = float((locked[0] + locked[2]) * 0.5)
        cy = float((locked[1] + locked[3]) * 0.5)
        x0 = int(max(0, min(W - p, round(cx - p / 2))))
        y0 = int(max(0, min(H - p, round(cy - p / 2))))
        crop = cp.ascontiguousarray(preprocessed[:, :, y0:y0 + p, x0:x0 + p])
        out = self.precision_sr.inference_cp(crop)[0]  # cupy (M, 6) in 80x80 local coords
        if out.shape[0]:
            out[:, 0] += x0
            out[:, 2] += x0
            out[:, 1] += y0
            out[:, 3] += y0
        return out

    def setup_bytetracker(self):
        #if engine is running just going to assume 144 is the target frame rate
        #if pt model is running its probably debug screen so 30
        target_frame_rate = 144 if self.base_model.model_ext == ".engine" else 30
        args = Namespace(
            track_high_thresh=0.65,
            track_low_thresh=0.4,
            track_buffer=20, #track_buffer -> time = track_buffer/30 so 20/30 = 0.66 seconds until lost
            fuse_score=0.5,
            match_thresh=0.6,
            new_track_thresh=0.65
        )

        self.tracker = BYTETracker(args, frame_rate=target_frame_rate)

    def aimbot(self, detections: np.ndarray):
        """
        Args:
            detections: (n, 10) array [x1, y1, x2, y2, track_id, score, cls, idx, start_frame, last_frame]
                        Typically from BYTETracker (BetterBYTETracker lib).get_active_tracks_with_lifetime()
        """
        min_age = self.cfg['targeting_settings']['min_frames_to_target']
        lifetimes = detections[:, 9] - detections[:, 8]  # last_frame - start_frame
        filtered_detections = detections[lifetimes >= min_age]

        raw_deltas = (0,0)
        scaled_deltas = (0,0)
        if len(filtered_detections) > 0:
            raw_deltas = self.target_selector.get_deltas(filtered_detections)
            if raw_deltas != (0,0):
                scaled_deltas = self.mousemover.move_mouse_humanized(raw_deltas[0],raw_deltas[1])
        return raw_deltas, scaled_deltas


    def cleanup(self):
        log("STARTING CLEANUP", "INFO")
        try:
            if hasattr(self, 'camera') and self.camera:
                log('Releasing camera', "INFO")
                self.camera.release()  # Ensure proper release
                del self.camera
                self.camera = None  # Explicitly clear reference

            if hasattr(self, 'gui_manager') and self.gui_manager:
                log('Cleaning up GUI', "INFO")
                self.gui_manager.cleanup()
                del self.gui_manager
                self.gui_manager = None
            if hasattr(self, 'base_model'):
                log('Removing base_model', "INFO")
                del self.base_model
            if hasattr(self, 'scan_sr') and self.scan_sr is not None:
                log('Removing scan_sr', "INFO")
                del self.scan_sr
            if hasattr(self, 'precision_sr') and self.precision_sr is not None:
                log('Removing precision_sr', "INFO")
                del self.precision_sr
            torch.cuda.empty_cache()
        except Exception as e:
            log(f"Cleanup error: {e}", "ERROR")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO11 Aimbot")
    parser.add_argument('--config', type=str, default='config/cfg.json', help='Path to the configuration file')
    args = parser.parse_args()
    log("About to create Aimbot instance and run main", "INFO")
    Aimbot(args.config).main()
