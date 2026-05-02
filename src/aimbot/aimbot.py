from BetterBYTETracker.trackers.byte_tracker import BYTETracker
import time
import torch
import sys
from pathlib import Path
from ultralytics.utils.ops import xyxy2xywh
import numpy as np
import cupy as cp
import json

# can replace with bettercam just no cupy support, when init camera object set nvidia_gpu = False for bettercam
import betterercam

from .data_parsing import targetselector
from .engine.detection_pipeline import DetectionPipeline
from .engine.hsv_crosshair import HSVCrosshairDetector
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
        self.pipeline = DetectionPipeline(self.cfg)
        self.base_hw_capture = self.pipeline.base_hw_capture
        log("DetectionPipeline initialized", "INFO")
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
        hsv_xh = ts['hsv_settings']['enabled']

        if lead_target and not predict_drop:
            raise ValueError(
                "Invalid targeting configuration: 'lead_target' requires 'predict_drop' to be enabled. "
                "Please enable 'predict_drop' in your config or disable 'lead_target'."
            )

        if model_xh and hsv_xh:
            raise ValueError(
                "Invalid targeting configuration: 'model_predict_crosshair' and 'hsv_settings.enabled' "
                "are mutually exclusive. Enable exactly one (or neither) in your config."
            )

        if hsv_xh:
            scheme = ts['hsv_settings']['voting_scheme']
            if scheme not in HSVCrosshairDetector.VOTING_SCHEMES:
                raise ValueError(
                    f"Invalid 'hsv_settings.voting_scheme': {scheme!r}. Must be one of {list(HSVCrosshairDetector.VOTING_SCHEMES)}."
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

                results = self.pipeline.run(
                    frame,
                    ads=self.inputdetector.is_rmb_pressed,
                    locked=self.target_selector._prev_detection,
                    locked_lifetime=self.target_selector._prev_detection_lifetime,
                )
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

    def setup_bytetracker(self):
        #if engine is running just going to assume 144 is the target frame rate
        #if pt model is running its probably debug screen so 30
        target_frame_rate = 144 if self.pipeline.base_model.model_ext == ".engine" else 30
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
        # Most of this is theater. Python GC + sys.exit handle nearly everything:
        # plain `del`/`= None` and `pipeline.cleanup()` (which itself just dels) are
        # redundant — refs drop on process exit and __del__ chains run automatically.
        # `torch.cuda.empty_cache()` returns cached pool memory but the OS reclaims
        # everything anyway when the process dies. Leaving it for cosmetic shutdown
        # logging more than anything.
        # The two things that actually matter:
        #   - camera.release(): betterercam holds OS capture handles (DXGI/nvidia),
        #     releasing avoids leaving them dangling if we ever do an in-process restart.
        #   - gui_manager.cleanup(): destroys cv2 windows / DPG context cleanly.
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
            if hasattr(self, 'pipeline'):
                log('Cleaning up DetectionPipeline', "INFO")
                self.pipeline.cleanup()
                del self.pipeline
            torch.cuda.empty_cache()
        except Exception as e:
            log(f"Cleanup error: {e}", "ERROR")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO11 Aimbot")
    parser.add_argument('--config', type=str, default='config/cfg.json', help='Path to the configuration file')
    args = parser.parse_args()
    log("About to create Aimbot instance and run main", "INFO")
    Aimbot(args.config).main()
