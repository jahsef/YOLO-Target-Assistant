from BetterBYTETracker.trackers.byte_tracker import BYTETracker
import time
import torch
import sys
from pathlib import Path
from ultralytics.utils.ops import xyxy2xywh
import numpy as np
import json

# can replace with bettercam just no cupy support, when init camera object set nvidia_gpu = False for bettercam
import betterercam

from .data_parsing import targetselector
from .engine import model
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

        self.gui_manager = gui_manager.GUI_Manager(config = self.cfg,hw_capture = self.scanning_hw_capture)

        log("GUI_Manager initialized", "INFO")


        self.fps_tracker = fpstracker.FPSTracker()
        self.init_camera()
        log("init_camera complete", "INFO")

        self.target_selector = targetselector.TargetSelector(
            cfg=self.cfg,
            detection_window_dim=self.ads_hw_capture,
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
        lead_target = self.cfg['targeting_settings']['lead_target']
        predict_drop = self.cfg['targeting_settings']['predict_drop']

        if lead_target and not predict_drop:
            raise ValueError(
                "Invalid targeting configuration: 'lead_target' requires 'predict_drop' to be enabled. "
                "Please enable 'predict_drop' in your config or disable 'lead_target'."
            )

    def init_model(self):
        #working on multi model support (scanning/ads)
        ads_path = Path.cwd() / self.cfg['model']['ads_dir'] / self.cfg['model']['ads_filename']

        if not ads_path.exists():
            raise FileNotFoundError(f"ADS model missing: {ads_path}")

        scanning_path = Path.cwd() / self.cfg['model']['scanning_dir'] / self.cfg['model']['scanning_filename']

        if not scanning_path.exists() or scanning_path == ads_path:
            log('SCANNING MODEL PATH DOES NOT EXIST OR SAME AS ADS MODEL, FALLING BACK TO ADS MODEL', "WARNING")
            scanning_path = ads_path

        #this capture dimension is only used for pt models, engine models load dimensions internally
        pt_hw_capture = tuple(self.cfg['model']['pt_hw_capture'])

        #load both models
        #if models are the same, only load 1 model make both regions the same

        self.ads_model = model.Model(ads_path, hw_capture=pt_hw_capture)
        self.ads_hw_capture = self.ads_model.hw_capture

        if ads_path == scanning_path:
            self.scanning_model = self.ads_model
        else:
            self.scanning_model = model.Model(scanning_path, hw_capture=pt_hw_capture)
        self.scanning_hw_capture = self.scanning_model.hw_capture

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
        scanning_x_offset = (self.screen_x - self.scanning_hw_capture[1]) // 2
        scanning_y_offset = (self.screen_y - self.scanning_hw_capture[0]) // 2
        log(f'screen_x: {self.screen_x}', "DEBUG")
        log(f'scanning_hw_capture[1]: {self.scanning_hw_capture[1]}', "DEBUG")
        ads_x_offset = (self.scanning_hw_capture[1] - self.ads_hw_capture[1]) // 2
        ads_y_offset = (self.scanning_hw_capture[0] - self.ads_hw_capture[0]) // 2

        #need to modify for dual region i think later
        self.scanning_region = (0 + scanning_x_offset, 0 + scanning_y_offset, self.screen_x - scanning_x_offset, self.screen_y - scanning_y_offset)

        self.ads_region = (
            self.scanning_region[0] + ads_x_offset,
            self.scanning_region[1] + ads_y_offset,
            self.scanning_region[2] - ads_x_offset,
            self.scanning_region[3] - ads_y_offset
        )

        #by default uses scanning region when ads, use ads region
        self.camera = betterercam.create(region = self.scanning_region, output_color='RGB',max_buffer_len=2, nvidia_gpu = True)

        #if using yolo inference need to use BGR since they assume your input is BGR



    def main(self):
        log("Entering main loop", "INFO")
        #frame_model  is the specific model we are using for the frame
        #could be either  ads or scanning model
        frame_model:model.Model
        try:
            while True:
                aimbot_active = (self.inputdetector.is_toggled and (self.inputdetector.is_rmb_pressed)) or not self.cfg['input_settings']['right_click_toggle']

                if not aimbot_active:
                    #throttling so when scanning we dont use all resources
                    time.sleep(self.cfg['other']['inactive_throttle_ms'] / 1000)

                #use either scanning  or ads model
                if self.inputdetector.is_rmb_pressed:
                    frame = self.camera.grab(region = self.ads_region)
                    frame_model = self.ads_model
                    w = self.ads_region[2] - self.ads_region[0]
                    h = self.ads_region[3] - self.ads_region[1]
                    #might be kinda bloaty to update it constantly but whatever
                    self.target_selector.update_detection_window_center(window_dim=(w, h))
                else:
                    frame = self.camera.grab()
                    frame_model = self.scanning_model
                    w = self.scanning_region[2] - self.scanning_region[0]
                    h = self.scanning_region[3] - self.scanning_region[1]
                    self.target_selector.update_detection_window_center(window_dim=(w, h))

                #capture lib sometimes may return none
                if frame is None:
                    continue

                results = frame_model.inference(src = frame) #model returns [x1,y1,x2,y2,conf,cls]
                results[:,0:4] = xyxy2xywh(results[:,0:4])
                self.tracker.update(results) # expects (N, 6) [x, y, w, h, conf, cls]
                self.tracker.multi_predict(tracks = None) # ultralytics expects stracks, our custom impl uses internal state (tracks arg unused)
                tracked_detections = self.tracker.get_active_tracks_with_lifetime() # returns (M, 10) [x1,y1,x2,y2,track_id,score,cls,idx,start_frame,last_frame]
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
        target_frame_rate = 144 if self.ads_model.model_ext == ".engine" else 30
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
            if hasattr(self, 'ads_model'):
                log('Removing ads_model', "INFO")
                del self.ads_model
            if hasattr(self, 'scanning_model'):
                log('Removing scanning_model', "INFO")
                del self.scanning_model
            torch.cuda.empty_cache()
        except Exception as e:
            log(f"Cleanup error: {e}", "ERROR")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO11 Aimbot")
    parser.add_argument('--config', type=str, default='config/cfg.json', help='Path to the configuration file')
    args = parser.parse_args()
    log("About to create Aimbot instance and run main", "INFO")
    Aimbot(args.config).main()
