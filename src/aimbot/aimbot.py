from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker

import cv2
import time
import torch
import sys
from pathlib import Path
import cupy as cp
import numpy as np
import json

# Need to add utils to path before importing log
github_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(github_dir / 'BettererCam'))  # Ensure BettererCam is in sys.path
print(f'Adding GITHUB DIR:{github_dir / "BettererCam"} to sys.path')  # Can't use log here yet
# can replace with bettercam just no cupy support, when init camera object set nvidia_gpu = False for bettercam
import betterercam
# print(betterercam.__file__)

# sys.path.insert(0,str(Path.cwd())) 

from .data_parsing import targetselector
from .engine import model
from .input import mousemover, inputdetector
from .gui import gui_manager
from .utils import fpstracker
from ioutrack import ByteTrack

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
            level=logging.getLevelNamesMapping()[self.cfg['logging']['logging_level']],
            format='%(levelname)s: %(message)s',
            force=True
        )
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
        #buffer for  tracked detections, we move from  
        self._tracked_buffer = np.empty((256, 8), dtype=np.float32)
        
        
        
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

        # print(f'ads dimensions: {self.ads_hw_capture}')
        # print(f'scanning dimensions: {self.scanning_hw_capture}')
        # print(f'ads region: {self.ads_region}')
        # print(f'scanning region: {scanning_region}')
        
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
                
                results = frame_model.inference(src = frame) #(n,6)
                self.tracker.update(results.to('cpu', dtype=torch.float32, non_blocking=True).numpy())
                
                BYTETracker.multi_predict(self.tracker,self.tracker.tracked_stracks)
                #pop stuff into preallocated array
                count = 0
                for t in self.tracker.tracked_stracks:
                    if t.is_activated:
                        self._tracked_buffer[count] = t.result
                        count += 1
                tracked_detections = self._tracked_buffer[:count]
                
                raw_deltas = (0,0)
                scaled_deltas = (0,0)
                if self.inputdetector.is_rmb_pressed:
                    self.target_selector.update_zoom_interpolation()#while right clicking we interpolate zoom till final zoom level
                else:
                    self.target_selector.reset_zoom()
                    
                if aimbot_active and len(self.tracker.tracked_stracks) > 0:
                    #print(f'AIMBOT ACTIVE')
                    raw_deltas, scaled_deltas = self.aimbot(self.tracker.tracked_stracks)
                
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
            track_buffer=int(target_frame_rate * 0.5),
            fuse_score=0.5,
            match_thresh=0.6,
            new_track_thresh=0.65
        )
        
        # Monkey-patch with Rust based tracker
        tracker = ByteTrack(max_age=5, min_hits=2, init_tracker_min_score=0.25)
        def update(self, dets, *args, **kwargs):
            boxes, cls = dets.data[:,:5], dets.data[:, -1:]
            tracks = tracker.update(boxes, return_indices=True)
            idxs = tracks[:, -1:].astype(int)
            confs = boxes[idxs.flatten(), 4:5]
            tracks = np.hstack((tracks[:, :-1], confs, cls[idxs.flatten()], idxs))
            return tracks
        BYTETracker.update = update
        
        self.tracker = BYTETracker(args, frame_rate=target_frame_rate)

    def aimbot(self, stracks:list):  
        """
        Args:
            stracks (list): list of STrack objects
        """
        
        # Filter stracks based on min_frames_to_target
        filtered_stracks = [s for s in stracks if s.frame_id - s.start_frame >= self.cfg['targeting_settings']['min_frames_to_target']]
        raw_deltas = (0,0)
        scaled_deltas = (0,0)
        if len(filtered_stracks) > 0:
            # Convert filtered_stracks back to the numpy array format for target_selector
            filtered_detections = np.asarray([x.result for x in filtered_stracks if x.is_activated])
            raw_deltas = self.target_selector.get_deltas(filtered_detections)
            if raw_deltas != (0,0):#if valid target
                scaled_deltas = self.mousemover.move_mouse_humanized(raw_deltas[0],raw_deltas[1])
        if self.cfg['targeting_settings']['lead_target']:
            self.target_selector.update_movement_buffer(scaled_deltas)
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