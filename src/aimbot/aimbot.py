from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker

import cv2
import time
import torch
import sys
from pathlib import Path
import cupy as cp
import numpy as np
github_dir = Path(__file__).parent.parent.parent.parent

sys.path.insert(0, str(github_dir / 'BettererCam'))  # Ensure BettererCam is in sys.path
print(f'Adding GITHUB DIR:{github_dir / "BettererCam"} to sys.path')
# can replace with bettercam just no cupy support, when init camera object set nvidia_gpu = False for bettercam
import betterercam
# print(betterercam.__file__)

sys.path.insert(0,str(Path.cwd())) 

from data_parsing import targetselector
from engine import model
from input import mousemover, inputdetector
from gui import gui_manager
from src.aimbot.utils import fpstracker
from src.aimbot.config_loader import ConfigLoader

from argparse import Namespace  
from screeninfo import get_monitors
import traceback
import argparse

class Aimbot:
    def __init__(self, config_path):
        print("Aimbot: Initializing...")
        self.config = ConfigLoader(config_path)
        self.init_model()
        print("Aimbot: init_model complete.")
        self.init_monitor()
        print("Aimbot: init_monitor complete.")
        #i think these have to be after monitor/model init
        self.init_input()
        print("Aimbot: init_input complete.")
        
        self.gui_manager = gui_manager.GUI_Manager(self.config.cfg['gui_settings'],self.scanning_hw_capture)
        print("Aimbot: GUI_Manager initialized.")
        
        if self.config.is_fps_tracked:
            self.fps_tracker = fpstracker.FPSTracker()
        
        self.init_camera()
        print("Aimbot: init_camera complete.")

        self.target_selector = targetselector.TargetSelector(
            cfg = self.config.cfg,
            detection_window_dim=self.ads_hw_capture,
            head_toggle=self.config.head_toggle,
            target_cls_id=self.config.target_cls_id,
            crosshair_cls_id=self.config.crosshair_cls_id,
            max_deltas = self.config.max_deltas,
            projectile_velocity=self.config.projectile_velocity,
            base_head_offset=self.config.base_head_offset,
            screen_hw= (self.screen_y,self.screen_x),
            zoom = self.config.zoom,
            hFOV_degrees= self.config.fov,
            predict_drop=self.config.predict_drop,
            predict_crosshair = self.config.predict_crosshair
        )
        print("Aimbot: target_selector initialized.")
        self.setup_bytetracker()
        print("Aimbot: setup_bytetracker complete.")
        
    def init_model(self):
        #working on multi model support (scanning/ads)
        ads_path = Path.cwd() / self.config.cfg['model']['ads_dir'] / self.config.cfg['model']['ads_filename']
        
        if not ads_path.exists():  
            raise FileNotFoundError(f"ADS model missing: {ads_path}")  
        
        scanning_path = Path.cwd() / self.config.cfg['model']['scanning_dir'] / self.config.cfg['model']['scanning_filename']
        
        if not scanning_path.exists() or scanning_path == ads_path:
            print(f'SCANNING MODEL PATH DOES NOT EXIST OR SAME AS ADS MODEL, FALLING BACK TO ADS MODEL')
            scanning_path = ads_path
        
        #this capture dimension is only used for pt models, engine models load dimensions internally
        pt_hw_capture = tuple(self.config.cfg['model']['pt_hw_capture'])
        
        #load both models
        #if models are the same, only load 1 model make both regions the same
        
        self.ads_model = model.Model(ads_path, hw_capture=pt_hw_capture)
        self.ads_hw_capture = self.ads_model.hw_capture
        
        if ads_path == scanning_path:
            self.scanning_model = self.ads_model
        else: 
            self.scanning_model = model.Model(scanning_path, hw_capture=pt_hw_capture)
        self.scanning_hw_capture = self.scanning_model.hw_capture
        self._tracked_buffer = np.empty((256, 8), dtype=np.float32)
        
        
        
    def init_monitor(self):
        #dynamic monitor settings
        monitor_idx = self.config.cfg['display_settings']['monitor_idx']
        monitor = get_monitors()[monitor_idx]
        print(f'LOOKING AT MONITOR: {monitor_idx}')
        self.screen_x = monitor.width
        self.screen_y = monitor.height
        print(f'MONITOR DIMS: {monitor.width} x {monitor.height}')
        self.screen_center = (self.screen_x // 2, self.screen_y // 2)

    
    def init_input(self):
        self.mousemover = mousemover.MouseMover(self.config.overall_sens,self.config.sens_scaling,self.config.max_deltas,self.config.jitter_strength,self.config.overshoot_strength,self.config.overshoot_chance,self.config.debug)
        self.inputdetector = inputdetector.InputDetector(self.config.debug, self.config.toggle_hotkey)
        self.inputdetector.start_input_detection()

    def init_camera(self):
        scanning_x_offset = (self.screen_x - self.scanning_hw_capture[1]) // 2
        scanning_y_offset = (self.screen_y - self.scanning_hw_capture[0]) // 2
        print(self.screen_x)
        print(self.scanning_hw_capture[1])
        ads_x_offset = (self.scanning_hw_capture[1] - self.ads_hw_capture[1]) // 2
        ads_y_offset = (self.scanning_hw_capture[0] - self.ads_hw_capture[0]) // 2
        
        #need to modify for dual region i think later
        scanning_region = (0 + scanning_x_offset, 0 + scanning_y_offset, self.screen_x - scanning_x_offset, self.screen_y - scanning_y_offset)
        
        self.ads_region = (
            scanning_region[0] + ads_x_offset,
            scanning_region[1] + ads_y_offset,
            scanning_region[2] - ads_x_offset,
            scanning_region[3] - ads_y_offset
        )

        # print(f'ads dimensions: {self.ads_hw_capture}')
        # print(f'scanning dimensions: {self.scanning_hw_capture}')
        # print(f'ads region: {self.ads_region}')
        # print(f'scanning region: {scanning_region}')
        
        #by default uses scanning region when ads, use ads region
        self.camera = betterercam.create(region = scanning_region, output_color='RGB',max_buffer_len=2, nvidia_gpu = True)
        
        #if using yolo inference need to use BGR since they assume your input is BGR
    
    
    
    def main(self):     
        print("Aimbot: Entering main loop.")
        frame_model:model.Model
        try:
            while True:
                aimbot_active = self.inputdetector.is_toggled and (self.inputdetector.is_rmb_pressed if self.config.right_click_toggle else True)
                
                if not aimbot_active:
                    time.sleep(self.config.inactive_throttle_ms / 1000)
                
                if self.inputdetector.is_rmb_pressed:
                    frame = self.camera.grab(region = self.ads_region)
                    frame_model = self.ads_model
                else:    
                    frame = self.camera.grab()
                    frame_model = self.scanning_model

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
                
                if aimbot_active and len(self.tracker.tracked_stracks) > 0:
                    self.aimbot(self.tracker.tracked_stracks)
                    
                if self.config.is_fps_tracked:
                    self.fps_tracker.update()
            
                if self.gui_manager:  
                    self.gui_manager.render(frame, tracked_detections, self.inputdetector.is_rmb_pressed)  
                
        except KeyboardInterrupt:
            print("\nShutting down...")
            self.cleanup()
            sys.exit(0)  # Clean exit
            
        except Exception as e:
            print(f"Fatal error: {traceback.print_exc()}")
            self.cleanup()
            sys.exit(1)
    
    def setup_bytetracker(self):
        #if engine is running just going to assume 144 is the target frame rate
        #if pt model is running its probably debug screen so 30
        target_frame_rate = 144 if self.ads_model.model_ext == ".engine" else 30
        args = self.config.get_tracker_args(target_frame_rate)
        self.tracker = BYTETracker(args, frame_rate=target_frame_rate)

    def aimbot(self, stracks:list):  
        """
        Args:
            stracks (list): list of STrack objects
        """
        
        # Filter stracks based on min_frames_to_target
        filtered_stracks = [s for s in stracks if s.frame_id - s.start_frame >= self.config.min_frames_to_target]

        if len(filtered_stracks) > 0:
            # Convert filtered_stracks back to the numpy array format for target_selector
            filtered_detections = np.asarray([x.result for x in filtered_stracks if x.is_activated])
            deltas = self.target_selector.get_deltas(filtered_detections)
            if deltas != (0,0):#if valid target
                self.mousemover.move_mouse_humanized(deltas[0],deltas[1])

    def cleanup(self):
        print("STARTING CLEANUP")
        try:
            if hasattr(self, 'camera') and self.camera:
                print('Releasing camera')
                self.camera.release()  # Ensure proper release
                del self.camera
                self.camera = None  # Explicitly clear reference
                
            if hasattr(self, 'gui_manager') and self.gui_manager:
                print('Cleaning up GUI')
                self.gui_manager.cleanup()
                del self.gui_manager
                self.gui_manager = None
            if hasattr(self, 'ads_model'):
                print('Removing ads_model')
                del self.ads_model
            if hasattr(self, 'scanning_model'):
                print('Removing scanning_model')
                del self.scanning_model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Cleanup error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO11 Aimbot")
    parser.add_argument('--config', type=str, default='config/cfg.json', help='Path to the configuration file')
    args = parser.parse_args()
    print("Aimbot: About to create Aimbot instance and run main.")
    Aimbot(args.config).main()