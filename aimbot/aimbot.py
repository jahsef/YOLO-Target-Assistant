from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker

import cv2
import time
import torch
import sys
from pathlib import Path
import cupy as cp
import numpy as np
github_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(github_dir / 'Betterercam'))
# can replace with bettercam just no cupy support, when init camera object set nvidia_gpu = False for bettercam
import betterercam
# print(betterercam.__file__)

from data_parsing import targetselector
from engine import model
from input import mousemover, inputdetector
from gui import gui_manager
from utils import fpstracker

from argparse import Namespace  
from screeninfo import get_monitors
import traceback


class Aimbot:
    def __init__(self):

        self.load_config_file()
        self.init_model()
        self.init_monitor()
        #i think these have to be after monitor/model init
        self.load_aim_config()
        self.load_other_config()
        self.init_input()
        
        self.gui_manager = gui_manager.GUI_Manager(self.cfg['gui_settings'],self.scanning_hw_capture)
        
        if self.is_fps_tracked:
            self.fps_tracker = fpstracker.FPSTracker()
        
        self.init_camera()
        self.setup_bytetracker()
        
    def load_config_file(self):
        import json
        config_path = Path.cwd() / "aimbot_config.json"
        with open(config_path) as f:
            self.cfg = json.load(f)
    
    def init_model(self):
        #working on multi model support (scanning/ads)
        ads_path = Path.cwd() / self.cfg['model']['ads_dir'] / self.cfg['model']['ads_filename']
        
        if not ads_path.exists():  
            raise FileNotFoundError(f"ADS model missing: {ads_path}")  
        
        scanning_path = Path.cwd() / self.cfg['model']['scanning_dir'] / self.cfg['model']['scanning_filename']
        
        if not scanning_path.exists() or scanning_path == ads_path:
            print(f'SCANNING MODEL PATH DOES NOT EXIST OR SAME AS ADS MODEL, FALLING BACK TO ADS MODEL')
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
        monitor = get_monitors()[0]
        self.screen_x = monitor.width
        self.screen_y = monitor.height
        self.screen_center = (self.screen_x // 2, self.screen_y // 2)

    
    def init_input(self):
        self.mousemover = mousemover.MouseMover(self.overall_sens,self.sens_scaling,self.max_deltas,self.jitter_strength,self.overshoot_strength,self.overshoot_chance,self.debug)
        self.inputdetector = inputdetector.InputDetector(self.debug)
        self.inputdetector.start_input_detection()

    def init_camera(self):
        scanning_x_offset = (self.screen_x - self.scanning_hw_capture[1]) // 2
        scanning_y_offset = (self.screen_y - self.scanning_hw_capture[0]) // 2
        
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
        frame_model:model.Model
        try:
            while True:
                # if not self.inputdetector.is_rmb_pressed:
                # #     #15ms sleep when not ads
                #     time.sleep(15e-3)
                
                #if rmb pressed, use smaller region else just use main region
                if self.inputdetector.is_rmb_pressed:
                    frame = self.camera.grab(region = self.ads_region)
                    frame_model = self.ads_model

                else:    
                    #uses default region (scanning)
                    frame = self.camera.grab()
                    frame_model = self.scanning_model

                if frame is None:
                    #if no fresh frame available from directx frame buffer
                    continue    

                results = frame_model.inference(src = frame) #(n,6)
                #results are converted into ultralytics boxes already
                # boxes_results = frame_model.parse_results_into_ultralytics_boxes(results)
                self.tracker.update(results.cpu().numpy())
                
                #could possibly convert somewhere upstream but idk if thats even more efficient
                #predicts next positions of tracked detections then gets results
                BYTETracker.multi_predict(self.tracker,self.tracker.tracked_stracks)
                tracked_detections = np.asarray([x.result for x in self.tracker.tracked_stracks if x.is_activated])#(n,8)
                
                if self.inputdetector.is_rmb_pressed and len(tracked_detections) > 0:
                    self.aimbot(tracked_detections)
                    # print(tracked_detections)
                    
                if self.is_fps_tracked:
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
    
    def load_other_config(self):
        #logging
        self.debug = self.cfg['logging']['debug']
        self.is_fps_tracked = self.cfg['logging']['fps']
        
    def load_aim_config(self):
        """
        cfg loading is order dependent so cant change where this method is called really
        """
        #sens settings
        self.overall_sens = self.cfg['sensitivity_settings']['overall_sens']
        self.sens_scaling = self.cfg['sensitivity_settings']['sens_scaling']
        self.max_deltas = self.cfg['sensitivity_settings']['max_deltas']
        self.jitter_strength = self.cfg['sensitivity_settings']['jitter_strength']
        self.overshoot_strength = self.cfg['sensitivity_settings']['overshoot_strength']
        self.overshoot_chance = self.cfg['sensitivity_settings']['overshoot_chance']
        #targeting/ bullet prediction settings
        #need self scope for debug stuff below
        self.target_cls_id = self.cfg['targeting_settings']['target_cls_id']
        self.crosshair_cls_id = self.cfg['targeting_settings']['crosshair_cls_id']
        head_toggle = self.cfg['targeting_settings']['head_toggle']
        predict_drop = self.cfg['targeting_settings']['predict_drop']
        predict_crosshair = self.cfg['targeting_settings']['predict_crosshair']
        zoom = self.cfg['targeting_settings']['zoom']
        projectile_velocity = self.cfg['targeting_settings']['projectile_velocity']
        base_head_offset = self.cfg['targeting_settings']['base_head_offset']
        fov = self.cfg['targeting_settings']['fov']
        
        self.target_selector = targetselector.TargetSelector(
            detection_window_dim=self.ads_hw_capture,
            head_toggle=head_toggle,
            target_cls_id=self.target_cls_id,
            crosshair_cls_id=self.crosshair_cls_id,
            max_deltas = self.max_deltas,
            projectile_velocity=projectile_velocity,
            base_head_offset=base_head_offset,
            screen_hw= (self.screen_y,self.screen_x),
            zoom = zoom,
            hFOV_degrees= fov,
            predict_drop=predict_drop,
            predict_crosshair = predict_crosshair
        )
        
        
    def setup_bytetracker(self):
        #if engine is running just going to assume 144 is the target frame rate
        #if pt model is running its probably debug screen so 30
        target_frame_rate = 144 if self.ads_model.model_ext == ".engine" else 30
        args = Namespace(
            track_high_thresh=.65,
            track_low_thresh=.4,
            track_buffer=target_frame_rate//2,
            fuse_score=.5,
            match_thresh=.6,
            new_track_thresh=0.65
        )
        self.tracker = BYTETracker(args, frame_rate=target_frame_rate)

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
            
    def aimbot(self, detections:np.ndarray):  
        """
        Args:
            detections (np.ndarray):detection array FROM TRACKER of shape (n, 8) where columns are:
            [x1, y1, x2, y2, track_id, confidence, class_id, strack_idx]
        """
        deltas = self.target_selector.get_deltas(detections)
        if deltas != (0,0):#if valid target
            self.mousemover.move_mouse_humanized(deltas[0],deltas[1])


if __name__ == "__main__":
    Aimbot().main()