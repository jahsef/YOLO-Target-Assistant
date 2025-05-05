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
from gui import manager
from utils import fpstracker

from argparse import Namespace  
from screeninfo import get_monitors

class Aimbot:
    def __init__(self):
        # self._load_config()          # 1. Config first  
        # self._init_model()           # 2. Model (sets hw_capture)  
        # self._init_monitor()         # 3. Monitor (depends on hw_capture)  
        # self._init_input()           # 4. Mouse/input  
        # self._init_gui()             # 5. Debug/overlay (optional) 
        
        import json
        config_path = Path.cwd() / "aimbot_config.json"
        with open(config_path) as f:
            cfg = json.load(f)
        model_path = Path.cwd() / cfg['model']['base_dir'] / cfg['model']['name']
        pt_hw_capture = tuple(cfg['model']['hw_capture'])
        #okay this is a little bit confusing
        #hw_capture is only used for pt models, engine models load it themselves
        
        self.model = model.Model(model_path, hw_capture=pt_hw_capture)
        self.hw_capture = self.model.hw_capture#grab hw_capture from model since engine determines by itself
        #maybe i shouldnt even have hw_capture as a param but pt models can do whatever size so not sure
        self.load_monitor_settings()
        
        self.load_cfg(cfg)
        
        self.mousemover = mousemover.MouseMover(self.overall_sens,self.sens_scaling,self.max_deltas,self.jitter_strength,self.overshoot_strength,self.overshoot_chance,self.debug)
        self.inputdetector = inputdetector.InputDetector(self.debug)
        self.inputdetector.start_input_detection()
        if self.is_fps_tracked:
            self.fps_tracker = fpstracker.FPSTracker()
        self.setup_tracker()
        
            
        #empty boxes to pass into tracker if no detections
        
        capture_region = (0 + self.x_offset, 0 + self.y_offset, self.screen_x - self.x_offset, self.screen_y - self.y_offset)
        self.camera = betterercam.create(region = capture_region, output_color='RGB',max_buffer_len=2, nvidia_gpu = True)
        
        self.manager = manager.GUIManager(cfg['gui_settings'],self.hw_capture)
        
        #if using yolo inference need to use BGR since they assume your input is BGR
        
    def main(self):     
        
        try:
            while True:
                # if not self.inputdetector.is_rmb_pressed:
                #     #15ms sleep when not ads
                #     time.sleep(15e-3)
                frame = self.camera.grab()
                if frame is None:
                    #if no fresh frame available from directx frame buffer
                    continue
                
                results = self.model.inference(src = frame) #(n,6)
                boxes_results = self.model.parse_results_into_ultralytics_boxes(results)
                self.tracker.update(boxes_results.cpu().numpy())#kind of annoying to convert to cpu but whatever
                #could this be done more efficiently or convert somewhere earlier upstream?
                #predicts next positions of tracked detections then gets results
                BYTETracker.multi_predict(self.tracker,self.tracker.tracked_stracks)
                tracked_detections = np.asarray([x.result for x in self.tracker.tracked_stracks if x.is_activated])#(n,8)
                if self.manager:
                    self.manager.render_gui(frame,tracked_detections)
                    
                if self.inputdetector.is_rmb_pressed and len(tracked_detections) > 0:
                    self.aimbot(tracked_detections)
                if self.is_fps_tracked:
                    self.fps_tracker.update()
        except KeyboardInterrupt:
            print('Shutting down...')
        finally:
            self.cleanup()
            
    def load_monitor_settings(self):
        #dynamic monitor settings
        monitor = get_monitors()[0]
        self.screen_x = monitor.width
        self.screen_y = monitor.height
        self.screen_center = (self.screen_x // 2, self.screen_y // 2)
        self.x_offset = (self.screen_x - self.hw_capture[1]) // 2
        self.y_offset = (self.screen_y - self.hw_capture[0]) // 2
        
    def load_cfg(self,cfg:dict):
        """
        cfg loading is order dependent so cant change where this method is called really
        """
        #logging
        self.debug = cfg['logging']['debug']
        self.is_fps_tracked = cfg['logging']['fps']
        
        #sens settings
        self.overall_sens = cfg['sensitivity_settings']['overall_sens']
        self.sens_scaling = cfg['sensitivity_settings']['sens_scaling']
        self.max_deltas = cfg['sensitivity_settings']['max_deltas']
        self.jitter_strength = cfg['sensitivity_settings']['jitter_strength']
        self.overshoot_strength = cfg['sensitivity_settings']['overshoot_strength']
        self.overshoot_chance = cfg['sensitivity_settings']['overshoot_chance']
        #targeting/ bullet prediction settings
        #need self scope for debug stuff below
        self.target_cls_id = cfg['targeting_settings']['target_cls_id']
        self.crosshair_cls_id = cfg['targeting_settings']['crosshair_cls_id']
        head_toggle = cfg['targeting_settings']['head_toggle']
        predict_drop = cfg['targeting_settings']['predict_drop']
        predict_crosshair = cfg['targeting_settings']['predict_crosshair']
        zoom = cfg['targeting_settings']['zoom']
        projectile_velocity = cfg['targeting_settings']['projectile_velocity']
        base_head_offset = cfg['targeting_settings']['base_head_offset']
        fov = cfg['targeting_settings']['fov']
        
        self.target_selector = targetselector.TargetSelector(
            detection_window_dim=self.hw_capture,
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
        
        
    def setup_tracker(self):
        #if engine is running just going to assume 144 is the target frame rate
        #if pt model is running its probably debug screen as well so 30
        target_frame_rate = 144 if self.model.model_ext == ".engine" else 30
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
        """
        may need to improve cleanup though its probably not needed
        """
        if self.camera:
            self.camera.release()
            del self.camera
        torch.cuda.empty_cache()
        
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