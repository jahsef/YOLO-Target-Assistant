from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.engine.results import Boxes

import cv2
import threading
import time
import keyboard
import win32api
import win32con
# import os
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
from utils import targetselector, tensorrt_engine
from argparse import Namespace  
from screeninfo import get_monitors


class Aimbot:
    def __init__(self):
        import json
        config_path = Path.cwd() / "aimbot_config.json"
        with open(config_path) as f:
            cfg = json.load(f)
        self.debug = cfg['debug']
        model_path = Path.cwd() / cfg['model']['base_dir'] / cfg['model']['name']
        pt_hw_capture = tuple(cfg['model']['hw_capture'])
        #hw_capture is only used for pt models, engine models load it themselves
        self.load_model(model_path, hw_capture=pt_hw_capture)
        self.load_monitor_settings()
        self.load_targeting_cfg(cfg)

        self.is_key_pressed = False
        self.fps_tracker = FPSTracker()
        self.setup_tracker()

        if self.debug:
            window_height, window_width = self.hw_capture  
            cv2.namedWindow("Screen Capture Detection", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Screen Capture Detection", window_width, window_height)
            
        #empty boxes to pass into tracker if no detections
        self.empty_boxes = Boxes(boxes=torch.empty((0, 6), device=torch.device('cuda')),orig_shape=self.hw_capture)
        
        capture_region = (0 + self.x_offset, 0 + self.y_offset, self.screen_x - self.x_offset, self.screen_y - self.y_offset)
        self.camera = betterercam.create(region = capture_region, output_color='RGB',max_buffer_len=2, nvidia_gpu = True)
        #if using yolo inference need to use BGR since they assume your input is BGR
        
    def main(self):     
        threading.Thread(target=self.input_detection, daemon=True).start()
        
        try:
            while True:
                frame = self.camera.grab()
                if frame is None:
                    #if no fresh frame available from directx frame buffer
                    continue
                
                if self.model_ext == '.engine':
                    processed_frame = self.preprocess_tensorrt(frame)
                    self.inference_tensorrt(processed_frame)
                elif self.model_ext == '.pt':
                    processed_frame = self.preprocess_torch(frame)
                    self.inference_torch(processed_frame)
                
                #predicts next positions of tracked detections then gets results
                BYTETracker.multi_predict(self.tracker,self.tracker.tracked_stracks)
                tracked_detections = np.asarray([x.result for x in self.tracker.tracked_stracks if x.is_activated])
                
                if self.debug:
                    display_frame = cp.asnumpy(frame)
                    display_frame = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
                    self.debug_render(display_frame)
                
                if self.is_key_pressed and len(tracked_detections) > 0:
                    self.aimbot(tracked_detections)
                    
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
        
    def load_targeting_cfg(self,cfg:dict):
        #sens settings
        sens = cfg['sensitivity_settings']['sens']
        rand_sens_mult_std_dev = cfg['sensitivity_settings']['rand_sens_mult_std_dev']
        min_sens_mult = cfg['sensitivity_settings']['min_sens_mult']
        max_deltas = cfg['sensitivity_settings']['max_deltas']
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
            max_deltas = max_deltas,
            sensitivity = sens,
            projectile_velocity=projectile_velocity,
            base_head_offset=base_head_offset,
            screen_hw= (self.screen_y,self.screen_x),
            zoom = zoom,
            hFOV_degrees= fov,
            rand_sens_mult_std_dev= rand_sens_mult_std_dev,
            predict_drop=predict_drop,
            min_sens_mult=min_sens_mult,
            predict_crosshair = predict_crosshair
        )
        
        
    def load_model(self, model_path: Path, hw_capture = (640,640)):
        self.model_ext = model_path.suffix
        if self.model_ext == '.engine':
            self.model = tensorrt_engine.TensorRT_Engine(engine_file_path= model_path, conf_threshold= .25,verbose = False)
            self.hw_capture = self.model.imgsz
            if self.model == None:
                raise Exception("tensorrt engine not loading correctly maybe set verbose = True")
        elif self.model_ext == '.pt':
            self.hw_capture = hw_capture#can be set to whatever
            self.model = YOLO(model = model_path)
        else:
            raise Exception(f'not supported file format: {self.model_ext} <- file format should be here lol')
        
    def setup_tracker(self):
        #if engine is running just going to assume 144 is the target frame rate
        #if pt model is running its probably debug screen as well so 30
        target_frame_rate = 144 if self.model_ext == ".engine" else 30
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
        if self.camera:
            self.camera.release()
            del self.camera
        if self.debug:
            cv2.destroyAllWindows()
        torch.cuda.empty_cache()
        
    def aimbot(self, detections):  
        deltas = self.target_selector.get_deltas(detections)
        if deltas is not None:#if valid target
            self.move_mouse_to_bounding_box(deltas)

    def move_mouse_to_bounding_box(self, deltas):
        delta_x = deltas[0]
        delta_y = deltas[1]
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(delta_x), int(delta_y), 0, 0)
            
    def input_detection(self):
        #SHOULD ADD A ZOOM TOGGLE THING
        def on_key_press(event):
            if event.name.lower() == 'y':
                self.is_key_pressed = not self.is_key_pressed
                print(f"Key toggled: {self.is_key_pressed}")
                
        keyboard.on_press(on_key_press)
        while True:#keeps thread alive
            time.sleep(.1)
            
    def parse_results_into_boxes(self,results: torch.Tensor, img_sz: tuple ) -> Boxes:
        #need to convert into boxes to pass into the ultralytics BYTETracker
        if len(results) == 0:#xyxy, conf, cls, smth else?
            return Boxes(boxes=torch.empty((0, 6)), orig_shape=img_sz)
        converted_boxes = Boxes(
            boxes=results,
            orig_shape=img_sz
        )
        return converted_boxes 
    
    #almost 25% speedup over np (preprocess + inference)
    #also python doesnt have method overloading by parameter type
    def _preprocess_frame(self,frame:cp.ndarray) -> cp.ndarray:
        bchw = frame.transpose(2, 0, 1)[cp.newaxis, ...]
        float_frame = bchw.astype(cp.float32, copy=False)#engine expects float 32 unless i export it differently
        float_frame /= 255.0 #/= is inplace, / creates a new cp arr
        return float_frame
    
    def preprocess_tensorrt(self,frame: cp.ndarray) -> cp.ndarray:
        return cp.ascontiguousarray(self._preprocess_frame(frame))

    def inference_tensorrt(self,src:cp.ndarray) -> np.ndarray:
        results = self.model.inference_cp(src = src)
        results = torch.as_tensor(results)#should be pretty inexpensive since it references the same memory
        results = self.parse_results_into_boxes(results, self.hw_capture)
        return self.tracker.update(results.cpu().numpy())
        
    def preprocess_torch(self,frame: cp.ndarray) -> torch.Tensor:
        return torch.as_tensor(self._preprocess_frame(frame)).contiguous()   
    
    @torch.inference_mode()
    def inference_torch(self,source:torch.Tensor) -> np.ndarray:
        results = self.model(source=source,
            conf = .25,
            imgsz=self.hw_capture,
            verbose = False
        )
        if results[0].boxes.data.numel() == 0: 
            return self.tracker.update(self.empty_boxes.cpu().numpy())

        return self.tracker.update(results[0].boxes.cpu().numpy())

    def debug_render(self,frame):
        display_frame = cp.asnumpy(frame)
        stracks = [x for x in self.tracker.tracked_stracks if x.is_activated]#not sure if activated is needed
        #strack.results doesnt have what i want
        
        for strack in stracks:
            x1, y1, x2, y2, = map(int,strack.xyxy)
            if strack.cls == self.crosshair_cls_id:#if crosshair
                cv2.circle(display_frame, ((x1 + x2) // 2, (y1 + y2) // 2),4, (0, 255, 0), -1)
                cv2.putText(display_frame, f"score: {strack.score:.2f}", (int(x1), int(y1) - 12),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 204), thickness = 1)
                cv2.circle(display_frame, ((x1 + x2) // 2, (y1 + y2) // 2),4, (255, 0, 204), -1)
                # cv2.circle(display_frame, ((x1 + x2) // 2, (y1 + y2)// 2 - int((y2-y1)*.39)),4, (255, 0, 204), -1)
                cv2.putText(display_frame, f"score: {strack.score:.2f}", (int(x1), int(y1) - 48),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(display_frame, f"cls_id: {strack.cls}", (int(x1), int(y1) - 36),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(display_frame, f"ID: {strack.track_id}", (int(x1), int(y1) - 24),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(display_frame, f"nums_frames_seen: {strack.end_frame - strack.start_frame}", (int(x1), int(y1) - 12),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Screen Capture Detection", display_frame)
        cv2.waitKey(1)

class FPSTracker:
    def __init__(self, update_interval=10.0):
        self.frame_count = 0
        self.last_update = time.perf_counter()
        self.update_interval = update_interval

    def update(self):
        self.frame_count += 1
        current_time = time.perf_counter()
        
        if current_time - self.last_update >= self.update_interval:
            fps = self.frame_count / (current_time - self.last_update)
            print(f'FPS: {fps:.2f}')
            self.frame_count = 0
            self.last_update = current_time

if __name__ == "__main__":
    Aimbot().main()