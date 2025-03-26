from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.engine.results import Boxes

import cv2
import threading
import time
import keyboard
import win32api
import win32con
import os
import torch
import sys
from pathlib import Path
import cupy as cp
import numpy as np
github_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(os.path.join(github_dir,'BettererCam')))
# can replace with bettercam just no cupy support
import betterercam
# print(betterercam.__file__)
from utils import targetselector, tensorrt_engine
from argparse import Namespace  

# import pycuda.autoinit#auto init cuda mem context
#no no init context use default context


class Main:
    def __init__(self):
        self.debug = False
        self.screen_x = 2560
        self.screen_y = 1440#should add auto detection for dimensions
        self.target_cls_id = 0#make sure class is is correct
        
        base_dir = "runs/train/EFPS_4000img_11n_1440p_batch11_epoch100\weights"
        model_name = "320x320_fp16True_int8False_stripped.engine"#tensorrt api needs a stripped metadata model if trained using yolo
        model_path = os.path.join(os.getcwd(), base_dir, model_name)
        #hw_capture for engine format defined by engine file
        #not using yolo for loading engine they stink
        self.load_model(model_path, hw_capture = (640,640))
        self.is_key_pressed = False
        self.screen_center = (self.screen_x // 2,self.screen_y // 2)
        self.x_offset = (self.screen_x - self.hw_capture[1])//2
        self.y_offset = (self.screen_y - self.hw_capture[0])//2
        self.fps_tracker = FPSTracker()
        self.head_toggle = True
        self.target_dimensions = self.hw_capture#aimbot window, dont really need
        self.setup_tracking()
        self.setup_targeting()
        if self.debug:
            window_height, window_width = self.hw_capture
            cv2.namedWindow("Screen Capture Detection", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Screen Capture Detection", window_width, window_height)
            
        self.empty_boxes = Boxes(boxes=torch.empty((0, 6), device=torch.device('cuda')),orig_shape=self.hw_capture)
        capture_region = (0 + self.x_offset, 0 + self.y_offset, self.screen_x - self.x_offset, self.screen_y - self.y_offset)
        self.camera = betterercam.create(region = capture_region, output_color='RGB',max_buffer_len=2, nvidia_gpu = True)#yolo does bgr -> rgb conversion in model.predict automatically
        
    def main(self):     
        threading.Thread(target=self.input_detection, daemon=True).start()
        
        try:
            while True:
                # time.sleep(1)
                frame = self.camera.grab()
                if frame is None:
                    continue
                if self.model_ext == '.engine':
                    processed_frame = self.preprocess(frame)
                    self.inference(processed_frame)
                elif self.model_ext == '.pt':
                    processed_frame = self.preprocess_torch(frame)
                    self.inference_torch(processed_frame)

                BYTETracker.multi_predict(self.tracker,self.tracker.tracked_stracks)
                tracked_objects = np.asarray([x.result for x in self.tracker.tracked_stracks if x.is_activated])
                if self.debug:
                    display_frame = cp.asnumpy(frame)
                    display_frame = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
                    self.debug_render(display_frame)
                if self.is_key_pressed and len(tracked_objects) > 0:
                    self.aimbot(tracked_objects)
                self.fps_tracker.update()
        except KeyboardInterrupt:
            print('Shutting down...')
        finally:
            self.cleanup()

    def load_model(self, model_path, hw_capture: tuple[int,int]):
        model_name = os.path.basename(model_path)
        self.model_ext = os.path.splitext(model_name)[1]
        if self.model_ext == '.engine':
            self.model = tensorrt_engine.TensorRT_Engine(engine_file_path= model_path, conf_threshold= .25,verbose = False)
            self.hw_capture = self.model.imgsz
            if self.model == None:
                raise Exception("tensorrt engine not loading correctly maybe set verbose = True")
        elif self.model_ext == '.pt':
            self.hw_capture = hw_capture#can be set to whatever
            self.model = YOLO(model = model_path)
        else:
            raise Exception(f'wrong file format dingus: {self.model_ext}')
        
    def setup_tracking(self):
        target_frame_rate = 144
        args = Namespace(
            track_high_thresh=.65,
            track_low_thresh=.4,
            track_buffer=target_frame_rate//2,
            fuse_score=.5,
            match_thresh=.6,
            new_track_thresh=0.65
        )
        self.tracker = BYTETracker(args, frame_rate=target_frame_rate)
    
    def setup_targeting(self):
        self.screen_center = (self.screen_x // 2, self.screen_y // 2)
        self.x_offset = (self.screen_x - self.hw_capture[1])//2
        self.y_offset = (self.screen_y - self.hw_capture[0])//2
        
        self.target_selector = targetselector.TargetSelector(
            screen_center=self.screen_center,
            x_offset=self.x_offset,
            y_offset=self.y_offset,
            head_toggle=self.head_toggle,
            target_dimensions= self.target_dimensions
        )

            
    def cleanup(self):
        if self.camera:
            self.camera.release()
            del self.camera
        if self.debug:
            cv2.destroyAllWindows()
        torch.cuda.empty_cache()
        
    def aimbot(self, tracked_objects):  
        deltas = self.target_selector.return_deltas(tracked_objects)
        # print(type(deltas))
        if deltas:#if valid target
            self.move_mouse_to_bounding_box(deltas)

    def move_mouse_to_bounding_box(self, deltas):
        delta_x = deltas[0]
        delta_y = deltas[1]
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(delta_x), int(delta_y), 0, 0)
            
    def input_detection(self):
        def on_key_press(event):
            if event.name.lower() == 'e':
                self.is_key_pressed = not self.is_key_pressed
                print(f"Key toggled: {self.is_key_pressed}")

                
        keyboard.on_press(on_key_press)
        while True:
            time.sleep(.1)
            
    def parse_results_into_boxes(self,results: torch.Tensor, orig_shape: tuple, ):
        #should probably make this work for both .engine and .pt
        # Check if results are empty
        if len(results) == 0:
            return Boxes(boxes=torch.empty((0, 6)), orig_shape=orig_shape)

        # Filter results by class ID
        cls_mask = results[:, 5] == self.target_cls_id  # Create mask for cls_target
        filtered_results = results[cls_mask]   # Apply mask to filter results

        # If no boxes match the target class, return empty Boxes object
        if len(filtered_results) == 0:
            return self.empty_boxes

        # Construct Boxes object
        boxes = Boxes(
            boxes=filtered_results,  # Filtered results [x1, y1, x2, y2, conf, cls_id]
            orig_shape=orig_shape    # Original image dimensions (height, width)
        )

        return boxes 
    
    #almost 25% speedup over np (preprocess + inference)
    #also python doesnt have method overloading by parameter type
    def preprocess(self,frame: cp.ndarray):
        bchw = frame.transpose(2, 0, 1)[cp.newaxis, ...]
        float_frame = bchw.astype(cp.float32, copy=False)
        float_frame /= 255.0 #/= is inplace, / creates a new cp arr
        return cp.ascontiguousarray(float_frame)

    def inference(self,source):
        results = self.model.inference_cp(input_data = source)
        results = torch.as_tensor(results)#should be pretty inexpensive since it still stays on gpu
        results = self.parse_results_into_boxes(results, self.hw_capture)
        return self.tracker.update(results.cpu().numpy())
    
    @torch.inference_mode()
    def inference_torch(self,source):
        results = self.model(source=source,
             conf = .25,
             imgsz=self.hw_capture,
             verbose = False
         )
        if results[0].boxes.data.numel() == 0: 
            enemy_boxes = self.empty_boxes
        else:
            cls_target = 0
            enemy_mask = results[0].boxes.cls == cls_target
            filtered_data = results[0].boxes.data[enemy_mask]
            enemy_boxes = Boxes(
                boxes=filtered_data,
                orig_shape=self.hw_capture
            )
    # coords = self.xyxy if self.angle is None else self.xywha
    # return coords.tolist() + [self.track_id, self.score, self.cls, self.idx]
    #STrack object results (byte tracker list[STrack])
        return self.tracker.update(enemy_boxes.cpu().numpy())

    def preprocess_torch(self,frame: cp.ndarray) -> torch.Tensor:
         bchw = cp.ascontiguousarray(frame.transpose(2, 0, 1)[cp.newaxis, ...])
         float_frame = bchw.astype(cp.float16, copy=False)
         float_frame /= 255.0 #/= is inplace, / creates a new cp arr
         return torch.as_tensor(float_frame)
    
    def debug_render(self,frame):
        display_frame = cp.asnumpy(frame)
        stracks = [x for x in self.tracker.tracked_stracks if x.is_activated]#not sure if activated is needed
        #strack.results doesnt have what i want
        
        for strack in stracks:
            x1, y1, x2, y2, = map(int,strack.xyxy)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 204), thickness = 2)
            cv2.putText(display_frame, f"ID: {strack.track_id}", (int(x1), int(y1) - 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display_frame, f"nums_frames_seen: {strack.end_frame - strack.start_frame}", (int(x1), int(y1) - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("Screen Capture Detection", display_frame)
        cv2.waitKey(1)
        


class FPSTracker:
    def __init__(self, update_interval=1.0):
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
    Main().main()