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
from utils import targetselector
from argparse import Namespace  


class Main:
    def __init__(self):
        self.debug = False
        self.screen_x = 2560
        self.screen_y = 1440
        self.h_w_capture = (896,1440)#height,width
        self.is_key_pressed = False

        self.detections = []
        self.screen_center = (self.screen_x // 2,self.screen_y // 2)
        self.x_offset = (self.screen_x - self.h_w_capture[1])//2
        self.y_offset = (self.screen_y - self.h_w_capture[0])//2
        head_toggle = True
        self.target_selector = targetselector.TargetSelector(hysteresis= 1.5, proximity_threshold_sq= 75**2, screen_center=self.screen_center, x_offset=self.x_offset, y_offset= self.y_offset, head_toggle= head_toggle)
        
        args = Namespace(track_high_thresh = .7, track_low_thresh = .4,track_buffer=30, fuse_score = .5, match_thresh = .6,new_track_thresh=0.6)
        #threshold is for conf, buffer is how long to wait before drop detection, fuse score to fuse detections, lower is aggressive, high is conservative, match thresh matching detections between frames with iou
        self.tracker = BYTETracker(args,frame_rate=60)
        
    def main(self):     
        threading.Thread(target=self.input_detection, daemon=True).start()
        self.run_screen_capture_detection()

    def aimbot(self, detections):  
        target_bb = self.target_selector.get_target(detections)
        self.move_mouse_to_bounding_box(target_bb)
                
    def move_mouse_to_bounding_box(self, detection):
        center_bb_x = detection[0]
        center_bb_y = detection[1]
        delta_x = center_bb_x - self.screen_center[0]
        delta_y = center_bb_y - self.screen_center[1]
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(delta_x), int(delta_y), 0, 0)

    def input_detection(self):
        def on_key_press(event):
            if event.name.lower() == 'e':
                self.is_key_pressed = not self.is_key_pressed
                print(f"Key toggled: {self.is_key_pressed}")
        keyboard.on_press(on_key_press)
        while True:
            time.sleep(1)
    
    #almost 25% speedup over np (preprocess + inference)
    #also python doesnt have method overloading by parameter type
    def preprocess(self,frame: cp.ndarray) -> torch.Tensor:
        bchw = cp.ascontiguousarray(frame.transpose(2, 0, 1)[cp.newaxis, ...])
        float_frame = bchw.astype(cp.float16, copy=False)
        float_frame /= 255.0 #/= is inplace, / creates a new cp arr
        return torch.as_tensor(float_frame, device='cuda')
    
    # def preprocess(self,frame: np.ndarray) -> torch.Tensor:
    #     # Use in-place normalization to avoid copies
    #     return (
    #         torch.as_tensor(frame,device = 'cuda',dtype = torch.uint8)
    #         .permute(2, 0, 1)
    #         .unsqueeze_(0)  # In-place add batch dim
    #         .half()        
    #         .div_(255.0)    # In-place normalization
    #         .contiguous() #need for tensorrt
    #     )
    @torch.inference_mode()
    def run_screen_capture_detection(self):
        capture_region = (0 + self.x_offset, 0 + self.y_offset, self.screen_x - self.x_offset, self.screen_y - self.y_offset)
        cwd = os.getcwd()

        model = YOLO(os.path.join(cwd,"runs/train/EFPS_4000img_11s_1440p_batch6_epoch200/weights/best.engine"))
        # model = YOLO(os.path.join(cwd,"runs/train/EFPS_3000image_realtrain_1440x1440_100epoch_batch6_11s/weights/best.engine"))
        
        # model = YOLO(os.path.join(cwd,'runs/train/EFPS_4000img_11m_1440p_batch6_epoch200/weights/best.pt'))
  
        
        if self.debug:
            window_height, window_width = self.h_w_capture
            cv2.namedWindow("Screen Capture Detection", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Screen Capture Detection", window_width, window_height)
        camera = betterercam.create(region = capture_region, output_color='BGR',max_buffer_len=2, nvidia_gpu = True)#yolo does bgr -> rgb conversion in model.predict automatically
        
        frame_count = 0
        last_fps_update = time.perf_counter()
        while True:
            frame = camera.grab()
            if frame is None:
                continue
            gpu_frame = self.preprocess(frame)

            results = model(source=gpu_frame,
                conf = .6,
                imgsz=self.h_w_capture,
                verbose = False
            )

            if results[0].boxes.data.numel() == 0:  # No detections
                # print("No detections found in this frame.")
                enemy_boxes = Boxes(
                    boxes=torch.empty((0, 6), device="cuda:0"),  # Empty tensor with shape (0, 6)
                    orig_shape=results[0].boxes.orig_shape
                )
            else:
                # Filter for enemies (cls == 0)
                cls_target = 0
                enemy_mask = results[0].boxes.cls == cls_target

                # Extract filtered data
                filtered_data = results[0].boxes.data[enemy_mask]  # Filter the 'data' attribute

                # Create a new Boxes object
                enemy_boxes = Boxes(
                    boxes=filtered_data,  # Pass the filtered 'data' tensor
                    orig_shape=results[0].boxes.orig_shape
                )

            tracked_objects = self.tracker.update(enemy_boxes.cpu().numpy())
            
            if self.debug:
                display_frame = cp.asnumpy(frame)
                for i in range(len(tracked_objects)):
                    x1, y1, x2, y2, = map(int,tracked_objects[i][:4])
                    # print(tracked_objects[i])

                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (200, 25, 225), thickness = 2)
                    # cv2.putText(display_frame, f"ID: {track_id}", (int(x1), int(y1) - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.imshow("Screen Capture Detection", display_frame)
                cv2.waitKey(1)
                
            if(self.is_key_pressed and len(tracked_objects) > 0):
                self.aimbot(tracked_objects)
            self.detections.clear()
            frame_count+=1
            current_time = time.perf_counter()
            if current_time - last_fps_update >= 1:
                fps = frame_count / (current_time - last_fps_update)
                last_fps_update = current_time
                frame_count = 0
                print(f'fps: {fps:.2f}')

        
            

if __name__ == "__main__":
    Main().main()