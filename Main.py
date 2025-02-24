from ultralytics import YOLO
import cv2
import numpy as np
import dxcam
import threading
import time
import keyboard
import logging
import win32api
import win32con
import math
from multiprocessing import Process, Manager

import os
import torch


# Suppress YOLO logs by adjusting the logging level
logging.getLogger('ultralytics').setLevel(logging.ERROR)


class Main:
     
    def main(self):
        self.debug = False
        self.screen_center_x = 2560 / 2  # Replace with your screen's width / 2
        self.screen_center_y = 1440 / 2  # Replace with your screen's height / 2
        self.is_key_pressed = False

        self.frame_times = []
        self.time_interval_frames = 60
        self.results = []
        self.detections = []
        
        threading.Thread(target=self.input_detection, daemon=True).start()
        threading.Thread(target=self.calculate_fps, daemon=True).start()

        # Run screen capture in the main thread
        self.run_screen_capture_detection()

    def aimbot(self):  
        detection = self.select_target_bounding_box()
        self.move_mouse_to_bounding_box(detection)
                

    def move_mouse_to_bounding_box(self, detection):
        center_bb_x = detection[0]
        center_bb_y = detection[1]
        delta_x = center_bb_x - self.screen_center_x
        delta_y = center_bb_y - self.screen_center_y
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(delta_x), int(delta_y), 0, 0)



    def select_target_bounding_box(self) -> tuple[int,int]:
        head_detection_dict = {}# center(tuple) : list[dist(int NEED TO CONVERT),area(int)]
        zombie_detection_dict = {}

        for detection in self.detections:
            x1, y1, x2, y2 = detection['bbox']
            curr_center_coords = ((x2 + x1) / 2, (y2 + y1) / 2)
            curr_area = (x2-x1)*(y2-y1)
            curr_dist = ((x2-x1)**2 + (y2-y1)**2)**.5
        
            if curr_dist == 0:
                score = float('inf')
            else:
                score = curr_area**1.5/curr_dist**.5

            if detection['class_name'] == 'Head':
                head_detection_dict[curr_center_coords] = score
            else:
                zombie_detection_dict[curr_center_coords] = score
        if len(head_detection_dict) > 0: 
            sorted_heads = sorted(head_detection_dict, key = head_detection_dict.__getitem__,reverse = True)
            return sorted_heads[0]
        else:
            sorted_zombies = sorted(zombie_detection_dict, key = zombie_detection_dict.__getitem__,reverse = True)
            return sorted_zombies[0]
            
        
        
        # return largest_head_coords if largest_head_area > 0 else largest_body_coords#edgecase no detections handled in aimbot prolly 
    
    def input_detection(self):
        def on_key_press(event):
            if event.name == 'e':
                self.is_key_pressed = not self.is_key_pressed
                print(f"Key toggled: {self.is_key_pressed}")
        keyboard.on_press(on_key_press)
        while True:
            time.sleep(2)
        
        # while True:
        #     time.sleep(0.1)
            # if self.is_key_pressed:
            #     if keyboard.is_pressed('a'):
            #         print('moving left')
            #         win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, 40, 0, 0, 0)
                    
            #     elif keyboard.is_pressed('d'):
            #         print('moving right')
            #         win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, -40, 0, 0, 0)


    def preprocess(self,frame: np.ndarray, target_size: tuple = (1440, 1440)) -> torch.Tensor:
        """
        Converts a BGR numpy frame to a GPU tensor in YOLO format (FP16, normalized, resized).
        """
        # 1. Resize with letterboxing (maintain aspect ratio)
        h, w = frame.shape[:2]
        scale = min(target_size[0] / h, target_size[1] / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # GPU-accelerated resize (using PyTorch)
        frame_tensor = torch.as_tensor(frame, device="cuda", dtype=torch.float16)  # Zero-copy to GPU
        frame_tensor = torch.permute(frame_tensor, (2, 0, 1))  # HWC → CHW (3, H, W)
        
        # Resize using bilinear interpolation (on GPU)
        frame_resized = torch.nn.functional.interpolate(
            frame_tensor.unsqueeze(0),  # Add batch dim
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)
        
        # 2. Normalize (if your model expects [0,1] instead of [0,255])
        frame_resized /= 255.0  # Normalize to [0,1]
        
        # 3. Pad to target_size (1440x1440) with 114s (YOLO convention)
        pad_h = target_size[0] - new_h
        pad_w = target_size[1] - new_w
        frame_padded = torch.nn.functional.pad(
            frame_resized,
            (0, pad_w, 0, pad_h),  # (left, right, top, bottom)
            value=114.0 / 255.0  # YOLO's "fill" value
        )
        
        # 4. Add batch dimension and return
        return frame_padded.unsqueeze(0)  # Shape: [1, 3, 1440, 1440]

    def run_screen_capture_detection(self):
        capture_region =[560,0,2000,1440]
        #[560,0,2000,1440] center of screen
        #[768, 180, 1792, 1260]
        cwd = os.getcwd()
        
        model = YOLO(os.path.join(cwd,"runs//train//train_run//weights//best.engine"))

        if self.debug:
            window_width, window_height = capture_region[2] - capture_region[0],capture_region[3]-capture_region[1]
            cv2.namedWindow("Screen Capture Detection", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Screen Capture Detection", window_width, window_height)
        
        camera = dxcam.create(region = capture_region, output_color='BGR',max_buffer_len=2)
        # camera.start(target_fps = 120, video_mode= True)
        if camera is None:
            print("Camera initialization failed.")
            return

        self.results = model.predict(source=np.zeros((2560,1440,3),dtype = np.uint8) ,conf=0.6, imgsz=1440)#preallocate the thingymabob?
        while True:
            
            start = time.time_ns()
            frame = camera.grab()
            
            if frame is None:
                print("frame none")
                time.sleep(.01)
                continue

            self.results = model.predict(source=frame, conf=0.6, imgsz=(1440,1440)) 

 
            self.detections.clear()#preallocated, uses same memory i think?
            for box in self.results[0].boxes:
                self.detections.append(
                    {
                    "class_name": model.names[int(box.cls[0])],
                    "bbox": list(map(int, box.xyxy[0])),
                    "confidence":float(box.conf[0])
                    }
                )

            if(self.is_key_pressed and self.detections):

                self.aimbot()


            
            

            # Draw bounding boxes and labels
            if self.debug:
                for obj in self.detections:
                    x1, y1, x2, y2 = obj['bbox']
                    label = f"{obj['class_name']} {obj['confidence']:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness = 1)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness = 1)

                # Show the output frame
                
                cv2.imshow("Screen Capture Detection", frame)
                cv2.waitKey(1)
            end = time.time_ns()
            runtime_ms = (end - start)/1000000

            self.append_time(runtime_ms)

    def append_time(self, time):
        
        if len(self.frame_times) >= self.time_interval_frames:
            self.frame_times.pop(0)

        self.frame_times.append(time)
    
    def calculate_fps(self):
        while True:
            if len(self.frame_times) > 0:
                avg_time_ms = sum(self.frame_times) / len(self.frame_times)  # Average time for the last N frames
                fps = 1000 / avg_time_ms  # FPS calculation based on average frame time (ms -> fps)
                print(f"FPS: {fps:.2f} (based on last {len(self.frame_times)} frames)")
            time.sleep(1)  # Adjust the reporting frequency as needed
        
            

if __name__ == "__main__":
    Main().main()