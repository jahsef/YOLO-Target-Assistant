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
import tkinter
import TargetSelector as ts

# Suppress YOLO logs by adjusting the logging level
logging.getLogger('ultralytics').setLevel(logging.ERROR)

class Main:
     
    def main(self):     
        self.debug = False
        self.screen_x = 2560
        self.screen_y = 1440
        self.capture_dim = (1024,1024)
        
        self.is_key_pressed = False
        self.frame_times = []
        self.time_interval_frames = 60
        self.results = []
        self.detections = []

        self.screen_center_x = self.screen_x // 2
        # print(self.screen_center_x)
        self.screen_center_y = self.screen_y // 2
        # print(self.screen_center_y)
        self.x_offset = (self.screen_x - self.capture_dim[0])//2
        # print(self.x_offset)
        self.y_offset = (self.screen_y - self.capture_dim[1])//2
        # print(self.y_offset)
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

    def select_target_bounding_box(self) -> tuple[int, int]:
        HYSTERESIS_FACTOR = 1.25
        PROXIMITY_THRESHOLD_SQ = 50**2
        screen_center = (self.screen_center_x, self.screen_center_y)
        
        # Track best candidates
        best_head = (None, -1)
        best_zombie = (None, -1)
        prev_center = getattr(self, 'prev_center', None)
        prev_class = getattr(self, 'prev_class', None)

        for detection in self.detections:
            # Calculate bounding box properties
            x1, y1, x2, y2 = detection['bbox']
            width, height = x2 - x1, y2 - y1
            area = width * height
            
            # Calculate center coordinates
            center = ((x1 + x2)/2 + self.x_offset, 
                    (y1 + y2)/2 + self.y_offset)
            
            # Calculate proximity score
            dx = center[0] - screen_center[0]
            dy = center[1] - screen_center[1]
            dist_sq = dx*dx + dy*dy + 1e-6
            score = (area ** 2) / (dist_sq ** 0.75)

            # Apply hysteresis to previous target
            if prev_class == detection['class_name'] and prev_center:
                dx_prev = center[0] - prev_center[0]
                dy_prev = center[1] - prev_center[1]
                if (dx_prev*dx_prev + dy_prev*dy_prev) <= PROXIMITY_THRESHOLD_SQ:
                    score *= HYSTERESIS_FACTOR

            # Update best candidates
            if detection['class_name'] == 'Head':
                if score > best_head[1]:
                    best_head = (center, score)
            else:
                if score > best_zombie[1]:
                    best_zombie = (center, score)

        # Select target with class priority
        new_target = None
        if best_head[0]:
            new_target = best_head[0]
        elif best_zombie[0]:
            new_target = best_zombie[0]

        # Update previous target tracking
        if new_target:
            self.prev_center = new_target
            self.prev_class = 'Head' if best_head[0] else 'Zombie'
        
        # Fallback to previous target or screen center
        return new_target or getattr(self, 'prev_center', None) or screen_center

    def input_detection(self):
        def on_key_press(event):
            if event.name == 'e':
                self.is_key_pressed = not self.is_key_pressed
                print(f"Key toggled: {self.is_key_pressed}")
        keyboard.on_press(on_key_press)
        while True:
            time.sleep(2)

    def run_screen_capture_detection(self):
        capture_region = (0 + self.x_offset, 0 + self.y_offset, self.screen_x - self.x_offset, self.screen_y - self.y_offset)

        cwd = os.getcwd()
        
        # model = YOLO(os.path.join(cwd,"runs//train//train_run//weights//best.pt"))
        # model = YOLO(os.path.join(cwd, "runs/detect/tune4/weights/best.engine"))
        model = YOLO(os.path.join(cwd, "runs/train/1024x1024_batch12/weights/best.engine"))
        
        
        if self.debug:
            window_width, window_height = self.capture_dim
            cv2.namedWindow("Screen Capture Detection", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Screen Capture Detection", window_width, window_height)
        
        camera = dxcam.create(region = capture_region, output_color='BGR',max_buffer_len=2)#yolo does bgr -> rgb conversion in model.predict automatically

        if camera is None:
            print("Camera initialization failed.")
            return
        frame_count = 0
        last_fps_update = time.perf_counter()
        while True:
            
            frame = camera.grab()
            
            if frame is None:
                # print("frame none")
                time.sleep(.01)
                continue

            
            # with torch.no_grad():
            img_tensor = (
                torch.from_numpy(frame)
                .to(torch.device("cuda"))
                .permute(2, 0, 1)
                .unsqueeze(0)
                .half()
                .div(255.0)
                .contiguous()
            )

            self.results = list(model(source=img_tensor,
                imgsz=self.capture_dim,
                stream=True,
                iou=0.5,
                device=0,
                half=True,
                max_det=16,
                agnostic_nms=False,
                augment=False,
                vid_stride=False,
                visualize=False,
                verbose=True,
                show_boxes=False,
                show_labels=False,
                show_conf=False,
                save=False,
                show=False))

            for box in self.results[0].boxes:
                detection = {
                    "class_name": model.names[int(box.cls[0])],
                    "bbox": list(map(int, box.xyxy[0])),
                    "confidence": float(box.conf[0])
                }
                self.detections.append(detection)
            
            if self.debug:
                # display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                for obj in self.detections:
                    x1, y1, x2, y2 = obj['bbox']
                    label = f"{obj['class_name']} {obj['confidence']:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness = 1)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness = 1)
                
                cv2.imshow("Screen Capture Detection", frame)
                cv2.waitKey(1)
                
            if(self.is_key_pressed and self.detections):
                self.aimbot()
            # Trim unused entries
            self.detections.clear()
            frame_count+=1
            current_time = time.perf_counter()
            if current_time - last_fps_update >= 1:
                fps = frame_count / (current_time - last_fps_update)
                last_fps_update = current_time
                frame_count = 0
                print(f'fps: {fps:.2f}')

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