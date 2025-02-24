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
        self.prev_target = None
        
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

    # def select_target_bounding_box(self) -> tuple[int,int]:
    #     head_detection_dict = {}# center(tuple) : list[dist(int NEED TO CONVERT),area(int)]
    #     zombie_detection_dict = {}

    #     for detection in self.detections:
    #         x1, y1, x2, y2 = detection['bbox']
    #         curr_center_coords = ((x2 + x1) / 2, (y2 + y1) / 2)
    #         curr_area = (x2-x1)*(y2-y1)
    #         curr_dist = ((curr_center_coords[0] - self.screen_center_x) ** 2 + (curr_center_coords[1] - self.screen_center_y) ** 2) ** 0.5
        
    #         if curr_dist == 0:
    #             score = float('inf')
    #         else:
    #             score = curr_area**1.5/curr_dist**.75

    #         if detection['class_name'] == 'Head':
    #             head_detection_dict[curr_center_coords] = score
    #         else:
    #             zombie_detection_dict[curr_center_coords] = score
    #     if len(head_detection_dict) > 0: 
    #         sorted_heads = sorted(head_detection_dict, key = head_detection_dict.__getitem__,reverse = True)
    #         return [sorted_heads[0][0] + 560,sorted_heads[0][1]]
    #     else:
    #         sorted_zombies = sorted(zombie_detection_dict, key = zombie_detection_dict.__getitem__,reverse = True)
    #         return [sorted_zombies[0][0] + 560,sorted_zombies[0][1]]
        
    # def select_target_bounding_box(self) -> tuple[int, int]:
    #     head_detection_dict = {}
    #     zombie_detection_dict = {}

    #     for detection in self.detections:
    #         x1, y1, x2, y2 = detection['bbox']
    #         center_x = ((x2 + x1) / 2) + 560  # Apply x-axis offset
    #         center_y = (y2 + y1) / 2
    #         curr_center_coords = (center_x, center_y)

    #         curr_area = (x2 - x1) * (y2 - y1)
    #         curr_dist = ((curr_center_coords[0] - self.screen_center_x) ** 2 + (curr_center_coords[1] - self.screen_center_y) ** 2) ** 0.5 + 1e-6  # Prevent div by zero

    #         score = curr_area **1.5/ curr_dist**.75  # More stable scoring function

    #         if detection['class_name'] == 'Head':
    #             head_detection_dict[curr_center_coords] = score
    #         else:
    #             zombie_detection_dict[curr_center_coords] = score

    #     def get_best_target(detection_dict):
    #         return max(detection_dict, key=detection_dict.get) if detection_dict else None

    #     best_head = get_best_target(head_detection_dict)
    #     best_zombie = get_best_target(zombie_detection_dict)

    #     new_target = best_head if best_head else best_zombie

    #     # Keep previous target unless new one is significantly better
    #     if self.prev_target and new_target:
    #         prev_score = head_detection_dict.get(self.prev_target, zombie_detection_dict.get(self.prev_target, 0))
    #         new_score = head_detection_dict.get(new_target, zombie_detection_dict.get(new_target, 0))

    #         if prev_score * 0.9 > new_score:  # Adds inertia
    #             return self.prev_target

    #     self.prev_target = new_target
    #     return new_target if new_target else (self.screen_center_x, self.screen_center_y)  # Default if no detection
    
    def select_target_bounding_box(self) -> tuple[int, int]:
        head_detection_dict = {}
        zombie_detection_dict = {}

        for detection in self.detections:
            x1, y1, x2, y2 = detection['bbox']
            center_x = ((x2 + x1) / 2) + 560  # Apply x-axis offset
            center_y = (y2 + y1) / 2
            curr_center_coords = (center_x, center_y)

            curr_area = (x2 - x1) * (y2 - y1)
            curr_dist = curr_dist = ((curr_center_coords[0] - self.screen_center_x) ** 2 + (curr_center_coords[1] - self.screen_center_y) ** 2) ** 0.5 + 1e-6  # Prevent div by zero
            # Calculate score with distance-based penalty (close targets get penalized more)
            score = (curr_area**1.5 / curr_dist**.75)
            if detection['class_name'] == 'Head':
                head_detection_dict[curr_center_coords] = score
            else:
                zombie_detection_dict[curr_center_coords] = score

        def get_best_target(detection_dict):
            return max(detection_dict.items(), key=lambda x: x[1], default=(None, 0))

        best_head, head_score = get_best_target(head_detection_dict)
        best_zombie, zombie_score = get_best_target(zombie_detection_dict)

        # Prioritize heads, then zombies
        new_target = best_head if best_head else best_zombie

        # Keep previous target only for 1 frame
        target = new_target if new_target else self.prev_target
        self.prev_target = new_target  

        return target if target else (self.screen_center_x, self.screen_center_y)  # Default if no detection
    
    def input_detection(self):
        def on_key_press(event):
            if event.name == 'e':
                self.is_key_pressed = not self.is_key_pressed
                print(f"Key toggled: {self.is_key_pressed}")
        keyboard.on_press(on_key_press)
        while True:
            time.sleep(2)

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