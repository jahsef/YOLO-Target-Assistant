from ultralytics import YOLO
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
github_dir = Path(__file__).parent.parent
sys.path.insert(0, str(os.path.join(github_dir,'BettererCam')))
#can replace with bettercam just no cupy support
import betterercam
print(betterercam.__file__)

class Main:
     
    def main(self):     
        self.debug = False
        self.screen_x = 2560
        self.screen_y = 1440
        self.h_w_capture = (896,1440)#height,width
        self.head_toggle = True
        self.is_key_pressed = False
        self.results = []
        self.detections = []
        self.screen_center_x = self.screen_x // 2
        self.screen_center_y = self.screen_y // 2
        self.x_offset = (self.screen_x - self.h_w_capture[1])//2
        self.y_offset = (self.screen_y - self.h_w_capture[0])//2#reversed because h_w
        threading.Thread(target=self.input_detection, daemon=True).start()
        # self.y_bottom_deadzone = 200
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
        
        # Track best candidate
        best_enemy = (None, -1)  # (center, score)
        prev_center = getattr(self, 'prev_center', None)
        prev_class = getattr(self, 'prev_class', None)

        for detection in self.detections:
            if detection['class_name'] != 'Enemy':  # Only process Enemy class
                continue

            # Calculate bounding box properties
            x1, y1, x2, y2 = detection['bbox']
            # if y1 >= self.screen_y - self.y_bottom_deadzone:
            #     continue
            width, height = x2 - x1, y2 - y1
            area = width * height
            
            # Calculate center coordinates
            center = ((x1 + x2) / 2 + self.x_offset, 
                      (y1 + y2) / 2 + self.y_offset)
            
            # Calculate proximity score
            dx = center[0] - screen_center[0]
            dy = center[1] - screen_center[1]
            dist_sq = dx * dx + dy * dy + 1e-6  # Avoid division by zero
            score = (area ** 2) / (dist_sq ** 0.75)

            # Apply hysteresis to previous target
            if prev_class == 'Enemy' and prev_center:
                dx_prev = center[0] - prev_center[0]
                dy_prev = center[1] - prev_center[1]
                if (dx_prev * dx_prev + dy_prev * dy_prev) <= PROXIMITY_THRESHOLD_SQ:
                    score *= HYSTERESIS_FACTOR

            # Update best enemy candidate
            if score > best_enemy[1]:
                best_enemy = (center, score)

        # Select target
        new_target = best_enemy[0] if best_enemy[0] else None

        # Update previous target tracking
        if new_target:
            self.prev_center = new_target
            self.prev_class = 'Enemy'
            if self.head_toggle:
                 #aims 25% above center
                height = y2 - y1

                if height > 95:  # Big target
                    offset_percentage = 0.35
                elif height > 35:  # Medium target
                    offset_percentage = 0.25
                else:  # Small target
                    offset_percentage = 0.15

                offset = int(height * offset_percentage)
                
                return (new_target[0], new_target[1] - offset)
        
        # Fallback to previous target or screen centerew 

        return new_target or screen_center

    def input_detection(self):
        def on_key_press(event):
            if event.name.lower() == 'e':
                self.is_key_pressed = not self.is_key_pressed
                print(f"Key toggled: {self.is_key_pressed}")
        keyboard.on_press(on_key_press)
        while True:
            time.sleep(1)
            

    
    #only like a 5% speedup using cupy, need to look into that some more
    #also python doesnt have method overloading by parameter type
    def preprocess(self,frame: cp.ndarray) -> torch.Tensor:
        bchw = cp.ascontiguousarray(frame.transpose(2, 0, 1)[cp.newaxis, ...])
        float_frame = bchw.astype(cp.float16, copy=False)/255.0
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
        
        # model  = YOLO(os.path.join(cwd,"runs/train/EFPS_3000image_realtrain_1440x1440_100epoch_batch6_11s/weights/best.pt"))
        model  = YOLO(os.path.join(cwd,"runs/train/EFPS_3000image_realtrain_1440x1440_100epoch_batch6_11s/weights/best.engine"))
        # model = YOLO(os.path.join(cwd,"runs/train/EFPS_3000image_640x896_tensorrt/best.engine"))
        # model = YOLO(os.path.join(cwd,'runs/train/EFPS_3000image_1440p_200epoch_batch3_11m/weights/best.engine'))
  
        
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

            self.results = model(source=gpu_frame,
                conf = .6,
                imgsz=self.h_w_capture,
                verbose = False
            )
            for box in self.results[0].boxes:
                detection = {
                    "class_name": model.names[int(box.cls[0])],
                    "bbox": list(map(int, box.xyxy[0])),
                    "confidence": float(box.conf[0])
                }
                self.detections.append(detection)
            
            if self.debug:
                display_frame = cp.asnumpy(frame)
                for obj in self.detections:
                    x1, y1, x2, y2 = obj['bbox']
                    label = f"{obj['class_name']} {obj['confidence']:.2f}"
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), thickness = 1)
                    cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness = 1)
                
                cv2.imshow("Screen Capture Detection", display_frame)
                cv2.waitKey(1)
                
            if(self.is_key_pressed and self.detections):
                self.aimbot()
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