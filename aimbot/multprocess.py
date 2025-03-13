import bettercam
import cv2
import os
import torch
from multiprocessing import Process, Event, shared_memory, Array, Lock, Value
import threading
import logging
from ultralytics import YOLO
import keyboard
import time
import numpy as np
from ctypes import Structure, c_int, c_float, c_bool, c_double
import win32api
import win32con



logging.getLogger('ultralytics').setLevel(logging.ERROR)

class Detection(Structure):
    _fields_ = [
        ('x1', c_int),
        ('y1', c_int),
        ('x2', c_int),
        ('y2', c_int),
        ('confidence', c_float),
        ('class_id', c_int)
    ]

class Threaded:
    def __init__(self):
        self.fps_debug = False
        self.cv_debug = False

        self.screen_x = 2560
        self.screen_y = 1440
        self.capture_dim = (896,1440)#hxw
        

        self.max_detections = 16
        self.screen_center_x = self.screen_x // 2
        self.screen_center_y = self.screen_y // 2
        self.x_offset = (self.screen_x - self.capture_dim[1])//2
        self.y_offset = (self.screen_y - self.capture_dim[0])//2
        
        self.is_key_pressed = Value(c_bool, False)
        self.key_lock = Lock()
        
        self.shape = list(self.capture_dim)
        self.shape.append(3)# for *3 in np.prod(np.ndarray)
        
        self.prev_center = None
        self.head_toggle = True

        # Single shared memory buffer
        # self.frame_shm = shared_memory.SharedMemory(
        #     create=True, 
        #     size=int(np.prod(self.shape))  # 1440*1440*3 = 6,220,800 bytes
        # )
        
        self.frame_buffer = [
            shared_memory.SharedMemory(
            create=True, 
            size=int(np.prod(self.shape))
            ),
            shared_memory.SharedMemory(
            create=True, 
            size=int(np.prod(self.shape))
            )
            
        ]
        self.frame_buffer_idx = Value(c_int,0)
        
        self.frame_shm_lock = Lock()


        # Detection shared memory
        self.detections_shm = Array(Detection, self.max_detections, lock=False)
        self.detection_lock = Lock()
        
        self.is_sc_ready = Event()
        self.is_inference_ready = Event()
        self.is_detections_ready = Event()
        self.is_inference_ready.set()
        self.detection_count = Value(c_int,0)
        
        # import psutil
        # p = psutil.Process()
        # p.cpu_affinity([0, 1, 2])  # Dedicate cores
        # p.nice(psutil.REALTIME_PRIORITY_CLASS)
        # self.y_bottom_deadzone = 200
        # 1. Add atomic frame counter


    def main(self):
        processes = [
            Process(target=self.screen_cap, daemon=True),
            Process(target=self.inference, daemon=True),
            Process(target=self.aimbot_logic, daemon=True)
        ]




        threading.Thread(target=self.input_detection, daemon=True).start()

        for p in processes:
            p.start()


        try:
            while True: time.sleep(1)
        except KeyboardInterrupt:
            print('Shutting down')
            for p in processes: p.terminate()
            self.frame_shm.close()
            self.frame_shm.unlink()

    def screen_cap(self):
        camera = bettercam.create(
            region=(self.x_offset, self.y_offset, 
                   self.screen_x - self.x_offset, self.screen_y - self.y_offset),
            output_color="BGR",
            max_buffer_len=2
        )
        
        capture_frames = 0
        last_report = time.perf_counter()
        # self.is_inference_ready.set()
        write_num = 0
        
        while True:
            print(f"[CAP] WAITING {write_num} @ {time.perf_counter()}")
            self.is_inference_ready.wait(.006)
            frame = camera.grab()
            if frame is None: 
                continue
            print(f"[CAP] CAPTURED {write_num} @ {time.perf_counter()}")
            with self.frame_shm_lock:
                write_idx = self.frame_buffer_idx.value
                shared_arr = np.ndarray(self.shape, dtype=np.uint8, buffer=self.frame_buffer[write_idx].buf)
                np.copyto(shared_arr, frame)
                self.frame_buffer_idx.value = 1 - write_idx  # Toggle for next write
            self.is_sc_ready.set()
            print(f"[CAP] SENT {write_num} @ {time.perf_counter()}")
            write_num+=1
            
            self.is_inference_ready.clear()#clearing it so doesnt loop
            
            if self.fps_debug:
                capture_frames += 1
                if time.perf_counter() - last_report >= 1.0:
                    print(f"Capture FPS: {capture_frames}")
                    capture_frames = 0
                    last_report = time.perf_counter()
                
    def process_results(self, results, class_names):

        with self.detection_lock:
            self.detection_count.value = len(results.boxes)
            for i, box in enumerate(results.boxes):
                if i >= self.max_detections:
                    break
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                self.detections_shm[i] = Detection(
                    xyxy[0], xyxy[1], xyxy[2], xyxy[3],
                    float(box.conf[0]), int(box.cls[0])
                )
    @torch.inference_mode()
    def inference(self):
        
        model = YOLO(os.path.join(os.getcwd(),"runs/train/EFPS_4000img_11s_1440p_batch6_epoch200/weights/best.engine"))#dynamic engine size is POOP

        read_num = 0
        while True:
            print(f"[INF] WAITING {read_num} @ {time.perf_counter()}")
            self.is_sc_ready.wait()

                
            with self.frame_shm_lock:
                read_idx = 1 - self.frame_buffer_idx.value  # Read from last completed buffer
                

                curr_buffer = self.frame_buffer[read_idx].buf
                frame_buffer = np.ndarray(self.shape, dtype=np.uint8, buffer=curr_buffer)
            
            self.is_sc_ready.clear()
            print(f"[INF] GRABBED {read_num} @ {time.perf_counter()}")

            tensor = (
                torch.as_tensor(frame_buffer, dtype = torch.uint8)
                .to(device = 'cuda')#non blocking arg might cause weird latency flicking thing?
                .permute(2, 0, 1)
                .unsqueeze(0)
                .div(255)
                .contiguous()
            )
            torch.cuda.synchronize()  # Wait for GPU copy
            results = model(source=tensor,
                imgsz=self.capture_dim,
                conf = .6,
                max_det=self.max_detections
            )
            #clearing flag later will result in more up to date data me thinks but lower fps
            
            
            self.process_results(results[0], model.names)
            
            print(f"[INF] SEND RESULTS {read_num} @ {time.perf_counter()}")
            read_num+=1
            self.is_inference_ready.set() 
            self.is_detections_ready.set()


    def aimbot_logic(self):
        last_fps_update = time.perf_counter()
        frame_count = 0
        
        if self.cv_debug:
            cv2.namedWindow("Screen Capture Detection", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Screen Capture Detection", *self.capture_dim[::-1])
        
        while True:
            
            self.is_detections_ready.wait()

            with self.frame_shm_lock:
                read_idx = 1 - self.frame_buffer_idx.value  # Sync with inference
            
            if self.cv_debug:
                curr_buffer = self.frame_buffer[read_idx].buf
                frame = np.ndarray(self.shape, dtype=np.uint8, buffer=curr_buffer).copy()


                


            with self.detection_lock:
                num_detections = self.detection_count.value
                detections = [self.detections_shm[i] for i in range(num_detections)]
            # print('clearing')
            self.is_detections_ready.clear()#clear flag after accessing thingymabob
            with self.key_lock:
                key_state = self.is_key_pressed.value

            if key_state and detections:
                # print('2')
                target_detection = self.select_target_bounding_box(detections)
                self.move_mouse_to_bounding_box(target_detection)

            if self.cv_debug and 'frame' in locals():
                for d in detections:
                    #detections are stored differently in this script than main
                    label = f"{d.class_id} {d.confidence:.2f}"
                    cv2.rectangle(frame, (d.x1, d.y1), (d.x2, d.y2), (0,255,0), 1)
                    cv2.putText(frame, label, (d.x1, d.y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1)
                cv2.imshow("Screen Capture Detection", frame)
                cv2.waitKey(1)

            if self.fps_debug:
                frame_count+=1
                current_time = time.perf_counter()
                if current_time - last_fps_update >= 1:
                    fps = frame_count / (current_time - last_fps_update)
                    last_fps_update = current_time
                    frame_count = 0
                    print(f"Actual FPS: {fps:.1f}")
    
    def select_target_bounding_box(self,detection) -> tuple[int, int]:
        HYSTERESIS_FACTOR = 1.25
        PROXIMITY_THRESHOLD_SQ = 50**2
        screen_center = (self.screen_center_x, self.screen_center_y)
        
        # Track best candidate
        best_enemy = (None, -1)  # (center, score)
        prev_center = getattr(self, 'prev_center', None)
        prev_class = getattr(self, 'prev_class', None)

        for d in detection:
            if d.class_id != 0:  # Only process Enemy class
                continue

            # Calculate bounding box properties
            x1, y1, x2, y2 = d.x1, d.y1, d.x2, d.y2
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
            if prev_class == 0 and prev_center:
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
            self.prev_class = 0
            if self.head_toggle:
                 #aims 25% above center
                height = y2 - y1

                if height > 75:  # Big target
                    offset_percentage = 0.35
                elif height > 35:  # Medium target
                    offset_percentage = 0.25
                else:  # Small target
                    offset_percentage = 0.15

                offset = int(height * offset_percentage)
                
                return (new_target[0], new_target[1] - offset)
            
        # Fallback to previous target or screen center
        return new_target or screen_center
    
    def move_mouse_to_bounding_box(self, target_detection):
        center_bb_x = target_detection[0]
        center_bb_y = target_detection[1]
        delta_x = center_bb_x - self.screen_center_x
        delta_y = center_bb_y - self.screen_center_y
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(delta_x), int(delta_y), 0, 0)
        # win32api.SetCursorPos((int(center_bb_x), int(center_bb_y)))
        
    def input_detection(self):
        def on_key_press(event):
            if event.name == 'e':
                with self.key_lock:
                    self.is_key_pressed.value = not self.is_key_pressed.value
                print(f"Key toggled: {self.is_key_pressed.value}")
        keyboard.on_press(on_key_press)
        while True:
            time.sleep(1)

if __name__ == '__main__':
    Threaded().main()