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
import cupy as cp





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
        self.fps_debug = True
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
        

        

        self.detections_shm = Array(Detection, self.max_detections, lock=False)
        self.detection_lock = Lock()

        self.is_detections_ready = Event()

        self.detection_count = Value(c_int,0)
        
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
        
        self.buffer_ready = [Value(c_bool, False), Value(c_bool, False)]
        self.current_write_idx = Value(c_int, 0)



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
            for buffer in self.frame_buffer: 
                buffer.close()
                buffer.unlink()
            

    def screen_cap(self):
        camera = bettercam.create(
            region=(self.x_offset, self.y_offset, 
                   self.screen_x - self.x_offset, self.screen_y - self.y_offset),
            output_color="BGR",
            max_buffer_len=2
        )
        
        capture_frames = 0
        last_report = time.perf_counter()
        write_num = 0
        
        while True:



            frame = camera.grab()
            if frame is None: 
                # time.sleep(.0001)
                continue
            
            write_idx = self.current_write_idx.value
            shared_arr = np.ndarray(self.shape, dtype=np.uint8, buffer=self.frame_buffer[write_idx].buf)
            np.copyto(shared_arr, frame)
            with self.buffer_ready[write_idx].get_lock():
                self.buffer_ready[write_idx].value = True  # Memory fence
                self.current_write_idx.value = 1 - write_idx  # Toggle AFTER write

            # print(f"[CAP] SENT {write_num} @ {time.perf_counter()}")
            write_num+=1
            
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
        def _process_frame(frame):
            tensor = torch.from_numpy(frame).to(device='cuda', non_blocking=True)
            tensor = tensor.permute(2, 0, 1).unsqueeze_(0).half().div_(255)
            return tensor.contiguous()
        
        # def _process_frame(frame: cp.ndarray) -> torch.Tensor:
        #     bchw = cp.ascontiguousarray(frame.transpose(2, 0, 1)[cp.newaxis, ...])
        #     float_frame = bchw.astype(cp.float16, copy=False)/255.0
        #     return torch.as_tensor(float_frame, device='cuda')

        model = YOLO(os.path.join(os.getcwd(),"runs/train/EFPS_4000img_11s_1440p_batch6_epoch200/weights/best.engine"))

        read_num = 0
        
        #if frame not ready yet wont explode
        tensor = np.zeros((896,1440,3),dtype = np.uint8)
        
        while True:
            
            # print(f"[INF] _TRY GRAB {read_num} @ {time.perf_counter()}")
            read_idx = 1 - self.current_write_idx.value  # Opposite of current write
            
            with self.buffer_ready[read_idx].get_lock():
                if self.buffer_ready[read_idx].value:
                    frame = np.ndarray(self.shape, dtype=np.uint8, buffer=self.frame_buffer[read_idx].buf)
                    tensor = _process_frame(frame)
                    self.buffer_ready[read_idx].value = False  # Reset flag
                    # print(f"[INF] SUCCESSFUL GRAB {read_num} @ {time.perf_counter()}")
            
            # print(f"[INF] INFERENCING {read_num} @ {time.perf_counter()}")
            results = model(source=tensor,
                imgsz=self.capture_dim,
                conf = .6,
                verbose = False
            )

            self.process_results(results[0], model.names)
            
            # print(f"[INF] DONE {read_num} @ {time.perf_counter()}")
            

            self.is_detections_ready.set()
            # print(f"[INF] SET {read_num} @ {time.perf_counter()}")
            read_num+=1


    def aimbot_logic(self):
        last_fps_update = time.perf_counter()
        frame_count = 0
        
        if self.cv_debug:
            cv2.namedWindow("Screen Capture Detection", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Screen Capture Detection", *self.capture_dim[::-1])
        
        while True:
            
            self.is_detections_ready.wait()

            read_idx = 1 - self.current_write_idx.value 
            
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