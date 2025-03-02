import dxcam
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
        self.fps_debug = True
        self.cv_debug = False

        
        self.screen_x = 2560
        self.screen_y = 1440
        self.capture_dim = (1024,1024)
        self.max_detections = 16
        self.screen_center_x = self.screen_x // 2
        self.screen_center_y = self.screen_y // 2
        self.x_offset = (self.screen_x - self.capture_dim[0])//2
        self.y_offset = (self.screen_y - self.capture_dim[1])//2
        
        self.is_key_pressed = Value(c_bool, False)
        self.key_lock = Lock()
        
        self.shape = list(self.capture_dim)
        self.shape.append(3)
        
        self.prev_center = None
        self.prev_class_id = None

        # Single shared memory buffer
        self.frame_shm = shared_memory.SharedMemory(
            create=True, 
            size=int(np.prod(self.shape))  # 1440*1440*3 = 6,220,800 bytes
        )
        
        self.frame_shm_lock = Lock()


        # Detection shared memory
        self.detections_shm = Array(Detection, self.max_detections, lock=False)
        self.detection_lock = Lock()
        
        self.is_sc_ready = Event()
        self.is_inference_ready = Event()
        self.is_inference_ready.set()

        self.detection_count = Value(c_int,0)
        self.detection_count_lock = Lock()

        

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
        camera = dxcam.create(
            region=(self.x_offset, self.y_offset, 
                   self.screen_x - self.x_offset, self.screen_y - self.y_offset),
            output_color="BGR",
            max_buffer_len=2
        )
        
        capture_frames = 0
        last_report = time.perf_counter()

        while True:
            
            self.is_inference_ready.wait(.001)
            
            frame = camera.grab()
            if frame is None: 
                time.sleep(.00005)
                continue
        
            with self.frame_shm_lock:
                np.ndarray(self.shape, dtype=np.uint8, buffer=self.frame_shm.buf)[:] = frame#loading frame into the buffer
                #seems like loading frame into buffer taking way too long?

            self.is_sc_ready.set()
            self.is_inference_ready.clear()#assuming its not gonna be ready 
            
            capture_frames += 1
            if time.perf_counter() - last_report >= 1.0:
                print(f"Capture FPS: {capture_frames}")
                capture_frames = 0
                last_report = time.perf_counter()
                
    def process_results(self, results, class_names):
        with self.detection_count_lock:
            self.detection_count.value = len(results.boxes)
        with self.detection_lock:
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
        #below is 1440 engine
        # model = YOLO(os.path.join(os.getcwd(), "runs/detect/tune4/weights/best.engine"))
        model = YOLO(os.path.join(os.getcwd(), "runs/train/1024x1024_batch12/weights/best.engine"))


        stream = torch.cuda.Stream()
        while True:
            
            self.is_sc_ready.wait()

            
            with self.frame_shm_lock:
                frame_buffer = np.ndarray(self.shape, dtype=np.uint8, buffer=self.frame_shm.buf)
            
            #would accessing the buffer "directly" cause weird memory issues??
            # frame_copy = np.copy(frame_buffer)
            
            #if clear flag early then the aimbot will have older data but higher fps
            self.is_sc_ready.clear()
            with torch.cuda.stream(stream):
                tensor = (
                    torch.as_tensor(frame_buffer, dtype = torch.uint8)
                    .to(device = 'cuda',non_blocking= True)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .div(255)
                    .contiguous()
                )
                
                results = list(model(source=tensor,
                    imgsz=self.capture_dim,
                    conf = .5,
                    stream=True,
                    iou=0.5,
                    device=0,
                    half=True,
                    max_det=self.max_detections,
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
            #clearing flag later will result in more up to date data me thinks but lower fps
            
            
            self.process_results(results[0], model.names)
            
            self.is_inference_ready.set() 


    def aimbot_logic(self):
        last_fps_update = time.perf_counter()
        frame_count = 0

        
        while True:
            
            self.is_inference_ready.wait()
            if self.cv_debug:
                frame = np.ndarray(self.shape, dtype=np.uint8, buffer=self.frame_shm.buf).copy()
            if self.cv_debug:
                cv2.namedWindow("Screen Capture Detection", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Screen Capture Detection", *self.capture_dim[:2])
                

            with self.detection_lock:
                with self.detection_count_lock:
                    num_detections = self.detection_count.value
                    detections = [self.detections_shm[i] for i in range(num_detections)]

            self.is_inference_ready.clear()#clear flag so it doesnt loop me thinks
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
            frame_count+=1
            
            current_time = time.perf_counter()
            if self.fps_debug and current_time - last_fps_update >= 1:
                fps = frame_count / (current_time - last_fps_update)
                last_fps_update = current_time
                frame_count = 0
                print(f"Actual FPS: {fps:.1f}")
    
    def select_target_bounding_box(self, detections) -> tuple[int, int]:
        #detections are passed in because they are parsed into c struct
        #i dont think instance var would be better maybe probably
        HYSTERESIS_FACTOR = 1.15
        PROXIMITY_THRESHOLD_SQ = 50**2
        screen_center = (self.screen_center_x, self.screen_center_y)
        
        # Track best candidates
        best_head = (None, -1)
        best_zombie = (None, -1)

        for d in detections:
            # Calculate bounding box properties
            #extract bounding box from struct
            x1, y1, x2, y2 = d.x1, d.y1, d.x2, d.y2
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
            if self.prev_class_id == d.class_id and self.prev_center:
                dx_prev = center[0] - self.prev_center[0]
                dy_prev = center[1] - self.prev_center[1]
                if (dx_prev*dx_prev + dy_prev*dy_prev) <= PROXIMITY_THRESHOLD_SQ:
                    score *= HYSTERESIS_FACTOR

            # Update best candidates
            if d.class_id == 3:#class_id 3 is head btw
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
            # self.prev_center = new_target
            self.prev_class_id = 3 if best_head[0] else 0#if head exists return head class_id if not return zombie class_id
        
        # Fallback to previous target or screen center
        
        return new_target or screen_center #prev_center omitted dont want to aim on old data
    def move_mouse_to_bounding_box(self, target_detection):
        center_bb_x = target_detection[0]
        center_bb_y = target_detection[1]
        delta_x = center_bb_x - self.screen_center_x
        delta_y = center_bb_y - self.screen_center_y
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(delta_x), int(delta_y), 0, 0)
        
    def input_detection(self):
        def on_key_press(event):
            if event.name == 'e':
                with self.key_lock:
                    self.is_key_pressed.value = not self.is_key_pressed.value
                print(f"Key toggled: {self.is_key_pressed.value}")
        keyboard.on_press(on_key_press)
        while True:
            time.sleep(2)

if __name__ == '__main__':
    Threaded().main()