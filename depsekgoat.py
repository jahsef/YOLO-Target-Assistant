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
        self.is_key_pressed = False
        self.shape = list(self.capture_dim)
        self.shape.append(3)
        
        # Single shared memory buffer
        self.latest_frame = shared_memory.SharedMemory(
            create=True, 
            size=int(np.prod(self.shape))  # 1440*1440*3 = 6,220,800 bytes
        )
        self.frame_ready = Value(c_bool, False)
        self.frame_timestamp = Value(c_double, 0.0)
        self.detection_timestamp = Value(c_double, 0.0)  # Added missing timestamp

        # Detection shared memory
        self.detections_shm = Array(Detection, self.max_detections, lock=False)
        self.detection_count = Array(c_int, 1, lock=False)
        self.detection_lock = Lock()
        self.new_detection_event = Event()

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
            self.latest_frame.close()
            self.latest_frame.unlink()

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
            frame = camera.grab()
            if frame is None: 
                time.sleep(.001)
                continue
        

            buffer = np.ndarray(self.shape, dtype=np.uint8, buffer=self.latest_frame.buf)
            np.copyto(buffer, frame)
            self.frame_ready.value = True
            self.frame_timestamp.value = time.perf_counter()
            
            capture_frames += 1
            if time.perf_counter() - last_report >= 1.0:
                print(f"Capture FPS: {capture_frames}")
                capture_frames = 0
                last_report = time.perf_counter()
    def process_results(self, results, class_names):
        count = len(results.boxes)
        self.detection_count[0] = count
        
        with self.detection_lock:
            for i, box in enumerate(results.boxes):
                if i >= self.max_detections:
                    break
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                self.detections_shm[i] = Detection(
                    xyxy[0], xyxy[1], xyxy[2], xyxy[3],
                    float(box.conf[0]), int(box.cls[0])
                )
    def inference(self):
        # model = YOLO(os.path.join(os.getcwd(), "runs/detect/tune4/weights/best.engine"))
        model = YOLO(os.path.join(os.getcwd(), "runs/train/1024x1024_batch12/weights/best.engine"))
        stream = torch.cuda.Stream()
        last_processed_time = 0.0

        while True:
            if not self.frame_ready.value or self.frame_timestamp.value <= last_processed_time:
                time.sleep(0.0001)
                continue
            

            frame_buffer = np.ndarray(self.shape, dtype=np.uint8, buffer=self.latest_frame.buf)
            frame_copy = np.copy(frame_buffer)
            last_processed_time = self.frame_timestamp.value
            self.frame_ready.value = False

            with torch.cuda.stream(stream): 
                tensor = (
                    torch.as_tensor(frame_copy,device = 'cuda')
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .div(255)
                    .contiguous()
                )
                results = model(tensor, imgsz=self.capture_dim, conf=0.5, max_det=self.max_detections)
            
            stream.synchronize()
            
            self.process_results(results[0], model.names)
            self.new_detection_event.set()
            self.detection_timestamp.value = time.perf_counter()  # Update detection timestamp

    def aimbot_logic(self):
        last_fps_update = time.perf_counter()
        last_detection_time = 0.0
        frame_count = 0

        
        while True:
            self.new_detection_event.wait()
            if self.cv_debug:
                frame = np.ndarray(self.shape, dtype=np.uint8, buffer=self.latest_frame.buf).copy()
            if self.cv_debug:
                cv2.namedWindow("Screen Capture Detection", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Screen Capture Detection", *self.capture_dim[:2])



                
            with self.detection_timestamp.get_lock():
                if self.detection_timestamp.value > last_detection_time:
                    with self.detection_lock:
                        count = self.detection_count[0]
                        detections = [self.detections_shm[i] for i in range(count)]
                    last_detection_time = self.detection_timestamp.value
                    self.new_detection_event.clear()  # Reset flag
            
            if self.is_key_pressed and detections:
                execute_aimbot()
            # Visualization
            
            if self.cv_debug and 'frame' in locals():

                for d in detections:
                    label = f"{d.class_id} {d.confidence:.2f}"
                    cv2.rectangle(frame, (d.x1, d.y1), (d.x2, d.y2), (0,255,0), 1)
                    cv2.putText(frame, label, (d.x1, d.y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 1)
                
                cv2.imshow("Screen Capture Detection", frame)
                cv2.waitKey(1)

            frame_count+=1
            
            current_time = time.perf_counter()
            if current_time - last_fps_update >= 1:
                fps = frame_count / (current_time - last_fps_update)
                last_fps_update = current_time
                frame_count = 0
                if self.fps_debug:
                    print(f"Actual FPS: {fps:.1f}")


    def input_detection(self):
        keyboard.on_press(lambda e: 
            setattr(self, 'is_key_pressed', not self.is_key_pressed) if e.name == 'e' else None)
        while True:
            time.sleep(1)

if __name__ == '__main__':
    Threaded().main()