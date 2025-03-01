import dxcam
import cv2
import os
import torch
from multiprocessing import Process, Event, shared_memory
import threading
import logging
from ultralytics import YOLO
import keyboard
import time
import numpy as np

logging.getLogger('ultralytics').setLevel(logging.ERROR)

class Threaded:
    def __init__(self):
        self.debug = True
        self.screen_x = 2560
        self.screen_y = 1440
        self.capture_dim = (1440,1440)
        self.screen_center_x = self.screen_x // 2
        self.screen_center_y = self.screen_y // 2
        self.x_offset = (self.screen_x - self.capture_dim[0])//2
        self.y_offset = (self.screen_y - self.capture_dim[1])//2
        self.is_key_pressed = False
        self.shape = list(self.capture_dim)
        self.shape.append(3)#3 channels for hcw
        self.frame_times = []
        self.time_interval_frames = 60
        self.results = []
        self.detections = []
        

    def main(self):
        
        screenshot_ready = Event()
        inference_ready = Event()
        inference_ready.set()


        shm = shared_memory.SharedMemory(create=True, size=int(np.prod(self.shape)))
        
        sc_process = Process(
            target=self.screen_cap,
            args=(screenshot_ready, inference_ready, shm.name,),
            daemon=True
        )
        inference_process = Process(
            target=self.inference,
            args=(screenshot_ready, inference_ready, shm.name,),
            daemon=True
        )
        input_detection_thread = threading.Thread(target=self.input_detection, daemon=True)
        
        input_detection_thread.start()
        sc_process.start()
        inference_process.start() 
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print('keyboard interrupt')
            sc_process.terminate()
            inference_process.terminate()
            input_detection_thread.join()
        
        shm.close()
        shm.unlink()

    def screen_cap(self, screenshot_ready, inference_ready, shm_name):
        capture_region = (0 + self.x_offset, 0 + self.y_offset, self.screen_x - self.x_offset, self.screen_y - self.y_offset)
        camera = dxcam.create(region = capture_region,output_color="BGR", max_buffer_len=2)
        # camera.start(target_fps=160)
        existing_shm = shared_memory.SharedMemory(name=shm_name)
        shared_array = np.ndarray(self.shape, dtype=np.uint8, buffer=existing_shm.buf)

        while True:
            inference_ready.wait()
            frame = camera.grab()
            if frame is not None:
                np.copyto(shared_array, frame)
                screenshot_ready.set()
                inference_ready.clear()

    def inference(self, screenshot_ready, inference_ready, shm_name):
        existing_shm = shared_memory.SharedMemory(name=shm_name)
        shared_array = np.ndarray(self.shape, dtype=np.uint8, buffer=existing_shm.buf)
        
        device = torch.device('cuda')
        model = YOLO(os.path.join(os.getcwd(), "runs/detect/tune4/weights/best.engine"))
        if self.debug:
            window_width, window_height = self.capture_dim
            cv2.namedWindow("Screen Capture Detection", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Screen Capture Detection", window_width, window_height)

        while True:
            screenshot_ready.wait()
            
            # Copy from shared memory and convert to tensor
            frame = shared_array.copy()
            with torch.no_grad():
                # next_tensor = prefetch_next_frame_async()
                # compute_current_frame()
                # postprocess_previous_frame()
                tensor = (
                    torch.from_numpy(frame)
                    .to(torch.device("cuda"))
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .half()
                    .div(255.0)
                    .contiguous()
                )
                self.results = model(tensor, imgsz = (1440,1440),conf = .5, max_det = 16)

                
            
            # Process results heree
            # print(f'Detected {len(results[0])} objects')
            
            screenshot_ready.clear()
            inference_ready.set()
            
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
            self.detections.clear()
            if self.is_key_pressed:
                print('key pressed')

    def input_detection(self):
        def on_key_press(event):
            if event.name == 'e':
                self.is_key_pressed = not self.is_key_pressed
                print(f"Key toggled: {self.is_key_pressed}")
        keyboard.on_press(on_key_press)
        while True:
            time.sleep(2)

if __name__ == '__main__':
    Threaded().main()