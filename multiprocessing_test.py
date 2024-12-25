from multiprocessing import Process, Queue
from ultralytics import YOLO
import dxcam
import time
import logging
import threading
import keyboard
import win32api
import win32con
import math
import cv2

logging.getLogger('ultralytics').setLevel(logging.ERROR)


class Poop:
    def __init__(self):
        self.detections_queue = Queue()  # Queue for detections
        self.results_queue = Queue()    # Queue for YOLO results
        self.model_path = r"C:\Users\kevin\Documents\GitHub\YOLO11-Final-Poop-2\runs\train\train_run\weights\best.pt"
        self.frame_times = []
        self.is_key_pressed = False
        threading.Thread(target=self.input_detection, daemon=True).start()
        
    def input_detection(self):
        def on_key_press(event):
            if event.name == 'e':
                self.is_key_pressed = not self.is_key_pressed
                print(f"Key toggled: {self.is_key_pressed}")

        keyboard.on_press(on_key_press)

        while True:
            time.sleep(3)
            
    def start_detection(self):
        window_width, window_height = 2560, 1440
        cv2.namedWindow("Screen Capture Detection", cv2.WINDOW_NORMAL)
        # Start the detection parsing process
        parser_process = Process(target=self.parse_detections, args=(self.model_path, self.results_queue, self.detections_queue))
        parser_process.start()
        
        camera = dxcam.create(output_color='BGR')  # Initialize camera in the main process
        model = YOLO(self.model_path)  # Initialize YOLO model in the main process

        while True:
            # print(f"Results Queue Size: {self.results_queue.qsize()}, Detections Queue Size: {self.detections_queue.qsize()}")

            # Frame capture and YOLO inference
            frame = camera.grab()
            if frame is None:
                time.sleep(0.05)
                continue
            
            start_time = time.time_ns()
            results = model.predict(source=frame, conf=0.6, imgsz=1440)
            
            # Send YOLO results to the queue
            boxes = results[0].boxes.data.cpu().numpy().tolist()
            self.results_queue.put(boxes)
            
            if self.is_key_pressed:
                target = self.select_target_bounding_box()
                if target[0] != -1:  
                    self.move_mouse_to_bounding_box(target)

            # Measure frame processing time
            end_time = time.time_ns()
            runtime_ms = (end_time - start_time) / 1e6
            self.append_time_fps_calc(runtime_ms)
            detections
            while not self.detections_queue.empty():
                detections = self.detections_queue.get()
            for obj in detections:
                x1, y1, x2, y2 = obj['bbox']
                label = f"{obj['class_name']} {obj['confidence']:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness = 1)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness = 1)

            # Show the output frame
            cv2.imshow("Screen Capture Detection", frame)
            
    def move_mouse_to_bounding_box(self, detection):
        center_bb_x = detection[0]
        center_bb_y = detection[1]
        def sigmoid_abs_scaled(input, beta=0.5, sensitivity = .9):
            input = input * sensitivity
            scaled_value = (abs(input) / (1 + math.exp(-beta * abs(input)))) * sensitivity
            return int(scaled_value)
        delta_x = center_bb_x - (2560 / 2)
        delta_y = center_bb_y - (1440 / 2)
        move_x = sigmoid_abs_scaled(delta_x) if delta_x > 0 else 0-sigmoid_abs_scaled(delta_x)
        move_y = sigmoid_abs_scaled(delta_y) if delta_y > 0 else 0-sigmoid_abs_scaled(delta_y)
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, move_x, move_y, 0, 0)
        # print(f'Mouse movement: {delta_x, delta_y}')
        

    def select_target_bounding_box(self) -> tuple[int, int]:
        head_detection_dict = {}
        zombie_detection_dict = {}
        
        while not self.detections_queue.empty():
            detections = self.detections_queue.get()  # Retrieve detections from the queue
            for detection in detections:
                x1, y1, x2, y2 = detection['bbox']
                curr_center_coords = ((x2 + x1) / 2, (y2 + y1) / 2)
                curr_area = (x2 - x1) * (y2 - y1)
                curr_dist = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
                
                score = curr_area**3 / curr_dist**4 if curr_dist != 0 else float('inf')

                if detection['class_name'] == 'Head':
                    head_detection_dict[curr_center_coords] = score
                else:
                    zombie_detection_dict[curr_center_coords] = score
                    
        if head_detection_dict:
            sorted_heads = sorted(head_detection_dict, key=head_detection_dict.get, reverse=True)
            return sorted_heads[0]
        elif zombie_detection_dict:
            sorted_zombies = sorted(zombie_detection_dict, key=zombie_detection_dict.get, reverse=True)
            return sorted_zombies[0]
        return -1, -1
        
    @staticmethod
    def parse_detections(model_path, results_queue, detections_queue):
        model = YOLO(model_path)  # Initialize YOLO model in the child process
        
        while True:
            if not results_queue.empty():
                local_results = results_queue.get()  # Retrieve YOLO results from the queue
                parsed_results = [
                    {
                        "class_name": model.names[int(box[5])],
                        "bbox": list(map(int, box[:4]))
                    }
                    for box in local_results
                ]
                # while not results_queue.empty():
                #     results_queue.get()
                # while not detections_queue.empty():
                #     detections_queue.get()
                detections_queue.put(parsed_results)

    def append_time_fps_calc(self, runtime_ms):
        if len(self.frame_times) >= 60:
            self.frame_times.pop(0)
        self.frame_times.append(runtime_ms)
        
        # Calculate FPS
        avg_time = sum(self.frame_times) / len(self.frame_times) if self.frame_times else 0
        if avg_time > 0:
            print(f"FPS: {1000 / avg_time:.2f}")


if __name__ == "__main__":
    Poop().start_detection()
