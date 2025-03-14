import os
import cv2
import cupy as cp
import numpy as np
from ultralytics import YOLO
from ultralytics import trackers
import torch
import time
from multiprocessing import Process
import bettercam
from pathlib import Path
import sys

github_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(os.path.join(github_dir,'BettererCam')))
# can replace with bettercam just no cupy support
import betterercam

cwd = os.getcwd()
img_path = os.path.join(cwd, 'train/datasets/EFPS_4000img/images/train/frame_0(1)_1.jpg')


img = cv2.imread(img_path)
img = cv2.resize(img, (1440, 896))  # Resize to model input size


def process_frame(frame):
    return  (
        torch.as_tensor(frame, dtype = torch.uint8)
        .to(device = 'cuda')#non blocking arg might cause weird latency flicking thing?
        .permute(2, 0, 1)
        .unsqueeze(0)
        .div(255)
        .contiguous()
    )
tensor = process_frame(img)

class Fart:

    @torch.inference_mode()
    def inference(self):
        def _process_frame(frame):
            tensor = torch.from_numpy(frame).to(device='cuda', non_blocking=True)
            tensor = tensor.permute(2, 0, 1).unsqueeze_(0).half().div_(255)
            return tensor.contiguous()
        model = YOLO(os.path.join(os.getcwd(),"runs/train/EFPS_4000img_11s_1440p_batch6_epoch200/weights/best.engine"))


        while True:
            results = model(source=_process_frame(img),
                imgsz=(896,1440),
                conf = .6,
                verbose = True
            )
    def screen_cap(self):
        camera = betterercam.create(
            region=(self.x_offset, self.y_offset, 
                   self.screen_x - self.x_offset, self.screen_y - self.y_offset),
            output_color="BGR",
            max_buffer_len=2,
            nvidia_gpu = True
        )
        while True:
            camera.grab()
    def main(self):
        
        self.screen_x = 2560
        self.screen_y = 1440
        self.capture_dim = (896,1440)#hxw
        self.max_detections = 16
        self.screen_center_x = self.screen_x // 2
        self.screen_center_y = self.screen_y // 2
        self.x_offset = (self.screen_x - self.capture_dim[1])//2
        self.y_offset = (self.screen_y - self.capture_dim[0])//2
        
        p1 = Process(target = self.inference,args = (), daemon= True).start()
        p2 = Process(target = self.screen_cap,args = (), daemon= True).start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            p1.terminate()
            p2.terminate()
if __name__ == '__main__':
    Fart().main()