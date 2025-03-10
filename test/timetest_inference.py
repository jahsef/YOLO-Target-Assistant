from ultralytics import YOLO
import os
import time
import logging
import cv2
import torch
import numpy as np
import cupy as cp
import sys
from torch.utils.dlpack import from_dlpack
from pathlib import Path
github_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(os.path.join(github_dir,'BettererCam')))

import betterercam
print(betterercam.__file__)


cwd = os.getcwd()
img_path = os.path.join(cwd, 'train/split_dataset/images/train/frame_1.jpg')

# model  = YOLO(os.path.join(os.getcwd(),"runs/train/EFPS_3000image_realtrain_1440x1440_100epoch_batch6_11s/weights/best.engine"))

img = cv2.imread(img_path)
np_img = cv2.resize(img, (1440, 896))  # Resize to model input size
cp_img = cp.ascontiguousarray(cp.asarray(img))

screen_x = 2560
screen_y = 1440
h_w_capture = (896,1440)#height,width
x_offset = (screen_x - h_w_capture[1])//2
y_offset = (screen_y - h_w_capture[0])//2#reversed because h_w
capture_region = (0 + x_offset, 0 + y_offset, screen_x - x_offset, screen_y - y_offset)
    

import cupy as cp
import torch

def preprocess(frame: cp.ndarray) -> torch.Tensor:
    bchw = cp.ascontiguousarray(frame.transpose(2, 0, 1)[cp.newaxis, ...])
    float_frame = bchw.astype(cp.float16, copy=False)
    float_frame *= 1.0 / 255.0  # Faster than division
    return torch.as_tensor(float_frame, device='cuda')

def preprocess(frame: np.ndarray):
    # Use in-place normalization to avoid copies
    return (
        torch.as_tensor(frame,device = 'cuda',dtype = torch.uint8)
        .permute(2, 0, 1)
        .unsqueeze_(0)  # In-place add batch dim
        .half()        
        .div_(255.0)    # In-place normalization
        .contiguous() #need for tensorrt
    )




camera = betterercam.create(region = capture_region, nvidia_gpu = True)




# img_tensor = preprocess(cp_img)
# for _ in range(16):
#     with torch.no_grad():
#         _ = model(img_tensor,imgsz = (896,1440), verbose = False)


print('Starting FPS test')

@torch.inference_mode()
def fart():
    for _ in range(640):
        
        # frame = camera.grab()
        # if frame is None:
        #     continue
        gpu_frame = preprocess(np_img)
        # results = model(source=gpu_frame,
        #     conf = .6,
        #     imgsz=h_w_capture,
        #     device=0,
        #     verbose = False
        # )

start = time.perf_counter()
for _ in range(10):
    fart()
inference_time = time.perf_counter() - start
# print(f"Model: {model_name}")
print(f"Inference time: {inference_time:.4f} sec")
print(f"FPS: {(640*10)/inference_time:.2f}")