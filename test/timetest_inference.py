from ultralytics import YOLO
import os
import time
import logging
import cv2
import torch
import numpy as np
import cupy as cp
import sys
import ultralytics
from pathlib import Path
import torch.jit as jit

github_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(os.path.join(github_dir,'BettererCam')))

import betterercam
# print(betterercam.__file__)

torch.cuda.empty_cache()

cwd = os.getcwd()
img_path = os.path.join(cwd, 'train/datasets/EFPS_4000img/images/train/frame_0(1)_1.jpg')

img = cv2.imread(img_path)
img = cv2.resize(img, (1440, 896))  # Resize to model input size
cp_img = cp.ascontiguousarray(cp.asarray(img))

def preprocess(frame: cp.ndarray) -> torch.Tensor:
    bchw = cp.ascontiguousarray(frame.transpose(2, 0, 1)[cp.newaxis, ...])
    float_frame = bchw.astype(cp.float16, copy=False)
    float_frame *= 1.0 / 255.0  # Faster than division
    return torch.as_tensor(float_frame, device='cuda')

# model = ultralytics.YOLO(os.path.join(os.getcwd(),"runs/train/EFPS_4000img_11s_1440p_batch6_epoch200/weights/best.engine"))
model = ultralytics.YOLO(os.path.join(os.getcwd(),"runs/train/EFPS_4000img_11n_1440p_batch11_epoch100/weights/best.engine"))
img_tensor = preprocess(cp_img)
for _ in range(16):
    with torch.no_grad():
        _ = model(img_tensor,imgsz = (896,1440), verbose = False)

print('Starting FPS test')




@torch.inference_mode()

def inference(frame,imgsz):
    return model.predict(
        source = frame,
        conf = .6,
        verbose = False,
        imgsz = imgsz
    )
poo_start = time.perf_counter()
for i in range(64):
    print(f'\niteration: {i}')
    start = time.perf_counter()
    for _ in range(1000):
        inference(frame = img_tensor,imgsz= (896,1440))#preprocess(cp_img)

    inference_time = time.perf_counter() - start

    print(f"Inference time: {inference_time:.4f} sec")
    print(f"FPS: {(1000)/inference_time:.2f}")

avg = time.perf_counter() - poo_start

print(f" time: {avg:.4f} sec")
print(f"avg FPS: {(64*1000)/avg:.2f}")