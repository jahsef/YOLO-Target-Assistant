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



# model = ultralytics.YOLO(os.path.join(os.getcwd(),"runs/train/EFPS_4000img_11s_1440p_batch6_epoch200/weights/best.engine"))
cwd = os.getcwd()
base_dir = "runs/train/EFPS_4000img_11n_1440p_batch11_epoch100/weights"
engine_name = "320x320.engine"
imgsz = (320,320)
model = ultralytics.YOLO(os.path.join(cwd,base_dir,engine_name))

img_path = os.path.join(cwd, 'train/datasets/EFPS_4000img/images/train/frame_1012(1013).jpg')
img = cv2.imread(img_path)
img = cv2.resize(img, imgsz[::-1])  # Resize to model input size
cp_img = cp.ascontiguousarray(cp.asarray(img))

def preprocess(frame: cp.ndarray) -> torch.Tensor:
    bchw = cp.ascontiguousarray(frame.transpose(2, 0, 1)[cp.newaxis, ...])
    float_frame = bchw.astype(cp.float16, copy=False)
    float_frame *= 1.0 / 255.0  # Faster than division
    return torch.as_tensor(float_frame, device='cuda')
img_tensor = preprocess(cp_img)


img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.transpose(2, 0, 1) # HWC to CHW
img = np.expand_dims(img, axis=0).astype(np.float32)#add batch dim
img /= 255.0
processed_np_img = np.ascontiguousarray(img)
print(type(processed_np_img))
print(processed_np_img.shape)
for _ in range(32):
    with torch.no_grad():
        _ = model(img_tensor,imgsz = imgsz, verbose = True)

print('Starting FPS test')
@torch.inference_mode()
def inference(frame,imgsz):
    return model(
        source = frame,
        conf = .6,
        verbose = False,
        imgsz = imgsz
    )
poo_start = time.perf_counter()
for i in range(16):
    print(f'\niteration: {i}')
    start = time.perf_counter()
    for _ in range(1000):
        inference(frame = img_tensor,imgsz= imgsz)#preprocess(cp_img)

    inference_time = time.perf_counter() - start

    print(f"Inference time: {inference_time:.4f} sec")
    print(f"FPS: {(1000)/inference_time:.2f}")

avg = time.perf_counter() - poo_start

print(f" time: {avg:.4f} sec")
print(f"avg FPS: {(16*1000)/avg:.2f}")