from ultralytics import YOLO
import os
import time
import logging
import cv2
import torch
import numpy as np

logging.getLogger('ultralytics').setLevel(logging.ERROR)
model_name = "best.engine"

cwd = os.getcwd()
img_path = os.path.join(cwd, 'train/split_dataset/images/val/frame_4.jpg')

# Initialize TensorRT engine
model = YOLO(os.path.join(cwd, "runs/train/train_run/weights/best.engine"))

# Preprocess image once and reuse numpy array
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (1440, 1440))
img_np = np.ascontiguousarray(img)  # Ensure memory continuity

# Create pinned memory buffer and precompute normalization
pinned_img = torch.from_numpy(img_np).pin_memory()
scale_factor = torch.tensor(1/255.0, dtype=torch.float16, device='cuda')

# Preallocate GPU memory for tensor operations
gpu_buffer = torch.empty((1, 3, 1440, 1440), 
                        dtype=torch.float16,
                        device='cuda',
                        pin_memory=False)

# Warmup with direct tensor operations
with torch.no_grad():
    for _ in range(10):
        # Efficient memory transfer pipeline
        gpu_buffer[0] = pinned_img.to('cuda', non_blocking=True) \
                              .permute(2, 0, 1) \
                              .to(torch.float16) \
                              .mul_(scale_factor)
        _ = model(gpu_buffer, conf=0.6, imgsz=1440, max_det=16)

# FPS test with precise timing
print('Starting FPS test')
torch.cuda.synchronize()
start = time.time()

# Time pure inference (no pre/post-processing)
with torch.no_grad():
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(1000):
        _ = model(gpu_buffer)
    torch.cuda.synchronize()

torch.cuda.synchronize()
inference_time = time.time() - start

print(f"Model: {model_name}")
print(f"Inference time: {inference_time:.4f} sec")
print(f"FPS: {1000/inference_time:.2f}")