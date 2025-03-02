from ultralytics import YOLO
import os
import time
import logging
import cv2
import torch
from torch import cuda

logging.getLogger('ultralytics').setLevel(logging.ERROR)
# model_name = "best.engine"

cwd = os.getcwd()
img_path = os.path.join(cwd, 'train/split_dataset/images/val/frame_4.jpg')

# Initialize model
model = YOLO(os.path.join(cwd, "runs/train/train_run/weights/best.engine"))
# model = YOLO(os.path.join(cwd, "runs/detect/tune4/weights/best.engine"))
# model.fuse()
device = torch.device("cuda")

# Preprocess image
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (1440, 1440))  # Resize to model input size
# print(f'type img: {type(img)} {img.dtype}')
# 2. Convert to tensor WITH PROPER FORMATTING
img_tensor = torch.from_numpy(img).to(device)
img_tensor = img_tensor.permute(2, 0, 1)        # HWC -> CHW
img_tensor = img_tensor.unsqueeze(0)            # Add batch dim -> BCHW
img_tensor = img_tensor.half() / 255.0          # Normalize AFTER casting
img_tensor = img_tensor.contiguous()            # Critical for TensorRT


# Warmup
for _ in range(10):
    with torch.no_grad():
        _ = model(img_tensor,device = device,imgsz = (1440,1440))

# FPS test
print('Starting FPS test')

start = time.time()
@torch.inference_mode()
def fart():
    for _ in range(1000):

        
        img_tensor = (
            torch.from_numpy(img)
            .to(torch.device("cuda"))
            .permute(2, 0, 1)
            .unsqueeze(0)
            .half()
            .div(255.0)
            .contiguous()
        )

        
        results = model(source=img_tensor,
            imgsz=(1440,1440),
            stream=False,
            iou=0.5,
            device=0,
            half=True,
            max_det=16,
            agnostic_nms=False,
            augment=False,
            vid_stride=False,
            visualize=False,
            verbose=True,
            show_boxes=False,
            show_labels=False,
            show_conf=False,
            save=False,
            show=False)
fart()
inference_time = time.time() - start
# print(f"Model: {model_name}")
print(f"Inference time: {inference_time:.4f} sec")
print(f"FPS: {1000/inference_time:.2f}")