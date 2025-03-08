from ultralytics import YOLO
import os
import time
import logging
import cv2
import torch
import numpy as np

# logging.getLogger('ultralytics').setLevel(logging.ERROR)
# model_name = "best.engine"

cwd = os.getcwd()
img_path = os.path.join(cwd, 'train/split_dataset/images/train/frame_1.jpg')

# Initialize model
model  = YOLO(os.path.join(os.getcwd(),"runs/train/EFPS_3000image_realtrain_1440x1440_100epoch_batch6_11s/weights/best.engine"))
# model = YOLO(os.path.join(cwd,"runs/train/EFPS_3000image_640x896_tensorrt/best.engine"))

device = torch.device("cuda")

# Preprocess image
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (1440, 896))  # Resize to model input size
# print(f'type img: {type(img)} {img.dtype}')
# 2. Convert to tensor WITH PROPER FORMATTING
img_tensor = torch.from_numpy(img).to(device)
img_tensor = img_tensor.permute(2, 0, 1)        # HWC -> CHW
img_tensor = img_tensor.unsqueeze(0)            # Add batch dim -> BCHW
img_tensor = img_tensor.half() / 255.0          # Normalize AFTER casting
img_tensor = img_tensor.contiguous()            # Critical for TensorRT


# Warmup
for _ in range(16):
    with torch.no_grad():
        _ = model(img_tensor,device = device,imgsz = (896,1440), verbose = False)



# Preallocate a reusable tensor (adjust size to your model's input)
BATCH_SIZE = 1
CHANNELS = 3
HEIGHT, WIDTH = 896, 1440  # Your capture resolution

# Create a persistent tensor to reuse memory
reusable_tensor = torch.empty(
    (BATCH_SIZE, CHANNELS, HEIGHT, WIDTH),
    dtype=torch.uint8,
    device="cuda"
    
)

def preprocess(frame_gpu):
    # frame_gpu: (H, W, C) uint8 tensor on GPU
    global reusable_tensor
    
    # Convert to FP16 and normalize in one step
    # Reshape to (C, H, W) and add batch dimension
    return (
        frame_gpu.permute(2, 0, 1)  # HWC -> CHW
        .unsqueeze(0)                # Add batch dim
        .div(255.0)                # Normalize to [0, 1]
        .half()                     # Convert to FP16
        # .contiguous()               # Ensure memory layout
    )
# FPS test
print('Starting FPS test')
print(img.nbytes)


def preprocess(frame: np.ndarray):
    # Use in-place normalization to avoid copies
    return (
        torch.from_numpy(frame)
        .to("cuda", non_blocking=True)
        .permute(2, 0, 1)
        .unsqueeze_(0)  # In-place add batch dim
        .half()        # In-place FP16 conversion
        .div_(255.0)    # In-place normalization
    )
@torch.inference_mode()
def fart():
    for _ in range(1000):
        # np.copyto(cpu_tensor.numpy(),img)
        # gpu_frame = cpu_tensor.to(device="cuda", non_blocking=True)
        # gpu_frame = (
        #     gpu_frame.permute(2, 0, 1)
        #     .unsqueeze(0)
        #     .half()
        #     .div(255.0)
        # )
        gpu_frame = preprocess(img)
        results = model(source=gpu_frame,
            conf = .6,
            imgsz=(896,1440),
            device=0,
            verbose = False

        )
start = time.perf_counter()
fart()
inference_time = time.perf_counter() - start
# print(f"Model: {model_name}")
print(f"Inference time: {inference_time:.4f} sec")
print(f"FPS: {1000/inference_time:.2f}")