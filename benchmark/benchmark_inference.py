import os
import time
import torch
import cupy as cp
import sys
from pathlib import Path

# Add project root to sys.path for absolute imports
github_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(Path.cwd()))

# Import the Model class from src/aimbot/engine
from src.aimbot.engine.model import Model

torch.cuda.empty_cache()

cwd = os.getcwd()
# Ensure these paths are correct after refactoring
model_path = Path(cwd) / "data/models/pf_1550img_11s/weights/320x320_stripped.engine" # Example model path
img_path = Path(cwd) / 'data/datasets/pf_1550img/images/train/frame5(0).png' # Example image path

# Load a sample image for inference
import cv2
img = cv2.imread(str(img_path))
if img is None:
    raise FileNotFoundError(f"Image not found at {img_path}")

# Initialize the Model
# The imgsz for the Model constructor should match the model's expected input size
# For .engine models, the imgsz is determined internally, but a placeholder can be provided if needed for the constructor.
# Let's assume the model expects 640x640 based on the filename.

hw_capture = (640, 640) # Height, Width
model_instance = Model(model_path=model_path, hw_capture=hw_capture)

# Determine target input size from the model
target_h, target_w = model_instance.hw_capture

# Get current image dimensions
H, W, _ = img.shape

# Calculate cropping coordinates to center the image
start_x = (W - target_w) // 2
start_y = (H - target_h) // 2

# Crop the image
cropped_img = img[start_y:start_y + target_h, start_x:start_x + target_w]

# Convert cropped image to CuPy array for processing by the Model's inference method
cp_img = cp.asarray(cropped_img)

print('Starting FPS test for inference...')

num_iterations = 10 # Number of times to run the inner loop
num_inferences_per_iteration = 1000 # Number of inferences in each inner loop

total_start_time = time.perf_counter()

for i in range(num_iterations):
    print(f'\nIteration: {i + 1}/{num_iterations}')
    iteration_start_time = time.perf_counter()

    for _ in range(num_inferences_per_iteration):
        _ = model_instance.inference(src=cp_img)

    iteration_inference_time = time.perf_counter() - iteration_start_time
    iteration_fps = num_inferences_per_iteration / iteration_inference_time

    print(f"Inference time for {num_inferences_per_iteration} inferences: {iteration_inference_time:.4f} sec")
    print(f"FPS for this iteration: {iteration_fps:.2f}")

total_inference_time = time.perf_counter() - total_start_time
average_fps = (num_iterations * num_inferences_per_iteration) / total_inference_time

print(f"\nTotal inference time for {num_iterations * num_inferences_per_iteration} inferences: {total_inference_time:.4f} sec")
print(f"Average FPS across all iterations: {average_fps:.2f}")
