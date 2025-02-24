from ultralytics import YOLO
import os
import time
import logging
import cv2
import dxcam
cwd = os.getcwd()
# logging.getLogger('ultralytics').setLevel(logging.ERROR)
model_name = "best.engine"
model = YOLO(os.path.join(cwd,"runs//train//train_run//weights//",model_name), task = 'detect')

img_path = os.path.join(cwd, 'train/split_dataset/images/val/frame_4.jpg')
img_size = 1440  # Set the desired input size

img = cv2.imread(img_path)
# # print(img.shape)
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert to RGB
img = cv2.resize(img, (img_size, img_size))  # Resize to match model input
# print(img.shape)

camera = camera = dxcam.create( output_color='BGR')
frame = camera.grab([560,0,2000,1440])


start = time.time()

for _ in range(1,60):
    print(f'iteration {_}: ')
    results = model(img, imgsz = (1440,1440))

poop = time.time() - start
print(f"model: {model_name}")
print(f"Inference time: {poop:.4f} sec")
print(f"rough fps: {60/poop:.2f}")