from ultralytics import YOLO
import os
import torch
import cv2
import shutil
import numpy as np
import random
import logging
import cupy as cp
import time
counters = {
    'total': 0,
    'background': 0,
    'friendly_only': 0,
    'high_conf': 0,
    'high_conf_crosshair': 0,
    'bombed': 0,
    'kept': 0
}

cwd = os.getcwd()
images_path = os.path.join(cwd, 'data_processing/_auto_annotation/data/images')
labels_path = os.path.join(cwd, 'data_processing/_auto_annotation/data/labels/train')
os.makedirs(labels_path, exist_ok=True)
img_dim = (640, 640)

def clear_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            logging.error(f"Error deleting {file_path}: {e}")

clear_directory(labels_path)

model = YOLO(os.path.join(os.getcwd(),"models/pf_1200img_11s/weights/best.pt"))
def preprocess(frame: cp.ndarray) -> torch.Tensor:
    bchw = cp.ascontiguousarray(frame.transpose(2, 0, 1)[cp.newaxis, ...])
    float_frame = bchw.astype(cp.float16, copy=False)/255.0
    return torch.as_tensor(float_frame, device='cuda')

def inference(img_path):
    try:
        img = cv2.imread(img_path)
        if img is None:
            logging.warning(f"Failed to read image: {img_path}")
            return None

        img = preprocess(cp.asarray(img))
        return model.predict(device = 0, source = img, imgsz=(img_dim[1], img_dim[0]), nms = True, verbose=False)
    except Exception as e:
        logging.error(f"Error during inference for {img_path}: {e}")
        # print('bomb')
        return None
    
    
def write_annotations(results, img_path):
    counters['total'] += 1
    boxes = results[0].boxes if results else []
    if len(boxes) == 0: 
        if random.random() > 0.75:
            os.remove(img_path)
            counters['background'] += 1
            return
        
    only_friendly = True
    for box in boxes:
        class_name = model.names[int(box.cls[0])]
        if class_name == 'Enemy':
            only_friendly = False
            break  
        
    if only_friendly:
        if random.random() > 0.6:
            os.remove(img_path)
            counters['friendly_only'] += 1
            return
        
    if all(box.conf[0] > 0.9 for box in boxes):#if all confidence is high
        os.remove(img_path)
        counters['high_conf'] += 1
        return
    
    #if crosshair is high (other ones can be low still)
    if any(box.cls[0] == 2 for box in boxes):
        if all(box.conf[0] > 0.9 for box in boxes if box.cls[0] == 2):
            os.remove(img_path)
            counters['high_conf_crosshair'] += 1
            return
        
    if random.random() > 1:  #bomb 70% anyway
        os.remove(img_path)
        counters['bombed'] += 1
        return
    counters['kept'] += 1
    #write annotations with normalized nums
    for box in boxes:
        class_name = model.names[int(box.cls[0])]
        x1, y1, x2, y2 = box.xyxy[0] 
        class_id = -1
        if class_name == "Enemy":
            class_id = 0
        elif class_name == "Friendly":
            class_id = 1
        elif class_name == "Crosshair":
            class_id = 2

        center_x = (x2 + x1) / 2
        center_y = (y2 + y1) / 2
        dim_x = x2 - x1
        dim_y = y2 - y1
        norm_center_x = round(float(center_x / img_dim[0]), 6)
        norm_center_y = round(float(center_y / img_dim[1]), 6)
        norm_dim_x = round(float(dim_x / img_dim[0]), 6)
        norm_dim_y = round(float(dim_y / img_dim[1]), 6)

        label_file = os.path.join(labels_path, f"{os.path.splitext(os.path.basename(img_path))[0]}.txt")
        with open(label_file, "a") as f:
            f.write(f"{class_id} {norm_center_x} {norm_center_y} {norm_dim_x} {norm_dim_y}\n")
    
    # logging.info(f"Annotations written for: {img_path}")

from tqdm import tqdm
# Process all images in the images directory
list_images = os.listdir(images_path)
# print(list_images)
progress_bar = tqdm(list_images, desc = 'progress', ncols=128)

for image in progress_bar:
    image_path = os.path.join(images_path, image)
    results = inference(image_path)
    # if results:
    write_annotations(results, image_path)

print(f"Image processing statistics:")
print(f"Total processed: {counters['total']}")
print(f"Background images: {counters['background']}")
print(f"Friendly-only images: {counters['friendly_only']}")
print(f"High confidence images: {counters['high_conf']}")
print(f"High confidence crosshair: {counters['high_conf_crosshair']}")
print(f"bombed: {counters['bombed']}")
print(f"Final kept images: {counters['kept']}")