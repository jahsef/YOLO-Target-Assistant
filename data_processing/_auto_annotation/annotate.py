from ultralytics import YOLO
import os
import torch
import cv2
import shutil
import numpy as np
import logging
import cupy as cp


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

cwd = os.getcwd()
images_path = os.path.join(cwd, 'data_processing/_auto_annotation/data/images')
labels_path = os.path.join(cwd, 'data_processing/_auto_annotation/data/labels/train')

# Ensure the labels directory exists
os.makedirs(labels_path, exist_ok=True)

img_dim = (640, 640)  # Image dimensions (width, height)

def clear_directory(directory):
    """Clears all files and subdirectories in the given directory."""
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            logging.error(f"Error deleting {file_path}: {e}")


# Clear the labels directory before generating new annotations
clear_directory(labels_path)

# Load the trained model
model = YOLO(os.path.join(os.getcwd(),"models/pf_300img_11n/weights/best.pt"))
def preprocess(frame: cp.ndarray) -> torch.Tensor:
    bchw = cp.ascontiguousarray(frame.transpose(2, 0, 1)[cp.newaxis, ...])
    float_frame = bchw.astype(cp.float16, copy=False)/255.0
    return torch.as_tensor(float_frame, device='cuda')

def inference(img_path):
    """Performs inference on the given image."""
    try:
        img = cv2.imread(img_path)
        if img is None:
            logging.warning(f"Failed to read image: {img_path}")
            return None
        

        # print('here')
        
        img = preprocess(cp.asarray(img))
        return model.predict(device = 0, source = img, imgsz=(img_dim[1], img_dim[0]), conf=0.4, verbose=False)
    except Exception as e:
        logging.error(f"Error during inference for {img_path}: {e}")
        # print('bomb')
        return None


def write_annotations(results, img_path):
    """Writes annotations for detected objects or deletes background images."""
    boxes = results[0].boxes if results else []

    # Handle background images
    # if len(boxes) == 0:  # No detections
    #     if np.random.rand() > 0.2:  # Keep 15% of background images
    #         os.remove(img_path)
    #         logging.info(f"Deleted background image: {img_path}")
    #     return
    # Determine if the image is friendly-only
    # only_friendly = True
    # for box in boxes:
    #     class_name = model.names[int(box.cls[0])]
    #     if class_name == 'Enemy':
    #         only_friendly = False
    #         break  # Exit early if an "Enemy" is found
    # Handle friendly-only images
    # if only_friendly:
    #     if np.random.rand() > 0.08:  # Keep 8% of friendly-only images
    #         os.remove(img_path)
    #         logging.info(f"Deleted friendly-only image: {img_path}")
    #         return

    # if any(box.cls[0] == 'Enemy' for box in boxes):  #removes only high confidence enemy detections
    #     if all(box.conf[0] > 0.85 for box in boxes if box.cls[0] == 'Enemy'):
    #         os.remove(img_path)
    #         logging.info(f"Removed high-confidence enemy image: {img_path}")
    #         return
        
    # if np.random.rand() > 0.4:  #bomb 60% of the stuff anyway
    #     os.remove(img_path)
    #     logging.info(f"bombed image: {img_path}")
    #     return
    # Write annotations for detected objects
    
    for box in boxes:
        class_name = model.names[int(box.cls[0])]
        x1, y1, x2, y2 = box.xyxy[0]  # Get bounding box coordinates

        # Map class name to class ID
        class_id = -1
        if class_name == "Enemy":
            print('enemy detected')
            class_id = 0
        elif class_name == "Friendly":
            print('friendly detected')
            class_id = 1
        elif class_name == "Crosshair":
            print('crosshair detected')
            class_id = 2

        # Calculate normalized YOLO format values
        center_x = (x2 + x1) / 2
        center_y = (y2 + y1) / 2
        dim_x = x2 - x1
        dim_y = y2 - y1

        norm_center_x = round(float(center_x / img_dim[0]), 6)
        norm_center_y = round(float(center_y / img_dim[1]), 6)
        norm_dim_x = round(float(dim_x / img_dim[0]), 6)
        norm_dim_y = round(float(dim_y / img_dim[1]), 6)

        # Save the annotation to a .txt file in YOLO format
        label_file = os.path.join(labels_path, f"{os.path.splitext(os.path.basename(img_path))[0]}.txt")
        with open(label_file, "a") as f:
            f.write(f"{class_id} {norm_center_x} {norm_center_y} {norm_dim_x} {norm_dim_y}\n")

    logging.info(f"Annotations written for: {img_path}")


# Process all images in the images directory
list_images = os.listdir(images_path)
# print(len(list_images))
for image in list_images:
    # print('iteration')
    image_path = os.path.join(images_path, image)
    # print('inferencing!!!')
    results = inference(image_path)
    # print(type(results))
    if results:
        write_annotations(results, image_path)