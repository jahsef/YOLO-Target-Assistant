from ultralytics import YOLO
import os
import torch
import cv2
import shutil
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

cwd = os.getcwd()
images_path = os.path.join(cwd, 'train/auto_annotation/data/images')
labels_path = os.path.join(cwd, 'train/auto_annotation/data/labels/train')

# Ensure the labels directory exists
os.makedirs(labels_path, exist_ok=True)

img_dim = (1440, 896)  # Image dimensions (width, height)


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
model = YOLO(os.path.join(cwd, "runs/train/EFPS_1863transferfrom1400_1440x1440_10epoch_batch6_11s/weights/best.pt"))


def inference(img_path):
    """Performs inference on the given image."""
    try:
        img = cv2.imread(img_path)
        if img is None:
            logging.warning(f"Failed to read image: {img_path}")
            return None

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        img_tensor = (
            torch.from_numpy(img)
            .to(torch.device("cuda"))
            .permute(2, 0, 1)
            .unsqueeze(0)
            .half()
            .div(255.0)
            .contiguous()
        )
        return model.predict(img_tensor, imgsz=(896, 1440), conf=0.4, verbose=False)
    except Exception as e:
        logging.error(f"Error during inference for {img_path}: {e}")
        return None


def write_annotations(results, img_path):
    """Writes annotations for detected objects or deletes background images."""
    boxes = results[0].boxes if results else []

    # Handle background images
    if len(boxes) == 0:  # No detections
        if np.random.rand() > 0.15:  # Keep 15% of background images
            os.remove(img_path)
            logging.info(f"Deleted background image: {img_path}")
        return

    # Determine if the image is friendly-only
    only_friendly = True
    for box in boxes:
        class_name = model.names[int(box.cls[0])]
        if class_name == 'Enemy':
            only_friendly = False
            break  # Exit early if an "Enemy" is found

    # Handle friendly-only images
    if only_friendly:
        if np.random.rand() > 0.08:  # Keep 8% of friendly-only images
            os.remove(img_path)
            logging.info(f"Deleted friendly-only image: {img_path}")
            return

    # Handle high-confidence enemy images
    if any(box.cls[0] == 'Enemy' for box in boxes):  # Check if there are any "Enemy" detections
        if all(box.conf[0] > 0.75 for box in boxes if box.cls[0] == 'Enemy'):
            os.remove(img_path)
            logging.info(f"Removed high-confidence enemy image: {img_path}")
            return
    # Write annotations for detected objects
    for box in boxes:
        class_name = model.names[int(box.cls[0])]
        x1, y1, x2, y2 = box.xyxy[0]  # Get bounding box coordinates

        # Map class name to class ID
        class_id = -1
        if class_name == "Enemy":
            class_id = 0
        elif class_name == "Friendly":
            class_id = 1

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
for image in list_images:
    image_path = os.path.join(images_path, image)
    results = inference(image_path)
    if results:
        write_annotations(results, image_path)