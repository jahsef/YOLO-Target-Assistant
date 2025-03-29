import os
import cv2
from multiprocessing import Process
import itertools
import numpy as np
import shutil
def clear_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path):
                # print(f"Deleting file: {file_path}")
                os.chmod(file_path, 0o777)  # Make writable
                os.remove(file_path)
            elif os.path.isdir(file_path):
                # print(f"Deleting directory: {file_path}")
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

def crop_labels(labels, crop_region, img_shape,file_name):
    y1_crop, y2_crop, x1_crop, x2_crop = crop_region
    h_img, w_img = img_shape
    cropped_labels = []
    for cls_id, rel_x, rel_y, rel_w, rel_h in labels:
        x_center = rel_x * w_img
        y_center = rel_y * h_img
        box_w = rel_w * w_img
        box_h = rel_h * h_img
        x1_box = x_center - box_w / 2
        y1_box = y_center - box_h / 2
        x2_box = x_center + box_w / 2
        y2_box = y_center + box_h / 2

        x1_box_clipped = max(x1_box, x1_crop)
        y1_box_clipped = max(y1_box, y1_crop)
        x2_box_clipped = min(x2_box, x2_crop)
        y2_box_clipped = min(y2_box, y2_crop)

        if x1_box_clipped >= x2_box_clipped or y1_box_clipped >= y2_box_clipped:
            continue 
        og_size_x,og_size_y = x2_box-x1_box , y2_box-y1_box
        new_size_x,new_size_y = x2_box_clipped-x1_box_clipped , y2_box_clipped-y1_box_clipped
        if new_size_x < .25*og_size_x or new_size_y < .25*og_size_y:
            print(file_name)
            print('new bounding box too small, <25%')
            continue
        x_center_new = (x1_box_clipped + x2_box_clipped) / 2
        y_center_new = (y1_box_clipped + y2_box_clipped) / 2
        box_w_new = x2_box_clipped - x1_box_clipped
        box_h_new = y2_box_clipped - y1_box_clipped

        w_crop = x2_crop - x1_crop
        h_crop = y2_crop - y1_crop
        rel_x_new = (x_center_new - x1_crop) / w_crop
        rel_y_new = (y_center_new - y1_crop) / h_crop
        rel_w_new = box_w_new / w_crop
        rel_h_new = box_h_new / h_crop

        cropped_labels.append((cls_id, rel_x_new, rel_y_new, rel_w_new, rel_h_new))

    return cropped_labels    
        
def crop(images_dir, cropped_dir,labels_dir, cropped_labels_dir, images_np_arr, crop_dim, process_idx):

    for img_name in images_np_arr:
        # Load the image
        img_path = os.path.join(images_dir, img_name)
        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        # Define the crop region (center crop)
        y1 = (h - crop_dim[0]) // 2
        x1 = (w - crop_dim[1]) // 2
        y2 = y1 + crop_dim[0]
        x2 = x1 + crop_dim[1]
        crop_region = (y1, y2, x1, x2)

        # Crop the image
        cropped_img = img[y1:y2, x1:x2]

        # Save the cropped image
        cropped_path = os.path.join(cropped_dir, img_name)
        cv2.imwrite(cropped_path, cropped_img)

        # Load and crop the labels
        
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(labels_dir, label_name)
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                labels = [tuple(map(float, line.strip().split())) for line in f]
            cropped_labels = crop_labels(labels, crop_region, (h, w), file_name= label_name)   
            if not cropped_labels:
                continue 

            cropped_label_path = os.path.join(cropped_labels_dir, label_name)
            with open(cropped_label_path, "w") as f:
                for cls_id, rel_x, rel_y, rel_w, rel_h in cropped_labels:
                    f.write(f"{int(cls_id)} {rel_x:.6f} {rel_y:.6f} {rel_w:.6f} {rel_h:.6f}\n")

    print(f"Process {process_idx} finished")

if __name__ == '__main__':
    import time
    cwd = os.getcwd()
    images_dir = os.path.join(cwd,'train/crop_data_and_labels/data/uncropped')
    cropped_dir = os.path.join(os.path.join(cwd,'train/crop_data_and_labels/data/cropped'))
    images_list = os.listdir(images_dir)
    images_np_arr = np.asarray(images_list)
    labels_path = os.path.join(cwd,'train/crop_data_and_labels/labels/uncropped')
    cropped_labels_path = os.path.join(cwd,'train/crop_data_and_labels/labels/cropped')
    n_processes = 16
    crop_dimensions = (640,640)
    os.makedirs(images_dir,exist_ok= True)
    os.makedirs(cropped_dir,exist_ok= True)
    os.makedirs(labels_path,exist_ok= True)
    os.makedirs(cropped_labels_path,exist_ok= True)
    
    clear_directory(cropped_dir)
    clear_directory(cropped_labels_path)
    
    for i in range(n_processes):
        curr_img_name_arr = images_np_arr[i::n_processes]
        Process(target = crop, args = (images_dir,cropped_dir,labels_path, cropped_labels_path, curr_img_name_arr,crop_dimensions,i,)).start()
    
    

