import os
import shutil
from sklearn.model_selection import train_test_split
import yaml

# Define paths
cwd = os.getcwd()
print(cwd)
base_dir = os.path.join(cwd, 'train')
# pre_split_dir = os.path.join(base_dir, 'pre_split_dataset')
split_dir = os.path.join(base_dir, 'split_dataset')

# images_dir = os.path.join(pre_split_dir, 'images/train')
# labels_dir = os.path.join(pre_split_dir, 'labels/traincm')
# data_yaml_path = os.path.join(pre_split_dir, 'data.yml')
# train_txt_path = os.path.join(pre_split_dir, 'train.txt')

train_images_dir = os.path.join(split_dir, 'images/train')
val_images_dir = os.path.join(split_dir, 'images/val')
train_labels_dir = os.path.join(split_dir, 'labels/train')
val_labels_dir = os.path.join(split_dir, 'labels/val')
split_train_txt_path = os.path.join(split_dir, 'train.txt')
split_val_txt_path = os.path.join(split_dir, 'val.txt')

# # Create directories for split_dataset if they don't exist
os.makedirs(split_dir, exist_ok=True)
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

# 'C:\\Users\\kevin\\OneDrive\\Desktop\\YOLO11-Enfartment-PoopPS\\train\\pre_split_dataset\\data.yml'
# 'C:\Users\kevin\OneDrive\Desktop\YOLO11-Enfartment-PoopPS\train\pre_split_dataset\data.yaml'