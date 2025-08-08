from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
import yaml
from collections import deque
import time

TEST_SIZE = 0.2
RANDOM_STATE = 42
NEW_DIR_NAME = "pf_delta_test"

cwd = Path.cwd()
base_dir = 'datasets'
#makes a new split dir copying contents from a presplit dir
pre_split_dir = cwd / base_dir / 'pre_split_dataset'
new_dir_name = NEW_DIR_NAME
split_dir = cwd / base_dir / new_dir_name#make sure no important data is here lol 

pre_split_images_dir =     pre_split_dir / 'images' / 'train'
pre_split_labels_dir =     pre_split_dir / 'labels' / 'train' 
pre_split_data_yaml_path = pre_split_dir / 'data.yaml' #only creates updated yaml in split dir
#not sure when i would even have a yaml in presplit but yea

train_images_dir = split_dir / 'images/train'
# print(train_images_dir)
# time.sleep(10000)
val_images_dir =   split_dir / 'images/val'
train_labels_dir = split_dir / 'labels/train'
val_labels_dir =   split_dir / 'labels/val'
split_train_txt =  split_dir / 'train.txt'
split_val_txt =    split_dir / 'val.txt'



Path.mkdir(pre_split_images_dir, exist_ok=True, parents=False)
Path.mkdir(pre_split_labels_dir, exist_ok=True, parents=False)
#clear that ho then reconstruct file struct
if Path.exists(split_dir):
    shutil.rmtree(split_dir)
    
Path.mkdir(split_dir, exist_ok=True, parents=True)
Path.mkdir(train_images_dir, exist_ok=True, parents=True)
Path.mkdir(val_images_dir, exist_ok=True, parents=True)
Path.mkdir(train_labels_dir, exist_ok=True, parents=True)
Path.mkdir(val_labels_dir, exist_ok=True, parents=True)

file_extensions = ('.jpg', '.png', '.jpeg')
image_paths = [p for p in pre_split_images_dir.iterdir() 
               if p.suffix.lower() in file_extensions]
# print(image_paths)
# print(type(image_files[0]))
# time.sleep(10000)
train_img_paths, val_img_paths = train_test_split(image_paths, test_size=TEST_SIZE, random_state=RANDOM_STATE)

def copy_image_and_label(img_path:Path, target_img_dir, target_label_dir):
    shutil.copy2(img_path, target_img_dir / img_path.name)
    
    label_name = img_path.stem + '.txt'
    label_path = pre_split_labels_dir / label_name
    target_label_path = target_label_dir / label_name
    if Path.exists(label_path):
        shutil.copy2(label_path, target_label_path)
        # print(f'moving:')
        # print(f'src:{label_path}')
        # print(f'dst:{target_label_path}')

for path in train_img_paths:
    copy_image_and_label(path, train_images_dir, train_labels_dir)

for path in val_img_paths:
    copy_image_and_label(path, val_images_dir, val_labels_dir)

# Create text files with new filenames
with open(split_train_txt, 'w') as f:
    for path in train_img_paths:
        f.write(f"images/train/{path.name}\n")

with open(split_val_txt, 'w') as f:
    for path in val_img_paths:
        f.write(f"images/val/{path.name}\n")

with open(pre_split_data_yaml_path, 'r') as f:
    data_yaml = yaml.safe_load(f)

data_yaml['path'] = str(split_dir)#'.\\data\\' + new_dir_name
# data_yaml['flip_idx'] = [0]#num of keypoints, empty for 1 ig
data_yaml['train'] = "images\\train"
data_yaml['val'] = "images\\val"

with open(split_dir / 'data.yaml', 'w') as f:
    yaml.dump(data_yaml, f)

print("Split complete")