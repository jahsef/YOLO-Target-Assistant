import os
import shutil
from sklearn.model_selection import train_test_split
import yaml
from collections import deque

# Define paths
cwd = os.getcwd()
base_dir = os.path.join(cwd, 'train')
pre_split_dir = os.path.join(base_dir, 'pre_split_dataset')
split_dir = os.path.join(base_dir, 'split_dataset')

images_dir = os.path.join(pre_split_dir, 'images')
labels_dir = os.path.join(pre_split_dir, 'labels')
data_yaml_path = os.path.join(pre_split_dir, 'data.yaml')

train_images_dir = os.path.join(split_dir, 'images/train')
val_images_dir = os.path.join(split_dir, 'images/val')
train_labels_dir = os.path.join(split_dir, 'labels/train')
val_labels_dir = os.path.join(split_dir, 'labels/val')

split_train_txt = os.path.join(split_dir, 'train.txt')
split_val_txt = os.path.join(split_dir, 'val.txt')

def clear_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

clear_directory(split_dir)
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

def bfs_find_files(root_dir, extensions):
    files = []
    queue = deque([root_dir])
    while queue:
        current_dir = queue.popleft()
        try:
            for entry in os.listdir(current_dir):
                full_path = os.path.join(current_dir, entry)
                if os.path.isdir(full_path):
                    queue.append(full_path)
                elif os.path.isfile(full_path) and entry.lower().endswith(extensions):
                    files.append(full_path)
        except Exception as e:
            print(f"Error accessing {current_dir}: {e}")
    return files

image_files = bfs_find_files(images_dir, ('.jpg', '.png', '.jpeg'))

train_images, val_images = train_test_split(image_files, test_size=0.2, random_state=42)

def copy_image_and_label(img, target_img_dir, target_label_dir):
    base_name = os.path.basename(img)
    new_name = base_name
    counter = 1
    
    # Handle duplicate filenames for images
    while os.path.exists(os.path.join(target_img_dir, new_name)):
        name_without_ext, ext = os.path.splitext(base_name)
        new_name = f"{name_without_ext}({counter}){ext}"
        counter += 1
    
    # Copy image
    shutil.copy2(img, os.path.join(target_img_dir, new_name))
    
    # Determine original label path using the original image's relative path
    img_rel_path = os.path.relpath(img, images_dir)
    original_label_path = os.path.splitext(img_rel_path)[0] + '.txt'
    original_label_full_path = os.path.join(labels_dir, original_label_path)
    
    if os.path.exists(original_label_full_path):
        # Create new label name based on new image name
        new_label_name = os.path.splitext(new_name)[0] + '.txt'
        # Copy label to target directory
        shutil.copy2(original_label_full_path, os.path.join(target_label_dir, new_label_name))
    
    return new_name

# Collect new filenames
train_new_names = []
for img in train_images:
    new_name = copy_image_and_label(img, train_images_dir, train_labels_dir)
    train_new_names.append(new_name)

val_new_names = []
for img in val_images:
    new_name = copy_image_and_label(img, val_images_dir, val_labels_dir)
    val_new_names.append(new_name)

# Create text files with new filenames
with open(split_train_txt, 'w') as f:
    for name in train_new_names:
        f.write(f"images/train/{name}\n")

with open(split_val_txt, 'w') as f:
    for name in val_new_names:
        f.write(f"images/val/{name}\n")

# Update data.yaml
with open(data_yaml_path, 'r') as f:
    data_yaml = yaml.safe_load(f)

data_yaml['path'] = split_dir
data_yaml['train'] = os.path.join('images', 'train')
data_yaml['val'] = os.path.join('images', 'val')

with open(os.path.join(split_dir, 'data.yaml'), 'w') as f:
    yaml.dump(data_yaml, f)

print("Split complete. Labels properly handled with flat directory structure")