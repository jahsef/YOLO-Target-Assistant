import os
import shutil
from sklearn.model_selection import train_test_split
import yaml

# Define paths
cwd = os.getcwd()
base_dir = os.path.join(cwd, 'train')
pre_split_dir = os.path.join(base_dir, 'pre_split_dataset')
split_dir = os.path.join(base_dir, 'split_dataset')

images_dir = os.path.join(pre_split_dir, 'images/train')
labels_dir = os.path.join(pre_split_dir, 'labels/train')
train_images_dir = os.path.join(split_dir, 'images/train')
val_images_dir = os.path.join(split_dir, 'images/val')
train_labels_dir = os.path.join(split_dir, 'labels/train')
val_labels_dir = os.path.join(split_dir, 'labels/val')
data_yaml_path = os.path.join(pre_split_dir, 'data.yml')
train_txt_path = os.path.join(pre_split_dir, 'train.txt')
split_train_txt_path = os.path.join(split_dir, 'train.txt')
split_val_txt_path = os.path.join(split_dir, 'val.txt')

# Create directories for split_dataset if they don't exist
os.makedirs(split_dir, exist_ok=True)
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

def clear_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path, ignore_errors=True)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

# Get the list of all images (assuming .jpg, .png, or similar extensions)
image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

# Split the list of images into train and validation sets (80% train, 20% val)
train_images, val_images = train_test_split(image_files, test_size=0.2, random_state=42)

# Function to safely copy images and labels
def copy_image_and_label(img, source_img_dir, target_img_dir, source_label_dir, target_label_dir):
    # Copy image file
    try:
        shutil.copy2(os.path.join(source_img_dir, img), os.path.join(target_img_dir, img))
    except FileNotFoundError as e:
        print(f"Error copying image file {img}: {e}")
        return

    # Copy corresponding label file
    label_file = os.path.splitext(img)[0] + ".txt"
    label_path = os.path.join(source_label_dir, label_file)
    if os.path.exists(label_path):
        try:
            shutil.copy2(label_path, os.path.join(target_label_dir, label_file))
        except FileNotFoundError as e:
            print(f"Error copying label file {label_file}: {e}")
    else:
        print(f"Warning: No label found for image {img}. Skipping this image.")

# Clear split_dataset directory before copying new files
clear_directory(split_dir)

# Copy images and labels into their respective directories
for img in train_images:
    copy_image_and_label(img, images_dir, train_images_dir, labels_dir, train_labels_dir)

for img in val_images:
    copy_image_and_label(img, images_dir, val_images_dir, labels_dir, val_labels_dir)

# Update data.yml file to reflect the new train/val paths
with open(data_yaml_path, 'r') as yaml_file:
    data_yaml = yaml.safe_load(yaml_file)

# Modify the paths in data.yml to point to the new train/val directories
data_yaml['train'] = os.path.join(split_dir, 'images/train')
data_yaml['val'] = os.path.join(split_dir, 'images/val')

# Save the updated data.yml
updated_data_yaml_path = os.path.join(split_dir, 'data.yml')
with open(updated_data_yaml_path, 'w') as yaml_file:
    yaml.dump(data_yaml, yaml_file)

# Create or update train.txt and val.txt for listing image filenames
with open(split_train_txt_path, 'w') as f:
    for img in train_images:
        f.write(f"{os.path.join('images/train', img)}\n")

with open(split_val_txt_path, 'w') as f:
    for img in val_images:
        f.write(f"{os.path.join('images/val', img)}\n")

print("Dataset split into train and validation sets successfully!")
