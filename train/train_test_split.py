import os
import shutil
from sklearn.model_selection import train_test_split
import yaml

# Define paths
cwd = os.getcwd()
base_dir = 'C:\\Users\\kevin\\Documents\\GitHub\\YOLO11-Final-Poop-2\\train\\train_test_split\\dataset'
images_dir = os.path.join(base_dir, 'images\\train')
labels_dir = os.path.join(base_dir, 'labels\\train')
train_images_dir = os.path.join(base_dir, 'images\\train')
val_images_dir = os.path.join(base_dir, 'images\\val')
train_labels_dir = os.path.join(base_dir, 'labels\\train')
val_labels_dir = os.path.join(base_dir, 'labels\\val')
data_yaml_path = os.path.join(base_dir, 'data.yaml')
train_txt_path = os.path.join(base_dir, 'train.txt')

# Create directories for train/val split if they don't exist
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

# Get the list of all images (assuming .jpg, .png, or similar extensions)
image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

# Split the list of images into train and validation sets (80% train, 20% val)
train_images, val_images = train_test_split(image_files, test_size=0.2, random_state=42)

# Function to safely move images and labels
def move_image_and_label(img, source_img_dir, target_img_dir, source_label_dir, target_label_dir):
    # Move image file
    try:
        shutil.move(os.path.join(source_img_dir, img), os.path.join(target_img_dir, img))
    except FileNotFoundError as e:
        print(f"Error moving image file {img}: {e}")
        return

    # Move corresponding label file
    label_file = os.path.splitext(img)[0] + ".txt"
    label_path = os.path.join(source_label_dir, label_file)
    if os.path.exists(label_path):
        try:
            shutil.move(label_path, os.path.join(target_label_dir, label_file))
        except FileNotFoundError as e:
            print(f"Error moving label file {label_file}: {e}")
    else:
        print(f"Warning: No label found for image {img}. Skipping this image.")

# Move images and labels into their respective directories
for img in train_images:
    move_image_and_label(img, images_dir, train_images_dir, labels_dir, train_labels_dir)

for img in val_images:
    move_image_and_label(img, images_dir, val_images_dir, labels_dir, val_labels_dir)

# Update data.yaml file to reflect the new train/val paths
with open(data_yaml_path, 'r') as yaml_file:
    data_yaml = yaml.safe_load(yaml_file)

# Modify the paths in data.yaml to point to the new train/val directories
data_yaml['train'] = os.path.join(base_dir, 'images/train')
data_yaml['val'] = os.path.join(base_dir, 'images/val')
data_yaml['names'] = ['Zombie', 'Berserker', 'Hunter', 'Head']  # Update with your class names

# Save the updated data.yaml
with open(data_yaml_path, 'w') as yaml_file:
    yaml.dump(data_yaml, yaml_file)

# Optionally, create or update train.txt and val.txt for listing image filenames
with open(train_txt_path, 'w') as f:
    for img in train_images:
        f.write(f"{os.path.join('images/train', img)}\n")

# Create val.txt
val_txt_path = os.path.join(base_dir, 'val.txt')
with open(val_txt_path, 'w') as f:
    for img in val_images:
        f.write(f"{os.path.join('images/val', img)}\n")

print("Dataset split into train and validation sets successfully!")
