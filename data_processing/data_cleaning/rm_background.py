import os
import random

# Configuration
DATASET_DIR = os.path.join(os.getcwd(),"data_processing/data_cleaning/_clean_dataset")
IMAGE_EXTS = ('.jpg', '.jpeg', '.png')  # Supported image extensions

# Path setup
images_dir = os.path.join(DATASET_DIR, "images", "train")
labels_dir = os.path.join(DATASET_DIR, "labels", "train")
train_txt_path = os.path.join(DATASET_DIR, "train.txt")

def find_background_images():
    """Identify images without corresponding label files"""
    background_images = []
    
    for image_file in os.listdir(images_dir):
        if not image_file.lower().endswith(IMAGE_EXTS):
            continue
        
        base_name = os.path.splitext(image_file)[0]
        label_file = os.path.join(labels_dir, f"{base_name}.txt")
        
        if not os.path.exists(label_file):
            background_images.append(image_file)
    
    return background_images

def update_train_txt(remaining_images):
    """Update train.txt with remaining images"""
    with open(train_txt_path, 'w') as f:
        for img in remaining_images:
            rel_path = os.path.join("images", "train", img)
            f.write(f"{rel_path}\n")

def main():
    # Identify background images
    background_images = find_background_images()
    
    if not background_images:
        print("No background images found")
        return
    
    # Calculate number to remove
    num_to_remove = len(background_images) // 2
    if num_to_remove < 1:
        print("Not enough background images to remove (need at least 2)")
        return
    
    # Randomly select images to remove
    random.seed(42)  # For reproducibility
    to_remove = random.sample(background_images, num_to_remove)
    
    # Remove selected images
    for img in to_remove:
        img_path = os.path.join(images_dir, img)
        os.remove(img_path)
        print(f"Removed: {img}")
    
    # Get remaining images
    all_images = [f for f in os.listdir(images_dir) if f.lower().endswith(IMAGE_EXTS)]
    
    # Update train.txt
    update_train_txt(all_images)
    
    print(f"\nRemoved {num_to_remove} background images")
    print(f"Total remaining images: {len(all_images)}")

if __name__ == "__main__":
    main()