import os
import shutil

# Configuration
DIRTY_DIR = "./data_processing/data_cleaning/_dirty_dataset"
CLEAN_DIR = "./data_processing/data_cleaning/_clean_dataset"
AREA_THRESHOLD = 0.025 * 0.025  # 0.000625
IMAGE_EXTS = ('.jpg', '.png')  # Supported image extensions

# Create clean directory structure
os.makedirs(os.path.join(CLEAN_DIR, "images", "train"), exist_ok=True)
os.makedirs(os.path.join(CLEAN_DIR, "labels", "train"), exist_ok=True)

def process_annotations(label_path):
    """Process annotations and return (filtered_lines, keep_image)"""
    valid_annotations = []
    original_has_annotations = False
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            original_has_annotations = True
            
            parts = line.split()
            if len(parts) != 5:
                continue  # Skip invalid lines
            
            cls_id, x, y, w, h = parts
            try:
                w_float = float(w)
                h_float = float(h)
            except ValueError:
                continue  # Skip invalid numbers
            
            if cls_id == '2' and (w_float * h_float < AREA_THRESHOLD):
                continue  # Skip this annotation
            
            valid_annotations.append(line)
    
    # Determine if we should keep the image
    if original_has_annotations:
        keep_image = len(valid_annotations) > 0
    else:
        keep_image = True  # Empty label file means keep image
    
    return valid_annotations, keep_image

# Process all images
kept_images = []
dirty_images_dir = os.path.join(DIRTY_DIR, "images", "train")
dirty_labels_dir = os.path.join(DIRTY_DIR, "labels", "train")

for image_file in os.listdir(dirty_images_dir):
    if not image_file.lower().endswith(IMAGE_EXTS):
        continue
    
    image_path = os.path.join(dirty_images_dir, image_file)
    base_name = os.path.splitext(image_file)[0]
    label_file = f"{base_name}.txt"
    label_path = os.path.join(dirty_labels_dir, label_file)
    
    label_exists = os.path.exists(label_path)
    filtered_annotations = []
    keep_image = True

    if label_exists:
        # Process annotations if label exists
        filtered_annotations, keep_image = process_annotations(label_path)
    
    if not keep_image:
        print(f"Removing {image_file} (invalid annotations)")
        continue
    
    # Copy image
    clean_image_path = os.path.join(CLEAN_DIR, "images", "train", image_file)
    shutil.copy2(image_path, clean_image_path)
    
    # Handle labels
    if label_exists and keep_image:
        # Only create label file if there are valid annotations
        if filtered_annotations:
            clean_label_path = os.path.join(CLEAN_DIR, "labels", "train", label_file)
            with open(clean_label_path, 'w') as f:
                f.write("\n".join(filtered_annotations))
    
    # Add to kept images list regardless of label existence
    kept_images.append(f"images/train/{image_file}")

# Copy data.yaml
shutil.copy2(
    os.path.join(DIRTY_DIR, "data.yaml"),
    os.path.join(CLEAN_DIR, "data.yaml")
)

# Write new train.txt
with open(os.path.join(CLEAN_DIR, "train.txt"), 'w') as f:
    f.write("\n".join(kept_images))

print(f"Cleaning complete. Kept {len(kept_images)} images (including {len(kept_images) - len(os.listdir(os.path.join(CLEAN_DIR, 'labels', 'train')))} without labels.")