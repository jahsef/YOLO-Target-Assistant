import os

def normalize_filenames_for_cvat(directory):
    """Renames files in the directory to remove special characters and ensure numerical order."""
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    
    # Sort files based on the numeric part of the filename
    sorted_files = sorted(files, key=lambda x: int(''.join(filter(str.isdigit, os.path.splitext(x)[0]))))

    renamed_files = set()  # Track renamed files to avoid duplicates
    next_index = 0  # Start from index 0

    for old_name in sorted_files:
        extension = os.path.splitext(old_name)[1]  # Get file extension (e.g., .jpg)
        
        # Generate a unique new filename
        while True:
            new_name = f"frame_{next_index}{extension}"
            if new_name not in renamed_files and not os.path.exists(os.path.join(directory, new_name)):
                break
            next_index += 1  # Increment index if the name is already taken
        
        # Rename the file
        old_path = os.path.join(directory, old_name)
        new_path = os.path.join(directory, new_name)
        os.rename(old_path, new_path)
        
        # Track the renamed file
        renamed_files.add(new_name)
        
        # print(f"Renamed: {old_name} -> {new_name}")
        
        next_index += 1  # Move to the next index

# Define the directory containing the images
image_dir = os.path.join(os.getcwd(),"train/video_processing/converted_videos")

# Normalize filenames for CVAT
normalize_filenames_for_cvat(image_dir)
