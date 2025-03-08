import os

# def reset_file_attributes(file_path):
#     """
#     Resets the file attributes of the specified file to 'Normal' (N).
    
#     Args:
#         file_path (str): Path to the file whose attributes need to be reset.
#     """
#     try:
#         # Use the 'attrib' command to remove the Archive attribute and set Normal attribute
#         os.system(f'attrib -A +N "{file_path}"')  # Remove Archive and set Normal
#         print(f"File attributes reset to Normal for: {file_path}")
#     except Exception as e:
#         print(f"Error resetting attributes for {file_path}: {e}")


def generate_train_txt(images_path, train_txt_path):
    """
    Generates a train.txt file with the correct image paths and resets its attributes to Normal.

    Args:
        images_path (str): Path to the directory containing the images.
        train_txt_path (str): Path to save the generated train.txt file.
    """
    # Ensure the images directory exists
    if not os.path.exists(images_path):
        print(f"Error: Images directory '{images_path}' does not exist.")
        return

    # Delete the existing train.txt file if it exists
    try:
        os.remove(train_txt_path)
    except FileNotFoundError:
        pass  # File doesn't exist; no action needed

    # Initialize the helper path prefix
    helper = 'data/images/train/'

    # Get the list of image files in the images directory
    list_images = [
        f for f in os.listdir(images_path)
        if os.path.isfile(os.path.join(images_path, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    # Sort the image files numerically (if possible)
    try:
        list_images = sorted(list_images, key=lambda x: int(''.join(filter(str.isdigit, os.path.splitext(x)[0]))))
    except ValueError:
        list_images = sorted(list_images)  # Fallback to alphabetical sorting

    # Write the image paths to train.txt
    with open(train_txt_path, 'w', encoding='UTF-8') as f:
        for image in list_images:
            train_string = helper + image + '\n'
            f.write(train_string)

    print(f"train.txt has been successfully generated at: {train_txt_path}")

    # Reset the attributes of the train.txt file to Normal
    # reset_file_attributes(train_txt_path)


# Define paths
cwd = os.getcwd()
images_path = os.path.join(cwd, 'train/auto_annotation/data/images')
train_txt_path = os.path.join(cwd, 'train/auto_annotation/data/train.txt')

# Generate train.txt and reset its attributes
generate_train_txt(images_path, train_txt_path)