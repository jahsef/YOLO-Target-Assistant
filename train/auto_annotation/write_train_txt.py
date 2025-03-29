import os


def generate_train_txt(images_path, train_txt_path):

    if not os.path.exists(images_path):
        print(f"Error: Images directory '{images_path}' does not exist.")
        return

    try:
        os.remove(train_txt_path)
    except FileNotFoundError:
        pass 

    # Initialize the helper path prefix
    helper = 'data/images/train/'
    # helper = ''
    

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

cwd = os.getcwd()
images_path = os.path.join(cwd, 'train\crop_data_and_labels\data\cropped')
train_txt_path = os.path.join(cwd, 'train\crop_data_and_labels/train.txt')


generate_train_txt(images_path, train_txt_path)