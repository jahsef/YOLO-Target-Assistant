import os
import torch
import cv2

cwd = os.getcwd()
images_dir = os.path.join(cwd,'train/video_processing/converted_videos')
images_list = os.listdir(images_dir)
cropped_dir = os.path.join(os.path.join(cwd,'train/video_processing/cropped_images'))


for image_name in images_list:
    # print(image_name)
    image_path = os.path.join(images_dir,image_name)
    image = cv2.imread(image_path)
    shape = image.shape
    # print(image.shape)
    
    
    original_dim = (image.shape[1],image.shape[0])#wxh
    crop_dim = (1440,896)#wxh
    offset_width = (original_dim[0] - crop_dim[0])//2
    offset_height= (original_dim[1] - crop_dim[1])//2

    crop_region = [offset_height, original_dim[1]-offset_height, offset_width, original_dim[0]-offset_width]
    # image = image[100:1124, 560:2000]#h x w
    image = image[crop_region[0]:crop_region[1], crop_region[2]:crop_region[3]]#h x w
    cv2.imwrite(os.path.join(cropped_dir,image_name), image)