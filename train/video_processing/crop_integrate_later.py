import os
import torch
import cv2
from multiprocessing import Process

cwd = os.getcwd()
images_dir = os.path.join(cwd,'train/video_processing/converted_videos')
images_list = os.listdir(images_dir)
cropped_dir = os.path.join(os.path.join(cwd,'train/video_processing/cropped_images'))



def crop(image_list):
    for image_name in image_list:
        # print(image_name)
        image_path = os.path.join(images_dir,image_name)
        image = cv2.imread(image_path)
        # shape = image.shape
        # print(image.shape)
        
        
        original_dim = (image.shape[1],image.shape[0])#wxh
        crop_dim = (1440,1440)#wxh
        offset_width = (original_dim[0] - crop_dim[0])//2
        offset_height= (original_dim[1] - crop_dim[1])//2

        crop_region = [offset_height, original_dim[1]-offset_height, offset_width, original_dim[0]-offset_width]
        # image = image[100:1124, 560:2000]#h x w
        image = image[crop_region[0]:crop_region[1], crop_region[2]:crop_region[3]]#h x w
        cv2.imwrite(os.path.join(cropped_dir,image_name), image)

if __name__ == '__main__':
    n_processes = 16
    for i in range(n_processes):
        curr_images_list = images_list[i::n_processes]
        print(len(curr_images_list))
        Process(target = crop, args = (curr_images_list,)).start()

