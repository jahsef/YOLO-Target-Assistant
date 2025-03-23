# from ultralytics import YOLO
# import os

# if __name__ == '__main__':
    
#     model = YOLO(os.path.join(os.getcwd(),"runs/train/EFPS_3000image_old11m/weights/best.pt"))
#     model.val(data = os.path.join(os.getcwd(),"train/datasets/EFPS_4000img/data.yaml"))
    
#     model = YOLO(os.path.join(os.getcwd(),"runs/train/EFPS_4000img_11s_1440p_batch6_epoch200/weights/best.pt"))
#     model.val(data = os.path.join(os.getcwd(),"train/datasets/EFPS_4000img/data.yaml"))
    
#     model = YOLO(os.path.join(os.getcwd(),"runs/train/\EFPS_4000img_11n_1440p_batch11_epoch100/weights/best.pt"))
#     model.val(data = os.path.join(os.getcwd(),"train/datasets/EFPS_4000img/data.yaml"))
    
    
    
import torch
print(torch.__version__)
torch.as_tensor()