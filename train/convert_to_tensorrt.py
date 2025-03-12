from ultralytics import YOLO
import os
cwd = os.getcwd()

# model  = YOLO(os.path.join(cwd,"runs/train/EFPS_3000image_realtrain_1440x1440_100epoch_batch6_11s/weights/onnx 1440p/best.pt"))
model = YOLO(os.path.join(cwd,"runs/train/EFPS_4000img_11s_1440p_batch6_epoch200/weights/int8/best.pt"))

model.export(device = 0,format = "engine",workspace = 8, half = True,  batch = 1, imgsz = (1440,1440), nms = True,data = os.path.join(os.getcwd(),"train/datasets/EFPS_4000img/data.yaml"))