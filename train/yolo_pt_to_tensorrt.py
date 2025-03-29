from ultralytics import YOLO
import os
cwd = os.getcwd()

# model  = YOLO(os.path.join(cwd,"runs/train/EFPS_3000image_realtrain_1440x1440_100epoch_batch6_11s/weights/onnx 1440p/best.pt"))
base_dir ="runs/train/EFPS_11n_4000img_640x640_batch16/weights"
model = YOLO(os.path.join(cwd,os.path.join(base_dir,'best.pt')))

imgsz = (320,320)
fp16 = True

path = model.export(device = 0,format = "engine",workspace = 8, half = fp16,int8 = False, simplify = True, batch = 1, imgsz = imgsz, nms = True, dynamic = False
                    ,data = os.path.join(os.getcwd(),"train/datasets/EFPS_4000img/data.yaml"))

os.rename(path,f'{base_dir}/{imgsz[0]}x{imgsz[1]}_fp16{fp16}.engine')