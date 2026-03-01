from ultralytics import YOLO
import os

BASE_DIR = "data/models/pf_1550img_26s_val/weights"
MODEL_FILENAME = "best.pt"
IMGSZ = (640,640)
FP16 = True
DEVICE = 0
WORKSPACE = 8
HALF = True
SIMPLIFY = True
BATCH = 1
NMS = True
DYNAMIC = False

cwd = os.getcwd()

# model  = YOLO(os.path.join(cwd,"runs/train/EFPS_3000image_realtrain_1440x1440_100epoch_batch6_11s/weights/onnx 1440p/best.pt"))
base_dir = BASE_DIR
model = YOLO(os.path.join(cwd,os.path.join(base_dir,MODEL_FILENAME)))
imgsz = IMGSZ
fp16 = FP16

path = model.export(device = DEVICE,format = "engine",workspace = WORKSPACE, half = HALF, simplify = SIMPLIFY, batch = BATCH, imgsz = imgsz, nms = NMS, dynamic = DYNAMIC)
                    # ,data = os.path.join(os.getcwd(),"train/datasets/EFPS_4000img/data.yaml"))

os.rename(path,f'{base_dir}/{imgsz[0]}x{imgsz[1]}.engine')