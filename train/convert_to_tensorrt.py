from ultralytics import YOLO
import os
cwd = os.getcwd()

model = YOLO(os.path.join(cwd,"runs/train/EFPS_3000img_640engine/best.pt"))

model.export(format = "engine", workspace = 8, batch = 1, half = True, nms = True, imgsz = (640,1024)) 