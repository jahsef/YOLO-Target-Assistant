from ultralytics import YOLO
import os
cwd = os.getcwd()

model  = YOLO(os.path.join(cwd,"runs/train/EFPS_3000image_realtrain_1440x1440_100epoch_batch6_11s/weights/best.pt"))

model.export(format = "engine", workspace = 8, batch = 1, half = True, imgsz = (896,1440)) 