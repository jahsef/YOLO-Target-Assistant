from ultralytics import YOLO
import os
cwd = os.getcwd()

model = YOLO(os.path.join(cwd,"runs//train//train_run//weights//best.pt"))

model.export(format="engine", imgsz=1440, half=True, device=0, 
             simplify=True, workspace=8, batch=1)