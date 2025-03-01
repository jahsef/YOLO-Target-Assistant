from ultralytics import YOLO
import os
cwd = os.getcwd()

model = YOLO(r"C:\Users\kevin\Documents\GitHub\YOLO11-Enfartment-PoopPS\runs\train\1024x1024_batch12\weights\best.pt")

model.export(format = "engine", workspace = 8, batch = 1, half = True, nms = True)