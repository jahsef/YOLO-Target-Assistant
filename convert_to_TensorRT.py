from ultralytics import YOLO
import os
cwd = os.getcwd()

model = YOLO(os.path.join(cwd,"runs//detect//tune4//weights//best.pt"))#tuned model jajaja

model.export(format = "engine", half = True, workspace = 8, batch = 1, simplify = False)