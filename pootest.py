from ultralytics import YOLO
import os

cwd = os.getcwd()

model = YOLO(os.path.join(cwd,"runs//train//train_run//weights//best.pt"))
print(model.task)
print(type(model.metrics))
print(model.metrics)
print(model.info(detailed=False, verbose = True))
print(model.cfg)
print(model.model_name)