# from tensorrt import Inspector
from ultralytics import YOLO
import os

cwd = os.getcwd()

model = YOLO(os.path.join(cwd, "runs/train/EFPS_4000img_11n_1440p_batch11_epoch100/weights/best.pt"))
model.tune(use_ray=False, iterations = 10, batch = 11)