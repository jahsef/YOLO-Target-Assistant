
from ultralytics import YOLO
import os

# model = YOLO(os.path.join(os.getcwd(), "runs/train/EFPS_3000image_realtrain_1440x1440_100epoch_batch6_11s/weights/best.pt"))
model = YOLO("yolo11n.pt").to("cpu")
results = model.predict(source=os.path.join(os.getcwd(),r'train\split_dataset\images\train\frame_1.jpg'), device="cpu")
print(results[0])

