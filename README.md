Another YOLO Aimbot (For Research & Educational Use Only)

A real-time target selection system leveraging YOLO-based object detection and fast preprocessing for high-FPS environments.

📌 Overview

This project demonstrates the use of real-time computer vision and deep learning techniques for automated target acquisition. It utilizes YOLOv11 for object detection, with a custom pipeline optimized for low-latency inference at 240+ FPS using TensorRT, CuPy, and Torch.

⚙️ Features

Real-time screen capture with BettererCam (BetterCam fork)

Fast preprocessing using CuPy and Torch (0.1ms)

YOLOv11 inference with TensorRT

Kalman-based object tracking for smoother predictions

Custom target selector with delta optimization logic

Configurable bounding box filters

🧪 Tech Stack

Python

YOLOv11 (Ultralytics)

TensorRT

CuPy / PyTorch

BettererCam (custom BetterCam fork)

🚀 Performance

Tested at 320x320, 220 FPS on RTX 3080

Achieved 220 FPS on RTX 3070 with lower monitor resolutions

📷 Demo

TODO ADD DEMO

⚠️ Disclaimer

This repository is intended solely for educational and research purposes.

No model weights are provided. I do not condone the use of this project in any competitive or commercial environments. Please use responsibly.

📄 Future Work

Explore capture/mouse movement spoofing options

Integrate reinforcement learning for more dynamic human-like aim adjustments

