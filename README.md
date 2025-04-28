YOLO Target Assistant (For Research & Educational Use Only)

Real-time object tracking with YOLOv11 + TensorRT, optimized for low-latency environments.


📌 Overview

This project demonstrates the use of real-time computer vision and deep learning techniques for automated target acquisition. It utilizes YOLOv11 for object detection, with a custom pipeline optimized for low-latency inference at 240+ FPS using TensorRT, CuPy, and Torch.


Requirements (Tested)

Python 3.11

CUDA 12.6

TensorRT 10.8

PyTorch 2.6.0+cu126

Betterercam (custom fork; can be adapted for Bettercam or DXCam, with some performance trade-offs)


📷 Demo (Click to View)

[![Tracking Showcase](https://img.youtube.com/vi/tjGYJhSO0tg/hqdefault.jpg)](https://youtu.be/tjGYJhSO0tg)
[![Multitarget Showcase](https://img.youtube.com/vi/AezA8emdlb4/hqdefault.jpg)](https://youtu.be/AezA8emdlb4)


🚀 Performance

Tested at 320×320 resolution

200+ FPS on RTX 3070 in live environment

Lowering game settings recommended to maximize FPS

FPS may bottleneck at frame buffer limits during live use (monitor refresh rate)


⚠️ Disclaimer

This repository is intended for educational and research purposes only.

No model weights are provided. Use in competitive or commercial environments is not supported.

