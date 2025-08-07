# Project: YOLO11-Enfartment-PoopPS

## Project Overview

This project is a real-time target selection system (aimbot) that leverages YOLOv11 for object detection. It is highly optimized for low-latency inference (achieving 240+ FPS) by utilizing TensorRT, CuPy, and PyTorch. Key features include real-time screen capture, efficient preprocessing, Kalman-based object tracking for smoother predictions, and a custom target selection mechanism with configurable filters.

**Disclaimer:** This repository is intended solely for educational and research purposes. No model weights are provided, and the use of this project in competitive or commercial environments is not condoned.

## Key Technologies

*   **Python:** Primary programming language.
*   **YOLOv11 (Ultralytics):** Object detection framework.
*   **TensorRT:** NVIDIA's SDK for high-performance deep learning inference.
*   **CuPy:** NumPy-compatible array library for GPU-accelerated computing.
*   **PyTorch:** Open-source machine learning framework.
*   **BettererCam:** A custom fork of BetterCam for efficient screen capture.

## Project Structure and Key Files

*   `aimbot/`: Contains the core aimbot logic.
    *   `aimbot.py`: The main script orchestrating model loading, camera capture, input detection, and mouse movement.
    *   `data_parsing/targetselector.py`: Implements the target selection logic.
    *   `engine/model.py`: Handles model inference.
    *   `input/mousemover.py`: Manages mouse movement.
    *   `input/inputdetector.py`: Detects user input.
    *   `gui/gui_manager.py`: Manages the graphical user interface overlay.
    *   `utils/fpstracker.py`: Tracks frames per second.
*   `aimbot_config.json`: Configuration file for the aimbot, defining model paths, sensitivity, targeting, and GUI settings.
*   `models/`: Directory for storing trained YOLOv11 models (TensorRT engine files).
*   `train/`: Contains scripts related to model training and conversion.
    *   `train.py`: Script for training YOLOv11 models.
    *   `onnx_to_tensorrt.py`: Script for converting ONNX models to TensorRT engines.
    *   `yolo_pt_to_tensorrt.py`: Script for converting PyTorch YOLO models to TensorRT engines.
*   `data_getting/`: Likely contains scripts for data acquisition (e.g., `hard_mining.py`, `screenshots/`).
*   `data_processing/`: Contains scripts for data manipulation (e.g., `convert_videos.py`, `crop_images.py`, `train_test_split.py`, `_auto_annotation/`).
*   `datasets/`: Stores datasets used for training.
*   `test/`: Contains various test scripts (e.g., `benchmark.py`, `dpg_class_test.py`).

## Building and Running

**Model Preparation:**
This project requires TensorRT engine files for inference. You will need to either:
1.  Train your own YOLOv11 models using `train/train.py`.
2.  Convert existing PyTorch YOLO models to TensorRT engines using `train/yolo_pt_to_tensorrt.py`.
3.  Convert ONNX models to TensorRT engines using `train/onnx_to_tensorrt.py`.
Ensure the paths to your `.engine` files are correctly configured in `aimbot_config.json`.

**Running the Aimbot:**
The primary entry point for running the aimbot is `aimbot/aimbot.py`. You can execute it directly:

```bash
python aimbot/aimbot.py
```

Alternatively, batch scripts like `aimbot.bat` or `autoclicker.bat` might be provided for convenience, which likely execute the Python script with specific configurations.

**Configuration:**
Modify `aimbot_config.json` to adjust settings such as:
*   `model`: Paths to your ADS and scanning models.
*   `sensitivity_settings`: Mouse sensitivity, max deltas, jitter, and overshoot.
*   `targeting_settings`: Target class IDs, head toggle, prediction, and FOV.
*   `gui_settings`: Overlay options.

## Development Conventions

*   **Python:** The codebase is primarily in Python.
*   **Configuration:** Project settings are managed through `aimbot_config.json`.
*   **Performance:** Heavy reliance on GPU-accelerated libraries like CuPy and TensorRT for critical performance sections.
*   **Modularity:** The project is structured into logical modules (e.g., `aimbot`, `data_parsing`, `engine`, `input`, `gui`, `utils`).
