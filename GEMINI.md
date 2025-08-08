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
    
*   `src/`: Contains the main source code.
    *   `src/aimbot/`: Contains the core aimbot logic.
        *   `aimbot.py`: The main script orchestrating model loading, camera capture, input detection, and mouse movement.
        *   `data_parsing/targetselector.py`: Implements the target selection logic.
        *   `engine/model.py`: Handles model inference.
        *   `input/mousemover.py`: Manages mouse movement.
        *   `input/inputdetector.py`: Detects user input.
        *   `gui/gui_manager.py`: Manages the graphical user interface overlay.
        *   `utils/fpstracker.py`: Tracks frames per second.
    *   `src/data_getting/`: Likely contains scripts for data acquisition (e.g., `hard_mining.py`, `screenshots/`).
    *   `src/data_processing/`: Contains scripts for data manipulation (e.g., `convert_videos.py`, `crop_images.py`, `train_test_split.py`, `_auto_annotation/`).
    *   `src/train/`: Contains scripts related to model training and conversion.
        *   `train.py`: Script for training YOLOv11 models.
        *   `onnx_to_tensorrt.py`: Script for converting ONNX models to TensorRT engines.
        *   `yolo_pt_to_tensorrt.py`: Script for converting PyTorch YOLO models to TensorRT engines.
*   `data/`: Contains data-related assets.
    *   `data/datasets/`: Stores datasets used for training.
    *   `data/models/`: Directory for storing trained YOLOv11 models (TensorRT engine files).
*   `config/cfg.json`: Configuration file for the aimbot, defining model paths, sensitivity, targeting, and GUI settings.
*   `tests/`: Contains various test scripts (e.g., `test_targetselector.py`).
*   `benchmark/`: Contains scripts for benchmarking (e.g., `benchmark_inference.py`).

## New Features and Configuration

### Input Settings

A new `input_settings` section has been added to `config/cfg.json` to control aimbot activation:

*   `toggle_hotkey`: (String) Specifies a keyboard key to toggle the aimbot on/off. If left empty (`""`), this feature is disabled.
*   `right_click_toggle`: (Boolean) If `true`, the aimbot will only be active when the right mouse button is pressed. The `toggle_hotkey` overrides this setting.

### Other Settings

*   `inactive_throttle_ms`: (Integer) Renamed from `throttle_non_ads_ms`. This setting controls the delay (in milliseconds) when the aimbot is inactive to reduce CPU usage.

## Data Preparation Workflow

This project supports a workflow for preparing custom datasets, including auto-annotation and manual inspection.

### 1. Initial Data Splitting
Use `src/data_processing/train_test_split.py` to split your raw image data. This script takes images from `data/datasets/pre_split_dataset` and organizes them into a new dataset directory (e.g., `data/datasets/new_dir_here`) with `train` and `val` splits.

### 2. Auto-Annotation and CVAT Integration
For initial labeling, you can leverage the auto-annotation scripts:
*   Run `src/data_processing/_auto_annotation/annotate.py` to generate preliminary annotations for your images. This script will process images and create corresponding label files.
*   After annotation, run `src/data_processing/_auto_annotation/write_train_txt.py` to generate a `train.txt` file listing the paths to your annotated images.

To manually inspect and refine the auto-annotations using CVAT:
1.  Locate the images and generated labels in `src/data_processing/_auto_annotation/data/images` and `src/data_processing/_auto_annotation/data/labels/train` respectively.
2.  For CVAT upload, you typically need to create a `.zip` file containing the entire `data` directory from `src/data_processing/_auto_annotation/` (e.g., `data.zip` containing `data/images/...` and `data/labels/train/...`). Ensure that any `labels.zip` file within this directory is *not* included in your new zip.
3.  Upload this `data.zip` to CVAT. CVAT will automatically detect and import the images and their corresponding labels.
4.  Perform manual inspection and correction of annotations within CVAT.
5.  Export the refined dataset from CVAT in YOLO format.

### 3. Post-CVAT Processing and Training
After exporting from CVAT, the refined dataset can be used for model training as described in the "Model Preparation" section. You may need to use other scripts in `src/data_processing` (e.g., `convert_videos.py`, `crop_images.py`, `data_cleaning/clean_data.py`, `data_cleaning/rm_background.py`, `z_crop_data_and_labels/crop.py`) to further process your data before training, depending on your specific needs.

## Building and Running

**Model Preparation:**
This project requires TensorRT engine files for inference. You will need to either:
1.  Train your YOLOv11 model using `src/train/train.py`.
2.  Convert your trained model to a TensorRT engine using `src/train/yolo_pt_to_tensorrt.py` (for PyTorch models) or `src/train/onnx_to_tensorrt.py` (for ONNX models).
3.  Strip metadata from the TensorRT engine using `src/train/strip_metadata.py`.
Ensure the paths to your `.engine` files are correctly configured in `config/cfg.json`.

**Running the Aimbot:**
The primary entry point for running the aimbot is `src/aimbot/aimbot.py`. You can execute it directly:

```bash
python src/aimbot/aimbot.py --config "config/cfg.json"
```

Alternatively, batch scripts like `aimbot.bat` or `autoclicker.bat` might be provided for convenience, which likely execute the Python script with specific configurations.

**Configuration:**
Modify `config/cfg.json` to adjust settings such as:
*   `model`: Paths to your ADS and scanning models.
*   `sensitivity_settings`: Mouse sensitivity, max deltas, jitter, and overshoot.
*   `targeting_settings`: Target class IDs, head toggle, prediction, and FOV.
*   `gui_settings`: Overlay options.

## Development Conventions

*   **Python:** The codebase is primarily in Python.
*   **Configuration:** Project settings are managed through `config/cfg.json`.
*   **Performance:** Heavy reliance on GPU-accelerated libraries like CuPy and TensorRT for critical performance sections.
*   **Modularity:** The project is structured into logical modules (e.g., `aimbot`, `data_parsing`, `engine`, `input`, `gui`, `utils`).
