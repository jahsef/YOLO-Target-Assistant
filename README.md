# Overview
This project is for highly optimized real time object detection. Full pipeline runs at ~160 fps on desktop on a 3080.

# Performance at a glance
| Component | Optimization | Speedup |
|---|---|---|
| Inference | TensorRT fp16 vs PyTorch | 2-3x |
| Inference | Direct TensorRT bindings vs Ultralytics | ~20% |
| Preprocessing | CuPy fused kernel vs Ultralytics | 23x |
| Screenshot | CuPy-native fork of BetterCam | ~20% |
| Tracker | C++/Eigen vs Ultralytics BYTETracker | 60x |

# Data
Collected and labeled all data by hand using CVAT. Trained models on proprietary data up to 4000 images.

# Inference
We leverage TensorRT python bindings directly sidestepping Ultralytics implementation overhead (~20% speedup).
TensorRT fp16 engine also gives a 2-3x speedup over PyTorch runtime.

# Preprocessing
CuPy fused kernel achieves a 23x speedup over Ultralytics. CuPy (nonfused) optimizations allows us a 10x speedup over Ultralytics. Within Ultralytics ecosystem, we can achieve 4x speedup for single frame inputs (PR LINK HERE).

# Screenshotting
The fastest windows screenshotting lib previously was bettercam (link), but it did not support CuPy natively. We forked it to support CuPy for a ~20% speedup over base library (link to our lib).

# Tracker
Ultralytics BYTETracker is quite slow, eating up 1 ms when inference can be as fast as 5 ms. We found some existing tracker implementations matching Ultralytics API, but found them to be unsatisfactory. Vendored and vectorized Ultralytics version for 5x speedups on dense scenes. Wrote a C++ version leveraging eigen for 60x speedups on sparse scenarios, and 25x speedups on high density (link to our lib). Our custom c_bytetracker lib we argue also has a nicer API to work with requiring less explicit passing of args when they shouldn't be needed, while keeping API parity. 

# HSV object detection
We encountered a problem where a class of red reticles was quite diverse in shape and size. Our object detection models could not reliably detect these due to data scarcity and how small these reticles were. Instead, we opted to use classical CV instead of throwing compute at this issue. We use a multistep HSV pipeline with fused custom kernels and FP reduction methods to accurately regress keypoints. Our FP reduction methods include row col ratio masking, morphological approaches with erosion and dilation, avg pooling for dilution, and a weighted center approach with cached weight tensor. 

# Small object detection
With many object detection models struggling with small objects, we sought a way to improve detection without modifying our architecture to satisfy real time constraints and limit engineering scope. Given the real time usage of this project, we can leverage temporal priors and split detection across three tiers, routed per frame:

- **base**: full-frame detector at capture resolution. Cheap, runs every frame, catches anything big enough.
- **scan SR**: while scanning, we patchify the capture region and batch those frames for super resolution and re-detect with a 'scan SR' model, then union with base under a shared NMS, recovering small objects the base model misses.
- **precision SR**: once a small target is locked, we crop a tight window around it, super resolve just that crop, and detect with a model trained on super resolution crops, for maximum detail where it actually matters.

Routing is mutually exclusive per frame: scanning runs base / base + scan SR. A locked small target switches to precision SR on its crop, sufficiently large targets deferred to base model only. A temporal hysteresis budget keeps the precision crop at the last known location for a few frames to survive flickery detections. Note that scan SR compute (typically 16 - 64 patches from the capture region) is tractable because of TensorRT optimizations and because of high fps tracking only being needed for when a target is locked onto, not when scanning for targets.

# Targeting
We predict a moving target's future position with a WMA velocity buffer over our own actuator outputs. Soft gating based on WMA norm and RSI (yes, the one from finance) handles uncertainty hedging and oscillatory dampening. We also present a novel frame of reference problem here. Why not PID? Our actuator perturbs its own observation frame, so the reference shifts every time we act and a naive feedback loop has no fixed setpoint to converge to.

