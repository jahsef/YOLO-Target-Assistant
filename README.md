# Overview

This project is for highly optimized real time object detection. Full pipeline runs at \~160 fps on (640,640) on a 3080 10gb.

# Performance at a glance

|Component|Optimization|Speedup|
|-|-|-|
|Inference|TensorRT fp16 vs PyTorch|2-3x|
|Inference|Direct TensorRT bindings vs Ultralytics|\~20%|
|Preprocessing|CuPy fused kernel vs Ultralytics|23x|
|Screenshot|CuPy-native fork of BetterCam|\~20%|
|Tracker|C++/Eigen vs Ultralytics BYTETracker|60x|

# Data

Collected and labeled all data by hand using CVAT. Trained models on proprietary data up to 4000 images.

# Inference

Leverages TensorRT python bindings directly sidestepping Ultralytics implementation overhead (\~20% speedup).
TensorRT fp16 engine also gives a 2-3x speedup over PyTorch runtime.

# Preprocessing

CuPy fused kernel achieves a 23x speedup over Ultralytics on preprocessing. CuPy (nonfused) optimizations allows us a 10x speedup over Ultralytics. Within Ultralytics ecosystem, we can achieve 4x speedup for single frame inputs (https://github.com/ultralytics/ultralytics/pull/23743), (https://github.com/ultralytics/ultralytics/pull/25982), (https://github.com/ultralytics/ultralytics/pull/25989).

# Screenshotting

The fastest Windows screenshotting lib previously was bettercam (https://github.com/RootKit-Org/BetterCam), but it did not support CuPy natively. Forked it to support CuPy for a \~20% speedup over base library (https://github.com/jahsef/BettererCam).

# Tracker

Ultralytics BYTETracker is quite slow, eating up 1 ms when inference can be as fast as 5 ms. Found existing tracker implementations, but found them to be unsatisfactory. Vendored and vectorized Ultralytics version for 5x speedups on dense scenes. Wrote a C++ version leveraging eigen for 60x speedups on sparse scenarios, and 25x speedups on high density (https://github.com/jahsef/c_bytetracker). Forked lib also has QOL features unique to it while maintaining API parity.

# Small object detection

With many object detection models struggling with small objects, I sought a way to improve detection without modifying our architecture to satisfy real time constraints and limit engineering scope. Given the real time usage of this project, we can leverage temporal priors and split detection across three tiers, routed per frame:

* **base**: full-frame detector at capture resolution. Cheap, runs every frame, catches anything big enough.
* **scan SR**: while scanning, the capture region is patchified and batch those frames for super resolution and re-detect with a 'scan SR' model, then union with base under a shared NMS, recovering small objects the base model misses.
* **precision SR**: once a small target is locked, we crop a tight window around it, super resolve just that crop, and detect with a model trained on super resolution crops, for maximum detail where it actually matters.

*SR = Super resolution*

Routing is mutually exclusive per frame: scanning runs base / base + scan SR. A locked small target switches to precision SR on its crop, sufficiently large targets deferred to base model only. A temporal hysteresis budget keeps the precision crop at the last known location for a few frames to survive flickery detections. Note that scan SR compute (typically 16 - 64 patches from the capture region) is tractable because of TensorRT optimizations and because of high fps tracking only being needed for when a target is locked onto, not when scanning for targets.

# HSV object detection

Encountered problem with class of objects that was diverse in shape and size. Our yolo11 models could not reliably detect these due to data scarcity and the small size of these objects. We could have gone the architectural route, using a p2 branch and NWD loss instead of standard IOU, but that is extremely expensive for slight improvement on small objects. Though, these objects are generally red so we opted for classical CV instead of throwing compute and data at this issue. I use a multi step HSV pipeline using fused custom kernels, and FP reduction methods like morphology, row col ratio masking, avg pooling dilution, and a weighted center approach using a cached weight tensor. We do not use any template matching as that adds compute and data acquisition costs.



# Targeting

We predict a moving target's future position with a WMA velocity buffer over per-frame raw aim deltas (the target's observed offset from our crosshair, before sensitivity scaling). Soft gating based on WMA norm and RSI (yes, the one from finance) handles uncertainty hedging and oscillatory dampening. We also present a novel frame of reference problem here. Why not PID? Our actuator perturbs its own observation frame, so the reference shifts every time we act and a naive feedback loop has no fixed setpoint to converge to.

