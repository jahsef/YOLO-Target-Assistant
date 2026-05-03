fix base data (relabel base set, label some of the unlabeled set for generalization)
semi supervised using augmentation invariance (STAC, or other semi supervised methods) for rest of unlabeled im too lazy to label

could update RSI harmonic dampening to allow rsi = 20 or 'oversold' to allow MORE movement to move through. however just dampening is probably the safest default
could try bollinger bands and possibly autocorr for our momentum vs mean reversion routing

REAL-USAGE FINDINGS (after shipping)
- scan_sr: false positives noticeable in scanning
- precision_sr: clear feel improvement on small/distant targets when ADS-locked. keep this path.

NEXT: REPLACE PRECISION SR WITH BILINEAR + ADD HYSTERESIS
- drop the SR engine slot from precision path. rebuild bundle as a YOLO-only checkpoint + metadata (input_size, upscale_factor, class_names, bb_largest_side_threshold). caller bilinear-upscales the crop (cupy/torch resize, sub-µs) before YOLO. simpler, cheaper, ~8% mAP50-95 we measured was inside the n=150 noise floor anyway.
- scan_sr stays gone. base-only scanning is what we're shipping.
- multi-crop ambition (N concurrent precision crops, one per small tracked enemy) is still appealing but secondary — single-crop on the lock is what feels good today; multi-crop is a "later" win for crowded scenes.

HYSTERESIS (the actual bad-feel sources)
- bb-size boundary thrashing: target with max(h,w) hovering around bb_largest_side_threshold flips between base-only and precision-crop every frame. add asymmetric thresholds (enter precision at < T_low, leave at > T_high, e.g. T_low=48, T_high=64) so the routing decision is sticky.
- missed-detection in crop: if precision sr model misses a single frame, then we are fucked. add 1-2 frame hysteresis for crop locations
(added hysteresis for N frames already, need to implement bb size hysteresis still)

bilinear simplification + sticky routing + small-window grace = the next coherent unit of work.

HSV CONV-GATED VOTING SCHEME
- avg_pool2d(k=4, s=2) on mask → argmax patch → gate mask to that window → weighted_center on gated mask.
- top-K patches or last-frame stickiness to avoid argmax flipping between near-tied blobs.

HSV DUAL-MODE
- opening (minpool→maxpool) + avgpool stack as front-end. mode select via top-K argmax spread: cluster → small-dot (centroid in argmax ROI), scatter → multi-element (weighted_center on opened mask).
- thin reticles eaten by erosion. template matching skipped (authoring cost). connected-components deprioritized.


move momentum args / other magic numbers from src.aimbot.data_parsing.targetselector to config
