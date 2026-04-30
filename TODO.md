remove crosshair class entirely [DONE]
- HSV red-mask path shipped (src/aimbot/engine/hsv_crosshair.py + cfg flags model_predict_crosshair / hsv_predict_crosshair). real-usage win: targets no longer occluded by the big crosshair the model needed to see.
- still want to retire crosshair_cls_id from training data so YOLO capacity goes fully to enemies.

labeled dataset for hsv detection params color range and center, min,max of v/s then using grid search / bayesian.
100 images of hsv filtering with the cupy kernel should be insanely fast (19k fps supposedly in batch 1)
so batched filtering we should be able to grid search like 1000 combinations in a second
or use bayesian optimizaiton because its cool

fix base data
semi supervised using augmentation invariance (STAC, or other semi supervised methods)

tracker from scratch (but actually follow through this time) (would need to do on gpu to make it worth) (cupy tracker proved poop, new vectorized tracker only better for like 5+ targets which is rare)

could update RSI harmonic dampening to allow rsi = 20 or 'oversold' to allow MORE movement to move through. however just dampening is probably the safest default

REAL-USAGE FINDINGS (after shipping)
- scan_sr: net negative. false positives noticeable in scanning running with scan_sr disabled (config: scan_sr_bundle = "") feels better. worst case is FP when on ur gun, pulling down aim to floor
- precision_sr: clear feel improvement on small/distant targets when ADS-locked. keep this path.

NEXT: REPLACE PRECISION SR WITH BILINEAR + ADD HYSTERESIS
- drop the SR engine slot from precision path. rebuild bundle as a YOLO-only checkpoint + metadata (input_size, upscale_factor, class_names, bb_largest_side_threshold). caller bilinear-upscales the crop (cupy/torch resize, sub-µs) before YOLO. simpler, cheaper, ~8% mAP50-95 we measured was inside the n=150 noise floor anyway.
- scan_sr stays gone. base-only scanning is what we're shipping.
- multi-crop ambition (N concurrent precision crops, one per small tracked enemy) is still appealing but secondary — single-crop on the lock is what feels good today; multi-crop is a "later" win for crowded scenes.

HYSTERESIS (the actual bad-feel sources)
- bb-size boundary thrashing: target with max(h,w) hovering around bb_largest_side_threshold flips between base-only and precision-crop every frame. add asymmetric thresholds (enter precision at < T_low, leave at > T_high, e.g. T_low=48, T_high=64) so the routing decision is sticky.
- missed-detection in crop: if precision sr model misses a single frame, then we are fucked. add 1-2 frame hysteresis for crop locations

bilinear simplification + sticky routing + small-window grace = the next coherent unit of work.
