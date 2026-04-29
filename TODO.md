remove crosshair class entirely, use HSV / classical CV to find crosshairs (improve enemy recall and reduce data need for crosshairs)
fix base data
semi supervised using augmentation invariance (STAC, or other semi supervised methods)

tracker from scratch (but actually follow through this time) (would need to do on gpu to make it worth) (cupy tracker proved poop, new vectorized tracker only better for like 5+ targets which is rare)

could update RSI harmonic dampening to allow rsi = 20 or 'oversold' to allow MORE movement to move through. however just dampening is probably the safest default

SMALL TARGET IMPROVEMENTS [DONE — POC]
dynamically crop 160x160 from 640x640 then use upsampler like real-esrgan_4x to 640x640
train using 160x160 crops, upsample offline, then boom new dataset of upsampled
train on that hoe, then bam model good on small targets
if a target < 50 pixels on smallest dimension, we center crop of 160x160 to that target im thinking. we need to scale everything by 1/4 to bring us back to normal space.
then we also need to apply crop offsets to keep consistency between switching from normal mode to small target mode.
shipped via SRBundleEngine + base/scan_sr/precision_sr routing in aimbot.py. SR gave only ~8% mAP50-95 over bilinear on a small (n=150) val set — not worth the runtime hit (80fps SR vs 140fps base).

NEXT: REPLACE SR WITH BILINEAR + MULTI-CROP
- drop the SR engine slot entirely. rebuild bundle as a YOLO-only checkpoint with metadata: input_size, upscale_factor, class_names, bb_largest_side_threshold. caller does the upscale (cupy/torch resize, sub-µs).
- replace the single locked-target precision crop with N concurrent crops, one per small tracked enemy:
    - drive crop placement from PREVIOUS frame's tracker state (kalman extrapolates fine), so base + crop-batch inputs are all known up front and can dispatch async.
    - export the YOLO crop engine with a fixed max batch of 8. zero-pad unused slots, mask outputs by valid_count before applying per-crop offsets.
    - if >8 small targets exist, top-K by LRU age (or distance to crosshair) — bottom of the list isn't aim-relevant.
- keep the cross-model NMS pass (base ∪ all crops) — multiple crops can overlap and base detections overlap with crops too. this is union-of-models dedup, not SR-specific.
- routing semantics stay: scanning → base + (crops for last-frame small targets); ADS+small lock → just precision crop on the lock; ADS+large lock → base only.

bassically just improve performance and also some bad feel in real usage