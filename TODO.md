

tracker from scratch (but actually follow through this time) (would need to do on gpu to make it worth) (cupy tracker proved poop, new vectorized tracker only better for like 5+ targets which is rare)

could update RSI harmonic dampening to allow rsi = 20 or 'oversold' to allow MORE movement to move through. however just dampening is probably the safest default

SMALL TARGET IMPROVEMENTS
dynamically crop 160x160 from 640x640 then use upsampler like real-esrgan_4x to 640x640
train using 160x160 crops, upsample offline, then boom new dataset of upsampled
train on that hoe, then bam model good on small targets
if a target < 50 pixels on smallest dimension, we center crop of 160x160 to that target im thinking. we need to scale everything by 1/4 to bring us back to normal space.
then we also need to apply crop offsets to keep consistency between switching from normal mode to small target mode. 