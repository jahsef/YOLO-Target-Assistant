perf:
true async (main hot loop, hsv ). pros: more fps, cons: slightly mroe latency (maybe 3ms, more fps may outweigh)
if we get true async though, we might be able to ~1.2-1.5x fps? even 5ms latency isnt a concern. humans are 100ms+, input is already like 10-50ms.
might also be more cache friendly (hits cpu/gpu more constantly, keeping aimbot stuff in cache). unsure if 100 fps -> 150 fps brings any meaningful cache hits though

instead of pynput input polling for input/inputdetector.py we can use pywin32 async key state polling (removes dep)
torch dlpack round trip in hsv_crosshair.py:339-343, or write some kernel for it instead

offload mouse movements to separate thread for 1000hz polling (try to mimic hardware speed) ,decouple from aimbot speed 
keep a buffer of deltas,frametimes
drain the 'leftover deltas' buffer (not executed stuff yet). mimics accel/decel. prolly just use exp drain of the buffer

humanized movements idea:
train on my specific mouse movements,
model sees window of n previous movements + current delta to execute. based on those, predicts how to move.
should be distribution output, forget the exact model name. then choose from that distribution to simulate randomness based on my actual patterns.
training setup: 10 minutes of actual fighting data @ 100fps+ is probably plenty. 
should train with a variety of frametimes as well. synthetic slowing down is probably shit here.
probably important to get mouse movement data without deltas to execute as well, good pretext task of just mouse movements generally vs mouse movements on a target

update logging from our utils log() -> stdlib logging.debug(). apparently those use lazy execution. benched 2026-06-11: 1.63 microseconds per discarded call (vs 0.13 stdlib lazy), 8 calls -> ~13 microseconds/frame, 7 ms frametimes -> ~0.19% performance loss. worth considering in the future
if done: cfg logging level must move from logger 'aimbot' to 'src.aimbot' subtree, aimbot.py needs explicit logger name (runs as __main__), update fart_avgpool.py log import

fix base data (relabel base set, label some of the unlabeled set for generalization)
semi supervised using augmentation invariance (STAC, or other semi supervised methods) for rest of unlabeled im too lazy to label

could update RSI harmonic dampening to allow rsi = 20 or 'oversold' to allow MORE movement to move through. however just dampening is probably the safest default
could try bollinger bands and possibly autocorr for our momentum vs mean reversion routing

bilinear simplification (remove sr model, simplify code alot for same performance)
possibly remove scanning model, or optimize it so that another final nms run isnt needed. 
cleanest code option entails scanning model and base model both having no nms baked into model for us to run nms on it finally. but we would probably use baked nms for all other models so makes it kinda wonky

move momentum args / other magic numbers from src.aimbot.data_parsing.targetselector to config
possibly separate into advanced / basic config stuff
docs for config, code docs for my own reference


update readme:

Baseline the numbers. 23x / 160fps over what res, hardware, batch? One line each.
"We" → "I" for a solo signaling doc.
Push the 3-tier SR up — it's your most novel system, currently buried under HSV.
Underrated signal: knowing when not to use the heavy tool (HSV over a detector, rejecting PID). That's your most senior flex — surface it. explain why hsv was so much better decision here: data problem, architectural problems (p2 branch / nwd to solve, p2 branch is expensive, nwd into ultralytics i think is too much work), all withotu template matching just using clever morphological tricks and kernels to destroy most FPs
