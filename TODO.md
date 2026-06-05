perf:
true async (main hot loop, hsv ). pros: more fps, cons: slightly mroe latency (maybe 3ms, more fps may outweigh)
instead of pynput input polling for input/inputdetector.py we can use pywin32 async key state polling (removes dep)
torch dlpack round trip in hsv_crosshair.py:339-343, or write some kernel for it instead


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
