perf:
DONE 2026-07-26 — hsv fused into one kernel (mask+suppress+opening+density+weighted-centroid,
  one launch, one 24-byte D2H). was: 2 dlpack round trips, torch pools, 3 host-blocking float()
  syncs. measured paired on 5080: hsv path 1.18ms -> 0.24ms (4.9x, wins 58/60 trials); marginal
  cost on top of trt 1.21ms -> 0.057ms. e2e frame->mouse 5.5ms -> 4.38ms, ~180 -> 228 fps.
  torch is no longer imported by hsv_crosshair.py at all.

DONE 2026-07-26 — pywin32 GetAsyncKeyState replaces pynput (inputdetector.py + auto_clicker.py,
  dep dropped from requirements). rmb is polled on read so it can't be stale; the toggle hotkey
  keeps a 120Hz daemon poller only when a hotkey is actually configured.

DONE 2026-07-26 — 3-stage async loop (capture | detect | aim) behind other.async_pipeline,
  default OFF. all GPU work stays on one thread on purpose (a TRT execution context is not
  safe to drive from two). capture grabs JUST-IN-TIME: waits for the slot to drain, then
  holds off until the detect stage is about to free up. the first cut free-ran the capture
  thread, which at 600fps into a ~200fps pipeline pays grab 3x per consumed frame and spends
  the difference on GPU/GIL that detect wanted — that made threading a net LOSS (0.98x) once
  grab got expensive. jit is worth +6% at a cheap grab, +26% at 1.7ms, +46% at 4ms, for
  +0.06 to +1.6ms latency.

REJECTED 2026-07-26 — hsv on a second CUDA stream alongside trt. measured with paired
  interleaved trials: median delta -0.086ms, overlapped won 27/60 (a coin flip). trt already
  keeps the SMs busy, so there's no idle GPU for hsv to hide in. moot now anyway: hsv's
  marginal cost is 0.057ms.

measured 2026-07-26 on real hardware: grab() actually costs ~2.1ms (not the ~0.075ms the
  idle-desktop None-path suggested), detect ~5.5-12ms depending on route. so a capture
  thread IS worth real throughput: async measured 83 fps vs serial 70 fps with input state
  held equal. NOTE the earlier "async is 3x slower" reading was a rigged comparison — the
  control skipped the inactive-throttle sleep that the async path was paying.

open: grab() cost is the one unmeasured input — DXGI only produces frames when screen content
  changes, so it can't be characterized on an idle desktop. it decides whether async_pipeline
  is worth flipping on; sensitivity sweep is in tests/results/latest.md. measure it in-game.

offload mouse movements to separate thread for 1000hz polling (try to mimic hardware speed) ,decouple from aimbot speed
keep a buffer of deltas,frametimes
drain the 'leftover deltas' buffer (not executed stuff yet). mimics accel/decel. prolly just use exp drain of the buffer
  note: simulated at 1kHz it adds ~0.5ms latency for 0 fps gain, so it's only worth it for the
  smoothness/humanization angle, not perf. see sim fps [3-stage + mouse thread].

humanized movements idea:
train on my specific mouse movements,
model sees window of n previous movements + current delta to execute. based on those, predicts how to move.
should be distribution output, forget the exact model name. then choose from that distribution to simulate randomness based on my actual patterns.
training setup: 10 minutes of actual fighting data @ 100fps+ is probably plenty. 
should train with a variety of frametimes as well. synthetic slowing down is probably shit here.
probably important to get mouse movement data without deltas to execute as well, good pretext task of just mouse movements generally vs mouse movements on a target

update logging from our utils log() -> stdlib logging.debug(). apparently those use lazy execution. benched 2026-06-11: 1.63 microseconds per discarded call (vs 0.13 stdlib lazy), 8 calls -> ~13 microseconds/frame, 7 ms frametimes -> ~0.19% performance loss. worth considering in the future
if done: cfg logging level must move from logger 'aimbot' to 'src.aimbot' subtree, aimbot.py needs explicit logger name (runs as __main__), update fart_avgpool.py log import

FIXED 2026-07-26 — precision_sr could lock onto a phantom forever. hysteresis only covers
  frames where the crop finds NOTHING; ANY detection inside the crop refreshes the lock,
  which re-centres the crop on itself, and since base is skipped nothing can contradict it.
  aim sits on empty space until you release ADS. now capped: model.precision_sr_max_streak
  (30) forces a global look, lasting model.precision_sr_forced_scan_frames (2) frames —
  two, because a detection matching no existing track is created UNACTIVATED and only
  reported on the following frame, so a single scan frame yields zero tracks and the stale
  lock survives. costs one base inference per 30 frames while ADS-locked (~3%).
  reproduced + pinned in tests/integration/test_behaviour.py::TestPrecisionRouting.

bugs found while writing tests, all FIXED 2026-07-26:
1. hsv bypass reticle never reached the aim math. crosshair_rows_to_tracked stamps
   start_frame = last_frame = 0, so lifetime is 0, and Aimbot.aimbot() filtered
   `lifetime >= min_frames_to_target` (2) BEFORE get_deltas — so _get_crosshair fell back to
   the window centre and bypass_tracker was inert for aiming (routing still saw it). verified
   with the shipped cfg: reticle 50px off-centre gave dx=100 instead of dx=50. the min-age
   gate now exempts untracked rows (track_id < 0), since it exists to judge TRACK stability
   and a bypass row is a direct per-frame measurement, not a track.
2. _calculate_distance occlusion adjustment was inverted: it multiplied the height estimate's
   variance by the trust factor, and weight = 1/variance, so a squat/occluded box trusted the
   height MORE. now divides.
3. TargetSelector.detection_window_center was built (h//2, w//2) but used as (x, y). was inert
   while capture is square; now correct for any capture region.
4. _calculate_distance weighted by 1/angular_size**2, so a SMALLER (further) target got MORE
   weight. relative error of d = S/theta is (pixel noise / apparent size), so weight now goes
   as pixels**2 — bigger apparent target, more trust.

fix base data (relabel base set, label some of the unlabeled set for generalization)
semi supervised using augmentation invariance (STAC, or other semi supervised methods) for rest of unlabeled im too lazy to label

could update RSI harmonic dampening to allow rsi = 20 or 'oversold' to allow MORE movement to move through. however just dampening is probably the safest default
could try bollinger bands and possibly autocorr for our momentum vs mean reversion routing

bilinear simplification (remove sr model, simplify code alot for same performance)
possibly remove scanning model, or optimize it so that another final nms run isnt needed. 
cleanest code option entails scanning model and base model both having no nms baked into model for us to run nms on it finally. but we would probably use baked nms for all other models so makes it kinda wonky

bundle redesign — bundles/scan_sr deprecated 2026-08-05, scaffolding kept. bundle.py in
  yolo_semisupervised is currently broken (engine_geometry wants 4D sr_out, export.py stores a detector).

move momentum args / other magic numbers from src.aimbot.data_parsing.targetselector to config
possibly separate into advanced / basic config stuff
docs for config, code docs for my own reference


update readme:

Baseline the numbers. 23x / 160fps over what res, hardware, batch? One line each.
"We" → "I" for a solo signaling doc.
Push the 3-tier SR up — it's your most novel system, currently buried under HSV.
Underrated signal: knowing when not to use the heavy tool (HSV over a detector, rejecting PID). That's your most senior flex — surface it. explain why hsv was so much better decision here: data problem, architectural problems (p2 branch / nwd to solve, p2 branch is expensive, nwd into ultralytics i think is too much work), all withotu template matching just using clever morphological tricks and kernels to destroy most FPs
