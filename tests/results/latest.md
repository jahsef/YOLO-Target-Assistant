perf results
============

run      : 2026-07-26T18:40:24
host     : jevinc / AMD64 / py3.11.2
gpu      : NVIDIA GeForce RTX 5080
baseline : tests/results/baseline.json
vs base  : compares min (steady state), not mean -- see tests/conftest.py
flagged  : '!' = moved >25% the wrong way AND more than the metric's
           own run-internal jitter. never fails a test.
           sub-100us GPU metrics are launch-overhead dominated, so treat
           their flags as advisory.

architecture (simulated)
------------------------

metric                                                 min      mean       p50       p95       p99     n unit vs base  note
---------------------------------------------------------------------------------------------------------------------------
sim fps [serial]                                  213.2500                                             1 fps    +0.0%  fused hsv
sim latency p50 [serial]                            5.5227                                             1 ms     +0.0%  screenshot -> mouse
sim fps [two_stage]                               214.2500                                             1 fps    +0.0%  fused hsv
sim latency p50 [two_stage]                         5.4993                                             1 ms     +0.0%  screenshot -> mouse
sim fps [three_stage]                             225.0000                                             1 fps    +0.0%  fused hsv
sim latency p50 [three_stage]                       5.5839                                             1 ms     +0.0%  screenshot -> mouse
sim fps [3-stage + mouse thread]                  225.0000                                             1 fps    +0.0%
sim latency p50 [3-stage + mouse thread]            6.0880                                             1 ms     +0.0%
sim fps [serial, pre-fusion hsv]                  172.2500                                             1 fps    +0.0%  torch chain, 3 syncs
sim hsv-fusion speedup                              1.2380                                             1 x      +0.0%  serial vs serial
sim hsv-fusion latency saved                        1.1060                                             1 ms     +0.0%
sim 3-stage fps gain                                1.0551                                             1 x      +0.0%
sim 3-stage latency cost                            0.0612                                             1 ms     +0.0%  positive = worse than serial
sim 3-stage gain @ grab=0.05ms                      1.0551                                             1 x      +0.0%  serial 213 -> 3-stage 225 fps
sim 3-stage gain @ grab=0.5ms                       1.1298                                             1 x      +0.0%  serial 194 -> 3-stage 220 fps
sim 3-stage gain @ grab=1.67ms                      1.2555                                             1 x      +0.0%  serial 158 -> 3-stage 199 fps
sim fps [3-stage capture=free]                    154.5000                                             1 fps    +0.0%  grab 1.67ms
sim latency p50 [3-stage capture=free]             13.4063                                             1 ms     +0.0%  grab 1.67ms
sim fps [3-stage capture=prefetch]                198.2500                                             1 fps    +0.0%  grab 1.67ms
sim latency p50 [3-stage capture=prefetch]         12.1547                                             1 ms     +0.0%  grab 1.67ms
sim fps [3-stage capture=jit]                     199.0000                                             1 fps    +0.0%  grab 1.67ms
sim latency p50 [3-stage capture=jit]               7.7663                                             1 ms     +0.0%  grab 1.67ms
sim serial fps @ capture=144                      144.0000                                             1 fps    +0.0%  latency p50 4.69 ms
sim serial fps @ capture=360                      213.2500                                             1 fps    +0.0%  latency p50 6.07 ms
sim serial fps @ capture=600                      213.2500                                             1 fps    +0.0%  latency p50 5.52 ms

cpu tail
--------

metric                                                 min      mean       p50       p95       p99     n unit vs base  note
---------------------------------------------------------------------------------------------------------------------------
tracker.step[ultralytics]                           0.3915    0.4083    0.4067    0.4296    0.4320     6 ms     -3.7%  fastest of 6 trials
tracker.step[ultralytics_vectorized]                0.4166    0.4352    0.4325    0.4551    0.4578     6 ms     -1.7%  fastest of 6 trials
tracker.step[cpp]                                   0.0075    0.0102    0.0100    0.0131    0.0131     6 ms     +1.4%  fastest of 6 trials
targetselector.update_prev_detection                0.0251    0.0302    0.0282    0.0401    0.0426     6 ms     +0.6%  fastest of 6 trials
targetselector.get_deltas                           0.0848    0.0886    0.0869    0.0950    0.0957     6 ms     -2.4%  fastest of 6 trials
targetselector.lead_target                          0.0140    0.0154    0.0146    0.0188    0.0198     6 ms     -1.9%  fastest of 6 trials
targetselector.buffer.ordered                       0.0048    0.0049    0.0049    0.0050    0.0050     6 ms     +1.8%  fastest of 6 trials
crosshair_rows_to_tracked                           0.0033    0.0034    0.0033    0.0036    0.0036     6 ms     -4.1%  fastest of 6 trials
mousemover.move_mouse_humanized (win32 stubbed)     0.0027    0.0031    0.0029    0.0038    0.0040     6 ms     -0.3%  fastest of 6 trials
_Slot.put+get roundtrip                             0.0011    0.0011    0.0011    0.0011    0.0011     6 ms     +3.9%  fastest of 6 trials
inputdetector.is_rmb_pressed (GetAsyncKeyState)     0.0003    0.0003    0.0003    0.0003    0.0003     6 ms     +1.1%  fastest of 6 trials

pipeline
--------

metric                                                 min      mean       p50       p95       p99     n unit vs base  note
---------------------------------------------------------------------------------------------------------------------------
pipeline.run [base + hsv]                           3.1702    3.2149    3.2145    3.2642    3.2682     6 ms     -5.3%  no ADS / no lock
pipeline.run [precision_sr + hsv]                   4.3591    4.4712    4.4523    4.5727    4.5827     6 ms     -6.1%  ADS + small lock
trt base inference_cp                               2.8527    2.9003    2.8957    2.9607    2.9673     6 ms     -7.0%  detector only, no hsv
hsv marginal cost on top of trt                     0.1952    0.2915    0.2776    0.3991    0.4043     6 ms         —  paired difference

latency
-------

metric                                                 min      mean       p50       p95       p99     n unit vs base  note
---------------------------------------------------------------------------------------------------------------------------
e2e: frame -> mouse move (serial)                   2.8384    3.2249    3.1341    3.7778    4.2671   200 ms     +6.5%  excludes camera.grab; per-frame, not batched
e2e fps (steady state)                            352.3112                                             1 fps    -6.1%  1000 / fastest frame; the comparable one
e2e fps (median, as-observed)                     319.0709                                             1 fps    +2.8%  what this run actually did, desktop contention included

hsv
---

metric                                                 min      mean       p50       p95       p99     n unit vs base  note
---------------------------------------------------------------------------------------------------------------------------
hsv.detect[simple]                                  0.5810    0.6130    0.6162    0.6353    0.6355     6 ms    -11.3%  240x240 roi, fastest of 6
hsv.detect[weighted_center]                         0.5654    0.5877    0.5747    0.6404    0.6568     6 ms     -2.0%  240x240 roi, fastest of 6
hsv.detect[connected]                               1.7664    1.8149    1.7872    1.9358    1.9738     6 ms     -9.2%  240x240 roi, fastest of 6
hsv.detect[heuristic_spam]                          0.2198    0.2280    0.2241    0.2443    0.2477     6 ms     -2.6%  240x240 roi, fastest of 6
hsv.fused_pipeline (launch, no sync)                0.0996    0.1007    0.1003    0.1025    0.1029     6 ms     +0.0%  one kernel, no D2H
hsv.fused_red_mask_suppress                         0.0291    0.0304    0.0297    0.0335    0.0344     6 ms     -0.6%  mask + row-suppress only
one host<-device sync (float())                     0.0315    0.0341    0.0338    0.0368    0.0369     6 ms     -4.2%  baseline for sync-count changes

gpu misc
--------

metric                                                 min      mean       p50       p95       p99     n unit vs base  note
---------------------------------------------------------------------------------------------------------------------------
model._preprocess_cp                                0.0221    0.0238    0.0230    0.0274    0.0285     6 ms     -8.8%  640x640 hwc u8 -> nchw f32
frame.copy() (640x640x3 u8)                         0.0281    0.0287    0.0286    0.0293    0.0293     6 ms     -1.8%  async capture ownership
D2H 3 float64 (.get())                              0.0314    0.0330    0.0324    0.0363    0.0372     6 ms     -0.5%  hsv result transfer
