"""Threading-architecture projections, driven by this machine's measured stage costs.

The simulator (tests/support/pipesim.py) models the two things that actually serialize
the loop — the GIL and the GPU — so these numbers say what a threading change would
buy BEFORE anyone writes one. Recorded alongside the real measurements so the
projection can be checked against reality over time.
"""

import numpy as np
import pytest

from tests.support.pipesim import Costs, Rig

pytestmark = pytest.mark.perf

UNTIL_MS = 4000.0

# Measured on an RTX 5080 (see tests/results/latest.md for the live numbers).
# capture_period: the user-reported ~600 fps capture rate.
# capture_gil/gpu: grab() is cheap on this setup; the sensitivity sweep below covers
#   the case where it isn't, since it can't be measured on an idle desktop (DXGI only
#   produces frames when screen content actually changes).
BASE = dict(capture_period=1000.0 / 600, capture_gil=0.031, capture_gpu=0.019,
            preprocess_gpu=0.025, infer_gpu=4.13, post_gil=0.20, mouse_gil=0.043)
HSV_OLD = dict(hsv_gpu=0.73, hsv_gil=0.62)   # torch chain + 3 host syncs
HSV_NEW = dict(hsv_gpu=0.20, hsv_gil=0.04)   # fused kernel + 1 host sync

ARCHES = ["serial", "two_stage", "three_stage"]


def run(costs, arch, **kw):
    if arch == "three_stage":
        kw.setdefault("capture_mode", "jit")
    return getattr(Rig(costs), arch)(UNTIL_MS, **kw)


class TestArchitectureProjection:
    @pytest.mark.parametrize("arch", ARCHES)
    def test_projected(self, perf, arch):
        r = run(Costs(**BASE, **HSV_NEW), arch)
        perf.record_value(f"sim fps [{arch}]", r["fps"], unit="fps",
                          group="architecture (simulated)", note="fused hsv")
        perf.record_value(f"sim latency p50 [{arch}]", r["lat_p50"], unit="ms",
                          group="architecture (simulated)", note="screenshot -> mouse")

    def test_mouse_thread_variant(self, perf):
        r = run(Costs(**BASE, **HSV_NEW), "three_stage", mouse_thread=True)
        perf.record_value("sim fps [3-stage + mouse thread]", r["fps"], unit="fps",
                          group="architecture (simulated)")
        perf.record_value("sim latency p50 [3-stage + mouse thread]", r["lat_p50"],
                          unit="ms", group="architecture (simulated)")

    def test_hsv_fusion_gain(self, perf):
        """What the kernel fusion bought, holding the architecture fixed."""
        old = run(Costs(**BASE, **HSV_OLD), "serial")
        new = run(Costs(**BASE, **HSV_NEW), "serial")
        perf.record_value("sim fps [serial, pre-fusion hsv]", old["fps"], unit="fps",
                          group="architecture (simulated)", note="torch chain, 3 syncs")
        perf.record_value("sim hsv-fusion speedup", new["fps"] / old["fps"], unit="x",
                          group="architecture (simulated)", note="serial vs serial")
        perf.record_value("sim hsv-fusion latency saved",
                          old["lat_p50"] - new["lat_p50"], unit="ms",
                          group="architecture (simulated)")

    def test_threading_tradeoff(self, perf):
        c = Costs(**BASE, **HSV_NEW)
        s, t = run(c, "serial"), run(c, "three_stage")
        perf.record_value("sim 3-stage fps gain", t["fps"] / s["fps"], unit="x",
                          group="architecture (simulated)")
        perf.record_value("sim 3-stage latency cost", t["lat_p50"] - s["lat_p50"],
                          unit="ms", group="architecture (simulated)",
                          note="positive = worse than serial")


class TestSensitivity:
    @pytest.mark.parametrize("grab_ms", [0.05, 0.5, 1.67])
    def test_capture_cost_sensitivity(self, perf, grab_ms):
        """How much a capture thread is worth depends entirely on what grab() costs —
        the one input that can't be measured on an idle desktop."""
        c = Costs(**{**BASE, "capture_gil": grab_ms * 0.62, "capture_gpu": grab_ms * 0.38},
                  **HSV_NEW)
        s, t = run(c, "serial"), run(c, "three_stage")
        perf.record_value(f"sim 3-stage gain @ grab={grab_ms}ms", t["fps"] / s["fps"],
                          unit="x", group="architecture (simulated)",
                          note=f"serial {s['fps']:.0f} -> 3-stage {t['fps']:.0f} fps")

    @pytest.mark.parametrize("mode", ["free", "prefetch", "jit"])
    def test_capture_mode(self, perf, mode):
        """When the capture thread does its grab, at a grab cost where it matters."""
        c = Costs(**{**BASE, "capture_gil": 1.03, "capture_gpu": 0.64}, **HSV_NEW)
        r = Rig(c).three_stage(UNTIL_MS, capture_mode=mode)
        perf.record_value(f"sim fps [3-stage capture={mode}]", r["fps"], unit="fps",
                          group="architecture (simulated)", note="grab 1.67ms")
        perf.record_value(f"sim latency p50 [3-stage capture={mode}]", r["lat_p50"],
                          unit="ms", group="architecture (simulated)", note="grab 1.67ms")

    @pytest.mark.parametrize("cap_fps", [144, 360, 600])
    def test_capture_rate_sensitivity(self, perf, cap_fps):
        c = Costs(**{**BASE, "capture_period": 1000.0 / cap_fps}, **HSV_NEW)
        s = run(c, "serial")
        perf.record_value(f"sim serial fps @ capture={cap_fps}", s["fps"], unit="fps",
                          group="architecture (simulated)",
                          note=f"latency p50 {s['lat_p50']:.2f} ms")
