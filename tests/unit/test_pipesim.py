"""The architecture simulator's own correctness.

It's used to justify design decisions, so it gets checked against hand-computable
cases: a serial loop must cost the sum of its stages, a pipelined loop must cost the
busiest resource, and neither may invent throughput the GIL/GPU can't support.
"""

import pytest

from tests.support.pipesim import Costs, Resource, Rig, Sim, Slot

FAST_CAPTURE = 0.25  # ms; well below any stage cost so capture is never the limiter


def costs(**kw):
    base = dict(capture_period=FAST_CAPTURE, capture_gil=0.0, capture_gpu=0.0,
                preprocess_gpu=1.0, infer_gpu=2.0, hsv_gpu=0.5, hsv_gil=0.0,
                post_gil=0.25, mouse_gil=0.05)
    base.update(kw)
    return Costs(**base)


class TestCore:
    def test_timeouts_advance_the_clock_in_order(self):
        sim = Sim()
        seen = []

        def proc(name, dt):
            yield sim.timeout(dt)
            seen.append((name, round(sim.now, 6)))

        sim.process(proc("b", 5.0))
        sim.process(proc("a", 1.0))
        sim.run(10.0)
        assert seen == [("a", 1.0), ("b", 5.0)]

    def test_resource_is_exclusive(self):
        sim = Sim()
        res = Resource(sim)
        overlaps = []
        state = {"held": False}

        def proc():
            yield res.request()
            overlaps.append(state["held"])
            state["held"] = True
            yield sim.timeout(1.0)
            state["held"] = False
            res.release()

        for _ in range(3):
            sim.process(proc())
        sim.run(10.0)
        assert overlaps == [False, False, False]
        assert sim.now >= 3.0, "three 1ms holds cannot finish in under 3ms"

    def test_slot_is_latest_wins(self):
        sim = Sim()
        slot = Slot(sim)
        got = []

        def producer():
            for i in range(3):
                slot.put(i)
                yield sim.timeout(1.0)

        def consumer():
            yield sim.timeout(0.5)
            while True:
                item = yield slot.get()
                got.append(item)
                yield sim.timeout(10.0)  # far slower than the producer

        sim.process(producer())
        sim.process(consumer())
        sim.run(40.0)
        assert got[0] == 0
        assert slot.dropped >= 1

    def test_values_flow_back_into_the_generator(self):
        """Regression: the scheduler used next() instead of send(), so every
        `yield slot.get()` evaluated to None and latency maths silently broke."""
        sim = Sim()
        slot = Slot(sim)
        got = []

        def consumer():
            got.append((yield slot.get()))

        sim.process(consumer())
        slot.put("payload")
        sim.run(5.0)
        assert got == ["payload"]

    def test_spinning_process_hits_the_event_cap(self):
        sim = Sim()

        def spin():
            while True:
                yield sim.timeout(0.0)

        sim.process(spin())
        with pytest.raises(RuntimeError, match="spinning"):
            sim.run(10.0, max_events=5000)


class TestArchitectures:
    def test_serial_costs_the_sum_of_its_stages(self):
        r = Rig(costs()).serial(2000.0)
        # 1.0 + 2.0 + 0.5 + 0.25 + 0.05 = 3.80, plus up to one capture period of wait
        assert 3.80 <= r["frame_ms"] <= 3.80 + FAST_CAPTURE

    def test_three_stage_costs_the_busiest_resource(self):
        r = Rig(costs()).three_stage(2000.0)
        # gpu demand 1.0 + 2.0 + 0.5 = 3.50; gil demand only 0.30
        assert r["frame_ms"] == pytest.approx(3.50, abs=0.05)

    def test_pipelining_cannot_beat_the_gpu_bound(self):
        c = costs(post_gil=5.0)  # make the CPU tail dominant instead
        r = Rig(c).three_stage(2000.0)
        assert r["frame_ms"] == pytest.approx(5.0, abs=0.1), "now GIL-bound"

    def test_latency_is_never_less_than_the_work(self):
        for arch in ("serial", "two_stage", "three_stage"):
            r = getattr(Rig(costs()), arch)(2000.0)
            assert r["lat_min"] >= 3.80 - 1e-9, f"{arch} reported impossible latency"

    def test_three_stage_trades_latency_for_throughput(self):
        c = costs()
        s = Rig(c).serial(2000.0)
        t = Rig(c).three_stage(2000.0)
        assert t["fps"] > s["fps"]
        assert t["lat_p50"] > s["lat_p50"], "extra pipeline stages cost latency"

    def test_capture_rate_caps_everything(self):
        c = costs(capture_period=20.0)  # 50 fps camera, ~4ms of work
        for arch in ("serial", "three_stage"):
            r = getattr(Rig(c), arch)(4000.0)
            assert r["fps"] <= 50.5, f"{arch} invented frames the camera never produced"

    def test_mouse_thread_adds_latency_without_adding_fps(self):
        c = costs()
        base = Rig(c).three_stage(2000.0)
        mouse = Rig(c).three_stage(2000.0, mouse_thread=True)
        assert mouse["fps"] == pytest.approx(base["fps"], rel=0.02)
        assert mouse["lat_p50"] > base["lat_p50"]

    def test_free_running_capture_wastes_an_expensive_grab(self):
        """With a 600 fps camera feeding a ~200 fps pipeline, a free-running capture
        thread pays grab 3x per consumed frame and spends the difference on GPU/GIL the
        detect stage wanted. Once grab is expensive that makes threading a net LOSS."""
        c = costs(capture_period=1000 / 600, capture_gil=1.03, capture_gpu=0.64,
                  preprocess_gpu=0.025, infer_gpu=4.13, hsv_gpu=0.20, hsv_gil=0.04,
                  post_gil=0.20)
        serial = Rig(c).serial(4000.0)["fps"]
        free = Rig(c).three_stage(4000.0, capture_mode="free")["fps"]
        jit = Rig(c).three_stage(4000.0, capture_mode="jit")["fps"]
        assert free < serial, "free-running capture should lose to no threading at all"
        assert jit > serial * 1.2, "grabbing just-in-time should recover the waste"

    def test_jit_capture_keeps_prefetch_throughput_without_the_staleness(self):
        """prefetch grabs the moment the slot drains, so the frame ages through the
        whole GPU stage. jit delays the grab to land as the stage frees up."""
        c = costs(capture_period=1000 / 600, capture_gil=1.03, capture_gpu=0.64,
                  preprocess_gpu=0.025, infer_gpu=4.13, hsv_gpu=0.20, hsv_gil=0.04,
                  post_gil=0.20)
        pre = Rig(c).three_stage(4000.0, capture_mode="prefetch")
        jit = Rig(c).three_stage(4000.0, capture_mode="jit")
        assert jit["fps"] == pytest.approx(pre["fps"], rel=0.05)
        assert jit["lat_p50"] < pre["lat_p50"] - 3.0

    def test_capture_mode_gain_grows_with_grab_cost(self):
        """The whole point of a capture thread: the more grab costs, the more there is
        to overlap. If a mode gets relatively worse as grab grows, it's wasting work."""
        gains = []
        for grab in (0.05, 0.5, 1.67, 4.0):
            c = costs(capture_period=1000 / 600, capture_gil=grab * 0.62,
                      capture_gpu=grab * 0.38, preprocess_gpu=0.025, infer_gpu=4.13,
                      hsv_gpu=0.20, hsv_gil=0.04, post_gil=0.20)
            s = Rig(c).serial(4000.0)["fps"]
            j = Rig(c).three_stage(4000.0, capture_mode="jit")["fps"]
            gains.append(j / s)
        assert gains == sorted(gains), gains
        assert gains[0] > 1.0

    def test_expensive_grab_hurts_the_serial_loop_most(self):
        cheap = costs(capture_gil=0.02, capture_gpu=0.01)
        dear = costs(capture_gil=1.03, capture_gpu=0.64)
        s_cheap = Rig(cheap).serial(3000.0)["fps"]
        s_dear = Rig(dear).serial(3000.0)["fps"]
        t_dear = Rig(dear).three_stage(3000.0, capture_mode="jit")["fps"]
        assert s_dear < s_cheap
        assert t_dear > s_dear, "a capture thread should absorb an expensive grab"
