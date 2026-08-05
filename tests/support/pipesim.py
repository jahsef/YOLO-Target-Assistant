"""Discrete-event simulator for the aimbot loop's threading architectures.

Why not real threads: the effects being compared are 0.2-1.7 ms and Windows' default
timer granularity is ~1-15 ms, so a real-thread sim would measure the timer, not the
design. This models the two resources that actually serialize work — the GIL and the
GPU — and reports throughput plus screenshot->mouse-move latency.

Model, per stage: `gil_ms` needs the GIL exclusively (pure-Python work: tracker,
target selection, the parts of a capture call that don't release). `gpu_ms` needs the
GPU exclusively while the calling thread is blocked but NOT holding the GIL — which is
what cupy/TensorRT/torch actually do (measured: they retain 0.93-1.10 of a competing
thread's throughput).

Latency is measured from the frame's CONTENT timestamp (the capture boundary, not when
grab() returned) to the mouse_event call, so frame staleness counts against every
architecture equally.
"""

import heapq
from collections import deque
from dataclasses import dataclass, field


class Sim:
    def __init__(self):
        self.now = 0.0
        self._q = []
        self._seq = 0

    def _push(self, at, fn):
        self._seq += 1
        heapq.heappush(self._q, (at, self._seq, fn))

    def process(self, gen):
        """Drive a generator that yields wait-objects (or None to reschedule now).
        The waiter's value is sent back in, so `item = yield slot.get()` works."""
        def step(value=None):
            try:
                waiter = gen.send(value)
            except StopIteration:
                return
            if waiter is None:
                self._push(self.now, step)
            else:
                waiter.then(step)
        self._push(self.now, step)

    def timeout(self, dt):
        w = _Waiter()
        self._push(self.now + dt, w.fire)
        return w

    def run(self, until, max_events=20_000_000):
        n = 0
        while self._q and self._q[0][0] <= until:
            at, _, fn = heapq.heappop(self._q)
            self.now = at
            fn()
            n += 1
            if n > max_events:
                raise RuntimeError(
                    f"event cap hit at t={self.now:.6f} of {until} — a process is "
                    f"spinning without advancing time")
        self.now = until


class _Waiter:
    __slots__ = ("_cb", "value")

    def __init__(self):
        self._cb = None
        self.value = None

    def then(self, cb):
        self._cb = cb

    def fire(self, value=None):
        self.value = value
        if self._cb:
            cb, self._cb = self._cb, None
            cb(value)


class Resource:
    """Exclusive, FIFO-queued."""

    def __init__(self, sim):
        self.sim = sim
        self.busy = False
        self._waiting = deque()

    def request(self):
        w = _Waiter()
        if self.busy:
            self._waiting.append(w)
        else:
            self.busy = True
            self.sim._push(self.sim.now, w.fire)
        return w

    def release(self):
        if self._waiting:
            self.sim._push(self.sim.now, self._waiting.popleft().fire)
        else:
            self.busy = False


class Slot:
    """Latest-wins mailbox. A producer overwriting an unread item drops the old one —
    exactly what you want for frames: never process a stale frame when a newer exists."""

    def __init__(self, sim):
        self.sim = sim
        self.item = None
        self.dropped = 0
        self._waiting = deque()
        self._free_waiting = deque()

    def _wake_free(self):
        while self._free_waiting:
            self.sim._push(self.sim.now, self._free_waiting.popleft().fire)

    def put(self, item):
        if self._waiting:
            self.sim._push(self.sim.now, lambda w=self._waiting.popleft(), i=item: w.fire(i))
            self._wake_free()
            return
        if self.item is not None:
            self.dropped += 1
        self.item = item

    def get(self):
        w = _Waiter()
        if self.item is not None:
            item, self.item = self.item, None
            self.sim._push(self.sim.now, lambda: w.fire(item))
            self._wake_free()
        else:
            self._waiting.append(w)
        return w

    def wait_free(self):
        """Block until nothing is sitting unread. Lets a producer avoid doing work
        whose result would only be dropped."""
        w = _Waiter()
        if self.item is None:
            self.sim._push(self.sim.now, w.fire)
        else:
            self._free_waiting.append(w)
        return w


@dataclass
class Costs:
    """All values in milliseconds."""
    capture_period: float = 1000.0 / 600     # new frame available every X ms
    capture_gil: float = 0.0                 # GIL-held part of grab()
    capture_gpu: float = 0.0                 # DMA / copy part
    preprocess_gpu: float = 0.025
    infer_gpu: float = 4.13
    hsv_gpu: float = 0.70                    # queued GPU work
    hsv_gil: float = 0.62                    # host-blocking vote (float() syncs)
    post_gil: float = 0.20                   # tracker + target selection
    mouse_gil: float = 0.043
    mouse_thread_hz: float = 1000.0


@dataclass
class Stats:
    frames: int = 0
    latencies: list = field(default_factory=list)
    dropped: int = 0

    def summary(self, wall_ms):
        lat = sorted(self.latencies)
        n = len(lat)
        pct = lambda p: lat[min(n - 1, int(p * n))] if n else float("nan")
        return {
            "fps": self.frames / (wall_ms / 1000.0),
            "frame_ms": wall_ms / self.frames if self.frames else float("nan"),
            "lat_p50": pct(0.50), "lat_p95": pct(0.95), "lat_p99": pct(0.99),
            "lat_mean": sum(lat) / n if n else float("nan"),
            "lat_min": lat[0] if n else float("nan"),
            "dropped": self.dropped,
        }


def _use(res, sim, dt):
    """Occupy `res` for dt ms."""
    yield res.request()
    yield sim.timeout(dt)
    res.release()


class Rig:
    def __init__(self, costs: Costs):
        self.c = costs
        self.sim = Sim()
        self.gil = Resource(self.sim)
        self.gpu = Resource(self.sim)
        self.stats = Stats()

    # --- shared stage bodies --------------------------------------------------

    def _gpu_section(self):
        c, sim = self.c, self.sim
        if c.preprocess_gpu:
            yield from _use(self.gpu, sim, c.preprocess_gpu)
        yield from _use(self.gpu, sim, c.infer_gpu)
        if c.hsv_gpu:
            yield from _use(self.gpu, sim, c.hsv_gpu)
        if c.hsv_gil:
            yield from _use(self.gil, sim, c.hsv_gil)

    def _post_section(self, t_content, emit=True):
        c, sim = self.c, self.sim
        yield from _use(self.gil, sim, c.post_gil)
        if emit:
            yield from _use(self.gil, sim, c.mouse_gil)
            self.stats.latencies.append(sim.now - t_content)
        self.stats.frames += 1

    def _capture(self, t_content):
        c, sim = self.c, self.sim
        if c.capture_gil:
            yield from _use(self.gil, sim, c.capture_gil)
        if c.capture_gpu:
            yield from _use(self.gpu, sim, c.capture_gpu)

    # Frame boundaries are indexed as integers, never derived from float modulo —
    # `now % period` can leave a 1e-15 residue that makes a wait-for-next-boundary
    # loop creep forward forever instead of advancing.
    def _frame_index(self):
        """Index of the newest frame the camera has produced."""
        return int(self.sim.now / self.c.capture_period + 1e-9)

    def _wait_for_frame_after(self, idx):
        """Sleep until frame `idx + 1` exists. Returns its (index, content timestamp)."""
        nxt = idx + 1
        at = nxt * self.c.capture_period
        if at > self.sim.now:
            yield self.sim.timeout(at - self.sim.now)
        return nxt, nxt * self.c.capture_period

    # --- architectures --------------------------------------------------------

    def serial(self, until):
        """Today: one thread does grab -> gpu -> post -> mouse."""
        sim = self.sim

        def loop():
            last = -1
            while True:
                idx = self._frame_index()
                if idx <= last:  # no new frame yet; wait for the next boundary
                    idx, ts = yield from self._wait_for_frame_after(last)
                else:
                    ts = idx * self.c.capture_period
                last = idx
                yield from self._capture(ts)
                yield from self._gpu_section()
                yield from self._post_section(ts)

        sim.process(loop())
        sim.run(until)
        return self.stats.summary(until)

    def two_stage(self, until):
        """Capture thread fills a latest-wins slot; worker does gpu + post."""
        sim = self.sim
        slot = Slot(sim)

        def capture_thread():
            idx = -1
            while True:
                idx, ts = yield from self._wait_for_frame_after(max(idx, self._frame_index() - 1))
                yield from self._capture(ts)
                slot.put(ts)

        def worker():
            while True:
                ts = yield slot.get()
                yield from self._gpu_section()
                yield from self._post_section(ts)

        sim.process(capture_thread())
        sim.process(worker())
        sim.run(until)
        self.stats.dropped = slot.dropped
        return self.stats.summary(until)

    def three_stage(self, until, mouse_thread=False, capture_mode="free"):
        """capture | gpu | post(+mouse). Optional dedicated mouse thread.

        capture_mode picks when the capture thread does its grab:
          "free"     — free-run, always holding the newest frame. Freshest possible
                       input, but every grab the consumer never reaches is wasted
                       GPU/GIL, which sinks throughput once grab gets expensive.
          "prefetch" — grab as soon as the slot drains. No waste, but the frame then
                       waits out the whole GPU stage, so it arrives stale.
          "jit"      — grab as soon as the slot drains MINUS the time the grab takes,
                       so it lands just as the GPU stage comes free. No waste and no
                       staleness; needs a running estimate of the GPU stage duration.
        """
        sim = self.sim
        frames = Slot(sim)
        results = Slot(sim)
        deltas = Slot(sim)
        gpu_est = [self.c.preprocess_gpu + self.c.infer_gpu + self.c.hsv_gpu + self.c.hsv_gil]
        grab_cost = self.c.capture_gil + self.c.capture_gpu

        def capture_thread():
            idx = -1
            while True:
                if capture_mode in ("prefetch", "jit"):
                    yield frames.wait_free()
                if capture_mode == "jit":
                    lag = gpu_est[0] - grab_cost
                    if lag > 0:
                        yield sim.timeout(lag)
                idx, ts = yield from self._wait_for_frame_after(max(idx, self._frame_index() - 1))
                yield from self._capture(ts)
                frames.put(ts)

        def gpu_thread():
            while True:
                ts = yield frames.get()
                t0 = sim.now
                yield from self._gpu_section()
                gpu_est[0] = 0.8 * gpu_est[0] + 0.2 * (sim.now - t0)
                results.put(ts)

        def post_thread():
            while True:
                ts = yield results.get()
                yield from self._post_section(ts, emit=not mouse_thread)
                if mouse_thread:
                    deltas.put(ts)

        def mouse_loop():
            period = 1000.0 / self.c.mouse_thread_hz
            while True:
                yield sim.timeout(period)
                ts = deltas.item
                if ts is None:
                    continue
                deltas.item = None
                yield from _use(self.gil, sim, self.c.mouse_gil)
                self.stats.latencies.append(sim.now - ts)

        sim.process(capture_thread())
        sim.process(gpu_thread())
        sim.process(post_thread())
        if mouse_thread:
            sim.process(mouse_loop())
        sim.run(until)
        self.stats.dropped = frames.dropped + results.dropped
        return self.stats.summary(until)


ARCHITECTURES = {
    "serial (today)": lambda rig, t: rig.serial(t),
    "2-stage (capture thread)": lambda rig, t: rig.two_stage(t),
    "3-stage": lambda rig, t: rig.three_stage(t),
    "3-stage + mouse thread": lambda rig, t: rig.three_stage(t, mouse_thread=True),
}


def run_all(costs: Costs, until_ms=4000.0):
    return {name: fn(Rig(costs), until_ms) for name, fn in ARCHITECTURES.items()}
