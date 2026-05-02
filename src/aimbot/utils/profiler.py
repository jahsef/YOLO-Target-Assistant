import time


class Profiler:
    """Per-frame inline profiler.

    Usage:
        prof = Profiler()
        while True:
            prof.reset()
            ...work...
            prof.time("grab")
            ...work...
            prof.time("inference")
            ...
            prof.print()

    reset() snapshots the reference time. each time(desc) records the elapsed
    nanoseconds since the previous mark (or reset if no prior mark this frame),
    then updates the reference. print() dumps the full ordered list as ms.
    """

    def __init__(self):
        self._marks: list[tuple[str, int]] = []  # (description, delta_ns)
        self._last_ns: int = time.perf_counter_ns()

    def reset(self):
        self._marks.clear()
        self._last_ns = time.perf_counter_ns()

    def time(self, description: str):
        now = time.perf_counter_ns()
        self._marks.append((description, now - self._last_ns))
        self._last_ns = now

    def print(self):
        if not self._marks:
            return
        total_ms = sum(d for _, d in self._marks) / 1e6
        parts = [f"{desc}={d / 1e6:.3f}ms" for desc, d in self._marks]
        print(f"[prof] total={total_ms:.3f}ms | " + " | ".join(parts))
