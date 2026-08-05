import threading
import time

import win32api
import win32con
import random
import math
from ..utils.utils import log

class MouseMover:
    def __init__(self,overall_sens:float,sens_scaling:float,max_deltas:int,jitter_strength:float,overshoot_strength:float,overshoot_chance:float):
        """
        Args:
            overall_sens: sensitivity 0-2 (more than 2 not recommended)
            max_deltas: max pixels to move
        """
        self.overall_sens = overall_sens
        self.sens_scaling = sens_scaling
        self.max_deltas = max_deltas
        self.jitter_strength = jitter_strength
        self.overshoot_strength = overshoot_strength
        self.overshoot_chance = overshoot_chance
    
    def _shape_delta(self, dx: float, dy: float) -> tuple[float, float]:
        """raw deltas -> sensitivity-scaled, humanized floats. No rounding, no output,
        so the threaded variant can queue the exact value instead of a rounded one."""
        return self._humanize_movement(self._scale_delta(dx), self._scale_delta(dy))

    def move_mouse_humanized(self,dx:float,dy:float) -> tuple[int, int]:
        """
        takes raw deltas, scales and humanizes them and moves mouse, returns scaled deltas too
        """
        humanized_xy = self._shape_delta(dx, dy)
        round_x, round_y = round(humanized_xy[0]), round(humanized_xy[1])
        log(f'moving mouse: {round_x,round_y}', "DEBUG")
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, round_x, round_y)
        return (round_x ,round_y)

    def pull(self, dx: float, dy: float) -> None:
        """Movement that is not aim (recoil), in raw screen px — no sensitivity curve,
        since it compensates what the gun did on screen rather than chasing a target."""
        x, y = round(dx), round(dy)
        if x or y:
            log(f'pull: {x},{y}', "DEBUG")
            win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, x, y)

    def _humanize_movement(self,dx:float, dy:float) -> tuple[float,float]:
        
        jitter = self.jitter_strength * (abs(dx) + abs(dy))
        dx += random.uniform(-jitter, jitter)
        dy += random.uniform(-jitter, jitter)
        
        if random.random() < self.overshoot_chance:
            dx *= self.overshoot_strength
            dy *= self.overshoot_strength
        
        return (dx, dy)


    def _scale_delta(self, delta):
        """
        - Low deltas (near zero) are scaled minimally (close to raw).
        - Higher sensitivity amplifies scaling, but does NOT invert behavior.
        - Smooth exponential transition.
        """
        x = abs(delta) / self.max_deltas  # Normalized delta (0 to 1)
        
        # How much to blend toward scaling (tune for desired curve)
        blend = 1 - math.exp(-x * 8)  # curve steepness (coeff on x). higher = steeper.
	# steep curve (ie 8) has near 1.0 sens when deltas are small (desired, do not want fine tune deltas to be thrown away)
	# high steepness results in lower sens on the high end of deltas (prevents wild flicking, pseudo accel decel patterns)
	# https://www.desmos.com/calculator/7ehwmgth7v
        
        # Apply sensitivity (higher sensitivity = more scaling, but not inverted)
        
        return self.overall_sens * delta * (1.0 + (self.sens_scaling - 1.0) * blend)

class DeltaDrain:
    """Holds pending mouse movement and hands it out in small integer steps.

    Per tick: take `drain_alpha` of what is pending, but never less than
    `min_step_px`. Benched against plain exponential, linear rate and a deadline
    scheme (src/tests/bench_drain.py):

      - the exponential part makes a big jump start soft instead of teleporting
      - the floor is what removes lag. In steady state the buffer sits small, so the
        exponential term goes tiny and the floor takes over, draining at
        min_step_px * poll_hz px/s. That outruns anything the aim loop produces, so a
        backlog cannot build. Plain exponential settles at a permanent 4-11 ms of lag;
        with the floor it is 0.00 ms.
      - nothing here depends on WHEN deltas arrive, so irregular frame timing costs
        nothing.

    `output_alpha` is a second EMA over the emitted step, so speed ramps up rather
    than starting at full tilt (a real hand accelerates). The floor overrides it, so
    the tail still terminates. 1.0 disables it.

    Both alphas are per tick, so the feel is tied to poll_hz. Change one, retune the
    other.

    Two details that are easy to get wrong and both silently eat movement:
      - the floor applies to the VECTOR magnitude, with both axes scaled by the same
        factor. A per-axis floor would drain (10, 5) at 45 degrees instead of 63.
      - mouse_event only takes ints, and at ~650 Hz most steps are under a pixel.
        Truncating instead of carrying the remainder loses ~45% of the movement.
    """

    def __init__(self, drain_alpha: float, min_step_px: float, output_alpha: float = 1.0):
        if not 0.0 < drain_alpha <= 1.0:
            raise ValueError(f"drain_alpha must be in (0, 1], got {drain_alpha}")
        if min_step_px < 0:
            raise ValueError(f"min_step_px must be >= 0, got {min_step_px}")
        if not 0.0 < output_alpha <= 1.0:
            raise ValueError(f"output_alpha must be in (0, 1], got {output_alpha}")
        self.drain_alpha = drain_alpha
        self.min_step_px = min_step_px
        self.output_alpha = output_alpha
        self._lock = threading.Lock()
        self._px = 0.0
        self._py = 0.0
        self._carry_x = 0.0
        self._carry_y = 0.0
        # last emitted step, so output speed ramps instead of starting at full tilt
        self._ema_x = 0.0
        self._ema_y = 0.0

    def add(self, dx: float, dy: float) -> None:
        with self._lock:
            self._px += dx
            self._py += dy

    @property
    def pending(self) -> tuple[float, float]:
        with self._lock:
            return self._px, self._py

    def tick(self) -> tuple[int, int]:
        """Advance one tick and return the integer step to emit."""
        with self._lock:
            norm = math.hypot(self._px, self._py)
            if norm <= 1e-9:
                # nothing pending: let the output EMA fall back to rest, so the next
                # burst ramps up again instead of resuming at the old speed
                self._ema_x = self._ema_y = 0.0
            else:
                ux, uy = self._px / norm, self._py / norm
                want = norm * self.drain_alpha
                take_x, take_y = ux * want, uy * want

                a = self.output_alpha
                if a < 1.0:
                    take_x = self._ema_x + a * (take_x - self._ema_x)
                    take_y = self._ema_y + a * (take_y - self._ema_y)

                # linear phase overrides the smoothing, so the tail still terminates
                floor = min(norm, self.min_step_px)
                if math.hypot(take_x, take_y) < floor:
                    take_x, take_y = ux * floor, uy * floor

                if math.hypot(take_x, take_y) > norm:  # never exceed what is pending
                    take_x, take_y = ux * norm, uy * norm

                # EMA follows what actually went out, including a floored step
                self._ema_x, self._ema_y = take_x, take_y
                self._px -= take_x
                self._py -= take_y
                self._carry_x += take_x
                self._carry_y += take_y
            step_x = math.trunc(self._carry_x)
            step_y = math.trunc(self._carry_y)
            self._carry_x -= step_x
            self._carry_y -= step_y
            return step_x, step_y


class ThreadedMouseMover(MouseMover):
    """MouseMover that queues movement for a high-rate thread instead of emitting it.

    A frame's worth of aim can be tens of pixels; sent as one mouse_event that is a
    teleport. This spreads it over many small events at poll_hz, which is also the
    natural home for recoil compensation later, since recoil is a function of
    wall-clock time rather than of how many frames the detector managed.

    move_mouse_humanized still returns the deltas it queued, so callers (GUI, lead
    buffer) see the same numbers as before.
    """

    def __init__(self, *args, poll_hz: float = 1000.0, drain_alpha: float = 0.2,
                 min_step_px: float = 2.0, output_alpha: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        if poll_hz <= 0:
            raise ValueError(f"poll_hz must be > 0, got {poll_hz}")
        self.poll_hz = poll_hz
        self.drain = DeltaDrain(drain_alpha=drain_alpha, min_step_px=min_step_px,
                                output_alpha=output_alpha)
        self.ticks = 0
        self._stop = threading.Event()
        self._thread = None

    def move_mouse_humanized(self, dx: float, dy: float) -> tuple[int, int]:
        hx, hy = self._shape_delta(dx, dy)
        self.drain.add(hx, hy)
        log(f'queued mouse: {hx:.2f},{hy:.2f}', "DEBUG")
        return (round(hx), round(hy))

    def pull(self, dx: float, dy: float) -> None:
        self.drain.add(dx, dy)
        log(f'queued pull: {dx:.2f},{dy:.2f}', "DEBUG")

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="mouse", daemon=True)
        self._thread.start()
        log(f"mouse thread started at {self.poll_hz:g} Hz "
            f"(drain {self.drain.drain_alpha:g}, floor {self.drain.min_step_px:g} px, "
            f"output {self.drain.output_alpha:g})", "INFO")

    def _loop(self) -> None:
        # time.sleep, NOT self._stop.wait(). Event.wait goes through the OS wait
        # machinery and gets quantised to the ~15.6ms Windows timer tick, measured at
        # 15.5ms for a 1ms request â€” the thread would run at 65Hz and hand out the
        # movement in a few fat steps, defeating the point. time.sleep uses a
        # high-resolution timer and measures 1.53ms for the same request. Cost is that
        # stop() is noticed up to one period late, which is irrelevant here.
        period = 1.0 / self.poll_hz
        while not self._stop.is_set():
            time.sleep(period)
            self.ticks += 1
            step_x, step_y = self.drain.tick()
            if step_x or step_y:
                win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, step_x, step_y)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
