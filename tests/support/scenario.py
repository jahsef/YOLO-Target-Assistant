"""Closed-loop behavioural harness.

The unit tests check single frames in isolation, which cannot answer the questions that
actually matter: does it converge, does it chase a mover, does it recover when the
detector goes blind. Those need the feedback loop — moving the mouse moves the view, so
the target's screen position changes in response to the aimbot's own output.

There is no ground truth for "aimed correctly", so scenarios assert coarse properties:
error shrinks, error stays small while chasing, aim recovers within N frames. Crude, but
it catches whole-system failures (aiming the wrong way, locking onto nothing, never
converging) that per-function tests cannot see.

World coordinates are the screen positions the targets WOULD have if the player never
moved. `view` accumulates every mouse delta actually issued; observed position is
world - view. Aiming right moves the view right, which slides targets left on screen.
"""

import threading

import numpy as np
from ultralytics.utils.ops import xyxy2xywh

from src.aimbot import bootstrap
from src.aimbot.aimbot import Aimbot
from src.aimbot.data_parsing import targetselector
from src.aimbot.engine.tracker_adapter import crosshair_rows_to_tracked
from tests.support.fakes import FakeFPS
from tests.support.loop_harness import FakeInput

CAPTURE = 640
CENTER = CAPTURE // 2


class Target:
    """Ground-truth target. `path(frame)` returns its (x, y) in world coords."""

    def __init__(self, path, w=40, h=80, conf=0.9, cls=0, visible_from=0, visible_to=10**9):
        self.path = path
        self.w, self.h, self.conf, self.cls = w, h, conf, cls
        self.visible_from, self.visible_to = visible_from, visible_to

    def box(self, frame, view):
        x, y = self.path(frame)
        x -= view[0]
        y -= view[1]
        return [x - self.w / 2, y - self.h / 2, x + self.w / 2, y + self.h / 2,
                self.conf, self.cls]

    def visible(self, frame):
        return self.visible_from <= frame < self.visible_to


def still(x, y):
    return lambda f: (x, y)


def linear(x0, y0, dx=0.0, dy=0.0):
    return lambda f: (x0 + dx * f, y0 + dy * f)


class ScriptedPipeline:
    """Stands in for DetectionPipeline, reproducing the routing decision that governs
    what the detector can SEE.

    The consequence that matters behaviourally: on the sr path the base model is
    skipped, so anything outside the crop is invisible that frame. Mirrors
    DetectionPipeline._run_engine_path.
    """

    def __init__(self, targets, crop=80, bb_thresh=48, hysteresis=2,
                 reticle=None):
        self.targets = targets
        self.crop, self.bb_thresh, self.hysteresis = crop, bb_thresh, hysteresis
        self.reticle = reticle          # (x, y) in view coords, or None
        self.frame = 0
        self.route = []                 # "sr" / "base" per frame

    def _visible_boxes(self, view):
        return [t.box(self.frame, view) for t in self.targets if t.visible(self.frame)]

    def run(self, view, ads, locked, locked_lifetime):
        boxes = self._visible_boxes(view)
        route = "base"
        if ads and locked is not None:
            side = max(locked[2] - locked[0], locked[3] - locked[1])
            if side < self.bb_thresh and locked_lifetime <= self.hysteresis:
                route = "sr"
        if route == "sr":
            cx = (locked[0] + locked[2]) * 0.5
            cy = (locked[1] + locked[3]) * 0.5
            x0 = max(0, min(CAPTURE - self.crop, round(cx - self.crop / 2)))
            y0 = max(0, min(CAPTURE - self.crop, round(cy - self.crop / 2)))
            inside = []
            for b in boxes:
                bx, by = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                if x0 <= bx < x0 + self.crop and y0 <= by < y0 + self.crop:
                    inside.append(b)
            boxes = inside
        self.route.append(route)

        bypass = np.empty((0, 6), dtype=np.float32)
        if self.reticle is not None:
            rx, ry = self.reticle
            bypass = np.array([[rx - 32, ry - 32, rx + 32, ry + 32, 1.0, 2]], dtype=np.float32)
        dets = np.asarray(boxes, dtype=np.float32).reshape(-1, 6)
        return dets, bypass


class Trace:
    """Per-frame record of what the system saw and did."""

    def __init__(self):
        self.rows = []

    def add(self, **kw):
        self.rows.append(kw)

    def col(self, key):
        return [r[key] for r in self.rows]

    def error(self):
        """Distance from the aim origin to the primary target, per frame."""
        return np.asarray([r["error"] for r in self.rows], dtype=np.float64)

    def settled_error(self, after):
        return self.error()[after:]


def run_scenario(cfg, pipeline, frames=30, ads=True, active=True, primary=0):
    """Drive the real Aimbot stages over `frames`, closing the loop through `view`.
    Returns a Trace."""
    fps = FakeFPS(144.0)
    ts = targetselector.TargetSelector(cfg=cfg, detection_window_dim=(CAPTURE, CAPTURE),
                                       screen_hw=(1440, 2560), fps_tracker=fps)
    bot = object.__new__(Aimbot)
    bot.cfg = cfg
    bot.target_selector = ts
    bot.mousemover = bootstrap.create_mousemover(cfg)
    bot.tracker = bootstrap.create_tracker(cfg, ".engine")
    bot.fps_tracker = fps
    bot.inputdetector = FakeInput(rmb=ads)
    bot.gui_manager = None
    bot._frame_count = 0
    bot._route_lock = threading.Lock()
    bot._stop = threading.Event()
    bot._worker_error = None
    bot._detect_ema = 0.0
    bot._grab_ema = 0.0
    bot._starve_ema = 0.0
    bot._lmb_was_down = False

    view = np.zeros(2, dtype=np.float64)
    trace = Trace()
    for f in range(frames):
        pipeline.frame = f
        with bot._route_lock:
            locked = ts._prev_detection
            lifetime = ts._prev_detection_lifetime
        dets, bypass = pipeline.run(view, ads, locked, lifetime)

        results = dets.copy()
        if len(results):
            results[:, 0:4] = xyxy2xywh(results[:, 0:4])
        tracked = bot._stage_track(results, bypass)
        raw, scaled = bot._stage_aim(tracked, active)

        # close the loop: the mouse moved, so the view moved
        view += np.asarray(scaled, dtype=np.float64)

        tgt = pipeline.targets[primary]
        if tgt.visible(f):
            tx, ty = tgt.path(f)
            crosshair = ts._get_crosshair(tracked) if len(tracked) else ts.detection_window_center
            ex = tx - view[0] - crosshair[0]
            ey = ty - view[1] - crosshair[1]
            err = float(np.hypot(ex, ey))
        else:
            ex = ey = err = float("nan")

        trace.add(frame=f, raw=raw, scaled=scaled, view=view.copy(), error=err,
                  err_x=float(ex), err_y=float(ey),
                  route=pipeline.route[-1], n_dets=len(dets), n_tracked=len(tracked))
    return trace


def geometry_cfg(cfg):
    """Strip the deliberate aim-point offsets so 'distance to target centre' is the
    right yardstick. Head offset alone biases the aim ~28px up on an 80px box, which
    would otherwise read as a permanent 28px error."""
    cfg["targeting_settings"].update(head_toggle=False, predict_drop=False,
                                     lead_target=False)
    return cfg
