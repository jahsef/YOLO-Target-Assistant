"""Stand-ins for hardware / heavy components so tests can exercise real logic."""

import time

import numpy as np


class FakeFPS:
    """fps_tracker stub. TargetSelector only calls get_fps()."""

    def __init__(self, fps=144.0):
        self.fps = fps
        self.buffer = []
        self.fps_buffer_len = 69

    def get_fps(self):
        return self.fps

    def update(self):
        pass

    def print_fps(self):
        pass


class RecordingMouse:
    """MouseMover stub that records instead of calling win32api."""

    def __init__(self):
        self.moves = []

    def move_mouse_humanized(self, dx, dy):
        out = (round(dx), round(dy))
        self.moves.append((out, time.perf_counter()))
        return out


def tracked_rows(rows) -> np.ndarray:
    """(N, 10) tracker-format array from [x1,y1,x2,y2,id,conf,cls,idx,start,last] tuples."""
    if not rows:
        return np.empty((0, 10), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


def enemy(x1, y1, x2, y2, track_id=1, conf=0.9, cls=0, start=0, last=30):
    return [x1, y1, x2, y2, track_id, conf, cls, 0, start, last]


def det_rows(rows) -> np.ndarray:
    """(N, 6) detector-format array from [x1,y1,x2,y2,conf,cls] tuples."""
    if not rows:
        return np.empty((0, 6), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


class FakeWin32:
    """Scriptable stand-in for the win32api surface the input code touches.

    key_state maps VK code -> bool 'currently down'. GetAsyncKeyState returns the
    real API's shape: bit 15 = down now, bit 0 = pressed since last call.
    """

    def __init__(self):
        self.key_state = {}
        self.mouse_events = []

    def set_down(self, vk, down=True):
        self.key_state[vk] = down

    def GetAsyncKeyState(self, vk):
        return -32768 if self.key_state.get(vk, False) else 0

    def mouse_event(self, flags, dx, dy, *rest):
        self.mouse_events.append((flags, dx, dy))


def crosshair_frame(h, w, cy, cx, arm=12, thickness=2, seed=0, noise=True):
    """RGB uint8 frame with a saturated-red plus-shaped reticle at (cy, cx).

    Red is (222, 40, 14) — the real game reticle color noted in hsv_crosshair.py.
    Background noise is deliberately kept off the red hue band so it can't leak
    into the mask.
    """
    rng = np.random.default_rng(seed)
    if noise:
        frame = rng.integers(0, 120, size=(h, w, 3), dtype=np.uint8)
        # push background toward blue/green so nothing lands in the red band
        frame[..., 0] = (frame[..., 0] // 3).astype(np.uint8)
    else:
        frame = np.zeros((h, w, 3), dtype=np.uint8)

    half = thickness // 2 + 1
    y0, y1 = max(0, cy - arm), min(h, cy + arm + 1)
    x0, x1 = max(0, cx - arm), min(w, cx + arm + 1)
    frame[y0:y1, max(0, cx - half):min(w, cx + half + 1)] = (222, 40, 14)
    frame[max(0, cy - half):min(h, cy + half + 1), x0:x1] = (222, 40, 14)
    return frame


def nametag_bar(frame, cy, cx, half_w=40, thickness=3, color=(208, 84, 73)):
    """Paint a wide, thin horizontal red-ish bar — the nametag false positive the
    row-suppression heuristic exists to kill. Mutates and returns frame."""
    h, w = frame.shape[:2]
    frame[max(0, cy - thickness // 2):min(h, cy + thickness // 2 + 1),
          max(0, cx - half_w):min(w, cx + half_w)] = color
    return frame
