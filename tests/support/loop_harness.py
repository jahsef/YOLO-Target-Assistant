"""Drives the post-detection half of the real main loop over a scripted detection stream.

Calls Aimbot._stage_track / _stage_aim / _stage_render directly — the same methods
main_serial() and main_async() use — so this exercises production code rather than a
copy of it. Only the capture and GPU stages are stubbed out.
"""

import threading

import numpy as np
from ultralytics.utils.ops import xyxy2xywh

from src.aimbot.aimbot import Aimbot
from src.aimbot.data_parsing import targetselector
from tests.support.fakes import FakeFPS, RecordingMouse


class FakeInput:
    """InputDetector stand-in with directly settable state."""

    def __init__(self, rmb=True, toggled=True):
        self.is_rmb_pressed = rmb
        self.is_toggled = toggled
        self.is_lmb_pressed = False

    def stop(self):
        pass


class LoopHarness:
    def __init__(self, cfg, tracker, hw_capture=(640, 640), screen_hw=(1440, 2560), fps=144.0):
        self.cfg = cfg
        self.tracker = tracker
        self.fps_tracker = FakeFPS(fps)
        self.mousemover = RecordingMouse()
        self.inputdetector = FakeInput()
        self.target_selector = targetselector.TargetSelector(
            cfg=cfg, detection_window_dim=hw_capture, screen_hw=screen_hw,
            fps_tracker=self.fps_tracker,
        )
        bot = object.__new__(Aimbot)
        bot.cfg = cfg
        bot.target_selector = self.target_selector
        bot.mousemover = self.mousemover
        bot.tracker = tracker
        bot.fps_tracker = self.fps_tracker
        bot.inputdetector = self.inputdetector
        bot.gui_manager = None
        bot._frame_count = 0
        bot._route_lock = threading.Lock()
        bot._stop = threading.Event()
        bot._worker_error = None
        bot._detect_ema = 0.0
        bot._grab_ema = 0.0
        bot._starve_ema = 0.0
        bot._lmb_was_down = False
        self.bot = bot

    @property
    def frames(self):
        return self.bot._frame_count

    def step(self, dets, bypass_rows=None, *, active=True, ads=True):
        """dets: (N, 6) xyxy detector rows. Returns (raw_deltas, scaled_deltas)."""
        self.inputdetector.is_rmb_pressed = ads
        # _stage_detect's tail, minus the GPU work it wraps
        results = np.array(dets, dtype=np.float32, copy=True).reshape(-1, 6)
        results[:, 0:4] = xyxy2xywh(results[:, 0:4])
        bypass = (np.asarray(bypass_rows, dtype=np.float32).reshape(-1, 6)
                  if bypass_rows is not None and len(bypass_rows)
                  else np.empty((0, 6), dtype=np.float32))

        tracked = self.bot._stage_track(results, bypass)
        raw, scaled = self.bot._stage_aim(tracked, active)
        self.bot._stage_render(None, tracked, raw, scaled)
        self.tracked = tracked
        return raw, scaled

    @property
    def moves(self):
        return [m for m, _ in self.mousemover.moves]
