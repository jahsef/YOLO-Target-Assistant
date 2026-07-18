"""Adapter giving the stock ultralytics ``BYTETracker`` the same call shape the main
loop uses for the vectorized / cpp trackers.

The loop (aimbot.py) drives a tracker as:
    tracker.update(detections)                 # (N, 6) [x, y, w, h, conf, cls]
    tracker.multi_predict(tracks=None)
    tracker.get_active_tracks_with_lifetime()  # (M, 10) ..., start_frame, last_frame

Stock ultralytics instead wants a *results object* (not a raw array) for ``update``,
returns only ``(M, 8)`` with no lifetime columns, and its ``multi_predict`` needs the
strack pool rather than ``None``. This module bridges that gap; the vectorized and cpp
trackers already satisfy the contract natively and are used unwrapped.
"""

from __future__ import annotations

import numpy as np
from ultralytics.utils.ops import xywh2xyxy


def crosshair_rows_to_tracked(crosshair_rows: np.ndarray) -> np.ndarray:
    """(K, 6) [x1,y1,x2,y2,conf,cls] -> (K, 10) tracked-format rows for crosshair
    dets that bypass the tracker (hsv_settings.bypass_tracker). track_id/idx/
    start_frame/last_frame are placeholders (-1/0); downstream only reads the
    xyxy box + cls."""
    k = len(crosshair_rows)
    out = np.empty((k, 10), dtype=np.float32)
    out[:, 0:4] = crosshair_rows[:, 0:4]   # xyxy
    out[:, 4] = -1                         # track_id (untracked)
    out[:, 5] = crosshair_rows[:, 4]       # score
    out[:, 6] = crosshair_rows[:, 5]       # cls
    out[:, 7] = -1                         # idx
    out[:, 8] = 0                          # start_frame
    out[:, 9] = 0                          # last_frame
    return out


class _UltralyticsResults:
    """Minimal stand-in for the detection results object ultralytics ``update`` expects.

    Exposes ``xywh`` / ``conf`` / ``cls`` / ``xyxy`` plus ``len()`` and boolean-mask
    indexing, which is all ``BYTETracker.update`` and ``init_track`` touch.
    """

    def __init__(self, xywh: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self.xywh = xywh
        self.conf = conf
        self.cls = cls
        self.xyxy = xywh2xyxy(xywh) if len(xywh) else xywh.reshape(0, 4)

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, idx) -> "_UltralyticsResults":
        return _UltralyticsResults(self.xywh[idx], self.conf[idx], self.cls[idx])


class UltralyticsAdapter:
    """Wrap stock ultralytics ``BYTETracker`` to match the vectorized tracker's API."""

    def __init__(self, impl):
        self.impl = impl
        self.frame_id = 0
        self._starts: dict[int, int] = {}  # track_id -> start_frame
        self._last = np.empty((0, 8), dtype=np.float32)

    @property
    def args(self):
        return self.impl.args

    @property
    def max_time_lost(self) -> int:
        return self.impl.max_time_lost

    @max_time_lost.setter
    def max_time_lost(self, value: int) -> None:
        self.impl.max_time_lost = int(value)

    def update(self, detections: np.ndarray) -> np.ndarray:
        """(N, 6) [x, y, w, h, conf, cls] -> (M, 8) [x1,y1,x2,y2,id,score,cls,idx]."""
        self.frame_id += 1
        detections = np.asarray(detections, dtype=np.float32)
        if detections.ndim == 1:
            detections = detections[np.newaxis]
        results = _UltralyticsResults(
            detections[:, :4], detections[:, 4], detections[:, 5]
        )
        self._last = np.asarray(self.impl.update(results), dtype=np.float32)
        if self._last.ndim != 2 or self._last.shape[1] < 8:
            self._last = np.empty((0, 8), dtype=np.float32)

        # External lifetime bookkeeping: record first-seen frame per id, prune the rest.
        live_ids = set()
        for track_id in self._last[:, 4].astype(np.int64):
            live_ids.add(track_id)
            self._starts.setdefault(track_id, self.frame_id)
        self._starts = {tid: f for tid, f in self._starts.items() if tid in live_ids}
        return self._last

    def multi_predict(self, tracks=None) -> None:
        """Forward the real strack pool (the loop passes ``None``, which would throw)."""
        self.impl.multi_predict(self.impl.tracked_stracks)

    def get_active_tracks_with_lifetime(self) -> np.ndarray:
        """(M, 8) -> (M, 10) by appending start_frame and last_frame columns."""
        if len(self._last) == 0:
            return np.empty((0, 10), dtype=np.float32)
        ids = self._last[:, 4].astype(np.int64)
        starts = np.array([self._starts.get(int(t), self.frame_id) for t in ids], dtype=np.float32)
        last = np.full(len(self._last), self.frame_id, dtype=np.float32)
        return np.concatenate(
            [self._last, starts.reshape(-1, 1), last.reshape(-1, 1)], axis=1
        )
