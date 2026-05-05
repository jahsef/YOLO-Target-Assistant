"""Test FP suppression quality of fine vs coarse weighted_center.

4 scenes per scheme:
  dot (no FP)        — gt: (160, 160)
  dot + small FPs
  ring (no FP)       — gt: (160, 160)
  ring + small FPs

FP = a few small red blobs scattered at periphery, square enough to survive
row-suppression (so they're a fair test). Measures Euclidean error between
detected center and ground truth.
"""
import sys
from pathlib import Path

import numpy as np
import cupy as cp
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.aimbot.engine.hsv_crosshair import cupy_red_mask, _ROW_SUPPRESS_KERNEL

CAPTURE = (320, 320)
_H, _W = CAPTURE
GT = (160, 160)  # all reticle scenes are centered here

_HSV_COLOR_CENTER = 0
_HSV_COLOR_RANGE = 10
_HSV_S_MIN, _HSV_S_MAX = 160, 255
_HSV_V_MIN, _HSV_V_MAX = 140, 255
_RATIO = 1.5
_GAUSS_EDGE = 1.665


class MinPool2d(nn.Module):
    def __init__(self, kernel_size, stride=1, padding=0):
        super().__init__()
        self.mp = nn.MaxPool2d(kernel_size, stride, padding)
    def forward(self, x):
        return -self.mp(-x)


opening = nn.Sequential(
    nn.ZeroPad2d((1, 0, 1, 0)),
    MinPool2d(kernel_size=2, stride=1),
    nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
).cuda().eval()

density = nn.Sequential(
    nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
    nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
).cuda().eval()
_DOWNSAMPLE = 4


def make_wc_buffers(h: int, w: int):
    ys = cp.arange(h, dtype=cp.float32).reshape(h, 1)
    xs = cp.arange(w, dtype=cp.float32).reshape(1, w)
    cy0 = (h - 1) * 0.5
    cx0 = (w - 1) * 0.5
    sigma_y = (h * 0.5) / _GAUSS_EDGE
    sigma_x = (w * 0.5) / _GAUSS_EDGE
    weights = cp.exp(
        -(((ys - cy0) ** 2) / (2.0 * sigma_y * sigma_y)
          + ((xs - cx0) ** 2) / (2.0 * sigma_x * sigma_x))
    ).astype(cp.float32)
    wm = cp.empty((h, w), dtype=cp.float32)
    return ys, xs, weights, wm


_fine_ys,   _fine_xs,   _fine_weights,   _fine_wm   = make_wc_buffers(_H, _W)
_coarse_h, _coarse_w = _H // _DOWNSAMPLE, _W // _DOWNSAMPLE
_coarse_ys, _coarse_xs, _coarse_weights, _coarse_wm = make_wc_buffers(_coarse_h, _coarse_w)


def row_suppress(mask: cp.ndarray) -> cp.ndarray:
    H, W = mask.shape
    mask_u8 = mask.view(cp.uint8)
    row_sum = mask_u8.sum(axis=1, dtype=cp.int64)
    col_sum = mask_u8.sum(axis=0, dtype=cp.int64)
    out = cp.empty_like(mask_u8)
    n = H * W
    threads = 256
    blocks = (n + threads - 1) // threads
    _ROW_SUPPRESS_KERNEL(
        (blocks,), (threads,),
        (mask_u8, row_sum, col_sum, out, np.int32(H), np.int32(W), np.float32(_RATIO)),
    )
    return out.view(cp.bool_)


def front_end(frame_rgb_gpu: cp.ndarray):
    cp_mask = cupy_red_mask(
        frame_rgb_gpu[cp.newaxis, ...],
        color_center=_HSV_COLOR_CENTER, color_range=_HSV_COLOR_RANGE,
        s_min=_HSV_S_MIN, s_max=_HSV_S_MAX,
        v_min=_HSV_V_MIN, v_max=_HSV_V_MAX,
    )[0, ...]
    mask_filtered = row_suppress(cp_mask)
    mask_f32 = mask_filtered.astype(cp.float32)
    mask_t = torch.from_dlpack(mask_f32)[None, None, ...]
    with torch.no_grad():
        opened = opening(mask_t)
        pooled = density(opened)
    opened_cp = cp.from_dlpack(opened.detach())[0, 0]
    pooled_cp = cp.from_dlpack(pooled.detach())[0, 0]
    return opened_cp, pooled_cp


def weighted_center(mask: cp.ndarray, ys, xs, weights, wm):
    cp.multiply(weights, mask, out=wm)
    total = float(wm.sum())
    if total <= 0.0:
        return None
    cy = float((wm * ys).sum()) / total
    cx = float((wm * xs).sum()) / total
    return cy, cx


def detect_fine(frame):
    opened_cp, _ = front_end(frame)
    return weighted_center(opened_cp, _fine_ys, _fine_xs, _fine_weights, _fine_wm)


def detect_coarse(frame):
    _, pooled_cp = front_end(frame)
    pt = weighted_center(pooled_cp, _coarse_ys, _coarse_xs, _coarse_weights, _coarse_wm)
    if pt is None:
        return None
    cy, cx = pt
    return cy * _DOWNSAMPLE, cx * _DOWNSAMPLE


# --- scene builders ---------------------------------------------------------

RED = (220, 30, 30)


def add_dot(frame, cy=160, cx=160, r=5):
    frame[cy - r:cy + r, cx - r:cx + r] = RED


def add_ring(frame, cy=160, cx=160, r_inner=45, r_outer=55):
    yy, xx = np.ogrid[:frame.shape[0], :frame.shape[1]]
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    frame[(rr >= r_inner) & (rr <= r_outer)] = RED


def add_small_fps(frame):
    """Small square red blobs at periphery — square enough to survive row-suppression
    (~5x5 each, so row_sum ≈ col_sum locally), placed far from screen center to
    bias the centroid away from the true reticle if not suppressed."""
    for cy, cx in [(40, 50), (50, 280), (280, 60), (270, 270)]:
        frame[cy - 3:cy + 3, cx - 3:cx + 3] = RED


def make(name: str) -> cp.ndarray:
    f = np.zeros((_H, _W, 3), dtype=np.uint8)
    if name == "dot":
        add_dot(f)
    elif name == "dot+fp":
        add_dot(f)
        add_small_fps(f)
    elif name == "ring":
        add_ring(f)
    elif name == "ring+fp":
        add_ring(f)
        add_small_fps(f)
    return cp.asarray(f)


# --- evaluate ---------------------------------------------------------------

def err(pt):
    if pt is None:
        return None
    dy = pt[0] - GT[0]
    dx = pt[1] - GT[1]
    return float(np.sqrt(dy * dy + dx * dx))


def main():
    print(f"GT center = {GT}, capture={_H}x{_W}, coarse={_coarse_h}x{_coarse_w}")
    print("-" * 96)
    print(f"{'scene':12s} | {'fine (cy, cx)':22s} {'fine err':>10s} | "
          f"{'coarse (cy, cx)':22s} {'coarse err':>11s}")
    print("-" * 96)
    for name in ["dot", "dot+fp", "ring", "ring+fp"]:
        frame = make(name)
        # warm + run a couple times to settle
        for _ in range(3):
            detect_fine(frame); detect_coarse(frame)
        f_pt = detect_fine(frame)
        c_pt = detect_coarse(frame)
        f_err = err(f_pt)
        c_err = err(c_pt)
        f_str = f"({f_pt[0]:7.2f}, {f_pt[1]:7.2f})" if f_pt else "None"
        c_str = f"({c_pt[0]:7.2f}, {c_pt[1]:7.2f})" if c_pt else "None"
        f_e = f"{f_err:8.3f}px" if f_err is not None else "  N/A"
        c_e = f"{c_err:9.3f}px" if c_err is not None else "  N/A"
        print(f"{name:12s} | {f_str:22s} {f_e:>10s} | {c_str:22s} {c_e:>11s}")


if __name__ == "__main__":
    main()
