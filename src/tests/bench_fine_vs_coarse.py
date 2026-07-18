"""Benchmark fine vs coarse weighted_center.

Both run identical front-end: fused mask+row-suppress kernel (1 launch) -> opening.
Fine:   weighted_center on 320x320 opened mask
Coarse: weighted_center on 80x80 density output (2x stride-2 avgpools, 4x total downsample)
"""
import sys
import time
from pathlib import Path

import numpy as np
import cupy as cp
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.aimbot.engine.hsv_crosshair import fused_red_mask_suppress

# --- shared front-end -------------------------------------------------------

CAPTURE = (320, 320)
_H, _W = CAPTURE

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
_DOWNSAMPLE = 4  # cumulative stride


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


# fine: 320x320, coarse: 80x80
_fine_ys,   _fine_xs,   _fine_weights,   _fine_wm   = make_wc_buffers(_H, _W)
_coarse_h, _coarse_w = _H // _DOWNSAMPLE, _W // _DOWNSAMPLE
_coarse_ys, _coarse_xs, _coarse_weights, _coarse_wm = make_wc_buffers(_coarse_h, _coarse_w)

_mask_buf = cp.empty((_H, _W), dtype=cp.uint8)  # reused fused-kernel output


def front_end(frame_rgb_gpu: cp.ndarray):
    """Run shared fused mask+row-suppress -> opening. Returns (opened_cp, pooled_cp)."""
    mask_filtered = fused_red_mask_suppress(
        frame_rgb_gpu,
        color_center=_HSV_COLOR_CENTER, color_range=_HSV_COLOR_RANGE,
        s_min=_HSV_S_MIN, s_max=_HSV_S_MAX,
        v_min=_HSV_V_MIN, v_max=_HSV_V_MAX,
        ratio=_RATIO, out=_mask_buf,
    )
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


def fine_full(frame):
    opened_cp, _ = front_end(frame)
    return weighted_center(opened_cp, _fine_ys, _fine_xs, _fine_weights, _fine_wm)


def coarse_full(frame):
    _, pooled_cp = front_end(frame)
    pt = weighted_center(pooled_cp, _coarse_ys, _coarse_xs, _coarse_weights, _coarse_wm)
    if pt is None:
        return None
    # density output is at 1/_DOWNSAMPLE resolution, corner-aligned. map back.
    cy, cx = pt
    return cy * _DOWNSAMPLE, cx * _DOWNSAMPLE


# --- synthetic frames -------------------------------------------------------

def make_empty():
    return cp.zeros((_H, _W, 3), dtype=cp.uint8)

def make_dot():
    f = np.zeros((_H, _W, 3), dtype=np.uint8)
    f[155:165, 155:165] = (220, 30, 30)
    return cp.asarray(f)

def make_ring():
    f = np.zeros((_H, _W, 3), dtype=np.uint8)
    yy, xx = np.ogrid[:_H, :_W]
    r = np.sqrt((yy - 160) ** 2 + (xx - 160) ** 2)
    f[(r >= 45) & (r <= 55)] = (220, 30, 30)
    return cp.asarray(f)


# --- bench ------------------------------------------------------------------

def bench(label, fn, frame, n_warmup=50, n_iter=500):
    for _ in range(n_warmup):
        fn(frame)
    cp.cuda.runtime.deviceSynchronize()
    times = np.empty(n_iter, dtype=np.int64)
    for i in range(n_iter):
        cp.cuda.runtime.deviceSynchronize()
        t0 = time.perf_counter_ns()
        fn(frame)
        cp.cuda.runtime.deviceSynchronize()
        times[i] = time.perf_counter_ns() - t0
    ms = times / 1e6
    pt = fn(frame)
    pt_str = f"({pt[0]:7.2f},{pt[1]:7.2f})" if pt is not None else "      None"
    print(f"  {label:18s} mean={ms.mean():6.3f}ms  std={ms.std():5.3f}ms  fps={1000/ms.mean():7.1f}  result={pt_str}")


def main():
    print(f"capture={_H}x{_W}, downsample={_DOWNSAMPLE}x -> coarse={_coarse_h}x{_coarse_w}")
    print("-" * 90)
    for name, frame in [("empty", make_empty()), ("dot", make_dot()), ("ring", make_ring())]:
        print(f"[{name}]")
        bench("fine (320x320)",  fine_full,   frame)
        bench("coarse (80x80)",  coarse_full, frame)


if __name__ == "__main__":
    main()
