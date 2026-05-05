"""Benchmark the HSV pipeline from fart_avgpool.py on 3 synthetic frames.

Cases:
  1. empty            — 320x320 all-zero RGB
  2. dot              — 10x10 red square at center
  3. ring             — radius 50, width 10 red ring at center
"""
import sys
import time
from pathlib import Path

import numpy as np
import cupy as cp
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.aimbot.engine.hsv_crosshair import cupy_red_mask

# ---- pipeline config (mirrors fart_avgpool.py) -----------------------------

CAPTURE = (320, 320)
_H, _W = CAPTURE

_HSV_COLOR_CENTER = 0
_HSV_COLOR_RANGE = 10
_HSV_S_MIN, _HSV_S_MAX = 160, 255
_HSV_V_MIN, _HSV_V_MAX = 140, 255

TOP_K = 48
SCATTER_THRESHOLD = 5
SMALL_ROI_HW = 24
RATIO_THRESHOLD = 1.5


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

# weighted_center buffers
_ys = cp.arange(_H, dtype=cp.float32).reshape(_H, 1)
_xs = cp.arange(_W, dtype=cp.float32).reshape(1, _W)
_cy0 = (_H - 1) * 0.5
_cx0 = (_W - 1) * 0.5
_sigma_y = (_H * 0.5) / 1.665
_sigma_x = (_W * 0.5) / 1.665
wc_weights = cp.exp(
    -(((_ys - _cy0) ** 2) / (2.0 * _sigma_y * _sigma_y)
      + ((_xs - _cx0) ** 2) / (2.0 * _sigma_x * _sigma_x))
).astype(cp.float32)
wc_wm = cp.empty((_H, _W), dtype=cp.float32)


def run_pipeline(frame_rgb_gpu: cp.ndarray):
    """Full pipeline as in fart_avgpool.py main loop. Returns mode string."""
    cp_mask = cupy_red_mask(
        frame_rgb_gpu[cp.newaxis, ...],
        color_center=_HSV_COLOR_CENTER, color_range=_HSV_COLOR_RANGE,
        s_min=_HSV_S_MIN, s_max=_HSV_S_MAX,
        v_min=_HSV_V_MIN, v_max=_HSV_V_MAX,
    )[0, ...]

    mask_u8 = cp_mask.astype(cp.uint8)
    row_sum = mask_u8.sum(axis=1, keepdims=True)
    col_sum = mask_u8.sum(axis=0, keepdims=True)
    suppress = row_sum > RATIO_THRESHOLD * col_sum
    cp_mask_filtered = cp_mask & ~suppress

    mask_f32 = cp_mask_filtered.astype(cp.float32)
    mask_t = torch.from_dlpack(mask_f32)[None, None, ...]
    with torch.no_grad():
        opened = opening(mask_t)
        pooled = density(opened)[0, 0]

    Hp, Wp = pooled.shape
    stride_y = _H / Hp
    stride_x = _W / Wp
    roi_half = SMALL_ROI_HW // 2

    flat = pooled.flatten()
    K = min(TOP_K, flat.numel())
    vals, idxs = flat.topk(K)
    ys_top = (idxs // Wp).float()
    xs_top = (idxs % Wp).float()
    top1_val = float(vals[0])

    valid = vals > 0
    n_valid = int(valid.sum().item())
    if n_valid > 1:
        valid_ys = ys_top[valid]
        valid_xs = xs_top[valid]
        cy_topk = valid_ys.mean()
        cx_topk = valid_xs.mean()
        dy = (valid_ys - cy_topk) * stride_y
        dx = (valid_xs - cx_topk) * stride_x
        l2 = torch.sqrt(dy * dy + dx * dx)
        mean_dist = float(torch.sqrt(l2).mean())
    else:
        mean_dist = 0.0

    if top1_val <= 0:
        return "empty", None

    if mean_dist <= SCATTER_THRESHOLD:
        py0 = int(ys_top[0].item())
        px0 = int(xs_top[0].item())
        cy_op = py0 * stride_y
        cx_op = px0 * stride_x
        y_lo = max(0, int(cy_op - roi_half))
        y_hi = min(_H, int(cy_op + roi_half))
        x_lo = max(0, int(cx_op - roi_half))
        x_hi = min(_W, int(cx_op + roi_half))
        opened_cp = cp.from_dlpack(opened.detach())[0, 0]
        roi_mask = opened_cp[y_lo:y_hi, x_lo:x_hi] > 0
        cys, cxs = cp.where(roi_mask)
        if cys.size > 0:
            return "small", (float(cys.mean()) + y_lo, float(cxs.mean()) + x_lo)
        return "small_empty_roi", (cy_op, cx_op)
    else:
        opened_cp = cp.from_dlpack(opened.detach())[0, 0]
        cp.multiply(wc_weights, opened_cp, out=wc_wm)
        total = float(wc_wm.sum())
        if total > 0:
            cy = float((wc_wm * _ys).sum()) / total
            cx = float((wc_wm * _xs).sum()) / total
            return "multi", (cy, cx)
        return "multi_empty", None


# ---- synthetic frames ------------------------------------------------------

def make_empty():
    return cp.zeros((_H, _W, 3), dtype=cp.uint8)


def make_dot():
    f = np.zeros((_H, _W, 3), dtype=np.uint8)
    cy, cx = _H // 2, _W // 2
    f[cy - 5:cy + 5, cx - 5:cx + 5] = (220, 30, 30)  # RGB ~ red
    return cp.asarray(f)


def make_ring():
    f = np.zeros((_H, _W, 3), dtype=np.uint8)
    cy, cx = _H // 2, _W // 2
    yy, xx = np.ogrid[:_H, :_W]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    ring = (r >= 45) & (r <= 55)
    f[ring] = (220, 30, 30)
    return cp.asarray(f)


# ---- bench ------------------------------------------------------------------

def bench(name, frame, n_warmup=50, n_iter=500):
    # warmup
    for _ in range(n_warmup):
        run_pipeline(frame)
    cp.cuda.runtime.deviceSynchronize()

    times_ns = np.empty(n_iter, dtype=np.int64)
    for i in range(n_iter):
        cp.cuda.runtime.deviceSynchronize()
        t0 = time.perf_counter_ns()
        mode, _ = run_pipeline(frame)
        cp.cuda.runtime.deviceSynchronize()
        t1 = time.perf_counter_ns()
        times_ns[i] = t1 - t0

    times_ms = times_ns / 1e6
    mean_ms = float(times_ms.mean())
    std_ms = float(times_ms.std())
    fps = 1000.0 / mean_ms
    print(f"{name:8s} | mode={mode:6s} | mean={mean_ms:6.3f} ms | std={std_ms:5.3f} ms | fps={fps:7.1f}")


# ---- per-stage breakdown ---------------------------------------------------

def bench_stage(label, fn, n_warmup=50, n_iter=500):
    """Time a single callable with sync barriers around each call."""
    for _ in range(n_warmup):
        fn()
    cp.cuda.runtime.deviceSynchronize()

    times_ns = np.empty(n_iter, dtype=np.int64)
    for i in range(n_iter):
        cp.cuda.runtime.deviceSynchronize()
        t0 = time.perf_counter_ns()
        fn()
        cp.cuda.runtime.deviceSynchronize()
        t1 = time.perf_counter_ns()
        times_ns[i] = t1 - t0
    arr = times_ns / 1e6
    print(f"  {label:32s} mean={arr.mean():6.3f} ms  std={arr.std():5.3f} ms")


def stage_breakdown(name: str, frame: cp.ndarray):
    """Bench each stage of the pipeline individually using inputs precomputed
    from a single forward pass. Sync barriers force per-stage timings, so the sum
    of stages will exceed the end-to-end mean (no overlap)."""
    print(f"\n[per-stage on '{name}']")

    # Stage 1: HSV mask
    bench_stage(
        "1. cupy_red_mask",
        lambda: cupy_red_mask(
            frame[cp.newaxis, ...],
            color_center=_HSV_COLOR_CENTER, color_range=_HSV_COLOR_RANGE,
            s_min=_HSV_S_MIN, s_max=_HSV_S_MAX,
            v_min=_HSV_V_MIN, v_max=_HSV_V_MAX,
        ),
    )

    # Precompute mask for downstream stages
    cp_mask = cupy_red_mask(
        frame[cp.newaxis, ...],
        color_center=_HSV_COLOR_CENTER, color_range=_HSV_COLOR_RANGE,
        s_min=_HSV_S_MIN, s_max=_HSV_S_MAX,
        v_min=_HSV_V_MIN, v_max=_HSV_V_MAX,
    )[0, ...]

    # Stage 2: row-suppression
    def stage_rowsup():
        mask_u8 = cp_mask.astype(cp.uint8)
        row_sum = mask_u8.sum(axis=1, keepdims=True)
        col_sum = mask_u8.sum(axis=0, keepdims=True)
        suppress = row_sum > RATIO_THRESHOLD * col_sum
        return cp_mask & ~suppress
    bench_stage("2. row-suppression", stage_rowsup)

    cp_mask_filtered = stage_rowsup()
    mask_f32_static = cp_mask_filtered.astype(cp.float32)
    mask_t_static = torch.from_dlpack(mask_f32_static)[None, None, ...]

    # Stage 3: opening (torch)
    def stage_opening():
        with torch.no_grad():
            return opening(mask_t_static)
    bench_stage("3. opening (torch)", stage_opening)

    opened_static = stage_opening()

    # Stage 4: density (torch)
    def stage_density():
        with torch.no_grad():
            return density(opened_static)[0, 0]
    bench_stage("4. density (torch)", stage_density)

    pooled_static = stage_density()
    Hp, Wp = pooled_static.shape

    # Stage 5: top-K extraction
    def stage_topk():
        flat = pooled_static.flatten()
        K = min(TOP_K, flat.numel())
        vals, idxs = flat.topk(K)
        ys_top = (idxs // Wp).float()
        xs_top = (idxs % Wp).float()
        return vals, ys_top, xs_top
    bench_stage("5. top-K extraction", stage_topk)

    vals_static, ys_top_static, xs_top_static = stage_topk()
    stride_y = _H / Hp
    stride_x = _W / Wp

    # Stage 6: scatter calculation (incl. sync via .item() on n_valid)
    def stage_scatter():
        valid = vals_static > 0
        n_valid = int(valid.sum().item())
        if n_valid > 1:
            valid_ys = ys_top_static[valid]
            valid_xs = xs_top_static[valid]
            cy_topk = valid_ys.mean()
            cx_topk = valid_xs.mean()
            dy = (valid_ys - cy_topk) * stride_y
            dx = (valid_xs - cx_topk) * stride_x
            l2 = torch.sqrt(dy * dy + dx * dx)
            return float(torch.sqrt(l2).mean())
        return 0.0
    bench_stage("6. scatter calc (+ topk sync)", stage_scatter)

    # Stage 7: small-dot ROI centroid (only meaningful when there's red)
    if float(vals_static[0]) > 0:
        roi_half = SMALL_ROI_HW // 2
        py0 = int(ys_top_static[0].item())
        px0 = int(xs_top_static[0].item())
        cy_op = py0 * stride_y
        cx_op = px0 * stride_x
        y_lo = max(0, int(cy_op - roi_half))
        y_hi = min(_H, int(cy_op + roi_half))
        x_lo = max(0, int(cx_op - roi_half))
        x_hi = min(_W, int(cx_op + roi_half))

        def stage_smalldot():
            opened_cp = cp.from_dlpack(opened_static.detach())[0, 0]
            roi_mask = opened_cp[y_lo:y_hi, x_lo:x_hi] > 0
            cys, cxs = cp.where(roi_mask)
            if cys.size > 0:
                return float(cys.mean()) + y_lo, float(cxs.mean()) + x_lo
            return cy_op, cx_op
        bench_stage("7. small-dot ROI centroid", stage_smalldot)

        # Stage 8: multi-element weighted_center
        def stage_multi():
            opened_cp = cp.from_dlpack(opened_static.detach())[0, 0]
            cp.multiply(wc_weights, opened_cp, out=wc_wm)
            total = float(wc_wm.sum())
            if total > 0:
                cy = float((wc_wm * _ys).sum()) / total
                cx = float((wc_wm * _xs).sum()) / total
                return cy, cx
            return None
        bench_stage("8. multi-elem weighted_center", stage_multi)


def main():
    print(f"capture={_H}x{_W}, top_k={TOP_K}, scatter_thresh={SCATTER_THRESHOLD}, "
          f"small_roi={SMALL_ROI_HW}, ratio_thresh={RATIO_THRESHOLD}")
    print("-" * 80)
    print("[end-to-end]")
    bench("empty", make_empty())
    bench("dot",   make_dot())
    bench("ring",  make_ring())

    # per-stage breakdown on dot (small-dot mode) and ring (multi-elem mode)
    print()
    print("=" * 80)
    print("per-stage breakdown (sync per stage; sum > end-to-end mean by design)")
    print("=" * 80)
    stage_breakdown("dot",  make_dot())
    stage_breakdown("ring", make_ring())


if __name__ == "__main__":
    main()
