"""Visual prototype for the simplified `heuristic_spam` voting scheme.

Pipeline: cupy_red_mask -> row-suppression (fused kernel) -> opening -> density ->
weighted_center on density output (coarse). Coarse beats fine on FP suppression
(see bench_fine_vs_coarse_fp.py).
"""
import sys
import logging
from pathlib import Path

import betterercam
import cv2
import numpy as np
import cupy as cp
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.aimbot.engine.hsv_crosshair import cupy_red_mask, _ROW_SUPPRESS_KERNEL
from src.aimbot.utils.utils import log

# root stays at INFO to suppress betterercam DEBUG spam; only our 'aimbot' logger goes to DEBUG.
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s', force=True)
logging.getLogger('aimbot').setLevel(logging.DEBUG)
logging.getLogger('betterercam').setLevel(logging.INFO)

screen_size = (2560, 1440)
screen_center = (screen_size[0] // 2, screen_size[1] // 2)
capture_size = (320, 320)
center_crop = (
    screen_center[0] - capture_size[0] // 2,
    screen_center[1] - capture_size[1] // 2,
    screen_center[0] + capture_size[0] // 2,
    screen_center[1] + capture_size[1] // 2,
)
camera = betterercam.create(
    nvidia_gpu=True, max_buffer_len=2, region=center_crop, output_color="BGR"
)
N_PANELS = 5
cv2.namedWindow("screen", cv2.WINDOW_NORMAL)
cv2.resizeWindow("screen", capture_size[1] * N_PANELS, capture_size[0])

combined_frame_buffer = np.ndarray(
    shape=(capture_size[0], capture_size[1] * N_PANELS, 3), dtype=np.uint8
)

# kernel args
_HSV_COLOR_CENTER = 0
_HSV_COLOR_RANGE = 10
_HSV_S_MIN = 160
_HSV_S_MAX = 255
_HSV_V_MIN = 140
_HSV_V_MAX = 255

# row-suppression: kill mask pixels where row_sum[y] > RATIO_THRESHOLD * col_sum[x].
RATIO_THRESHOLD = 1.5


# morphological opening (erosion -> dilation). pytorch has no minpool primitive,
# so wrap maxpool with sign flips: min(x) = -max(-x).
class MinPool2d(nn.Module):
    def __init__(self, kernel_size, stride=1, padding=0):
        super().__init__()
        self.mp = nn.MaxPool2d(kernel_size, stride, padding)

    def forward(self, x):
        return -self.mp(-x)


opening = nn.Sequential(
    nn.ZeroPad2d((1, 0, 1, 0)),
    MinPool2d(kernel_size=2, stride=1),                  # erode: kills 1-2px scattered noise
    nn.MaxPool2d(kernel_size=3, stride=1, padding=1),    # dilate: restore survivor blob sizes
).cuda().eval()

# density downsamples the opened mask. weighted_center runs on this output (coarse mode).
# downsampling smooths small FPs into near-zero density while preserving the reticle's
# concentrated mass — cuts dot+fp centroid error by ~40% vs running on the opened mask.
density = nn.Sequential(
    nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
    nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
).cuda().eval()
_DOWNSAMPLE = 4  # cumulative stride of the density stack

# weighted_center buffers at DENSITY resolution (80x80 for 320x320 input @ 4x downsample).
_H, _W = capture_size
_dh, _dw = _H // _DOWNSAMPLE, _W // _DOWNSAMPLE
_ys = cp.arange(_dh, dtype=cp.float32).reshape(_dh, 1)
_xs = cp.arange(_dw, dtype=cp.float32).reshape(1, _dw)
_cy0 = (_dh - 1) * 0.5
_cx0 = (_dw - 1) * 0.5
_sigma_y = (_dh * 0.5) / 1.665
_sigma_x = (_dw * 0.5) / 1.665
wc_weights = cp.exp(
    -(((_ys - _cy0) ** 2) / (2.0 * _sigma_y * _sigma_y)
      + ((_xs - _cx0) ** 2) / (2.0 * _sigma_x * _sigma_x))
).astype(cp.float32)
wc_wm = cp.empty((_dh, _dw), dtype=cp.float32)


def row_suppress(mask: cp.ndarray, ratio: float) -> cp.ndarray:
    """Fused row-suppression via _ROW_SUPPRESS_KERNEL (shared with hsv_crosshair.py)."""
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
        (mask_u8, row_sum, col_sum, out,
         np.int32(H), np.int32(W), np.float32(ratio)),
    )
    return out.view(cp.bool_)


LOG_EVERY = 144
frame_idx = -1

while True:
    frame_idx += 1
    should_log = (frame_idx % LOG_EVERY == 0)
    frame = camera.grab()  # hwc, cp.uint8 BGR on GPU
    if frame is None:
        continue

    frame_rgb_gpu = frame[..., ::-1]
    frame_rgb_gpu_bhwc = frame_rgb_gpu[cp.newaxis, ...]
    cp_mask = cupy_red_mask(
        frame_rgb_gpu_bhwc,
        color_center=_HSV_COLOR_CENTER,
        color_range=_HSV_COLOR_RANGE,
        s_min=_HSV_S_MIN, s_max=_HSV_S_MAX,
        v_min=_HSV_V_MIN, v_max=_HSV_V_MAX,
    )[0, ...]  # (H, W) bool

    # 1) row-suppress (fused kernel) on raw mask
    cp_mask_filtered = row_suppress(cp_mask, RATIO_THRESHOLD)

    # 2) opening (erosion -> dilation) -> density (avgpool x2) on torch
    mask_f32 = cp_mask_filtered.astype(cp.float32)
    mask_t = torch.from_dlpack(mask_f32)[None, None, ...]
    with torch.no_grad():
        opened = opening(mask_t)             # (1, 1, H, W) float32 in {0,1}
        pooled = density(opened)             # (1, 1, dh, dw) float32 density

    # 3) weighted_center on density output (coarse). result is in density-grid coords;
    #    multiply by _DOWNSAMPLE to get back to input-pixel space (corner-aligned mapping).
    pooled_cp = cp.from_dlpack(pooled.detach())[0, 0]  # (dh, dw)
    cp.multiply(wc_weights, pooled_cp, out=wc_wm)
    total = float(wc_wm.sum())
    if total > 0:
        cy_d = float((wc_wm * _ys).sum()) / total
        cx_d = float((wc_wm * _xs).sum()) / total
        final_pt = (cy_d * _DOWNSAMPLE, cx_d * _DOWNSAMPLE)
    else:
        final_pt = None

    # ---- viz ---------------------------------------------------------------
    cp_mask_np = cp.asnumpy(cp_mask)[..., None]
    frame_np = cp.asnumpy(frame)
    opened_u8 = (opened[0, 0].detach().cpu().numpy() * 255).astype(np.uint8)
    opened_vis_bgr = cv2.cvtColor(opened_u8, cv2.COLOR_GRAY2BGR)

    # density vis: nearest-upscale the (dh, dw) density map back to capture size.
    pooled_np = pooled[0, 0].detach().cpu().numpy()
    pooled_u8 = np.clip(pooled_np * 255.0, 0, 255).astype(np.uint8)
    pooled_vis = cv2.resize(
        pooled_u8, (capture_size[1], capture_size[0]), interpolation=cv2.INTER_NEAREST
    )
    pooled_vis_bgr = cv2.cvtColor(pooled_vis, cv2.COLOR_GRAY2BGR)

    # panel 5: OG with green marker on weighted_center result. easiest sanity check.
    result_vis = frame_np.copy()
    if final_pt is not None:
        fy, fx = int(final_pt[0]), int(final_pt[1])
        cv2.drawMarker(result_vis, (fx, fy), (0, 255, 0),
                       markerType=cv2.MARKER_CROSS, markerSize=14, thickness=1)
        cv2.circle(result_vis, (fx, fy), 6, (0, 255, 0), 1)

    w = capture_size[1]
    combined_frame_buffer[:, :w, :]            = frame_np
    combined_frame_buffer[:, w:2 * w, :]       = frame_np * cp_mask_np
    combined_frame_buffer[:, 2 * w:3 * w, :]   = opened_vis_bgr
    combined_frame_buffer[:, 3 * w:4 * w, :]   = pooled_vis_bgr
    combined_frame_buffer[:, 4 * w:5 * w, :]   = result_vis

    panel_labels = [
        ("OG image", 0),
        ("cupy fused kernel", w),
        ("opened (erode->dilate)", 2 * w),
        (f"density {_dh}x{_dw}", 3 * w),
        ("weighted_center result", 4 * w),
    ]
    for text, x_off in panel_labels:
        org = (x_off + 10, 26)
        cv2.putText(combined_frame_buffer, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(combined_frame_buffer, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)

    if should_log:
        if final_pt is not None:
            log(f"frame={frame_idx} final=({final_pt[0]:.2f},{final_pt[1]:.2f}) total={total:.1f}", "DEBUG")
        else:
            log(f"frame={frame_idx} no detection (empty mask)", "DEBUG")

    cv2.imshow("screen", combined_frame_buffer)
    cv2.waitKey(1)
