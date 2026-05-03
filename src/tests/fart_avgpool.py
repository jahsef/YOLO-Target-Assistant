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

from src.aimbot.engine.hsv_crosshair import cupy_red_mask
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

# morphological opening (erosion -> dilation) followed by avgpool density stack.
# pytorch has no minpool, so wrap maxpool with sign flips: min(x) = -max(-x).
class MinPool2d(nn.Module):
    def __init__(self, kernel_size, stride=1, padding=0):
        super().__init__()
        self.mp = nn.MaxPool2d(kernel_size, stride, padding)

    def forward(self, x):
        return -self.mp(-x)

opening = nn.Sequential(
    nn.ZeroPad2d((1,0,1,0)),
    MinPool2d(kernel_size=2, stride=1),       # erode: kills 1-2px scattered noise
    nn.MaxPool2d(kernel_size=3, stride=1, padding=1),    # dilate: restore survivor blob sizes
    # nn.MaxPool2d(kernel_size=5, stride=1, padding=2), 
).cuda().eval()
# density downsample stack. cumulative stride is derived at runtime from output shape
# so changing this stack doesn't require touching the small-dot mapping below.
density = nn.Sequential(
    nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
    nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
    # nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
    # nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
).cuda().eval()

# dual-mode tuning knobs
TOP_K = 48 # topk too high would destroy small dot precision if any name FPs
# mean sqrt-of-L2 dist (units = sqrt(input pixels)) of valid top-K members from their
# centroid, above which we declare scatter and fall through to multi-element mode.
# sqrt compresses outliers so a couple of far-away top-K patches don't dominate the metric.
SCATTER_THRESHOLD = 3
# small-dot ROI side length in input pixels. window centered on argmax patch's pixel center.
# bigger = more slack against off-center argmax, smaller = tighter precision (less periphery).
SMALL_ROI_HW = 24
# horizontal-stripe (red nametag) suppression: kill mask pixels where row_sum[y] >
# RATIO_THRESHOLD * col_sum[x]. applied to raw mask before opening so opening sees
# a nametag-free input. 2 = "this row is more than 2x wider than this column is tall."
RATIO_THRESHOLD = 1.5

# weighted_center buffers for the opened (capture-size) mask. mirrors HSVCrosshairDetector.
_H, _W = capture_size
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

    # row-suppression (kill horizontally-elongated structures e.g. red nametags) on raw mask.
    # applied pre-opening so opening doesn't grow nametag remnants back via dilation.
    mask_u8 = cp_mask.astype(cp.uint8)
    row_sum = mask_u8.sum(axis=1, keepdims=True)              # (H, 1)
    col_sum = mask_u8.sum(axis=0, keepdims=True)              # (1, W)
    suppress = row_sum > RATIO_THRESHOLD * col_sum            # (H, W) via broadcast
    cp_mask_filtered = cp_mask & ~suppress                    # (H, W) bool

    # cupy bool -> torch float32 (1,1,H,W) via dlpack on the cast tensor
    mask_f32 = cp_mask_filtered.astype(cp.float32)
    mask_t = torch.from_dlpack(mask_f32)[None, None, ...]
    with torch.no_grad():
        opened = opening(mask_t)              # (1,1,H,W) float32 in {0,1}
        pooled = density(opened)[0, 0]        # (Hp,Wp) float32 density

    Hp, Wp = pooled.shape
    pooled_np = pooled.detach().cpu().numpy()
    # patch -> pixel: derived from actual density-output shape so any pool stack works.
    stride_y = _H / Hp
    stride_x = _W / Wp
    roi_half = SMALL_ROI_HW // 2

    # top-K argmax positions in pooled grid (units = patches).
    flat = pooled.flatten()
    K = min(TOP_K, flat.numel())
    vals, idxs = flat.topk(K)
    ys_top = (idxs // Wp).float()
    xs_top = (idxs % Wp).float()
    top1_val = float(vals[0])

    # mean SQRT-DISTANCE (input pixels^0.5) of valid (val>0) top-K members from their
    # unweighted centroid. sqrt compresses outliers — a pixel at distance 100 contributes
    # 10, not 100 — so a few far-away top-K members no longer dominate mean_dist.
    # SCATTER_THRESHOLD is now in sqrt(pixel) units (~5 ≈ 24 raw px before).
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
    
    # mode select + final centroid. branch-internal extras (mask_px count, weighted total)
    # are captured into `branch_extra` for the consolidated log block at end-of-frame.
    final_pt = None  # (cy, cx) in opened-mask pixel coords
    mode = "empty"
    branch_extra = ""
    if top1_val > 0:
        if mean_dist <= SCATTER_THRESHOLD:
            # SMALL-DOT MODE: localize to argmax patch, centroid OPENED mask in pixel ROI.
            # opened (not raw) because argmax was computed over density(opened) — that's the
            # signal we're triangulating against. opened in the patch's RF is non-empty by
            # construction whenever top1_val > 0, so empty-ROI is structurally impossible.
            py0 = int(ys_top[0].item())
            px0 = int(xs_top[0].item())
            # same-pad pools are corner-aligned: output[i] ↔ input[i*stride], no +0.5 offset.
            cy_op = py0 * stride_y
            cx_op = px0 * stride_x
            y_lo = max(0, int(cy_op - roi_half))
            y_hi = min(_H, int(cy_op + roi_half))
            x_lo = max(0, int(cx_op - roi_half))
            x_hi = min(_W, int(cx_op + roi_half))
            opened_cp = cp.from_dlpack(opened.detach())[0, 0]  # (H, W) float32 {0,1}
            roi_mask = opened_cp[y_lo:y_hi, x_lo:x_hi] > 0
            cys, cxs = cp.where(roi_mask)
            final_pt = (float(cys.mean()) + y_lo, float(cxs.mean()) + x_lo)
            branch_extra = (
                f"argmax=({py0},{px0}) patch_px=({cy_op:.1f},{cx_op:.1f}) "
                f"roi=[{y_lo}:{y_hi},{x_lo}:{x_hi}] opened_px={int(cys.size)}"
            )
            mode = f"small (d={mean_dist:.1f})"
        else:
            # MULTI-ELEMENT MODE: weighted_center over opened mask.
            opened_cp = cp.from_dlpack(opened.detach())[0, 0]
            cp.multiply(wc_weights, opened_cp, out=wc_wm)
            total = float(wc_wm.sum())
            if total > 0:
                cy = float((wc_wm * _ys).sum()) / total
                cx = float((wc_wm * _xs).sum()) / total
                final_pt = (cy, cx)
                branch_extra = f"weighted_total={total:.1f}"
            else:
                log("multi-elem: opened mask empty after weighting", "WARNING")
                branch_extra = "EMPTY_WEIGHTED"
            mode = f"multi (d={mean_dist:.1f})"

    # rescale density [0,1] -> [0,255] uint8 for display, then nearest-resize back to capture size.
    pooled_u8 = np.clip(pooled_np * 255.0, 0, 255).astype(np.uint8)
    pooled_vis = cv2.resize(
        pooled_u8, (capture_size[1], capture_size[0]), interpolation=cv2.INTER_NEAREST
    )
    pooled_vis_bgr = cv2.cvtColor(pooled_vis, cv2.COLOR_GRAY2BGR)

    # three diagnostic markers, corner-aligned patch-to-pixel mapping (no +0.5):
    #   orange = argmax (top-1 patch center)
    #   purple = unweighted centroid of valid top-K patches (only if n_valid > 1)
    #   green  = voted final location (small-dot ROI centroid OR multi-element weighted_center)
    if top1_val > 0:
        argmax_y = int(int(ys_top[0].item()) * stride_y)
        argmax_x = int(int(xs_top[0].item()) * stride_x)
        cv2.drawMarker(pooled_vis_bgr, (argmax_x, argmax_y), (0, 165, 255),
                       markerType=cv2.MARKER_CROSS, markerSize=10, thickness=1)
        cv2.circle(pooled_vis_bgr, (argmax_x, argmax_y), 4, (0, 165, 255), 1)

    if n_valid > 1:
        cent_y = int(float(cy_topk.item()) * stride_y)
        cent_x = int(float(cx_topk.item()) * stride_x)
        cv2.drawMarker(pooled_vis_bgr, (cent_x, cent_y), (255, 0, 255),
                       markerType=cv2.MARKER_TILTED_CROSS, markerSize=10, thickness=1)
        cv2.circle(pooled_vis_bgr, (cent_x, cent_y), 4, (255, 0, 255), 1)

    if final_pt is not None:
        fy, fx = int(final_pt[0]), int(final_pt[1])
        cv2.drawMarker(pooled_vis_bgr, (fx, fy), (0, 255, 0),
                       markerType=cv2.MARKER_CROSS, markerSize=12, thickness=1)
        cv2.circle(pooled_vis_bgr, (fx, fy), 5, (0, 255, 0), 1)

    # legend at top of density panel: argmax (orange) / centroid (purple) / voted (green).
    # drawn under the panel-label text band so it doesn't collide.
    legend_entries = [
        ("argmax",   (0, 165, 255)),
        ("centroid", (255, 0, 255)),
        ("voted",    (0, 255, 0)),
    ]
    legend_y = 56
    legend_x = 10
    for text, color in legend_entries:
        cv2.circle(pooled_vis_bgr, (legend_x + 5, legend_y - 4), 4, color, -1)
        cv2.putText(pooled_vis_bgr, text, (legend_x + 14, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(pooled_vis_bgr, text, (legend_x + 14, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        legend_y += 16

    # mode + dist readout in bottom-left of density panel.
    cv2.putText(pooled_vis_bgr, mode, (10, capture_size[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(pooled_vis_bgr, mode, (10, capture_size[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

    cp_mask_np = cp.asnumpy(cp_mask)[..., None]  # (H, W, 1) bool
    frame_np = cp.asnumpy(frame)
    # row-masked mask as standalone grayscale viz (binary, scaled to 0/255).
    filtered_u8 = (cp.asnumpy(cp_mask_filtered).astype(np.uint8)) * 255
    filtered_vis_bgr = cv2.cvtColor(filtered_u8, cv2.COLOR_GRAY2BGR)
    # opened mask as standalone grayscale viz (binary, scaled to 0/255).
    opened_u8 = (opened[0, 0].detach().cpu().numpy() * 255).astype(np.uint8)
    opened_vis_bgr = cv2.cvtColor(opened_u8, cv2.COLOR_GRAY2BGR)

    w = capture_size[1]
    combined_frame_buffer[:, :w, :] = frame_np
    combined_frame_buffer[:, w:2 * w, :] = frame_np * cp_mask_np
    combined_frame_buffer[:, 2 * w:3 * w, :] = filtered_vis_bgr
    combined_frame_buffer[:, 3 * w:4 * w, :] = opened_vis_bgr
    combined_frame_buffer[:, 4 * w:5 * w, :] = pooled_vis_bgr

    panel_labels = [
        ("OG image", 0),
        ("cupy fused kernel", w),
        (f"row-mask (ratio>{RATIO_THRESHOLD})", 2 * w),
        ("opened (erode->dilate)", 3 * w),
        (f"density {pooled_np.shape[0]}x{pooled_np.shape[1]}", 4 * w),
    ]
    for text, x_off in panel_labels:
        org = (x_off + 10, 26)
        cv2.putText(combined_frame_buffer, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(combined_frame_buffer, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)

    if should_log:
        topk_str = ", ".join(
            f"({int(ys_top[r].item())},{int(xs_top[r].item())},{float(vals[r]):.2f})"
            for r in range(K)
        )
        final_str = f"({final_pt[0]:.2f},{final_pt[1]:.2f})" if final_pt is not None else "None"
        log(
            f"frame={frame_idx} mode={mode} top1={top1_val:.3f} mean_dist={mean_dist:.2f} "
            f"(thresh={SCATTER_THRESHOLD}) final={final_str} | topK: {topk_str}"
            + (f" | {branch_extra}" if branch_extra else ""),
            "DEBUG",
        )

    cv2.imshow("screen", combined_frame_buffer)
    cv2.waitKey(1)
