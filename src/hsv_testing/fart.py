import sys
from pathlib import Path

import betterercam
import cv2
import numpy as np
import cupy as cp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.aimbot.engine.hsv_crosshair import cupy_red_mask

screen_size = (2560,1440)
screen_center = (screen_size[0]//2, screen_size[1]//2)
capture_size = (640,640)
center_crop = (screen_center[0]-capture_size[0]//2, screen_center[1]-capture_size[1]//2, screen_center[0]+capture_size[0]//2, screen_center[1]+capture_size[1]//2)
camera = betterercam.create(nvidia_gpu=True, max_buffer_len=2, region = center_crop, output_color="BGR")
cv2.namedWindow("screen", cv2.WINDOW_NORMAL)
cv2.resizeWindow("screen", capture_size[1]*3, capture_size[0])

combined_frame_buffer = np.ndarray(shape = (capture_size[0], capture_size[1]*3, 3), dtype = np.uint8)

def get_color_mask(hsv, hue_ranges, s_min=150, v_min=150, s_max=255, v_max=255):
    """
    hue_ranges: list of (h_low, h_high) tuples. Multiple ranges OR'd together (for wraparound colors like red).
    """
    masks = [cv2.inRange(hsv, (h_lo, s_min, v_min), (h_hi, s_max, v_max))
             for h_lo, h_hi in hue_ranges]
    combined = masks[0]
    for m in masks[1:]:
        combined = cv2.bitwise_or(combined, m)
    return combined.astype(bool)[..., None]

color_range = 4
s_min, v_min = 90, 90
 
while True:
    frame = camera.grab() # hwc, cp.uint8 BGR on GPU
    if frame is None:
        continue

    # GPU path: BGR -> RGB via channel reverse, then cupy kernel.
    frame_rgb_gpu = frame[..., ::-1]
    cp_mask = cupy_red_mask(frame_rgb_gpu, color_range, s_min=s_min, v_min=v_min)
    cp_mask_np = cp.asnumpy(cp_mask)[..., None]  # (H, W, 1) bool

    # CPU path: BGR -> HSV -> inRange.
    frame_np = cp.asnumpy(frame)
    hsv = cv2.cvtColor(frame_np, cv2.COLOR_BGR2HSV)
    cv_mask = get_color_mask(hsv, [(0, color_range//2), (179 - color_range//2, 179)],
                             s_min=s_min, v_min=v_min)

    w = capture_size[1]
    combined_frame_buffer[:, :w, :]       = frame_np
    combined_frame_buffer[:, w:2*w, :]    = frame_np * cv_mask
    combined_frame_buffer[:, 2*w:3*w, :]  = frame_np * cp_mask_np

    # Panel labels — drawn last so the text sits over the imagery.
    panel_labels = [("OG image", 0), ("opencv path", w), ("cupy fused kernel", 2 * w)]
    for text, x_off in panel_labels:
        org = (x_off + 12, 32)
        cv2.putText(combined_frame_buffer, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 0), 4, cv2.LINE_AA)  # black outline
        cv2.putText(combined_frame_buffer, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 1, cv2.LINE_AA)  # white fill

    cv2.imshow("screen", combined_frame_buffer)
    cv2.waitKey(1)
