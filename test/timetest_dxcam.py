# import dxcam
import bettercam
import os
import time
import cv2



# camera = dxcam.create(region = capture_region)
camera = bettercam.create(nvidia_gpu=False)

window_height, window_width = 1440,2560
# cv2.namedWindow("Screen Capture", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Screen Capture", window_width, window_height)

last_fps_update = time.perf_counter()
frame_count = 0

while True:
    
    frame = camera.grab()
    if frame is None:
        continue
    
    # cv2.imshow("Screen Capture", frame)
    # cv2.waitKey(1)
    frame_count+=1
    current_time = time.perf_counter()
    if current_time - last_fps_update >= 1:
        fps = frame_count / (current_time - last_fps_update)
        last_fps_update = current_time
        frame_count = 0
        print(f'fps: {fps:.2f}')