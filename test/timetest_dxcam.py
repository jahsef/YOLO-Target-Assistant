import dxcam
import bettercam
import os
import time
import cv2

screen_x = 2560
screen_y = 1440
h_w_capture = (896,1440)#height,width
x_offset = (screen_x - h_w_capture[1])//2
y_offset = (screen_y - h_w_capture[0])//2#reversed because h_w
capture_region = (0 + x_offset, 0 + y_offset, screen_x - x_offset, screen_y - y_offset)

# camera = dxcam.create(region = capture_region)
camera = bettercam.create(region = capture_region, nvidia_gpu=False)

window_height, window_width = h_w_capture
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