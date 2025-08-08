import sys
import bettercam#no need for betterercam cupy support
from screeninfo import get_monitors
import imageio.v3 as iio
import time
from pathlib import Path
import win32api
import win32con
import time
import threading

#this script watches for right click then 
hw_capture = (640,640)
monitor = get_monitors()[0]#monitor num
screen_x = monitor.width
screen_y = monitor.height
x_offset = (screen_x - hw_capture[1])//2
y_offset = (screen_y - hw_capture[0])//2
capture_region = (0 + x_offset, 0 + y_offset, screen_x - x_offset, screen_y - y_offset)
camera = bettercam.create(region = capture_region, output_color='RGB',max_buffer_len=2, nvidia_gpu = False)
frame_counter = 0
ambient_fps = .15
target_fps = .75
output_folder = Path.cwd() / 'data_getting_69/screenshots/'


# Global variable to track if right-click is held down
right_click_held = False

def check_right_click():
    global right_click_held
    while True:
        # Get the state of the right mouse button
        if win32api.GetAsyncKeyState(win32con.VK_RBUTTON) < 0:  # Right mouse button is held down
            if not right_click_held:
                right_click_held = True
                print("Right click held - Capture more frames")
                # Trigger higher frame capture rate here
        else:
            if right_click_held:
                right_click_held = False
                print("Right click released - Capture at lower frame rate")
                # Switch back to low frame capture rate here
        time.sleep(0.1)

# Run the listener in a separate thread to avoid blocking
listener_thread = threading.Thread(target=check_right_click)
listener_thread.daemon = True
listener_thread.start()

while True:

    
    frame = camera.grab()#should be in hwc format
    if frame is None:
        continue
    frame_filename = Path(f'frame{frame_counter}.png')
    output_destination = output_folder / frame_filename
    while Path.exists(output_destination):
        name_str = str(frame_filename.name)
        lparentheses_idx = name_str.find("(")
        rparentheses_idx = name_str.find(")")
        if lparentheses_idx != -1 and rparentheses_idx != -1:
            dupe_counter = int(name_str[lparentheses_idx+1: rparentheses_idx]) + 1
            print(f'current counter: {dupe_counter}')
            frame_name = name_str[:lparentheses_idx]
            frame_filename = Path(f"{frame_name}({dupe_counter}).png")
        else:
            dupe_counter = 0
            print('no counter found starting from 0')
            frame_filename = Path(f"{frame_filename.stem}({dupe_counter}).png")
        output_destination = output_folder / frame_filename
    frame_path = output_folder / frame_filename
    iio.imwrite(frame_path, frame, extension=".png")
    # print(frame.shape)
    frame_counter+=1
    if right_click_held:
        time.sleep(1 // target_fps)
    else:
        time.sleep(1 // ambient_fps)#not perfect but close enough