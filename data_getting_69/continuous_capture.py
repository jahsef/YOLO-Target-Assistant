

import sys
import bettercam#no need for betterercam cupy support
from screeninfo import get_monitors
import imageio.v3 as iio
import time
from pathlib import Path

hw_capture = (640,640)
monitor = get_monitors()[0]#monitor num
screen_x = monitor.width
screen_y = monitor.height
x_offset = (screen_x - hw_capture[1])//2
y_offset = (screen_y - hw_capture[0])//2
capture_region = (0 + x_offset, 0 + y_offset, screen_x - x_offset, screen_y - y_offset)

camera = bettercam.create(region = capture_region, output_color='RGB',max_buffer_len=2, nvidia_gpu = False)
frame_counter = 0
target_fps = 2
output_folder = Path.cwd() / 'data_getting_69/screenshots/'

while True:
    time.sleep(1 // target_fps)
    
    frame = camera.grab()#should be in hwc format
    if frame is None:
        continue
    
    # print(f'{counter}, grabbed successfully')
    frame_filename = Path(f'frame{frame_counter}.png')
    
    
    # has_dupe_already = False
    # if Path.exists(output_destination):
    #     # print(frame_filename)
    #     stem = str(frame_filename.stem)
    #     # print(stem)
    #     if stem.count("(") != 0:
    #         lparentheses_idx = stem.index("(")
    #         rparentheses_idx = stem.index(")")
    #         dupe_counter = stem[lparentheses_idx+1: rparentheses_idx]
    #         has_dupe_already = True
    #     else:
    #         lparentheses_idx = len(str(stem))
    #         print(frame_filename)
    #         print(lparentheses_idx)
    output_destination = output_folder / frame_filename
    dupe_counter = -1
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
    
    print(frame.shape)
    frame_counter+=1
    