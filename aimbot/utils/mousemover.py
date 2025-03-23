import win32api
import win32con
import numpy as np
import time

class MouseMover:
    def __init__(self):
        print('init mouse mover object')
    
    def move_mouse(self, deltas):
        #detection is (dx,dy)
        delta_x = deltas[0]
        delta_y = deltas[1]
        # print(f'dx,dy: {delta_x,delta_y}')
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(delta_x), int(delta_y), 0, 0)
    
    def async_smooth_linear_move_mouse(self,deltas, steps = 10, timeout = .0002):
        #should be called async
        #detection is (dx,dy)
        delta_x = deltas[0]
        delta_y = deltas[1]
        
        step_x = delta_x // steps
        step_y = delta_y // steps
        # print(f'dx,dy: {delta_x,delta_y}')
        for _ in range(steps):
            win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(step_x), int(step_y), 0, 0)
            time.sleep(timeout)
            