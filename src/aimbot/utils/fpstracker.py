import time
from collections import deque
import sys

class FPSTracker:
    def __init__(self, update_interval=1):

        self.last_update = time.perf_counter()
        self.update_interval = update_interval
        self.fps_buffer_len = 69
        self.buffer = deque(maxlen=self.fps_buffer_len)

    def update(self):
        current_time = time.perf_counter()
        self.buffer.appendleft(current_time)
    
    def print_fps(self):
        """prints fps according to update interval"""
        current_time = time.perf_counter()
        if current_time - self.last_update >= self.update_interval:
            fps = self.get_fps()
            print(f'FPS: {fps:.2f}')
            self.last_update = current_time
    
    def get_fps(self):
        time_elapsed = self.buffer[0] - self.buffer[-1]#index for peek
        return self.fps_buffer_len / time_elapsed
            

