import time
from collections import deque

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
        # N timestamps span N-1 frame intervals; 0.0 while warming up so callers
        # (e.g. lead prediction) degrade to no-op instead of crashing/over-reporting.
        if len(self.buffer) < 2:
            return 0.0
        time_elapsed = self.buffer[0] - self.buffer[-1]
        if time_elapsed <= 0:
            return 0.0
        return (len(self.buffer) - 1) / time_elapsed
            

