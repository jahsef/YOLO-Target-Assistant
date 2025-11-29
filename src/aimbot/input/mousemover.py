import win32api
import win32con
import random
import math
import logging
from ..utils.utils import log

class MouseMover:
    def __init__(self,overall_sens:float,sens_scaling:float,max_deltas:int,jitter_strength:float,overshoot_strength:float,overshoot_chance:float):
        """_summary_

        Args:
            sensitivity (_type_): sensitivity 0-2 (more than 2 not recommended)
            max_deltas (_int_): max pixels to move

        """
        self.overall_sens = overall_sens
        self.sens_scaling = sens_scaling
        self.max_deltas = max_deltas
        self.jitter_strength = jitter_strength
        self.overshoot_strength = overshoot_strength
        self.overshoot_chance = overshoot_chance
    
    def move_mouse_raw(self,dx:int,dy:int):
        """
        moves mouse with raw deltas no scaling        
        """
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, dx = int(dx), dy = int(dy))
    
    def move_mouse_humanized(self,dx:float,dy:float) -> tuple[int, int]:
        """
        takes raw deltas, scales and humanizes them and moves mouse, returns scaled deltas too
        """
        scaled_x, scaled_y = self._scale_delta(dx), self._scale_delta(dy)
        humanized_xy = self._humanize_movement(scaled_x,scaled_y)
        round_x, round_y = round(humanized_xy[0]), round(humanized_xy[1])
        log(f'moving mouse: {round_x,round_y}', "DEBUG")
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, round_x, round_y)
        return (round_x ,round_y)
        
    # Apply confidence scaling (0.0 to 1.0)
    # smoothed_dx = dx * conf  # Higher conf = stronger aim
    # smoothed_dy = dy * conf

    # # Inverse confidence jitter (low conf = more wobble)
    # jitter_scale = (1.0 - conf) * max_jitter
    # smoothed_dx += random.uniform(-jitter_scale, jitter_scale)
    # smoothed_dy += random.uniform(-jitter_scale, jitter_scale)
    
    
    
    def _humanize_movement(self,dx:float, dy:float) -> tuple[float,float]:
        
        jitter = self.jitter_strength * (abs(dx) + abs(dy))
        dx += random.uniform(-jitter, jitter)
        dy += random.uniform(-jitter, jitter)
        
        if random.random() < self.overshoot_chance:
            dx *= self.overshoot_strength
            dy *= self.overshoot_strength
        
        return (dx, dy)


    def _scale_delta(self, delta):
        """
        - Low deltas (near zero) are scaled minimally (close to raw).
        - Higher sensitivity amplifies scaling, but does NOT invert behavior.
        - Smooth exponential transition.
        """
        x = abs(delta) / self.max_deltas  # Normalized delta (0 to 1)
        
        # How much to blend toward scaling (tune for desired curve)
        blend = 1 - math.exp(-x * 2.5)  # 3.0 = curve steepness (higher = sharper transition)
        
        # Apply sensitivity (higher sensitivity = more scaling, but not inverted)
        
        return self.overall_sens * delta * (1.0 + (self.sens_scaling - 1.0) * blend)