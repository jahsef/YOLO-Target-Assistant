import numpy as np
import math
from collections import OrderedDict
from ..utils.utils import log


class RingBuffer2D:
    """Pre-allocated (capacity, 2) ring buffer. O(1) push, no per-frame allocation."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buf = np.zeros((capacity, 2), dtype=np.float64)
        self.idx = 0  # next write slot

    def push(self, x: float, y: float):
        self.buf[self.idx, 0] = x
        self.buf[self.idx, 1] = y
        self.idx = (self.idx + 1) % self.capacity

    def ordered(self) -> np.ndarray:
        """(capacity, 2) copy ordered most-recent-first."""
        return np.roll(self.buf, -self.idx, axis=0)[::-1]
        
    @property
    def newest(self) -> np.ndarray:
        """Most recent entry as (2,) array."""
        return self.buf[(self.idx - 1) % self.capacity]

    def decay(self, factor: float):
        """Multiply all entries by factor (e.g. target switch momentum decay)."""
        self.buf *= factor


class _RSIChannel:
    """Single-period RSI with Wilder EMA smoothing."""

    def __init__(self, period: int):
        self.period = period
        self._prev_magnitude = None
        self._avg_gain = 0.0
        self._avg_loss = 0.0

    def update(self, magnitude: float) -> float | None:
        """Feed magnitude, return RSI value (0-100) or None if first frame."""
        if self._prev_magnitude is None:
            self._prev_magnitude = magnitude
            return None

        change = magnitude - self._prev_magnitude
        self._prev_magnitude = magnitude
        gain = max(change, 0.0)
        loss = max(-change, 0.0)

        self._avg_gain = (self._avg_gain * (self.period - 1) + gain) / self.period
        self._avg_loss = (self._avg_loss * (self.period - 1) + loss) / self.period

        if self._avg_loss == 0:
            return None

        rs = self._avg_gain / self._avg_loss
        return 100 - 100 / (1 + rs)


class RSIDampener:
    """Multi-period RSI dampener. Combines short/medium/long RSI via RMS.
    RSI near 50 = oscillating = dampen. RSI near extremes = trending = don't dampen.
    Output factor in [0.33, 1.0]."""

    def __init__(self, periods: tuple[int, ...] = (7, 14, 28), k: float = 24):
        self._channels = [_RSIChannel(p) for p in periods]
        self.k = k

    def update(self, dx: float, dy: float) -> float:
        """Feed new delta, return combined RSI factor.

        Args:
            dx, dy: raw movement deltas this frame
        Returns:
            factor in [0.33, 1.0]
        """
        magnitude = (dx**2 + dy**2)**0.5

        factors = []
        for ch in self._channels:
            rsi = ch.update(magnitude)
            if rsi is not None:
                distance = (abs(rsi - 50) / 50) #can use sub 1 power scaling to punish larger oscillation more than small oscillations
                factor = 1 - math.exp(-self.k * distance)  # ~1.0 unless RSI is right at 50
                factors.append(factor)

        if not factors:
            return 1.0

        # RMS combination, capped at 1.0
        rms = (sum(f**2 for f in factors) / len(factors))**0.5
        return max(0.33, rms)
        # return 1.0  


class TargetSelector:
    
    def __init__(
            self,
            cfg: dict,
            detection_window_dim : tuple[int,int],
            screen_hw:tuple[int,int],
            fps_tracker: object
            ):
        # Tuning constants
        #for lead (momentum based leading)
        self.MOVEMENT_BUFFER_LENGTH = 64
        self.JITTER_SCORE_THRESHOLD = 40 #soft gating, handles noise
        self.WMA_VELOCITY_THRESHOLD = 100 #soft gating, starts thresholding @ 0.5 of this threshold. the hard gate is the num u put here
        self.ZERO_DECAY = 0.90 # multiply confidence by this on zero frames (lose 15%)
        self.ZERO_RECOVERY = 0.30 # recover this fraction of headroom on non-zero frames
        self._zero_confidence = 1.0 # running confidence [0.33, 1]
        self.BASE_LEAD_SENS = 0.25 #
        self.TARGET_SWITCH_DECAY = 0.3 #keep 30% of old momentum when switching targets
        self.LEAD_X_SCALE = 1.0
        self.LEAD_Y_SCALE = 1.0#may want to lessen y momentum sometimes tbh
        self.LEAD_SENS_EMA_ALPHA = 0.12 # EMA smoothing for lead_sensitivity (lower = smoother)
        self.rsi_dampener = RSIDampener(periods=(32, 128, 512), k=24) #higher k = less dampening, more responsive. lower k = higher dampening, less responsive
        self._lead_sens_ema = self.BASE_LEAD_SENS
        
        # Physics and targeting constants
        self.FOV_DEGREES = 80
        self.GRAVITY = 128  # Roblox default studs/s^2
        self.TARGET_REAL_HEIGHT = 5
        self.TARGET_REAL_WIDTH = 3.5
        self.DISTANCE_CALIBRATION_FACTOR = 0.425
        
        

        self.cfg = cfg
        self.detection_window_center = (detection_window_dim[0]//2 , detection_window_dim[1]//2)
        self.screen_height = screen_hw[0]
        self.screen_width = screen_hw[1]
        self.base_zoom = 1.0#i have this here so in case we want to change base zoom for whatever reason
        self.zoom = self.base_zoom
        self.final_zoom = cfg['targeting_settings']['zoom']
        self.zoom_interpolation_frames = cfg['targeting_settings']['zoom_interpolation_frames']
        self.zoom_progress = 0.0
        
        self.fps_tracker = fps_tracker

        # Load targeting settings from config
        self.head_toggle = cfg['targeting_settings']['head_toggle']
        self.target_cls_id = cfg['targeting_settings']['target_cls_id']
        self.crosshair_cls_id = cfg['targeting_settings']['crosshair_cls_id']
        self.predict_drop = cfg['targeting_settings']['predict_drop']
        self.predict_crosshair = cfg['targeting_settings']['predict_crosshair']
        self.projectile_velocity = cfg['targeting_settings']['projectile_velocity']
        self.base_head_offset = cfg['targeting_settings']['base_head_offset']

        # Load sensitivity settings from config
        self.max_deltas = cfg['sensitivity_settings']['max_deltas']
        self.hfov_rad = np.deg2rad(self.FOV_DEGREES)
        self.vfov_rad = 2 * np.arctan(np.tan(self.hfov_rad/2) * (self.screen_height/self.screen_width))

        self.buffer = RingBuffer2D(self.MOVEMENT_BUFFER_LENGTH)
        self.weights = np.arange(self.MOVEMENT_BUFFER_LENGTH, 0, -1, dtype=np.float64)  # [n,n-1,...,1]
        self.weights = self.weights / self.weights.sum()  # normalize
        # self.jitter_relu_breakpoint = 5#5 or greater counts as 'jitter'
        #1.5 pixels avg jitter /frame pretyty conservative
        self.target_lru = OrderedDict()
        self.last_target_id = None #for target-aware momentum decay
        

    def update_detection_window_center(self, window_dim):
        w, h = window_dim
        self.detection_window_center = (w // 2, h // 2)
            
    def _calculate_distance(self, target_height_pixels = None, target_real_height=None,
                            target_width_pixels= None, target_real_width=None):
        """
        Robust distance calculation with perspective-safe head detection and weighted uncertainty.
        Uses aspect ratio to detect vertical occlusion (legs covered, headglitching) and
        dynamically favor width measurement when occlusion is likely.
        """
        distances = []
        variances = []


        #eff_vert_fov = self.vfov_rad / self.zoom
	# supposedly most games use tan fov scaling rather than linear
        eff_vert_fov = 2 * np.atan(np.tan(self.vfov_rad / 2) / self.zoom)
        eff_horiz_fov =  2 * np.atan(np.tan(self.hfov_rad / 2) / self.zoom)

        # Detect occlusion via aspect ratio
        # Expected ratio is TARGET_REAL_HEIGHT / TARGET_REAL_WIDTH (~1.43 for 5/3.5)
        # If observed ratio is lower, target is likely vertically occluded
        height_penalty = 1.0
        if target_height_pixels and target_width_pixels:
            expected_ratio = self.TARGET_REAL_HEIGHT / self.TARGET_REAL_WIDTH
            observed_ratio = target_height_pixels / target_width_pixels
            # ratio_factor < 1 means target appears "too wide" (vertically occluded)
            ratio_factor = observed_ratio / expected_ratio
            # Clamp between 0.3 and 1.0 - below 0.3 is extreme occlusion
            ratio_factor = max(0.3, min(1.0, ratio_factor))
            # Square it to make the penalty more aggressive for occluded targets
            height_penalty = ratio_factor ** 2

        # 4. Height-based calculation with uncertainty weighting
        if target_height_pixels:
            if target_real_height is None:
                target_real_height = self.TARGET_REAL_HEIGHT
            px_per_rad = self.screen_height / eff_vert_fov
            angular_size = target_height_pixels / px_per_rad
            dist = target_real_height / (2 * math.tan(angular_size / 2))
            distances.append(dist)
            # Weight by inverse square of angular size (smaller = less reliable)
            # Apply occlusion penalty to height variance (higher variance = less trusted)
            variances.append(angular_size ** 2 * height_penalty)

        # 5. Width-based calculation (same logic)
        if target_width_pixels:
            if target_real_width is None:
                target_real_width = self.TARGET_REAL_WIDTH
            px_per_rad = self.screen_width / eff_horiz_fov
            angular_size = target_width_pixels / px_per_rad
            dist = target_real_width / (2 * math.tan(angular_size / 2))
            distances.append(dist)
            variances.append(angular_size ** 2)

        # 6. Physics-based weighting instead of magic numbers
        weights = [1/v for v in variances] if len(variances) == 2 else [1]
        weighted_distance = sum(d * w for d, w in zip(distances, weights)) / sum(weights)

        return weighted_distance * self.DISTANCE_CALIBRATION_FACTOR
    
    def _calculate_travel_time(self, delta_x):
        time_of_flight = delta_x / self.projectile_velocity
        return time_of_flight
    
    def _calculate_bullet_drop(self, time_of_flight):
        """
        Calculate bullet drop in meters using physical projectile motion
        """
        drop = 0.5 * self.GRAVITY * (time_of_flight ** 2)
        return drop

    def _convert_to_screen_drop(self, real_drop, distance):
        eff_vert_fov = 2 * np.atan(np.tan(self.vfov_rad / 2) / self.zoom)
        pixels_per_radian = self.screen_height / eff_vert_fov
        angular_drop_rad = real_drop / distance
        screen_drop = angular_drop_rad * pixels_per_radian
        return screen_drop

    
        

    def _get_closest_detection(self,detections:np.ndarray,reference_point:tuple[int,int]) -> tuple:
        """
        returns the closest detection to a given reference point, which is a crosshair or the center of the screen
        
        uses L1 dist
        
        returns detection, l1_dist
        """
        
        xyxy_arr = detections[:,:4]
        #needs to be transposed, this tries to unpack row wise, so x1 would try to get the first row xyxy
        x1, y1, x2, y2 = xyxy_arr[:, 0:4].T
        curr_centers_x = (x1 + x2) / 2
        curr_centers_y = (y1 + y2) / 2
        dx = curr_centers_x - reference_point[0]
        dy = curr_centers_y - reference_point[1]
        sum_deltas = np.abs(dx) + np.abs(dy)
        min_idx = np.argmin(sum_deltas)
        return detections[min_idx], sum_deltas[min_idx]

    def _get_distance_to_crosshair(self,detection, crosshair):    
        """
        L1 dist from crosshair
        """
        x1, y1, x2, y2 = detection[:4]
        curr_centers_x = (x1 + x2) / 2
        curr_centers_y = (y1 + y2) / 2
        dx = curr_centers_x - crosshair[0]
        dy = curr_centers_y - crosshair[1]
        sum_deltas = np.abs(dx) + np.abs(dy)#just use l1 
        return sum_deltas
        
    def _get_highest_priority_target(self, detections, crosshair):

        for detection in detections:
            #if we have seen the target before (track_id)
            #then 
            if detection[4] in self.target_lru:
                # get distance to this one vs closest
                closest, dist_closest = self._get_closest_detection(detections, crosshair)
                dist_current = self._get_distance_to_crosshair(detection, crosshair)
                
                if dist_current < dist_closest * 1.5:  # 50% hysteresis
                    self.target_lru.move_to_end(detection[4])
                    return detection
        
        # fallback to closest if weve never seen the target before
        closest, _ = self._get_closest_detection(detections, crosshair)
        self.target_lru[closest[4]] = True
        if len(self.target_lru) > 64:
            self.target_lru.popitem(last=False)
        return closest
    
    def _get_deltas(self,detection_xy,crosshair_xy):
        x2,y2 = detection_xy[:]
        x1,y1 = crosshair_xy[:]
        deltas = (x2-x1 , y2-y1)
        if abs(deltas[0]) < self.max_deltas and abs(deltas[1])< self.max_deltas:

            return (round(deltas[0]) , round(deltas[1]))
        else:
            return (0,0)
        
    def _get_crosshair(self,detections:np.ndarray) -> tuple[int,int]:
        crosshair = self.detection_window_center
        if self.predict_crosshair:
            crosshair_mask = detections[:,6]  == self.crosshair_cls_id
            #if crosshair (red dots/scopes) is detected, get closest one
            if np.count_nonzero(crosshair_mask) != 0:
                crosshair_detections = detections[crosshair_mask]
                closest_crosshair, _ = self._get_closest_detection(crosshair_detections,crosshair)
                x1,y1,x2,y2 = closest_crosshair[:4] #xywh returns center_x, center_y, width height
                crosshair = ((x1+x2)//2, (y1+y2)//2)  
        return crosshair
    
    def update_movement_buffer(self, scaled_deltas:tuple[int,int]):
        #okay so currently this is called in aimbot.aimbot() after it gets scaled deltas from mouse movement humanizer thing
        #i think this is the cleanest solution i hope?
        self.buffer.push(scaled_deltas[0], scaled_deltas[1])
    
    def reset_zoom(self):
        """resets zoom for when you aren't right clicking anymore"""
        self.zoom = self.base_zoom
        self.zoom_progress = 0.0
        log(f'ZOOM RESET', "DEBUG")

    def update_zoom_interpolation(self):
        if self.zoom_progress < 1.0:
            self.zoom_progress += 1.0 / self.zoom_interpolation_frames
            self.zoom_progress = min(self.zoom_progress, 1.0)

            # Easing functions - uncomment desired curve:
            # t = self.zoom_progress  # linear
            # t = self.zoom_progress ** 2  # quadratic ease-in (slow start, fast end)
            t = self.zoom_progress ** 3  # cubic ease-in
            # t = 1 - (1 - self.zoom_progress) ** 2  # quadratic ease-out (fast start, slow end)
            # t = 1 - (1 - self.zoom_progress) ** 3  # cubic ease-out

            self.zoom = self.base_zoom + (self.final_zoom - self.base_zoom) * t
        log(f'zoom: {self.zoom}', "DEBUG")
        
    def get_deltas(self,detections:np.ndarray):
        """
        processes detection array of shape (n, 8) where columns are:
        
        [x1, y1, x2, y2, track_id, confidence, class_id, Strack idx]
        """
        cls_mask = detections[:,6] == self.target_cls_id
        if np.count_nonzero(cls_mask) != 0:#checking if any targets exist
            enemy_strack_results = detections[cls_mask]
        else:
            return (0,0) #no enemies 

        
        crosshair = self._get_crosshair(detections)    
        if self.cfg['targeting_settings']['prioritize_oldest']:
            #if we want to prio oldest detection, else we get closest
            highest_prio_enemy_detection = self._get_highest_priority_target(enemy_strack_results,crosshair)
        else:
            highest_prio_enemy_detection,_ = self._get_closest_detection(enemy_strack_results, crosshair)

        x1,y1,x2,y2 = highest_prio_enemy_detection[:4]
        w = x2 - x1
        h = y2 - y1
        aim_x = (x1+x2)/2#getting center
        aim_y = (y1+y2)/2
        
        if self.head_toggle:
            aim_y -= h * self.base_head_offset
        


        if self.predict_drop:
            predicted_dist = self._calculate_distance(target_height_pixels=h, target_width_pixels=w)
            predicted_bullet_travel_time = self._calculate_travel_time(predicted_dist)
            real_drop = self._calculate_bullet_drop(predicted_bullet_travel_time)
            screen_drop = self._convert_to_screen_drop(real_drop,predicted_dist)
            aim_y -= screen_drop
            log(f'screen_drop: {screen_drop}', "DEBUG")
            #if leading target is on then predicting drop is almost definitely on 
            if self.cfg['targeting_settings']['lead_target']:
                #lead target function handles all the sensitivity lead scaling etc
                unscaled_deltas = self._get_deltas((aim_x, aim_y), crosshair_xy=crosshair)
                target_id = highest_prio_enemy_detection[4]
                lead_pixels_x, lead_pixels_y = self.lead_target(predicted_bullet_travel_time=predicted_bullet_travel_time, unscaled_deltas=unscaled_deltas, target_id=target_id)
        
                aim_x += lead_pixels_x#sort of acts like a momentum factor while also leading shots
                aim_y += lead_pixels_y
        
        aim_xy = (aim_x, aim_y) #since tuples immutable
        deltas = self._get_deltas(aim_xy,crosshair)

        log(f'\npredicted dist: {predicted_dist}', "DEBUG")
        log(f'predicted bullet travel time: {predicted_bullet_travel_time}', "DEBUG")
        log(f'predicted screen drop:{screen_drop}', "DEBUG")
        log(f'aim_xy: {aim_xy}', "DEBUG")
        log(f'deltas: {deltas}', "DEBUG")

        return deltas
    
    def lead_target(self, predicted_bullet_travel_time:int, unscaled_deltas:tuple[float,float], target_id:int):
        """
        Calculates lead offset based on tracked mouse velocity

        raw_aim_xy is tuple of raw deltas

        if tracking is too unstable, return (0,0) which is lead_x, lead_y
        """
        # Target switch detection - decay old momentum when switching targets
        if target_id != self.last_target_id:
            self.buffer.decay(self.TARGET_SWITCH_DECAY)
            log(f'target switch detected: {self.last_target_id} -> {target_id}, decayed buffer', "DEBUG")
            self.last_target_id = target_id

        arr_buffer = self.buffer.ordered()
        lead_sensitivity = self.BASE_LEAD_SENS

        # EMA-style zero confidence: decay on zero frames, recover on non-zero, floor 0.33
        if unscaled_deltas == (0, 0):
            self._zero_confidence = max(self._zero_confidence * self.ZERO_DECAY, 0.33)
        else:
            self._zero_confidence += self.ZERO_RECOVERY * (1.0 - self._zero_confidence)
        zero_confidence = self._zero_confidence

        diffs = arr_buffer[:-1] - arr_buffer[1:]  # (n-1, 2)
        squared_diffs = diffs ** 2
        jitter_score = np.sqrt(squared_diffs[:, 0].mean() + squared_diffs[:, 1].mean())
        jitter_factor = max(0.33, (max((self.JITTER_SCORE_THRESHOLD - jitter_score),0)/self.JITTER_SCORE_THRESHOLD)**(1/2))

        wma_velocity_x = (arr_buffer[:, 0] * self.weights).sum()
        wma_velocity_y = (arr_buffer[:, 1] * self.weights).sum()
        wma_velocity = (wma_velocity_x**2 + wma_velocity_y**2)**(0.5)
        # soft gate: linear ramp from 1.0 at half threshold to 0.0 at full threshold, floor 0.33
        wma_factor = max(0.33, min(1.0, 2.0 * (1.0 - wma_velocity / self.WMA_VELOCITY_THRESHOLD)))

        lead_pixels_x = wma_velocity_x * self.fps_tracker.get_fps() * predicted_bullet_travel_time #pixels/frame * frames/s * s => pixels
        lead_pixels_y = wma_velocity_y * self.fps_tracker.get_fps() * predicted_bullet_travel_time

        rsi_factor = self.rsi_dampener.update(unscaled_deltas[0], unscaled_deltas[1])

        # combine all multipliers
        raw_lead_sens = lead_sensitivity * zero_confidence * jitter_factor * wma_factor * rsi_factor

        # EMA smoothing to stabilize lead_sensitivity across frames
        self._lead_sens_ema += self.LEAD_SENS_EMA_ALPHA * (raw_lead_sens - self._lead_sens_ema)
        lead_sensitivity = self._lead_sens_ema

        #BELOW IS USED FOR DEBUG INFO FOR GRAPHING LEAVE COMMENTED
        with open('dampening_factors.csv', 'a') as f:
            f.write(f'{zero_confidence},{jitter_factor},{wma_factor},{rsi_factor},{raw_lead_sens},{lead_sensitivity},{wma_velocity},{jitter_score},{lead_pixels_x},{lead_pixels_y},{unscaled_deltas[0]},{unscaled_deltas[1]}\n')

        lead_sensitivity_x = lead_sensitivity*self.LEAD_X_SCALE
        lead_sensitivity_y = lead_sensitivity*self.LEAD_Y_SCALE
        return (lead_pixels_x* lead_sensitivity_x, lead_pixels_y * lead_sensitivity_y)

if __name__ == '__main__':
    # python -m src.aimbot.data_parsing.targetselector
    import time
    import json
    from pathlib import Path

    config_path = Path.cwd() / "config" / "cfg.json"
    with open(config_path) as f:
        cfg = json.load(f)

    class FakeFPS:
        def get_fps(self): return 144.0

    ts = TargetSelector(cfg=cfg, detection_window_dim=(640, 640), screen_hw=(1440, 2560), fps_tracker=FakeFPS())

    # fake detections: 5 enemies + 1 crosshair, (n, 8) [x1,y1,x2,y2,track_id,conf,cls_id,strack_idx]
    detections = np.array([
        [100, 200, 130, 260, 1, 0.9, 0, 0],
        [300, 150, 340, 230, 2, 0.85, 0, 1],
        [200, 300, 225, 355, 3, 0.7, 0, 2],
        [400, 100, 430, 170, 4, 0.8, 0, 3],
        [150, 250, 180, 310, 5, 0.75, 0, 4],
        [318, 318, 322, 322, 6, 0.95, 2, 5],
    ], dtype=np.float32)

    # fill buffer with some movement data
    for i in range(48):
        ts.buffer.push(np.sin(i * 0.3) * 2, np.cos(i * 0.3) * 1.5)

    N_SAMPLES = 256
    N_ITERS = 512

    def bench(name, fn):
        sprints = np.empty(N_SAMPLES)
        for s in range(N_SAMPLES):
            t0 = time.perf_counter_ns()
            for _ in range(N_ITERS):
                fn()
            sprints[s] = (time.perf_counter_ns() - t0) / N_ITERS / 1e3
        mean = sprints.mean()
        std = sprints.std(ddof=1)
        ci95 = 1.96 * std / np.sqrt(N_SAMPLES)
        print(f"  {name:<35}: {mean:7.2f} ±{std:5.2f} µs  CI95=[{mean-ci95:.2f}, {mean+ci95:.2f}]")

    enemy_dets = detections[detections[:, 6] == 0]
    crosshair = (320, 320)

    print(f"TargetSelector benchmarks  ({N_SAMPLES} sprints × {N_ITERS} iters)\n")

    bench("RingBuffer2D.push",             lambda: ts.buffer.push(1.0, 2.0))
    bench("RingBuffer2D.ordered",          lambda: ts.buffer.ordered())
    bench("RingBuffer2D.decay",            lambda: ts.buffer.decay(0.3))
    bench("_get_closest_detection",        lambda: ts._get_closest_detection(enemy_dets, crosshair))
    bench("_get_highest_priority_target",  lambda: ts._get_highest_priority_target(enemy_dets, crosshair))
    bench("_calculate_distance",           lambda: ts._calculate_distance(target_height_pixels=60, target_width_pixels=30))
    bench("_get_crosshair",               lambda: ts._get_crosshair(detections))
    bench("get_deltas",                    lambda: ts.get_deltas(detections))
    bench("lead_target",                   lambda: ts.lead_target(predicted_bullet_travel_time=0.05, unscaled_deltas=(3.0, -2.0), target_id=1))

    # ── lead_target internals breakdown ──
    print(f"\nlead_target internals:\n")
    arr_buf = ts.buffer.ordered()
    diffs = arr_buf[:-1] - arr_buf[1:]
    current_xy = np.array([5.0, -3.0])
    current_norm = np.linalg.norm(current_xy)
    recent = arr_buf[:8]

    bench("  ordered()",                     lambda: ts.buffer.ordered())
    bench("  zero_ratio",                    lambda: np.sum(np.all(arr_buf == 0, axis=1)))
    bench("  jitter (diffs+sqrt+mean)",      lambda: np.sqrt((diffs**2)[:, 0].mean() + (diffs**2)[:, 1].mean()))
    bench("  wma dot product",               lambda: (arr_buf[:, 0] * ts.weights).sum())
    def _cos_sim_vec():
        dots = recent @ current_xy
        norms = np.linalg.norm(recent, axis=1) * current_norm + 1e-6
        return (dots / norms).mean()
    bench("  cos_sim (8-frame vectorized)",  _cos_sim_vec)
    bench("  np.linalg.norm (single)",       lambda: np.linalg.norm(current_xy))
    bench("  np.dot (single 2d)",            lambda: np.dot(current_xy, recent[0]))
    bench("  np.exp (dampening)",            lambda: np.exp(-0.5))
