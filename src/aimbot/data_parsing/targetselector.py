import numpy as np
import math

class TargetSelector:
    
    def __init__(
            self,
            cfg: dict,
            detection_window_dim : tuple[int,int],
            screen_hw:tuple[int,int],
            head_toggle : bool,
            base_head_offset:float,
            target_cls_id: int,
            crosshair_cls_id: int,
            max_deltas: int,
            predict_drop:bool,
            predict_crosshair:bool,
            zoom:float,
            projectile_velocity:float,
            hFOV_degrees:float
 
            ):
        self.cfg = cfg
        self.detection_window_center = (detection_window_dim[0]//2 , detection_window_dim[1]//2)
        self.head_toggle = head_toggle
        self.target_cls_id = target_cls_id
        self.crosshair_cls_id = crosshair_cls_id
        self.max_deltas = max_deltas
        self.GRAVITY = self.cfg['targeting_settings']['gravity']#THIS VALUE IS ROBLOX DEFAULT studs/s^2
        self.screen_height = screen_hw[0]
        self.screen_width = screen_hw[1]
        self.projectile_velocity = projectile_velocity
        self.base_head_offset = base_head_offset
        self.predict_drop = predict_drop
        self.predict_crosshair = predict_crosshair
        self.zoom = zoom
        self.hfov_rad = np.deg2rad(hFOV_degrees)
        self.vfov_rad = 2 * np.arctan(np.tan(self.hfov_rad/2) * (self.screen_height/self.screen_width))
        self.DISTANCE_CONST = self.cfg['targeting_settings']['distance_calibration_factor']
        self.debug = False

    
    def _calculate_distance(self, target_height_pixels = None, target_real_height=None,
                            target_width_pixels= None, target_real_width=None):
        """
        Robust distance calculation with perspective-safe head detection and weighted uncertainty.
        """
        distances = []
        variances = []
        

        eff_vert_fov = self.vfov_rad / self.zoom  # More intuitive than 1/zoom
        eff_horiz_fov = self.hfov_rad / self.zoom

        # 4. Height-based calculation with uncertainty weighting
        if target_height_pixels:
            if target_real_height is None:
                target_real_height = self.cfg['targeting_settings']['target_real_height']
            px_per_rad = self.screen_height / eff_vert_fov
            angular_size = target_height_pixels / px_per_rad
            dist = target_real_height / (2 * math.tan(angular_size / 2))
            distances.append(dist)
            # Weight by inverse square of angular size (smaller = less reliable)
            variances.append(angular_size ** 2)

        # 5. Width-based calculation (same logic)
        if target_width_pixels:
            if target_real_width is None:
                target_real_width = self.cfg['targeting_settings']['target_real_width']
            px_per_rad = self.screen_width / eff_horiz_fov
            angular_size = target_width_pixels / px_per_rad
            dist = target_real_width / (2 * math.tan(angular_size / 2))
            distances.append(dist)
            variances.append(angular_size ** 2)

        # 6. Physics-based weighting instead of magic numbers
        weights = [1/v for v in variances] if len(variances) == 2 else [1]
        weighted_distance = sum(d * w for d, w in zip(distances, weights)) / sum(weights)

        return weighted_distance * self.DISTANCE_CONST  # Document this as "calibration factor"
    
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
        """
        Convert real-world bullet drop to screen pixels
        """

        pixels_per_radian = self.screen_height / self.vfov_rad
        angular_drop_rad = real_drop / distance
        screen_drop = angular_drop_rad * pixels_per_radian
        return screen_drop


    def _get_closest_detection(self,detections:np.ndarray,reference_point:tuple[int,int]) -> tuple:
        """
        returns the closest detection to a given reference point, which is a crosshair or the center of the screen
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
        return detections[min_idx]
        
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
                closest_crosshair = self._get_closest_detection(crosshair_detections,crosshair)
                x1,y1,x2,y2 = closest_crosshair[:4] #xywh returns center_x, center_y, width height
                crosshair = ((x1+x2)//2, (y1+y2)//2)  
        return crosshair
            
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
        
        closest_enemy_detection = self._get_closest_detection(enemy_strack_results,crosshair)

        x1,y1,x2,y2 = closest_enemy_detection[:4]
        w = x2 - x1
        h = y2 - y1
        aim_x = (x1+x2)/2
        aim_y = (y1+y2)/2
        
        if self.head_toggle:
            aim_y -= h * self.base_head_offset
            
        if self.predict_drop:
            predicted_dist = self._calculate_distance(target_height_pixels=h, target_width_pixels=w)
            predicted_bullet_travel_time = self._calculate_travel_time(predicted_dist)
            real_drop = self._calculate_bullet_drop(predicted_bullet_travel_time)
            screen_drop = self._convert_to_screen_drop(real_drop,predicted_dist)
            aim_y -= screen_drop
        
        aim_xy = (aim_x, aim_y)
        deltas = self._get_deltas(aim_xy,crosshair)

        if self.debug:
            print(f'\npredicted dist: {predicted_dist}')
            print(f'predicted bullet travel time: {predicted_bullet_travel_time}')
            print(f'predicted screen drop:{screen_drop}')
            print(f'aim_xy: {aim_xy}')
            print(f'deltas: {deltas}')

        return deltas


if __name__ == '__main__':#test
    import json
    from pathlib import Path
    config_path = Path.cwd() / "config" / "cfg.json"
    with open(config_path) as f:
        cfg = json.load(f)
    ts = TargetSelector(
        cfg=cfg,
        detection_window_dim=(320,320),
        head_toggle=True,
        target_cls_id=cfg['targeting_settings']['target_cls_id'],
        crosshair_cls_id=cfg['targeting_settings']['crosshair_cls_id'],
        max_deltas = cfg['sensitivity_settings']['max_deltas'],
        projectile_velocity=cfg['targeting_settings']['projectile_velocity'],
        base_head_offset=cfg['targeting_settings']['base_head_offset'],
        screen_hw=(1440,2560),
        zoom=cfg['targeting_settings']['zoom'],
        hFOV_degrees=cfg['targeting_settings']['fov']
    )
    dist = ts._calculate_distance(target_height_pixels=207,target_width_pixels=97)
    print(dist)
    