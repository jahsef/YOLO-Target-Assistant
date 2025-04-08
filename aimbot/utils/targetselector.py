import numpy as np
import math

class TargetSelector:
    
    def __init__(
            self,
            detection_window_dim : tuple[int,int],
            head_toggle : bool,
            target_cls_id,
            crosshair_cls_id,
            max_deltas,
            sensitivity,
            zoom,
            projectile_velocity = 2000,
            base_head_offset = .33,
            screen_hw = (1440,2560),
            hFOV_degrees = 105
            ):
        self.detection_window_center = (detection_window_dim[0]//2 , detection_window_dim[1]//2)
        self.head_toggle = head_toggle
        self.target_cls_id = target_cls_id
        self.crosshair_cls_id = crosshair_cls_id
        self.max_deltas = max_deltas
        self.sensitivity = sensitivity
        # self.DIST_CALC_CONST = 25000
        self.GRAVITY = 196.2#THIS VALUE IS ROBLOX DEFAULT studs/s^2
        self.screen_height = screen_hw[0]
        self.screen_width = screen_hw[1]
        self.projectile_velocity = projectile_velocity
        self.base_head_offset = base_head_offset
        self.zoom = zoom
        
        self.hfov_rad = np.deg2rad(hFOV_degrees)
        self.vfov_rad = 2 * np.arctan(np.tan(self.hfov_rad/2) * (self.screen_height/self.screen_width))
        print(f'hfov: {np.rad2deg(self.hfov_rad)}')
        print(f'vfov: {np.rad2deg(self.vfov_rad)}')
        self.DISTANCE_CONST =.5

    def _calculate_distance(self, target_height_pixels=None, target_real_height=5,
                            target_width_pixels=None, target_real_width=3.5):
        """
        Enhanced distance calculator with partial visibility handling
        """
        # Default human proportions (height/width ratio)
        EXPECTED_RATIO = target_real_height / target_real_width  # ~3.4 for humans
        HEAD_RATIO_THRESHOLD = 0.4 * EXPECTED_RATIO  # Threshold for head detection
        
        distances = []
        weights = []
        use_head_dimensions = False

        # Detect partial visibility using aspect ratio
        if target_height_pixels and target_width_pixels:
            actual_ratio = target_height_pixels / target_width_pixels
            
            # Check for head-like proportions
            if actual_ratio < HEAD_RATIO_THRESHOLD:
                use_head_dimensions = True
                target_real_height = 0.25  # Average head height
                target_real_width = 0.3    # Average head width

        # Height-based calculation
        if target_height_pixels and target_real_height:
            eff_vert_fov = self.vfov_rad * (1/self.zoom)
            px_per_rad = self.screen_height / eff_vert_fov
            angular_size = target_height_pixels / px_per_rad
            dist = target_real_height / (2 * math.tan(angular_size/2))
            distances.append(dist)
            weights.append(0.4 if use_head_dimensions else 0.6)

        # Width-based calculation
        if target_width_pixels and target_real_width:
            eff_horiz_fov = self.hfov_rad * (1/self.zoom)
            px_per_rad = self.screen_width / eff_horiz_fov
            angular_size = target_width_pixels / px_per_rad
            dist = target_real_width / (2 * math.tan(angular_size/2))
            distances.append(dist)
            weights.append(0.6 if use_head_dimensions else 0.4)

        if not distances:
            raise ValueError("Provide at least one valid dimension")

        # Final distance calculation
        if len(distances) == 2:
            # Prioritize width when head-like proportions detected
            base_distance = (distances[0]*weights[0] + distances[1]*weights[1]) / sum(weights)
        else:
            base_distance = distances[0]

        return base_distance * self.DISTANCE_CONST
    
    
    def _calculate_bullet_drop(self, distance):
        """
        Calculate bullet drop in meters using physical projectile motion
        """
        time_of_flight = distance / self.projectile_velocity
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
    
    
    #returns closest detection (sum of deltas not actual distance)
    def _get_closest_detection(self,detections,reference_point):
        #reference point is center of screen for crosshair calculations
        #reference point is crosshair if detected
        x1, y1, x2, y2 = detections[:, 0], detections[:, 1], detections[:, 2] ,detections[:, 3]
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
            return (int(deltas[0]* self.sensitivity) , int(deltas[1] * self.sensitivity) )
        else:
            return (0,0)
        
    def get_deltas(self,detections):
        cls_mask = detections[:,6] == self.target_cls_id

        if np.count_nonzero(cls_mask) != 0:
            enemy_detections = detections[cls_mask]
        else:
            return (0,0) #no enemies 
        
        crosshair_mask = detections[:,6] == self.crosshair_cls_id
        crosshair = self.detection_window_center
        #if crosshair (red dots/scopes) is detected, get closest one
        if np.count_nonzero(crosshair_mask) != 0:
            crosshair_detections = detections[crosshair_mask]
            closest_crosshair = self._get_closest_detection(crosshair_detections,crosshair)
            x1,y1,x2,y2 = closest_crosshair[:4]
            crosshair = ((x1+x2)/2, (y1+y2)/2)  
        # x1, y1, x2, y2 = enemy_detections[:, 0], enemy_detections[:, 1], enemy_detections[:, 2], enemy_detections[:, 3]
        # heights = y2 - y1
        
        closest_detection = self._get_closest_detection(enemy_detections,crosshair)
        
        #doesnt return true least mouse movement, just aims towards the closest enemy
        x1, y1, x2, y2 = closest_detection[:4]
        # print(closest_detection)
        h = y2 - y1
        w = x2 - x1
        # print(f'predicted,real,screen:')
        predicted_dist = self._calculate_distance(target_height_pixels=h, target_width_pixels=w)
        # print(predicted_dist)
        real_drop = self._calculate_bullet_drop(predicted_dist)
        # print(real_drop)
        screen_drop = self._convert_to_screen_drop(real_drop,predicted_dist)
        # print(screen_drop)
        # print(f'predicted dist: {predicted_dist}')
        # pixel_drop = self._predict_bullet_drop(predicted_dist,self.projectile_velocity)
        # print(f'predicted 3d drop:{real_drop}')
        # print(f'predicted screen drop:{screen_drop}')
        if self.head_toggle:
            pixel_head_offset = h * self.base_head_offset
            # print(f'head_offset: {pixel_head_offset}')
        else:
            pixel_head_offset = 0
        center_x = (x1+x2)/2
        center_y = (y1+y2)/2  
        # print(center_y)
        offset_y = center_y - screen_drop - pixel_head_offset
        # print(offset_y)
        center_detection = (center_x, offset_y)
        
        deltas = self._get_deltas(center_detection,crosshair)
        # print(deltas)
        return deltas


if __name__ == '__main__':#test
    ts = TargetSelector(detection_window_dim=(320,320),head_toggle=True,target_cls_id=0,crosshair_cls_id=2,
                        max_deltas = 16,sensitivity=1,zoom=10,projectile_velocity=2000,base_head_offset=.33,screen_hw=(1440,2560),hFOV_degrees=105)
    dist = ts._calculate_distance(target_height_pixels=207,target_width_pixels=97)
    print(dist)
    