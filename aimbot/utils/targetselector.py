

class TargetSelector:
    
    def __init__(self,
                 hysteresis : float,
                 proximity_threshold_sq : int,
                 screen_center : tuple[int,int],
                 x_offset : int,
                 y_offset: int,
                 head_toggle : bool
                 ):
        self.hysteresis = hysteresis
        self.proximity_threshold_sq = proximity_threshold_sq
        self.screen_center = screen_center
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.head_toggle = head_toggle
        self.prev_center = None

        
    def get_target(self,detections: dict) -> tuple[int, int]:
        # Track best candidate
        best_enemy = (None, -1)  # (center, score)
        keys = list(detections.keys())
        for uid in keys:
            detection = detections[uid]
            # Calculate bounding box properties
            x1, y1, x2, y2 = detection['x1'],detection['y1'],detection['x2'],detection['y2']
            # if y1 >= self.screen_y - self.y_bottom_deadzone:
            #     continue
            width, height = x2 - x1, y2 - y1
            area = width * height
            
            # Calculate center coordinates
            center = ((x1 + x2) / 2 + self.x_offset, 
                        (y1 + y2) / 2 + self.y_offset)
            
            # Calculate proximity score
            dx = center[0] - self.screen_center[0]
            dy = center[1] - self.screen_center[1]
            dist_sq = dx * dx + dy * dy + 1e-6  # Avoid division by zero
            score = (area ** 2) / (dist_sq)
            # Apply hysteresis to previous target
            if self.prev_center:
                dx_prev = center[0] - self.prev_center[0]
                dy_prev = center[1] - self.prev_center[1]
                if (dx_prev * dx_prev + dy_prev * dy_prev) <= self.proximity_threshold_sq:
                    score *= self.hysteresis

            # Update best enemy candidate
            if score > best_enemy[1]:
                best_enemy = (center, score)

        # Select target
        new_target = best_enemy[0] if best_enemy[0] else None

        # Update previous target tracking
        if new_target:
            self.prev_center = new_target
            if self.head_toggle:
                height = y2 - y1
                if height > 85:  # Big target
                    offset_percentage = 0.35
                elif height > 25:  # Medium target
                    offset_percentage = 0.27
                else:  # Small target
                    offset_percentage = 0.15
                offset = int(height * offset_percentage)
                return (new_target[0], new_target[1] - offset)
            
        return new_target or self.screen_center