class TargetTracker:
    def __init__(self, screen_width=2560, screen_height=1440):
        self.screen_center = (screen_width/2, screen_height/2)
        self.tracking_history = []
        self.current_target = None
        self.x_offset = 560  # From your original code

    
    def select_target(self, detections):
        # 1. Update tracking history
        self._update_tracking(detections)
        
        # 2. Score candidates with temporal consistency
        candidates = self._score_candidates()
        
        # 3. Select best target with inertia
        return self._select_best_target(candidates)

    def _update_tracking(self, detections):
        # Maintain tracking history of last 5 frames (adjust as needed)
        self.tracking_history = (self.tracking_history + [detections])[-5:]

    def _score_candidates(self):
        candidate_scores = {}
        
        # Analyze last 3 frames for stability
        for frame in self.tracking_history[-3:]:
            for detection in frame:
                center = self._get_center(detection)
                score = self._calculate_score(detection, center)
                
                # Temporal scoring: Older detections decay in influence
                age_weight = 0.8 ** (len(self.tracking_history) - 1)
                candidate_scores[center] = candidate_scores.get(center, 0) + score * age_weight
                
                # Bonus for persistent appearances
                if center in candidate_scores:
                    candidate_scores[center] *= 1.2
                    
        return candidate_scores
    
    def _get_center(self, detection):
        x1, y1, x2, y2 = detection['bbox']
        return ((x1 + x2)/2 + self.x_offset, (y1 + y2)/2)

    def _calculate_score(self, detection, center):
        # Extract bounding box coordinates from detection
        x1, y1, x2, y2 = detection['bbox']
        
        # Spatial scoring components
        area = (x2 - x1) * (y2 - y1)
        dx = center[0] - self.screen_center[0]
        dy = center[1] - self.screen_center[1]
        distance = (dx**2 + dy**2) ** 0.5
        
        # Class priority weights
        class_weight = 2.0 if detection['class'] == 'Head' else 1.0
        
        # Motion prediction bonus
        motion_bonus = self._calculate_motion_bonus(center)
        
        return (area ** 1.5) / (distance ** 0.5) * class_weight * motion_bonus

    def _calculate_motion_bonus(self, current_center):
        # Predict position based on previous movement
        if not self.current_target:
            return 1.0
            
        # Calculate velocity-based prediction
        predicted_pos = self._predict_position()
        dx = current_center[0] - predicted_pos[0]
        dy = current_center[1] - predicted_pos[1]
        distance_error = (dx**2 + dy**2) ** 0.5
        
        # Scale bonus based on prediction accuracy
        return max(0.5, 1.0 - (distance_error / 100))  # 100px tolerance

    def _predict_position(self):
        # Simple linear prediction using last 2 positions
        if len(self.tracking_history) < 2:
            return self.current_target
            
        prev1 = self.tracking_history[-1]
        prev2 = self.tracking_history[-2]
        
        dx = prev1[0] - prev2[0]
        dy = prev1[1] - prev2[1]
        return (prev1[0] + dx, prev1[1] + dy)

    def _select_best_target(self, candidates):
        if not candidates:
            return self.screen_center
            
        # Get top 2 candidates
        sorted_candidates = sorted(candidates.items(), key=lambda x: -x[1])
        best = sorted_candidates[0]
        second_best = sorted_candidates[1] if len(sorted_candidates) > 1 else (None, 0)
        
        # Require 25% better score to switch targets
        if self.current_target and best[0] != self.current_target:
            if best[1] < (self.current_target[1] * 1.25):
                return self.current_target[0]
            
        # Update current target
        self.current_target = (best[0], best[1])
        return best[0]