# Add new imports at the top
from collections import deque
import numpy as np

class TargetSelector:
    def __init__(self, motion_prediction=False):
        self.motion_prediction = motion_prediction
        self.detections_buffer = deque(maxlen=5)  # Buffer last 5 frames
        self.tracked_objects = {}
        self.next_id = 0
        self.prev_center = None
        self.prev_class = None

    def _match_detections_to_tracks(self, current_detections):
        # Simple tracking using centroid proximity
        matched = {}
        unmatched = current_detections.copy()
        
        for obj_id, history in self.tracked_objects.items():
            last_pos = history['positions'][-1]
            min_dist = float('inf')
            match_idx = None
            
            for i, det in enumerate(unmatched):
                det_pos = self._get_centroid(det['bbox'])
                distance = np.linalg.norm(np.array(last_pos) - np.array(det_pos))
                
                if distance < min_dist and distance < 50:  # 50px threshold
                    min_dist = distance
                    match_idx = i
            
            if match_idx is not None:
                matched[obj_id] = unmatched.pop(match_idx)
        
        # Create new tracks for unmatched detections
        for det in unmatched:
            self.tracked_objects[self.next_id] = {
                'positions': [self._get_centroid(det['bbox'])],
                'velocities': [(0, 0)],
                'class_history': [det['class_name']]
            }
            self.next_id += 1

        return matched

    def _predict_positions(self, detections):
        if not self.motion_prediction or len(self.detections_buffer) < 2:
            return detections

        # Simple linear prediction using velocity
        for obj_id, det in detections.items():
            history = self.tracked_objects[obj_id]
            if len(history['positions']) >= 2:
                dx = history['positions'][-1][0] - history['positions'][-2][0]
                dy = history['positions'][-1][1] - history['positions'][-2][1]
                predicted_x = det['bbox'][0] + dx
                predicted_y = det['bbox'][1] + dy
                det['bbox'] = (
                    int(predicted_x), int(predicted_y),
                    det['bbox'][2] + dx, det['bbox'][3] + dy
                )
        return detections

    def _get_centroid(self, bbox):
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

    def select_target(self, detections, screen_center):
        # Update tracking information
        matched_detections = self._match_detections_to_tracks(detections)
        predicted_detections = self._predict_positions(matched_detections)
        
        # Convert tracked objects to detection format
        processed_detections = []
        for obj_id, det in predicted_detections.items():
            processed_detections.append({
                'bbox': det['bbox'],
                'class_name': self.tracked_objects[obj_id]['class_history'][-1],
                'confidence': 1.0  # Tracked objects get max confidence
            })

        # Fallback to original detection if no tracks
        if not processed_detections:
            processed_detections = detections

        # Original selection logic with optimizations
        head_targets = {}
        zombie_targets = {}
        HYSTERESIS_FACTOR = 1.1
        PROXIMITY_THRESHOLD_SQ = 50**2

        screen_center_x, screen_center_y = screen_center
        x_offset = 560  # Should be parameterized

        # Vectorized calculations
        bboxes = np.array([d['bbox'] for d in processed_detections])
        centers = np.stack([
            (bboxes[:, 0] + bboxes[:, 2]) / 2 + x_offset,
            (bboxes[:, 1] + bboxes[:, 3]) / 2
        ], axis=1)
        
        areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        deltas = centers - np.array(screen_center)
        dist_sq = np.sum(deltas**2, axis=1) + 1e-6
        scores = (areas ** 2) / (dist_sq ** 0.75)

        for i, score in enumerate(scores):
            detection = processed_detections[i]
            class_name = detection['class_name']
            
            # Apply hysteresis
            if self.prev_center and detection['class_name'] == self.prev_class:
                dx_prev = centers[i][0] - self.prev_center[0]
                dy_prev = centers[i][1] - self.prev_center[1]
                if dx_prev**2 + dy_prev**2 <= PROXIMITY_THRESHOLD_SQ:
                    score *= HYSTERESIS_FACTOR

            target_dict = head_targets if class_name == 'Head' else zombie_targets
            target_dict[tuple(centers[i])] = score

        # Rest of original selection logic
        best_head = max(head_targets.items(), key=lambda x: x[1], default=(None, 0))
        best_zombie = max(zombie_targets.items(), key=lambda x: x[1], default=(None, 0))

        new_target = best_head[0] or best_zombie[0]
        new_class = 'Head' if best_head[0] else 'Zombie' if best_zombie[0] else None

        self.prev_center = new_target if new_target else self.prev_center
        self.prev_class = new_class if new_target else self.prev_class

        return new_target or self.prev_center or screen_center