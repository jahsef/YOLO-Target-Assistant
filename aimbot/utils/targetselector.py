import numpy as np

class TargetSelector:
    
    def __init__(self,
                 detection_window_dim : tuple[int,int],
                #  x_offset : int,
                #  y_offset: int,
                 head_toggle : bool,
                 target_cls_id,
                 crosshair_cls_id
                 ):
        self.detection_window_center = (detection_window_dim[0]//2 , detection_window_dim[1]//2)
        # self.x_offset = x_offset#offset needed because capture dim could be lower than real screen dim
        # self.y_offset = y_offset
        self.head_toggle = head_toggle
        self.target_cls_id = target_cls_id
        self.crosshair_cls_id = crosshair_cls_id
        # self.target_dimensions = target_dimensions
        # self.target_region = [x_center - target_dimensions[0],y_center - target_dimensions[1],x_center + target_dimensions[0],y_center + target_dimensions[1]]#x +- target_dim[0]
        #
    def _get_head_offset(self,target):
        height = target[3] - target[1]
        # print(height)
        if height > 85:  # Big target
            offset_percentage = 0.35
        elif height > 25:  # Medium target
            offset_percentage = 0.28
        else:  # Small target
            offset_percentage = 0.15
        offset = int(height * offset_percentage)
        return offset
        # print(f'returning: {target_center[0], target_center[1] - offset}')
        
        #returns closest detection (sum of deltas not actual distance)
    def _get_min_deltas(self,detections,crosshair,get_detection: bool = False,head_offsets = 0):
        x1, y1, x2, y2 = detections[:, 0], detections[:, 1], detections[:, 2], detections[:, 3]
        curr_centers_x = (x1 + x2) / 2
        curr_centers_y = (y1 + y2) / 2 - head_offsets
        dx = curr_centers_x - crosshair[0]
        dy = curr_centers_y - crosshair[1]
        sum_deltas = np.abs(dx) + np.abs(dy)
        min_idx = np.argmin(sum_deltas)
        if get_detection:
            return detections[min_idx]
        #returns deltas otherwise
        deltas = (int(dx[min_idx]), int(dy[min_idx]))
        return deltas
    def return_deltas_vectorized(self,detections):
        cls_mask = detections[:,6] == self.target_cls_id

        if np.count_nonzero(cls_mask) != 0:
            enemy_detections = detections[cls_mask]
        else:
            return None #no enemies 
        crosshair_mask = detections[:,6] == self.crosshair_cls_id
        crosshair = self.detection_window_center
        #if crosshair (red dots/scopes) is detected, get closest one
        if np.count_nonzero(crosshair_mask) != 0:
            crosshair_detections = detections[crosshair_mask]
            closest_crosshair = self._get_min_deltas(crosshair_detections,crosshair,get_detection=True)
            x1,y1,x2,y2 = closest_crosshair[:4]
            crosshair = ((x1+x2)/2, (y1+y2)/2)  

        x1, y1, x2, y2 = enemy_detections[:, 0], enemy_detections[:, 1], enemy_detections[:, 2], enemy_detections[:, 3]
        heights = y2 - y1
        
        if self.head_toggle:
            offset_percentages = np.where(
                heights > 85, 0.35,
                np.where(heights > 25, 0.28, 0.20)
            )
            head_offsets = (heights * offset_percentages)
        else:
            head_offsets = 0

        deltas = self._get_min_deltas(enemy_detections,crosshair,get_detection=False,head_offsets=head_offsets)
        return deltas
    
    # def return_deltas(self,detections:np.ndarray) -> tuple[int, int]:
        
    #     #returns the closest target thats also within the tracking region
    #     shortest_dist =  float('inf')
    #     target_deltas = None
    #     # x1, y1, x2, y2 = detections[::0], detections[::1], detections[::2], detections[::3]
        
    #     for i, detection in enumerate(detections):
    #         x1, y1, x2, y2 = detection[:4]
    #         head_offset = 0
    #         if self.head_toggle:
    #             #slower to run this in here but like max 16 detections anyway
    #             head_offset = self._get_head_offset(detection)
                
    #         curr_center = ((x1 + x2) / 2 + self.x_offset, 
    #                     (y1 + y2) / 2 + self.y_offset - head_offset)
    #         dx = curr_center[0] - self.screen_center[0]
    #         dy = curr_center[1] - self.screen_center[1]
    #         abs_dx, abs_dy = abs(dx), abs(dy)

    #         sum_deltas = abs_dx + abs_dy
    #         if sum_deltas < shortest_dist:
    #             shortest_dist = sum_deltas
    #             target_deltas = (dx,dy)

                
    #     # print(type(target_deltas))
    #     # print(target_deltas)
    #     return target_deltas if target_deltas else (0,0)