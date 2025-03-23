

class TargetSelector:
    
    def __init__(self,
                 screen_center : tuple[int,int],
                 x_offset : int,
                 y_offset: int,
                 head_toggle : bool,
                 target_dimensions
                 ):
        self.screen_center = screen_center
        self.x_offset = x_offset#offset needed because capture dim could be lower than real screen dim
        self.y_offset = y_offset
        self.head_toggle = head_toggle
        x_center = screen_center[0]
        y_center = screen_center[1]
        self.target_dimensions = target_dimensions
        self.target_region = [x_center - target_dimensions[0],y_center - target_dimensions[1],x_center + target_dimensions[0],y_center + target_dimensions[1]]#x +- target_dim[0]
        
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
        
    def return_deltas(self,detections) -> tuple[int, int]:
        #returns the closest target thats also within the tracking region
        shortest_dist =  float('inf')
        target_deltas = None
        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = detection[:4]
            head_offset = 0
            if self.head_toggle:
                #slower to run this in here but like max 16 detections anyway
                head_offset = self._get_head_offset(detection)
                
            curr_center = ((x1 + x2) / 2 + self.x_offset, 
                        (y1 + y2) / 2 + self.y_offset - head_offset)
            dx = curr_center[0] - self.screen_center[0]
            dy = curr_center[1] - self.screen_center[1]
            abs_dx, abs_dy = abs(dx), abs(dy)
            if abs_dx <= self.target_dimensions[0] and abs_dy <= self.target_dimensions[1]:
                sum_deltas = abs_dx + abs_dx
                if sum_deltas < shortest_dist:
                    shortest_dist = sum_deltas
                    target_deltas = (dx,dy)
            else:
                continue
                
        # print(type(target_deltas))
        # print(target_deltas)
        return target_deltas if target_deltas else (0,0)