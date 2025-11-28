import cv2
import cupy as cp
import numpy as np




class OpenCVGUI:
    
    def __init__(self,window_width:int,window_height:int, config:dict):
        cv2.namedWindow("Screen Capture Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Screen Capture Detection", window_width, window_height)
        self.config = config
    def render(self,frame:cp.ndarray,tracked_detections:np.ndarray, raw_deltas:tuple[int,int], scaled_deltas:tuple[int,int]):
        display_frame = cp.asnumpy(frame)
        display_frame = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR) #since opencv uses bgr
        #tracker_results (np.ndarray): [x1, y1, x2, y2, track_id, confidence, class_id, Strack idx]
        for detection in tracked_detections:
            self._draw_rectangle(display_frame,detection)


        if raw_deltas != (0,0):

            #extra loop here but who cares its just debug code
            crosshair = None
            for detection in tracked_detections:
                if detection[6] == self.config['targeting_settings']['crosshair_cls_id']:
                    crosshair = detection
                    
            if crosshair is not None:
                #if we have detected crosshair we point vectors from crosshair
                center = (int((crosshair[0] + crosshair[2]) // 2), int((crosshair[1] + crosshair[3]) // 2))
            else:
                #otherwise we just use center screen heuristic
                #display frame is hw, center should be xy
                center = (display_frame.shape[1]//2, display_frame.shape[0]//2)
            
            vector_endpoint = (center[0] + raw_deltas[0], center[1] + raw_deltas[1])
            cv2.line(display_frame,center, vector_endpoint, color = (255,0,204), thickness=2)
            
            #scaled deltas
            center = (center[0], center[1] - 10)
            vector_endpoint = (center[0] + scaled_deltas[0], center[1] + scaled_deltas[1])
            cv2.line(display_frame,center, vector_endpoint, color = (255, 0, 255), thickness=2)
            

        cv2.imshow("Screen Capture Detection", display_frame)
        cv2.waitKey(1)
        
    def _draw_rectangle(self,display_frame:np.ndarray,detection:np.ndarray):
        #when drawing need int
        x1,y1,x2,y2 = map(int,detection[:4])
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 204), thickness = 1)#bb
        center_bb = ((x1 + x2) // 2, (y1 + y2) // 2)
        cv2.circle(display_frame, center_bb,2, (255, 0, 204), -1)#center of bb
        if detection[6] == self.config['targeting_settings']['target_cls_id']:
            height = y2-y1
            cv2.circle(display_frame, (center_bb[0], int(center_bb[1] - height* self.config['targeting_settings']['base_head_offset'])),2, (255, 0, 204), -1)#head offset set by the user
        cv2.putText(display_frame, f"conf: {detection[5]:.2f}", (int(x1), int(y1) - 24),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display_frame, f"cls_id: {detection[6]}", (int(x1), int(y1) - 12),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            
    def cleanup(self):
        cv2.destroyAllWindows()