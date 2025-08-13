import cv2
import cupy as cp
import numpy as np




class OpenCVGUI:
    
    def __init__(self,window_width,window_height):
        cv2.namedWindow("Screen Capture Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Screen Capture Detection", window_width, window_height)
    
    def render(self,frame:cp.ndarray,tracked_detections:np.ndarray):
        display_frame = cp.asnumpy(frame)
        display_frame = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
        #tracker_results (np.ndarray): [x1, y1, x2, y2, track_id, confidence, class_id, Strack idx]
        for detection in tracked_detections:
            self._draw_rectangle(display_frame,detection)

        cv2.imshow("Screen Capture Detection", display_frame)
        cv2.waitKey(1)
        
    def _draw_rectangle(self,display_frame:np.ndarray,detection:np.ndarray):
        #when drawing need int
        x1,y1,x2,y2 = map(int,detection[:4])
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 204), thickness = 1)#bb
        cv2.circle(display_frame, ((x1 + x2) // 2, (y1 + y2) // 2),2, (255, 0, 204), -1)#center of bb
        cv2.putText(display_frame, f"conf: {detection[5]:.2f}", (int(x1), int(y1) - 24),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display_frame, f"cls_id: {detection[6]}", (int(x1), int(y1) - 12),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)\
            
            
    def cleanup(self):
        cv2.destroyAllWindows()