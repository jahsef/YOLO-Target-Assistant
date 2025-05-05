import cv2
import cupy as cp
import numpy as np
from typing import Optional


class GUIManager:
    
    def __init__(self, config:dict,hw_capture:tuple[int,int]):
        """
        Args:
            config (dict): config dict (only the gui_settings portion) loaded from the config json file passed from aimbot.py
        """
        
        self.opencv_enabled = config.get("opencv_render", False)  
        self.dpg_enabled = config.get("dpg_overlay", False) 
        print(f'starting gui manager, opencv: {self.opencv_enabled}, dpg: {self.dpg_enabled}')
        
        if not self.opencv_enabled and not self.dpg_enabled:
            return None
        
        window_height, window_width = hw_capture  
        if self.opencv_enabled:
            cv2.namedWindow("Screen Capture Detection", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Screen Capture Detection", window_width, window_height)

    def render_gui(self,frame:Optional[cp.ndarray],tracker_results:np.ndarray):
        """_summary_

        Args:
            frame (Optional[cp.ndarray]): i think frame is only needed for opencv render
            tracker_results (np.ndarray): [x1, y1, x2, y2, track_id, confidence, class_id, Strack idx]
        """
        if self.opencv_enabled:
            display_frame = cp.asnumpy(frame)
            display_frame = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
            self._opencv_render(display_frame,tracker_results)
        
    def _opencv_render(self,frame:cp.ndarray,tracker_results:np.ndarray):
        display_frame = cp.asnumpy(frame)
        #strack.results doesnt have what i want
        
        for result in tracker_results:
            x1, y1, x2, y2, = result[:4]
            if result[6] == self.crosshair_cls_id:#if crosshair
                cv2.circle(display_frame, ((x1 + x2) // 2, (y1 + y2) // 2),4, (0, 255, 0), -1)
                cv2.putText(display_frame, f"conf: {result[5]:.2f}", (int(x1), int(y1) - 12),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 204), thickness = 1)
                cv2.circle(display_frame, ((x1 + x2) // 2, (y1 + y2) // 2),4, (255, 0, 204), -1)
                # cv2.circle(display_frame, ((x1 + x2) // 2, (y1 + y2)// 2 - int((y2-y1)*.39)),4, (255, 0, 204), -1)
                cv2.putText(display_frame, f"conf: {result[5]:.2f}", (int(x1), int(y1) - 48),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(display_frame, f"cls_id: {result[6]}", (int(x1), int(y1) - 36),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(display_frame, f"ID: {result[7]}", (int(x1), int(y1) - 24),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Screen Capture Detection", display_frame)
        cv2.waitKey(1)
        
    def _dpg_overlay(self):
        pass
    