import cv2
import cupy as cp
import numpy as np
from typing import Optional

import dearpygui.dearpygui as dpg
import ctypes
from ctypes import wintypes 
from . import dpgoverlay



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
        
        self.window_height, self.window_width = hw_capture  
        
        if self.opencv_enabled:
            cv2.namedWindow("Screen Capture Detection", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Screen Capture Detection", self.window_width, self.window_height)
        if self.dpg_enabled:
            print('creating dpg overlay')
            self._create_dpg_overlay()    
            

    def render_gui(self,frame:Optional[cp.ndarray],tracked_detections:np.ndarray):
        """_summary_

        Args:
            frame (Optional[cp.ndarray]): i think frame is only needed for opencv render?
            tracked_detections (np.ndarray): [x1, y1, x2, y2, track_id, confidence, class_id, Strack idx]
        """
        if self.opencv_enabled:
            self._opencv_render(frame,tracked_detections)
            
        if self.dpg_enabled:
            self._dpg_render(tracked_detections)
            
        
    def _opencv_render(self,frame:cp.ndarray,tracked_detections:np.ndarray):
        display_frame = cp.asnumpy(frame)
        display_frame = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
        #tracker_results (np.ndarray): [x1, y1, x2, y2, track_id, confidence, class_id, Strack idx]
        for detection in tracked_detections:
            self._draw_opencv_rectangle(display_frame,detection)

        cv2.imshow("Screen Capture Detection", display_frame)
        cv2.waitKey(1)
        
    def _dpg_render(self, tracked_detections:np.ndarray, target_cls = 0):
        #probably just write tracker results to dpg overlay then render
        self.dpg_overlay.clear_canvas()
        
        if tracked_detections.size == 0:
            self.dpg_overlay.render()
            return
        
        cls_mask = (tracked_detections[:,6] == target_cls)
        target_cls_detections = tracked_detections[cls_mask]
        
        for detection in target_cls_detections:
            x1,y1,x2,y2 = detection[:4]
            self.dpg_overlay.draw_bounding_box(x1,y1,x2,y2)
            
        self.dpg_overlay.render()
        
    def _draw_opencv_rectangle(self,display_frame:np.ndarray,detection:np.ndarray):
        #when drawing need int
        x1,y1,x2,y2 = map(int,detection[:4])
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 204), thickness = 1)#bb
        cv2.circle(display_frame, ((x1 + x2) // 2, (y1 + y2) // 2),2, (255, 0, 204), -1)#center of bb
        cv2.putText(display_frame, f"conf: {detection[5]:.2f}", (int(x1), int(y1) - 24),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display_frame, f"cls_id: {detection[6]}", (int(x1), int(y1) - 12),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _create_dpg_overlay(self):
        self.dpg_overlay = dpgoverlay.DPGOverlay(height=self.window_height, width=self.window_width)
        self.dpg_overlay.start()
        self.dpg_overlay.clear_canvas()
        self.dpg_overlay.draw_bounding_box(100, 100, 200, 200)  # Example bounding box
        self.dpg_overlay.render()

    