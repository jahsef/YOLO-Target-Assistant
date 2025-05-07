import cv2
import cupy as cp
import numpy as np
from typing import Optional

import dearpygui.dearpygui as dpg
import ctypes
from ctypes import wintypes 
from . import dpgoverlay, opencvgui



class GUI_Manager:
    
    def __init__(self, config:dict,hw_capture:tuple[int,int],overlay_render_cls_id:int = 0):
        """
        Args:
            config (dict): config dict (only the gui_settings portion) loaded from the config json file passed from aimbot.py
            
            overlay_render_cls_id (int): which class id you want the overlay to render, this is usually used for enemy class
        """
        
        self.opencv_enabled = config["opencv_render"]
        self.dpg_enabled = config["dpg_overlay"]
        window_height, window_width = hw_capture  
        only_render_overlay_non_ads = config["only_render_overlay_non_ads"]
        self.overlay_render_cls_id = overlay_render_cls_id
        print(f'starting gui manager, opencv: {self.opencv_enabled}, dpg: {self.dpg_enabled}')
        
        if not self.opencv_enabled and not self.dpg_enabled:
            return None
        
        if self.opencv_enabled:
            self.opencvgui = opencvgui.OpenCVGUI(window_height=window_height, window_width=window_width)
            
        if self.dpg_enabled:
            self.dpgoverlay = dpgoverlay.DPGOverlay(height=window_height, width=window_width,only_render_overlay_non_ads=only_render_overlay_non_ads)
            

    def render(self,frame:Optional[cp.ndarray],tracked_detections:np.ndarray,is_rmb_pressed:bool):
        """_summary_

        Args:
            frame (Optional[cp.ndarray]): i think frame is only needed for opencv render?
            tracked_detections (np.ndarray): [x1, y1, x2, y2, track_id, confidence, class_id, Strack idx]
        """
        if self.opencv_enabled:
            self.opencvgui.render(frame,tracked_detections)
            
        if self.dpg_enabled:
            #should probably check this inside of the dpg overlay class lol
            if tracked_detections.size > 0:
                class_mask = tracked_detections[:,6] == self.overlay_render_cls_id
                masked_detections = tracked_detections[class_mask]
            else:
                masked_detections = np.asarray([])
            
            self.dpgoverlay.render(masked_detections,is_rmb_pressed)
                    
    def cleanup(self):
        if self.opencv_enabled:
            self.opencvgui.cleanup()
        if self.dpg_enabled:
            self.dpgoverlay.cleanup()