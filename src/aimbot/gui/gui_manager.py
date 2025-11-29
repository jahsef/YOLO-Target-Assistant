import cv2
import cupy as cp
import numpy as np
from typing import Optional

import dearpygui.dearpygui as dpg
import ctypes
from ctypes import wintypes
from . import dpgoverlay, opencvgui
from ..utils.utils import log



class GUI_Manager:
    
    def __init__(self, config:dict,hw_capture:tuple[int,int],overlay_render_cls_id:int = 0):
        """
        Args:
            config (dict): config dict loaded from the config json file passed from aimbot.py
            
            overlay_render_cls_id (int): which class id you want the overlay to render, this is usually used for enemy class
        """
        self.gui_settings = config['gui_settings'] #we pass in whole dict since some things may require whole dict access
        self.config = config
        self.opencv_enabled = self.gui_settings["opencv_render"]
        self.dpg_enabled = self.gui_settings["dpg_overlay"]
        window_height, window_width = hw_capture  
        only_render_overlay_non_ads = self.gui_settings["only_render_overlay_non_ads"]
        self.overlay_render_cls_id = overlay_render_cls_id
        log(f'starting gui manager, opencv: {self.opencv_enabled}, dpg: {self.dpg_enabled}', "INFO")
        
        if not self.opencv_enabled and not self.dpg_enabled:
            return None
        
        if self.opencv_enabled:
            self.opencvgui = opencvgui.OpenCVGUI(window_height=window_height, window_width=window_width, config = self.config)
            
        if self.dpg_enabled:
            self.dpgoverlay = dpgoverlay.DPGOverlay(height=window_height, width=window_width,only_render_overlay_non_ads=only_render_overlay_non_ads,
                                                    overlay_render_cls_id=self.overlay_render_cls_id)
            

    def render(self,frame:Optional[cp.ndarray],tracked_detections:np.ndarray,is_rmb_pressed:bool, raw_deltas:tuple[int,int], scaled_deltas:tuple[int,int]):
        """_summary_

        Args:
            frame (Optional[cp.ndarray]): i think frame is only needed for opencv render?
            tracked_detections (np.ndarray): [x1, y1, x2, y2, track_id, confidence, class_id, Strack idx]
        """
        #for weird edgecase where model returns nans near startup
        if np.isnan(tracked_detections).any():
            return
        
        if self.opencv_enabled:
            self.opencvgui.render(frame = frame,tracked_detections=tracked_detections, raw_deltas = raw_deltas, scaled_deltas = scaled_deltas)
            
        if self.dpg_enabled:

            self.dpgoverlay.render(tracked_detections,is_rmb_pressed)
                    
    def cleanup(self):
        if self.opencv_enabled:
            self.opencvgui.cleanup()
        if self.dpg_enabled:
            self.dpgoverlay.cleanup()