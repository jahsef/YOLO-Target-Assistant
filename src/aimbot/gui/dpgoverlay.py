import dearpygui.dearpygui as dpg
import win32gui
import win32con
import win32api
import time
import numpy as np
from ..utils.utils import log

class DPGOverlay:
    def __init__(self,height:int,width:int,only_render_overlay_non_ads:bool, overlay_render_cls_id:int):
        self.padding = 16
        self.draw_offset = self.padding//2
        self.vp_width = width
        self.vp_height = height
        self.drawlist_tag = "drawing_area"
        self.overlay_render_cls_id = overlay_render_cls_id
        dpg.create_context()
        self._setup_viewport()
        self._create_window()
        self._draw_corner_edges(offset=self.padding)
        #store bounding box tags to delete later
        self.bbox_tags = []
        self.are_bb_drawn = False
        self.only_render_overlay_non_ads = only_render_overlay_non_ads
        
        self._start()

        #we use this so if not updated within time we manually update to avoid window freeze
        self.last_updated_time = time.time()

    def _setup_viewport(self):
        screen_width = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
        screen_height = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)
        center_x = (screen_width - self.vp_width) // 2
        center_y = (screen_height - self.vp_height) // 2

        dpg.create_viewport(
            title='Transparent Overlay',
            width=self.vp_width,
            height=self.vp_height,
            clear_color=[0.0, 0.0, 0.0, 0.0],
            always_on_top=True,
            decorated=False,
            vsync=True,
        )
        
        dpg.set_viewport_pos([center_x, center_y])
        dpg.configure_app(manual_callback_management=True)  # ← Takes full control of rendering

    def _create_window(self):
        with dpg.window(
            label="Overlay",
            no_title_bar=True,
            no_resize=True,
            no_move=True,
            no_collapse=True,
            no_scrollbar=True,
            width=self.vp_width,
            height=self.vp_height,
            pos=[0, 0],
            tag="main_window",
            no_background=True
        ):
            dpg.add_drawlist(
                tag=self.drawlist_tag,
                width=self.vp_width,
                height=self.vp_height
            )

    def _draw_corner_edges(self, offset=16):
        draw_width = dpg.get_item_width(self.drawlist_tag)
        draw_height = dpg.get_item_height(self.drawlist_tag)
        corner_length = 12
        thickness = 2

        # Top-left corner
        dpg.draw_line((0, 0), (corner_length, 0), color=(255, 0, 0, 255), thickness=thickness, parent=self.drawlist_tag)
        dpg.draw_line((0, 0), (0, corner_length), color=(255, 0, 0, 255), thickness=thickness, parent=self.drawlist_tag)

        # Top-right corner
        dpg.draw_line((draw_width - corner_length - offset, 0), (draw_width - offset, 0), color=(255, 0, 0, 255), thickness=thickness, parent=self.drawlist_tag)
        dpg.draw_line((draw_width - offset, 0), (draw_width - offset, corner_length), color=(255, 0, 0, 255), thickness=thickness, parent=self.drawlist_tag)

        # Bottom-left corner
        dpg.draw_line((0, draw_height - offset), (corner_length, draw_height - offset), color=(255, 0, 0, 255), thickness=thickness, parent=self.drawlist_tag)
        dpg.draw_line((0, draw_height - corner_length - offset), (0, draw_height - offset), color=(255, 0, 0, 255), thickness=thickness, parent=self.drawlist_tag)

        # Bottom-right corner
        dpg.draw_line((draw_width - corner_length - offset, draw_height - offset), (draw_width - offset, draw_height - offset), color=(255, 0, 0, 255), thickness=thickness, parent=self.drawlist_tag)
        dpg.draw_line((draw_width - offset, draw_height - corner_length - offset), (draw_width - offset, draw_height - offset), color=(255, 0, 0, 255), thickness=thickness, parent=self.drawlist_tag)

    def _draw_bb_corners(self, x1: int, y1: int, x2: int, y2: int):
        x1 = x1 - self.draw_offset
        y1 = y1 - self.draw_offset
        x2 = x2 - self.draw_offset
        y2 = y2 - self.draw_offset

        bb_w = x2 - x1
        bb_h = y2 - y1

        outset_x = max(2, int(bb_w * 0.2))
        outset_y = max(2, int(bb_h * 0.2))
        x1 -= outset_x
        y1 -= outset_y
        x2 += outset_x
        y2 += outset_y

        corner_lx = max(2, int(bb_w * 0.15))
        corner_ly = max(2, int(bb_h * 0.15))
        thickness = 2
        color = (0, 205, 0, 255)

        # Top-left
        self.bbox_tags.append(dpg.draw_line((x1, y1), (x1 + corner_lx, y1), color=color, thickness=thickness, parent=self.drawlist_tag))
        self.bbox_tags.append(dpg.draw_line((x1, y1), (x1, y1 + corner_ly), color=color, thickness=thickness, parent=self.drawlist_tag))
        # Top-right
        self.bbox_tags.append(dpg.draw_line((x2 - corner_lx, y1), (x2, y1), color=color, thickness=thickness, parent=self.drawlist_tag))
        self.bbox_tags.append(dpg.draw_line((x2, y1), (x2, y1 + corner_ly), color=color, thickness=thickness, parent=self.drawlist_tag))
        # Bottom-left
        self.bbox_tags.append(dpg.draw_line((x1, y2), (x1 + corner_lx, y2), color=color, thickness=thickness, parent=self.drawlist_tag))
        self.bbox_tags.append(dpg.draw_line((x1, y2 - corner_ly), (x1, y2), color=color, thickness=thickness, parent=self.drawlist_tag))
        # Bottom-right
        self.bbox_tags.append(dpg.draw_line((x2 - corner_lx, y2), (x2, y2), color=color, thickness=thickness, parent=self.drawlist_tag))
        self.bbox_tags.append(dpg.draw_line((x2, y2 - corner_ly), (x2, y2), color=color, thickness=thickness, parent=self.drawlist_tag))
        
    def _clear_canvas(self):
        for tag in self.bbox_tags:
            dpg.delete_item(tag)
        self.bbox_tags.clear()

    def _apply_transparency(self):
        hwnd = win32gui.FindWindow(None, "Transparent Overlay")
        if hwnd:
            # print('for sure applying')
            win32gui.SetWindowLong(
                hwnd,
                win32con.GWL_EXSTYLE,
                win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE) | win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT
            )
            win32gui.SetLayeredWindowAttributes(hwnd, win32api.RGB(0, 0, 0), 0, win32con.LWA_COLORKEY)
        else:
            log("Window handle not found!", "WARNING")

    def _start(self):
        dpg.setup_dearpygui()
        dpg.show_viewport()

        # Wait for the window to be registered before applying transparency
        # print('waiting transparency')
        time.sleep(0.5)
        # print('applying transparency')
        self._apply_transparency()
        self._draw_corner_edges()
        self._render_frame()

    def render(self, tracked_detections: np.ndarray, is_rmb_pressed: bool) -> None:
        """render handler for bounding boxes"""
        # self.frame_count += 1
        
        if tracked_detections.size > 0:
            #we mask to  only display target class
            class_mask = tracked_detections[:,6] == self.overlay_render_cls_id
            masked_detections = tracked_detections[class_mask]
        else:
            #if no target class then nothing
            masked_detections = np.asarray([])
        
        # force clear every half second if theres no detections
        #avoids window freeze issues a lot of the time
        if (time.time() - self.last_updated_time > 0.5) and masked_detections.size == 0:
            #force refresh
            self._clear_canvas()
            self._render_frame()
            self.last_updated_time = time.time()
            return

        #get rendering decisions
        should_draw, should_clear = self._should_render(masked_detections, is_rmb_pressed)

        if not should_draw:
            if should_clear  and self.are_bb_drawn:
            #no new detection,  need to clear out old
                self._clear_canvas()
                self.are_bb_drawn = False
                self._render_frame()
                self.last_updated_time = time.time()
            return
        else:
            #if we draw, we always clear canvas
            #if should_draw already takes into account if masked_detections is empty or not
            self._clear_canvas()
            self.are_bb_drawn = True
            for detection in masked_detections:
                x1, y1, x2, y2 = map(int, detection[:4])
                self._draw_bb_corners(x1, y1, x2, y2)

        # Push updates to screen
        self._render_frame()
        self.last_updated_time = time.time()


    def _should_render(self, tracked_detections: np.ndarray, is_rmb_pressed: bool) -> tuple[bool, bool]:
        """State machine for overlay visibility decisions
        
        ai so much more useful for writing comments than actual code lmfao
        
        Returns:
            tuple[bool, bool]: 
                [0] True if should render this frame
                [1] True if should clear existing boxes
        """
        # Block rendering during ADS if configured
        if self.only_render_overlay_non_ads and is_rmb_pressed:
            if self.are_bb_drawn:
                return (False, True)  # Clear boxes without drawing new ones
            return (False, False)  # No action needed

        has_valid_detections = tracked_detections.size > 0
        
        # Clear logic: no detections but boxes exist
        if not has_valid_detections and self.are_bb_drawn:
            return (True, True)
            
        # Draw logic: new detections available
        if has_valid_detections:
            return (True, False)
            
        # Default: no action needed
        return (False, False)
        
    def _render_frame(self):
        dpg.render_dearpygui_frame()
    
    def cleanup(self):
        dpg.destroy_context()
        