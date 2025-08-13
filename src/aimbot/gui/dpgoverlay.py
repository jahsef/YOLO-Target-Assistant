import dearpygui.dearpygui as dpg
import win32gui
import win32con
import win32api
import time
import numpy as np

class DPGOverlay:
    def __init__(self,height:int,width:int,only_render_overlay_non_ads:bool):
        self.padding = 16
        self.draw_offset = self.padding//2
        self.vp_width = width
        self.vp_height = height
        self.drawlist_tag = "drawing_area"
        
        dpg.create_context()
        self._setup_viewport()
        self._create_window()
        self._draw_corner_edges(offset=self.padding)
        #store bounding box tags to delete later
        self.bbox_tags = []
        self.are_bb_drawn = False
        self.only_render_overlay_non_ads = only_render_overlay_non_ads
        
        self._start()
        #count frames of not updated
        self.frame_count = 0

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

    def _draw_bounding_box(self, x1: int, y1: int, x2: int, y2: int):
        x1_adj = x1 - self.draw_offset
        y1_adj = y1 - self.draw_offset
        x2_adj = x2 - self.draw_offset
        y2_adj = y2 - self.draw_offset

        tag = dpg.draw_rectangle(
            (x1_adj, y1_adj),
            (x2_adj, y2_adj),
            color=(0, 255, 0, 255),
            thickness=2,
            parent=self.drawlist_tag
        )
        self.bbox_tags.append(tag)
        
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
            print("Window handle not found!")

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
        self.frame_count += 1

        # force clear every 300 frames if theres no detections
        if self.frame_count % 300 == 0 and tracked_detections.size == 0:
            self._clear_canvas()
            self._render_frame()
            return

        #get rendering decisions
        should_draw, should_clear = self._should_render(tracked_detections, is_rmb_pressed)
        
        if not should_draw:
            return
        #need to clear canvas before drawing new bounding boxes
        self._clear_canvas()

        if should_clear:
            # Clear existing boxes state
            self.are_bb_drawn = False
        elif tracked_detections.size > 0:
            # Draw all valid detections
            self.are_bb_drawn = True
            for detection in tracked_detections:
                x1, y1, x2, y2 = map(int, detection[:4])
                self._draw_bounding_box(x1, y1, x2, y2)

        # Push updates to screen
        self._render_frame()


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
                return (True, True)  # Force clear existing boxes
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
        