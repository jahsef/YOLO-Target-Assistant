import dearpygui.dearpygui as dpg
import win32gui
import win32con
import win32api
import time

class DPGOverlay:
    def __init__(self,height:int,width:int):
        self.padding = 16
        self.draw_offset = self.padding//2
        self.vp_width = width
        self.vp_height = height
        self.drawlist_tag = "drawing_area"

        dpg.create_context()
        self._setup_viewport()
        self._create_window()
        self._draw_corner_edges(offset=self.padding)

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
            decorated=False
        )
        dpg.set_viewport_pos([center_x, center_y])

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

    def draw_bounding_box(self, x1: int, y1: int, x2: int, y2: int):
        x1_adj = x1 - self.draw_offset
        y1_adj = y1 - self.draw_offset
        x2_adj = x2 - self.draw_offset
        y2_adj = y2 - self.draw_offset

        dpg.draw_rectangle(
            (x1_adj, y1_adj),
            (x2_adj, y2_adj),
            color=(0, 255, 0, 255),
            thickness=2,
            parent=self.drawlist_tag
        )

    def clear_canvas(self):
        dpg.delete_item(self.drawlist_tag, children_only=True)
        self._draw_corner_edges(offset=self.padding)

    def apply_transparency(self):
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

    def start(self):
        dpg.setup_dearpygui()
        dpg.show_viewport()

        # Wait for the window to be registered before applying transparency
        # print('waiting transparency')
        time.sleep(0.5)
        # print('applying transparency')
        self.apply_transparency()

    def render(self):
        dpg.render_dearpygui_frame()
    
    def cleanup(self):
        dpg.destroy_context()
        