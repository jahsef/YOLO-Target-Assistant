from dpg_class_test import TransparentOverlay
import dearpygui as dpg
import time

if __name__ == "__main__":
    overlay = TransparentOverlay()
    overlay.start()
    timeout = 10e-3
    counter = 0
    sum = 0
    try:
        while True:
            counter+=1
            start = time.perf_counter()
            overlay.clear_canvas()
            overlay.draw_bounding_box(100, 100, 200, 200)  # Example bounding box
            overlay.render()
            sum += time.perf_counter() - start
    except KeyboardInterrupt:
        print(f'avg time/frame: {sum/counter}')