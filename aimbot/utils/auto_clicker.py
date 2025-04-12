import time
import threading
from pynput import mouse
from pynput.mouse import Controller, Button

class MouseClicker:
    def __init__(self, activation_button=Button.x1):  # Default to X1 (common back button)
        self.mouse_controller = Controller()
        self.clicking = False
        self.listener = None
        self.click_thread = None
        self.exit_flag = False
        self.activation_button = activation_button

    def on_click(self, x, y, button, pressed):
        if button == self.activation_button:
            self.clicking = pressed
            if pressed and not (self.click_thread and self.click_thread.is_alive()):
                self.click_thread = threading.Thread(target=self.autoclicker)
                self.click_thread.start()

    def autoclicker(self):
        while self.clicking and not self.exit_flag:
            self.mouse_controller.click(Button.left)
            time.sleep(0.001)

    def start(self):
        with mouse.Listener(on_click=self.on_click) as self.listener:
            try:
                while not self.exit_flag:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                self.exit_flag = True
                if self.click_thread:
                    self.click_thread.join()
                self.listener.stop()

def detect_mouse_buttons():
    def on_click(x, y, button, pressed):
        if pressed:
            print(f"Detected button: {button} (Use this name in the next step)")
    
    with mouse.Listener(on_click=on_click) as listener:
        print("Press your back button to detect it...")
        listener.join()

if __name__ == "__main__":
    print("First let's detect your back button...")
    detect_mouse_buttons()
    
    btn_name = input("Enter button name from detection (e.g., 'x1' or 'x2'): ").strip().lower()
    activation_button = getattr(Button, btn_name)
    
    print("\nAutoclicker started. Hold your back button to click.")
    print("Press Ctrl+C to exit.")
    MouseClicker(activation_button).start()