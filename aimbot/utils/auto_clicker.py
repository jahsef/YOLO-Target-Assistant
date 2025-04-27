import time
import threading
import win32api
import win32con
from pynput import mouse
from pynput.mouse import Button

class MouseClicker:
    def __init__(self,cpm):
        self.clicking = False
        self.listener = None
        self.click_thread = None
        self.exit_flag = False
        self.activation_button = Button.x1
        
        # More game-friendly settings
        self.click_duration = 0.01  # 10ms hold time
        self.clicks_per_minute = cpm
        self.clicks_per_second = self.clicks_per_minute / 60
        self.click_interval = 1 / self.clicks_per_second

    def on_click(self, x, y, button, pressed):
        """Modified to work with elevated privileges"""
        try:
            if button == self.activation_button:
                self.clicking = pressed
                if pressed and not (self.click_thread and self.click_thread.is_alive()):
                    self.click_thread = threading.Thread(target=self.autoclicker)
                    self.click_thread.start()
        except Exception as e:
            pass

    def autoclicker(self):
        """Enhanced click simulation with hold duration"""
        while self.clicking and not self.exit_flag:
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0)
            time.sleep(self.click_duration)  # Hold button briefly
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0)
            time.sleep(self.click_interval - self.click_duration)

    def start(self):
        """Run with try-except for privilege issues"""
        try:
            with mouse.Listener(on_click=self.on_click) as self.listener:
                while not self.exit_flag:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            self.exit_flag = True
            if self.click_thread:
                self.click_thread.join()
            self.listener.stop()
        except Exception as e:
            print(f"Permission error: Try running as Administrator")
            
if __name__ == "__main__":
    print('default button is mouse back, hold down')
    print('clicks per min should be *.95 if game has click speed caps')
    print('eval func in code')
    print('enter desired clicks per min:')
    input_cpm = float(eval(input()))
    print("Press Ctrl+C to exit.")
    
    clicker = MouseClicker(input_cpm)
    clicker.start()