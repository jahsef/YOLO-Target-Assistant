import time
import threading
import win32api
import win32con

_DOWN_BIT = 0x8000  # GetAsyncKeyState high bit = key is down now

class MouseClicker:
    def __init__(self,cpm):
        self.clicking = False
        self.click_thread = None
        self.exit_flag = False
        self.activation_vk = win32con.VK_XBUTTON1  # mouse back button
        self.poll_interval = 0.005

        # More game-friendly settings
        self.click_duration = 0.01  # 10ms hold time
        self.clicks_per_minute = cpm
        self.clicks_per_second = self.clicks_per_minute / 60
        self.click_interval = 1 / self.clicks_per_second

    def _button_down(self) -> bool:
        return bool(win32api.GetAsyncKeyState(self.activation_vk) & _DOWN_BIT)

    def poll_once(self):
        """Sync self.clicking to the button and spawn the click thread on press."""
        pressed = self._button_down()
        self.clicking = pressed
        if pressed and not (self.click_thread and self.click_thread.is_alive()):
            self.click_thread = threading.Thread(target=self.autoclicker)
            self.click_thread.start()

    def autoclicker(self):
        """Enhanced click simulation with hold duration"""
        while self.clicking and not self.exit_flag:
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0)
            time.sleep(self.click_duration)  # Hold button briefly
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0)
            time.sleep(self.click_interval - self.click_duration)

    def start(self):
        """Poll the activation button until interrupted."""
        try:
            while not self.exit_flag:
                self.poll_once()
                time.sleep(self.poll_interval)
        except KeyboardInterrupt:
            self.exit_flag = True
            self.clicking = False
            if self.click_thread:
                self.click_thread.join()
        except Exception:
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