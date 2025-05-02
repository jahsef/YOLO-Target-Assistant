from pynput import mouse
from pynput.mouse import Button
import threading

class InputDetector:
    
    def __init__(self,debug = False):
        self.is_rmb_pressed = False
        self.activation_button = mouse.Button.right
        # self.listener = None
        # self.listener_thread = None
        self.debug = debug
    
    def on_click(self, x, y, button, pressed):
        """Callback for mouse clicks (thread-safe)."""
        if button == self.activation_button:
            self.is_rmb_pressed = pressed
            if self.debug:
                print('Pressed' if pressed else 'Released')

    def start_input_detection(self):
        """Start the listener in a daemon thread (non-blocking)."""
        # if self.listener_thread and self.listener_thread.is_alive():
        #     return  # Already running

        self.listener = mouse.Listener(on_click=self.on_click)
        self.listener_thread = threading.Thread(
            target=self.listener.start, 
            daemon=True  # Kills thread when main program exits
        )
        self.listener_thread.start()
        print("Input listener started in background.")        
