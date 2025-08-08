from pynput import mouse, keyboard
from pynput.mouse import Button
import threading

class InputDetector:
    
    def __init__(self, debug=False, toggle_hotkey=''):
        self.is_rmb_pressed = False
        self.is_toggled = True  # Aimbot is active by default
        self.activation_button = mouse.Button.right
        self.toggle_hotkey = toggle_hotkey.lower() if toggle_hotkey else ''
        self.debug = debug

    def on_click(self, x, y, button, pressed):
        """Callback for mouse clicks (thread-safe)."""
        if button == self.activation_button:
            self.is_rmb_pressed = pressed
            if self.debug:
                print(f'RMB: {"Pressed" if pressed else "Released"}')

    def on_press(self, key):
        """Callback for keyboard presses (thread-safe)."""
        if self.toggle_hotkey:
            try:
                if key.char and key.char.lower() == self.toggle_hotkey:
                    self.is_toggled = not self.is_toggled
                    print(f'Aimbot Toggled: {self.is_toggled}')
            except AttributeError:
                # Special keys (e.g., 'Key.space', 'Key.esc') don't have .char
                if str(key).replace('Key.', '').lower() == self.toggle_hotkey:
                    self.is_toggled = not self.is_toggled
                    print(f'Aimbot Toggled: {self.is_toggled}')

    def start_input_detection(self):
        """Start the mouse and keyboard listeners in daemon threads (non-blocking)."""
        mouse_listener = mouse.Listener(on_click=self.on_click)
        mouse_listener_thread = threading.Thread(
            target=mouse_listener.start, 
            daemon=True
        )
        mouse_listener_thread.start()
        print("Mouse input listener started in background.")

        if self.toggle_hotkey:
            keyboard_listener = keyboard.Listener(on_press=self.on_press)
            keyboard_listener_thread = threading.Thread(
                target=keyboard_listener.start, 
                daemon=True
            )
            keyboard_listener_thread.start()
            print(f"Keyboard input listener for hotkey '{self.toggle_hotkey}' started in background.")