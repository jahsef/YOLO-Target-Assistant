import threading

import win32api
import win32con

from ..utils.utils import log

# Special-key names accepted in cfg's toggle_hotkey. Lowercase and underscore-separated
# (page_down, shift_r, f9); keep these spellings, configs are written against them.
_SPECIAL_VKS = {
    'space': win32con.VK_SPACE, 'tab': win32con.VK_TAB, 'esc': win32con.VK_ESCAPE,
    'escape': win32con.VK_ESCAPE, 'enter': win32con.VK_RETURN,
    'backspace': win32con.VK_BACK, 'delete': win32con.VK_DELETE,
    'insert': win32con.VK_INSERT, 'home': win32con.VK_HOME, 'end': win32con.VK_END,
    'page_up': win32con.VK_PRIOR, 'page_down': win32con.VK_NEXT,
    'up': win32con.VK_UP, 'down': win32con.VK_DOWN,
    'left': win32con.VK_LEFT, 'right': win32con.VK_RIGHT,
    'shift': win32con.VK_SHIFT, 'shift_l': win32con.VK_LSHIFT, 'shift_r': win32con.VK_RSHIFT,
    'ctrl': win32con.VK_CONTROL, 'ctrl_l': win32con.VK_LCONTROL, 'ctrl_r': win32con.VK_RCONTROL,
    'alt': win32con.VK_MENU, 'alt_l': win32con.VK_LMENU, 'alt_r': win32con.VK_RMENU,
    'caps_lock': win32con.VK_CAPITAL, 'num_lock': win32con.VK_NUMLOCK,
    **{f'f{i}': getattr(win32con, f'VK_F{i}') for i in range(1, 13)},
}

_DOWN_BIT = 0x8000  # GetAsyncKeyState: high bit set means the key is down right now


def _vk_for(name: str) -> int | None:
    """cfg hotkey string -> virtual-key code, or None if unmappable."""
    name = name.strip().lower()
    if not name:
        return None
    if name in _SPECIAL_VKS:
        return _SPECIAL_VKS[name]
    if len(name) == 1:
        # VkKeyScan handles layout and punctuation; low byte is the VK
        vk = win32api.VkKeyScan(name)
        return None if vk == -1 else vk & 0xFF
    return None


class InputDetector:
    """Reads input by polling GetAsyncKeyState.

    RMB is polled on read, so `is_rmb_pressed` is never stale. The call costs ~0.3 us,
    cheap enough to do per access.

    The toggle hotkey needs edge detection (press, not held), which means something has
    to watch it between frames — so it gets a daemon poller, and only when a hotkey is
    actually configured. With the shipped empty toggle_hotkey no thread starts at all.
    """

    POLL_HZ = 120  # hotkey poller; well under key-repeat, far above human press length

    def __init__(self, toggle_hotkey=''):
        self.is_toggled = True  # Aimbot is active by default
        self.toggle_hotkey = toggle_hotkey.lower() if toggle_hotkey else ''
        self._toggle_vk = _vk_for(self.toggle_hotkey) if self.toggle_hotkey else None
        if self.toggle_hotkey and self._toggle_vk is None:
            log(f"unmappable toggle_hotkey {self.toggle_hotkey!r}; toggling disabled", "WARNING")
        self._toggle_was_down = False
        self._stop = threading.Event()
        self._thread = None

    @property
    def is_rmb_pressed(self) -> bool:
        return bool(win32api.GetAsyncKeyState(win32con.VK_RBUTTON) & _DOWN_BIT)

    @property
    def is_lmb_pressed(self) -> bool:
        return bool(win32api.GetAsyncKeyState(win32con.VK_LBUTTON) & _DOWN_BIT)

    def poll_toggle(self) -> None:
        """Flip is_toggled on the rising edge of the hotkey. Safe to call from
        anywhere; the daemon thread calls it, and so can a host loop."""
        if self._toggle_vk is None:
            return
        down = bool(win32api.GetAsyncKeyState(self._toggle_vk) & _DOWN_BIT)
        if down and not self._toggle_was_down:
            self.is_toggled = not self.is_toggled
            log(f'Aimbot Toggled: {self.is_toggled}', "DEBUG")
        self._toggle_was_down = down

    def start_input_detection(self):
        """Start the hotkey poller. No-op when no toggle hotkey is configured —
        RMB needs no background work."""
        if self._toggle_vk is None:
            log("no toggle hotkey configured; RMB is polled on demand", "INFO")
            return
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        log(f"hotkey poller for '{self.toggle_hotkey}' started at {self.POLL_HZ} Hz", "INFO")

    def _poll_loop(self):
        period = 1.0 / self.POLL_HZ
        while not self._stop.wait(period):
            self.poll_toggle()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
