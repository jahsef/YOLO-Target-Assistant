"""GetAsyncKeyState-based input detection, driven through a fake win32api."""

import pytest
import win32con

from src.aimbot.input import inputdetector
from src.aimbot.input.inputdetector import InputDetector, _vk_for
from tests.support.fakes import FakeWin32


@pytest.fixture
def fake_win32(monkeypatch):
    fake = FakeWin32()
    fake.VkKeyScan = lambda ch: ord(ch.upper())  # good enough for a-z0-9
    monkeypatch.setattr(inputdetector, "win32api", fake)
    return fake


class TestVkMapping:
    def test_single_letter(self):
        assert _vk_for("k") == ord("K")

    def test_case_insensitive(self):
        assert _vk_for("K") == _vk_for("k")

    def test_special_key_names(self):
        """Lowercase underscore spellings — cfg files are written against these."""
        assert _vk_for("space") == win32con.VK_SPACE
        assert _vk_for("caps_lock") == win32con.VK_CAPITAL
        assert _vk_for("page_down") == win32con.VK_NEXT
        assert _vk_for("f9") == win32con.VK_F9
        assert _vk_for("shift_r") == win32con.VK_RSHIFT

    def test_empty_and_unknown(self):
        assert _vk_for("") is None
        assert _vk_for("   ") is None
        assert _vk_for("not_a_key") is None


class TestRmb:
    def test_reports_button_state(self, fake_win32):
        d = InputDetector()
        assert d.is_rmb_pressed is False
        fake_win32.set_down(win32con.VK_RBUTTON, True)
        assert d.is_rmb_pressed is True
        fake_win32.set_down(win32con.VK_RBUTTON, False)
        assert d.is_rmb_pressed is False

    def test_is_read_fresh_every_access(self, fake_win32):
        """No cached attribute: a hook-based value can lag by a scheduling quantum,
        a polled one cannot."""
        d = InputDetector()
        seen = []
        for state in (True, False, True):
            fake_win32.set_down(win32con.VK_RBUTTON, state)
            seen.append(d.is_rmb_pressed)
        assert seen == [True, False, True]

    def test_needs_no_background_thread(self, fake_win32):
        d = InputDetector(toggle_hotkey="")
        d.start_input_detection()
        assert d._thread is None


class TestToggle:
    def test_defaults_to_on(self, fake_win32):
        assert InputDetector().is_toggled is True

    def test_flips_once_per_press(self, fake_win32):
        d = InputDetector(toggle_hotkey="k")
        vk = ord("K")
        fake_win32.set_down(vk, True)
        d.poll_toggle()
        assert d.is_toggled is False

    def test_holding_does_not_retrigger(self, fake_win32):
        """Edge, not level — otherwise a held key strobes the aimbot every poll."""
        d = InputDetector(toggle_hotkey="k")
        fake_win32.set_down(ord("K"), True)
        for _ in range(20):
            d.poll_toggle()
        assert d.is_toggled is False

    def test_release_then_press_toggles_again(self, fake_win32):
        d = InputDetector(toggle_hotkey="k")
        vk = ord("K")
        for state in (True, True, False, True):
            fake_win32.set_down(vk, state)
            d.poll_toggle()
        assert d.is_toggled is True

    def test_no_hotkey_never_toggles(self, fake_win32):
        d = InputDetector(toggle_hotkey="")
        for vk in range(0x01, 0x100):
            fake_win32.set_down(vk, True)
        for _ in range(5):
            d.poll_toggle()
        assert d.is_toggled is True

    def test_unmappable_hotkey_degrades_to_disabled(self, fake_win32):
        d = InputDetector(toggle_hotkey="not_a_key")
        assert d._toggle_vk is None
        d.poll_toggle()
        assert d.is_toggled is True

    def test_other_keys_do_not_toggle(self, fake_win32):
        d = InputDetector(toggle_hotkey="k")
        fake_win32.set_down(ord("J"), True)
        d.poll_toggle()
        assert d.is_toggled is True


class TestPollerThread:
    def test_starts_and_stops_when_a_hotkey_is_set(self, fake_win32):
        d = InputDetector(toggle_hotkey="k")
        d.start_input_detection()
        assert d._thread is not None and d._thread.is_alive()
        assert d._thread.daemon, "must not keep the process alive on exit"
        d.stop()
        assert d._thread is None

    def test_thread_observes_a_press(self, fake_win32):
        import time
        d = InputDetector(toggle_hotkey="k")
        d.start_input_detection()
        try:
            fake_win32.set_down(ord("K"), True)
            deadline = time.perf_counter() + 2.0
            while d.is_toggled and time.perf_counter() < deadline:
                time.sleep(0.005)
            assert d.is_toggled is False
        finally:
            d.stop()


class TestNoPynput:
    def test_pynput_is_gone_from_the_input_package(self):
        """Checks real imports via AST — the word still appears in a comment
        documenting the hotkey spellings we stayed compatible with."""
        import ast
        import pathlib

        import src.aimbot.input as pkg
        for path in pathlib.Path(pkg.__file__).parent.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                else:
                    continue
                assert not any(n.split(".")[0] == "pynput" for n in names), \
                    f"{path.name} still imports pynput"
