"""Global hotkey (works while Roblox is focused) to switch the bot on and off."""

from __future__ import annotations

import sys
import threading
from typing import Callable, Optional

from .controller import normalize_key, short_error


def _key_name(key: object) -> Optional[str]:
    """Translate a pynput key object into BladeBot's key naming."""
    name = getattr(key, "name", None)  # pynput.keyboard.Key members
    if name:
        try:
            return normalize_key(name)
        except ValueError:
            return None
    char = getattr(key, "char", None)
    if char:
        try:
            return normalize_key(char)
        except ValueError:
            return None
    vk = getattr(key, "vk", None)
    if vk is not None:
        if 0x70 <= vk <= 0x7B and sys.platform == "win32":
            return f"f{vk - 0x6F}"
        if 0x41 <= vk <= 0x5A and sys.platform == "win32":
            return chr(vk + 32)
    return None


class HotkeyListener:
    """Calls ``on_toggle`` whenever the configured key is pressed (anywhere)."""

    def __init__(self, get_hotkey: Callable[[], str], on_toggle: Callable[[], None]) -> None:
        self.get_hotkey = get_hotkey
        self.on_toggle = on_toggle
        self.error: Optional[str] = None
        self._listener = None
        self._down: set[str] = set()
        self._lock = threading.Lock()

    def start(self) -> bool:
        try:
            from pynput import keyboard
        except Exception as exc:  # no display, not installed, no permission...
            self.error = f"global hotkey unavailable ({short_error(exc)})"
            return False
        try:
            self._listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
            self._listener.daemon = True
            self._listener.start()
        except Exception as exc:
            self.error = f"global hotkey unavailable ({short_error(exc)})"
            self._listener = None
            return False
        return True

    def stop(self) -> None:
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:
                pass
            self._listener = None

    def _on_press(self, key: object) -> None:
        name = _key_name(key)
        if name is None:
            return
        with self._lock:
            if name in self._down:
                return  # ignore auto-repeat while the key is held
            self._down.add(name)
        try:
            if name == normalize_key(self.get_hotkey()):
                self.on_toggle()
        except Exception:
            pass

    def _on_release(self, key: object) -> None:
        name = _key_name(key)
        if name is not None:
            with self._lock:
                self._down.discard(name)
