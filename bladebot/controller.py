"""Sends the parry input (a key press or a left click) to the game window.

* **Windows:** uses the Win32 ``SendInput`` API with hardware scan codes,
  which Roblox reliably accepts. No extra packages needed.
* **macOS / Linux:** falls back to ``pynput``.
* **Dry run / arena:** no real input at all.

The key (or mouse button) goes down immediately on the calling thread - no
thread hand-off delay - and a small worker thread releases it after the hold
time, so the vision loop never stalls while a key is held down.
"""

from __future__ import annotations

import queue
import sys
import threading
import time
from typing import Callable, Optional

# Windows virtual-key codes for every key name BladeBot understands.
KEY_VK: dict[str, int] = {
    **{chr(c): c - 32 for c in range(ord("a"), ord("z") + 1)},  # 'a' -> 0x41
    **{str(d): 0x30 + d for d in range(10)},
    **{f"f{n}": 0x6F + n for n in range(1, 13)},  # f1 -> 0x70
    "space": 0x20,
    "tab": 0x09,
    "enter": 0x0D,
    "shift": 0x10,
    "ctrl": 0x11,
    "alt": 0x12,
    "caps_lock": 0x14,
    "insert": 0x2D,
    "delete": 0x2E,
    "home": 0x24,
    "end": 0x23,
    "page_up": 0x21,
    "page_down": 0x22,
    "up": 0x26,
    "down": 0x28,
    "left": 0x25,
    "right": 0x27,
    "pause": 0x13,
    "scroll_lock": 0x91,
    "num_lock": 0x90,
    "print_screen": 0x2C,
    "menu": 0x5D,
}
KEY_NAMES = tuple(KEY_VK)

# Scan codes (set 1) used if MapVirtualKey is unavailable.
_FALLBACK_SCAN: dict[str, int] = {
    **dict(zip("qwertyuiop", range(0x10, 0x1A))),
    **dict(zip("asdfghjkl", range(0x1E, 0x27))),
    **dict(zip("zxcvbnm", range(0x2C, 0x33))),
    **dict(zip("1234567890", range(0x02, 0x0C))),
    "space": 0x39,
    "tab": 0x0F,
    "enter": 0x1C,
    "shift": 0x2A,
    "ctrl": 0x1D,
    "alt": 0x38,
}
_EXTENDED = {"insert", "delete", "home", "end", "page_up", "page_down", "up", "down", "left", "right", "menu"}


def normalize_key(name: str) -> str:
    """Canonical key name (``"F"`` -> ``"f"``, ``"PageUp"`` -> ``"page_up"``)."""
    n = str(name).strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "pageup": "page_up",
        "pagedown": "page_down",
        "pgup": "page_up",
        "pgdn": "page_down",
        "ins": "insert",
        "del": "delete",
        "return": "enter",
        "control": "ctrl",
        "ctrl_l": "ctrl",
        "ctrl_r": "ctrl",
        "shift_l": "shift",
        "shift_r": "shift",
        "alt_l": "alt",
        "alt_r": "alt",
        "alt_gr": "alt",
        "capslock": "caps_lock",
        "scrolllock": "scroll_lock",
        "numlock": "num_lock",
        "printscreen": "print_screen",
        "prtsc": "print_screen",
        "spacebar": "space",
        " ": "space",
    }
    n = aliases.get(n, n)
    if n not in KEY_VK:
        raise ValueError(f"unknown key {name!r}")
    return n


class InputBackend:
    name = "none"

    def key_down(self, key: str) -> None:
        raise NotImplementedError

    def key_up(self, key: str) -> None:
        raise NotImplementedError

    def mouse_down(self) -> None:
        raise NotImplementedError

    def mouse_up(self) -> None:
        raise NotImplementedError


class WindowsSendInput(InputBackend):
    """Win32 ``SendInput`` with scan codes (works with Roblox)."""

    name = "windows-sendinput"

    def __init__(self) -> None:
        import ctypes
        from ctypes import wintypes

        self._ctypes = ctypes
        ulong_ptr = ctypes.c_size_t

        class MOUSEINPUT(ctypes.Structure):
            _fields_ = [
                ("dx", wintypes.LONG),
                ("dy", wintypes.LONG),
                ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", ulong_ptr),
            ]

        class KEYBDINPUT(ctypes.Structure):
            _fields_ = [
                ("wVk", wintypes.WORD),
                ("wScan", wintypes.WORD),
                ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", ulong_ptr),
            ]

        class HARDWAREINPUT(ctypes.Structure):
            _fields_ = [("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD), ("wParamH", wintypes.WORD)]

        class _U(ctypes.Union):
            _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)]

        class INPUT(ctypes.Structure):
            _anonymous_ = ("u",)
            _fields_ = [("type", wintypes.DWORD), ("u", _U)]

        self.INPUT = INPUT
        self.user32 = ctypes.WinDLL("user32", use_last_error=True)
        self.user32.SendInput.argtypes = (wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int)
        self.user32.SendInput.restype = wintypes.UINT
        self.user32.MapVirtualKeyW.argtypes = (wintypes.UINT, wintypes.UINT)
        self.user32.MapVirtualKeyW.restype = wintypes.UINT

    def _scan(self, key: str) -> int:
        scan = int(self.user32.MapVirtualKeyW(KEY_VK[key], 0))  # MAPVK_VK_TO_VSC
        return scan or _FALLBACK_SCAN.get(key, 0)

    def _send(self, inp: object) -> None:
        sent = self.user32.SendInput(1, self._ctypes.byref(inp), self._ctypes.sizeof(self.INPUT))
        if sent != 1:
            raise OSError(f"SendInput failed (error {self._ctypes.get_last_error()})")

    def _key(self, key: str, up: bool) -> None:
        inp = self.INPUT()
        inp.type = 1  # INPUT_KEYBOARD
        inp.ki.wVk = 0
        inp.ki.wScan = self._scan(key)
        flags = 0x0008  # KEYEVENTF_SCANCODE
        if key in _EXTENDED:
            flags |= 0x0001  # KEYEVENTF_EXTENDEDKEY
        if up:
            flags |= 0x0002  # KEYEVENTF_KEYUP
        inp.ki.dwFlags = flags
        self._send(inp)

    def key_down(self, key: str) -> None:
        self._key(key, up=False)

    def key_up(self, key: str) -> None:
        self._key(key, up=True)

    def _mouse(self, flag: int) -> None:
        inp = self.INPUT()
        inp.type = 0  # INPUT_MOUSE
        inp.mi.dwFlags = flag
        self._send(inp)

    def mouse_down(self) -> None:
        self._mouse(0x0002)  # MOUSEEVENTF_LEFTDOWN

    def mouse_up(self) -> None:
        self._mouse(0x0004)  # MOUSEEVENTF_LEFTUP


class PynputBackend(InputBackend):
    name = "pynput"

    def __init__(self) -> None:
        from pynput import keyboard, mouse  # raises if no display / not installed

        self._kb = keyboard.Controller()
        self._mouse = mouse.Controller()
        self._Key = keyboard.Key
        self._Button = mouse.Button

    def _k(self, key: str):  # noqa: ANN202
        if len(key) == 1:
            return key
        special = {"ctrl": "ctrl", "alt": "alt", "shift": "shift"}
        return getattr(self._Key, special.get(key, key))

    def key_down(self, key: str) -> None:
        self._kb.press(self._k(key))

    def key_up(self, key: str) -> None:
        self._kb.release(self._k(key))

    def mouse_down(self) -> None:
        self._mouse.press(self._Button.left)

    def mouse_up(self) -> None:
        self._mouse.release(self._Button.left)


def short_error(exc: BaseException, limit: int = 140) -> str:
    """One readable line for an exception (pynput errors can be whole paragraphs)."""
    if isinstance(exc, ModuleNotFoundError):
        return f"{exc.name or 'a module'} is not installed (pip install -r requirements.txt)"
    text = str(exc)
    if "X connection" in text or "DISPLAY" in text:
        return "no graphical display found (DISPLAY is not set)"
    line = next((ln.strip() for ln in text.splitlines() if ln.strip()), exc.__class__.__name__)
    return line if len(line) <= limit else line[: limit - 3] + "..."


def create_backend() -> tuple[Optional[InputBackend], Optional[str]]:
    """Best input backend for this OS, or ``(None, reason)`` if none works."""
    errors = []
    if sys.platform == "win32":
        try:
            return WindowsSendInput(), None
        except Exception as exc:  # pragma: no cover - Windows only
            errors.append(f"SendInput: {short_error(exc)}")
    try:
        return PynputBackend(), None
    except Exception as exc:
        errors.append(f"pynput: {short_error(exc)}")
    return None, "; ".join(errors) or "no input backend available"


class ParryController:
    """Queues parry presses and performs them on a worker thread."""

    def __init__(self, backend: Optional[InputBackend] = None, backend_error: Optional[str] = None) -> None:
        if backend is None and backend_error is None:
            backend, backend_error = create_backend()
        self.backend = backend
        self.backend_error = backend_error
        self.last_error: Optional[str] = None
        self.presses = 0
        self._queue: "queue.Queue[tuple[str, str, float]]" = queue.Queue(maxsize=8)
        self._thread = threading.Thread(target=self._worker, name="bladebot-input", daemon=True)
        self._thread.start()

    @property
    def available(self) -> bool:
        return self.backend is not None

    def press(self, method: str, key: str, hold_s: float) -> bool:
        """Press now and release after ``hold_s``. Returns False if it couldn't be done."""
        backend = self.backend
        if backend is None or self._queue.full():
            return False
        try:
            if method == "mouse":
                backend.mouse_down()
            else:
                backend.key_down(key)
        except Exception as exc:
            self.last_error = f"input failed: {short_error(exc)}"
            return False
        self._queue.put_nowait((method, key, time.perf_counter() + hold_s))
        return True

    def _worker(self) -> None:
        while True:
            method, key, release_at = self._queue.get()
            backend = self.backend
            if backend is None:
                continue
            delay = release_at - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            try:
                if method == "mouse":
                    backend.mouse_up()
                else:
                    backend.key_up(key)
                self.presses += 1
                self.last_error = None
            except Exception as exc:
                self.last_error = f"input failed: {short_error(exc)}"


class CallbackController:
    """Controller used by the simulator arena: a press just calls a function."""

    def __init__(self, callback: Callable[[], None]) -> None:
        self.callback = callback
        self.presses = 0
        self.available = True
        self.backend_error = None
        self.last_error: Optional[str] = None

    def press(self, method: str, key: str, hold_s: float) -> bool:
        self.callback()
        self.presses += 1
        return True
