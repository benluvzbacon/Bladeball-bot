"""Finds the Roblox window: where it is (so the capture can follow it) and
whether it is the active window (so the bot never types into another program).

* **Windows:** Win32 API through ``ctypes`` (no extra packages).
* **macOS:** the Quartz window list (``pyobjc``, installed together with pynput).
* **Linux / X11:** ``python-xlib`` (installed together with pynput). Roblox has
  no official Linux client, so this only checks the active window's title.

Nothing here touches the Roblox process itself: it only asks the operating
system which windows exist, the same information a task bar shows.

Every lookup returns ``None`` when it can't tell. Callers treat that as
"unknown", not as an error.
"""

from __future__ import annotations

import math
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

# The smallest window that can be a game view (skips launchers and tool windows).
MIN_WIDTH, MIN_HEIGHT = 160, 120


@dataclass(frozen=True)
class WindowInfo:
    title: str
    owner: str  # process or application name
    left: int
    top: int
    width: int
    height: int
    focused: bool = False
    minimized: bool = False

    @property
    def rect(self) -> dict[str, int]:
        return {"left": self.left, "top": self.top, "width": self.width, "height": self.height}

    @property
    def usable(self) -> bool:
        return not self.minimized and self.width >= MIN_WIDTH and self.height >= MIN_HEIGHT


def looks_like_roblox(title: str = "", owner: str = "") -> bool:
    """True for the Roblox player window (not Roblox Studio or a browser tab)."""
    t = (title or "").strip().lower()
    o = (owner or "").strip().lower().replace("\\", "/").rsplit("/", 1)[-1]
    if "studio" in t or "studio" in o:
        return False
    if o == "explorer.exe":  # a File Explorer window showing a folder called "Roblox"
        return False
    if t in ("roblox", "sober"):  # Windows / Linux (Sober) window titles
        return True
    return o.startswith("robloxplayer") or o in ("roblox", "roblox.app", "sober")


# ===================================================================== backends
class _Backend:
    name = "none"

    def foreground(self) -> Optional[WindowInfo]:
        raise NotImplementedError

    def find_roblox(self) -> Optional[WindowInfo]:
        raise NotImplementedError


class _Win32(_Backend):
    """Win32 window list. The process is made DPI aware first (as ``mss`` does),
    so coordinates are real pixels even with display scaling."""

    name = "win32"

    def __init__(self) -> None:
        import ctypes
        from ctypes import wintypes

        self._ct = ctypes
        self._wt = wintypes
        u = ctypes.WinDLL("user32", use_last_error=True)
        k = ctypes.WinDLL("kernel32", use_last_error=True)
        _make_dpi_aware(ctypes)
        hwnd = wintypes.HWND
        u.GetForegroundWindow.argtypes = ()
        u.GetForegroundWindow.restype = hwnd
        u.GetWindowTextLengthW.argtypes = (hwnd,)
        u.GetWindowTextLengthW.restype = ctypes.c_int
        u.GetWindowTextW.argtypes = (hwnd, wintypes.LPWSTR, ctypes.c_int)
        u.GetWindowTextW.restype = ctypes.c_int
        u.IsWindowVisible.argtypes = (hwnd,)
        u.IsWindowVisible.restype = wintypes.BOOL
        u.IsIconic.argtypes = (hwnd,)
        u.IsIconic.restype = wintypes.BOOL
        u.GetClientRect.argtypes = (hwnd, ctypes.POINTER(wintypes.RECT))
        u.GetClientRect.restype = wintypes.BOOL
        u.ClientToScreen.argtypes = (hwnd, ctypes.POINTER(wintypes.POINT))
        u.ClientToScreen.restype = wintypes.BOOL
        u.GetWindowThreadProcessId.argtypes = (hwnd, ctypes.POINTER(wintypes.DWORD))
        u.GetWindowThreadProcessId.restype = wintypes.DWORD
        self._enum_proc = ctypes.WINFUNCTYPE(wintypes.BOOL, hwnd, wintypes.LPARAM)
        u.EnumWindows.argtypes = (self._enum_proc, wintypes.LPARAM)
        u.EnumWindows.restype = wintypes.BOOL
        k.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        k.OpenProcess.restype = wintypes.HANDLE
        k.QueryFullProcessImageNameW.argtypes = (
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.LPWSTR,
            ctypes.POINTER(wintypes.DWORD),
        )
        k.QueryFullProcessImageNameW.restype = wintypes.BOOL
        k.CloseHandle.argtypes = (wintypes.HANDLE,)
        k.CloseHandle.restype = wintypes.BOOL
        self._u, self._k = u, k

    def _title(self, h: Any) -> str:
        n = self._u.GetWindowTextLengthW(h)
        if n <= 0:
            return ""
        buf = self._ct.create_unicode_buffer(n + 1)
        self._u.GetWindowTextW(h, buf, n + 1)
        return buf.value

    def _exe(self, h: Any) -> str:
        pid = self._wt.DWORD(0)
        self._u.GetWindowThreadProcessId(h, self._ct.byref(pid))
        if not pid.value:
            return ""
        proc = self._k.OpenProcess(0x1000, False, pid.value)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not proc:
            return ""
        try:
            size = self._wt.DWORD(1024)
            buf = self._ct.create_unicode_buffer(size.value)
            if not self._k.QueryFullProcessImageNameW(proc, 0, buf, self._ct.byref(size)):
                return ""
            return buf.value.replace("\\", "/").rsplit("/", 1)[-1]
        finally:
            self._k.CloseHandle(proc)

    def _info(self, h: Any, title: str, exe: str, focused: bool) -> WindowInfo:
        rect = self._wt.RECT()
        origin = self._wt.POINT(0, 0)
        minimized = bool(self._u.IsIconic(h))
        if not minimized and self._u.GetClientRect(h, self._ct.byref(rect)) and self._u.ClientToScreen(h, self._ct.byref(origin)):
            left, top = int(origin.x), int(origin.y)
            width, height = int(rect.right - rect.left), int(rect.bottom - rect.top)
        else:
            left = top = width = height = 0
        return WindowInfo(title, exe, left, top, width, height, focused, minimized)

    def foreground(self) -> Optional[WindowInfo]:
        h = self._u.GetForegroundWindow()
        if not h:
            return None
        return self._info(h, self._title(h), self._exe(h), True)

    def find_roblox(self) -> Optional[WindowInfo]:
        fg = self._u.GetForegroundWindow()
        by_title: list[Any] = []
        others: list[Any] = []

        def collect(h: Any, _lparam: Any) -> bool:
            try:
                if self._u.IsWindowVisible(h):
                    title = self._title(h)
                    (by_title if looks_like_roblox(title) else others).append((h, title))
            except Exception:  # never let an exception escape into the Win32 callback
                pass
            return True

        self._u.EnumWindows(self._enum_proc(collect), 0)  # front-to-back order
        candidates = []
        for h, t in by_title:
            exe = self._exe(h)
            if looks_like_roblox(t, exe):
                candidates.append((h, t, exe))
        if not candidates:  # unusual window title: fall back to the process name
            for h, t in others:
                if t:
                    exe = self._exe(h)
                    if looks_like_roblox("", exe):
                        candidates.append((h, t, exe))
        infos = [self._info(h, t, exe, bool(fg) and h == fg) for h, t, exe in candidates]
        infos = [i for i in infos if i.usable] or infos
        if not infos:
            return None
        return next((i for i in infos if i.focused), infos[0])


def _make_dpi_aware(ctypes: Any) -> None:
    """Report real pixel coordinates on scaled displays (e.g. 150 %), like ``mss`` does."""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # per-monitor aware
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


class _Quartz(_Backend):
    """macOS window list. Bounds are in points, the unit ``mss`` uses on macOS."""

    name = "quartz"

    def __init__(self) -> None:
        import Quartz  # pyobjc-framework-Quartz, a dependency of pynput on macOS

        self._q = Quartz

    @staticmethod
    def _get(d: Any, key: str, default: Any = None) -> Any:
        try:
            value = d.get(key)
        except AttributeError:  # pragma: no cover - older PyObjC
            value = d.valueForKey_(key)
        return default if value is None else value

    def _windows(self) -> list[Any]:
        q = self._q
        opts = q.kCGWindowListOptionOnScreenOnly | q.kCGWindowListExcludeDesktopElements
        wins = q.CGWindowListCopyWindowInfo(opts, q.kCGNullWindowID) or []
        return [w for w in wins if int(self._get(w, "kCGWindowLayer", 0)) == 0]  # normal app windows

    def _info(self, w: Any, focused: bool) -> WindowInfo:
        b = self._get(w, "kCGWindowBounds", {})
        return WindowInfo(
            title=str(self._get(w, "kCGWindowName", "")),  # needs Screen Recording permission
            owner=str(self._get(w, "kCGWindowOwnerName", "")),
            left=int(self._get(b, "X", 0)),
            top=int(self._get(b, "Y", 0)),
            width=int(self._get(b, "Width", 0)),
            height=int(self._get(b, "Height", 0)),
            focused=focused,
        )

    def foreground(self) -> Optional[WindowInfo]:
        wins = self._windows()  # front-to-back: the first one belongs to the active app
        return self._info(wins[0], True) if wins else None

    def find_roblox(self) -> Optional[WindowInfo]:
        wins = self._windows()
        front_owner = self._get(wins[0], "kCGWindowOwnerName", "") if wins else ""
        infos = []
        for w in wins:
            owner = str(self._get(w, "kCGWindowOwnerName", ""))
            if looks_like_roblox(str(self._get(w, "kCGWindowName", "")), owner):
                infos.append(self._info(w, owner == front_owner))
        infos = [i for i in infos if i.usable] or infos
        return infos[0] if infos else None


class _X11(_Backend):
    """Active-window title on X11 (best effort; Roblox has no official Linux client)."""

    name = "x11"

    def __init__(self) -> None:
        from Xlib import X, display  # python-xlib, installed together with pynput

        self._X = X
        self._d = display.Display()
        self._root = self._d.screen().root
        self._active = self._d.intern_atom("_NET_ACTIVE_WINDOW")
        self._net_name = self._d.intern_atom("_NET_WM_NAME")

    def foreground(self) -> Optional[WindowInfo]:
        prop = self._root.get_full_property(self._active, self._X.AnyPropertyType)
        if prop is None or not prop.value or not int(prop.value[0]):
            return None
        win = self._d.create_resource_object("window", int(prop.value[0]))
        name_prop = win.get_full_property(self._net_name, 0)
        name = name_prop.value if name_prop is not None else win.get_wm_name()
        if isinstance(name, bytes):
            name = name.decode("utf-8", "replace")
        return WindowInfo(str(name or ""), "", 0, 0, 0, 0, True)

    def find_roblox(self) -> Optional[WindowInfo]:
        return None  # the capture falls back to the monitor area


def _make_backend() -> tuple[Optional[_Backend], Optional[str]]:
    try:
        if sys.platform == "win32":
            return _Win32(), None
        if sys.platform == "darwin":
            return _Quartz(), None
        if sys.platform.startswith("linux"):
            return _X11(), None
    except Exception as exc:
        return None, f"can't list windows on this system ({exc.__class__.__name__}: {exc})"
    return None, f"window detection is not supported on {sys.platform}"


# ======================================================================= finder
class RobloxWindowFinder:
    """Cached, exception-safe access to the platform backend."""

    def __init__(self, backend: Optional[_Backend] = None, ttl_s: float = 0.5) -> None:
        self._backend = backend
        self._created = backend is not None
        self.error: Optional[str] = None
        self.ttl_s = ttl_s
        self._cache: Optional[WindowInfo] = None
        self._cache_t = -math.inf
        self._focus: Optional[bool] = None
        self._focus_t = -math.inf
        self._lock = threading.Lock()  # the X11 connection is not thread-safe

    @property
    def backend(self) -> Optional[_Backend]:
        with self._lock:
            if not self._created:
                self._created = True
                self._backend, self.error = _make_backend()
            return self._backend

    @property
    def supported(self) -> bool:
        return self.backend is not None

    def invalidate(self) -> None:
        self._cache_t = self._focus_t = -math.inf

    def find(self, max_age_s: Optional[float] = None) -> Optional[WindowInfo]:
        """The Roblox window, or ``None`` if it isn't open (or can't be detected)."""
        now = time.monotonic()
        if now - self._cache_t <= (self.ttl_s if max_age_s is None else max_age_s):
            return self._cache
        backend = self.backend
        info = None
        if backend is not None:
            with self._lock:
                try:
                    info = backend.find_roblox()
                except Exception as exc:
                    self.error = f"window lookup failed ({exc.__class__.__name__}: {exc})"
        self._cache, self._cache_t = info, now
        return info

    def roblox_focused(self, max_age_s: float = 0.0) -> Optional[bool]:
        """Is Roblox the active window? ``None`` if this system can't tell."""
        now = time.monotonic()
        if now - self._focus_t <= max_age_s:
            return self._focus
        backend = self.backend
        focused: Optional[bool] = None
        if backend is not None:
            with self._lock:
                try:
                    fg = backend.foreground()
                    focused = fg is not None and looks_like_roblox(fg.title, fg.owner)
                except Exception as exc:
                    self.error = f"active-window check failed ({exc.__class__.__name__}: {exc})"
        self._focus, self._focus_t = focused, now
        return focused
