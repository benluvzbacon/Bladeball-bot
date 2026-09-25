"""Screen capture of the Roblox window area.

By default the capture follows the Roblox window (see :mod:`bladebot.window`),
so windowed and fullscreen Roblox both work without adjusting anything. If the
window can't be found it falls back to an area of the chosen monitor.

Two capture methods:

* **DXGI Desktop Duplication** (Windows, needs the optional ``dxcam``
  package): copies the region straight from the GPU in a couple of
  milliseconds and hands over each new frame as soon as it is on screen.
  Used automatically when available and the region is on the main monitor.
* **mss** (all systems): reliable, but a large region can take 10-25 ms per
  grab on Windows, which adds directly to the bot's delay.
"""

from __future__ import annotations

import sys
import time
from typing import Any, Optional

import numpy as np

from .window import RobloxWindowFinder, WindowInfo


def _short(exc: BaseException) -> str:
    text = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
    return f"{type(exc).__name__}: {text}"[:160] if text else type(exc).__name__


class DxgiGrabber:
    """Fast capture of the primary monitor through ``dxcam`` (Desktop Duplication API).

    The primary monitor always starts at desktop coordinate (0, 0), so a
    region fully inside ``(0, 0, width, height)`` can be grabbed directly.
    """

    def __init__(self) -> None:
        import dxcam  # optional dependency (Windows only)

        self._cam = dxcam.create(output_color="BGRA", processor_backend="numpy")
        self.width = int(self._cam.width)
        self.height = int(self._cam.height)

    def covers(self, region: dict[str, int]) -> bool:
        l, t = int(region["left"]), int(region["top"])
        return l >= 0 and t >= 0 and l + int(region["width"]) <= self.width and t + int(region["height"]) <= self.height

    def grab(self, region: dict[str, int]) -> Optional[np.ndarray]:
        """BGRA frame of ``region``, or ``None`` if nothing changed on screen since the last grab."""
        l, t = int(region["left"]), int(region["top"])
        box = (l, t, l + int(region["width"]), t + int(region["height"]))
        return self._cam.grab(region=box, copy=False, new_frame_only=True)

    def close(self) -> None:
        try:
            self._cam.release()
        except Exception:
            pass


class CaptureError(RuntimeError):
    pass


def clip_rect(rect: dict[str, int], bounds: dict[str, int]) -> Optional[dict[str, int]]:
    """``rect`` cut to ``bounds`` (e.g. a window partly off screen), or ``None`` if nothing is left."""
    left = max(int(rect["left"]), int(bounds["left"]))
    top = max(int(rect["top"]), int(bounds["top"]))
    right = min(int(rect["left"]) + int(rect["width"]), int(bounds["left"]) + int(bounds["width"]))
    bottom = min(int(rect["top"]) + int(rect["height"]), int(bounds["top"]) + int(bounds["height"]))
    if right - left < 16 or bottom - top < 16:
        return None
    return {"left": left, "top": top, "width": right - left, "height": bottom - top}


def compute_region(monitor: dict[str, int], cfg: dict[str, Any]) -> dict[str, int]:
    """Capture rectangle (absolute screen pixels) from the fractional settings.

    ``monitor`` is the area the fractions refer to: the Roblox window or a monitor.
    """
    mw, mh = int(monitor["width"]), int(monitor["height"])
    w = max(16, int(round(mw * float(cfg["region_w"]))))
    h = max(16, int(round(mh * float(cfg["region_h"]))))
    cx = mw / 2.0 + float(cfg["region_x"]) * mw
    cy = mh / 2.0 + float(cfg["region_y"]) * mh
    left = int(round(min(max(cx - w / 2.0, 0), mw - w)))
    top = int(round(min(max(cy - h / 2.0, 0), mh - h)))
    return {
        "left": int(monitor["left"]) + left,
        "top": int(monitor["top"]) + top,
        "width": min(w, mw),
        "height": min(h, mh),
    }


class ScreenSource:
    """Grabs the configured region. Create and use it on one thread only (mss rule)."""

    name = "screen"

    def __init__(self, finder: Optional[RobloxWindowFinder] = None, dxgi_factory: Any = None) -> None:
        self._sct: Any = None
        self.region: Optional[dict[str, int]] = None
        self.monitor_count = 0
        self.finder = finder or RobloxWindowFinder()
        self.target = "monitor"  # what is being captured right now: "roblox" or "monitor"
        self.window: Optional[WindowInfo] = None
        self.note: Optional[str] = None
        self.method = "mss"  # capture method used for the last frame: "dxgi" or "mss"
        self.dxgi_error: Optional[str] = None  # why the fast method isn't used
        self.grab_ms = 0.0  # smoothed time a grab takes
        self._dxgi: Optional[DxgiGrabber] = None
        self._dxgi_failed = False
        self._dxgi_factory = dxgi_factory or (DxgiGrabber if sys.platform == "win32" else None)

    def _base_area(self, cfg: dict[str, Any], monitors: list[dict[str, int]]) -> dict[str, int]:
        """The Roblox window if wanted and found, else the chosen monitor."""
        idx = int(cfg["monitor"])
        if idx < 1 or idx >= len(monitors):
            idx = 1 if len(monitors) > 1 else 0
        monitor = monitors[idx]
        self.note = None
        if cfg.get("capture_target", "roblox") == "roblox":
            win = self.finder.find()
            if win is not None and win.usable:
                area = clip_rect(win.rect, monitors[0])  # monitors[0] = all screens together
                if area is not None:
                    self.target, self.window = "roblox", win
                    return area
            if win is not None and win.minimized:
                self.note = "Roblox is minimised - capturing the monitor instead."
            elif win is not None:
                self.note = "The Roblox window is too small or off screen - capturing the monitor instead."
            else:
                self.note = "Roblox window not found - capturing the monitor instead."
        self.target, self.window = "monitor", None
        return monitor

    def _ensure(self) -> Any:
        if self._sct is None:
            try:
                import mss
            except ImportError as exc:  # pragma: no cover
                raise CaptureError("the 'mss' package is not installed (pip install -r requirements.txt)") from exc
            try:
                factory = getattr(mss, "MSS", None) or mss.mss
                self._sct = factory()
            except Exception as exc:
                raise CaptureError(f"screen capture is not available here ({exc})") from exc
            self.finder.invalidate()  # coordinates may change once mss makes the process DPI aware
        return self._sct

    def _fast_grabber(self, cfg: dict[str, Any]) -> Optional[DxgiGrabber]:
        if cfg.get("capture_method", "auto") != "auto" or self._dxgi_factory is None or self._dxgi_failed:
            return None
        if self._dxgi is None:
            try:
                self._dxgi = self._dxgi_factory()
            except ImportError:
                self._dxgi_failed = True
                self.dxgi_error = "the optional 'dxcam' package is not installed"
            except Exception as exc:
                self._dxgi_failed = True
                self.dxgi_error = f"fast capture unavailable ({_short(exc)})"
        return self._dxgi

    def _note_grab_time(self, t0: float) -> None:
        ms = (time.perf_counter() - t0) * 1000.0
        self.grab_ms = ms if self.grab_ms == 0.0 else 0.9 * self.grab_ms + 0.1 * ms

    def grab(self, cfg: dict[str, Any]) -> tuple[Optional[np.ndarray], float]:
        """Return ``(rgb_view, timestamp)`` with the configured downscale applied.

        The frame is ``None`` when the fast capture reports that nothing new has
        been drawn since the previous grab (just ask again a moment later).
        """
        sct = self._ensure()  # first: on Windows this also makes window coordinates DPI-exact
        monitors = sct.monitors
        self.monitor_count = max(len(monitors) - 1, 0)
        self.region = compute_region(self._base_area(cfg, monitors), cfg)
        s = max(int(cfg["scale"]), 1)
        fast = self._fast_grabber(cfg)
        if fast is not None and fast.covers(self.region):
            t = time.perf_counter()
            try:
                bgra = fast.grab(self.region)
            except Exception as exc:  # lost access (UAC prompt, display change...): use mss from now on
                self.dxgi_error = f"fast capture stopped ({_short(exc)}) - using mss"
                self._dxgi_failed = True
                fast.close()
                self._dxgi = None
            else:
                self.method = "dxgi"
                if bgra is None:
                    return None, t
                self._note_grab_time(t)
                return bgra[::s, ::s, 2::-1], t
        elif fast is not None:
            self.dxgi_error = "fast capture only works on the main monitor - using mss"
        self.method = "mss"
        t = time.perf_counter()
        try:
            shot = sct.grab(self.region)
        except Exception as exc:
            raise CaptureError(f"screen grab failed ({exc})") from exc
        self._note_grab_time(t)
        h, w = int(shot.height), int(shot.width)
        raw = shot.raw
        if len(raw) == h * w * 4:
            bgra = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 4)  # zero-copy
        else:  # unusual row padding: let mss sort it out
            bgra = np.asarray(shot, dtype=np.uint8).reshape(h, w, 4)
        rgb = bgra[::s, ::s, 2::-1]  # BGRA -> RGB view, downscaled
        return rgb, t

    def close(self) -> None:
        if self._dxgi is not None:
            self._dxgi.close()
            self._dxgi = None
        if self._sct is not None:
            try:
                self._sct.close()
            except Exception:
                pass
            self._sct = None
