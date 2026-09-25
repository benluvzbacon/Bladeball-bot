"""Screen capture of the Roblox window area with ``mss`` (fast, cross-platform).

By default the capture follows the Roblox window (see :mod:`bladebot.window`),
so windowed and fullscreen Roblox both work without adjusting anything. If the
window can't be found it falls back to an area of the chosen monitor.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np

from .window import RobloxWindowFinder, WindowInfo


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

    def __init__(self, finder: Optional[RobloxWindowFinder] = None) -> None:
        self._sct: Any = None
        self.region: Optional[dict[str, int]] = None
        self.monitor_count = 0
        self.finder = finder or RobloxWindowFinder()
        self.target = "monitor"  # what is being captured right now: "roblox" or "monitor"
        self.window: Optional[WindowInfo] = None
        self.note: Optional[str] = None

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

    def grab(self, cfg: dict[str, Any]) -> tuple[np.ndarray, float]:
        """Return ``(rgb_view, timestamp)`` with the configured downscale applied."""
        sct = self._ensure()  # first: on Windows this also makes window coordinates DPI-exact
        monitors = sct.monitors
        self.monitor_count = max(len(monitors) - 1, 0)
        self.region = compute_region(self._base_area(cfg, monitors), cfg)
        t = time.perf_counter()
        try:
            shot = sct.grab(self.region)
        except Exception as exc:
            raise CaptureError(f"screen grab failed ({exc})") from exc
        h, w = int(shot.height), int(shot.width)
        raw = shot.raw
        if len(raw) == h * w * 4:
            bgra = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 4)  # zero-copy
        else:  # unusual row padding: let mss sort it out
            bgra = np.asarray(shot, dtype=np.uint8).reshape(h, w, 4)
        s = max(int(cfg["scale"]), 1)
        rgb = bgra[::s, ::s, 2::-1]  # BGRA -> RGB view, downscaled
        return rgb, t

    def close(self) -> None:
        if self._sct is not None:
            try:
                self._sct.close()
            except Exception:
                pass
            self._sct = None
