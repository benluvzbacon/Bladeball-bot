"""Screen capture of the Roblox window area with ``mss`` (fast, cross-platform)."""

from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np


class CaptureError(RuntimeError):
    pass


def compute_region(monitor: dict[str, int], cfg: dict[str, Any]) -> dict[str, int]:
    """Capture rectangle (absolute screen pixels) from the fractional settings."""
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

    def __init__(self) -> None:
        self._sct: Any = None
        self.region: Optional[dict[str, int]] = None
        self.monitor_count = 0

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
        return self._sct

    def grab(self, cfg: dict[str, Any]) -> tuple[np.ndarray, float]:
        """Return ``(rgb_view, timestamp)`` with the configured downscale applied."""
        sct = self._ensure()
        monitors = sct.monitors
        self.monitor_count = max(len(monitors) - 1, 0)
        idx = int(cfg["monitor"])
        if idx < 1 or idx >= len(monitors):
            idx = 1 if len(monitors) > 1 else 0
        self.region = compute_region(monitors[idx], cfg)
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
