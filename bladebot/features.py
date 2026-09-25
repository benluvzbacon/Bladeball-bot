"""Turns a stream of ball detections into the feature vector the network reads.

The same :class:`TrackState` class is used by the live bot, by the simulator
that generates training data and by the recording replayer, so the network
always sees features computed in exactly the same way (no train/serve skew).

Coordinate convention (set up by :mod:`bladebot.vision`):

* ``x``/``y`` are the ball centre *relative to your character* (the "anchor"),
  divided by the height of the processed frame. ``y`` grows downwards.
* ``r`` is the apparent ball radius divided by the frame height.

Dividing by the frame height makes the features independent of screen
resolution and capture-region size.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Optional, Sequence

FEATURE_NAMES: tuple[str, ...] = (
    "log_r",        # log apparent radius  -> how close the ball is
    "dx",           # horizontal offset from your character
    "dy",           # vertical offset from your character
    "log_d",        # log screen distance to your character
    "d_over_r",     # screen distance measured in ball radii (scale invariant)
    "loom_rate",    # d(log r)/dt  = 1 / time-to-contact from looming (smooth)
    "loom_q",       # d(log r)/dt from a quadratic fit (less lag, more noise)
    "loom_acc",     # how fast the looming speeds up (d2 log r / dt2)
    "vx",           # screen velocity (frame heights per second)
    "vy",
    "closing",      # speed at which the ball approaches your character on screen
    "screen_rate",  # closing / distance = 1 / time-to-contact on screen
    "speed",        # screen speed magnitude
    "log_char",     # log height of your character box -> how far the camera is zoomed out
    "r_rel",        # log(ball radius / character height) -> ball depth vs. your character
    "age",          # how many detections the track has (0..1)
    "has_vel",      # 1 once we have at least two detections
    "span",         # time covered by the track window (0..1)
    "stale",        # seconds since the last detection
)
DEFAULT_CHAR_H = 0.30  # character box height / frame height when unknown
N_FEATURES = len(FEATURE_NAMES)


def _clip(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


class TrackState:
    """Short history of detections of the ball that is targeting you."""

    def __init__(
        self,
        max_points: int = 16,
        window_s: float = 0.30,
        reset_gap_s: float = 0.25,
        tau_s: float = 0.10,
    ) -> None:
        self.max_points = max_points
        self.window_s = window_s
        self.reset_gap_s = reset_gap_s
        self.tau_s = tau_s
        self.points: deque[tuple[float, float, float, float]] = deque()

    # ----------------------------------------------------------------- update
    def reset(self) -> None:
        self.points.clear()

    @property
    def active(self) -> bool:
        return bool(self.points)

    @property
    def last_time(self) -> Optional[float]:
        return self.points[-1][0] if self.points else None

    def update(self, t: float, det: Optional[Sequence[float]]) -> None:
        """Feed one frame. ``det`` is ``(x, y, r)`` or ``None`` if no ball was seen."""
        pts = self.points
        if pts and t - pts[-1][0] > self.reset_gap_s:
            pts.clear()  # lost the ball for too long: this is a new approach
        if det is None:
            return
        x, y, r = float(det[0]), float(det[1]), float(det[2])
        if pts and t <= pts[-1][0]:
            return  # ignore out-of-order / duplicate timestamps
        pts.append((t, x, y, math.log(max(r, 1e-4))))
        while len(pts) > self.max_points or (len(pts) > 1 and t - pts[0][0] > self.window_s):
            pts.popleft()

    # --------------------------------------------------------------- features
    def _fit(self) -> tuple[float, float, float, float, float, float]:
        """Recency-weighted least-squares line fit of x, y and log r over time.

        Returns ``(x, y, log_r, vx, vy, loom_rate)`` evaluated at the newest point.
        """
        pts = self.points
        t_last = pts[-1][0]
        if len(pts) == 1:
            _, x, y, lr = pts[0]
            return x, y, lr, 0.0, 0.0, 0.0
        s0 = s1 = s2 = 0.0
        sx = sy = sl = stx = sty = stl = 0.0
        inv_tau = 1.0 / self.tau_s
        for t, x, y, lr in pts:
            dt = t - t_last  # <= 0
            w = math.exp(dt * inv_tau)
            s0 += w
            s1 += w * dt
            s2 += w * dt * dt
            sx += w * x
            sy += w * y
            sl += w * lr
            stx += w * dt * x
            sty += w * dt * y
            stl += w * dt * lr
        det = s0 * s2 - s1 * s1
        if det <= 1e-12:
            _, x, y, lr = pts[-1]
            return x, y, lr, 0.0, 0.0, 0.0
        vx = (s0 * stx - s1 * sx) / det
        vy = (s0 * sty - s1 * sy) / det
        vl = (s0 * stl - s1 * sl) / det
        x0 = (sx - vx * s1) / s0
        y0 = (sy - vy * s1) / s0
        l0 = (sl - vl * s1) / s0
        return x0, y0, l0, vx, vy, vl

    def _loom_quadratic(self, fallback: float) -> tuple[float, float]:
        """Slope and curvature of log r at the newest point from a quadratic fit."""
        pts = self.points
        if len(pts) < 4 or pts[-1][0] - pts[0][0] < 0.03:
            return fallback, 0.0
        t_last = pts[-1][0]
        inv_tau = 1.0 / (self.tau_s * 1.5)
        m = [[0.0] * 3 for _ in range(3)]
        b = [0.0, 0.0, 0.0]
        for t, _, _, lr in pts:
            dt = t - t_last
            w = math.exp(dt * inv_tau)
            p = (1.0, dt, dt * dt)
            for i in range(3):
                b[i] += w * p[i] * lr
                for j in range(3):
                    m[i][j] += w * p[i] * p[j]
        # solve the 3x3 system with Cramer's rule
        def det3(a: list[list[float]]) -> float:
            return (
                a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
                - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
                + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
            )

        d = det3(m)
        if abs(d) < 1e-18:
            return fallback, 0.0
        m1 = [[m[i][0], b[i], m[i][2]] for i in range(3)]
        m2 = [[m[i][0], m[i][1], b[i]] for i in range(3)]
        return det3(m1) / d, 2.0 * det3(m2) / d

    def features(self, t_now: float, char_h: Optional[float] = None) -> Optional[list[float]]:
        """Feature vector for the current moment, or ``None`` if there is no track.

        ``char_h`` is the height of your character box divided by the frame
        height (the size of the yellow box in the menu); it tells the network
        how far the camera is zoomed out.
        """
        if not self.points:
            return None
        n = len(self.points)
        t_first = self.points[0][0]
        t_last = self.points[-1][0]
        x, y, lr, vx, vy, loom = self._fit()
        loom_q, loom_acc = self._loom_quadratic(loom)
        lr = _clip(lr, -7.0, 0.0)
        r = math.exp(lr)
        vx = _clip(vx, -20.0, 20.0)
        vy = _clip(vy, -20.0, 20.0)
        d = math.hypot(x, y)
        closing = -(x * vx + y * vy) / max(d, 1e-3)
        closing = _clip(closing, -20.0, 20.0)
        log_char = math.log(_clip(char_h if char_h else DEFAULT_CHAR_H, 0.02, 1.5))
        return [
            lr,
            _clip(x, -2.0, 2.0),
            _clip(y, -2.0, 2.0),
            math.log(d + 0.01),
            min(d / r, 60.0),
            _clip(loom, -10.0, 40.0),
            _clip(loom_q, -10.0, 40.0),
            _clip(loom_acc, -200.0, 400.0),
            vx,
            vy,
            closing,
            _clip(closing / max(d, r, 0.01), -10.0, 40.0),
            min(math.hypot(vx, vy), 30.0),
            log_char,
            _clip(lr - log_char, -8.0, 3.0),
            n / self.max_points,
            1.0 if n >= 2 else 0.0,
            min((t_last - t_first) / self.window_s, 1.5),
            _clip(t_now - t_last, 0.0, 0.5),
        ]

    # --------------------------------------------------------------- baseline
    def heuristic_tti(self) -> Optional[float]:
        """Classic (non-neural) time-to-impact guess used as a comparison baseline.

        Uses the looming time-to-contact ``r / (dr/dt)`` and the screen-space
        time-to-contact ``distance / closing speed`` and keeps the smaller one.
        """
        if len(self.points) < 2:
            return None
        x, y, lr, vx, vy, loom = self._fit()
        r = math.exp(lr)
        d = math.hypot(x, y)
        closing = -(x * vx + y * vy) / max(d, 1e-3)
        guesses = []
        if loom > 1e-3:
            guesses.append(1.0 / loom)
        if closing > 1e-4:
            guesses.append(max(d - r, 0.0) / closing)
        return min(guesses) if guesses else None
