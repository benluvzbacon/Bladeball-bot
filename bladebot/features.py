"""Turns a stream of ball detections into the feature vector the network reads.

The same classes are used by the live bot, by the simulator that generates
training data and by the recording replayer, so the network always sees
features computed in exactly the same way (no train/serve skew).

Coordinate convention (set up by :mod:`bladebot.vision`):

* ``x``/``y`` are the ball centre *relative to your character* (the "anchor"),
  divided by the height of the processed frame. ``y`` grows downwards.
* ``r`` is the apparent ball radius divided by the frame height.

Dividing by the frame height makes the features independent of screen
resolution and capture-region size.

Besides the ball's current position and motion, the network also gets:

* **curvature** - a slower quadratic fit that measures how the ball's path
  bends (curve balls),
* **the episode** - how long ago you turned red (the ball was deflected at
  you), whether and since when the ball has been seen, and the fastest
  approach seen so far - so a ball that curves out of view isn't forgotten,
* **the rally rhythm** - how long the ball was away before it came back and
  how long the previous ball took to reach you, which predicts when this one
  will arrive even before it's on screen.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

FEATURE_NAMES: tuple[str, ...] = (
    # --- where the ball is now (fast fit)
    "log_r",        # log apparent radius  -> how close the ball is
    "dx",           # horizontal offset from your character
    "dy",           # vertical offset from your character
    "log_d",        # log screen distance to your character
    "d_over_r",     # screen distance measured in ball radii (scale invariant)
    "loom_rate",    # d(log r)/dt  = 1 / time-to-contact from looming
    "vx",           # screen velocity (frame heights per second)
    "vy",
    "closing",      # speed at which the ball approaches your character on screen
    "screen_rate",  # closing / distance = 1 / time-to-contact on screen
    # --- how the path bends (slow quadratic fit)
    "loom_slow",    # smoother looming rate
    "loom_acc",     # how fast the looming speeds up (d2 log r / dt2)
    "ax",           # screen acceleration
    "ay",
    "turn",         # signed turn rate of the screen path (rad/s)
    "bend",         # acceleration towards your character on screen
    # --- rough 3D motion in ball radii (independent of zoom)
    "lat_speed",    # sideways speed
    "depth_speed",  # speed towards the camera
    "speed3",       # total speed
    "tti_straight", # log(distance / speed): arrival time if it flew straight at you
    # --- scene
    "log_char",     # log height of your character box -> how far the camera is zoomed out
    "r_rel",        # log(ball radius / character height) -> ball depth vs. your character
    # --- track quality
    "n_eff",        # how many recent detections the fast fit is based on
    "has_vel",      # 1 once the current sighting has at least two detections
    "seg_age",      # how long the ball has been continuously tracked
    "stale",        # seconds since the ball was last seen
    # --- this episode (since you turned red)
    "ep_age",       # seconds since you became the target
    "seen",         # 1 if the ball has been seen since then
    "seen_age",     # seconds since it was first seen
    "loom_max",     # fastest looming seen in this episode
    "speed_max",    # fastest 3D speed seen in this episode
    "r_max",        # biggest the ball has looked in this episode
    # --- the rally rhythm
    "has_prev",     # 1 if the ball came back to you recently
    "prev_gap",     # seconds it spent away before coming back
    "prev_dur",     # seconds the previous ball took to reach you
    "age_gap",      # ep_age / prev_gap
    "age_dur",      # ep_age / prev_dur
)
DEFAULT_CHAR_H = 0.30  # character box height / frame height when unknown
N_FEATURES = len(FEATURE_NAMES)
FOCAL_REL = 0.9  # focal length / capture height for a 70 deg camera and a 0.8 high capture
BALL_TO_CHAR = 0.2  # ball radius / character box height at the same distance (roughly)
RALLY_MAX_S = 4.0  # rally timings longer than this count as unknown
_UNSEEN_LOG_R = -7.0


def _clip(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _slog(v: float) -> float:
    """Signed log1p: keeps the sign, compresses large magnitudes."""
    return math.copysign(math.log1p(abs(v)), v)


class ExpFit:
    """Exponentially weighted least-squares polynomial fit, updated in O(1).

    Fits ``k`` signals at once against time with weights ``exp((t_i - t_last) / tau)``
    and returns value / slope / (curvature) at the newest sample.
    """

    __slots__ = ("inv_tau", "deg", "k", "S", "B", "t", "n")

    def __init__(self, tau: float, deg: int, k: int) -> None:
        if deg not in (1, 2):
            raise ValueError("deg must be 1 or 2")
        self.inv_tau = 1.0 / tau
        self.deg = deg
        self.k = k
        self.reset()

    def reset(self) -> None:
        self.S = [0.0] * (2 * self.deg + 1)
        self.B = [[0.0] * (self.deg + 1) for _ in range(self.k)]
        self.t: Optional[float] = None
        self.n = 0

    def add(self, t: float, values: Sequence[float]) -> None:
        if self.t is not None:
            d = t - self.t
            if d > 0:
                a = math.exp(-d * self.inv_tau)
                m = -d
                S = self.S
                if self.deg == 1:
                    s0, s1, s2 = S
                    self.S = [a * s0, a * (s1 + m * s0), a * (s2 + 2 * m * s1 + m * m * s0)]
                    for b in self.B:
                        b0, b1 = b
                        b[0] = a * b0
                        b[1] = a * (b1 + m * b0)
                else:
                    s0, s1, s2, s3, s4 = S
                    m2 = m * m
                    m3 = m2 * m
                    self.S = [
                        a * s0,
                        a * (s1 + m * s0),
                        a * (s2 + 2 * m * s1 + m2 * s0),
                        a * (s3 + 3 * m * s2 + 3 * m2 * s1 + m3 * s0),
                        a * (s4 + 4 * m * s3 + 6 * m2 * s2 + 4 * m3 * s1 + m2 * m2 * s0),
                    ]
                    for b in self.B:
                        b0, b1, b2 = b
                        b[0] = a * b0
                        b[1] = a * (b1 + m * b0)
                        b[2] = a * (b2 + 2 * m * b1 + m2 * b0)
        self.S[0] += 1.0
        for b, v in zip(self.B, values):
            b[0] += v
        self.t = t
        self.n += 1

    @property
    def weight(self) -> float:
        """Effective number of samples."""
        return self.S[0]

    def solve(self) -> list[tuple[float, float, float]]:
        """``(value, slope, second_derivative)`` per signal at the newest sample."""
        S = self.S
        if self.n == 0:
            return [(0.0, 0.0, 0.0)] * self.k
        if self.deg == 2 and self.n >= 4:
            s0, s1, s2, s3, s4 = S
            # inverse of the symmetric 3x3 moment matrix via cofactors
            c00 = s2 * s4 - s3 * s3
            c01 = s2 * s3 - s1 * s4
            c02 = s1 * s3 - s2 * s2
            c11 = s0 * s4 - s2 * s2
            c12 = s1 * s2 - s0 * s3
            c22 = s0 * s2 - s1 * s1
            det = s0 * c00 + s1 * c01 + s2 * c02
            scale = s0 * s2 * s4
            if scale > 0 and det > 1e-9 * scale:
                out = []
                for b0, b1, b2 in self.B:
                    v = (c00 * b0 + c01 * b1 + c02 * b2) / det
                    sl = (c01 * b0 + c11 * b1 + c12 * b2) / det
                    cu = (c02 * b0 + c12 * b1 + c22 * b2) / det
                    out.append((v, sl, 2.0 * cu))
                return out
        s0, s1, s2 = S[0], S[1], S[2]
        det = s0 * s2 - s1 * s1
        if self.n >= 2 and det > 1e-12:
            out = []
            for b in self.B:
                sl = (s0 * b[1] - s1 * b[0]) / det
                out.append(((b[0] - sl * s1) / s0, sl, 0.0))
            return out
        return [(b[0] / s0, 0.0, 0.0) for b in self.B]


class Episode:
    """Tracks "the ball is targeting me" periods and the rhythm of the rally.

    An episode starts on the first frame you are highlighted red and ends once
    the highlight has been gone for ``rearm_s``. The gap between two episodes
    is how long the ball spent with other players; together with the length of
    the previous episode it predicts when the next ball will arrive.
    """

    def __init__(self, rearm_s: float = 0.06, unknown_after_s: float = RALLY_MAX_S) -> None:
        self.rearm_s = rearm_s
        self.unknown_after_s = unknown_after_s
        self.reset()

    def reset(self) -> None:
        self.active = False
        self.start: Optional[float] = None
        self.last_on: Optional[float] = None
        self.prev_end: Optional[float] = None
        self.prev_len: Optional[float] = None
        self.gap: Optional[float] = None  # context of the current episode
        self.prev_dur: Optional[float] = None
        self.started_now = False

    def update(self, t: float, targeted: bool) -> None:
        self.started_now = False
        if targeted:
            if not self.active:
                self.active = True
                self.start = t
                self.started_now = True
                gap = t - self.prev_end if self.prev_end is not None else None
                if gap is None or gap > self.unknown_after_s:
                    self.gap, self.prev_dur = None, None
                else:
                    self.gap, self.prev_dur = gap, self.prev_len
            self.last_on = t
        elif self.active and self.last_on is not None and t - self.last_on > self.rearm_s:
            self.active = False
            self.prev_end = self.last_on
            self.prev_len = self.last_on - (self.start if self.start is not None else self.last_on)


class TrackState:
    """Detections of the ball that is targeting you, plus what we know about this episode."""

    def __init__(
        self,
        reset_gap_s: float = 0.25,
        tau_fast_s: float = 0.07,
        tau_slow_s: float = 0.2,
    ) -> None:
        self.reset_gap_s = reset_gap_s
        self.fast = ExpFit(tau_fast_s, 1, 3)
        self.slow = ExpFit(tau_slow_s, 2, 3)
        self.reset()

    # ----------------------------------------------------------------- update
    def reset(self) -> None:
        """Forget everything (a new episode)."""
        self.fast.reset()
        self.slow.reset()
        self.last_time: Optional[float] = None
        self.first_seen: Optional[float] = None
        self.seg_start: Optional[float] = None
        self.seg_n = 0
        self.loom_max = 0.0
        self.speed_max = 0.0
        self.r_max = _UNSEEN_LOG_R
        self._state: Optional[dict[str, float]] = None

    @property
    def active(self) -> bool:
        """True once the ball has been seen (since the last reset)."""
        return self.last_time is not None

    def recent(self, t: float, within_s: Optional[float] = None) -> bool:
        within = self.reset_gap_s if within_s is None else within_s
        return self.last_time is not None and t - self.last_time <= within

    def update(self, t: float, det: Optional[Sequence[float]]) -> None:
        """Feed one frame. ``det`` is ``(x, y, r)`` or ``None`` if no ball was seen."""
        if det is None:
            return
        if self.last_time is not None and t <= self.last_time:
            return  # ignore out-of-order / duplicate timestamps
        if self.last_time is None or t - self.last_time > self.reset_gap_s:
            self.fast.reset()  # lost the ball for a while: start a new sighting
            self.slow.reset()
            self.seg_start = t
            self.seg_n = 0
        x, y, r = float(det[0]), float(det[1]), float(det[2])
        lr = math.log(max(r, 1e-4))
        vals = (x, y, lr)
        self.fast.add(t, vals)
        self.slow.add(t, vals)
        self.seg_n += 1
        self.last_time = t
        if self.first_seen is None:
            self.first_seen = t
        self._state = self._compute()
        st = self._state
        if self.seg_n >= 3:
            self.loom_max = max(self.loom_max, st["loom"])
            self.speed_max = max(self.speed_max, st["speed3"])
        self.r_max = max(self.r_max, st["lr"])

    def _compute(self) -> dict[str, float]:
        (x, vx, _), (y, vy, _), (lr, loom, _) = self.fast.solve()
        (_, vxs, axs), (_, vys, ays), (_, loom_s, loom_acc) = self.slow.solve()
        lr = _clip(lr, _UNSEEN_LOG_R, 0.0)
        r = math.exp(lr)
        vx = _clip(vx, -20.0, 20.0)
        vy = _clip(vy, -20.0, 20.0)
        loom = _clip(loom, -10.0, 40.0)
        d = math.hypot(x, y)
        closing = _clip(-(x * vx + y * vy) / max(d, 1e-3), -20.0, 20.0)
        sp2 = vxs * vxs + vys * vys
        turn = (vxs * ays - vys * axs) / sp2 if sp2 > 1e-4 else 0.0
        bend = -(x * axs + y * ays) / max(d, 1e-3)
        # rough 3D motion in ball radii: sideways from the screen motion, depth from looming
        lat = math.hypot(vx - x * loom, vy - y * loom) / max(r, 1e-3)
        depth = FOCAL_REL * loom / max(r, 1e-3)
        speed3 = math.hypot(lat, depth)
        return {
            "x": x, "y": y, "lr": lr, "r": r, "vx": vx, "vy": vy, "loom": loom, "d": d,
            "closing": closing, "loom_s": _clip(loom_s, -10.0, 40.0), "loom_acc": _clip(loom_acc, -200.0, 400.0),
            "ax": _clip(axs, -200.0, 200.0), "ay": _clip(ays, -200.0, 200.0), "turn": _clip(turn, -40.0, 40.0),
            "bend": _clip(bend, -200.0, 200.0), "lat": lat, "depth": depth, "speed3": speed3,
        }

    # --------------------------------------------------------------- features
    def features(
        self,
        t_now: float,
        char_h: Optional[float] = None,
        ep_start: Optional[float] = None,
        prev_gap: Optional[float] = None,
        prev_dur: Optional[float] = None,
    ) -> list[float]:
        """Feature vector for the current moment.

        ``char_h`` is the height of your character box divided by the frame
        height (the size of the yellow box in the menu); it tells the network
        how far the camera is zoomed out. ``ep_start`` is when you became the
        target (defaults to when the ball was first seen) and ``prev_gap`` /
        ``prev_dur`` describe the rally so far (``None`` = unknown).
        """
        log_char = math.log(_clip(char_h if char_h else DEFAULT_CHAR_H, 0.02, 1.5))
        if ep_start is None:
            ep_start = self.first_seen if self.first_seen is not None else t_now
        ep_age = _clip(t_now - ep_start, 0.0, RALLY_MAX_S)
        st = self._state
        if st is None:
            track = [
                _UNSEEN_LOG_R, 0.0, 0.0, math.log(0.01), 80.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, math.log(8.0),
                log_char, _clip(_UNSEEN_LOG_R - log_char, -8.0, 3.0),
                0.0, 0.0, 0.0, 2.0,
            ]
            seen, seen_age = 0.0, 0.0
        else:
            x, y, r, d, loom = st["x"], st["y"], st["r"], st["d"], st["loom"]
            last = self.last_time if self.last_time is not None else t_now
            seg_start = self.seg_start if self.seg_start is not None else last
            first = self.first_seen if self.first_seen is not None else last
            # distance to you in ball radii (depth from the size difference to your character)
            r_char = BALL_TO_CHAR * math.exp(log_char)
            depth_gap = FOCAL_REL * (1.0 / max(r, 1e-3) - 1.0 / r_char)
            dist3 = math.sqrt((x / r) ** 2 + (y / r) ** 2 + depth_gap * depth_gap)
            tti_straight = _clip(dist3 / max(st["speed3"], 1e-3), 0.01, 8.0)
            track = [
                st["lr"],
                _clip(x, -2.0, 2.0),
                _clip(y, -2.0, 2.0),
                math.log(d + 0.01),
                min(d / r, 80.0),
                loom,
                st["vx"],
                st["vy"],
                st["closing"],
                _clip(st["closing"] / max(d, r, 0.01), -10.0, 40.0),
                st["loom_s"],
                st["loom_acc"],
                st["ax"],
                st["ay"],
                st["turn"],
                st["bend"],
                math.log1p(min(st["lat"], 5000.0)),
                _slog(_clip(st["depth"], -5000.0, 5000.0)),
                math.log1p(min(st["speed3"], 5000.0)),
                math.log(tti_straight),
                log_char,
                _clip(st["lr"] - log_char, -8.0, 3.0),
                min(self.fast.weight / 5.0, 3.0),
                1.0 if self.seg_n >= 2 else 0.0,
                _clip(last - seg_start, 0.0, 1.0),
                _clip(t_now - last, 0.0, 2.0),
            ]
            seen = 1.0
            seen_age = _clip(t_now - first, 0.0, RALLY_MAX_S)
        known = prev_gap is not None and prev_dur is not None and 0 < prev_gap <= RALLY_MAX_S
        gap = _clip(prev_gap, 0.0, RALLY_MAX_S) if known else RALLY_MAX_S
        dur = _clip(prev_dur, 0.0, RALLY_MAX_S) if known and prev_dur is not None else RALLY_MAX_S
        return track + [
            ep_age,
            seen,
            seen_age,
            self.loom_max,
            math.log1p(min(self.speed_max, 5000.0)),
            self.r_max,
            1.0 if known else 0.0,
            gap,
            dur,
            _clip(ep_age / max(gap, 0.02), 0.0, 4.0) if known else 0.0,
            _clip(ep_age / max(dur, 0.02), 0.0, 4.0) if known else 0.0,
        ]

    # --------------------------------------------------------------- baseline
    def heuristic_tti(self) -> Optional[float]:
        """Classic (non-neural) time-to-impact guess used as a comparison baseline.

        Uses the looming time-to-contact ``r / (dr/dt)`` and the screen-space
        time-to-contact ``distance / closing speed`` and keeps the smaller one.
        """
        st = self._state
        if st is None or self.seg_n < 2:
            return None
        guesses = []
        if st["loom"] > 1e-3:
            guesses.append(1.0 / st["loom"])
        if st["closing"] > 1e-4:
            guesses.append(max(st["d"] - st["r"], 0.0) / st["closing"])
        return min(guesses) if guesses else None
