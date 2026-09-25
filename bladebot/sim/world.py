"""A small Blade Ball-like 3D world used to train and test the bot offline.

Units are Roblox studs, +Y is up. Your character stands at the origin; the
ball homes in on your torso like the real ball does when it turns red. A
Roblox-style third-person camera (distance / pitch / field of view / optional
shift-lock offset) projects everything onto a virtual screen, which lets us
produce exactly the kind of detections the screen-reading bot sees.

Curve balls
-----------
In Blade Ball the ball leaves a deflect in the direction the deflecting
player's camera is looking and then bends towards its new target. Flicking
the camera just before a deflect therefore sends it out sideways, high up or
even backwards before it curves round to you. This module models that with
five *curve types* (``CURVE_NAMES``: straight at you, out to the side, high,
backwards, anything) and three different *steering laws* (``LAW_NAMES``)
because nobody outside the game's developers knows the exact formula:

* ``distance`` - the turn rate grows as the ball gets closer (pursuit curve),
* ``rate``     - a fixed turn rate that tightens over time,
* ``blend``    - the direction is blended from the launch direction to "at
  you" over a set time.

On top of that the ball can speed up during a flight, your character can be
moving and the camera can be turned while the ball is in the air.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

TORSO = np.array([0.0, 3.0, 0.0])  # centre of your character's hitbox
FOCUS_HEIGHT = 4.5  # the Roblox camera orbits around your head
HIT_RADIUS = 2.0  # impact when |ball - TORSO| <= ball_radius + HIT_RADIUS
CHAR_HALF_WIDTH = 1.4  # studs, including arms
CHAR_HALF_HEIGHT = 2.9
PARRY_WINDOW_S = 0.5  # the block shield lasts 500 ms ...
WHIFF_COOLDOWN_S = 2.0  # ... and missing it costs a 2 s cooldown

CURVE_NAMES: tuple[str, ...] = ("direct", "side", "high", "back", "wild")
LAW_NAMES: tuple[str, ...] = ("distance", "rate", "blend")
# How often each curve type appears in the training data. Straight shots are
# still the most common, but a third of all balls are sharp curves.
TRAINING_CURVE_MIX: tuple[float, ...] = (0.42, 0.2, 0.15, 0.15, 0.08)
LAW_MIX: tuple[float, ...] = (0.35, 0.35, 0.30)
# Rally context: no previous ball (new round), a 1v1-style back-and-forth, or
# a free-for-all where the ball visited other players in between.
RALLY_MIX: tuple[float, ...] = (0.3, 0.5, 0.2)


@dataclass
class Camera:
    """Third-person camera looking at your head from behind (like Roblox)."""

    yaw: float = 0.0  # radians, 0 = looking towards +Z
    pitch: float = math.radians(18)  # radians, positive = looking down
    distance: float = 14.0  # studs from the focus point (mouse-wheel zoom)
    fov: float = math.radians(70)  # vertical field of view (Roblox default 70 deg)
    shift: float = 0.0  # sideways shoulder offset (shift-lock is ~1.75 studs)
    width: int = 1280
    height: int = 720
    crop_w: float = 1.0  # centred capture region, as a fraction of the screen
    crop_h: float = 1.0
    forward: np.ndarray = field(init=False, repr=False)
    right: np.ndarray = field(init=False, repr=False)
    up: np.ndarray = field(init=False, repr=False)
    position: np.ndarray = field(init=False, repr=False)
    focal: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        self.forward = np.array([cp * sy, -sp, cp * cy])
        self.right = np.array([cy, 0.0, -sy])
        self.up = np.array([sp * sy, cp, sp * cy])
        focus = np.array([0.0, FOCUS_HEIGHT, 0.0])
        self.position = focus - self.distance * self.forward + self.shift * self.right
        self.focal = (self.height / 2.0) / math.tan(self.fov / 2.0)

    # ------------------------------------------------------------ projection
    def project(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """World points ``(N, 3)`` -> screen ``u``, ``v`` (pixels, v down) and depth ``z``."""
        rel = np.asarray(pts, dtype=np.float64) - self.position
        xc = rel @ self.right
        yc = rel @ self.up
        zc = rel @ self.forward
        zs = np.maximum(zc, 1e-6)
        u = self.width / 2.0 + self.focal * xc / zs
        v = self.height / 2.0 - self.focal * yc / zs
        return u, v, zc

    def project_moving(
        self, pts: np.ndarray, yaw: np.ndarray, pitch: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Like :meth:`project`, but point ``k`` is seen with camera angles ``yaw[k]``, ``pitch[k]``.

        Used for frames where the player turns the camera while the ball flies.
        """
        pts = np.asarray(pts, dtype=np.float64)
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        forward = np.stack([cp * sy, -sp, cp * cy], axis=1)
        right = np.stack([cy, np.zeros_like(cy), -sy], axis=1)
        up = np.stack([sp * sy, cp, sp * cy], axis=1)
        position = np.array([0.0, FOCUS_HEIGHT, 0.0]) - self.distance * forward + self.shift * right
        rel = pts - position
        xc = np.einsum("ij,ij->i", rel, right)
        yc = np.einsum("ij,ij->i", rel, up)
        zc = np.einsum("ij,ij->i", rel, forward)
        zs = np.maximum(zc, 1e-6)
        return self.width / 2.0 + self.focal * xc / zs, self.height / 2.0 - self.focal * yc / zs, zc

    @property
    def region(self) -> tuple[float, float, float, float]:
        """Capture region ``(x0, y0, width, height)`` in screen pixels."""
        rw, rh = self.crop_w * self.width, self.crop_h * self.height
        return (self.width - rw) / 2.0, (self.height - rh) / 2.0, rw, rh

    def anchor(self) -> tuple[float, float]:
        """Screen position of your character's torso (what you click to calibrate)."""
        u, v, _ = self.project(TORSO[None, :])
        return float(u[0]), float(v[0])

    def character_box(self) -> tuple[float, float, float, float]:
        """``(u, v, half_w, half_h)`` of your character on screen, in pixels."""
        u, v, z = self.project(TORSO[None, :])
        z0 = max(float(z[0]), 0.5)
        return float(u[0]), float(v[0]), self.focal * CHAR_HALF_WIDTH / z0, self.focal * CHAR_HALF_HEIGHT / z0


# ------------------------------------------------------------------ flight
@dataclass
class Flight:
    """How each ball of a batch steers and speeds up (arrays of length ``n``).

    ``law`` picks the steering law (index into ``LAW_NAMES``):

    * 0 ``distance``: ``gain`` = homing strength (turn rate ~ gain * speed / distance)
    * 1 ``rate``: ``gain`` = turn rate in 1/s at launch, growing by ``gain2`` per second
    * 2 ``blend``: the heading is blended from ``dir0`` to "at you" over ``gain``
      seconds, with ``gain2`` shaping the blend (``(t / gain) ** gain2``)
    """

    law: np.ndarray
    gain: np.ndarray
    gain2: np.ndarray
    dir0: np.ndarray  # (n, 3) launch direction (used by the blend law)
    accel: np.ndarray  # relative speed-up per second (0 = constant speed)
    accel_abs: np.ndarray  # extra speed-up in studs/s per second (round start)
    vmax: np.ndarray  # speed cap in studs/s

    @property
    def n(self) -> int:
        return int(self.law.shape[0])

    def subset(self, idx: np.ndarray) -> "Flight":
        return Flight(
            self.law[idx], self.gain[idx], self.gain2[idx], self.dir0[idx],
            self.accel[idx], self.accel_abs[idx], self.vmax[idx],
        )

    @staticmethod
    def homing(gain: Sequence[float] | np.ndarray, direction: np.ndarray) -> "Flight":
        """The classic distance-based homing at constant speed."""
        g = np.atleast_1d(np.asarray(gain, dtype=np.float64))
        n = g.shape[0]
        z = np.zeros(n)
        return Flight(np.zeros(n, np.int64), g, z.copy(), np.atleast_2d(direction).astype(np.float64), z.copy(), z.copy(), np.full(n, np.inf))


def flight_step(
    pos: np.ndarray,
    direction: np.ndarray,
    speed: np.ndarray,
    t: float | np.ndarray,
    fl: Flight,
    dt: float,
    char_vel: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Advance a batch of balls by ``dt`` seconds.

    ``pos`` is relative to your character (who stands at the origin and may be
    moving with velocity ``char_vel``); ``t`` is the time since the deflect.
    Returns ``(new_pos, new_dir, new_speed)``.
    """
    to_t = TORSO - pos
    dist = np.linalg.norm(to_t, axis=-1, keepdims=True)
    desired = to_t / np.maximum(dist, 1e-6)
    spd = speed[:, None]
    t_arr = np.broadcast_to(np.asarray(t, dtype=np.float64), speed.shape)[:, None]
    gain = fl.gain[:, None]
    law = fl.law[:, None]
    # 0: distance based (pursuit); 1: time based turn rate that tightens
    k_dist = np.minimum(gain * spd * dt / np.maximum(dist, 2.0), 1.0)
    k_rate = 1.0 - np.exp(-gain * (1.0 + fl.gain2[:, None] * t_arr) * dt)
    # near you every law homes in hard enough to arrive (no endless orbits)
    k_rate = np.maximum(k_rate, np.minimum(1.5 * spd * dt / np.maximum(dist, 2.0), 1.0))
    k = np.where(law == 0, k_dist, k_rate)
    new_dir = direction + k * (desired - direction)
    # 2: blend from the launch direction towards the target
    if np.any(fl.law == 2):
        s = np.clip(t_arr / np.maximum(gain, 1e-3), 0.0, 1.0) ** np.maximum(fl.gain2[:, None], 0.1)
        blend = (1.0 - s) * fl.dir0 + s * desired
        new_dir = np.where(law == 2, blend, new_dir)
    norm = np.linalg.norm(new_dir, axis=-1, keepdims=True)
    new_dir = np.where(norm > 1e-6, new_dir / np.maximum(norm, 1e-12), desired)
    new_speed = np.minimum(speed * (1.0 + fl.accel * dt) + fl.accel_abs * dt, fl.vmax)
    step = new_dir * (new_speed[:, None] * dt)
    if char_vel is not None:
        step = step - char_vel * dt
    return pos + step, new_dir, new_speed


class OneBall:
    """A single ball flying with :func:`flight_step`'s rules, in plain Python (fast for n = 1).

    Used by the practice arena; ``tests/test_sim.py`` checks it matches the batch version.
    """

    __slots__ = ("pos", "dir", "speed", "t", "law", "gain", "gain2", "dir0", "accel", "accel_abs", "vmax")

    def __init__(self, pos: Sequence[float], direction: Sequence[float], speed: float, fl: Flight, i: int = 0) -> None:
        self.pos = [float(v) for v in pos]
        self.dir = [float(v) for v in direction]
        self.speed = float(speed)
        self.t = 0.0
        self.law = int(fl.law[i])
        self.gain = float(fl.gain[i])
        self.gain2 = float(fl.gain2[i])
        self.dir0 = [float(v) for v in fl.dir0[i]]
        self.accel = float(fl.accel[i])
        self.accel_abs = float(fl.accel_abs[i])
        self.vmax = float(fl.vmax[i])

    def copy(self) -> "OneBall":
        other = OneBall.__new__(OneBall)
        for name in OneBall.__slots__:
            v = getattr(self, name)
            setattr(other, name, list(v) if isinstance(v, list) else v)
        return other

    def distance(self) -> float:
        px, py, pz = self.pos
        return math.sqrt(px * px + (py - TORSO[1]) * (py - TORSO[1]) + pz * pz)

    def step(self, dt: float) -> None:
        px, py, pz = self.pos
        tx, ty, tz = -px, float(TORSO[1]) - py, -pz
        dist = math.sqrt(tx * tx + ty * ty + tz * tz)
        inv = 1.0 / max(dist, 1e-6)
        ex, ey, ez = tx * inv, ty * inv, tz * inv
        dx, dy, dz = self.dir
        spd = self.speed
        if self.law == 2:
            s = min(max(self.t / max(self.gain, 1e-3), 0.0), 1.0) ** max(self.gain2, 0.1)
            ox, oy, oz = self.dir0
            nx, ny, nz = (1 - s) * ox + s * ex, (1 - s) * oy + s * ey, (1 - s) * oz + s * ez
        else:
            if self.law == 0:
                k = min(self.gain * spd * dt / max(dist, 2.0), 1.0)
            else:
                k = 1.0 - math.exp(-self.gain * (1.0 + self.gain2 * self.t) * dt)
                k = max(k, min(1.5 * spd * dt / max(dist, 2.0), 1.0))
            nx, ny, nz = dx + k * (ex - dx), dy + k * (ey - dy), dz + k * (ez - dz)
        norm = math.sqrt(nx * nx + ny * ny + nz * nz)
        if norm > 1e-6:
            nx, ny, nz = nx / norm, ny / norm, nz / norm
        else:
            nx, ny, nz = ex, ey, ez
        spd = min(spd * (1.0 + self.accel * dt) + self.accel_abs * dt, self.vmax)
        self.pos = [px + nx * spd * dt, py + ny * spd * dt, pz + nz * spd * dt]
        self.dir = [nx, ny, nz]
        self.speed = spd
        self.t += dt

    def time_to_hit(self, hit_radius: float, dt: float, t_max: float = 8.0) -> Optional[float]:
        """Seconds until this ball reaches you if it keeps flying (``None`` if not within ``t_max``)."""
        ball = self.copy()
        t = 0.0
        if ball.distance() <= hit_radius:
            return 0.0
        while t < t_max:
            ball.step(dt)
            t += dt
            if ball.distance() <= hit_radius:
                return t
        return None


def homing_step(
    pos: np.ndarray, direction: np.ndarray, speed: np.ndarray, homing: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """Advance classic homing balls by ``dt`` (distance-based steering, constant speed).

    Works on ``(n, 3)`` arrays or a single ``(3,)`` ball.
    """
    to_t = TORSO - pos
    dist = np.linalg.norm(to_t, axis=-1, keepdims=True)
    desired = to_t / np.maximum(dist, 1e-6)
    speed_arr = np.asarray(speed)[..., None]
    k = np.minimum(np.asarray(homing)[..., None] * speed_arr * dt / np.maximum(dist, 2.0), 1.0)
    new_dir = direction + k * (desired - direction)
    norm = np.linalg.norm(new_dir, axis=-1, keepdims=True)
    new_dir = np.where(norm > 1e-6, new_dir / np.maximum(norm, 1e-12), desired)
    return pos + new_dir * (speed_arr * dt), new_dir


def _dir_from_angles(yaw: np.ndarray, elev: np.ndarray) -> np.ndarray:
    return np.stack([np.cos(elev) * np.sin(yaw), np.sin(elev), np.cos(elev) * np.cos(yaw)], axis=-1)


def _log_uniform(rng: np.random.Generator, lo: float, hi: float, n: int) -> np.ndarray:
    return np.exp(rng.uniform(math.log(lo), math.log(hi), n))


def _mix(p: Sequence[float]) -> np.ndarray:
    a = np.asarray(p, dtype=np.float64)
    return a / a.sum()


def sample_launch(
    rng: np.random.Generator,
    start: np.ndarray,
    speed: np.ndarray,
    curve_mix: Sequence[float] = TRAINING_CURVE_MIX,
    law_mix: Sequence[float] = LAW_MIX,
    accel_prob: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, Flight]:
    """Launch direction and steering for balls deflected at ``start`` towards you.

    Returns ``(curve, direction, flight)`` where ``curve`` indexes ``CURVE_NAMES``.
    """
    start = np.atleast_2d(start).astype(np.float64)
    speed = np.atleast_1d(speed).astype(np.float64)
    n = start.shape[0]
    to_t = TORSO[None, :] - start
    dist = np.linalg.norm(to_t, axis=1)
    base_yaw = np.arctan2(to_t[:, 0], to_t[:, 2])
    base_elev = np.arctan2(to_t[:, 1], np.hypot(to_t[:, 0], to_t[:, 2]))
    curve = rng.choice(len(CURVE_NAMES), size=n, p=_mix(curve_mix))
    side = np.where(rng.random(n) < 0.5, -1.0, 1.0)
    rad = np.radians
    yaw = base_yaw + rng.normal(0.0, rad(10.0), n)
    elev = base_elev + rng.normal(rad(2.0), rad(5.0), n)
    c = curve == 1  # out to the side
    yaw[c] = base_yaw[c] + side[c] * rng.uniform(rad(35), rad(110), c.sum())
    elev[c] = base_elev[c] + rng.uniform(rad(-5), rad(25), c.sum())
    c = curve == 2  # high
    yaw[c] = base_yaw[c] + rng.normal(0.0, rad(25), c.sum())
    elev[c] = rng.uniform(rad(35), rad(80), c.sum())
    c = curve == 3  # backwards (away from you first)
    yaw[c] = base_yaw[c] + side[c] * rng.uniform(rad(110), rad(180), c.sum())
    elev[c] = base_elev[c] + rng.uniform(rad(-5), rad(45), c.sum())
    c = curve == 4  # anything
    yaw[c] = rng.uniform(-math.pi, math.pi, c.sum())
    elev[c] = rng.uniform(rad(-20), rad(85), c.sum())
    direction = _dir_from_angles(yaw, elev)

    law = rng.choice(len(LAW_NAMES), size=n, p=_mix(law_mix))
    gain = np.empty(n)
    gain2 = np.zeros(n)
    m = law == 0
    gain[m] = rng.uniform(1.5, 4.5, m.sum())
    m = law == 1
    # turn radius at most ~1.25x the distance so every ball comes back to you
    w_min = 0.8 * speed[m] / np.maximum(dist[m], 1.0)
    gain[m] = np.maximum(_log_uniform(rng, 1.5, 10.0, int(m.sum())), w_min)
    gain2[m] = rng.uniform(0.3, 3.0, m.sum())
    m = law == 2
    gain[m] = dist[m] / np.maximum(speed[m], 1.0) * rng.uniform(0.35, 1.3, m.sum())
    gain2[m] = rng.uniform(0.7, 2.0, m.sum())
    accel = np.where(rng.random(n) < accel_prob, rng.uniform(0.02, 0.4, n), 0.0)
    flight = Flight(law, gain, gain2, direction.copy(), accel, np.zeros(n), speed * 3.0)
    return curve, direction, flight


# ---------------------------------------------------------------- scenarios
@dataclass
class Approaches:
    """A batch of red-ball approaches (all arrays have length ``n``).

    Each approach starts at the moment the ball is deflected towards you (the
    moment your character turns red) and ends when it reaches you.
    """

    start: np.ndarray  # (n, 3) spawn position (relative to you)
    direction: np.ndarray  # (n, 3) initial unit direction
    speed: np.ndarray  # (n,) studs / s at launch
    flight: Flight
    curve: np.ndarray  # (n,) index into CURVE_NAMES
    radius: np.ndarray  # (n,) ball radius in studs
    cam_yaw: np.ndarray
    cam_pitch: np.ndarray
    cam_distance: np.ndarray
    cam_fov: np.ndarray
    cam_shift: np.ndarray
    crop_w: np.ndarray
    crop_h: np.ndarray
    # you walking around: velocity until ``move_switch`` seconds, then ``move_vel2``
    move_vel: np.ndarray = None  # type: ignore[assignment]
    move_vel2: np.ndarray = None  # type: ignore[assignment]
    move_switch: np.ndarray = None  # type: ignore[assignment]
    # camera turned during the flight: yaw(t) = yaw + amp*(sin(2 pi f t + ph) - sin(ph)) + drift*t
    cam_yaw_amp: np.ndarray = None  # type: ignore[assignment]
    cam_yaw_freq: np.ndarray = None  # type: ignore[assignment]
    cam_yaw_phase: np.ndarray = None  # type: ignore[assignment]
    cam_yaw_drift: np.ndarray = None  # type: ignore[assignment]
    cam_pitch_amp: np.ndarray = None  # type: ignore[assignment]
    # rally rhythm before this ball (inf = unknown): time the ball spent away
    # from you before coming back, and how long the previous ball took to reach you
    prev_gap: np.ndarray = None  # type: ignore[assignment]
    prev_dur: np.ndarray = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        n = self.n
        zeros = np.zeros(n)
        if self.move_vel is None:
            self.move_vel = np.zeros((n, 3))
        if self.move_vel2 is None:
            self.move_vel2 = self.move_vel.copy()
        if self.move_switch is None:
            self.move_switch = np.full(n, np.inf)
        for name in ("cam_yaw_amp", "cam_yaw_freq", "cam_yaw_phase", "cam_yaw_drift", "cam_pitch_amp"):
            if getattr(self, name) is None:
                setattr(self, name, zeros.copy())
        if self.prev_gap is None:
            self.prev_gap = np.full(n, np.inf)
        if self.prev_dur is None:
            self.prev_dur = np.full(n, np.inf)

    @property
    def n(self) -> int:
        return int(self.speed.shape[0])

    @property
    def homing(self) -> np.ndarray:
        """Steering gain (kept for backwards compatibility)."""
        return self.flight.gain

    def camera(self, i: int, width: int = 1280, height: int = 720) -> Camera:
        return Camera(
            yaw=float(self.cam_yaw[i]),
            pitch=float(self.cam_pitch[i]),
            distance=float(self.cam_distance[i]),
            fov=float(self.cam_fov[i]),
            shift=float(self.cam_shift[i]),
            width=width,
            height=height,
            crop_w=float(self.crop_w[i]),
            crop_h=float(self.crop_h[i]),
        )

    def camera_angles(self, i: int, times: np.ndarray) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Camera ``(yaw, pitch)`` at each frame time, or ``None`` if it doesn't move."""
        amp, drift, pamp = float(self.cam_yaw_amp[i]), float(self.cam_yaw_drift[i]), float(self.cam_pitch_amp[i])
        if amp == 0.0 and drift == 0.0 and pamp == 0.0:
            return None
        f, ph = float(self.cam_yaw_freq[i]), float(self.cam_yaw_phase[i])
        w = 2 * math.pi * f
        yaw = self.cam_yaw[i] + amp * (np.sin(w * times + ph) - math.sin(ph)) + drift * times
        pitch = self.cam_pitch[i] + pamp * (np.sin(0.7 * w * times + 2 * ph) - math.sin(2 * ph))
        return yaw, np.clip(pitch, math.radians(1.0), math.radians(70.0))

    def char_velocity(self, idx: np.ndarray, t: float) -> np.ndarray:
        late = (t >= self.move_switch[idx])[:, None]
        return np.where(late, self.move_vel2[idx], self.move_vel[idx])


def sample_approaches(
    rng: np.random.Generator,
    n: int,
    speed_range: tuple[float, float] = (25.0, 450.0),
    distance_range: tuple[float, float] = (12.0, 100.0),
    in_view_prob: float = 0.85,
    cam_distance_range: tuple[float, float] = (6.0, 45.0),
    cam_pitch_deg: tuple[float, float] = (2.0, 50.0),
    curve_mix: Sequence[float] = TRAINING_CURVE_MIX,
    moving_prob: float = 0.4,
    cam_motion_prob: float = 0.5,
    rally_mix: Sequence[float] = RALLY_MIX,
    serve_prob: float = 0.05,
) -> Approaches:
    """Randomised approaches covering many speeds, curves, camera setups and rallies."""
    cam_yaw = rng.uniform(0.0, 2 * math.pi, n)
    in_view = rng.random(n) < in_view_prob
    rel_az = np.where(in_view, rng.normal(0.0, math.radians(28), n), rng.uniform(-math.pi, math.pi, n))
    az = cam_yaw + rel_az
    speed = _log_uniform(rng, *speed_range, n)
    # slow balls start closer, so every approach arrives within a few seconds
    dist = np.minimum(rng.uniform(*distance_range, n), np.maximum(distance_range[0], speed * rng.uniform(1.5, 4.0, n)))
    # the ball leaves the other player at about body height (sometimes higher)
    low = rng.random(n) < 0.6
    height = np.clip(TORSO[1] + np.where(low, rng.uniform(-1.0, 3.0, n), rng.uniform(-1.0, 9.0, n)), 1.0, None)
    start = np.stack([dist * np.sin(az), height, dist * np.cos(az)], axis=1)

    curve, direction, flight = sample_launch(rng, start, speed, curve_mix)
    # the first ball of a round starts slow and speeds up
    serve = rng.random(n) < serve_prob
    if serve.any():
        k = int(serve.sum())
        vmax = speed[serve]
        speed[serve] = rng.uniform(5.0, 30.0, k)
        flight.accel_abs[serve] = rng.uniform(10.0, 60.0, k)
        flight.accel[serve] = 0.0
        flight.vmax[serve] = np.maximum(vmax, speed[serve])

    # you walking/dashing around while the ball comes at you
    moving = rng.random(n) < moving_prob

    def _walk() -> np.ndarray:
        heading = rng.uniform(0, 2 * math.pi, n)
        v = rng.uniform(4.0, 22.0, n) * moving
        return np.stack([v * np.sin(heading), np.zeros(n), v * np.cos(heading)], axis=1)

    move_vel = _walk()
    move_vel2 = np.where(rng.random(n)[:, None] < 0.5, _walk(), move_vel)
    move_switch = rng.uniform(0.2, 1.2, n)

    # the player turning the camera while the ball is in the air
    turning = rng.random(n) < cam_motion_prob
    cam_yaw_amp = np.radians(rng.uniform(0.0, 25.0, n)) * turning
    cam_yaw_freq = rng.uniform(0.3, 2.0, n)
    cam_yaw_phase = rng.uniform(0.0, 2 * math.pi, n)
    cam_yaw_drift = np.radians(rng.uniform(-60.0, 60.0, n)) * turning * (rng.random(n) < 0.5)
    cam_pitch_amp = np.radians(rng.uniform(0.0, 8.0, n)) * turning

    # rally rhythm
    ctx = rng.choice(3, size=n, p=_mix(rally_mix))
    ctx[serve] = 0
    prev_gap = np.full(n, np.inf)
    prev_dur = np.full(n, np.inf)
    duel = ctx == 1
    if duel.any():
        k = int(duel.sum())
        v0 = np.maximum(speed[duel], 5.0)
        v_out = v0 / rng.uniform(1.0, 1.15, k)  # the other player's deflect sped it up
        v_prev = v_out / rng.uniform(1.0, 1.15, k)  # ... and so did yours
        d = dist[duel]
        prev_gap[duel] = np.maximum(
            d * rng.uniform(0.85, 1.15, k) * rng.uniform(1.0, 1.35, k) / v_out + rng.uniform(-0.04, 0.1, k), 0.02
        )
        prev_dur[duel] = d * rng.uniform(0.8, 1.25, k) * rng.uniform(1.0, 1.9, k) / v_prev + rng.uniform(-0.02, 0.05, k)
    ffa = ctx == 2
    if ffa.any():
        k = int(ffa.sum())
        prev_gap[ffa] = rng.uniform(0.3, 6.0, k)
        prev_dur[ffa] = rng.uniform(0.15, 4.0, k)

    return Approaches(
        start=start,
        direction=direction,
        speed=speed,
        flight=flight,
        curve=curve,
        radius=rng.uniform(0.8, 1.6, n),
        cam_yaw=cam_yaw,
        cam_pitch=np.radians(rng.uniform(*cam_pitch_deg, n)),
        cam_distance=_log_uniform(rng, *cam_distance_range, n),
        cam_fov=np.radians(rng.uniform(62.0, 78.0, n)),
        cam_shift=np.where(rng.random(n) < 0.5, 1.75, 0.0),
        crop_w=rng.uniform(0.5, 1.0, n),
        crop_h=rng.uniform(0.55, 1.0, n),
        move_vel=move_vel,
        move_vel2=move_vel2,
        move_switch=move_switch,
        cam_yaw_amp=cam_yaw_amp,
        cam_yaw_freq=cam_yaw_freq,
        cam_yaw_phase=cam_yaw_phase,
        cam_yaw_drift=cam_yaw_drift,
        cam_pitch_amp=cam_pitch_amp,
        prev_gap=prev_gap,
        prev_dur=prev_dur,
    )


def simulate_trajectories(
    ap: Approaches, dt: float = 0.004, t_max: float = 5.0
) -> tuple[np.ndarray, np.ndarray]:
    """Fly every ball of the batch until it hits you (or ``t_max``).

    Returns ``(positions, t_impact)`` where ``positions`` has shape
    ``(steps + 1, n, 3)`` sampled every ``dt`` seconds (frozen after impact),
    relative to your (possibly moving) character, and ``t_impact`` is the
    exact impact time (``inf`` if it never arrived).
    """
    n = ap.n
    steps = int(math.ceil(t_max / dt))
    positions = np.empty((steps + 1, n, 3), dtype=np.float32)
    pos = ap.start.astype(np.float64).copy()
    direction = ap.direction.astype(np.float64).copy()
    speed = ap.speed.astype(np.float64).copy()
    hit_r = ap.radius + HIT_RADIUS
    t_impact = np.full(n, np.inf)
    alive = np.ones(n, dtype=bool)
    positions[0] = pos
    prev_dist = np.linalg.norm(pos - TORSO, axis=1)
    instant = prev_dist <= hit_r  # balls spawned inside the hit radius hit immediately
    t_impact[instant] = 0.0
    alive &= ~instant
    moving = bool(np.any(ap.move_vel != 0) or np.any(ap.move_vel2 != 0))
    idx = np.nonzero(alive)[0]
    fl = ap.flight.subset(idx)
    for s in range(1, steps + 1):
        if idx.size == 0:
            positions[s:] = positions[s - 1]
            break
        t = (s - 1) * dt
        cv = ap.char_velocity(idx, t) if moving else None
        new_pos, new_dir, new_speed = flight_step(pos[idx], direction[idx], speed[idx], t, fl, dt, cv)
        pos[idx] = new_pos
        direction[idx] = new_dir
        speed[idx] = new_speed
        dist = np.linalg.norm(new_pos - TORSO, axis=1)
        hit = dist <= hit_r[idx]
        if hit.any():
            hi = idx[hit]
            d0 = prev_dist[hi]
            d1 = dist[hit]
            frac = np.clip((d0 - hit_r[hi]) / np.maximum(d0 - d1, 1e-9), 0.0, 1.0)
            t_impact[hi] = (s - 1 + frac) * dt
            keep = ~hit
            prev_dist[idx[keep]] = dist[keep]
            idx = idx[keep]
            fl = fl.subset(np.nonzero(keep)[0])
        else:
            prev_dist[idx] = dist
        positions[s] = pos
    return positions, t_impact


def interpolate_positions(positions: np.ndarray, i: int, times: np.ndarray, dt: float) -> np.ndarray:
    """Positions of ball ``i`` at arbitrary ``times`` (linear interpolation)."""
    f = np.clip(times / dt, 0.0, positions.shape[0] - 1.000001)
    k = np.floor(f).astype(np.int64)
    w = (f - k)[:, None]
    return positions[k, i] * (1.0 - w) + positions[k + 1, i] * w


def frame_times(rng: np.random.Generator, fps: float, t_end: float, jitter: float = 0.2) -> np.ndarray:
    """Irregular frame timestamps like a real capture loop produces."""
    n_max = int(t_end * fps * 1.5) + 4
    steps = (1.0 / fps) * (1.0 + rng.uniform(-jitter, jitter, n_max))
    t = rng.uniform(0.0, 1.0 / fps) + np.concatenate([[0.0], np.cumsum(steps[:-1])])
    return t[t < t_end]


def detections_for_frames(
    rng: np.random.Generator,
    cam: Camera,
    pos: np.ndarray,
    ball_radius: float,
    proc_scale: int = 2,
    dropout: float = 0.03,
    angles: Optional[tuple[np.ndarray, np.ndarray]] = None,
) -> tuple[list[tuple[float, float, float] | None], float]:
    """Simulate what the vision module would report for each ball position.

    Returns ``(detections, char_h)`` where ``char_h`` is the height of the
    simulated character box divided by the capture height.

    Includes pixel noise, quantisation to the processed-pixel grid, random
    missed frames, balls outside the capture region, balls too small to see, a
    glow that makes the ball look bigger than it is, and the character box that
    the detector cuts out (a ball whose centre is inside it is not seen; one
    overlapping its edge is measured less precisely). ``angles`` (per-frame
    camera yaw and pitch) simulates the player turning the camera; the
    calibrated character position stays where it was.
    """
    if angles is None:
        u, v, z = cam.project(pos)
    else:
        u, v, z = cam.project_moving(pos, angles[0], angles[1])
    n = len(u)
    glow = rng.uniform(0.9, 1.5)  # the red aura makes the blob bigger than the ball
    r_px = cam.focal * ball_radius * glow / np.maximum(z, 1e-6)
    x0, y0, rw, rh = cam.region
    au, av = cam.anchor()
    cu, cv, chw, chh = cam.character_box()
    zw = chw * rng.uniform(0.85, 1.25)  # the user's character box (cut out by the detector)
    zh = chh * rng.uniform(0.85, 1.2)
    s = float(proc_scale)
    r_proc = r_px / s
    in_view = (z > 0.5) & (u >= x0) & (u <= x0 + rw) & (v >= y0) & (v <= y0 + rh) & (r_proc >= 1.5)
    hidden = (np.abs(u - cu) < zw) & (np.abs(v - cv) < zh)
    clipped = (np.abs(u - cu) < zw + r_px) & (np.abs(v - cv) < zh + r_px)
    clipped |= (u - r_px < x0) | (u + r_px > x0 + rw) | (v - r_px < y0) | (v + r_px > y0 + rh)
    dropped = rng.random(n) < dropout
    ok = in_view & ~hidden & ~dropped
    noise_scale = np.where(clipped, 2.5, 1.0)
    sig_pos = (0.35 + 0.03 * r_proc) * s * noise_scale
    sig_r = (0.30 + 0.05 * r_proc) * s * noise_scale
    un = np.round((u + rng.normal(0, 1, n) * sig_pos) / s) * s
    vn = np.round((v + rng.normal(0, 1, n) * sig_pos) / s) * s
    rn = np.maximum(r_px + rng.normal(0, 1, n) * sig_r, 0.5 * s)
    out: list[tuple[float, float, float] | None] = []
    for k in range(n):
        if ok[k]:
            out.append(((un[k] - au) / rh, (vn[k] - av) / rh, rn[k] / rh))
        else:
            out.append(None)
    return out, float(2.0 * zh / rh)
