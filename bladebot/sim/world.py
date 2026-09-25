"""A small Blade Ball-like 3D world used to train and test the bot offline.

Units are Roblox studs, +Y is up. Your character stands at the origin; the
ball homes in on your torso like the real ball does when it turns red. A
Roblox-style third-person camera (distance / pitch / field of view / optional
shift-lock offset) projects everything onto a virtual screen, which lets us
produce exactly the kind of detections the screen-reading bot sees.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

TORSO = np.array([0.0, 3.0, 0.0])  # centre of your character's hitbox
FOCUS_HEIGHT = 4.5  # the Roblox camera orbits around your head
HIT_RADIUS = 2.0  # impact when |ball - TORSO| <= ball_radius + HIT_RADIUS
CHAR_HALF_WIDTH = 1.4  # studs, including arms
CHAR_HALF_HEIGHT = 2.9
PARRY_WINDOW_S = 0.5  # the block shield lasts 500 ms ...
WHIFF_COOLDOWN_S = 2.0  # ... and missing it costs a 2 s cooldown


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


# ---------------------------------------------------------------- scenarios
@dataclass
class Approaches:
    """A batch of red-ball approaches (all arrays have length ``n``)."""

    start: np.ndarray  # (n, 3) spawn position
    direction: np.ndarray  # (n, 3) initial unit direction
    speed: np.ndarray  # (n,) studs / s
    homing: np.ndarray  # (n,) steering strength towards the target (1 = gentle, 4 = sharp)
    radius: np.ndarray  # (n,) ball radius in studs
    cam_yaw: np.ndarray
    cam_pitch: np.ndarray
    cam_distance: np.ndarray
    cam_fov: np.ndarray
    cam_shift: np.ndarray
    crop_w: np.ndarray
    crop_h: np.ndarray

    @property
    def n(self) -> int:
        return int(self.speed.shape[0])

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


def _log_uniform(rng: np.random.Generator, lo: float, hi: float, n: int) -> np.ndarray:
    return np.exp(rng.uniform(math.log(lo), math.log(hi), n))


def sample_approaches(
    rng: np.random.Generator,
    n: int,
    speed_range: tuple[float, float] = (20.0, 350.0),
    distance_range: tuple[float, float] = (12.0, 110.0),
    in_view_prob: float = 0.85,
    cam_distance_range: tuple[float, float] = (6.0, 45.0),
    cam_pitch_deg: tuple[float, float] = (2.0, 50.0),
) -> Approaches:
    """Randomised approaches covering many speeds, curves and camera setups."""
    cam_yaw = rng.uniform(0.0, 2 * math.pi, n)
    in_view = rng.random(n) < in_view_prob
    rel_az = np.where(
        in_view, rng.normal(0.0, math.radians(28), n), rng.uniform(-math.pi, math.pi, n)
    )
    az = cam_yaw + rel_az
    speed = _log_uniform(rng, *speed_range, n)
    # slow balls start closer, so every approach arrives within a few seconds
    dist = np.minimum(rng.uniform(*distance_range, n), np.maximum(distance_range[0], speed * rng.uniform(1.5, 4.0, n)))
    # many balls come straight from the player you are facing, at about body height
    low = rng.random(n) < 0.55
    height = np.clip(TORSO[1] + np.where(low, rng.uniform(-1.0, 3.0, n), rng.uniform(-1.0, 9.0, n)), 1.0, None)
    start = np.stack([dist * np.sin(az), height, dist * np.cos(az)], axis=1)

    to_target = TORSO[None, :] - start
    heading = np.arctan2(to_target[:, 0], to_target[:, 2]) + rng.uniform(
        -math.radians(60), math.radians(60), n
    )
    horiz = np.hypot(to_target[:, 0], to_target[:, 2])
    elev = np.arctan2(to_target[:, 1], horiz) + rng.uniform(
        -math.radians(15), math.radians(20), n
    )
    direction = np.stack(
        [np.cos(elev) * np.sin(heading), np.sin(elev), np.cos(elev) * np.cos(heading)], axis=1
    )

    return Approaches(
        start=start,
        direction=direction,
        speed=speed,
        homing=rng.uniform(1.0, 4.0, n),
        radius=rng.uniform(0.8, 1.6, n),
        cam_yaw=cam_yaw,
        cam_pitch=np.radians(rng.uniform(*cam_pitch_deg, n)),
        cam_distance=_log_uniform(rng, *cam_distance_range, n),
        cam_fov=np.radians(rng.uniform(62.0, 78.0, n)),
        cam_shift=np.where(rng.random(n) < 0.5, 1.75, 0.0),
        crop_w=rng.uniform(0.5, 1.0, n),
        crop_h=rng.uniform(0.55, 1.0, n),
    )


def homing_step(
    pos: np.ndarray, direction: np.ndarray, speed: np.ndarray, homing: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """Advance homing balls by ``dt``. Works on ``(n, 3)`` arrays (or a single ``(3,)``)."""
    to_t = TORSO - pos
    dist = np.linalg.norm(to_t, axis=-1, keepdims=True)
    desired = to_t / np.maximum(dist, 1e-6)
    # Steer towards the target in proportion to how fast the angle to it changes
    # (distance based, so balls of any speed curve in and always arrive).
    speed_arr = np.asarray(speed)[..., None]
    k = np.minimum(np.asarray(homing)[..., None] * speed_arr * dt / np.maximum(dist, 2.0), 1.0)
    new_dir = direction + k * (desired - direction)
    norm = np.linalg.norm(new_dir, axis=-1, keepdims=True)
    new_dir = np.where(norm > 1e-6, new_dir / np.maximum(norm, 1e-12), desired)
    new_pos = pos + new_dir * (speed_arr * dt)
    return new_pos, new_dir


def simulate_trajectories(
    ap: Approaches, dt: float = 0.004, t_max: float = 5.0
) -> tuple[np.ndarray, np.ndarray]:
    """Fly every ball of the batch until it hits you (or ``t_max``).

    Returns ``(positions, t_impact)`` where ``positions`` has shape
    ``(steps + 1, n, 3)`` sampled every ``dt`` seconds (frozen after impact) and
    ``t_impact`` is the exact impact time (``inf`` if it never arrived).
    """
    n = ap.n
    steps = int(math.ceil(t_max / dt))
    positions = np.empty((steps + 1, n, 3), dtype=np.float32)
    pos = ap.start.astype(np.float64).copy()
    direction = ap.direction.astype(np.float64).copy()
    hit_r = ap.radius + HIT_RADIUS
    t_impact = np.full(n, np.inf)
    alive = np.ones(n, dtype=bool)
    positions[0] = pos
    prev_dist = np.linalg.norm(pos - TORSO, axis=1)
    # balls spawned inside the hit radius hit immediately
    instant = prev_dist <= hit_r
    t_impact[instant] = 0.0
    alive &= ~instant
    last_step = 0
    for s in range(1, steps + 1):
        if not alive.any():
            positions[s:] = positions[s - 1]
            break
        idx = np.nonzero(alive)[0]
        new_pos, new_dir = homing_step(pos[idx], direction[idx], ap.speed[idx], ap.homing[idx], dt)
        pos[idx] = new_pos
        direction[idx] = new_dir
        dist = np.linalg.norm(new_pos - TORSO, axis=1)
        hit = dist <= hit_r[idx]
        if hit.any():
            hi = idx[hit]
            d0 = prev_dist[hi]
            d1 = dist[hit]
            frac = np.clip((d0 - hit_r[hi]) / np.maximum(d0 - d1, 1e-9), 0.0, 1.0)
            t_impact[hi] = (s - 1 + frac) * dt
            alive[hi] = False
        prev_dist[idx] = dist
        positions[s] = pos
        last_step = s
    del last_step
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
) -> tuple[list[tuple[float, float, float] | None], float]:
    """Simulate what the vision module would report for each ball position.

    Returns ``(detections, char_h)`` where ``char_h`` is the height of the
    simulated character box divided by the capture height.

    Includes pixel noise, quantisation to the processed-pixel grid, random
    missed frames, balls outside the capture region, balls too small to see, a
    glow that makes the ball look bigger than it is, and the character box that
    the detector cuts out (a ball whose centre is inside it is not seen; one
    overlapping its edge is measured less precisely).
    """
    u, v, z = cam.project(pos)
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
