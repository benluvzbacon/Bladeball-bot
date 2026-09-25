"""Computer vision: find the red ball on screen and check if *you* are targeted.

In Blade Ball the ball turns red (with a red glow) while it is targeting you,
and your own character gets a red highlight at the same time. This module:

1. builds a mask of "ball red" pixels with an HSV colour threshold,
2. groups the mask into blobs (fast run-length connected components, NumPy only),
3. picks the most ball-like blob (round, solid, not your own character), and
4. measures how much of the area around your character is red (the *gate*).

Everything is expressed relative to your character (the *anchor*) and divided
by the frame height, which is exactly what :class:`bladebot.features.TrackState`
expects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from typing import Any, Optional

import numpy as np


@dataclass
class VisionSettings:
    hue_center: float = 355.0  # degrees; Blade Ball's targeting red is ~345-5
    hue_tol: float = 20.0
    sat_min: float = 0.40
    val_min: float = 0.45
    min_area: int = 5  # processed pixels
    max_aspect: float = 1.8
    min_fill: float = 0.55
    max_fill: float = 0.92  # solid rectangles (UI, body parts) are not the ball
    dilate: int = 1  # close 1-px gaps (the ball's glow ring)
    anchor_x: float = 0.50  # your character, as a fraction of the capture width
    anchor_y: float = 0.64  # ... and height
    gate_enabled: bool = True
    gate_w: float = 0.14  # box around your whole character (fraction of the capture height)
    gate_h: float = 0.30
    gate_min_frac: float = 0.04  # this much red inside the box = "I'm targeted"

    @classmethod
    def from_mapping(cls, cfg: dict[str, Any]) -> "VisionSettings":
        kwargs = {}
        for f in fields(cls):
            if f.name in cfg:
                kwargs[f.name] = type(f.default)(cfg[f.name])
        return cls(**kwargs)


@dataclass
class Blob:
    cx: float
    cy: float
    area: float
    x0: int
    y0: int
    x1: int  # inclusive
    y1: int  # inclusive

    @property
    def w(self) -> int:
        return self.x1 - self.x0 + 1

    @property
    def h(self) -> int:
        return self.y1 - self.y0 + 1

    @property
    def aspect(self) -> float:
        return max(self.w, self.h) / max(min(self.w, self.h), 1)

    @property
    def fill(self) -> float:
        return self.area / float(self.w * self.h)


@dataclass
class Detection:
    ball: Optional[tuple[float, float, float]]  # anchor-relative, / frame height
    ball_px: Optional[tuple[float, float, float]]  # processed-frame pixels (x, y, r)
    targeted: bool
    gate_frac: float
    red_frac: float
    blobs: list[Blob]
    shape: tuple[int, int]
    anchor_px: tuple[float, float]
    gate_box: tuple[int, int, int, int]
    mask: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def usable_ball(self) -> Optional[tuple[float, float, float]]:
        return self.ball


# ---------------------------------------------------------------- colours
def hsv_match(r: np.ndarray, g: np.ndarray, b: np.ndarray, s: VisionSettings) -> np.ndarray:
    """Exact HSV threshold test for arrays of 0-255 channel values."""
    r = np.asarray(r, dtype=np.float32)
    g = np.asarray(g, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    delta = mx - mn
    safe = np.where(delta > 0, delta, 1.0)
    hue = np.where(
        mx == r, (g - b) / safe, np.where(mx == g, 2.0 + (b - r) / safe, 4.0 + (r - g) / safe)
    )
    hue = (hue * 60.0) % 360.0
    dist = np.abs(((hue - s.hue_center + 180.0) % 360.0) - 180.0)
    return (
        (delta > 0)
        & (mx >= s.val_min * 255.0)
        & (delta >= s.sat_min * mx)
        & (dist <= s.hue_tol)
    )


_LUT_CACHE: dict[tuple[float, float, float, float], np.ndarray] = {}


def color_lut(s: VisionSettings) -> np.ndarray:
    """64x64x64 lookup table (6 bits per channel) of the colour threshold."""
    key = (float(s.hue_center), float(s.hue_tol), float(s.sat_min), float(s.val_min))
    lut = _LUT_CACHE.get(key)
    if lut is None:
        levels = np.arange(64, dtype=np.float32) * 4.0 + 1.5  # bin centres
        r, g, b = np.meshgrid(levels, levels, levels, indexing="ij")
        lut = hsv_match(r, g, b, s).ravel()
        if len(_LUT_CACHE) > 16:
            _LUT_CACHE.clear()
        _LUT_CACHE[key] = lut
    return lut


def red_mask(rgb: np.ndarray, s: VisionSettings) -> np.ndarray:
    """Boolean mask of pixels whose colour matches the ball colour settings.

    Uses a precomputed lookup table so a whole frame costs a few milliseconds
    even when most of the scene is a saturated colour (like Blade Ball's sand).
    """
    lut = color_lut(s)
    idx = (rgb[..., 0] >> 2).astype(np.int32) << 12
    idx |= (rgb[..., 1] >> 2).astype(np.int32) << 6
    idx |= rgb[..., 2] >> 2
    return lut[idx]


def rgb_to_hsv_pixel(r: int, g: int, b: int) -> tuple[float, float, float]:
    """HSV of one pixel: hue in degrees, saturation and value in 0..1."""
    mx, mn = max(r, g, b), min(r, g, b)
    d = mx - mn
    if d == 0:
        h = 0.0
    elif mx == r:
        h = (60.0 * (g - b) / d) % 360.0
    elif mx == g:
        h = 60.0 * (2.0 + (b - r) / d)
    else:
        h = 60.0 * (4.0 + (r - g) / d)
    return h, (d / mx if mx else 0.0), mx / 255.0


def dilate(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """3x3 binary dilation (separable, NumPy only)."""
    m = mask
    for _ in range(max(iterations, 0)):
        h = m.copy()
        h[:, 1:] |= m[:, :-1]
        h[:, :-1] |= m[:, 1:]
        v = h.copy()
        v[1:, :] |= h[:-1, :]
        v[:-1, :] |= h[1:, :]
        m = v
    return m


def _or_pool2(mask: np.ndarray) -> np.ndarray:
    h, w = (mask.shape[0] // 2) * 2, (mask.shape[1] // 2) * 2
    m = mask[:h, :w]
    return m[0::2, 0::2] | m[1::2, 0::2] | m[0::2, 1::2] | m[1::2, 1::2]


# ---------------------------------------------------------------- blobs
def connected_components(mask: np.ndarray, max_runs: int = 8000) -> list[Blob]:
    """8-connected components via run-length encoding + union-find.

    Only the runs of set pixels are visited in Python, so it stays fast for
    the sparse masks we get (a ball and a few highlights). Very noisy masks
    are OR-pooled down until they are manageable.
    """
    scale = 1
    while True:
        h, w = mask.shape
        row_idx = np.flatnonzero(mask.any(axis=1))
        if row_idx.size == 0:
            return []
        padded = np.zeros((row_idx.size, w + 2), dtype=np.int8)
        padded[:, 1:-1] = mask[row_idx]
        d = np.diff(padded, axis=1)
        sr_local, sc = np.nonzero(d == 1)
        _, ec = np.nonzero(d == -1)
        sr = row_idx[sr_local]
        if len(sr) <= max_runs or min(h, w) < 8:
            break
        mask = _or_pool2(mask)
        scale *= 2
    n = len(sr)
    if n == 0:
        return []
    rows = sr.tolist()
    starts = sc.tolist()
    ends = ec.tolist()  # exclusive
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    # index ranges of runs per row (rows are sorted because nonzero is row-major)
    prev_row, p0, p1 = -10, 0, 0
    i = 0
    while i < n:
        row = rows[i]
        j = i
        while j < n and rows[j] == row:
            j += 1
        if row == prev_row + 1:
            a, b = p0, i
            while a < p1 and b < j:
                if starts[b] <= ends[a] and starts[a] <= ends[b]:
                    ra, rb = find(a), find(b)
                    if ra != rb:
                        parent[ra] = rb
                if ends[a] < ends[b]:
                    a += 1
                else:
                    b += 1
        prev_row, p0, p1 = row, i, j
        i = j

    roots = np.fromiter((find(k) for k in range(n)), dtype=np.int64, count=n)
    _, lab = np.unique(roots, return_inverse=True)
    k = int(lab.max()) + 1
    sr_f = sr.astype(np.float64)
    sc_f = sc.astype(np.float64)
    ec_f = ec.astype(np.float64)
    length = ec_f - sc_f
    area = np.bincount(lab, weights=length, minlength=k)
    sum_x = np.bincount(lab, weights=(sc_f + ec_f - 1.0) * length / 2.0, minlength=k)
    sum_y = np.bincount(lab, weights=sr_f * length, minlength=k)
    x0 = np.full(k, np.iinfo(np.int64).max)
    x1 = np.full(k, -1)
    y0 = np.full(k, np.iinfo(np.int64).max)
    y1 = np.full(k, -1)
    np.minimum.at(x0, lab, sc)
    np.maximum.at(x1, lab, ec - 1)
    np.minimum.at(y0, lab, sr)
    np.maximum.at(y1, lab, sr)
    blobs = []
    for q in range(k):
        cx = (sum_x[q] / area[q] + 0.5) * scale - 0.5
        cy = (sum_y[q] / area[q] + 0.5) * scale - 0.5
        blobs.append(
            Blob(
                cx=float(cx),
                cy=float(cy),
                area=float(area[q]) * scale * scale,
                x0=int(x0[q]) * scale,
                y0=int(y0[q]) * scale,
                x1=int(x1[q]) * scale + scale - 1,
                y1=int(y1[q]) * scale + scale - 1,
            )
        )
    return blobs


# ---------------------------------------------------------------- detector
class BallDetector:
    """Stateful detector (remembers the last ball to keep tracking the same one)."""

    def __init__(self, settings: Optional[VisionSettings] = None) -> None:
        self.settings = settings or VisionSettings()
        self._prev: Optional[tuple[float, float, float]] = None
        self._misses = 0

    def reset(self) -> None:
        self._prev = None
        self._misses = 0

    def geometry(self, shape: tuple[int, int]) -> tuple[float, float, tuple[int, int, int, int]]:
        h, w = shape
        s = self.settings
        ax, ay = s.anchor_x * w, s.anchor_y * h
        hw, hh = s.gate_w * h / 2.0, s.gate_h * h / 2.0
        box = (
            max(int(ax - hw), 0),
            max(int(ay - hh), 0),
            min(int(math.ceil(ax + hw)), w - 1),
            min(int(math.ceil(ay + hh)), h - 1),
        )
        return ax, ay, box

    def detect(self, rgb: np.ndarray, keep_mask: bool = False) -> Detection:
        s = self.settings
        h, w = rgb.shape[:2]
        mask = red_mask(rgb, s)
        ax, ay, box = self.geometry((h, w))
        gx0, gy0, gx1, gy1 = box
        gate_region = mask[gy0 : gy1 + 1, gx0 : gx1 + 1]
        gate_frac = float(gate_region.mean()) if gate_region.size else 0.0
        targeted = gate_frac >= s.gate_min_frac
        red_frac = float(mask.mean())

        # Your own character sits inside the gate box and turns red too, so the
        # box is cut out before looking for the ball. A ball overlapping the box
        # edge is still found; its size is estimated from the visible part.
        ball_mask = mask.copy()
        ball_mask[gy0 : gy1 + 1, gx0 : gx1 + 1] = False
        merged = dilate(ball_mask, s.dilate) if s.dilate > 0 else ball_mask
        blobs = connected_components(merged)
        ball_px = self._pick(blobs, ax, ay, box, (h, w))
        ball = None
        if ball_px is not None:
            bx, by, br = ball_px
            ball = ((bx - ax) / h, (by - ay) / h, br / h)
            self._prev = ball_px
            self._misses = 0
        else:
            self._misses += 1
            if self._misses > 8:
                self._prev = None
        return Detection(
            ball=ball,
            ball_px=ball_px,
            targeted=targeted,
            gate_frac=gate_frac,
            red_frac=red_frac,
            blobs=blobs[:64],
            shape=(h, w),
            anchor_px=(ax, ay),
            gate_box=box,
            mask=mask if keep_mask else None,
        )

    def _pick(
        self,
        blobs: list[Blob],
        ax: float,
        ay: float,
        box: tuple[int, int, int, int],
        shape: tuple[int, int],
    ) -> Optional[tuple[float, float, float]]:
        s = self.settings
        frame_h, frame_w = shape
        gx0, gy0, gx1, gy1 = box
        m = s.dilate + 1  # blobs this close to a cut edge count as clipped
        best = None
        best_score = -math.inf
        for b in blobs:
            if b.area < s.min_area:
                continue
            # which sides of the blob are cut off by the frame or the character box?
            overlap_x = b.x1 >= gx0 - m and b.x0 <= gx1 + m
            overlap_y = b.y1 >= gy0 - m and b.y0 <= gy1 + m
            cut_bottom = b.y1 >= frame_h - 1 - m or (overlap_x and gy0 - m <= b.y1 + 1 <= gy0 + m)
            cut_top = b.y0 <= m or (overlap_x and gy1 - m <= b.y0 - 1 <= gy1 + m)
            cut_right = b.x1 >= frame_w - 1 - m or (overlap_y and gx0 - m <= b.x1 + 1 <= gx0 + m)
            cut_left = b.x0 <= m or (overlap_y and gx1 - m <= b.x0 - 1 <= gx1 + m)
            clipped = cut_bottom or cut_top or cut_right or cut_left
            if clipped and (b.area < max(s.min_area, 12) or max(b.w, b.h) < 2 * s.dilate + 3):
                continue  # a sliver of something disappearing behind the box
            max_aspect = max(s.max_aspect, 2.2) if clipped else s.max_aspect
            if b.aspect > max_aspect or b.fill < s.min_fill:
                continue
            if b.area >= 40 and b.fill > s.max_fill:
                continue
            roundness = max(0.0, 1.0 - abs(b.fill - math.pi / 4) * 2.0)
            score = math.log(b.area) + 2.0 * roundness - 1.5 * (min(b.aspect, 2.0) - 1.0) - (0.5 if clipped else 0.0)
            if self._prev is not None:
                px, py, pr = self._prev
                dist = math.hypot(b.cx - px, b.cy - py)
                score += 2.0 * math.exp(-dist / (3.0 * pr + 0.05 * frame_h))
            if score > best_score:
                best_score = score
                best = (b, cut_top, cut_bottom, cut_left, cut_right, clipped)
        if best is None:
            return None
        b, cut_top, cut_bottom, cut_left, cut_right, clipped = best
        d = s.dilate
        if not clipped:
            radius = max((b.w + b.h) / 4.0 - d, 0.75)
            return b.cx, b.cy, radius
        # partially hidden ball: the widest visible extent is still the diameter
        radius = max(max(b.w, b.h) / 2.0 - d, 0.75)
        cx = (b.x0 + b.x1) / 2.0
        cy = (b.y0 + b.y1) / 2.0
        if cut_bottom and not cut_top:
            cy = b.y0 + d + radius
        elif cut_top and not cut_bottom:
            cy = b.y1 - d - radius
        if cut_right and not cut_left:
            cx = b.x0 + d + radius
        elif cut_left and not cut_right:
            cx = b.x1 - d - radius
        return cx, cy, radius


# ---------------------------------------------------------------- preview
def _put(img: np.ndarray, xs: np.ndarray, ys: np.ndarray, color: tuple[int, int, int]) -> None:
    h, w = img.shape[:2]
    xs = np.round(xs).astype(np.int64)
    ys = np.round(ys).astype(np.int64)
    ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    img[ys[ok], xs[ok]] = color


def draw_rect(img: np.ndarray, x0: float, y0: float, x1: float, y1: float, color: tuple[int, int, int]) -> None:
    xs = np.arange(int(x0), int(x1) + 1)
    ys = np.arange(int(y0), int(y1) + 1)
    _put(img, xs, np.full(xs.shape, y0), color)
    _put(img, xs, np.full(xs.shape, y1), color)
    _put(img, np.full(ys.shape, x0), ys, color)
    _put(img, np.full(ys.shape, x1), ys, color)


def draw_circle(img: np.ndarray, cx: float, cy: float, r: float, color: tuple[int, int, int], thick: int = 2) -> None:
    n = max(int(2 * math.pi * (r + thick) * 1.5), 16)
    ang = np.linspace(0, 2 * math.pi, n, endpoint=False)
    for k in range(thick):
        rr = r + k
        _put(img, cx + rr * np.cos(ang), cy + rr * np.sin(ang), color)


def draw_cross(img: np.ndarray, cx: float, cy: float, size: float, color: tuple[int, int, int]) -> None:
    t = np.arange(-size, size + 1)
    _put(img, cx + t, np.full(t.shape, cy), color)
    _put(img, np.full(t.shape, cx), cy + t, color)


def render_preview(
    rgb: np.ndarray, det: Detection, max_width: int = 520, active: Optional[bool] = None
) -> np.ndarray:
    """Downscaled copy of the processed frame with the detector's view drawn on top.

    The ball is circled green when the bot is paying attention to it (you are
    targeted) and grey when it is ignored. ``active`` overrides ``det.targeted``.
    """
    if active is None:
        active = det.targeted
    h, w = rgb.shape[:2]
    k = max(1, int(math.ceil(w / max_width)))
    img = np.ascontiguousarray(rgb[::k, ::k, :3]).copy()
    if det.mask is not None:
        m = det.mask[::k, ::k][: img.shape[0], : img.shape[1]]
        img[m] = (img[m] * 0.25 + np.array([255, 0, 255]) * 0.75).astype(np.uint8)
    gx0, gy0, gx1, gy1 = det.gate_box
    gate_color = (255, 60, 60) if det.targeted else (255, 220, 0)
    draw_rect(img, gx0 / k, gy0 / k, gx1 / k, gy1 / k, gate_color)
    for b in det.blobs[:24]:
        draw_rect(img, b.x0 / k, b.y0 / k, b.x1 / k, b.y1 / k, (255, 150, 0))
    ax, ay = det.anchor_px
    draw_cross(img, ax / k, ay / k, 6, (0, 230, 255))
    if det.ball_px is not None:
        bx, by, br = det.ball_px
        draw_circle(img, bx / k, by / k, max(br / k, 2.0) + 2, (0, 255, 90) if active else (190, 190, 190))
    return img
