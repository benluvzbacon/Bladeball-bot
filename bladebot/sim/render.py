"""Draws simple Blade Ball-looking frames (NumPy only) for the practice arena.

The frames go through the *real* vision pipeline, so the arena exercises the
whole bot: detection, the red-highlight gate, tracking, the network and the
parry decision.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

from .world import CHAR_HALF_HEIGHT, CHAR_HALF_WIDTH, TORSO, Camera

BALL_RED = np.array([214, 52, 74], dtype=np.float32)  # measured from the game's targeting red
BALL_GLOW = np.array([236, 112, 152], dtype=np.float32)
BALL_GRAY = np.array([196, 196, 202], dtype=np.float32)
HIGHLIGHT_RED = np.array([196, 44, 50], dtype=np.float32)


class Renderer:
    def __init__(self, width: int = 640, height: int = 360, seed: int = 3) -> None:
        self.width = width
        self.height = height
        rng = np.random.default_rng(seed)
        self._texture = rng.normal(0.0, 1.0, (height, width)).astype(np.float32)
        # blotchy low-frequency variation for the sand
        small = rng.normal(0.0, 1.0, (height // 24 + 2, width // 24 + 2)).astype(np.float32)
        self._blotches = np.kron(small, np.ones((24, 24), dtype=np.float32))[:height, :width]
        self._bg_key: Optional[tuple[float, ...]] = None
        self._bg: Optional[np.ndarray] = None
        yy, xx = np.mgrid[0:height, 0:width]
        self._yy = yy.astype(np.float32)
        self._xx = xx.astype(np.float32)

    def camera(self, **kwargs: float) -> Camera:
        return Camera(width=self.width, height=self.height, **kwargs)

    # ------------------------------------------------------------ background
    def background(self, cam: Camera) -> np.ndarray:
        key = (round(cam.pitch, 4), round(cam.fov, 4))
        if self._bg is not None and self._bg_key == key:
            return self._bg
        h, w = self.height, self.width
        horizon = h / 2.0 - cam.focal * math.tan(cam.pitch)
        y = self._yy
        img = np.empty((h, w, 3), dtype=np.float32)
        sky_t = np.clip(y / max(horizon, 1.0), 0.0, 1.0)[..., None]
        sky = (1 - sky_t) * np.array([120, 170, 225], np.float32) + sky_t * np.array([228, 214, 200], np.float32)
        depth = np.clip((y - horizon) / max(h - horizon, 1.0), 0.0, 1.0)[..., None]
        sand = (1 - depth) * np.array([170, 142, 104], np.float32) + depth * np.array([176, 136, 78], np.float32)
        sand = sand + (self._texture * 7.0 + self._blotches * 9.0)[..., None]
        ground = (y >= horizon)[..., None]
        img[:] = np.where(ground, sand, sky)
        # a distant dark-red barn: similar hue to the ball but too dark/grey to count
        if horizon > 20:
            x0, x1 = int(w * 0.04), int(w * 0.18)
            y0, y1 = int(max(horizon - h * 0.12, 0)), int(horizon)
            img[y0:y1, x0:x1] = np.array([128, 92, 88], np.float32) + self._texture[y0:y1, x0:x1, None] * 4
        self._bg = np.clip(img, 0, 255).astype(np.uint8)
        self._bg_key = key
        return self._bg

    # ------------------------------------------------------------ characters
    def _fill(self, img: np.ndarray, x0: float, y0: float, x1: float, y1: float, color: np.ndarray) -> None:
        h, w = img.shape[:2]
        a, b = int(max(math.floor(x0), 0)), int(min(math.ceil(x1), w))
        c, d = int(max(math.floor(y0), 0)), int(min(math.ceil(y1), h))
        if a < b and c < d:
            img[c:d, a:b] = color

    def draw_character(
        self,
        img: np.ndarray,
        u: float,
        v: float,
        half_w: float,
        half_h: float,
        red: bool,
        shirt: Sequence[int] = (34, 34, 40),
        pants: Sequence[int] = (44, 46, 60),
        skin: Sequence[int] = (232, 186, 150),
    ) -> None:
        """Blocky Roblox-style avatar centred on its torso at ``(u, v)``."""
        top, bottom = v - half_h, v + half_h
        hh = bottom - top
        cols = [np.array(c, np.float32) for c in (shirt, pants, skin)]
        if red:  # Blade Ball's red Highlight: red fill over the avatar + white outline
            cols = [np.clip(c * 0.35 + HIGHLIGHT_RED * 0.75, 0, 255) for c in cols]
            pad = max(1.0, half_w * 0.08)
            white = np.array([245, 245, 245], np.float32)
            self._fill(img, u - half_w * 0.55 - pad, top - pad, u + half_w * 0.55 + pad, top + hh * 0.22 + pad, white)
            self._fill(img, u - half_w - pad, top + hh * 0.22 - pad, u + half_w + pad, bottom + pad, white)
        shirt_c, pants_c, skin_c = cols
        # head, torso, arms, legs
        self._fill(img, u - half_w * 0.55, top, u + half_w * 0.55, top + hh * 0.22, skin_c)
        self._fill(img, u - half_w * 0.62, top + hh * 0.22, u + half_w * 0.62, top + hh * 0.58, shirt_c)
        self._fill(img, u - half_w, top + hh * 0.24, u - half_w * 0.64, top + hh * 0.56, skin_c if not red else shirt_c)
        self._fill(img, u + half_w * 0.64, top + hh * 0.24, u + half_w, top + hh * 0.56, skin_c if not red else shirt_c)
        self._fill(img, u - half_w * 0.6, top + hh * 0.58, u - half_w * 0.04, bottom, pants_c)
        self._fill(img, u + half_w * 0.04, top + hh * 0.58, u + half_w * 0.6, bottom, pants_c)

    # ------------------------------------------------------------ ball
    def draw_ball(self, img: np.ndarray, u: float, v: float, r: float, red: bool) -> None:
        h, w = img.shape[:2]
        glow_r = r * (1.7 if red else 1.15)
        x0, x1 = int(max(u - glow_r - 1, 0)), int(min(u + glow_r + 2, w))
        y0, y1 = int(max(v - glow_r - 1, 0)), int(min(v + glow_r + 2, h))
        if x0 >= x1 or y0 >= y1:
            return
        xx = self._xx[y0:y1, x0:x1]
        yy = self._yy[y0:y1, x0:x1]
        d = np.sqrt((xx - u) ** 2 + (yy - v) ** 2) / max(r, 0.5)
        patch = img[y0:y1, x0:x1].astype(np.float32)
        if red:
            glow = np.clip((1.7 - d) / 0.7, 0.0, 1.0)[..., None] * 0.75
            patch = patch * (1 - glow) + BALL_GLOW * glow
            shade = np.clip(1.08 - 0.25 * ((yy - v) / max(r, 0.5) + 1) / 2, 0.8, 1.1)[..., None]
            core = np.clip(BALL_RED * shade, 0, 255)
        else:
            ring = ((d > 0.85) & (d <= 1.15))[..., None]
            patch = np.where(ring, np.array([250, 250, 250], np.float32), patch)
            core = BALL_GRAY
        inside = (d <= 1.0)[..., None]
        patch = np.where(inside, core, patch)
        img[y0:y1, x0:x1] = np.clip(patch, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------ frame
    def render(
        self,
        cam: Camera,
        ball_pos: Optional[np.ndarray],
        ball_radius: float,
        ball_red: bool,
        player_red: bool,
        others: Sequence[tuple[np.ndarray, bool]] = (),
    ) -> np.ndarray:
        img = self.background(cam).copy()
        # other players, far ones first
        order = []
        for pos, red in others:
            u, v, z = cam.project(np.asarray(pos, dtype=np.float64)[None, :] + np.array([[0.0, TORSO[1], 0.0]]))
            if z[0] > 1.0:
                order.append((float(z[0]), float(u[0]), float(v[0]), red))
        for z, u, v, red in sorted(order, reverse=True):
            s = cam.focal / z
            self.draw_character(img, u, v, CHAR_HALF_WIDTH * s, CHAR_HALF_HEIGHT * s, red, shirt=(40, 70, 150), pants=(40, 40, 48))
        cu, cv, chw, chh = cam.character_box()
        self.draw_character(img, cu, cv, chw, chh, player_red)
        if ball_pos is not None:
            u, v, z = cam.project(np.asarray(ball_pos, dtype=np.float64)[None, :])
            if z[0] > 0.5:
                r = cam.focal * ball_radius / float(z[0])
                self.draw_ball(img, float(u[0]), float(v[0]), r, ball_red)
        return img
