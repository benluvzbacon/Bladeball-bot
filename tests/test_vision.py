import struct
import zlib

import numpy as np
import pytest

from bladebot.pngenc import encode_png
from bladebot.vision import (
    BallDetector,
    VisionSettings,
    connected_components,
    hsv_match,
    red_mask,
    render_preview,
    rgb_to_hsv_pixel,
)

SAND = (176, 140, 92)
BALL_RED = (214, 52, 74)


def frame(h=360, w=640, color=SAND):
    img = np.empty((h, w, 3), dtype=np.uint8)
    img[:] = color
    return img


def disc(img, cx, cy, r, color):
    yy, xx = np.mgrid[0 : img.shape[0], 0 : img.shape[1]]
    img[(xx - cx) ** 2 + (yy - cy) ** 2 <= r * r] = color


def test_hsv_of_blade_ball_red():
    h, s, v = rgb_to_hsv_pixel(*BALL_RED)
    assert 345 < h < 355 and s > 0.6 and v > 0.8


def test_red_mask_lut_matches_exact_hsv():
    settings = VisionSettings()
    rng = np.random.default_rng(0)
    px = rng.integers(0, 256, (50000, 3)).astype(np.uint8)
    exact = hsv_match(px[:, 0], px[:, 1], px[:, 2], settings)
    lut = red_mask(px.reshape(-1, 1, 3), settings).ravel()
    assert (exact == lut).mean() > 0.99
    # the sandy arena floor and a grey ball are not "red"
    assert not red_mask(frame(4, 4), settings).any()
    assert not red_mask(frame(4, 4, (200, 200, 205)), settings).any()


def test_connected_components_counts_blobs():
    m = np.zeros((50, 60), dtype=bool)
    m[5:10, 5:10] = True
    m[20:30, 40:45] = True
    m[40, 0:60] = True
    m[10, 10] = True  # diagonal neighbour of the first square -> same blob (8-connectivity)
    blobs = sorted(connected_components(m), key=lambda b: b.area)
    assert len(blobs) == 3
    assert blobs[0].area == 26 and blobs[1].area == 50 and blobs[2].area == 60
    sq = blobs[0]
    assert (sq.x0, sq.y0, sq.x1, sq.y1) == (5, 5, 10, 10)


def test_detects_red_ball_and_ignores_grey_ball():
    img = frame()
    disc(img, 420, 120, 14, BALL_RED)
    disc(img, 150, 100, 14, (200, 200, 205))
    det = BallDetector(VisionSettings(gate_enabled=False)).detect(img)
    assert det.ball_px is not None
    x, y, r = det.ball_px
    assert x == pytest.approx(420, abs=1.5) and y == pytest.approx(120, abs=1.5)
    assert r == pytest.approx(14, abs=2)
    # anchor-relative, normalised by the frame height
    s = VisionSettings()
    assert det.ball[0] == pytest.approx((420 - s.anchor_x * 640) / 360, abs=0.01)


def test_gate_detects_red_character_and_excludes_it():
    s = VisionSettings()
    img = frame()
    ax, ay = s.anchor_x * 640, s.anchor_y * 360
    hw, hh = s.gate_w * 360 / 2, s.gate_h * 360 / 2
    img[int(ay - hh * 0.8) : int(ay + hh * 0.8), int(ax - hw * 0.6) : int(ax + hw * 0.6)] = (170, 50, 55)
    det = BallDetector(s).detect(img)
    assert det.targeted and det.gate_frac > 0.2
    assert det.ball is None  # the character itself is not the ball
    disc(img, 100, 80, 10, BALL_RED)
    det = BallDetector(s).detect(img)
    assert det.targeted and det.ball_px is not None and abs(det.ball_px[0] - 100) < 2


def test_ball_partly_behind_character_box_keeps_its_size():
    s = VisionSettings()
    img = frame()
    ax, ay = s.anchor_x * 640, s.anchor_y * 360
    top = ay - s.gate_h * 360 / 2
    disc(img, ax, top - 6, 16, BALL_RED)  # centre just above the box, lower part hidden
    det = BallDetector(s).detect(img)
    assert det.ball_px is not None
    assert det.ball_px[2] == pytest.approx(16, abs=3)
    assert det.ball_px[1] == pytest.approx(top - 6, abs=3)


def test_rectangles_and_thin_shapes_are_rejected():
    img = frame()
    img[20:80, 20:90] = BALL_RED  # solid rectangle (UI panel)
    img[200:204, 300:420] = BALL_RED  # thin bar
    det = BallDetector(VisionSettings(gate_enabled=False)).detect(img)
    assert det.ball is None


def test_preview_and_png():
    img = frame(90, 160)
    disc(img, 40, 40, 8, BALL_RED)
    det = BallDetector().detect(img, keep_mask=True)
    prev = render_preview(img, det, max_width=80)
    assert prev.shape == (45, 80, 3) and prev.dtype == np.uint8
    png = encode_png(prev)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    # decode the IDAT back and compare
    w, h = struct.unpack(">II", png[16:24])
    assert (w, h) == (80, 45)
    idat_len = struct.unpack(">I", png[33:37])[0]
    raw = zlib.decompress(png[41 : 41 + idat_len])
    rows = np.frombuffer(raw, dtype=np.uint8).reshape(h, 1 + w * 3)
    assert np.array_equal(rows[:, 1:].reshape(h, w, 3), prev)
