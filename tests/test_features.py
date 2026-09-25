import math

import pytest

from bladebot.features import FEATURE_NAMES, N_FEATURES, TrackState


def test_empty_track_has_no_features():
    tr = TrackState()
    assert not tr.active
    assert tr.features(0.0) is None
    assert tr.heuristic_tti() is None


def test_single_detection_features():
    tr = TrackState()
    tr.update(0.0, (0.1, -0.2, 0.02))
    f = tr.features(0.0)
    assert len(f) == N_FEATURES == len(FEATURE_NAMES)
    d = dict(zip(FEATURE_NAMES, f))
    assert d["has_vel"] == 0.0
    assert d["dx"] == pytest.approx(0.1)
    assert d["log_r"] == pytest.approx(math.log(0.02))


def test_looming_ball_gives_positive_rates_and_sane_heuristic():
    tr = TrackState()
    impact = 1.0
    t = 0.0
    while t < 0.7:
        tti = impact - t
        tr.update(t, (0.3 * tti, -0.2 * tti, 0.01 / tti))  # grows and moves towards the character
        t += 1 / 60
    d = dict(zip(FEATURE_NAMES, tr.features(t)))
    assert d["loom_rate"] > 1.0
    assert d["loom_q"] > d["loom_rate"] * 0.9  # the quadratic fit lags less
    assert d["closing"] > 0
    assert d["has_vel"] == 1.0
    h = tr.heuristic_tti()
    assert h is not None and 0.1 < h < 0.8


def test_track_resets_after_a_long_gap():
    tr = TrackState(reset_gap_s=0.25)
    tr.update(0.0, (0.0, 0.0, 0.01))
    tr.update(0.1, None)
    assert tr.active  # short dropout keeps the track
    tr.update(0.5, (0.0, 0.0, 0.01))
    assert len(tr.points) == 1  # new approach
    assert tr.features(0.6)[FEATURE_NAMES.index("stale")] == pytest.approx(0.1)


def test_window_is_bounded():
    tr = TrackState(max_points=16, window_s=0.3)
    for k in range(200):
        tr.update(k / 144, (0.0, 0.0, 0.01))
    assert len(tr.points) <= 16
    assert tr.points[-1][0] - tr.points[0][0] <= 0.3 + 1e-9
