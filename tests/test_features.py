import math

import numpy as np
import pytest

from bladebot.features import FEATURE_NAMES, N_FEATURES, RALLY_MAX_S, Episode, ExpFit, TrackState


def feat(values):
    return dict(zip(FEATURE_NAMES, values))


def test_unseen_ball_still_gives_episode_features():
    tr = TrackState()
    assert not tr.active
    assert tr.heuristic_tti() is None
    d = feat(tr.features(0.25, 0.3, ep_start=0.05))
    assert len(d) == N_FEATURES
    assert d["seen"] == 0.0 and d["has_vel"] == 0.0
    assert d["ep_age"] == pytest.approx(0.2)
    assert d["stale"] == 2.0  # "never seen" looks like a very old sighting
    assert d["has_prev"] == 0.0 and d["prev_gap"] == RALLY_MAX_S


def test_single_detection_features():
    tr = TrackState()
    tr.update(0.0, (0.1, -0.2, 0.02))
    f = tr.features(0.0)
    assert len(f) == N_FEATURES == len(FEATURE_NAMES)
    d = feat(f)
    assert d["has_vel"] == 0.0 and d["seen"] == 1.0
    assert d["dx"] == pytest.approx(0.1)
    assert d["log_r"] == pytest.approx(math.log(0.02))
    assert all(math.isfinite(v) for v in f)


def test_looming_ball_gives_positive_rates_and_sane_heuristic():
    tr = TrackState()
    impact = 1.0
    t = 0.0
    while t < 0.7:
        tti = impact - t
        tr.update(t, (0.3 * tti, -0.2 * tti, 0.01 / tti))  # grows and moves towards the character
        t += 1 / 60
    d = feat(tr.features(t, 0.3, 0.0))
    assert d["loom_rate"] > 1.0
    assert d["loom_slow"] > 1.0
    assert d["loom_acc"] > 0  # looming speeds up as the ball gets closer
    assert d["closing"] > 0
    assert d["has_vel"] == 1.0
    assert d["loom_max"] >= d["loom_rate"] * 0.9
    h = tr.heuristic_tti()
    assert h is not None and 0.1 < h < 0.8


def test_curving_ball_has_a_turn_rate():
    tr = TrackState()
    for k in range(40):
        t = k / 100
        ang = 2.0 * t  # moving on a circle on screen: constant turn rate of 2 rad/s
        tr.update(t, (0.3 * math.cos(ang), 0.3 * math.sin(ang), 0.02))
    d = feat(tr.features(0.39))
    assert d["turn"] == pytest.approx(2.0, rel=0.15)


def test_long_gap_starts_a_new_sighting_but_keeps_the_episode():
    tr = TrackState(reset_gap_s=0.25)
    tr.update(0.0, (0.0, 0.0, 0.01))
    tr.update(0.05, (0.0, 0.0, 0.03))
    tr.update(0.1, None)
    assert tr.recent(0.1)
    tr.update(0.5, (0.2, 0.0, 0.01))
    d = feat(tr.features(0.6, 0.3, 0.0))
    assert d["has_vel"] == 0.0  # the fits restarted
    assert d["stale"] == pytest.approx(0.1)
    assert d["seen_age"] == pytest.approx(0.6)  # first seen at t = 0
    assert d["r_max"] == pytest.approx(math.log(0.03), abs=1e-6)  # the episode remembers the biggest size


def test_expfit_matches_weighted_least_squares():
    rng = np.random.default_rng(0)
    ts = np.cumsum(rng.uniform(0.005, 0.02, 50))
    vals = 0.3 + 1.2 * ts - 2.0 * ts**2
    for deg, tau in ((1, 0.07), (2, 0.2)):
        fit = ExpFit(tau, deg, 1)
        for t, v in zip(ts, vals):
            fit.add(float(t), (float(v),))
        value, slope, curv = fit.solve()[0]
        s = ts - ts[-1]
        w = np.sqrt(np.exp(s / tau))
        coef = np.linalg.lstsq(np.vander(s, deg + 1, increasing=True) * w[:, None], vals * w, rcond=None)[0]
        assert value == pytest.approx(coef[0], abs=1e-7)
        assert slope == pytest.approx(coef[1], abs=1e-5)
        if deg == 2:
            assert curv == pytest.approx(2 * coef[2], abs=1e-3)


def test_episode_measures_the_rally_rhythm():
    ep = Episode(rearm_s=0.06)
    t = 0.0
    for on, dur in ((True, 0.5), (False, 0.3), (True, 0.4)):
        end = t + dur
        while t < end - 1e-9:
            ep.update(t, on)
            t = round(t + 0.01, 6)
    assert ep.active
    assert ep.gap == pytest.approx(0.31, abs=0.011)  # away for 0.3 s (+1 frame)
    assert ep.prev_dur == pytest.approx(0.49, abs=0.011)
    assert ep.start == pytest.approx(0.8)
    # a long break (new round) makes the rhythm unknown again
    for _ in range(500):
        ep.update(t, False)
        t += 0.01
    ep.update(t, True)
    assert ep.started_now and ep.gap is None and ep.prev_dur is None


def test_rally_features():
    tr = TrackState()
    d = feat(tr.features(0.3, 0.3, 0.0, 0.6, 0.8))
    assert d["has_prev"] == 1.0
    assert d["prev_gap"] == pytest.approx(0.6) and d["prev_dur"] == pytest.approx(0.8)
    assert d["age_gap"] == pytest.approx(0.5)
    unknown = feat(tr.features(0.3, 0.3, 0.0, None, 0.8))
    assert unknown["has_prev"] == 0.0 and unknown["age_gap"] == 0.0
