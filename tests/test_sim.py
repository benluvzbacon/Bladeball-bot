import numpy as np
import pytest

from bladebot.sim.arena import Arena
from bladebot.sim.world import (
    CURVE_NAMES,
    HIT_RADIUS,
    TORSO,
    Camera,
    OneBall,
    detections_for_frames,
    flight_step,
    sample_approaches,
    simulate_trajectories,
)


def run_until(arena, cond, max_t=15.0, dt=0.004):
    t0 = arena.t
    while not cond():
        arena.step(dt)
        assert arena.t - t0 < max_t, "condition never happened"


def test_camera_puts_character_below_centre():
    cam = Camera(width=640, height=360)
    u, v = cam.anchor()
    assert abs(u - 320) < 1e-6
    assert v > 180  # the camera looks at the head from behind and above
    assert Camera(width=640, height=360, shift=1.75).anchor()[0] < 320  # shift-lock: character left of centre


def test_homing_balls_arrive():
    rng = np.random.default_rng(0)
    ap = sample_approaches(rng, 300, speed_range=(60, 300), distance_range=(15, 60))
    pos, t_imp = simulate_trajectories(ap)
    hit = np.isfinite(t_imp)
    assert hit.mean() > 0.98
    i = int(np.flatnonzero(hit)[0])
    assert np.linalg.norm(pos[-1, i] - TORSO) <= ap.radius[i] + HIT_RADIUS + 1e-6


def test_curve_balls_leave_sideways_and_still_arrive():
    rng = np.random.default_rng(5)
    ap = sample_approaches(rng, 600, speed_range=(60, 300), distance_range=(15, 60))
    pos, t_imp = simulate_trajectories(ap)
    for k, name in enumerate(CURVE_NAMES):
        sel = ap.curve == k
        assert sel.sum() > 10, name
        assert np.isfinite(t_imp[sel]).mean() > 0.93, name
    # a backwards curve first flies away from you
    to_you = TORSO - ap.start
    to_you /= np.linalg.norm(to_you, axis=1, keepdims=True)
    cos = np.einsum("ij,ij->i", ap.direction, to_you)
    assert np.all(cos[ap.curve == 3] < -0.3)
    assert np.all(cos[ap.curve == 0] > 0.8)
    # curves take longer than the straight line
    straight = np.linalg.norm(ap.start - TORSO, axis=1) / ap.speed
    ratio = t_imp / straight
    ok = np.isfinite(ratio)
    assert np.median(ratio[ok & (ap.curve == 3)]) > 1.3 * np.median(ratio[ok & (ap.curve == 0)])


def test_single_ball_flight_matches_the_batch_version():
    rng = np.random.default_rng(3)
    ap = sample_approaches(rng, 40, moving_prob=0.0)
    pos, d, sp = ap.start.copy(), ap.direction.copy(), ap.speed.copy()
    balls = [OneBall(ap.start[i], ap.direction[i], ap.speed[i], ap.flight, i) for i in range(ap.n)]
    for s in range(200):
        pos, d, sp = flight_step(pos, d, sp, s * 0.004, ap.flight, 0.004)
        for b in balls:
            b.step(0.004)
    assert np.allclose([b.pos for b in balls], pos, atol=1e-6)


def test_turning_the_camera_moves_the_ball_on_screen():
    cam = Camera(width=640, height=360)
    pts = np.array([[0.0, 5.0, 40.0]] * 3)
    u0, _, _ = cam.project(pts)
    u, _, _ = cam.project_moving(pts, np.radians([0.0, 10.0, -10.0]), np.full(3, cam.pitch))
    assert u[0] == pytest.approx(u0[0])
    assert u[1] < u0[0] < u[2]  # turning right moves things left on screen


def test_simulated_detections_hide_ball_behind_character():
    cam = Camera(width=640, height=360)
    rng = np.random.default_rng(1)
    pts = np.array([[0.0, 3.0, 60.0], TORSO + np.array([0.0, 0.0, 0.2])])
    dets, char_h = detections_for_frames(rng, cam, pts, 1.2, proc_scale=1, dropout=0.0)
    assert dets[0] is not None and dets[1] is None
    x, y, r = dets[0]
    assert y < 0  # above the character on screen
    assert 0.05 < char_h < 1.0


def test_arena_well_timed_block():
    a = Arena(seed=0)
    a.configure({"arena_ping_ms": 0})
    run_until(a, lambda: a.phase == "incoming")
    run_until(a, lambda: (a.true_tti() or 99) <= 0.2)
    a.request_parry("human")
    run_until(a, lambda: a.phase != "incoming")
    assert a.phase == "away"
    assert a.stats["blocks"] == 1 and a.stats["hits"] == 0
    assert a.feedback["result"] == "blocked"
    speed = a.speed
    run_until(a, lambda: a.phase == "incoming")  # the other player sends it back, faster
    assert a.speed == pytest.approx(speed)
    assert speed > 45.0


def test_arena_early_block_whiffs_and_costs_cooldown():
    a = Arena(seed=1)
    a.configure({"arena_ping_ms": 0, "arena_speed": 18.0})
    run_until(a, lambda: a.phase == "incoming")
    assert a.true_tti() > 0.8
    a.request_parry("human")
    a.step(0.6)
    assert a.stats["whiffs"] == 1
    assert a.cooldown_until - a.t > 1.0
    assert a.feedback["result"] == "early"
    run_until(a, lambda: a.phase != "incoming")
    assert a.phase == "eliminated" and a.stats["hits"] == 1


def test_arena_late_block_is_a_hit():
    a = Arena(seed=2)
    a.configure({"arena_ping_ms": 120})
    run_until(a, lambda: a.phase == "incoming")
    run_until(a, lambda: (a.true_tti() or 99) <= 0.05)
    a.request_parry("human")
    run_until(a, lambda: a.phase != "incoming")
    assert a.stats["hits"] == 1
    assert "Too late" in a.feedback["text"]


def test_arena_frames_show_targeting():
    a = Arena(seed=3)
    img = a.frame()
    assert img.shape == (360, 640, 3) and img.dtype == np.uint8
    run_until(a, lambda: a.phase == "incoming")
    from bladebot.vision import BallDetector

    det = BallDetector(a.vision_settings()).detect(a.frame())
    assert det.targeted


def test_arena_curve_setting():
    def curves(setting, n=12, seed=7):
        a = Arena(seed=seed)
        a.configure({"arena_curves": setting, "arena_ping_ms": 0})
        seen = []
        for _ in range(n):
            a._start_incoming(a.opponents[0])
            seen.append(a.curve)
            assert a.true_tti() is not None and a.true_tti() > 0
        return seen

    assert set(curves("off")) == {"direct"}
    assert len(set(curves("lots")) - {"direct"}) >= 2


def test_arena_true_tti_is_exact():
    a = Arena(seed=11)
    a.configure({"arena_curves": "lots", "arena_ping_ms": 0})
    run_until(a, lambda: a.phase == "incoming")
    tti = a.true_tti()
    t0 = a.t
    run_until(a, lambda: a.phase != "incoming")
    assert a.t - t0 == pytest.approx(tti, abs=0.0041)
