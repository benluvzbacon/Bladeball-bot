import time

import numpy as np
import pytest

from bladebot.config import Settings
from bladebot.engine import BotEngine, FramePipeline, ParryDecider, default_config, run_arena_benchmark
from bladebot.features import FEATURE_NAMES, N_FEATURES


class FakeController:
    backend = None
    backend_error = None
    last_error = None
    available = True

    def __init__(self):
        self.calls = []

    def press(self, method, key, hold_s):
        self.calls.append((method, key, hold_s))
        return True


def cfg(**kw):
    base = dict(confidence=0.5, confirm_frames=2, instant_confidence=0.95, min_interval_ms=100, rearm_ms=60, retry_ms=2600)
    base.update(kw)
    return default_config(**base)


def test_decider_presses_once_per_approach():
    d = ParryDecider()
    c = cfg()
    times = np.arange(0, 1.0, 1 / 60)
    presses = [d.update(t, True, 0.9 if t >= 0.1 else 0.1, c) for t in times]
    assert sum(presses) == 1
    first_high = int(np.argmax(times >= 0.1))
    assert presses.index(True) == first_high + 1  # confirmed on the second frame
    for t in np.arange(1.0, 1.2, 1 / 60):
        assert not d.update(t, False, 0.0, c)  # the ball left: episode ends
    again = [d.update(t, True, 0.9, c) for t in np.arange(1.2, 1.5, 1 / 60)]
    assert sum(again) == 1


def test_decider_instant_and_retry_and_gate():
    d = ParryDecider()
    c = cfg()
    assert d.update(0.0, True, 0.99, c)  # instant confidence skips confirmation
    presses = [d.update(t, True, 0.99, c) for t in np.arange(1 / 60, 3.0, 1 / 60)]
    assert sum(presses) == 1  # one retry after retry_ms (a whiff + cooldown)
    d2 = ParryDecider()
    assert not any(d2.update(t, False, 0.99, c) for t in np.arange(0, 1, 1 / 60))


def test_settings_are_validated_and_saved(tmp_path):
    path = tmp_path / "settings.json"
    s = Settings(path)
    s.update({"lead_ms": 99999, "parry_key": "F", "hotkey": "PageUp", "bogus": 1, "confidence": "0.7"})
    assert s["lead_ms"] == 700
    assert s["parry_key"] == "f" and s["hotkey"] == "page_up"
    assert s["confidence"] == pytest.approx(0.7)
    with pytest.raises(ValueError):
        s.update({"source": "moon"})
    with pytest.raises(ValueError):
        s.update({"parry_key": "not-a-key"})
    assert Settings(path)["lead_ms"] == 700


def test_screen_source_needs_practice_notice(tmp_path, bundled_model):
    settings = Settings(tmp_path / "s.json")
    engine = BotEngine(settings, controller=FakeController(), model=bundled_model)
    ok, msg = engine.set_enabled(True)
    assert not ok and "practice" in msg.lower()
    settings.update({"practice_ack": True})
    assert engine.set_enabled(True)[0]
    engine.set_enabled(False)
    settings.update({"practice_ack": False, "source": "arena"})
    assert engine.set_enabled(True)[0]  # the offline arena is always fine


def test_engine_without_model_refuses(tmp_path):
    engine = BotEngine(Settings(tmp_path / "s.json"), controller=FakeController(), autoload_model=False)
    ok, msg = engine.set_enabled(True)
    assert not ok and "network" in msg.lower()


def test_network_plays_the_arena(bundled_model):
    st = run_arena_benchmark(bundled_model, seconds=25, fps=60, seed=4)
    assert st["blocks"] >= 8
    assert st["block_rate"] > 0.75


def test_engine_thread_runs_arena(tmp_path, bundled_model):
    settings = Settings(tmp_path / "s.json")
    settings.update({"source": "arena"})
    ctrl = FakeController()
    engine = BotEngine(settings, controller=ctrl, model=bundled_model)
    engine.start()
    try:
        assert engine.set_enabled(True)[0]
        engine.preview()
        time.sleep(4.0)
        st = engine.status()
    finally:
        engine.stop()
    assert st["frames"] > 60
    assert st["arena"] is not None and st["enabled"]
    assert engine.preview() is not None  # a PNG was produced
    assert ctrl.calls == []  # arena presses never touch the real keyboard
    assert engine.arena.stats["blocks"] + engine.arena.stats["hits"] >= 1


def test_engine_autoloads_bundled_model(tmp_path, bundled_model):
    engine = BotEngine(Settings(tmp_path / "s.json"), controller=FakeController())
    assert engine.model is not None and engine.model_error is None
    assert engine.model_path == "models/parry_net.npz"
    info = engine.model_info()
    assert info["loaded"] and info["layers"][0] == N_FEATURES


def test_engine_reports_missing_model(tmp_path):
    settings = Settings(tmp_path / "s.json")
    settings.update({"model_path": "models/does_not_exist.npz"})
    engine = BotEngine(settings, controller=FakeController())
    assert engine.model is None and "not found" in engine.model_error


def test_human_block_through_engine_thread(tmp_path, bundled_model):
    settings = Settings(tmp_path / "s.json")
    settings.update({"source": "arena", "arena_mode": "human", "arena_speed": 30})
    engine = BotEngine(settings, controller=FakeController(), model=bundled_model)
    engine.start()
    pressed = False
    try:
        deadline = time.time() + 12
        while time.time() < deadline:
            engine.preview()  # keeps the loop running like an open arena tab
            a = engine.arena
            tti = a.true_tti() if a.phase == "incoming" else None
            if not pressed and tti is not None and tti < 0.3:
                engine.human_parry()
                pressed = True
            if pressed and a.phase != "incoming":
                break
            time.sleep(0.005)
    finally:
        engine.stop()
    assert pressed
    assert engine.arena.stats["blocks"] == 1
    assert engine.arena.feedback["result"] == "blocked"
    assert "network would have pressed" in engine.arena.feedback["text"]


def test_hidden_ball_widens_the_press_horizon(bundled_model):
    from bladebot.model import BLIND_AFTER_S, BLIND_HORIZON_S

    m = bundled_model
    probs = np.linspace(0.02, 0.98, len(m.horizons)).astype(np.float32)  # a monotone CDF
    visible = m.press_probability(probs, 0.30, 0.0)
    hidden = m.press_probability(probs, 0.30, BLIND_AFTER_S + 0.01)
    assert visible == pytest.approx(m.prob_within(probs, 0.30))
    assert hidden == pytest.approx(m.prob_within(probs, BLIND_HORIZON_S))
    assert hidden > visible
    # a lead time longer than the blind horizon is kept as it is
    assert m.press_probability(probs, 0.60, 0.2) == pytest.approx(m.prob_within(probs, 0.60))
    # a ball gone for a long time is not "just hidden"
    assert m.press_probability(probs, 0.30, 1.0) == pytest.approx(visible)
    # a ball not seen yet: wait a moment for its launch direction, then widen
    assert m.press_probability(probs, 0.30, 2.0, seen=False, ep_age=0.05) == pytest.approx(visible)
    assert m.press_probability(probs, 0.30, 2.0, seen=False, ep_age=0.3) == pytest.approx(hidden)


def test_vectorised_press_probability_matches(bundled_model):
    m = bundled_model
    rng = np.random.default_rng(0)
    probs = np.sort(rng.uniform(0, 1, (200, len(m.horizons))), axis=1).astype(np.float32)
    lead = rng.uniform(0.0, 1.2, 200)
    stale = rng.choice([0.0, 0.03, 0.1, 0.5, 1.0, 2.0], 200)
    seen = rng.random(200) < 0.7
    age = rng.uniform(0, 1, 200)
    many = m.press_probability_many(probs, lead, stale, seen, age)
    one = [m.press_probability(p, float(l), float(s), bool(v), float(a)) for p, l, s, v, a in zip(probs, lead, stale, seen, age)]
    assert np.allclose(many, one, atol=1e-6)


def test_old_custom_model_falls_back_to_the_bundled_one(tmp_path, bundled_model):
    from bladebot.nn import MLP

    old = tmp_path / "old_model.npz"
    MLP([19, 8, 20]).save(old, {"feature_names": [f"f{i}" for i in range(19)], "mean": [0.0] * 19,
                                "std": [1.0] * 19, "horizons": [0.05 * k for k in range(1, 21)]})
    settings = Settings(tmp_path / "s.json")
    settings.update({"model_path": str(old)})
    engine = BotEngine(settings, controller=FakeController())
    assert engine.model is not None and engine.model_error is None
    assert engine.model_path == "models/parry_net.npz"
    assert any("older BladeBot" in e["text"] for e in engine.events)


def test_pipeline_tracks_the_rally_and_compensates_half_a_frame(bundled_model):
    """Episodes start/stop with the red highlight and the rally rhythm reaches the network."""
    from bladebot.sim.arena import Arena

    arena = Arena(seed=5)
    arena.configure({"arena_ping_ms": 0, "arena_curves": "off"})
    pipe = FramePipeline(bundled_model)
    c = default_config(lead_ms=300)
    gaps = []
    for _ in range(int(20 * 60)):
        res = pipe.process(arena.frame(), arena.t, arena.vision_settings(), c)
        if res.press:
            arena.request_parry("bot")
        if pipe.episode.started_now and pipe.episode.gap is not None:
            gaps.append(pipe.episode.gap)
        arena.step(1 / 60)
    assert pipe.frame_dt == pytest.approx(1 / 60, rel=0.05)
    assert pipe.lead_s(c) == pytest.approx(0.3 + 0.5 / 60, rel=0.02)
    assert len(gaps) >= 3 and all(0.02 < g < 4.0 for g in gaps)
    d = dict(zip(FEATURE_NAMES, res.features if res.features is not None else [0.0] * N_FEATURES))
    assert set(d) == set(FEATURE_NAMES)
