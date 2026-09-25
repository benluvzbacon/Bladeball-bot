import time

import numpy as np
import pytest

from bladebot.config import Settings
from bladebot.engine import BotEngine, ParryDecider, default_config, run_arena_benchmark


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
    assert info["loaded"] and info["layers"][0] == 19


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
