import json
import math

import numpy as np
import pytest

from bladebot.features import N_FEATURES
from bladebot.model import HORIZONS
from bladebot.nn import bce_with_logits
from bladebot.training import (
    Dataset,
    TrainConfig,
    dataset_from_recordings,
    debounce,
    evaluate,
    generate_sim_dataset,
    sample_weights,
    simulate_parries,
    train_network,
)


def test_generate_small_dataset():
    ds = generate_sim_dataset(120, seed=3)
    assert len(ds) > 1000
    assert ds.X.shape == (len(ds), N_FEATURES)
    assert np.all(np.isfinite(ds.X))
    y = ds.labels()
    assert y.shape == (len(ds), len(HORIZONS))
    assert np.all(np.diff(y, axis=1) >= 0)  # "within 0.1 s" implies "within 0.2 s"
    assert np.all(np.diff(ds.group) >= 0)
    assert np.isinf(ds.tti).any()  # decoys
    assert (ds.tti[np.isfinite(ds.tti)] >= -0.02).all()


def test_training_beats_base_rates():
    tr = generate_sim_dataset(400, seed=1)
    va = generate_sim_dataset(100, seed=2, keep_all=True)
    model, hist = train_network(tr, va, TrainConfig(hidden=(16,), epochs=3, batch_size=512, patience=10))
    assert len(hist) == 3
    y = va.labels()
    w = sample_weights(va.tti, 2.0)
    base = np.clip(tr.labels().mean(axis=0), 1e-3, 1 - 1e-3)
    base_loss, _ = bce_with_logits(np.tile(np.log(base / (1 - base)), (len(va), 1)), y, w)
    assert min(h["val_loss"] for h in hist) < base_loss * 0.8
    report = evaluate(model, va)
    assert 0.0 <= report["nn"]["success"] <= 1.0
    assert report["nn"]["approaches"] > 50


def _one_approach(ttis, group=0):
    t = np.asarray(ttis, dtype=np.float32)
    n = len(t)
    return Dataset(np.zeros((n, N_FEATURES), np.float32), t, np.full(n, group), np.full(n, np.inf, np.float32), np.zeros(n, np.float32))


def test_parry_rules():
    ds = _one_approach([0.9, 0.7, 0.5, 0.3, 0.1, 0.0])
    press = np.zeros(6, bool)
    press[3] = True
    assert simulate_parries(ds, press, latency=0.06)["success"] == 1.0
    press[:] = False
    press[0] = True
    assert simulate_parries(ds, press, latency=0.06)["too_early"] == 1.0
    press[:] = False
    press[5] = True
    assert simulate_parries(ds, press, latency=0.06)["too_late"] == 1.0
    decoy = _one_approach([math.inf] * 4, group=1)
    both = Dataset.concat([ds, decoy])
    r = simulate_parries(both, np.array([0, 0, 0, 1, 0, 0, 1, 0, 0, 0], bool))
    assert r["success"] == 1.0 and r["false_press"] == 1.0


def test_debounce():
    press = np.array([1, 1, 0, 1, 1, 1, 0, 1, 1, 1], bool)
    group = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
    assert debounce(press, group, 1).tolist() == press.tolist()
    assert debounce(press, group, 2).astype(int).tolist() == [0, 1, 0, 0, 1, 1, 0, 0, 0, 1]


def test_hindsight_labels_from_recording(tmp_path):
    path = tmp_path / "session-test.jsonl"
    rows = [{"type": "header", "version": 1, "source": "screen", "gate": True, "char_h": 0.3}]
    t = 0.0
    dt = 1 / 60
    for k in range(70):
        targeted = 10 <= k < 40
        det = None
        if targeted:
            remaining = (40 - k) * dt
            det = [0.2 * remaining, -0.1, 0.01 / max(remaining, 0.02)]
        rows.append({"t": round(t, 5), "d": det, "g": int(targeted)})
        t += dt
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    ds = dataset_from_recordings([path])
    assert len(ds) == 30
    assert np.all(np.diff(ds.tti) < 0)
    assert ds.tti[-1] == pytest.approx(dt / 2, abs=1e-3)  # last red frame + half a frame
    assert ds.tti[0] == pytest.approx(29 * dt + dt / 2, abs=1e-3)


def test_truncated_recording_episode_is_skipped(tmp_path):
    path = tmp_path / "session-cut.jsonl"
    rows = [{"type": "header", "gate": True}]
    for k in range(30):
        rows.append({"t": k / 60, "d": [0.1, 0.1, 0.02] if k >= 10 else None, "g": int(k >= 10)})
    path.write_text("\n".join(json.dumps(r) for r in rows))
    assert len(dataset_from_recordings([path])) == 0  # still red when the recording stopped
