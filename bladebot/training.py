"""Train the parry network.

Training data comes from two places:

1. **The simulator** (:mod:`bladebot.sim.world`): thousands of randomised
   red-ball approaches with different speeds, curves, camera distances/angles,
   frame rates and detection noise. For each frame we know the exact
   time-to-impact, so labelling is free.
2. **Your recordings** (optional, :mod:`bladebot.recorder`): real practice
   sessions labelled in hindsight from the moment the red highlight ended.

Run ``python -m bladebot.training --help`` for the command-line interface.
"""

from __future__ import annotations

import argparse
import math
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np

from .features import DEFAULT_CHAR_H, N_FEATURES, TrackState
from .model import DEFAULT_MODEL_PATH, HORIZONS, STALE_INDEX, ParryNet
from .nn import MLP, Adam, bce_with_logits
from .recorder import iter_episodes, list_recordings, read_recording
from .sim.world import (
    PARRY_WINDOW_S,
    detections_for_frames,
    frame_times,
    interpolate_positions,
    sample_approaches,
    simulate_trajectories,
)

ProgressFn = Callable[[float, str], None]
SIM_DT = 0.004
SIM_T_MAX = 5.0


# =================================================================== data
@dataclass
class Dataset:
    X: np.ndarray  # (N, F) features
    tti: np.ndarray  # (N,) true time-to-impact in seconds (inf = never hits)
    group: np.ndarray  # (N,) approach id (frames of one approach are consecutive)
    heur: np.ndarray  # (N,) classic heuristic time-to-impact (inf = unknown)
    speed: np.ndarray  # (N,) ball speed in studs/s (0 if unknown)
    weight_scale: float = 1.0

    def __len__(self) -> int:
        return int(self.X.shape[0])

    @staticmethod
    def empty() -> "Dataset":
        z = np.zeros(0, dtype=np.float32)
        return Dataset(np.zeros((0, N_FEATURES), np.float32), z, np.zeros(0, np.int64), z, z)

    @staticmethod
    def concat(parts: Sequence["Dataset"]) -> "Dataset":
        parts = [p for p in parts if len(p)]
        if not parts:
            return Dataset.empty()
        offset = 0
        groups = []
        for p in parts:
            groups.append(p.group + offset)
            offset += int(p.group.max()) + 1
        return Dataset(
            np.concatenate([p.X for p in parts]),
            np.concatenate([p.tti for p in parts]),
            np.concatenate(groups),
            np.concatenate([p.heur for p in parts]),
            np.concatenate([p.speed for p in parts]),
        )

    def labels(self, horizons: Sequence[float] = HORIZONS) -> np.ndarray:
        h = np.asarray(horizons, dtype=np.float32)
        return (self.tti[:, None] <= h[None, :]).astype(np.float32)


class _Builder:
    def __init__(self) -> None:
        self.rows: list[list[float]] = []
        self.tti: list[float] = []
        self.group: list[int] = []
        self.heur: list[float] = []
        self.speed: list[float] = []

    def add(self, feats: list[float], tti: float, group: int, heur: Optional[float], speed: float) -> None:
        self.rows.append(feats)
        self.tti.append(tti)
        self.group.append(group)
        self.heur.append(math.inf if heur is None else heur)
        self.speed.append(speed)

    def build(self) -> Dataset:
        if not self.rows:
            return Dataset.empty()
        return Dataset(
            np.asarray(self.rows, dtype=np.float32),
            np.asarray(self.tti, dtype=np.float32),
            np.asarray(self.group, dtype=np.int64),
            np.asarray(self.heur, dtype=np.float32),
            np.asarray(self.speed, dtype=np.float32),
        )


def generate_sim_dataset(
    approaches: int = 12000,
    seed: int = 0,
    speed_range: tuple[float, float] = (20.0, 350.0),
    keep_all: bool = False,
    keep_far_prob: float = 0.35,
    distractor_frac: float = 0.08,
    progress: Optional[ProgressFn] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> Dataset:
    """Simulate ``approaches`` red-ball approaches and label every frame.

    With ``keep_all=False`` frames far from impact (> 1.5 s) are subsampled to
    keep the dataset compact; evaluation sets use ``keep_all=True``.
    """
    rng = np.random.default_rng(seed)
    b = _Builder()
    gid = 0
    chunk = 1000
    done = 0
    while done < approaches:
        if should_stop and should_stop():
            break
        m = min(chunk, approaches - done)
        ap = sample_approaches(rng, m, speed_range=speed_range)
        positions, t_imp = simulate_trajectories(ap, dt=SIM_DT, t_max=SIM_T_MAX)
        for i in range(m):
            fps = rng.uniform(30.0, 144.0)
            t_end = min(float(t_imp[i]), SIM_T_MAX)
            times = frame_times(rng, fps, t_end)
            if len(times) == 0:
                gid += 1
                continue
            pos = interpolate_positions(positions, i, times, SIM_DT)
            dets, char_h = detections_for_frames(
                rng,
                ap.camera(i),
                pos,
                float(ap.radius[i]),
                proc_scale=int(rng.choice([1, 2, 2, 3])),
                dropout=float(rng.uniform(0.0, 0.10)),
            )
            track = TrackState()
            speed = float(ap.speed[i])
            for t, det in zip(times, dets):
                track.update(float(t), det)
                if not track.active:
                    continue
                tti = float(t_imp[i]) - float(t)
                if not keep_all and tti > 1.5 and rng.random() > keep_far_prob:
                    continue
                feats = track.features(float(t), char_h)
                if feats is not None:
                    b.add(feats, tti, gid, track.heuristic_tti(), speed)
            gid += 1
        done += m
        if progress:
            progress(done / approaches, f"simulated {done}/{approaches} approaches")

    # Red things that are *not* the ball coming at you (decorations, effects...).
    n_distract = int(approaches * distractor_frac)
    for _ in range(n_distract):
        fps = rng.uniform(30.0, 144.0)
        times = frame_times(rng, fps, rng.uniform(0.3, 2.5))
        ang = rng.uniform(0, 2 * math.pi)
        dist = rng.uniform(0.08, 0.6)
        x0, y0 = dist * math.cos(ang), dist * math.sin(ang)
        r0 = math.exp(rng.uniform(math.log(0.004), math.log(0.05)))
        vx, vy = rng.normal(0, 0.05, 2)
        drop = rng.uniform(0.0, 0.2)
        char_h = rng.uniform(0.12, 0.6)
        track = TrackState()
        for t in times:
            det = None
            if rng.random() > drop:
                det = (
                    x0 + vx * t + rng.normal(0, 0.002),
                    y0 + vy * t + rng.normal(0, 0.002),
                    max(r0 * (1 + rng.normal(0, 0.05)), 0.002),
                )
            track.update(float(t), det)
            if track.active and (keep_all or rng.random() < 0.5):
                feats = track.features(float(t), char_h)
                if feats is not None:
                    b.add(feats, math.inf, gid, track.heuristic_tti(), 0.0)
        gid += 1
    return b.build()


def dataset_from_recordings(
    paths: Sequence[Path | str] | None = None, merge_gap_s: Optional[float] = None
) -> Dataset:
    """Hindsight-label recorded practice sessions (see :mod:`bladebot.recorder`)."""
    paths = list(paths) if paths is not None else list_recordings()
    b = _Builder()
    gid = 0
    for path in paths:
        header, frames = read_recording(path)
        if len(frames) < 10:
            continue
        use_gate = bool(header.get("gate")) and any(f.get("g") is not None for f in frames)
        gap = merge_gap_s if merge_gap_s is not None else (0.06 if use_gate else 0.15)
        dts = np.diff([f["t"] for f in frames])
        half_frame = float(np.median(dts)) / 2 if len(dts) else 0.0
        for first, last in iter_episodes(frames, use_gate, gap):
            if last >= len(frames) - 1:
                continue  # recording stopped mid-approach: we don't know the impact time
            t_start, t_last = frames[first]["t"], frames[last]["t"]
            if not (0.08 <= t_last - t_start <= 8.0):
                continue
            span = frames[first : last + 1]
            if sum(f.get("d") is not None for f in span) < 3:
                continue
            t_impact = t_last + half_frame
            char_h = float(header.get("char_h") or DEFAULT_CHAR_H)
            track = TrackState()
            for fr in span:
                t = float(fr["t"])
                track.update(t, fr.get("d"))
                if track.active:
                    feats = track.features(t, char_h)
                    if feats is not None:
                        b.add(feats, t_impact - t, gid, track.heuristic_tti(), 0.0)
            gid += 1
    return b.build()


# =============================================================== training
@dataclass
class TrainConfig:
    hidden: tuple[int, ...] = (64, 64)
    activation: str = "tanh"
    epochs: int = 30
    batch_size: int = 1024
    lr: float = 3e-3
    weight_decay: float = 1e-5
    seed: int = 0
    near_weight: float = 2.0  # extra weight for frames close to impact
    patience: int = 6


def sample_weights(tti: np.ndarray, near_weight: float) -> np.ndarray:
    t = np.where(np.isfinite(tti), np.maximum(tti, 0.0), 10.0)
    return (1.0 + near_weight * np.exp(-t / 0.6)).astype(np.float32)


def train_network(
    train: Dataset,
    val: Dataset,
    cfg: TrainConfig = TrainConfig(),
    progress: Optional[ProgressFn] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    extra_weights: Optional[np.ndarray] = None,
) -> tuple[ParryNet, list[dict[str, float]]]:
    """Mini-batch Adam training with early stopping on the validation loss."""
    rng = np.random.default_rng(cfg.seed)
    mean = train.X.mean(axis=0)
    std = np.maximum(train.X.std(axis=0), 1e-3)
    Xtr = ((train.X - mean) / std).astype(np.float32)
    Ytr = train.labels()
    Wtr = sample_weights(train.tti, cfg.near_weight)
    if extra_weights is not None:
        Wtr = Wtr * extra_weights.astype(np.float32)
    Xva = ((val.X - mean) / std).astype(np.float32)
    Yva = val.labels()
    Wva = sample_weights(val.tti, cfg.near_weight)

    net = MLP([N_FEATURES, *cfg.hidden, len(HORIZONS)], cfg.activation, seed=cfg.seed)
    # Start the output biases at the base rates: faster, more stable training.
    base = np.clip(Ytr.mean(axis=0), 1e-3, 1 - 1e-3)
    net.biases[-1][:] = np.log(base / (1 - base)).astype(np.float32)
    opt = Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    n = len(Xtr)
    steps_per_epoch = max(1, math.ceil(n / cfg.batch_size))
    total_steps = steps_per_epoch * cfg.epochs
    step = 0
    best = (math.inf, net.copy(), 0)
    history: list[dict[str, float]] = []
    for epoch in range(cfg.epochs):
        if should_stop and should_stop():
            break
        perm = rng.permutation(n)
        running = 0.0
        for s in range(steps_per_epoch):
            idx = perm[s * cfg.batch_size : (s + 1) * cfg.batch_size]
            logits = net.forward(Xtr[idx], train=True)
            loss, grad = bce_with_logits(logits, Ytr[idx], Wtr[idx])
            gw, gb = net.backward(grad)
            lr = cfg.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * step / total_steps)))
            opt.step([*gw, *gb], lr=lr)
            running += loss
            step += 1
        train_loss = running / steps_per_epoch
        val_loss = _eval_loss(net, Xva, Yva, Wva)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best[0] - 1e-5:
            best = (val_loss, net.copy(), epoch)
        if progress:
            progress((epoch + 1) / cfg.epochs, f"epoch {epoch + 1}/{cfg.epochs}  train {train_loss:.4f}  val {val_loss:.4f}")
        if epoch - best[2] >= cfg.patience:
            break
    model = ParryNet(best[1], mean, std, HORIZONS)
    return model, history


def _eval_loss(net: MLP, X: np.ndarray, Y: np.ndarray, W: np.ndarray, batch: int = 8192) -> float:
    if len(X) == 0:
        return math.nan
    total = 0.0
    wsum = 0.0
    for s in range(0, len(X), batch):
        logits = net.forward(X[s : s + batch])
        loss, _ = bce_with_logits(logits, Y[s : s + batch], W[s : s + batch])
        w = float(W[s : s + batch].sum())
        total += loss * w
        wsum += w
    return total / max(wsum, 1e-9)


# ============================================================= evaluation
def simulate_parries(
    ds: Dataset,
    press: np.ndarray,
    latency: float = 0.06,
    window: float = PARRY_WINDOW_S,
) -> dict[str, float]:
    """Replay each approach and check whether the *first* press would have worked.

    ``press`` is a boolean per frame ("the policy wants to parry now"). A press
    at a frame with true time-to-impact ``tti`` succeeds when the shield, which
    goes up ``latency`` seconds later and lasts ``window`` seconds, covers the
    impact: ``latency <= tti <= latency + window``.
    """
    if len(ds) == 0:
        return {"success": math.nan, "approaches": 0, "false_press": math.nan}
    starts = np.flatnonzero(np.r_[True, ds.group[1:] != ds.group[:-1]])
    ends = np.r_[starts[1:], len(ds)]
    hits = ok = early = late = never = decoys = decoy_pressed = 0
    for s, e in zip(starts, ends):
        tti = ds.tti[s:e]
        p = press[s:e]
        first = int(np.argmax(p)) if p.any() else -1
        if not np.isfinite(tti[0]):
            decoys += 1
            decoy_pressed += int(first >= 0)
            continue
        hits += 1
        if first < 0:
            never += 1
            continue
        t = float(tti[first])
        if t < latency:
            late += 1
        elif t > latency + window:
            early += 1
        else:
            ok += 1
    return {
        "success": ok / max(hits, 1),
        "too_early": early / max(hits, 1),
        "too_late": late / max(hits, 1),
        "never_pressed": never / max(hits, 1),
        "approaches": hits,
        "false_press": decoy_pressed / max(decoys, 1),
    }


def debounce(press: np.ndarray, group: np.ndarray, frames: int) -> np.ndarray:
    """Only keep presses whose condition held for ``frames`` consecutive frames."""
    out = press.astype(bool).copy()
    same = np.r_[False, group[1:] == group[:-1]]
    run = press.astype(bool)
    for _ in range(max(frames, 1) - 1):
        run = np.r_[False, run[:-1]] & same
        out &= run
        same = same & np.r_[False, same[:-1]]
    return out


def evaluate(
    model: ParryNet,
    ds: Dataset,
    lead: float = 0.30,
    confidence: float = 0.6,
    latency: float = 0.06,
    confirm_frames: int = 2,
    instant_confidence: float = 0.85,
) -> dict[str, Any]:
    """Parry success of the network vs. the classic looming heuristic.

    Uses the same decision rule as the live bot: parry when the probability of
    impact within ``lead`` seconds stays above ``confidence`` for
    ``confirm_frames`` frames in a row, or immediately above ``instant_confidence``
    (see :meth:`ParryNet.press_probability` for hidden balls).
    """
    probs = model.predict(ds.X) if len(ds) else np.zeros((0, len(HORIZONS)))
    stale = ds.X[:, STALE_INDEX] if len(ds) else np.zeros(0)
    p_lead = np.array([model.press_probability(p, lead, s) for p, s in zip(probs, stale)]) if len(ds) else np.zeros(0)
    nn_press = debounce(p_lead >= confidence, ds.group, confirm_frames) | (p_lead >= instant_confidence)
    nn = simulate_parries(ds, nn_press, latency)
    heur = simulate_parries(ds, debounce(ds.heur <= lead, ds.group, confirm_frames), latency)
    # timing error of the expected time-to-impact on frames that matter
    near = np.isfinite(ds.tti) & (ds.tti >= 0) & (ds.tti <= 0.8)
    if near.any():
        eta = np.array([model.expected_tti(p) for p in probs[near]])
        mae = float(np.mean(np.abs(eta - ds.tti[near])))
    else:
        mae = math.nan
    by_speed: dict[str, float] = {}
    for lo, hi in ((0, 60), (60, 150), (150, 1e9)):
        sel = (ds.speed >= lo) & (ds.speed < hi) & np.isfinite(ds.tti)
        if sel.any():
            sub = Dataset(ds.X[sel], ds.tti[sel], ds.group[sel], ds.heur[sel], ds.speed[sel])
            label = f"{lo}-{int(hi)}" if hi < 1e9 else f"{lo}+"
            by_speed[label] = simulate_parries(sub, nn_press[sel], latency)["success"]
    return {
        "lead_s": lead,
        "confidence": confidence,
        "latency_s": latency,
        "confirm_frames": confirm_frames,
        "instant_confidence": instant_confidence,
        "nn": nn,
        "heuristic": heur,
        "eta_mae_s": mae,
        "nn_success_by_speed": by_speed,
    }


# ================================================================ pipeline
@dataclass
class PipelineConfig:
    approaches: int = 12000
    val_approaches: int = 2000
    seed: int = 0
    speed_min: float = 20.0
    speed_max: float = 350.0
    use_recordings: bool = False
    recording_weight: float = 3.0
    train: TrainConfig = field(default_factory=TrainConfig)


def run_pipeline(
    cfg: PipelineConfig,
    out_path: Path | str,
    progress: Optional[ProgressFn] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    log: Callable[[str], None] = print,
) -> tuple[ParryNet, dict[str, Any]]:
    """Generate data, train, evaluate and save a model. Returns ``(model, report)``."""

    def stage(lo: float, hi: float) -> Optional[ProgressFn]:
        if progress is None:
            return None
        return lambda f, msg: progress(lo + (hi - lo) * f, msg)

    t0 = time.time()
    speed_range = (cfg.speed_min, cfg.speed_max)
    log(f"Simulating {cfg.approaches} training approaches...")
    train = generate_sim_dataset(cfg.approaches, cfg.seed, speed_range, progress=stage(0.0, 0.35), should_stop=should_stop)
    log(f"  {len(train):,} training frames")
    log(f"Simulating {cfg.val_approaches} validation approaches...")
    val = generate_sim_dataset(cfg.val_approaches, cfg.seed + 7919, speed_range, keep_all=True, progress=stage(0.35, 0.45), should_stop=should_stop)
    log(f"  {len(val):,} validation frames")

    extra = None
    rec_frames = 0
    if cfg.use_recordings:
        rec = dataset_from_recordings()
        rec_frames = len(rec)
        log(f"Recordings: {rec_frames:,} labelled frames")
        if rec_frames:
            n_sim = len(train)
            train = Dataset.concat([train, rec])
            extra = np.ones(len(train), dtype=np.float32)
            extra[n_sim:] = cfg.recording_weight

    if should_stop and should_stop():
        raise RuntimeError("training cancelled")
    log(f"Training {cfg.train.hidden} {cfg.train.activation} network for up to {cfg.train.epochs} epochs...")
    model, history = train_network(train, val, cfg.train, progress=stage(0.45, 0.95), should_stop=should_stop, extra_weights=extra)
    if should_stop and should_stop():
        raise RuntimeError("training cancelled")
    report = evaluate(model, val)
    report["history"] = history
    report["train_frames"] = len(train)
    report["recording_frames"] = rec_frames
    report["seconds"] = round(time.time() - t0, 1)
    model.save(
        out_path,
        {
            "created": datetime.now().isoformat(timespec="seconds"),
            "trained_on": {
                "approaches": cfg.approaches,
                "speed_range": list(speed_range),
                "recording_frames": rec_frames,
                "seed": cfg.seed,
            },
            "metrics": {
                "nn_success": report["nn"]["success"],
                "heuristic_success": report["heuristic"]["success"],
                "eta_mae_s": report["eta_mae_s"],
                "val_loss": min(h["val_loss"] for h in history) if history else None,
            },
        },
    )
    if progress:
        progress(1.0, "done")
    log(format_report(report))
    log(f"Saved model to {out_path} ({report['seconds']} s)")
    return model, report


def format_report(report: dict[str, Any]) -> str:
    nn, h = report["nn"], report["heuristic"]
    lines = [
        f"Parry success in simulation (lead {report['lead_s']*1000:.0f} ms, "
        f"confidence {report['confidence']:.2f}, latency {report['latency_s']*1000:.0f} ms):",
        f"  neural network : {nn['success']*100:5.1f}%  (early {nn['too_early']*100:.1f}%, late {nn['too_late']*100:.1f}%, "
        f"never {nn['never_pressed']*100:.1f}%, false presses on decoys {nn['false_press']*100:.1f}%)",
        f"  classic looming: {h['success']*100:5.1f}%  (early {h['too_early']*100:.1f}%, late {h['too_late']*100:.1f}%, "
        f"never {h['never_pressed']*100:.1f}%, false presses on decoys {h['false_press']*100:.1f}%)",
        f"  time-to-impact error (frames < 0.8 s out): {report['eta_mae_s']*1000:.0f} ms",
    ]
    if report.get("nn_success_by_speed"):
        parts = ", ".join(f"{k} studs/s: {v*100:.0f}%" for k, v in report["nn_success_by_speed"].items())
        lines.append(f"  network success by ball speed: {parts}")
    return "\n".join(lines)


# ============================================================ background job
class TrainingJob:
    """Runs :func:`run_pipeline` on a background thread for the web menu."""

    def __init__(self, cfg: PipelineConfig, out_path: Path | str, on_done: Optional[Callable[[ParryNet, Path], None]] = None) -> None:
        self.cfg = cfg
        self.out_path = Path(out_path)
        self.on_done = on_done
        self.progress = 0.0
        self.message = "starting"
        self.log_lines: list[str] = []
        self.report: Optional[dict[str, Any]] = None
        self.error: Optional[str] = None
        self.state = "running"
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="bladebot-training", daemon=True)
        self._thread.start()

    def _log(self, line: str) -> None:
        for part in str(line).splitlines():
            self.log_lines.append(part)
        del self.log_lines[:-200]

    def _progress(self, frac: float, msg: str) -> None:
        self.progress = float(min(max(frac, 0.0), 1.0))
        self.message = msg

    def _run(self) -> None:
        try:
            model, report = run_pipeline(self.cfg, self.out_path, self._progress, self._stop.is_set, self._log)
            self.report = report
            self.state = "done"
            if self.on_done:
                self.on_done(model, self.out_path)
        except Exception as exc:  # surfaced in the UI
            self.error = str(exc)
            self.state = "cancelled" if self._stop.is_set() else "error"
            self._log(f"Training stopped: {exc}")

    def cancel(self) -> None:
        self._stop.set()

    @property
    def running(self) -> bool:
        return self.state == "running"

    def status(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "progress": self.progress,
            "message": self.message,
            "log": self.log_lines[-60:],
            "error": self.error,
            "report": _jsonable(self.report),
            "out_path": str(self.out_path),
        }


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


# ==================================================================== CLI
def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Train BladeBot's parry network (practice use only).")
    ap.add_argument("--approaches", type=int, default=20000, help="simulated training approaches")
    ap.add_argument("--val-approaches", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--hidden", type=str, default="64,64", help="hidden layer sizes, e.g. 64,64")
    ap.add_argument("--activation", choices=["tanh", "relu", "leaky_relu"], default="tanh")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--speed-min", type=float, default=20.0)
    ap.add_argument("--speed-max", type=float, default=350.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--recordings", action="store_true", help="also learn from recordings/ (hindsight labels)")
    ap.add_argument("--out", type=Path, default=DEFAULT_MODEL_PATH)
    args = ap.parse_args(argv)
    cfg = PipelineConfig(
        approaches=args.approaches,
        val_approaches=args.val_approaches,
        seed=args.seed,
        speed_min=args.speed_min,
        speed_max=args.speed_max,
        use_recordings=args.recordings,
        train=TrainConfig(
            hidden=tuple(int(v) for v in args.hidden.split(",") if v.strip()),
            activation=args.activation,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
        ),
    )

    last = [0.0]

    def progress(frac: float, msg: str) -> None:
        if frac - last[0] >= 0.05 or frac >= 1.0:
            last[0] = frac
            print(f"[{frac*100:5.1f}%] {msg}", flush=True)

    run_pipeline(cfg, args.out, progress=progress)


if __name__ == "__main__":
    main()
