"""The bot itself: capture -> detect -> track -> neural network -> parry.

:class:`FramePipeline` is the pure per-frame logic (also used by the tests and
the arena benchmark); :class:`BotEngine` runs it on a background thread, talks
to the web menu and presses the parry input.
"""

from __future__ import annotations

import math
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

from . import __version__
from .capture import CaptureError, ScreenSource
from .config import ALL_SETTINGS, Settings
from .features import TrackState
from .model import PROJECT_DIR, STALE_INDEX, ParryNet
from .pngenc import encode_png
from .recorder import Recorder, list_recordings
from .sim.arena import Arena, ArenaSource
from .vision import BallDetector, Detection, VisionSettings, render_preview, rgb_to_hsv_pixel


# ====================================================================== logic
class ParryDecider:
    """Turns network outputs into (at most) one parry per incoming ball."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.episode_active = False
        self.last_target = -math.inf
        self.pressed = False
        self.last_press = -math.inf
        self.confirm = 0
        self.presses = 0

    def update(self, t: float, targeted: bool, p_lead: float, cfg: dict[str, Any]) -> bool:
        if targeted:
            if not self.episode_active:
                self.episode_active = True
                self.pressed = False
                self.confirm = 0
            self.last_target = t
        elif self.episode_active and t - self.last_target > cfg["rearm_ms"] / 1000.0:
            self.episode_active = False
            self.pressed = False
        if not self.episode_active:
            self.confirm = 0
            return False
        self.confirm = self.confirm + 1 if p_lead >= cfg["confidence"] else 0
        if not (self.confirm >= cfg["confirm_frames"] or p_lead >= cfg["instant_confidence"]):
            return False
        since = t - self.last_press
        if since < cfg["min_interval_ms"] / 1000.0:
            return False
        if self.pressed and since < cfg["retry_ms"] / 1000.0:
            return False
        self.pressed = True
        self.last_press = t
        self.presses += 1
        return True


@dataclass
class FrameResult:
    t: float
    det: Detection
    targeted: bool
    features: Optional[list[float]]
    probs: Optional[np.ndarray]
    p_lead: float
    eta: Optional[float]
    heuristic_tti: Optional[float]
    press: bool


class FramePipeline:
    def __init__(self, model: Optional[ParryNet]) -> None:
        self.model = model
        self.detector = BallDetector()
        self.tracker = TrackState()
        self.decider = ParryDecider()

    def reset(self) -> None:
        self.detector.reset()
        self.tracker.reset()
        self.decider.reset()

    def process(
        self,
        frame: np.ndarray,
        t: float,
        vision: VisionSettings,
        cfg: dict[str, Any],
        keep_mask: bool = False,
    ) -> FrameResult:
        self.detector.settings = vision
        det = self.detector.detect(frame, keep_mask=keep_mask)
        if vision.gate_enabled:
            targeted = det.targeted
            ball = det.ball if targeted else None
        else:
            ball = det.ball
            targeted = ball is not None or self.tracker.active
        self.tracker.update(t, ball)
        feats = self.tracker.features(t, vision.gate_h) if self.tracker.active else None
        probs = None
        p_lead = 0.0
        eta = None
        if feats is not None and self.model is not None:
            probs = self.model.predict(np.asarray(feats, dtype=np.float32))
            p_lead = self.model.press_probability(probs, cfg["lead_ms"] / 1000.0, feats[STALE_INDEX])
            eta = self.model.expected_tti(probs)
        heur = self.tracker.heuristic_tti() if feats is not None else None
        press = self.decider.update(t, targeted, p_lead, cfg)
        return FrameResult(t, det, targeted, feats, probs, p_lead, eta, heur, press)


def default_config(**overrides: Any) -> dict[str, Any]:
    cfg = {k: s.default for k, s in ALL_SETTINGS.items()}
    cfg.update(overrides)
    return cfg


def run_arena_benchmark(
    model: ParryNet,
    seconds: float = 60.0,
    fps: float = 60.0,
    seed: int = 0,
    **overrides: Any,
) -> dict[str, Any]:
    """Let the network play the arena headlessly (real renderer + vision)."""
    cfg = default_config(**overrides)
    arena = Arena(seed=seed)
    arena.configure(cfg)
    pipe = FramePipeline(model)
    dt = 1.0 / fps
    for _ in range(int(seconds * fps)):
        res = pipe.process(arena.frame(), arena.t, arena.vision_settings(), cfg)
        if res.press:
            arena.request_parry("bot")
        arena.step(dt)
    st = dict(arena.stats)
    decided = st["blocks"] + st["hits"]
    st["block_rate"] = st["blocks"] / decided if decided else math.nan
    return st


# ===================================================================== engine
def _beep(on: bool) -> None:
    if sys.platform != "win32":
        return
    try:
        import winsound

        threading.Thread(target=winsound.Beep, args=(1320 if on else 660, 110), daemon=True).start()
    except Exception:
        pass


def _clean(v: Any) -> Any:
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


class BotEngine:
    def __init__(
        self,
        settings: Settings,
        controller: Any = None,
        model: Optional[ParryNet] = None,
        autoload_model: bool = True,
    ) -> None:
        self.settings = settings
        self.enabled = False
        self.model: Optional[ParryNet] = model
        self.model_error: Optional[str] = None
        self.model_path: Optional[str] = None
        self.pipeline = FramePipeline(self.model)
        self.arena = Arena()
        self.arena_source = ArenaSource(self.arena)
        self.screen: Optional[ScreenSource] = None
        if controller is None:
            from .controller import ParryController

            controller = ParryController()
        self.controller = controller
        self.hotkey_error: Optional[str] = None
        self.events: deque[dict[str, Any]] = deque(maxlen=60)
        self.recorder: Optional[Recorder] = None
        self.capture_error: Optional[str] = None
        self.parries = 0
        self.frames = 0
        self._last_parry_wall: Optional[float] = None
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._preview_until = 0.0
        self._preview_png: Optional[bytes] = None
        self._preview_time = 0.0
        self._last_frame: Optional[np.ndarray] = None
        self._status: dict[str, Any] = {}
        self._fps = 0.0
        self._proc_ms = 0.0
        self._would_press_until = 0.0
        self._human_presses: deque[float] = deque(maxlen=8)
        if model is None and autoload_model:
            self.load_model()

    # ------------------------------------------------------------ lifecycle
    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="bladebot-engine", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.stop_recording()

    # ------------------------------------------------------------ logging
    def log(self, kind: str, text: str) -> None:
        self.events.append({"time": datetime.now().strftime("%H:%M:%S"), "kind": kind, "text": text})

    # ------------------------------------------------------------ model
    def _resolve(self, path: str) -> Path:
        p = Path(path)
        return p if p.is_absolute() else PROJECT_DIR / p

    def load_model(self, path: Optional[str] = None) -> bool:
        path = path or self.settings.get("model_path")
        try:
            model = ParryNet.load(self._resolve(path))
        except FileNotFoundError:
            self.model_error = f"model file not found: {path} (train one in the Train tab)"
            return False
        except Exception as exc:
            self.model_error = f"could not load {path}: {exc}"
            return False
        self.set_model(model, path)
        return True

    def set_model(self, model: ParryNet, path: Optional[str | Path] = None) -> None:
        with self._lock:
            self.model = model
            self.model_error = None
            if path is not None:
                p = Path(path)
                try:
                    rel = str(p.resolve().relative_to(PROJECT_DIR)) if p.is_absolute() else str(p)
                except ValueError:
                    rel = str(p)
                self.model_path = rel.replace("\\", "/")
                if self.settings.get("model_path") != self.model_path:
                    self.settings.update({"model_path": self.model_path})
            if hasattr(self, "pipeline"):
                self.pipeline.model = model
                self.pipeline.reset()

    # ------------------------------------------------------------ control
    def set_enabled(self, on: bool, via: str = "menu") -> tuple[bool, str]:
        cfg = self.settings.snapshot()
        if on:
            if self.model is None:
                return False, "No neural network loaded - train one in the Train tab first."
            if cfg["source"] == "screen" and not cfg["practice_ack"]:
                return False, "Accept the practice-only notice in the menu before using the bot on Roblox."
        with self._lock:
            changed = self.enabled != on
            self.enabled = on
            self.pipeline.decider.reset()
        if changed:
            where = "practice arena" if cfg["source"] == "arena" else "Roblox (screen)"
            self.log("info", f"Bot switched {'ON' if on else 'OFF'} via {via} - source: {where}")
        return True, ""

    def toggle_from_hotkey(self) -> None:
        ok, msg = self.set_enabled(not self.enabled, via="hotkey")
        if not ok:
            self.log("warn", msg)
        if self.settings.get("beep"):
            _beep(self.enabled if ok else False)

    def human_parry(self) -> None:
        """Block in the arena ("I play" mode). Applied by the engine thread on its next frame."""
        self._human_presses.append(time.perf_counter())

    def reset_arena(self) -> None:
        with self._lock:
            self.arena = Arena()
            self.arena.configure(self.settings.snapshot())
            self.arena_source = ArenaSource(self.arena)
            self.pipeline.reset()

    # ------------------------------------------------------------ preview / calibration
    def preview(self) -> Optional[bytes]:
        self._preview_until = time.perf_counter() + 1.5
        return self._preview_png

    def pick_color(self, x: float, y: float) -> dict[str, float]:
        frame = self._last_frame
        if frame is None:
            raise ValueError("no frame yet - open the preview first")
        h, w = frame.shape[:2]
        cx = int(min(max(x, 0.0), 0.999) * w)
        cy = int(min(max(y, 0.0), 0.999) * h)
        patch = np.asarray(frame[max(cy - 1, 0) : cy + 2, max(cx - 1, 0) : cx + 2, :3], dtype=np.float32)
        r, g, b = (int(v) for v in patch.reshape(-1, 3).mean(axis=0))
        hue, sat, val = rgb_to_hsv_pixel(r, g, b)
        changes = {
            "hue_center": round(hue, 1),
            "sat_min": round(max(0.15, sat * 0.65), 2),
            "val_min": round(max(0.15, val * 0.6), 2),
        }
        self.settings.update(changes)
        self.log("info", f"Ball colour picked: RGB({r},{g},{b}) -> hue {hue:.0f} deg, S {sat:.2f}, V {val:.2f}")
        return {**changes, "rgb": [r, g, b]}

    def set_anchor(self, x: float, y: float) -> dict[str, float]:
        changes = {"anchor_x": round(min(max(x, 0.0), 1.0), 3), "anchor_y": round(min(max(y, 0.0), 1.0), 3)}
        self.settings.update(changes)
        return changes

    # ------------------------------------------------------------ recording
    def start_recording(self) -> str:
        with self._lock:
            if self.recorder is None:
                cfg = self.settings.snapshot()
                if cfg["source"] == "arena":
                    gate, char_h = True, self.arena.vision_settings().gate_h
                else:
                    gate, char_h = bool(cfg["gate_enabled"]), float(cfg["gate_h"])
                self.recorder = Recorder(cfg["source"], gate, char_h)
                self.log("info", f"Recording to {self.recorder.path.name}")
            return str(self.recorder.path)

    def stop_recording(self) -> None:
        with self._lock:
            if self.recorder is not None:
                self.recorder.close()
                self.log("info", f"Saved recording ({self.recorder.frames} frames)")
                self.recorder = None

    # ------------------------------------------------------------ main loop
    def _run(self) -> None:
        version = -1
        cfg: dict[str, Any] = {}
        user_vision = VisionSettings()
        prev_source: Optional[str] = None
        next_frame = time.perf_counter()
        last_tick: Optional[float] = None
        while not self._stop.is_set():
            if self.settings.version != version:
                version = self.settings.version
                cfg = self.settings.snapshot()
                user_vision = VisionSettings.from_mapping(cfg)
            source = cfg["source"]
            if source != prev_source:
                prev_source = source
                self.pipeline.reset()
                self.arena_source.pause()
                if source != "screen" and self.screen is not None:
                    self.screen.close()
                    self.screen = None
            now = time.perf_counter()
            wants_preview = now < self._preview_until
            active = self.enabled or wants_preview or self.recorder is not None
            if not active:
                self.arena_source.pause()
                self._human_presses.clear()
                last_tick = None
                self._fps = 0.0
                self._publish(cfg, None, idle=True)
                time.sleep(0.1)
                next_frame = time.perf_counter()
                continue
            try:
                if source == "arena":
                    while self._human_presses:
                        self._human_presses.popleft()
                        self.arena.request_parry("human")
                    frame, t = self.arena_source.grab(cfg)
                    vision = self.arena.vision_settings()
                else:
                    if self.screen is None:
                        self.screen = ScreenSource()
                    frame, t = self.screen.grab(cfg)
                    vision = user_vision
                self.capture_error = None
            except CaptureError as exc:
                if self.capture_error != str(exc):
                    self.log("error", str(exc))
                self.capture_error = str(exc)
                self._publish(cfg, None)
                if self.screen is not None:
                    self.screen.close()
                    self.screen = None
                time.sleep(0.5)
                continue
            t0 = time.perf_counter()
            self._last_frame = frame
            res = self.pipeline.process(frame, t, vision, cfg, keep_mask=wants_preview)
            if res.press:
                self._handle_press(res, cfg, source)
            rec = self.recorder
            if rec is not None:
                rec.write(t, res.det.ball, res.det.targeted if vision.gate_enabled else None)
            self.frames += 1
            if wants_preview and t0 - self._preview_time >= 0.08:
                self._preview_time = t0
                self._preview_png = encode_png(render_preview(frame, res.det, active=res.targeted))
            proc = (time.perf_counter() - t0) * 1000.0
            self._proc_ms = proc if self._proc_ms == 0 else 0.9 * self._proc_ms + 0.1 * proc
            if last_tick is not None:
                inst = 1.0 / max(t0 - last_tick, 1e-4)
                self._fps = inst if self._fps == 0 else 0.93 * self._fps + 0.07 * inst
            last_tick = t0
            self._publish(cfg, res)
            max_fps = float(cfg["max_fps"])
            if source == "arena":
                max_fps = min(max_fps, 60.0)
            next_frame += 1.0 / max_fps
            delay = next_frame - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_frame = time.perf_counter()

    def _handle_press(self, res: FrameResult, cfg: dict[str, Any], source: str) -> None:
        detail = f"p={res.p_lead:.2f}, ETA {res.eta * 1000:.0f} ms" if res.eta is not None else f"p={res.p_lead:.2f}"
        if source == "arena" and cfg["arena_mode"] == "human":
            self.arena.note_ghost()
            self._would_press_until = time.perf_counter() + 0.35
            return
        if not self.enabled:
            self._would_press_until = time.perf_counter() + 0.35
            return
        if cfg["dry_run"]:
            self._would_press_until = time.perf_counter() + 0.35
            self.log("parry", f"Would parry now (observe only) - {detail}")
            return
        if source == "arena":
            self.arena.request_parry("bot")
        else:
            hold = cfg["key_hold_ms"] / 1000.0
            if not self.controller.press(cfg["parry_input"], cfg["parry_key"], hold):
                err = getattr(self.controller, "backend_error", None) or "input queue full"
                self.log("error", f"Could not press the parry input: {err}")
                return
        self.parries += 1
        self._last_parry_wall = time.time()
        if source == "arena":
            what = "arena block"
        else:
            what = "left click" if cfg["parry_input"] == "mouse" else f"key {cfg['parry_key'].upper()}"
        self.log("parry", f"Parry ({what}) - {detail}")

    # ------------------------------------------------------------ status
    def _publish(self, cfg: dict[str, Any], res: Optional[FrameResult], idle: bool = False) -> None:
        nn: dict[str, Any] = {"probs": None, "p_lead": 0.0, "eta": None, "heuristic": None}
        vision: dict[str, Any] = {"targeted": False, "gate_frac": 0.0, "ball": None, "red_frac": 0.0, "blobs": 0}
        if res is not None:
            d = res.det
            vision = {
                "targeted": bool(res.targeted),
                "gate_frac": round(d.gate_frac, 4),
                "ball": [round(v, 4) for v in d.ball] if d.ball else None,
                "red_frac": round(d.red_frac, 4),
                "blobs": len(d.blobs),
                "size": [d.shape[1], d.shape[0]],
            }
            nn = {
                "probs": [round(float(p), 4) for p in res.probs] if res.probs is not None else None,
                "p_lead": round(res.p_lead, 4),
                "eta": _clean(round(res.eta, 4)) if res.eta is not None else None,
                "heuristic": _clean(round(res.heuristic_tti, 4)) if res.heuristic_tti is not None else None,
                "tracking": res.features is not None,
            }
        ctrl = self.controller
        status = {
            "version": __version__,
            "enabled": self.enabled,
            "idle": idle,
            "source": cfg.get("source"),
            "arena_mode": cfg.get("arena_mode"),
            "dry_run": cfg.get("dry_run"),
            "practice_ack": cfg.get("practice_ack"),
            "fps": round(self._fps, 1),
            "proc_ms": round(self._proc_ms, 2),
            "frames": self.frames,
            "capture_error": self.capture_error,
            "region": self.screen.region if self.screen is not None else None,
            "monitors": self.screen.monitor_count if self.screen is not None else None,
            "vision": vision,
            "nn": nn,
            "horizons": [float(h) for h in self.model.horizons] if self.model is not None else None,
            "decision": {
                "episode": self.pipeline.decider.episode_active,
                "pressed": self.pipeline.decider.pressed,
                "would_press": time.perf_counter() < self._would_press_until,
                "parries": self.parries,
                "last_parry_ago": round(time.time() - self._last_parry_wall, 2) if self._last_parry_wall else None,
            },
            "model": {
                "loaded": self.model is not None,
                "path": self.model_path or cfg.get("model_path"),
                "error": self.model_error,
            },
            "input": {
                "backend": getattr(getattr(ctrl, "backend", None), "name", None) if not hasattr(ctrl, "callback") else "arena",
                "error": getattr(ctrl, "backend_error", None),
                "last_error": getattr(ctrl, "last_error", None),
            },
            "hotkey": {"key": cfg.get("hotkey"), "error": self.hotkey_error},
            "events": list(self.events)[-16:],
            "recording": {
                "active": self.recorder is not None,
                "frames": self.recorder.frames if self.recorder is not None else 0,
                "path": str(self.recorder.path.name) if self.recorder is not None else None,
                "files": self._recording_count(),
            },
            "arena": self.arena.state() if cfg.get("source") == "arena" else None,
        }
        self._status = status

    def _recording_count(self) -> int:
        now = time.monotonic()
        if now - getattr(self, "_rec_count_time", -10.0) > 2.0:
            self._rec_count_time = now
            self._rec_count = len(list_recordings())
        return self._rec_count

    def status(self) -> dict[str, Any]:
        if not self._status:
            self._publish(self.settings.snapshot(), None, idle=True)
        st = dict(self._status)
        st["enabled"] = self.enabled
        return st

    def model_info(self) -> dict[str, Any]:
        if self.model is None:
            return {"loaded": False, "error": self.model_error}
        info = self.model.summary()
        info.update(loaded=True, path=self.model_path, error=None)
        return info
