"""Practice arena: an endless Blade Ball-style rally against simulated players.

The ball homes in on you (red), you (or the neural network) block, it flies
to another player (who glows red instead), comes back a little faster, and so
on until someone misses. Blocking follows the real rules: the shield lasts
500 ms and blocking too early costs a 2 second cooldown.

The other players can curve the ball (sideways, high or backwards first, see
:mod:`bladebot.sim.world`) - set "Curve balls" to Off / Some / Lots.

In "Me" mode the arena gives timing feedback on every block ("too early by
120 ms") and tells you when the network would have pressed - a safe way to
practise your own parry timing.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from ..vision import VisionSettings
from .render import Renderer
from .world import CURVE_NAMES, HIT_RADIUS, PARRY_WINDOW_S, TORSO, WHIFF_COOLDOWN_S, Camera, OneBall, sample_launch

ARENA_DEFAULTS: dict[str, Any] = {
    "arena_ping_ms": 60,
    "arena_speed": 45.0,
    "arena_speed_growth": 1.07,
    "arena_speed_max": 300.0,
    "arena_camera_distance": 16.0,
    "arena_camera_pitch": 20.0,
    "arena_decoys": True,
    "arena_curves": "some",
}
# curve type mix (see CURVE_NAMES) and steering-law mix for each "Curve balls" setting
CURVE_SETTINGS: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = {
    "off": ((1.0, 0.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
    "some": ((0.55, 0.17, 0.1, 0.12, 0.06), (0.35, 0.35, 0.3)),
    "lots": ((0.25, 0.25, 0.18, 0.22, 0.1), (0.35, 0.35, 0.3)),
}
SUBSTEP_S = 0.004


@dataclass
class _Press:
    t_register: float
    t_pressed: float
    who: str
    tti_at_press: Optional[float]


def _steer(pos: np.ndarray, direction: np.ndarray, target: np.ndarray, speed: float, homing: float, dt: float) -> tuple[np.ndarray, np.ndarray]:
    to_t = target - pos
    dist = float(np.linalg.norm(to_t))
    desired = to_t / max(dist, 1e-6)
    k = min(homing * speed * dt / max(dist, 2.0), 1.0)
    new_dir = direction + k * (desired - direction)
    n = float(np.linalg.norm(new_dir))
    new_dir = new_dir / n if n > 1e-6 else desired
    return pos + new_dir * speed * dt, new_dir


class Arena:
    def __init__(self, width: int = 640, height: int = 360, seed: Optional[int] = None) -> None:
        self.renderer = Renderer(width, height)
        self.rng = np.random.default_rng(seed)
        self.cfg: dict[str, Any] = dict(ARENA_DEFAULTS)
        self.t = 0.0
        self.stats = {"blocks": 0, "hits": 0, "whiffs": 0, "streak": 0, "best_streak": 0, "rounds": 0, "top_speed": 0.0}
        self.events: deque[dict[str, Any]] = deque(maxlen=40)
        self.feedback: Optional[dict[str, Any]] = None
        self.ball_radius = 1.2
        self.ball_pos = np.array([0.0, 4.0, 20.0])
        self.ball_dir = np.array([0.0, 0.0, -1.0])
        self.ball: Optional[OneBall] = None  # the incoming ball's flight
        self.t_hit: Optional[float] = None  # flight time at which it will reach you
        self.curve = "direct"
        self._acc = 0.0
        self.speed = float(self.cfg["arena_speed"])
        self.shield: Optional[tuple[float, float]] = None
        self.shield_owner: Optional[str] = None
        self.shield_press: Optional[_Press] = None
        self.cooldown_until = 0.0
        self.pending: list[_Press] = []
        self.opponents: list[np.ndarray] = []
        self.target_opp = 0
        self.ghost: Optional[tuple[float, Optional[float]]] = None
        self.phase = "countdown"
        self.phase_end = 0.0
        self._cam_yaw = 0.0
        self.cam = self._make_camera()
        self._new_round()

    # ------------------------------------------------------------ settings
    def configure(self, cfg: dict[str, Any]) -> None:
        changed = False
        for key in ARENA_DEFAULTS:
            if key in cfg and cfg[key] != self.cfg.get(key):
                self.cfg[key] = cfg[key]
                changed = True
        if changed:
            self.cam = self._make_camera()
            if self.phase == "countdown":
                self.speed = float(self.cfg["arena_speed"])  # applies to the next serve

    def _make_camera(self) -> Camera:
        return self.renderer.camera(
            yaw=self._cam_yaw,
            pitch=math.radians(float(self.cfg["arena_camera_pitch"])),
            distance=float(self.cfg["arena_camera_distance"]),
        )

    def _aim_camera(self, target: np.ndarray) -> None:
        self._cam_yaw = math.atan2(float(target[0]), float(target[2])) + math.radians(self.rng.uniform(-35, 35))
        self.cam = self._make_camera()

    def vision_settings(self) -> VisionSettings:
        """Detector settings that match this renderer exactly."""
        u, v, hw, hh = self.cam.character_box()
        w, h = self.renderer.width, self.renderer.height
        return VisionSettings(
            anchor_x=u / w,
            anchor_y=v / h,
            gate_w=max(2 * hw * 1.15 / h, 0.02),
            gate_h=max(2 * hh * 1.12 / h, 0.02),
            gate_enabled=True,
            min_area=4,
        )

    # ------------------------------------------------------------ flow
    def _log(self, kind: str, text: str) -> None:
        self.events.append({"t": round(self.t, 3), "kind": kind, "text": text})

    def _place_opponents(self) -> None:
        base = self.rng.uniform(0, 2 * math.pi)
        self.opponents = []
        for k in range(3):
            az = base + k * 2 * math.pi / 3 + self.rng.uniform(-0.4, 0.4)
            d = self.rng.uniform(18.0, 40.0)
            self.opponents.append(np.array([d * math.sin(az), 0.0, d * math.cos(az)]))

    def _new_round(self) -> None:
        self.stats["rounds"] += 1
        self.speed = float(self.cfg["arena_speed"])
        self._place_opponents()
        self.target_opp = int(self.rng.integers(len(self.opponents)))
        opp = self.opponents[self.target_opp]
        self.ball_pos = opp + np.array([0.0, TORSO[1] + 1.5, 0.0]) - opp / max(np.linalg.norm(opp), 1e-6) * 3.0
        self._aim_camera(opp)
        self.phase = "countdown"
        self.phase_end = self.t + 1.5
        self.shield = None
        self.pending.clear()
        self.cooldown_until = 0.0
        self._log("round", f"Round {self.stats['rounds']} - get ready!")

    def _start_incoming(self, origin: np.ndarray) -> None:
        self.phase = "incoming"
        start = origin + np.array([0.0, TORSO[1] + self.rng.uniform(-0.5, 2.5), 0.0])
        curve_mix, law_mix = CURVE_SETTINGS.get(str(self.cfg.get("arena_curves", "some")), CURVE_SETTINGS["some"])
        curve, direction, flight = sample_launch(
            self.rng, start[None, :], np.array([self.speed]), curve_mix, law_mix, accel_prob=0.0
        )
        self.curve = CURVE_NAMES[int(curve[0])]
        self.ball = OneBall(start, direction[0], self.speed, flight)
        self.t_hit = self.ball.time_to_hit(self.ball_radius + HIT_RADIUS, SUBSTEP_S)
        self.ball_pos = start
        self.ball_dir = direction[0]
        self.ghost = None
        self.stats["top_speed"] = max(self.stats["top_speed"], self.speed)

    def _start_away(self) -> None:
        choices = [i for i in range(len(self.opponents)) if i != self.target_opp] or [0]
        self.target_opp = int(self.rng.choice(choices))
        self._aim_camera(self.opponents[self.target_opp])
        self.phase = "away"
        target = self.opponents[self.target_opp] + TORSO
        d = target - self.ball_pos
        self.ball_dir = d / max(float(np.linalg.norm(d)), 1e-6)

    # ------------------------------------------------------------ parrying
    def request_parry(self, who: str = "human") -> None:
        """Press block. It registers after the simulated ping."""
        ping = float(self.cfg["arena_ping_ms"]) / 1000.0
        self.pending.append(_Press(self.t + ping, self.t, who, self.true_tti()))

    def note_ghost(self) -> None:
        """Remember when the network *would* have pressed (used in "Me" mode)."""
        if self.phase == "incoming" and self.ghost is None:
            self.ghost = (self.t, self.true_tti())

    def _register(self, p: _Press) -> None:
        if self.t < self.cooldown_until:
            if p.who == "human":
                self.feedback = {
                    "result": "cooldown",
                    "text": f"On cooldown for another {self.cooldown_until - self.t:.1f} s (you blocked too early before).",
                }
            return
        if self.shield is not None:
            return  # shield already up
        self.shield = (self.t, self.t + PARRY_WINDOW_S)
        self.shield_owner = p.who
        self.shield_press = p

    def _ball_text(self) -> str:
        return {"direct": "a straight ball", "side": "a side curve", "high": "a high curve",
                "back": "a backwards curve", "wild": "a wild curve"}.get(self.curve, "the ball")

    def _ghost_text(self) -> str:
        if self.ghost is None or self.ghost[1] is None:
            return ""
        return f" The network would have pressed with {self.ghost[1] * 1000:.0f} ms to go."

    def _blocked(self) -> None:
        assert self.shield is not None
        margin = self.t - self.shield[0]
        self.stats["blocks"] += 1
        self.stats["streak"] += 1
        self.stats["best_streak"] = max(self.stats["best_streak"], self.stats["streak"])
        who = "Network" if self.shield_owner == "bot" else "You"
        self._log("block", f"{who} blocked {self._ball_text()} at {self.speed:.0f} studs/s (shield up {margin * 1000:.0f} ms before impact)")
        if self.shield_owner == "human":
            if margin < 0.12:
                quality = "Close call!"
            elif margin > 0.38:
                quality = "A bit early, but it worked."
            else:
                quality = "Perfect timing!"
            self.feedback = {
                "result": "blocked",
                "text": f"{quality} The ball arrived {margin * 1000:.0f} ms after your shield went up (window {PARRY_WINDOW_S * 1000:.0f} ms)."
                + self._ghost_text(),
            }
        self.shield = None
        self.pending.clear()
        self.speed = min(self.speed * float(self.cfg["arena_speed_growth"]), float(self.cfg["arena_speed_max"]))
        self._start_away()

    def _hit(self) -> None:
        self.stats["hits"] += 1
        self.stats["streak"] = 0
        late = [p for p in self.pending if p.t_register > self.t]
        if late:
            p = late[0]
            text = f"Too late: your block registered {(p.t_register - self.t) * 1000:.0f} ms after the ball hit you."
        elif self.t < self.cooldown_until:
            text = "Hit while on cooldown - the earlier block was too early."
        else:
            text = "Hit without blocking."
        if any(p.who == "human" for p in self.pending) or (self.shield_press and self.shield_press.who == "human") or self.cfg.get("_human"):
            self.feedback = {"result": "hit", "text": text + self._ghost_text()}
        self._log("hit", f"Hit by {self._ball_text()} at {self.speed:.0f} studs/s! {text}")
        self.phase = "eliminated"
        self.phase_end = self.t + 1.6
        self.shield = None
        self.pending.clear()

    def _whiff(self) -> None:
        remaining = self.true_tti()
        self.stats["whiffs"] += 1
        self.cooldown_until = self.t + WHIFF_COOLDOWN_S
        who = self.shield_owner
        self.shield = None
        if remaining is not None:
            extra = f" The ball was still {remaining * 1000:.0f} ms away."
        elif self.phase == "countdown":
            extra = " The ball had not been served yet."
        elif self.phase == "away":
            extra = " The ball was heading for someone else."
        else:
            extra = ""
        self._log("whiff", f"{'Network' if who == 'bot' else 'You'} blocked too early - 2 s cooldown.{extra}")
        if who == "human":
            self.feedback = {"result": "early", "text": "Too early! The shield ran out before the ball arrived." + extra + self._ghost_text()}

    # ------------------------------------------------------------ simulation
    def true_tti(self) -> Optional[float]:
        """Exact time until the incoming ball reaches you (None if it isn't coming)."""
        if self.phase != "incoming" or self.ball is None or self.t_hit is None:
            return None
        return max(self.t_hit - self.ball.t, 0.0)

    def step(self, dt: float) -> None:
        """Advance the arena by ``dt`` seconds (in fixed 4 ms steps, so flights are exactly repeatable)."""
        self._acc += max(float(dt), 0.0)
        while self._acc >= SUBSTEP_S - 1e-9:
            self._acc -= SUBSTEP_S
            self._substep(SUBSTEP_S)

    def _substep(self, h: float) -> None:
        self.t += h
        for p in [p for p in self.pending if p.t_register <= self.t]:
            self.pending.remove(p)
            self._register(p)
        if self.shield is not None and self.t > self.shield[1]:
            self._whiff()
        if self.phase == "countdown":
            if self.t >= self.phase_end:
                self._start_incoming(self.opponents[self.target_opp])
        elif self.phase == "incoming":
            ball = self.ball
            assert ball is not None
            ball.step(h)
            self.ball_pos = np.array(ball.pos)
            self.ball_dir = np.array(ball.dir)
            arrived = ball.distance() <= self.ball_radius + HIT_RADIUS
            if self.t_hit is None and ball.t > 8.0:
                arrived = True  # safety net: a ball that somehow never arrives still ends the flight
            if arrived:
                if self.shield is not None and self.shield[0] <= self.t <= self.shield[1]:
                    self._blocked()
                else:
                    self._hit()
        elif self.phase == "away":
            target = self.opponents[self.target_opp] + TORSO
            self.ball_pos, self.ball_dir = _steer(self.ball_pos, self.ball_dir, target, self.speed, 3.0, h)
            if np.linalg.norm(self.ball_pos - target) <= self.ball_radius + HIT_RADIUS:
                self._start_incoming(self.opponents[self.target_opp])
        elif self.phase == "eliminated":
            if self.t >= self.phase_end:
                self._new_round()

    def frame(self) -> np.ndarray:
        incoming = self.phase == "incoming"
        decoys = bool(self.cfg["arena_decoys"])
        others = [(pos, decoys and self.phase == "away" and j == self.target_opp) for j, pos in enumerate(self.opponents)]
        ball = None if self.phase == "eliminated" else self.ball_pos
        return self.renderer.render(self.cam, ball, self.ball_radius, incoming, incoming, others)

    def state(self) -> dict[str, Any]:
        eta = None
        if self.phase == "incoming":
            dist = float(np.linalg.norm(self.ball_pos - TORSO)) - self.ball_radius - HIT_RADIUS
            eta = max(dist, 0.0) / max(self.speed, 1e-6)
        shield_left = max(self.shield[1] - self.t, 0.0) if self.shield else 0.0
        return {
            "phase": self.phase,
            "speed": round(self.speed, 1),
            "stats": dict(self.stats),
            "events": list(self.events)[-12:],
            "feedback": self.feedback,
            "shield": shield_left,
            "cooldown": max(self.cooldown_until - self.t, 0.0),
            "eta": eta,
            "t": round(self.t, 3),
        }


class ArenaSource:
    """Frame source for the bot engine that plays the arena in real time."""

    name = "arena"

    def __init__(self, arena: Optional[Arena] = None) -> None:
        self.arena = arena or Arena()
        self._last: Optional[float] = None

    def grab(self, cfg: dict[str, Any]) -> tuple[np.ndarray, float]:
        now = time.perf_counter()
        dt = 0.0 if self._last is None else min(now - self._last, 0.05)
        self._last = now
        self.arena.configure(cfg)
        self.arena.cfg["_human"] = cfg.get("arena_mode") == "human"
        self.arena.step(dt)
        return self.arena.frame(), self.arena.t

    def pause(self) -> None:
        self._last = None

    def close(self) -> None:
        self._last = None
