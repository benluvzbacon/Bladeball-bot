"""All user-tweakable settings, their limits and help texts, plus persistence.

The web menu is generated from :data:`SCHEMA`, so a setting only has to be
declared once here.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from .controller import normalize_key
from .model import PROJECT_DIR

DEFAULT_SETTINGS_PATH = PROJECT_DIR / "settings.json"


@dataclass
class Setting:
    key: str
    default: Any
    kind: str  # int | float | bool | choice | key | str
    label: str
    group: str
    help: str = ""
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    unit: str = ""
    choices: list[list[str]] = field(default_factory=list)  # [value, label]


GROUPS = {
    "timing": "Parry timing (neural network)",
    "input": "Controls",
    "capture": "Screen capture",
    "vision": "Ball detection",
    "gate": "Targeting check",
    "arena": "Practice arena (simulator)",
}

SCHEMA: list[Setting] = [
    # ---------------------------------------------------------------- timing
    Setting("lead_ms", 300, "int", "Parry lead time", "timing",
            "Parry this long before the network expects the ball to hit you. Raise it if the ball "
            "hits you before the bot blocks (high ping); lower it if the bot blocks too early.",
            50, 700, 10, "ms"),
    Setting("confidence", 0.6, "float", "Confidence threshold", "timing",
            "How sure the network must be that the ball arrives within the lead time.",
            0.2, 0.95, 0.05),
    Setting("instant_confidence", 0.85, "float", "Instant-parry confidence", "timing",
            "Above this probability the bot parries on the very first frame instead of waiting "
            "for confirmation (helps against very fast balls).", 0.5, 0.99, 0.01),
    Setting("confirm_frames", 2, "int", "Confirm frames", "timing",
            "Number of consecutive frames the network must agree before parrying.", 1, 5, 1),
    Setting("min_interval_ms", 150, "int", "Minimum time between parries", "timing",
            "Never press the parry input more often than this.", 30, 2000, 10, "ms"),
    Setting("rearm_ms", 60, "int", "Re-arm delay", "timing",
            "How long the targeting must disappear before the next approach counts as a new one "
            "(one parry per approach).", 20, 500, 10, "ms"),
    Setting("retry_ms", 2600, "int", "Retry after", "timing",
            "If the same approach is still going this long after a parry (you whiffed and sat out "
            "the 2 s cooldown), allow one more parry.", 600, 6000, 100, "ms"),
    # ---------------------------------------------------------------- input
    Setting("parry_input", "key", "choice", "Parry input", "input",
            "Blade Ball blocks with F or a left click.", choices=[["key", "Keyboard key"], ["mouse", "Left mouse click"]]),
    Setting("parry_key", "f", "key", "Parry key", "input",
            "The key bound to Block in Blade Ball (default F)."),
    Setting("key_hold_ms", 25, "int", "Key hold time", "input",
            "How long the key/button is held down for one parry.", 5, 150, 5, "ms"),
    Setting("hotkey", "f6", "key", "On/off hotkey", "input",
            "Global hotkey that switches the bot on/off while Roblox is focused."),
    Setting("dry_run", False, "bool", "Observe only (never press anything)", "input",
            "The bot shows when it WOULD parry but sends no input. Great for calibrating."),
    Setting("require_focus", True, "bool", "Only press while Roblox is the active window", "input",
            "Keeps the bot from typing its key into other programs (like this menu or a chat window). "
            "Turn it off only if the menu says the Roblox window can't be detected on your system."),
    Setting("beep", True, "bool", "Beep when toggled by hotkey", "input", "High beep = on, low beep = off (Windows)."),
    Setting("max_fps", 120, "int", "Max frames per second", "input",
            "Upper limit for the capture loop. Higher = more precise timing, more CPU.", 20, 240, 5, "fps"),
    # ---------------------------------------------------------------- capture
    Setting("capture_target", "roblox", "choice", "Capture", "capture",
            "Follow the Roblox window automatically (windowed or fullscreen, any monitor), or capture an "
            "area of a fixed monitor. If the Roblox window can't be found, the monitor below is used.",
            choices=[["roblox", "Roblox window (automatic)"], ["monitor", "Whole monitor"]]),
    Setting("capture_method", "auto", "choice", "Capture method", "capture",
            "Auto uses fast DXGI capture on Windows when the optional 'dxcam' package is installed "
            "(start.bat installs it) and the game is on the main monitor - it cuts 10-20 ms of delay per "
            "frame. Compatible always uses mss.",
            choices=[["auto", "Auto (fastest available)"], ["mss", "Compatible (mss)"]]),
    Setting("monitor", 1, "int", "Monitor", "capture",
            "Which monitor to capture when not following the Roblox window (1 = primary).", 1, 8, 1),
    Setting("region_w", 0.70, "float", "Capture width", "capture",
            "Width of the captured area, as a fraction of the Roblox window (or monitor). "
            "Keep the side UI out of it.", 0.2, 1.0, 0.01),
    Setting("region_h", 0.80, "float", "Capture height", "capture",
            "Height of the captured area, as a fraction of the Roblox window (or monitor).", 0.2, 1.0, 0.01),
    Setting("region_x", 0.0, "float", "Horizontal offset", "capture",
            "Shift the captured area left/right (fraction of the window/monitor width).", -0.4, 0.4, 0.01),
    Setting("region_y", 0.0, "float", "Vertical offset", "capture",
            "Shift the captured area up/down (fraction of the window/monitor height).", -0.4, 0.4, 0.01),
    Setting("scale", 2, "int", "Downscale", "capture",
            "Process every Nth pixel. 2 is a good balance of speed and precision.", 1, 4, 1, "x"),
    # ---------------------------------------------------------------- vision
    Setting("hue_center", 355.0, "float", "Ball hue", "vision",
            "Colour of the ball while it targets you (degrees on the colour wheel; red = 0/360).", 0, 360, 1, "deg"),
    Setting("hue_tol", 20.0, "float", "Hue tolerance", "vision", "", 2, 60, 1, "deg"),
    Setting("sat_min", 0.40, "float", "Minimum saturation", "vision", "", 0.0, 1.0, 0.01),
    Setting("val_min", 0.45, "float", "Minimum brightness", "vision", "", 0.0, 1.0, 0.01),
    Setting("min_area", 5, "int", "Minimum ball size", "vision",
            "Smallest blob (in processed pixels) that can be the ball.", 1, 500, 1, "px"),
    Setting("max_aspect", 1.8, "float", "Max stretch", "vision",
            "Blobs more stretched than this (width vs height) are not the ball.", 1.0, 4.0, 0.05),
    Setting("min_fill", 0.55, "float", "Minimum roundness", "vision",
            "How solid the blob must be (a filled circle is ~0.79, a character outline ~0.4).", 0.1, 1.0, 0.01),
    Setting("dilate", 1, "int", "Gap closing", "vision",
            "Close small gaps in the mask (the ball's glow ring).", 0, 3, 1, "px"),
    Setting("anchor_x", 0.50, "float", "Character X", "vision",
            "Where your character is inside the capture area (click it in the preview).", 0.0, 1.0, 0.005),
    Setting("anchor_y", 0.64, "float", "Character Y", "vision", "", 0.0, 1.0, 0.005),
    # ---------------------------------------------------------------- gate
    Setting("gate_enabled", True, "bool", "Only parry while I'm highlighted red", "gate",
            "Blade Ball paints your character red while the ball targets you. Checking for it stops the "
            "bot reacting to other red things."),
    Setting("gate_w", 0.14, "float", "Character box width", "gate",
            "The box should cover your whole character (relative to the capture height). "
            "It is also cut out when looking for the ball.", 0.02, 0.6, 0.01),
    Setting("gate_h", 0.30, "float", "Character box height", "gate",
            "Relative to the capture height. Make it just as tall as your character: the network "
            "also uses the box size to judge how far the camera is zoomed out.", 0.02, 0.8, 0.01),
    Setting("gate_min_frac", 0.04, "float", "Red needed in box", "gate",
            "Fraction of red pixels in the box that means you are targeted.", 0.005, 0.6, 0.005),
    # ---------------------------------------------------------------- arena
    Setting("arena_mode", "bot", "choice", "Who plays", "arena",
            "Watch the neural network play, or practise your own timing (Space / click to block).",
            choices=[["bot", "Neural network"], ["human", "Me (Space / click)"]]),
    Setting("arena_ping_ms", 60, "int", "Simulated ping", "arena", "Delay before a block registers.", 0, 300, 5, "ms"),
    Setting("arena_speed", 45.0, "float", "Starting ball speed", "arena", "", 15, 200, 1, "studs/s"),
    Setting("arena_speed_growth", 1.07, "float", "Speed-up per block", "arena", "", 1.0, 1.3, 0.01, "x"),
    Setting("arena_speed_max", 300.0, "float", "Maximum ball speed", "arena", "", 50, 500, 5, "studs/s"),
    Setting("arena_camera_distance", 16.0, "float", "Camera distance", "arena", "", 6, 40, 0.5, "studs"),
    Setting("arena_camera_pitch", 20.0, "float", "Camera pitch", "arena", "", 2, 60, 1, "deg"),
    Setting("arena_curves", "some", "choice", "Curve balls", "arena",
            "How often the other players curve the ball out to the side, up high or backwards before it "
            "comes to you.", choices=[["off", "Off"], ["some", "Some"], ["lots", "Lots"]]),
    Setting("arena_decoys", True, "bool", "Red decoys", "arena",
            "Other players that glow red when the ball targets them (the bot must ignore them)."),
]

# Settings that are not shown as regular form fields.
HIDDEN: list[Setting] = [
    Setting("source", "screen", "choice", "Source", "hidden", choices=[["screen", "Roblox (screen)"], ["arena", "Practice arena"]]),
    Setting("model_path", "models/parry_net.npz", "str", "Model", "hidden"),
    Setting("practice_ack", False, "bool", "Practice-only acknowledged", "hidden"),
]

ALL_SETTINGS: dict[str, Setting] = {s.key: s for s in SCHEMA + HIDDEN}


def coerce(setting: Setting, value: Any) -> Any:
    """Validate/clamp a value for ``setting``; raises ``ValueError`` if impossible."""
    kind = setting.kind
    if kind == "bool":
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    if kind in ("int", "float"):
        v = float(value)
        if v != v:  # NaN
            raise ValueError(f"{setting.key}: not a number")
        if setting.min is not None:
            v = max(v, float(setting.min))
        if setting.max is not None:
            v = min(v, float(setting.max))
        return int(round(v)) if kind == "int" else round(v, 6)
    if kind == "choice":
        v = str(value)
        if v not in [c[0] for c in setting.choices]:
            raise ValueError(f"{setting.key}: must be one of {[c[0] for c in setting.choices]}")
        return v
    if kind == "key":
        return normalize_key(str(value))
    return str(value)


class Settings:
    """Thread-safe settings store backed by a JSON file."""

    def __init__(self, path: Optional[Path | str] = DEFAULT_SETTINGS_PATH) -> None:
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        self._values: dict[str, Any] = {k: s.default for k, s in ALL_SETTINGS.items()}
        self.version = 0
        self.load_error: Optional[str] = None
        if self.path and self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                self.update(data, save=False)
            except Exception as exc:
                self.load_error = f"could not read {self.path.name}: {exc}"

    def get(self, key: str) -> Any:
        with self._lock:
            return self._values[key]

    def __getitem__(self, key: str) -> Any:
        return self.get(key)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._values)

    def update(self, changes: dict[str, Any], save: bool = True) -> dict[str, Any]:
        """Apply valid changes, ignore unknown keys. Returns the applied values."""
        applied: dict[str, Any] = {}
        errors: list[str] = []
        for key, value in changes.items():
            setting = ALL_SETTINGS.get(key)
            if setting is None:
                continue
            try:
                applied[key] = coerce(setting, value)
            except (TypeError, ValueError) as exc:
                errors.append(str(exc))
        if applied:
            with self._lock:
                self._values.update(applied)
                self.version += 1
            if save:
                self.save()
        if errors and not applied:
            raise ValueError("; ".join(errors))
        return applied

    def reset(self, group: Optional[str] = None) -> None:
        defaults = {k: s.default for k, s in ALL_SETTINGS.items() if group is None or s.group == group}
        self.update(defaults)

    def save(self) -> None:
        if not self.path:
            return
        data = self.snapshot()
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.path)

    @staticmethod
    def schema() -> dict[str, Any]:
        return {
            "groups": GROUPS,
            "settings": [asdict(s) for s in SCHEMA],
        }
