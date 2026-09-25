"""Record what the bot sees during practice so the network can learn from it.

A recording is a JSON-lines file: one header line followed by one line per
frame with the detection (anchor-relative ``x, y, r``) and whether your
character was highlighted red (the "targeted" gate).

Because the red highlight switches off the moment the ball touches you (you
either deflect it or get hit), the end of every red period tells us *when the
ball arrived*. :func:`bladebot.training.dataset_from_recordings` uses this to
label every earlier frame with its true time-to-impact - free training data
from your own practice sessions ("hindsight labelling").
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence

from .model import PROJECT_DIR

RECORDINGS_DIR = PROJECT_DIR / "recordings"
FORMAT_VERSION = 1


class Recorder:
    """Buffered JSON-lines writer for one practice session."""

    def __init__(
        self, source: str, gate: bool, char_h: float = 0.30, directory: Path | str = RECORDINGS_DIR
    ) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.path = directory / f"session-{stamp}-{source}.jsonl"
        suffix = 1
        while self.path.exists():
            suffix += 1
            self.path = directory / f"session-{stamp}-{source}-{suffix}.jsonl"
        self._fh = open(self.path, "w", encoding="utf-8")
        self._t0: Optional[float] = None
        self._last_flush = time.monotonic()
        self.frames = 0
        header = {
            "type": "header",
            "version": FORMAT_VERSION,
            "created": datetime.now().isoformat(timespec="seconds"),
            "source": source,
            "gate": bool(gate),
            "char_h": round(float(char_h), 4),
        }
        self._fh.write(json.dumps(header) + "\n")

    def write(self, t: float, det: Optional[Sequence[float]], targeted: Optional[bool]) -> None:
        if self._fh is None:
            return
        if self._t0 is None:
            self._t0 = t
        row: dict[str, Any] = {"t": round(t - self._t0, 5)}
        row["d"] = None if det is None else [round(float(v), 6) for v in det]
        row["g"] = None if targeted is None else int(bool(targeted))
        self._fh.write(json.dumps(row, separators=(",", ":")) + "\n")
        self.frames += 1
        now = time.monotonic()
        if now - self._last_flush > 1.0:
            self._fh.flush()
            self._last_flush = now

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def read_recording(path: Path | str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return ``(header, frames)`` of a recording file (tolerates a truncated last line)."""
    header: dict[str, Any] = {}
    frames: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("type") == "header":
                header = obj
            elif "t" in obj:
                frames.append(obj)
    return header, frames


def list_recordings(directory: Path | str = RECORDINGS_DIR) -> list[Path]:
    directory = Path(directory)
    if not directory.exists():
        return []
    return sorted(directory.glob("session-*.jsonl"))


def iter_episodes(
    frames: list[dict[str, Any]], use_gate: bool, merge_gap_s: float
) -> Iterator[tuple[int, int]]:
    """Yield ``(first_index, last_index)`` of every "ball is coming at me" period."""
    start: Optional[int] = None
    last_active: Optional[int] = None
    for k, fr in enumerate(frames):
        active = bool(fr.get("g")) if use_gate else fr.get("d") is not None
        if active:
            if start is None:
                start = k
            elif last_active is not None and fr["t"] - frames[last_active]["t"] > merge_gap_s:
                yield start, last_active
                start = k
            last_active = k
    if start is not None and last_active is not None:
        yield start, last_active
