"""Local web server for the control menu (standard library only).

The menu is a small web page served on http://127.0.0.1:8765 - it works in
any browser, needs no GUI toolkit and can stay open on a second monitor.
"""

from __future__ import annotations

import json
import math
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import urlsplit

import numpy as np

from .config import Settings
from .engine import BotEngine
from .model import DEFAULT_MODEL_PATH, PROJECT_DIR
from .pngenc import encode_png
from .recorder import list_recordings
from .training import PipelineConfig, TrainConfig, TrainingJob

WEB_DIR = Path(__file__).resolve().parent / "web"
CUSTOM_MODEL = "models/custom_parry_net.npz"
STATIC = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "application/javascript; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
}


def _placeholder_png() -> bytes:
    img = np.zeros((180, 320, 3), dtype=np.uint8)
    img[:] = (24, 26, 34)
    img[88:92, 120:200] = (60, 64, 80)
    return encode_png(img)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


class ControlServer:
    def __init__(self, engine: BotEngine, settings: Settings, host: str = "127.0.0.1", port: int = 8765) -> None:
        self.engine = engine
        self.settings = settings
        self.host = host
        self.port = port
        self.job: Optional[TrainingJob] = None
        self._placeholder = _placeholder_png()
        self.check_host = host in ("127.0.0.1", "localhost", "::1")
        handler = self._make_handler()
        self.httpd = ThreadingHTTPServer((host, port), handler)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self._thread: Optional[threading.Thread] = None

    @property
    def url(self) -> str:
        host = "127.0.0.1" if self.host in ("0.0.0.0", "::", "") else self.host
        return f"http://{host}:{self.port}/"

    def start(self) -> None:
        self._thread = threading.Thread(target=self.httpd.serve_forever, name="bladebot-http", daemon=True)
        self._thread.start()

    def serve_forever(self) -> None:
        self.httpd.serve_forever()

    def shutdown(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    # ------------------------------------------------------------ API
    def api_get(self, path: str) -> tuple[int, Any]:
        e, s = self.engine, self.settings
        if path == "/api/state":
            return 200, e.status()
        if path == "/api/schema":
            return 200, {**s.schema(), "values": s.snapshot()}
        if path == "/api/settings":
            return 200, s.snapshot()
        if path == "/api/model":
            return 200, e.model_info()
        if path == "/api/train/status":
            return 200, self.job.status() if self.job else {"state": "idle"}
        if path == "/api/recordings":
            files = [
                {"name": p.name, "kb": round(p.stat().st_size / 1024, 1), "modified": datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")}
                for p in list_recordings()
            ]
            return 200, {"files": files}
        return 404, {"error": "not found"}

    def api_post(self, path: str, body: dict[str, Any]) -> tuple[int, Any]:
        e, s = self.engine, self.settings
        if path == "/api/toggle":
            on = body.get("on")
            on = (not e.enabled) if on is None else bool(on)
            ok, msg = e.set_enabled(on, via="menu")
            return (200 if ok else 409), {"ok": ok, "enabled": e.enabled, "error": msg or None}
        if path == "/api/settings":
            changes = body.get("changes", body)
            if not isinstance(changes, dict):
                return 400, {"error": "expected an object of settings"}
            try:
                applied = s.update(changes)
            except ValueError as exc:
                return 400, {"error": str(exc)}
            if "source" in applied and applied["source"] == "screen" and e.enabled and not s.get("practice_ack"):
                e.set_enabled(False, via="source change")
            if "model_path" in applied:
                e.load_model(applied["model_path"])
            return 200, {"ok": True, "applied": applied, "values": s.snapshot()}
        if path == "/api/settings/reset":
            group = body.get("group")
            ack = s.get("practice_ack")
            source = s.get("source")
            model_path = s.get("model_path")
            s.reset(group)
            s.update({"practice_ack": ack, "source": source, "model_path": model_path})
            return 200, {"ok": True, "values": s.snapshot()}
        if path == "/api/ack":
            accepted = bool(body.get("accepted"))
            s.update({"practice_ack": accepted})
            if not accepted and e.enabled and s.get("source") == "screen":
                e.set_enabled(False, via="notice declined")
            e.log("info", "Practice-only notice " + ("accepted" if accepted else "withdrawn"))
            return 200, {"ok": True, "practice_ack": accepted}
        if path == "/api/pick_color":
            try:
                return 200, {"ok": True, **e.pick_color(float(body["x"]), float(body["y"]))}
            except (KeyError, TypeError, ValueError) as exc:
                return 400, {"error": str(exc)}
        if path == "/api/anchor":
            try:
                return 200, {"ok": True, **e.set_anchor(float(body["x"]), float(body["y"]))}
            except (KeyError, TypeError, ValueError) as exc:
                return 400, {"error": str(exc)}
        if path == "/api/arena/parry":
            e.human_parry()
            return 200, {"ok": True}
        if path == "/api/arena/reset":
            e.reset_arena()
            return 200, {"ok": True}
        if path == "/api/record":
            if body.get("on"):
                return 200, {"ok": True, "path": e.start_recording()}
            e.stop_recording()
            return 200, {"ok": True}
        if path == "/api/train":
            return self._start_training(body)
        if path == "/api/train/cancel":
            if self.job and self.job.running:
                self.job.cancel()
            return 200, {"ok": True}
        if path == "/api/model/reload":
            ok = e.load_model(body.get("path"))
            return (200 if ok else 409), {"ok": ok, "error": e.model_error, "model": e.model_info()}
        if path == "/api/model/use_default":
            rel = str(DEFAULT_MODEL_PATH.relative_to(PROJECT_DIR)).replace("\\", "/")
            ok = e.load_model(rel)
            return (200 if ok else 409), {"ok": ok, "error": e.model_error, "model": e.model_info()}
        return 404, {"error": "not found"}

    def _start_training(self, body: dict[str, Any]) -> tuple[int, Any]:
        if self.job and self.job.running:
            return 409, {"error": "training is already running"}

        def num(key: str, default: float, lo: float, hi: float) -> float:
            try:
                return min(max(float(body.get(key, default)), lo), hi)
            except (TypeError, ValueError):
                return default

        hidden_raw = str(body.get("hidden", "96,96"))
        try:
            hidden = tuple(min(max(int(v), 4), 256) for v in hidden_raw.split(",") if v.strip())[:4] or (96, 96)
        except ValueError:
            return 400, {"error": "hidden layers must look like 96,96"}
        activation = body.get("activation", "tanh")
        if activation not in ("tanh", "relu", "leaky_relu"):
            activation = "tanh"
        cfg = PipelineConfig(
            approaches=int(num("approaches", 40000, 500, 300000)),
            val_approaches=int(num("val_approaches", 2000, 200, 10000)),
            seed=int(num("seed", 0, 0, 1_000_000)),
            speed_min=num("speed_min", 25, 3, 300),
            speed_max=num("speed_max", 450, 30, 600),
            use_recordings=bool(body.get("use_recordings", False)),
            train=TrainConfig(hidden=hidden, activation=activation, epochs=int(num("epochs", 25, 1, 200))),
        )
        if cfg.speed_max <= cfg.speed_min:
            cfg.speed_max = cfg.speed_min + 10
        out = PROJECT_DIR / CUSTOM_MODEL
        out.parent.mkdir(parents=True, exist_ok=True)

        def done(model: Any, path: Path) -> None:
            self.engine.set_model(model, path)
            self.engine.log("info", f"New network trained and loaded ({path.name})")

        self.job = TrainingJob(cfg, out, on_done=done)
        self.engine.log("info", f"Training started ({cfg.approaches} simulated approaches, {cfg.train.epochs} epochs)")
        return 200, {"ok": True, "status": self.job.status()}

    # ------------------------------------------------------------ HTTP plumbing
    def _make_handler(self) -> type[BaseHTTPRequestHandler]:
        server = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "BladeBot"
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - quiet
                return

            def _host_ok(self) -> bool:
                if not server.check_host:
                    return True
                host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]").lower()
                return host in ("127.0.0.1", "localhost", "::1")

            def _send(self, code: int, body: bytes, ctype: str, extra: Optional[dict[str, str]] = None) -> None:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                for k, v in (extra or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(body)

            def _json(self, code: int, obj: Any) -> None:
                data = json.dumps(obj, default=_json_default).encode("utf-8")
                self._send(code, data, "application/json; charset=utf-8")

            def do_HEAD(self) -> None:  # noqa: N802
                self.do_GET()

            def do_GET(self) -> None:  # noqa: N802
                if not self._host_ok():
                    self._json(403, {"error": "forbidden host"})
                    return
                path = urlsplit(self.path).path
                try:
                    if path in STATIC:
                        name, ctype = STATIC[path]
                        self._send(200, (WEB_DIR / name).read_bytes(), ctype)
                    elif path == "/api/preview.png":
                        png = server.engine.preview() or server._placeholder
                        self._send(200, png, "image/png")
                    elif path.startswith("/api/"):
                        code, obj = server.api_get(path)
                        self._json(code, obj)
                    elif path == "/favicon.ico":
                        self._send(204, b"", "image/x-icon")
                    else:
                        self._json(404, {"error": "not found"})
                except (BrokenPipeError, ConnectionResetError):
                    pass
                except Exception as exc:  # keep the menu alive whatever happens
                    try:
                        self._json(500, {"error": f"{exc.__class__.__name__}: {exc}"})
                    except Exception:
                        pass

            def do_POST(self) -> None:  # noqa: N802
                if not self._host_ok():
                    self._json(403, {"error": "forbidden host"})
                    return
                if self.headers.get("X-BladeBot") != "1":
                    self._json(403, {"error": "missing X-BladeBot header"})
                    return
                path = urlsplit(self.path).path
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                    if length > 1_000_000:
                        self._json(413, {"error": "request too large"})
                        return
                    raw = self.rfile.read(length) if length else b"{}"
                    body = json.loads(raw.decode("utf-8") or "{}")
                    if not isinstance(body, dict):
                        raise ValueError("expected a JSON object")
                except (ValueError, UnicodeDecodeError) as exc:
                    self._json(400, {"error": f"bad request: {exc}"})
                    return
                try:
                    code, obj = server.api_post(path, body)
                    self._json(code, obj)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                except Exception as exc:
                    try:
                        self._json(500, {"error": f"{exc.__class__.__name__}: {exc}"})
                    except Exception:
                        pass

        return Handler


def run_server(
    engine: BotEngine,
    settings: Settings,
    host: str,
    port: int,
    on_ready: Optional[Callable[[ControlServer], None]] = None,
) -> ControlServer:
    srv = ControlServer(engine, settings, host, port)
    if on_ready:
        on_ready(srv)
    return srv
