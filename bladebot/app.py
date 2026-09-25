"""Command-line entry point: starts the bot engine, the hotkey and the web menu."""

from __future__ import annotations

import argparse
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Optional, Sequence

from . import __version__
from .config import DEFAULT_SETTINGS_PATH, Settings

BANNER = f"""
================================================================
 BladeBot {__version__} - neural-network parry bot for Blade Ball
----------------------------------------------------------------
 PRACTICE USE ONLY. Use it in the built-in practice arena, in
 practice/training modes, or in private servers where everyone
 agrees. Never in public matches: automating gameplay can break
 Roblox's Terms of Use and get your account banned.
================================================================
"""


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="bladebot",
        description="BladeBot - neural-network parry bot for Roblox Blade Ball (practice use only).",
    )
    ap.add_argument("--host", default="127.0.0.1", help="menu address (default 127.0.0.1 = this PC only)")
    ap.add_argument("--port", type=int, default=8765, help="menu port (default 8765)")
    ap.add_argument("--no-browser", action="store_true", help="don't open the menu in the browser")
    ap.add_argument("--no-hotkey", action="store_true", help="disable the global on/off hotkey")
    ap.add_argument("--source", choices=["screen", "arena"], help="start with this source selected")
    ap.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS_PATH, help="settings file")
    ap.add_argument("--version", action="version", version=f"BladeBot {__version__}")
    args = ap.parse_args(argv)

    try:
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except Exception:
        pass
    print(BANNER)
    # imported here so --help/--version stay instant
    from .engine import BotEngine
    from .hotkeys import HotkeyListener
    from .server import ControlServer

    settings = Settings(args.settings)
    if settings.load_error:
        print(f"Warning: {settings.load_error} - using defaults")
    if args.source:
        settings.update({"source": args.source})

    engine = BotEngine(settings)
    if engine.model is None:
        print(f"Warning: {engine.model_error}")
    else:
        print(f"Neural network: {engine.model_path}")
    backend = getattr(getattr(engine.controller, "backend", None), "name", None)
    print(f"Input: {backend or 'unavailable - ' + str(getattr(engine.controller, 'backend_error', ''))}")

    hotkeys = None
    if not args.no_hotkey:
        hotkeys = HotkeyListener(lambda: settings.get("hotkey"), engine.toggle_from_hotkey)
        if hotkeys.start():
            print(f"Hotkey: {settings.get('hotkey').upper()} switches the bot on/off")
        else:
            engine.hotkey_error = hotkeys.error
            print(f"Hotkey: {hotkeys.error}")
    else:
        engine.hotkey_error = "disabled with --no-hotkey"

    try:
        server = ControlServer(engine, settings, args.host, args.port)
    except OSError as exc:
        print(f"Could not start the menu on {args.host}:{args.port} ({exc}).")
        print("Is BladeBot already running? Otherwise try another port: --port 8766")
        return 1

    engine.start()
    print(f"\nMenu: {server.url}   (press Ctrl+C here to quit)\n")
    if not args.no_browser:
        def _open() -> None:
            time.sleep(0.6)
            try:
                webbrowser.open(server.url)
            except Exception:
                pass

        threading.Thread(target=_open, daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        engine.stop()
        if hotkeys is not None:
            hotkeys.stop()
        try:
            server.httpd.server_close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
