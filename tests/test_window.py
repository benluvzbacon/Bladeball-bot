"""Roblox window detection, window-following capture and the focus safety check.

The operating-system calls can't run here, so these tests use fake backends and
check the logic around them.
"""

from types import SimpleNamespace

import pytest

from bladebot.capture import ScreenSource, clip_rect
from bladebot.config import Settings
from bladebot.engine import BotEngine, default_config
from bladebot.window import RobloxWindowFinder, WindowInfo, looks_like_roblox

from test_engine import FakeController


@pytest.mark.parametrize(
    "title, owner, expected",
    [
        ("Roblox", "", True),  # Windows player window title
        ("roblox", "RobloxPlayerBeta.exe", True),
        ("", "C:\\Users\\me\\AppData\\Local\\Roblox\\Versions\\v1\\RobloxPlayerBeta.exe", True),
        ("", "Roblox", True),  # macOS application name
        ("", "RobloxPlayer", True),
        ("Sober", "", True),  # Linux (Sober)
        ("Roblox", "explorer.exe", False),  # a folder called Roblox
        ("Roblox", "ApplicationFrameHost.exe", True),  # Microsoft Store version
        ("Roblox Studio", "", False),
        ("Baseplate - Roblox Studio", "RobloxStudioBeta.exe", False),
        ("Roblox - Google Chrome", "chrome.exe", False),
        ("BladeBot - Blade Ball practice bot", "firefox.exe", False),
        ("", "", False),
    ],
)
def test_looks_like_roblox(title, owner, expected):
    assert looks_like_roblox(title, owner) is expected


class FakeBackend:
    name = "fake"

    def __init__(self, window=None, front=None):
        self.window = window
        self.front = front
        self.find_calls = 0
        self.fail = False

    def find_roblox(self):
        self.find_calls += 1
        if self.fail:
            raise OSError("boom")
        return self.window

    def foreground(self):
        if self.fail:
            raise OSError("boom")
        return self.front


ROBLOX = WindowInfo("Roblox", "RobloxPlayerBeta.exe", 100, 50, 1280, 720, focused=True)
BROWSER = WindowInfo("BladeBot - Chrome", "chrome.exe", 0, 0, 800, 600, focused=True)


def test_finder_caches_and_survives_errors():
    backend = FakeBackend(window=ROBLOX)
    finder = RobloxWindowFinder(backend, ttl_s=10.0)
    assert finder.supported
    assert finder.find() == ROBLOX
    assert finder.find() == ROBLOX
    assert backend.find_calls == 1  # second call came from the cache
    finder.invalidate()
    backend.fail = True
    assert finder.find() is None
    assert "boom" in finder.error


def test_finder_focus_check():
    backend = FakeBackend(front=ROBLOX)
    finder = RobloxWindowFinder(backend)
    assert finder.roblox_focused() is True
    backend.front = BROWSER
    assert finder.roblox_focused() is False
    backend.front = None  # nothing in front (e.g. the desktop)
    assert finder.roblox_focused() is False
    backend.fail = True
    assert finder.roblox_focused() is None  # can't tell


def test_finder_without_backend_is_unknown(monkeypatch):
    import bladebot.window as window

    monkeypatch.setattr(window, "_make_backend", lambda: (None, "not supported"))
    finder = RobloxWindowFinder()
    assert not finder.supported
    assert finder.find() is None
    assert finder.roblox_focused() is None
    assert finder.error == "not supported"


def test_window_info_usable():
    assert ROBLOX.usable
    assert not WindowInfo("Roblox", "", 0, 0, 1280, 720, minimized=True).usable
    assert not WindowInfo("Roblox", "", 0, 0, 100, 80).usable


def test_clip_rect():
    screen = {"left": 0, "top": 0, "width": 1920, "height": 1080}
    assert clip_rect({"left": 100, "top": 50, "width": 800, "height": 600}, screen) == {
        "left": 100, "top": 50, "width": 800, "height": 600}
    assert clip_rect({"left": 1600, "top": -100, "width": 800, "height": 600}, screen) == {
        "left": 1600, "top": 0, "width": 320, "height": 500}
    assert clip_rect({"left": 3000, "top": 0, "width": 800, "height": 600}, screen) is None


class FakeShot:
    def __init__(self, region):
        self.width, self.height = region["width"], region["height"]
        self.raw = bytes(self.width * self.height * 4)


class FakeMSS:
    monitors = [
        {"left": 0, "top": 0, "width": 3840, "height": 1080},  # all monitors
        {"left": 0, "top": 0, "width": 1920, "height": 1080},
        {"left": 1920, "top": 0, "width": 1920, "height": 1080},
    ]

    def __init__(self):
        self.regions = []

    def grab(self, region):
        self.regions.append(dict(region))
        return FakeShot(region)

    def close(self):
        pass


def screen_with(window):
    src = ScreenSource(RobloxWindowFinder(FakeBackend(window=window), ttl_s=0.0))
    src._sct = FakeMSS()
    return src


def test_capture_follows_the_roblox_window():
    win = WindowInfo("Roblox", "RobloxPlayerBeta.exe", 2020, 100, 1000, 800, focused=False)  # on monitor 2
    src = screen_with(win)
    frame, _ = src.grab(default_config(region_w=0.5, region_h=0.5, scale=2))
    assert src.target == "roblox" and src.note is None
    assert src.region == {"left": 2270, "top": 300, "width": 500, "height": 400}  # centre of the window
    assert frame.shape == (200, 250, 3)


def test_capture_falls_back_to_the_monitor():
    cfg = default_config(region_w=1.0, region_h=1.0, monitor=2)
    src = screen_with(None)
    src.grab(cfg)
    assert src.target == "monitor" and "not found" in src.note
    assert src.region == {"left": 1920, "top": 0, "width": 1920, "height": 1080}

    src = screen_with(WindowInfo("Roblox", "", 0, 0, 1280, 720, minimized=True))
    src.grab(cfg)
    assert src.target == "monitor" and "minimised" in src.note

    src = screen_with(ROBLOX)
    src.grab(default_config(region_w=1.0, region_h=1.0, monitor=1, capture_target="monitor"))
    assert src.target == "monitor" and src.note is None
    assert src.region == {"left": 0, "top": 0, "width": 1920, "height": 1080}


class FakeFinder:
    supported = True
    error = None

    def __init__(self, focused):
        self.focused = focused

    def roblox_focused(self, max_age_s=0.0):
        return self.focused

    def find(self, max_age_s=None):
        return ROBLOX if self.focused else None

    def invalidate(self):
        pass


def screen_engine(tmp_path, bundled_model, focused, name="s.json", **settings):
    s = Settings(tmp_path / name)
    s.update({"practice_ack": True, "source": "screen", **settings})
    ctrl = FakeController()
    engine = BotEngine(s, controller=ctrl, model=bundled_model, window_finder=FakeFinder(focused))
    assert engine.set_enabled(True)[0]
    return engine, ctrl, s.snapshot()


PRESS = SimpleNamespace(p_lead=0.9, eta=0.25)


def test_no_presses_while_another_window_is_active(tmp_path, bundled_model):
    engine, ctrl, cfg = screen_engine(tmp_path, bundled_model, focused=False)
    engine._handle_press(PRESS, cfg, "screen")
    engine._handle_press(PRESS, cfg, "screen")
    assert ctrl.calls == [] and engine.parries == 0
    skipped = [e for e in engine.events if "not the active window" in e["text"]]
    assert len(skipped) == 1  # rate-limited

    engine.window_finder.focused = True
    engine._handle_press(PRESS, cfg, "screen")
    assert ctrl.calls == [("key", "f", 0.025)] and engine.parries == 1


def test_focus_check_can_be_switched_off_or_unknown(tmp_path, bundled_model):
    engine, ctrl, cfg = screen_engine(tmp_path, bundled_model, focused=False, require_focus=False)
    engine._handle_press(PRESS, cfg, "screen")
    assert len(ctrl.calls) == 1

    engine, ctrl, cfg = screen_engine(tmp_path, bundled_model, focused=None, name="s2.json")  # can't tell
    engine._handle_press(PRESS, cfg, "screen")
    engine._handle_press(PRESS, cfg, "screen")
    assert len(ctrl.calls) == 2
    assert sum("Can't check" in e["text"] for e in engine.events) == 1


def test_status_reports_the_window(tmp_path, bundled_model):
    engine, _, cfg = screen_engine(tmp_path, bundled_model, focused=True)
    engine._publish(cfg, None, idle=True)
    w = engine.status()["window"]
    assert w["found"] and w["focused"] and w["size"] == [1280, 720] and w["want"] == "roblox"
