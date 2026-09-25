import http.client
import json

import pytest

from bladebot.config import Settings
from bladebot.engine import BotEngine
from bladebot.server import ControlServer


class FakeController:
    backend = None
    backend_error = "test"
    last_error = None

    def press(self, method, key, hold_s):
        return True


@pytest.fixture
def server(tmp_path, bundled_model):
    settings = Settings(tmp_path / "s.json")
    engine = BotEngine(settings, controller=FakeController(), model=bundled_model)
    engine.start()
    srv = ControlServer(engine, settings, "127.0.0.1", 0)
    srv.start()
    yield srv
    srv.shutdown()
    engine.stop()


def request(srv, method, path, body=None, headers=None, host=None):
    conn = http.client.HTTPConnection("127.0.0.1", srv.port, timeout=10)
    h = {"Host": host or f"127.0.0.1:{srv.port}"}
    h.update(headers or {})
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        h["Content-Type"] = "application/json"
    conn.request(method, path, body=data, headers=h)
    resp = conn.getresponse()
    payload = resp.read()
    conn.close()
    return resp.status, resp.getheader("Content-Type"), payload


def post(srv, path, body=None):
    status, _, payload = request(srv, "POST", path, body or {}, {"X-BladeBot": "1"})
    return status, json.loads(payload)


def test_menu_page_and_assets(server):
    status, ctype, body = request(server, "GET", "/")
    assert status == 200 and ctype.startswith("text/html")
    assert b"BladeBot" in body and b"PRACTICE ONLY" in body
    for path in ("/app.js", "/style.css"):
        assert request(server, "GET", path)[0] == 200


def test_state_and_schema(server):
    status, _, body = request(server, "GET", "/api/state")
    st = json.loads(body)
    assert status == 200 and st["enabled"] is False and "nn" in st
    status, _, body = request(server, "GET", "/api/schema")
    schema = json.loads(body)
    assert any(s["key"] == "lead_ms" for s in schema["settings"])
    assert schema["values"]["source"] == "screen"


def test_posts_need_the_custom_header_and_local_host(server):
    status, _, _ = request(server, "POST", "/api/toggle", {"on": True})
    assert status == 403
    status, _, _ = request(server, "GET", "/api/state", host="evil.example.com")
    assert status == 403


def test_settings_roundtrip(server):
    status, r = post(server, "/api/settings", {"changes": {"lead_ms": 250, "parry_key": "G"}})
    assert status == 200 and r["values"]["lead_ms"] == 250 and r["values"]["parry_key"] == "g"
    status, r = post(server, "/api/settings", {"changes": {"source": "nowhere"}})
    assert status == 400


def test_practice_notice_gates_the_screen_bot(server):
    status, r = post(server, "/api/toggle", {"on": True})
    assert status == 409 and "practice" in r["error"].lower()
    assert post(server, "/api/ack", {"accepted": True})[0] == 200
    status, r = post(server, "/api/toggle", {"on": True})
    assert status == 200 and r["enabled"] is True
    status, r = post(server, "/api/toggle", {"on": False})
    assert r["enabled"] is False


def test_arena_endpoints_and_preview(server):
    assert post(server, "/api/settings", {"changes": {"source": "arena", "arena_mode": "human"}})[0] == 200
    assert post(server, "/api/arena/parry")[0] == 200
    assert post(server, "/api/arena/reset")[0] == 200
    status, ctype, png = request(server, "GET", "/api/preview.png")
    assert status == 200 and ctype == "image/png" and png[:4] == b"\x89PNG"
    assert request(server, "GET", "/api/nope")[0] == 404
    status, _, body = request(server, "GET", "/api/model")
    assert json.loads(body)["loaded"] is True
