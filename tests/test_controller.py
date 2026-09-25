import time

from bladebot.controller import ParryController


class FakeBackend:
    name = "fake"

    def __init__(self, fail=False):
        self.log = []
        self.fail = fail

    def key_down(self, key):
        if self.fail:
            raise OSError("access denied")
        self.log.append(("down", key, time.perf_counter()))

    def key_up(self, key):
        self.log.append(("up", key, time.perf_counter()))

    def mouse_down(self):
        self.log.append(("down", "mouse", time.perf_counter()))

    def mouse_up(self):
        self.log.append(("up", "mouse", time.perf_counter()))


def wait_for(cond, timeout=1.0):
    end = time.time() + timeout
    while not cond() and time.time() < end:
        time.sleep(0.002)
    return cond()


def test_key_goes_down_immediately_and_is_released_after_the_hold():
    b = FakeBackend()
    c = ParryController(backend=b)
    t0 = time.perf_counter()
    assert c.press("key", "f", 0.03)
    assert b.log and b.log[0][:2] == ("down", "f")  # no thread hand-off before the press
    assert b.log[0][2] - t0 < 0.005
    assert wait_for(lambda: len(b.log) == 2)
    assert b.log[1][:2] == ("up", "f")
    assert b.log[1][2] - b.log[0][2] >= 0.025
    assert wait_for(lambda: c.presses == 1)


def test_mouse_press_and_failures_are_reported():
    b = FakeBackend()
    c = ParryController(backend=b)
    assert c.press("mouse", "f", 0.01)
    assert wait_for(lambda: [e[:2] for e in b.log] == [("down", "mouse"), ("up", "mouse")])
    bad = ParryController(backend=FakeBackend(fail=True))
    assert not bad.press("key", "f", 0.01)
    assert "access denied" in bad.last_error
    assert not ParryController(backend=None, backend_error="no backend").press("key", "f", 0.01)
