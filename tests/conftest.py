import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bladebot.model import DEFAULT_MODEL_PATH, ParryNet  # noqa: E402


@pytest.fixture(scope="session")
def bundled_model():
    if not DEFAULT_MODEL_PATH.exists():
        pytest.skip("bundled model not trained yet")
    return ParryNet.load(DEFAULT_MODEL_PATH)
