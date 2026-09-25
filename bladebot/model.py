"""The parry network: features -> "will the ball hit me within h seconds?".

The network has one sigmoid output per time horizon ``h`` in :data:`HORIZONS`
(0.05 s, 0.10 s, ... 1.00 s). Output ``k`` is the probability that the ball
reaches you within ``HORIZONS[k]`` seconds. The bot parries when the
probability for your chosen *lead time* crosses the confidence threshold, so
you can compensate for ping without retraining the network.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from .features import FEATURE_NAMES, N_FEATURES
from .nn import MLP, sigmoid

HORIZONS: tuple[float, ...] = tuple(round(0.05 * k, 2) for k in range(1, 21))

# When the ball has vanished (usually behind your own character, just before it
# hits) waiting gains nothing, so the bot presses as soon as the ball will very
# likely arrive within the 0.5 s the shield lasts. The same applies to a ball
# that still hasn't been seen a moment after you turned red (it is coming from
# off screen) - but not on the very first frames, where waiting to see the
# launch direction is cheap and saves many early presses on curve balls.
BLIND_AFTER_S = 0.05
BLIND_HORIZON_S = 0.45
BLIND_MAX_STALE_S = 0.6  # a ball gone for longer than this is not "just hidden"
BLIND_UNSEEN_AFTER_S = 0.15
STALE_INDEX = FEATURE_NAMES.index("stale")
SEEN_INDEX = FEATURE_NAMES.index("seen")
EP_AGE_INDEX = FEATURE_NAMES.index("ep_age")

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent
DEFAULT_MODEL_PATH = PROJECT_DIR / "models" / "parry_net.npz"


class ModelVersionError(ValueError):
    """The model file was trained with a different feature set (older/newer BladeBot)."""


class ParryNet:
    """An :class:`~bladebot.nn.MLP` plus input normalisation and horizon metadata."""

    def __init__(
        self,
        net: MLP,
        mean: np.ndarray,
        std: np.ndarray,
        horizons: Sequence[float] = HORIZONS,
        meta: dict[str, Any] | None = None,
    ) -> None:
        self.net = net
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.horizons = np.asarray(horizons, dtype=np.float32)
        self.meta = dict(meta or {})
        if self.mean.shape != (N_FEATURES,):
            raise ModelVersionError(
                f"model expects {self.mean.shape[0]} features but this version computes {N_FEATURES}"
            )
        if net.layer_sizes[-1] != len(self.horizons):
            raise ValueError("network outputs do not match the horizons")

    # ------------------------------------------------------------ inference
    def logits(self, feats: np.ndarray) -> np.ndarray:
        x = (np.asarray(feats, dtype=np.float32) - self.mean) / self.std
        return self.net.forward(x)

    def predict(self, feats: Sequence[float] | np.ndarray) -> np.ndarray:
        """Probabilities P(time-to-impact <= h) for every horizon (monotone in h)."""
        x = np.asarray(feats, dtype=np.float32)
        single = x.ndim == 1
        probs = sigmoid(self.logits(x[None, :] if single else x))
        probs = np.maximum.accumulate(probs, axis=-1)  # enforce monotonicity
        return probs[0] if single else probs

    def prob_within(self, probs: np.ndarray, lead_s: float) -> float:
        """Interpolated probability that impact happens within ``lead_s`` seconds."""
        h = self.horizons
        if lead_s <= 0:
            return 0.0
        if lead_s <= h[0]:
            return float(probs[0] * lead_s / h[0])
        if lead_s >= h[-1]:
            return float(probs[-1])
        return float(np.interp(lead_s, h, probs))

    def prob_within_many(self, probs: np.ndarray, lead_s: np.ndarray | float) -> np.ndarray:
        """Vectorised :meth:`prob_within` for ``(N, H)`` probabilities and ``N`` lead times."""
        h = self.horizons.astype(np.float64)
        probs = np.asarray(probs, dtype=np.float64)
        lead = np.broadcast_to(np.asarray(lead_s, dtype=np.float64), probs.shape[:1])
        grid = np.concatenate([[0.0], h])
        p = np.concatenate([np.zeros((probs.shape[0], 1)), probs], axis=1)
        x = np.clip(lead, 0.0, h[-1])
        k = np.clip(np.searchsorted(grid, x, side="right") - 1, 0, len(grid) - 2)
        w = (x - grid[k]) / (grid[k + 1] - grid[k])
        rows = np.arange(probs.shape[0])
        return p[rows, k] * (1.0 - w) + p[rows, k + 1] * w

    def press_probability_many(
        self,
        probs: np.ndarray,
        lead_s: np.ndarray | float,
        stale_s: np.ndarray,
        seen: Optional[np.ndarray] = None,
        ep_age: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Vectorised :meth:`press_probability`."""
        p = self.prob_within_many(probs, lead_s)
        lead = np.broadcast_to(np.asarray(lead_s, dtype=np.float64), p.shape)
        stale = np.asarray(stale_s, dtype=np.float64)
        seen_a = np.ones(p.shape, bool) if seen is None else np.asarray(seen) > 0
        age = np.full(p.shape, np.inf) if ep_age is None else np.asarray(ep_age, dtype=np.float64)
        blind = (stale >= BLIND_AFTER_S) & (
            (seen_a & (stale <= BLIND_MAX_STALE_S)) | (~seen_a & (age >= BLIND_UNSEEN_AFTER_S))
        )
        if blind.any():
            wide = self.prob_within_many(probs, np.maximum(lead, BLIND_HORIZON_S))
            p = np.where(blind, np.maximum(p, wide), p)
        return p

    def press_probability_for(self, probs: np.ndarray, lead_s: float, feats: Sequence[float]) -> float:
        """:meth:`press_probability` with the hidden-ball details read from a feature vector."""
        return self.press_probability(probs, lead_s, feats[STALE_INDEX], feats[SEEN_INDEX] > 0, feats[EP_AGE_INDEX])

    def press_probability(
        self,
        probs: np.ndarray,
        lead_s: float,
        stale_s: float = 0.0,
        seen: bool = True,
        ep_age: float = float("inf"),
    ) -> float:
        """The probability the parry decision uses.

        Normally ``P(impact within lead_s)``. While the ball is hidden (last seen
        ``stale_s`` seconds ago), or still hasn't been seen ``ep_age`` seconds
        after you turned red, the horizon widens to :data:`BLIND_HORIZON_S`.
        """
        p = self.prob_within(probs, lead_s)
        if stale_s >= BLIND_AFTER_S and (
            (seen and stale_s <= BLIND_MAX_STALE_S) or (not seen and ep_age >= BLIND_UNSEEN_AFTER_S)
        ):
            p = max(p, self.prob_within(probs, max(lead_s, BLIND_HORIZON_S)))
        return p

    def expected_tti(self, probs: np.ndarray) -> float:
        """Expected time-to-impact in seconds, capped at the longest horizon."""
        h = np.concatenate([[0.0], self.horizons])
        p = np.concatenate([[0.0], probs])
        survival = 1.0 - 0.5 * (p[1:] + p[:-1])
        return float(np.sum(survival * np.diff(h)))

    # ------------------------------------------------------------- save/load
    def save(self, path: str | Path, extra: dict[str, Any] | None = None) -> None:
        meta = dict(self.meta)
        meta.update(extra or {})
        meta.update(
            feature_names=list(FEATURE_NAMES),
            horizons=[float(v) for v in self.horizons],
            mean=[float(v) for v in self.mean],
            std=[float(v) for v in self.std],
        )
        self.meta = meta
        self.net.save(path, meta)

    @classmethod
    def load(cls, path: str | Path = DEFAULT_MODEL_PATH) -> "ParryNet":
        net, meta = MLP.load(path)
        names = meta.get("feature_names")
        if names is not None and tuple(names) != FEATURE_NAMES:
            raise ModelVersionError(
                f"{Path(path).name} was made by an older BladeBot version (different inputs) - "
                "retrain it in the Train tab"
            )
        return cls(net, meta["mean"], meta["std"], meta["horizons"], meta)

    def summary(self) -> dict[str, Any]:
        return {
            "layers": self.net.layer_sizes,
            "activation": self.net.activation,
            "parameters": self.net.n_parameters(),
            "horizons": [float(v) for v in self.horizons],
            "trained_on": self.meta.get("trained_on"),
            "created": self.meta.get("created"),
            "metrics": self.meta.get("metrics", {}),
        }
