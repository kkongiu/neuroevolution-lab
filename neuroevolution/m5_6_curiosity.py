"""M5.6 candidate intrinsic-reward signals (not yet integrated into ecology).

This module does NOT simulate learning, survival, predators, or neural memory.
Per-individual state must be constructed at birth and never shared across worlds.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class IntrinsicReward:
    """Finite bounded curiosity rewards, kept separate from physical energy.

    Novelty is based ONLY on locations actually entered by this individual.
    Learning progress must be supplied as comparable pre/post prediction losses
    from a fixed probe set, not losses on different or secretly held-out data.
    """
    novelty_scale: float = 0.1
    progress_scale: float = 0.1
    maximum: float = 0.1
    visits: dict[tuple[int, int], int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not all(math.isfinite(x) and x >= 0 for x in
                   (self.novelty_scale, self.progress_scale, self.maximum)):
            raise ValueError("Reward scales and cap must be finite, nonnegative")

    def observe_arrival(self, location: tuple[int, int]) -> float:
        """Novelty reward issued only for an actual arrival at a visited cell."""
        if (not isinstance(location, tuple) or len(location) != 2 or
                any(type(v) is not int for v in location)):
            raise ValueError("Location must be a pair of integer coordinates")
        previous_visits = self.visits.get(location, 0)
        self.visits[location] = previous_visits + 1
        return min(self.maximum,
                   self.novelty_scale / math.sqrt(1 + previous_visits))

    def learning_progress(self, loss_before: float, loss_after: float) -> float:
        """Positive reduction in SAME-probe prediction loss, never raw surprise.

        Caller is responsible for obtaining honest comparable losses and for
        excluding probe samples from the gradient update used between them.
        """
        if not (math.isfinite(loss_before) and math.isfinite(loss_after)):
            raise ValueError("Losses must be finite")
        if loss_before < 0 or loss_after < 0:
            raise ValueError("Prediction losses must be nonnegative")
        return min(self.maximum,
                   self.progress_scale * max(0.0, loss_before - loss_after))
