"""M5.6 experimental self-supervised *learning progress*, not raw surprise.

Each organism builds a separate local model of what its sensory readings
will look like after its selected movement. Initial observed transitions become
a FIXED, NEVER TRAINED probe set; only later real action-consequence pairs
modify predictor weights. Compare prediction loss on exactly the same fixed
probe examples immediately before and after each training update.

A high-entropy/unlearnable transition cannot by itself produce reward:
only actual positive improvement on held-aside examples earns it.

Limitations: fixed early probe can bias progress toward early environments;
one-step local predictor is not a generative world model, long-horizon planning,
or a robust scientific demonstration of epistemic curiosity.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .m5_6_curiosity import IntrinsicReward

INPUTS = 24  # 19 observed sensor values + 4 action one-hot + bias
OUTPUTS = 19
PROBE_SIZE = 4
RATE = 0.035
MAX_WEIGHTS = 2.0


def _features(observation: tuple[float, ...], action: int) -> tuple[float, ...]:
    if (len(observation) != OUTPUTS or type(action) is not int
            or action not in range(4) or
            not all(math.isfinite(v) for v in observation)):
        raise ValueError("Expected 19 finite observations and a cardinal action")
    return tuple(math.tanh(x) for x in observation) + tuple(
        float(i == action) for i in range(4)) + (1.0,)


def _target(observation: tuple[float, ...]) -> tuple[float, ...]:
    if len(observation) != OUTPUTS or not all(
            math.isfinite(v) for v in observation):
        raise ValueError("Expected 19 finite resulting sensory readings")
    return tuple(math.tanh(x) for x in observation)


@dataclass
class LocalPredictor:
    """Personal numerical forward predictor with a strict disjoint probe."""

    weights: list[list[float]] = field(default_factory=lambda: [
        [0.0]*INPUTS for _ in range(OUTPUTS)])
    probes: list[tuple[tuple[float, ...], tuple[float, ...]]] = (
        field(default_factory=list))
    learned_transitions: int = 0
    probe_rewards: int = 0
    total_raw_progress: float = 0.

    def _predict(self, features: tuple[float, ...]) -> tuple[float, ...]:
        return tuple(math.tanh(math.fsum(w*x for w, x in zip(row, features)))
                     for row in self.weights)

    def _loss(self, features: tuple[float, ...],
              target: tuple[float, ...]) -> float:
        prediction = self._predict(features)
        return math.fsum((p-y)**2 for p, y in zip(prediction, target))/OUTPUTS

    def probe_loss(self) -> float | None:
        if len(self.probes) < PROBE_SIZE:
            return None
        return math.fsum(self._loss(f, t) for f, t in self.probes) / PROBE_SIZE

    def transition(self, observation: tuple[float, ...], action: int,
                   observed_next: tuple[float, ...]) -> float:
        """Update from actual transition, reward from distinct OLD probes.

        Never update on any stored probe transition. All gradient features are
        obtained from the agent's own current observation and actual action.
        """
        features = _features(observation, action)
        target = _target(observed_next)
        if len(self.probes) < PROBE_SIZE:
            self.probes.append((features, target))
            return 0.
        loss_before = self.probe_loss()
        prediction = self._predict(features)
        for row, pred, goal in zip(self.weights, prediction, target):
            # squared loss on current experience; tanh derivative
            delta = (goal - pred) * (1.0 - pred*pred) * RATE
            for i, f in enumerate(features):
                row[i] = max(-MAX_WEIGHTS, min(MAX_WEIGHTS, row[i]+delta*f))
        loss_after = self.probe_loss()
        if loss_before is None or loss_after is None:
            raise AssertionError("Probe not initialized")
        self.learned_transitions += 1
        raw_progress = max(0., loss_before-loss_after)
        self.total_raw_progress += raw_progress
        # Reward is bounded within 0.1, exactly as novelty; 4x downstream
        # policy gain is an experimental dose, NOT a matched-reward guarantee.
        reward = IntrinsicReward(progress_scale=4., maximum=.1).learning_progress(
            loss_before, loss_after)
        self.probe_rewards += int(reward > 0)
        return reward
