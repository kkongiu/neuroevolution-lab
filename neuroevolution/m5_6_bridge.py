"""M5.6 experimental RNN-to-ecology sensory/motor bridge.

Pre-integration contract only: *not* an autonomous survival experiment.
M4.4A's 19-channel architecture is reused, but its corridor-trained
weights have no validated meaning with this NEW ecological observation mapping.
No held-out worlds, gradients, or physical ecology are executed here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from . import m4_2, m5_0, m5_5b


@dataclass
class EcologicalRNN:
    weights: tuple[float, ...]
    hidden: list[float]

    @classmethod
    def create(cls, weights: tuple[float, ...]) -> "EcologicalRNN":
        settings = m4_2.Settings()
        if len(weights) != m4_2.genome_size("rnn", settings):
            raise ValueError("Expected exact M4.4A 260-weight recurrent architecture")
        if not all(math.isfinite(v) for v in weights):
            raise ValueError("All neural weights must be finite")
        return cls(weights, [0.] * settings.hidden)

    def reset(self) -> None:
        """Reset at newborn initialization or explicitly for memory ablation."""
        self.hidden = [0.] * len(self.hidden)

    def decide(self, observation: tuple[float, ...],
               legal_moves: tuple[int, ...]) -> int | None:
        """One causal recurrent update; only locally legal cardinal actions."""
        n = len(self.hidden)
        if len(observation) != m4_2.INPUTS or not all(
                math.isfinite(x) for x in observation):
            raise ValueError("Expected 19 finite, currently observed channels")
        if not legal_moves:
            return None
        if len(set(legal_moves)) != len(legal_moves) or any(
                type(a) is not int or a < 0 or a >= m4_2.ACTIONS
                for a in legal_moves):
            raise ValueError("Invalid local motor actions")
        stride = m4_2.INPUTS + n + 1
        prev = self.hidden
        next_hidden = []
        for i in range(n):
            start = i * stride
            net = math.fsum(self.weights[start + j] * observation[j]
                            for j in range(m4_2.INPUTS))
            net += math.fsum(
                self.weights[start + m4_2.INPUTS + j] * prev[j]
                for j in range(n))
            net += self.weights[start + stride - 1]
            next_hidden.append(math.tanh(net))
        self.hidden = next_hidden
        head = n * stride
        scores = {
            a: math.fsum(self.weights[head + a*(n+1) + j] *
                         self.hidden[j] for j in range(n))
               + self.weights[head + a*(n+1) + n]
            for a in legal_moves
        }
        return max(legal_moves, key=lambda a: (scores[a], -a))


def ecological_observation(eco: m5_0.Ecology, agent: m5_0.Agent,
                           predators: tuple[m5_0.Point, ...]) -> tuple[float, ...]:
    """Existing 15 local sensory channels + four local predator direction flags.

    Direction flags are 1 when a predator is directly adjacent in that
    cardinal direction; no global map, hidden targets or future state.
    """
    blocked, _ = m5_0.blocked_and_free(eco, agent)
    existing = tuple(m5_0.observe(eco, agent, blocked))
    if len(existing) != m5_0.INPUTS:
        raise AssertionError("Native M5 observation contract changed")
    x, y = agent.position
    nearby = set(predators)
    danger = tuple(float((x + dx, y + dy) in nearby)
                   for dx, dy in m5_0.DIRECTIONS)
    observation = existing + danger
    if len(observation) != m4_2.INPUTS:
        raise AssertionError("Unexpected RNN sensory dimensionality")
    return observation


def legal_cardinal_moves(eco: m5_0.Ecology,
                         agent: m5_0.Agent) -> tuple[int, ...]:
    blocked, _ = m5_0.blocked_and_free(eco, agent)
    return tuple(i for i, is_blocked in enumerate(blocked) if not is_blocked)
