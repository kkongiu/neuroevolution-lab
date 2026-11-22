"""M5.7 DEVELOPMENT-only one-step choice using a locally learned world model.

The forward model was trained only on the agent's own experienced sensory
transitions. Score each *legal* hypothetical next observation, without access
to real future food/predators/map. Explicit engineered utility: expected
next-step food signal, energy reserve, locally sensed predator proximity.
This utility is an externally specified survival objective, NOT an emergent
value function or learned risk preference. The real causal test is whether
a trained model's decisions outperform disabling/permuting model access.
"""
from __future__ import annotations

import math
import random

from .m5_6_prediction import LocalPredictor, _features

# Original M5.0 sensor indices plus 4 predator-adjacent flags.
NEXT_ENERGY = 12
NEXT_FOOD_CONSUMED = 14
NEXT_PREDATORS = (15, 16, 17, 18)
MODELS = ("disabled", "model", "permuted")


def predicted_value(next_sensors: tuple[float, ...]) -> float:
    """Fixed transparent homeostatic proxy; not a handcoded correct action."""
    if len(next_sensors) != 19 or not all(
            math.isfinite(x) for x in next_sensors):
        raise ValueError("Expected 19 finite *predicted* readings")
    return (2.0 * next_sensors[NEXT_FOOD_CONSUMED]
            + .5 * next_sensors[NEXT_ENERGY]
            - 1.5 * sum(max(0., next_sensors[i]) for i in NEXT_PREDATORS))


def choose_from_model(predictor: LocalPredictor,
                      observation: tuple[float, ...],
                      legal: tuple[int, ...],
                      *, mode: str,
                      rng: random.Random | None = None) -> int | None:
    """Pure counterfactual inference; never changes model or environment.

    Requires >=4 probe observations and >=1 *actual* gradient-trained
    transition, otherwise returns None (RNN policy remains in charge).
    """
    if mode not in MODELS:
        raise ValueError("Unknown model-access intervention")
    if mode == "disabled" or not legal:
        return None
    if predictor.learned_transitions < 1 or len(predictor.probes) < 4:
        return None
    if len(legal) != len(set(legal)) or any(
            type(a) is not int or a not in range(4) for a in legal):
        raise ValueError("Invalid motor alternatives")
    if mode == "permuted" and rng is None:
        raise ValueError("Permuted model requires seeded RNG")
    actions = tuple(sorted(legal))
    # For the placebo, break the action-to-forecast association by permuting
    # the same model's forecasts. Preserve the set of predicted outcomes,
    # and all original neural weights and physical world state.
    forecast_actions = list(actions)
    if mode == "permuted":
        assert rng is not None
        rng.shuffle(forecast_actions)
        if len(actions) > 1 and forecast_actions == list(actions):
            forecast_actions = forecast_actions[1:] + forecast_actions[:1]
    values = {}
    for real_action, imagined_action in zip(actions, forecast_actions):
        prediction = predictor._predict(_features(observation, imagined_action))
        values[real_action] = predicted_value(prediction)
    return max(actions, key=lambda a: (values[a], -a))
