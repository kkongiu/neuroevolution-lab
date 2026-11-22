"""M5.6 development-only curiosity-conditioned online RNN in native physical ecology.

The original M4.4A and M5.5B modules are not modified. The action policy
updates its input, recurrent, and readout weights from *observed* food and
bounded individual visit-novelty rewards, never from a correct-action label.
Updates use ONE-STEP truncated recurrent policy gradients, not full BPTT.
Reproduction remains an explicitly programmed baseline, not neural learning.
The shared physical world follows M5.0 movement/food/birth and M5.5B predators.
Not preregistered training or a sealed held-out M5.6 survival result.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from . import m4_2, m4_4a, m5_0 as base, m5_1 as actuator, m5_5b
from .m5_6_bridge import ecological_observation, legal_cardinal_moves
from .m5_6_curiosity import IntrinsicReward
from .m5_6_prediction import LocalPredictor
from .m5_7_decision import choose_from_model

ARMS = ("novelty", "no_novelty", "frozen", "shuffled_novelty", "learning_progress")
RATE = 0.02
GRADIENT_CLIP = 2.0


@dataclass
class Learner:
    """Personal, non-heritable learned synapses and sensory history."""
    weights: list[float]
    hidden: list[float] = field(default_factory=lambda: [0.] * 8)
    curiosity: IntrinsicReward = field(default_factory=IntrinsicReward)
    predictor: LocalPredictor = field(default_factory=LocalPredictor)
    update_count: int = 0

    @classmethod
    def fresh(cls, genome: tuple[float, ...], start: base.Point) -> "Learner":
        if len(genome) != 260 or not all(math.isfinite(v) for v in genome):
            raise ValueError("Original M4.4A 260 synapses required")
        a = cls(list(genome))
        a.curiosity.visits[start] = 1  # birth location is known, not a discovery
        return a

    def action(self, obs: tuple[float, ...], legal: tuple[int, ...],
               rng: random.Random) -> tuple[int | None, tuple | None]:
        if not legal:
            return None, None
        if len(obs) != 19 or len(set(legal)) != len(legal) or any(
                type(a) is not int or a not in range(4) for a in legal):
            raise ValueError("Invalid RNN observation/actions")
        prev = self.hidden[:]
        n, stride = 8, 28
        new = []
        for i in range(n):
            offset = i * stride
            z = math.fsum(self.weights[offset+j]*obs[j] for j in range(19))
            z += math.fsum(self.weights[offset+19+j]*prev[j] for j in range(n))
            new.append(math.tanh(z + self.weights[offset+27]))
        self.hidden = new
        head = n * stride
        logits = {a: math.fsum(
            self.weights[head+a*9+j]*new[j] for j in range(n)
        ) + self.weights[head+a*9+8] for a in legal}
        maximum = max(logits.values())
        exp = {a: math.exp(logits[a] - maximum) for a in legal}
        total = sum(exp.values())
        probabilities = {a: exp[a]/total for a in legal}
        ticket = rng.random()
        cumulative = 0.
        selected = legal[-1]
        for a in legal:
            cumulative += probabilities[a]
            if ticket < cumulative:
                selected = a
                break
        return selected, (obs, prev, new, probabilities, selected)

    def learn(self, cache: tuple | None, reward: float,
              *, enabled: bool = True) -> float:
        """One-step REINFORCE: true local gradient through recurrence weights.

        Does not backpropagate through earlier hidden states (truncated BPTT).
        No reinforcement update for a deterministic single legal action.
        """
        if cache is None or not enabled or reward == 0:
            return 0.
        if not math.isfinite(reward) or abs(reward) > 2:
            raise ValueError("Reward outside predeclared safe range")
        obs, prev, hidden, probs, chosen = cache
        if len(probs) < 2:
            return 0.
        n, stride, head = 8, 28, 224
        gradients = [0.] * len(self.weights)
        dh = [0.] * n
        for action, p in probs.items():
            coefficient = reward * (float(action == chosen) - p)
            for i in range(n):
                gradients[head+action*9+i] += coefficient * hidden[i]
                dh[i] += coefficient * self.weights[head+action*9+i]
            gradients[head+action*9+8] += coefficient
        for i in range(n):
            delta = dh[i] * (1 - hidden[i]**2)
            offset = i * stride
            for j in range(19):
                gradients[offset+j] += delta*obs[j]
            for j in range(n):
                gradients[offset+19+j] += delta*prev[j]
            gradients[offset+27] += delta
        norm = math.sqrt(math.fsum(g*g for g in gradients))
        scale = min(1., GRADIENT_CLIP / max(norm, 1e-12))
        drift = 0.
        for i, old in enumerate(self.weights):
            updated = max(-4., min(4., old + RATE*scale*gradients[i]))
            drift += (updated - old)**2
            self.weights[i] = updated
        self.update_count += 1
        return math.sqrt(drift)


def simulate_development(seed: int, settings: m5_5b.Settings, arm: str,
                         *, predator_count: int = 1,
                         curiosity_gain: float = 1.,
                         energy_gate: bool = False,
                         learning_cutoff_tick: int | None = None,
                         reset_weights_at_cutoff: bool = False,
                         reset_hidden_at_cutoff: bool = False,
                         model_decision_mode: str = "disabled") -> dict:
    """Small reproducible training run; must never consume final world seeds."""
    if model_decision_mode not in ("disabled", "model", "permuted"):
        raise ValueError("Unregistered M5.7 decision intervention")
    if model_decision_mode != "disabled" and (arm != "learning_progress" or learning_cutoff_tick is None):
        raise ValueError("M5.7 model-based decision requires prediction training cutoff")
    if arm not in ARMS or predator_count not in m5_5b.PREDATOR_LEVELS:
        raise ValueError("Unregistered development arm/predator count")
    settings.validate()
    if (not math.isfinite(curiosity_gain) or curiosity_gain < 0 or
            curiosity_gain > 4):
        raise ValueError("Development curiosity gain must be finite in [0, 4]")
    if seed not in range(1000, 1100) or settings.ticks > 60:
        raise ValueError("Development-only worlds: seeds 1000–1099, <=60 ticks")
    if learning_cutoff_tick is None:
        if reset_weights_at_cutoff or reset_hidden_at_cutoff:
            raise ValueError("A cutoff is required for retention interventions")
    elif not (1 <= learning_cutoff_tick < settings.ticks):
        raise ValueError("Cutoff must precede the final tick")
    eco = base.create_ecology(seed, settings, "heritable_no_mutation")
    # Predator capture audits victim genetic identity using the original
    # nine-gene M5.5B contract. Keep neural weights in the separate personal
    # Learner: they are never mistaken for inherited predator/fear genes.
    source_genes = m5_5b.founder_genes(seed, settings)
    for agent, genes in zip(eco.agents.values(), source_genes):
        agent.genome = genes
    eco.initial_genome_digests = tuple(
        base.genome_digest(genes) for genes in source_genes
    )
    predators = m5_5b.initial_predators(eco, predator_count)
    food_schedule = base._food_schedule(seed, settings, eco, False)
    # Source weights are the only hereditary neural component; each child
    # starts with untrained weights, never the parent's in-life updates.
    initial = {
        a.id: tuple(m4_4a._initial_genome(124000+a.founder,
                                         "rnn", m4_2.Settings()))
        for a in eco.agents.values()
    }
    learners = {a.id: Learner.fresh(initial[a.id], a.position)
                for a in eco.agents.values()}
    initial_fingerprints = {a.id: tuple(learners[a.id].weights)
                            for a in eco.agents.values()}
    update_count = 0
    positive_curiosity = 0.
    positive_food_reward = 0.
    actual_movements = 0
    predation_kills = 0
    energy_before_actions = 0.
    action_count = 0
    unique_arrivals = 0
    predictor_train_transitions = 0
    predictor_reward_events = 0
    predictor_raw_progress_total = 0.
    model_decisions = 0
    rnn_fallbacks = 0
    survived_at_cutoff = None
    births_at_cutoff = None
    food_at_cutoff = None
    predation_at_cutoff = None
    synaptic_updates_at_cutoff = None
    for tick in range(1, settings.ticks + 1):
        # Paired counterfactual branches have identical pre-cutoff ecology,
        # agency, updates, food and predator states. Both can erase transient
        # recurrent hidden activations; only one erases learned synapses.
        if learning_cutoff_tick is not None and tick == learning_cutoff_tick + 1:
            if reset_hidden_at_cutoff:
                for learner in learners.values():
                    learner.hidden = [0.] * len(learner.hidden)
            if reset_weights_at_cutoff:
                for aid, learner in learners.items():
                    learner.weights = list(initial[aid])
        training_phase = (learning_cutoff_tick is None
                          or tick <= learning_cutoff_tick)
        ids = list(eco.agents)
        base._local_rng(seed, tick, 0, 0).shuffle(ids)
        for aid in ids:
            agent = eco.agents.get(aid)
            if agent is None:
                continue
            learner = learners[aid]
            legal = legal_cardinal_moves(eco, agent)
            old_position = agent.position
            old_food = eco.food_eaten
            energy_before_actions += agent.energy
            action_count += 1
            # Programmed reproduction is explicitly a shared baseline,
            # not learned. Native birth costs and inherited original genotype.
            _, free = base.blocked_and_free(eco, agent)
            birth = (base.can_reproduce(eco, agent, free)
                     and agent.energy >= 46.
                     and base._local_rng(seed, tick, aid, 84).random() < .20)
            cache = None
            if birth:
                action = base.REPRODUCE
            elif legal:
                obs = ecological_observation(eco, agent, tuple(predators))
                action, cache = learner.action(
                    obs, legal, base._local_rng(seed, tick, aid, 118))
                if not training_phase and model_decision_mode != "disabled":
                    selected = choose_from_model(
                        learner.predictor, obs, legal,
                        mode=model_decision_mode,
                        rng=base._local_rng(seed, tick, aid, 217))
                    if selected is not None:
                        action = selected
                        model_decisions += 1
                    else:
                        rnn_fallbacks += 1
            else:
                action = None
            previous_id = eco.next_id
            actuator._act_scripted(eco, aid, tick, action)
            if eco.next_id != previous_id:
                child = eco.agents[previous_id]
                initial[child.id] = initial[aid]
                learners[child.id] = Learner.fresh(initial[aid], child.position)
            new_agent = eco.agents.get(aid)
            # The action and food event are physical. Novelty uses visited
            # location only, and is NEVER assigned to physical body energy.
            novelty = 0.
            if action is not None and action in range(4):
                dx, dy = base.DIRECTIONS[action]
                arrived = (old_position[0]+dx, old_position[1]+dy)
                first_visit = arrived not in learner.curiosity.visits
                novelty = learner.curiosity.observe_arrival(arrived)
                unique_arrivals += int(first_visit)
                actual_movements += 1
            # Predict only the physically observed next sensory state. The
            # predictor does not receive the map, hidden food or future state.
            progress = 0.
            if (arm == "learning_progress" and training_phase
                    and cache is not None and aid in eco.agents
                    and action is not None and action in range(4)):
                current = ecological_observation(eco, eco.agents[aid],
                                                 tuple(predators))
                progress = learner.predictor.transition(cache[0], action, current)
                predictor_train_transitions += int(
                    learner.predictor.learned_transitions > 0)
                predictor_reward_events += int(progress > 0)
                predictor_raw_progress_total += progress
            external = float(eco.food_eaten - old_food)
            if arm == "shuffled_novelty" and action is not None and action in range(4):
                novelty = learner.curiosity.maximum * base._local_rng(
                    seed, tick, aid, 119).random()
            elif arm == "learning_progress":
                novelty = progress
            elif arm != "novelty":
                novelty = 0.
            # This optional *engineered* energy gate is a diagnostic control,
            # NOT evidence that survival-vs-curiosity tradeoffs emerged.
            if energy_gate:
                novelty *= max(0., min(1., agent.energy / settings.max_energy))
            novelty *= curiosity_gain
            if not training_phase:
                novelty = 0.
            reward = external + novelty if training_phase else 0.
            drift = learner.learn(cache, reward,
                                  enabled=arm != "frozen" and training_phase)
            positive_curiosity += novelty
            positive_food_reward += external
            update_count += int(drift > 0)
            if abs(base.energy_balance(eco)) > 1e-6:
                raise AssertionError("Intrinsic reward violated energy balance")
            if new_agent is not None and new_agent.position != old_position and (
                    action is None or action == base.REPRODUCE):
                raise AssertionError("Unexpected movement")
        predation_kills += m5_5b.step_predators(eco, predators, tick)
        if abs(base.energy_balance(eco)) > 1e-6:
            raise AssertionError("Predator capture violated energy balance")
        # Real scheduled food regrowth, native rules.
        # Use the original keyed schedule rather than arbitrary food gifts.
        base._food_regeneration(eco, food_schedule[tick-1])
        eco.ticks_elapsed = tick
        if not eco.agents and eco.extinct_at is None:
            eco.extinct_at = tick
        if learning_cutoff_tick is not None and tick == learning_cutoff_tick:
            survived_at_cutoff = int(bool(eco.agents))
            births_at_cutoff = eco.births
            food_at_cutoff = eco.food_eaten
            predation_at_cutoff = predation_kills
            synaptic_updates_at_cutoff = update_count
    return {
        "seed": seed, "arm": arm, "predators": predator_count,
        "learning_cutoff_tick": learning_cutoff_tick,
        "model_decision_mode": model_decision_mode,
        "post_cutoff_model_decisions": model_decisions,
        "post_cutoff_rnn_fallbacks": rnn_fallbacks,
        "reset_weights_at_cutoff": reset_weights_at_cutoff,
        "reset_hidden_at_cutoff": reset_hidden_at_cutoff,
        "survived_at_cutoff": survived_at_cutoff,
        "post_cutoff_food_eaten": (eco.food_eaten - food_at_cutoff
                                 if food_at_cutoff is not None else None),
        "post_cutoff_births": (eco.births - births_at_cutoff
                              if births_at_cutoff is not None else None),
        "post_cutoff_predation_kills": (predation_kills - predation_at_cutoff
                                      if predation_at_cutoff is not None else None),
        "post_cutoff_weight_updates": (update_count - synaptic_updates_at_cutoff
                                      if synaptic_updates_at_cutoff is not None else None),
        "curiosity_gain": curiosity_gain, "energy_gate": energy_gate,
        "predictor_reward_events": predictor_reward_events,
        "predictor_reward_total": predictor_raw_progress_total,
        "predictor_training_transitions": predictor_train_transitions,
        "physical_moves": actual_movements,
        "unique_individual_arrivals": unique_arrivals,
        "predation_kills": predation_kills,
        "mean_action_start_energy": (energy_before_actions / action_count
                                     if action_count else 0.),
        "survived": int(bool(eco.agents)), "population": len(eco.agents),
        "births": eco.births, "food_eaten": eco.food_eaten,
        "novelty_reward_total": positive_curiosity,
        "physical_food_reward_total": positive_food_reward,
        "updates_with_nonzero_drift": update_count,
        "weights_changed": sum(tuple(learners[i].weights) != w
                               for i, w in initial_fingerprints.items()),
        "energy_residual": base.energy_balance(eco),
        "claim": "development only; no held-out curiosity benefit demonstrated",
    }
