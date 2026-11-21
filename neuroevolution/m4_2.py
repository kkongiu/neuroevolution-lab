"""M4.2: delayed spatial cues, neural state-memory ablation and matched FF.

This M4.2 module does not modify M4.0/M4.1. Eight independent evolutionary
replicates (recurrent vs near-parameter-matched feedforward) face pairs of
opposite-goal T-mazes. Neither policy receives a food-odor or target-location
signal once the starting cue disappears. The motor action mask is shared.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

from .m4_0 import (
    DIRECTIONS, World, initial_state, observation as reactive_observation,
    apply_action, decide_baseline, render_world_svg,
)
from .m4_1 import (
    Settings as PreviousSettings, hierarchical_bootstrap, make_diverse_world,
)

Point = tuple[int, int]
INPUTS = 19
ACTIONS = 4
DELAYED_ARMS = (
    "evolved_rnn", "evolved_ff", "rnn_reset_at_fork", "rnn_shuffled_cue",
    "legal_random", "scripted_memory", "rnn_no_cue",
)
REACTIVE_ARMS = (
    "rnn_reactive", "rnn_reactive_no_odor", "ff_reactive",
    "legal_random", "greedy_odor",
)
CONTRASTS = (
    ("evolved_rnn", "evolved_ff", "primary_rnn_vs_ff"),
    ("evolved_rnn", "rnn_reset_at_fork", "reset_memory_effect"),
    ("evolved_rnn", "rnn_shuffled_cue", "cue_flip_effect"),
    ("evolved_rnn", "rnn_no_cue", "cue_removal_effect"),
    ("evolved_rnn", "legal_random", "rnn_vs_legal_random"),
    ("evolved_rnn", "scripted_memory", "rnn_vs_scripted"),
)


@dataclass(frozen=True)
class Settings(PreviousSettings):
    train_start: int = 55000
    train_worlds: int = 10
    validation_start: int = 56000
    validation_worlds: int = 6
    test_start: int = 57000
    test_worlds: int = 32
    shifted_start: int = 57500
    shifted_worlds: int = 16
    genetic_seed_start: int = 85000
    genetic_runs: int = 8
    hidden: int = 8
    ff_hidden: int = 11
    population: int = 12
    generations: int = 6
    stem_min: int = 3
    stem_max: int = 5
    shifted_stem_min: int = 5
    shifted_stem_max: int = 6
    cue_ticks: int = 2
    horizon: int = 20
    reactive_start: int = 58000
    reactive_worlds: int = 12

    def validate(self) -> None:
        super().validate()
        if self.size != 13 or self.ff_hidden < 1:
            raise ValueError("M4.2 requires fixed 13x13 paired maze and FF width")
        if (self.stem_min < 3 or self.stem_max > 6
                or self.stem_min > self.stem_max
                or self.shifted_stem_min < 3 or self.shifted_stem_max > 6
                or self.shifted_stem_min > self.shifted_stem_max):
            raise ValueError("Stem corridor length must be between 3 and 6")
        if not 1 <= self.cue_ticks < self.stem_min:
            raise ValueError("Cue must end before a decision can be reached")
        if self.horizon < self.shifted_stem_max + 2:
            raise ValueError("Time horizon too small for shifted task")
        other_ranges = (
            (self.train_start, self.train_worlds * self.generations),
            (self.validation_start, self.validation_worlds),
            (self.test_start, self.test_worlds),
            (self.shifted_start, self.shifted_worlds),
        )
        for start, n in other_ranges:
            if set(range(self.reactive_start, self.reactive_start + self.reactive_worlds)).intersection(range(start, start + n)):
                raise ValueError("Reactive transfer seeds overlap other splits")


def rotate(point: Point, quarter_turns: int, reflected: bool, size: int) -> Point:
    x, y = point
    if reflected:
        x = size - 1 - x
    for _ in range(quarter_turns):
        x, y = size - 1 - y, x
    return x, y


def direction(src: Point, dest: Point) -> int:
    delta = (dest[0] - src[0], dest[1] - src[1])
    if delta not in DIRECTIONS:
        raise ValueError("Only adjacent cardinal moves supported")
    return DIRECTIONS.index(delta)


@dataclass(frozen=True)
class TrialWorld:
    seed: int
    side: int
    size: int
    start: Point
    fork: Point
    gates: tuple[Point, Point]
    walls: frozenset[Point]
    forward: int
    target_direction: int
    stem_length: int
    quarter_turns: int
    reflected: bool
    shifted: bool

    def open_at(self, p: Point) -> list[int]:
        return [
            action for action, (dx, dy) in enumerate(DIRECTIONS)
            if (p[0] + dx, p[1] + dy) not in self.walls
        ]

    @property
    def target(self) -> Point:
        return self.gates[self.side]


def make_pair(seed: int, settings: Settings,
              shifted: bool = False) -> tuple[TrialWorld, TrialWorld]:
    """Two worlds with *identical* geometry, start and wall sensors."""
    rng = random.Random(seed * 19_913 + 227)
    quarter = rng.randrange(4)
    reflected = bool(rng.randrange(2))
    min_length, max_length = (
        (settings.shifted_stem_min, settings.shifted_stem_max)
        if shifted else (settings.stem_min, settings.stem_max)
    )
    length = rng.randint(min_length, max_length)
    size = settings.size
    canonical_fork = (size // 2, 5)
    canonical_start = (size // 2, 5 + length)
    canonical_gates = ((size // 2 - 1, 5), (size // 2 + 1, 5))
    open_cells = {
        (size // 2, y) for y in range(5, 6 + length)
    } | set(canonical_gates)
    walls = frozenset(
        (x, y) for x in range(size) for y in range(size)
        if (x, y) not in {
            rotate(p, quarter, reflected, size) for p in open_cells
        }
    )
    start = rotate(canonical_start, quarter, reflected, size)
    fork = rotate(canonical_fork, quarter, reflected, size)
    gates = tuple(rotate(p, quarter, reflected, size) for p in canonical_gates)
    first_step = rotate((canonical_start[0], canonical_start[1] - 1),
                        quarter, reflected, size)
    forward = direction(start, first_step)
    worlds = []
    for side in (0, 1):
        target_direction = direction(fork, gates[side])
        w = TrialWorld(
            seed, side, size, start, fork, gates, walls, forward,
            target_direction, length, quarter, reflected, shifted
        )
        if w.open_at(start) != [forward]:
            raise AssertionError("Start must have a single legal direction")
        if sorted(w.open_at(fork)) != sorted((forward ^ 2, direction(fork, gates[0]),
                                             direction(fork, gates[1]))):
            # Cardinal index opposites differ by two.
            raise AssertionError("Fork must expose two branches plus approach")
        worlds.append(w)
    return tuple(worlds)


def genome_size(architecture: str, settings: Settings) -> int:
    if architecture == "rnn":
        n = settings.hidden
        return n * (INPUTS + n + 1) + ACTIONS * (n + 1)
    if architecture == "ff":
        n = settings.ff_hidden
        return n * (INPUTS + 1) + ACTIONS * (n + 1)
    raise ValueError(architecture)


def fresh_genome(rng: random.Random, settings: Settings,
                 architecture: str) -> tuple[float, ...]:
    return tuple(rng.gauss(0., settings.initial_weight_sd)
                 for _ in range(genome_size(architecture, settings)))


def mutate(genome: tuple[float, ...], rng: random.Random,
           settings: Settings) -> tuple[float, ...]:
    return tuple(max(-5., min(5., weight + rng.gauss(0., settings.mutation_sd)))
                 for weight in genome)


class Controller:
    def __init__(self, genome: tuple[float, ...], architecture: str,
                 settings: Settings):
        if len(genome) != genome_size(architecture, settings):
            raise ValueError("Genome does not fit requested neural architecture")
        self.genome = genome
        self.architecture = architecture
        self.n = settings.hidden if architecture == "rnn" else settings.ff_hidden
        self.hidden = [0.] * self.n
        self.steps = 0
        self.illegal = 0
        self.corrected = 0

    def reset_memory(self) -> None:
        self.hidden = [0.] * self.n

    def choose(self, inputs: tuple[float, ...]) -> int:
        if len(inputs) != INPUTS:
            raise ValueError("Observation dimensions wrong")
        h = []
        stride = INPUTS + (self.n if self.architecture == "rnn" else 0) + 1
        for i in range(self.n):
            off = i * stride
            weight_input = sum(w * x for w, x in zip(
                self.genome[off:off + INPUTS], inputs
            ))
            weight_recurrent = (
                sum(w * val for w, val in zip(
                    self.genome[off + INPUTS:off + INPUTS + self.n],
                    self.hidden
                )) if self.architecture == "rnn" else 0.
            )
            h.append(math.tanh(weight_input + weight_recurrent
                               + self.genome[off + stride - 1]))
        self.hidden = h
        head = self.n * stride
        logits = [
            sum(w * val for w, val in zip(
                self.genome[head + action * (self.n + 1):
                            head + action * (self.n + 1) + self.n],
                self.hidden
            )) + self.genome[head + action * (self.n + 1) + self.n]
            for action in range(ACTIONS)
        ]
        preference = max(range(ACTIONS), key=lambda a: (logits[a], -a))
        if inputs[preference] > .5:
            self.illegal += 1
        legal = [action for action in range(ACTIONS) if inputs[action] < .5]
        if not legal:
            return preference
        action = max(legal, key=lambda a: (logits[a], -a))
        self.corrected += int(action != preference)
        self.steps += 1
        return action


def observation(world: TrialWorld, position: Point, steps: int,
                energy: float, last_collision: int, last_food: int,
                settings: Settings, *, flip_cue: bool = False,
                no_cue: bool = False) -> tuple[float, ...]:
    wall_bits = tuple(float(i not in world.open_at(position))
                      for i in range(ACTIONS))
    cue = [0.] * ACTIONS
    if steps < settings.cue_ticks and not no_cue:
        action = (
            direction(world.fork, world.gates[1 - world.side])
            if flip_cue else world.target_direction
        )
        cue[action] = 1.
    values = (
        wall_bits + (0.,) * 8 +
        (1.0,  # Delayed memory benchmark: hide energy/time as external cue
         float(last_collision), float(last_food)) + tuple(cue)
    )
    if len(values) != INPUTS:
        raise AssertionError("Wrong sensor shape")
    return values


def simulate(world: TrialWorld, settings: Settings, arm: str,
             genome: tuple[float, ...] | None = None,
             trace: bool = False) -> dict:
    if arm not in DELAYED_ARMS:
        raise ValueError(arm)
    is_neural = arm not in ("legal_random", "scripted_memory")
    if is_neural and genome is None:
        raise ValueError("Neural arm needs a genome")
    architecture = "ff" if arm == "evolved_ff" else "rnn"
    agent = Controller(genome, architecture, settings) if is_neural else None
    rng = random.Random(world.seed * 3391 + 13)
    position = world.start
    energy = settings.initial_energy
    steps = 0
    visited = {position}
    max_progress = 0
    arrived = False
    committed = False
    branch_correct = False
    food = 0
    corrected = 0
    illegal = 0
    cue_memory = None
    heading_memory = None
    last_collision = 0
    last_food = 0
    path = [position] if trace else []
    events = []
    fork_reset_done = False
    while energy > 0 and steps < settings.horizon and not committed:
        obs = observation(
            world, position, steps, energy, last_collision,
            last_food, settings,
            flip_cue=arm == "rnn_shuffled_cue",
            no_cue=arm == "rnn_no_cue",
        )
        if position == world.fork and not arrived:
            arrived = True
        if is_neural:
            if arm == "rnn_reset_at_fork" and arrived and not fork_reset_done:
                agent.reset_memory()
                fork_reset_done = True
            action = agent.choose(obs)
            corrected = agent.corrected
            illegal = agent.illegal
        elif arm == "legal_random":
            allowed = [a for a in range(ACTIONS) if obs[a] == 0.]
            action = rng.choice(allowed)
        else:
            if cue_memory is None:
                cue_memory = obs[15:].index(1.)  # read cue, not true hidden goal
            if heading_memory is None:
                heading_memory = obs[:4].index(0.)  # sole legal first movement
            if position == world.fork:
                action = cue_memory
            else:
                action = heading_memory
            if obs[action] > 0.5:
                raise AssertionError("Scripted memory selected blocked movement")
        dx, dy = DIRECTIONS[action]
        nxt = (position[0] + dx, position[1] + dy)
        if nxt in world.walls:
            raise AssertionError("Legal decoder allowed a wall collision")
        position = nxt
        steps += 1
        energy = max(0., energy - settings.movement_cost)
        visited.add(position)
        progress = world.stem_length - abs(
            world.fork[0] - position[0]) - abs(world.fork[1] - position[1])
        max_progress = max(max_progress, min(world.stem_length, progress))
        if position in world.gates:
            committed = True
            branch_correct = position == world.target
            if branch_correct:
                food = 1
                last_food = 1
                energy = min(settings.max_energy, energy + settings.food_energy)
        if trace:
            path.append(position)
            events.append({
                "step": steps, "position": list(position),
                "cue_present": int(any(obs[15:])),
                "action": action, "food": food,
                "reached_fork": int(arrived),
                "energy": round(energy, 4),
            })
    fitness = (
        12 * food + 4 * int(branch_correct)
        + .15 * max_progress - .03 * steps - .08 * corrected
    )
    return {
        "world_seed": world.seed,
        "side": world.side,
        "arm": arm,
        "shifted": world.shifted,
        "success": food,
        "fork_reached": int(arrived),
        "committed": int(committed),
        "correct_branch": int(branch_correct),
        "steps": steps,
        "corrected": corrected,
        "illegal_preferences": illegal,
        "collisions": 0,
        "visited": len(visited),
        "starved": int(energy <= 0),
        "end_energy": energy,
        "max_stem_progress": max_progress,
        "fitness": fitness,
        "path": path,
        "events": events,
    }


def evolve_one(settings: Settings, genetic_seed: int,
               architecture: str) -> tuple[tuple[float, ...], list[dict], dict]:
    """No final-world data enters genotype selection."""
    rng = random.Random(genetic_seed)
    population = [fresh_genome(rng, settings, architecture)
                  for _ in range(settings.population)]
    validation = [
        pair for seed in range(settings.validation_start,
                               settings.validation_start + settings.validation_worlds)
        for pair in make_pair(seed, settings)
    ]
    winner_arm = "evolved_rnn" if architecture == "rnn" else "evolved_ff"
    leaders = []
    history = []
    candidate_evaluations = 0
    for generation in range(settings.generations):
        training = [
            pair for seed in range(
                settings.train_start + generation * settings.train_worlds,
                settings.train_start + (generation + 1) * settings.train_worlds
            ) for pair in make_pair(seed, settings)
        ]
        ranked = []
        for genome in population:
            results = [simulate(w, settings, winner_arm, genome)
                       for w in training]
            candidate_evaluations += len(results)
            ranked.append((
                statistics.mean(r["fitness"] for r in results),
                genome,
                statistics.mean(r["success"] for r in results),
            ))
        ranked.sort(key=lambda r: r[0], reverse=True)
        leader = ranked[0]
        val_metrics = [simulate(w, settings, winner_arm, leader[1])
                       for w in validation]
        val_score = statistics.mean(r["fitness"] for r in val_metrics)
        leaders.append((val_score, generation, leader[1]))
        history.append({
            "genetic_seed": genetic_seed,
            "architecture": architecture,
            "generation": generation,
            "best_training_fitness": leader[0],
            "mean_training_fitness": statistics.mean(r[0] for r in ranked),
            "best_training_success": leader[2],
            "validation_fitness": val_score,
            "validation_success": statistics.mean(r["success"] for r in val_metrics),
            "cumulative_candidate_trials": candidate_evaluations,
        })
        print(f"M4.2 evolution gene={genetic_seed} arch={architecture}"
              f" gen={generation} train={leader[2]:.3f}"
              f" val={history[-1]['validation_success']:.3f}", flush=True)
        elites = [x[1] for x in ranked[:settings.elites]]
        population = list(elites)
        while len(population) < settings.population - settings.immigrants:
            population.append(mutate(rng.choice(elites), rng, settings))
        while len(population) < settings.population:
            population.append(fresh_genome(rng, settings, architecture))
    val_fitness, gen, champion = max(
        leaders, key=lambda r: (r[0], -r[1])
    )
    expected = (settings.generations * settings.population *
                settings.train_worlds * 2)
    if candidate_evaluations != expected:
        raise AssertionError("Genetic trials do not match registered budget")
    return champion, history, {
        "genetic_seed": genetic_seed,
        "architecture": architecture,
        "selected_generation": gen,
        "validation_fitness": val_fitness,
        "candidate_trials": expected,
        "validation_trials": settings.generations * settings.validation_worlds * 2,
        "genome_weights": len(champion),
    }


def reactive_trial(world: World, settings: Settings, arm: str,
                   genome: tuple[float, ...] | None) -> dict:
    """Exploratory task transfer: no training on open reactive food mazes."""
    if arm not in REACTIVE_ARMS:
        raise ValueError(arm)
    ctrl = None
    if arm in REACTIVE_ARMS[:3]:
        ctrl = Controller(
            genome, "ff" if arm == "ff_reactive" else "rnn", settings
        )
    state = initial_state(world, settings)
    rng = random.Random(world.seed * 1717 + (0 if arm == "legal_random" else 97) + 7)
    while state.energy > 0 and state.steps < settings.horizon:
        signals = reactive_observation(world, state, settings)
        if ctrl:
            padded = (
                signals[:4] + (0.,)*4 + signals[8:] + (0.,)*4
                if arm == "rnn_reactive_no_odor" else signals + (0.,)*4
            )
            action = ctrl.choose(padded)
        else:
            action = decide_baseline(
                signals, "wall_avoiding_random"
                if arm == "legal_random" else "greedy_odor", rng
            )
        apply_action(world, state, action, settings)
    if state.collisions:
        raise AssertionError("Mask should prevent reactive collisions")
    return {
        "seed": world.seed, "arm": arm, "food": state.eaten,
        "steps": state.steps, "visited": len(state.visited),
        "energy": state.energy, "collisions": state.collisions,
        "corrected": ctrl.corrected if ctrl else 0,
    }


def summarize_delayed(records: list[dict], genetics: list[int],
                      world_seeds: list[int], name: str) -> dict:
    by_key = {(r["genetic_seed"], r["world_seed"], r["side"], r["arm"]): r
              for r in records}
    measures = ("success", "fork_reached", "committed", "correct_branch",
                "steps", "corrected", "illegal_preferences", "visited", "starved")
    means = {
        arm: {m: statistics.mean(
            by_key[g, s, side, arm][m]
            for g in genetics for s in world_seeds for side in (0, 1)
        ) for m in measures} for arm in DELAYED_ARMS
    }
    contrasts = {}
    for a, b, label in CONTRASTS:
        matrix = [
            [statistics.mean(
                by_key[g, s, side, a]["success"]
                - by_key[g, s, side, b]["success"]
                for side in (0, 1)
            ) for s in world_seeds]
            for g in genetics
        ]
        per_run = [statistics.mean(row) for row in matrix]
        per_world = [statistics.mean(matrix[i][j]
                     for i in range(len(genetics)))
                     for j in range(len(world_seeds))]
        confidence = hierarchical_bootstrap(
            matrix, genetics[0] * 173 + world_seeds[0]
            + list(c[2] for c in CONTRASTS).index(label)
        )
        contrasts[label] = {
            "mean_delta_success": statistics.mean(per_run),
            "hierarchical_bootstrap_95": list(confidence),
            "per_genetic_run": per_run,
            "per_world": per_world,
            "positive_runs": sum(x > 0 for x in per_run),
            "negative_runs": sum(x < 0 for x in per_run),
            "tied_runs": sum(x == 0 for x in per_run),
        }
    primary = contrasts["primary_rnn_vs_ff"]
    reset = contrasts["reset_memory_effect"]
    cue = contrasts["cue_flip_effect"]
    return {
        "split": name,
        "genetic_runs": len(genetics),
        "paired_world_seeds": len(world_seeds),
        "descriptive_side_trials": len(genetics) * len(world_seeds) * 2,
        "means": means,
        "contrasts": contrasts,
        "primary_gate": bool(
            primary["mean_delta_success"] > 0
            and primary["hierarchical_bootstrap_95"][0] > 0
            and primary["positive_runs"] >= 6
        ),
        "memory_gate": bool(
            reset["mean_delta_success"] > 0 and
            reset["hierarchical_bootstrap_95"][0] > 0
        ),
        "cue_gate": bool(
            cue["mean_delta_success"] > 0 and
            cue["hierarchical_bootstrap_95"][0] > 0
        ),
    }


def summarize_reactive(records: list[dict], genetics: list[int],
                       worlds: list[int]) -> dict:
    keys = {(r["genetic_seed"], r["seed"], r["arm"]): r for r in records}
    means = {arm: {
        k: statistics.mean(keys[g, seed, arm][k]
                           for g in genetics for seed in worlds)
        for k in ("food", "steps", "visited", "energy", "collisions", "corrected")
    } for arm in REACTIVE_ARMS}
    contrast = [[keys[g, w, "rnn_reactive"]["food"] -
                 keys[g, w, "rnn_reactive_no_odor"]["food"] for w in worlds]
                for g in genetics]
    return {
        "exploratory_only": True,
        "genetic_runs": len(genetics),
        "worlds": len(worlds),
        "means": means,
        "odor_ablation_delta_food": statistics.mean(
            statistics.mean(row) for row in contrast
        ),
        "odor_ablation_hierarchical_95": list(hierarchical_bootstrap(
            contrast, genetics[0] * 211 + worlds[0]
        )),
    }


def render_maze_svg(world: TrialWorld, route: list[Point]) -> str:
    scale = 24
    shapes = []
    for y in range(world.size):
        for x in range(world.size):
            cell = (x, y)
            bg = "#475569" if cell in world.walls else "#e4f2e7"
            shapes.append(
                f'<rect x="{scale*x}" y="{scale*y}" width="23" height="23" '
                f'fill="{bg}" rx="2"/>'
            )
    for i, gate in enumerate(world.gates):
        x, y = gate
        bg = "#22a45c" if i == world.side else "#b9c6b9"
        shapes.append(f'<circle cx="{scale*x+12}" cy="{scale*y+12}" r="7" '
                      f'fill="{bg}"/>')
    points = " ".join(f"{scale*x+12},{scale*y+12}" for x, y in route)
    if points:
        shapes.append(f'<polyline points="{points}" fill="none" '
                      'stroke="#f97316" stroke-width="3"/>')
    x, y = world.start
    shapes.append(f'<circle cx="{scale*x+12}" cy="{scale*y+12}" '
                  'r="7" fill="#2563eb"/>')
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{scale*world.size}" '
        f'height="{scale*world.size}">'
        + "".join(shapes) + "</svg>\n"
    )


def run(settings: Settings, output: Path) -> dict:
    settings.validate()
    output.mkdir(parents=True, exist_ok=True)
    genetics = [settings.genetic_seed_start + i
                for i in range(settings.genetic_runs)]
    main_seeds = list(range(settings.test_start,
                            settings.test_start + settings.test_worlds))
    shifted_seeds = list(range(settings.shifted_start,
                               settings.shifted_start + settings.shifted_worlds))
    reactive_seeds = list(range(settings.reactive_start,
                                settings.reactive_start + settings.reactive_worlds))
    delayed_worlds = {
        "main": [w for seed in main_seeds for w in make_pair(seed, settings)],
        "shifted": [w for seed in shifted_seeds
                    for w in make_pair(seed, settings, shifted=True)],
    }
    # Reactive transfer has M4.1's full 100-action horizon, separate from
    # the 20-action irreversible memory-gate trial.
    reactive_settings = replace(settings, horizon=100)
    open_worlds = [make_diverse_world(seed, reactive_settings)
                   for seed in reactive_seeds]
    genotypes = {}
    history = []
    selection = []
    records = []
    reactive_records = []
    for genetic_seed in genetics:
        rnn, hist_rnn, select_rnn = evolve_one(settings, genetic_seed, "rnn")
        ff, hist_ff, select_ff = evolve_one(settings, genetic_seed, "ff")
        genotypes[genetic_seed] = {
            "rnn": list(rnn), "ff": list(ff)
        }
        history.extend(hist_rnn + hist_ff)
        selection.extend((select_rnn, select_ff))
        for label, worlds in delayed_worlds.items():
            for world in worlds:
                for arm in DELAYED_ARMS:
                    genome = ff if arm == "evolved_ff" else (
                        None if arm in ("legal_random", "scripted_memory")
                        else rnn
                    )
                    trial = simulate(world, settings, arm, genome)
                    records.append({
                        "split": label, "genetic_seed": genetic_seed,
                        **{k: v for k, v in trial.items() if k not in (
                            "path", "events",
                        )}
                    })
            print(f"M4.2 holdout gene={genetic_seed} {label} "
                  f"world_pairs={len(worlds)//2} done", flush=True)
        for world in open_worlds:
            for arm in REACTIVE_ARMS:
                genome = ff if arm == "ff_reactive" else (
                    rnn if arm in ("rnn_reactive", "rnn_reactive_no_odor")
                    else None
                )
                trial = reactive_trial(world, reactive_settings, arm, genome)
                reactive_records.append({
                    "genetic_seed": genetic_seed, **trial,
                })
    grouped = {label: [r for r in records if r["split"] == label]
               for label in delayed_worlds}
    summary = {
        "protocol": "M4.2 frozen delayed-cue fork, recurrent-state memory benchmark",
        "settings": asdict(settings),
        "selection": selection,
        "main": summarize_delayed(grouped["main"], genetics, main_seeds, "main"),
        "shifted": summarize_delayed(
            grouped["shifted"], genetics, shifted_seeds, "shifted"
        ),
        "reactive_transfer": summarize_reactive(
            reactive_records, genetics, reactive_seeds
        ),
        "delayed_main_primary": "paired food success evolved_rnn minus evolved_ff",
        "necessary_memory_ablations": "junction reset and flipped cue",
        "limitations": [
            "A one-way branch commitment gate is a toy memory benchmark, not an ecology",
            "Cue is externally supplied and absolute cardinal, not self-discovered",
            "Selection optimizes externally specified food and progress fitness",
            "No within-life weight learning or independent organism reproduction",
            "Reactive transfer uses no retraining; no odor skill may emerge",
            "Evolved RNN has 260 weights, feedforward has 268 (3.1% more)",
            "Hierarchical bootstrap respects 8 genetic runs x shared paired worlds",
            "A positive mean without reset ablation does not show functional memory",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "genotypes.json").write_text(
        json.dumps({
            "genomes": genotypes, "selections": selection,
            "input_features": INPUTS,
            "rnn_params": genome_size("rnn", settings),
            "ff_params": genome_size("ff", settings),
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output / "train_validation.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        columns = list(history[0])
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(history)
    with (output / "delayed_trial_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        columns = ("split", "genetic_seed", "world_seed", "side", "arm",
                   "shifted", "success", "fork_reached", "committed",
                   "correct_branch", "steps", "corrected",
                   "illegal_preferences", "collisions", "visited",
                   "starved", "end_energy", "max_stem_progress", "fitness")
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows({k: r[k] for k in columns} for r in records)
    with (output / "reactive_transfer_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        columns = ("genetic_seed", "seed", "arm", "food", "steps",
                   "visited", "energy", "collisions", "corrected")
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows({k: r[k] for k in columns}
                         for r in reactive_records)
    for label, shifted in (("main", False), ("shifted", True)):
        w = make_pair(
            settings.shifted_start if shifted else settings.test_start,
            settings, shifted
        )[0]
        detail = simulate(w, settings, "evolved_rnn",
                          tuple(genotypes[genetics[0]]["rnn"]), trace=True)
        (output / f"example_{label}.svg").write_text(
            render_maze_svg(w, detail["path"]), encoding="utf-8"
        )
        (output / f"example_{label}_trace.json").write_text(
            json.dumps({
                "genetic_seed": genetics[0],
                "world_seed": w.seed,
                "side": w.side,
                "fork": list(w.fork),
                "start": list(w.start),
                "target": list(w.target),
                "metrics": {k: v for k, v in detail.items()
                            if k not in ("path", "events")},
                "events": detail["events"],
            }, indent=2) + "\n", encoding="utf-8"
        )
    print("M4.2 MAIN PRIMARY",
          summary["main"]["contrasts"]["primary_rnn_vs_ff"], flush=True)
    print("M4.2 MAIN RESET",
          summary["main"]["contrasts"]["reset_memory_effect"], flush=True)
    print("M4.2 MAIN CUE",
          summary["main"]["contrasts"]["cue_flip_effect"], flush=True)
    print("M4.2 SHIFTED",
          summary["shifted"]["contrasts"]["primary_rnn_vs_ff"], flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = Settings()
    for field in fields(Settings):
        val = getattr(defaults, field.name)
        parser.add_argument("--" + field.name.replace("_", "-"),
                            type=type(val), default=val)
    parser.add_argument("--out", type=Path, default=Path("results/m4_2"))
    args = parser.parse_args()
    settings = Settings(**{
        field.name: getattr(args, field.name) for field in fields(Settings)
    })
    run(settings, args.out)


if __name__ == "__main__":
    main()
