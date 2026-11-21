"""M4.0: first embodied survival world for controlled neural evolution.

Small deterministic 2D mazes with walls, finite food and energy. A simple
recurrent controller can evolve inherited numerical weights BETWEEN episodes
but cannot edit its weights, goals, or source during any single lifetime.
All world/test seeds and scientific criteria are preregistered in the protocol.
Standard library only.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import deque
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from .plasticity import paired_bootstrap

Point = tuple[int, int]
DIRECTIONS: tuple[Point, ...] = ((0, -1), (1, 0), (0, 1), (-1, 0))
ARMS = ("evolved", "random_neural", "wall_avoiding_random",
        "greedy_odor", "evolved_no_odor")
INPUTS = 15
ACTIONS = 4


@dataclass(frozen=True)
class Settings:
    train_start: int = 41000
    train_worlds: int = 8
    validation_start: int = 42000
    validation_worlds: int = 4
    test_start: int = 43000
    test_worlds: int = 32
    shifted_start: int = 44000
    shifted_worlds: int = 16
    size: int = 13
    obstacle_density: float = 0.16
    shifted_obstacle_density: float = 0.26
    food_count: int = 12
    shifted_food_count: int = 8
    odor_radius: int = 5
    shifted_odor_radius: int = 2
    initial_energy: float = 24.0
    max_energy: float = 42.0
    food_energy: float = 16.0
    movement_cost: float = 1.0
    collision_extra_cost: float = 0.3
    horizon: int = 100
    hidden: int = 12
    population: int = 12
    generations: int = 8
    elites: int = 2
    immigrants: int = 2
    initial_weight_sd: float = 0.45
    mutation_sd: float = 0.28
    evolution_seed: int = 61042
    random_neural_seed: int = 73001

    def validate(self) -> None:
        counts = (
            self.train_worlds, self.validation_worlds, self.test_worlds,
            self.shifted_worlds, self.food_count, self.shifted_food_count,
            self.odor_radius, self.shifted_odor_radius, self.horizon,
            self.hidden, self.population, self.generations,
        )
        if min(counts) < 1 or self.size < 9 or self.size % 2 != 1:
            raise ValueError("Odd world size >=9 and positive dimensions required")
        if self.elites < 1 or self.population <= self.elites + self.immigrants:
            raise ValueError("Population must exceed fixed elite and immigrant counts")
        if self.immigrants < 1 or self.food_count >= (self.size - 2) ** 2:
            raise ValueError("Invalid immigrant count or food population")
        if self.shifted_food_count >= (self.size - 2) ** 2:
            raise ValueError("Shifted world has too much food")
        if not (0 <= self.obstacle_density < 0.75 and
                0 <= self.shifted_obstacle_density < 0.75):
            raise ValueError("Obstacle densities out of range")
        positive = (
            self.initial_energy, self.max_energy, self.food_energy,
            self.movement_cost, self.initial_weight_sd, self.mutation_sd,
        )
        if min(positive) <= 0 or self.initial_energy > self.max_energy:
            raise ValueError("Energy and weight scales must be positive")
        if self.collision_extra_cost < 0:
            raise ValueError("Collision penalty must be nonnegative")
        groups = (
            (self.train_start, self.train_worlds),
            (self.validation_start, self.validation_worlds),
            (self.test_start, self.test_worlds),
            (self.shifted_start, self.shifted_worlds),
        )
        for i, (start, count) in enumerate(groups):
            if start < 0:
                raise ValueError("Seed ranges must be nonnegative")
            a = set(range(start, start + count))
            for s, n in groups[i + 1:]:
                if a.intersection(range(s, s + n)):
                    raise ValueError("Training/validation/final world seed overlap")


@dataclass(frozen=True)
class World:
    seed: int
    size: int
    start: Point
    walls: frozenset[Point]
    initial_food: tuple[Point, ...]
    odor_radius: int
    shifted: bool

    def reachable(self) -> frozenset[Point]:
        seen = {self.start}
        q = deque([self.start])
        while q:
            x, y = q.popleft()
            for dx, dy in DIRECTIONS:
                nxt = (x + dx, y + dy)
                if (0 < nxt[0] < self.size - 1 and
                    0 < nxt[1] < self.size - 1 and
                    nxt not in self.walls and nxt not in seen):
                    seen.add(nxt)
                    q.append(nxt)
        return frozenset(seen)


def make_world(seed: int, settings: Settings, shifted: bool = False) -> World:
    rng = random.Random(seed * 1009 + 421)
    size = settings.size
    start = (size // 2, size // 2)
    density = (settings.shifted_obstacle_density if shifted
               else settings.obstacle_density)
    nfood = settings.shifted_food_count if shifted else settings.food_count
    odor = settings.shifted_odor_radius if shifted else settings.odor_radius
    protected = {start} | {
        (start[0] + dx, start[1] + dy) for dx, dy in DIRECTIONS
    }
    border = {
        (x, y) for x in range(size) for y in range(size)
        if x in (0, size - 1) or y in (0, size - 1)
    }
    grid = [(x, y) for y in range(1, size - 1)
            for x in range(1, size - 1) if (x, y) not in protected]
    min_reachable = min(40, (size - 2) ** 2 - 1)
    for _ in range(100):
        walls = border | {p for p in grid if rng.random() < density}
        candidate = World(seed, size, start, frozenset(walls), (), odor, shifted)
        reachable = sorted(candidate.reachable() - {start})
        if len(reachable) >= max(min_reachable, nfood):
            food = tuple(sorted(rng.sample(reachable, nfood)))
            return World(seed, size, start, frozenset(walls),
                         food, odor, shifted)
    raise RuntimeError("World generator failed to find reachable maze")


@dataclass
class State:
    position: Point
    energy: float
    food: set[Point]
    eaten: int
    collisions: int
    steps: int
    visited: set[Point]
    last_collision: int = 0
    last_food: int = 0


def initial_state(world: World, settings: Settings) -> State:
    return State(
        position=world.start, energy=settings.initial_energy,
        food=set(world.initial_food), eaten=0, collisions=0,
        steps=0, visited={world.start},
    )


def observation(world: World, state: State, settings: Settings) -> tuple[float, ...]:
    x, y = state.position
    blocked = []
    adjacent_food = []
    for dx, dy in DIRECTIONS:
        p = (x + dx, y + dy)
        blocked.append(float(p in world.walls))
        adjacent_food.append(float(p in state.food))

    # Odor supplies a bounded, local and explicitly programmed gradient;
    # this is NOT food-location access by the neural policy.
    odor = [0.0] * ACTIONS
    for fx, fy in state.food:
        delta_x, delta_y = fx - x, fy - y
        distance = abs(delta_x) + abs(delta_y)
        if not 0 < distance <= world.odor_radius:
            continue
        intensity = 1.0 / (1.0 + distance)
        for i, (dx, dy) in enumerate(DIRECTIONS):
            directional = max(0, dx * delta_x + dy * delta_y)
            if directional:
                odor[i] = max(odor[i],
                              intensity * directional / distance)
    values = (
        blocked + odor + adjacent_food +
        [min(1., state.energy / settings.max_energy),
         float(state.last_collision), float(state.last_food)]
    )
    if len(values) != INPUTS:
        raise AssertionError("Invalid sensor dimensions")
    return tuple(values)


def apply_action(world: World, state: State, action: int,
                 settings: Settings) -> None:
    if action not in range(ACTIONS):
        raise ValueError("Action must be cardinal direction index")
    if state.energy <= 0 or state.steps >= settings.horizon:
        raise ValueError("Cannot act after starvation or horizon")
    dx, dy = DIRECTIONS[action]
    new_pos = (state.position[0] + dx, state.position[1] + dy)
    hit = new_pos in world.walls
    state.steps += 1
    state.last_collision = int(hit)
    state.last_food = 0
    state.energy -= settings.movement_cost
    if hit:
        state.collisions += 1
        state.energy -= settings.collision_extra_cost
    else:
        state.position = new_pos
        state.visited.add(new_pos)
        if new_pos in state.food:
            state.food.remove(new_pos)
            state.eaten += 1
            state.last_food = 1
            state.energy = min(settings.max_energy,
                               state.energy + settings.food_energy)
    state.energy = max(0., state.energy)


def genome_size(hidden: int) -> int:
    return hidden * (INPUTS + hidden + 1) + ACTIONS * (hidden + 1)


def fresh_genome(rng: random.Random, settings: Settings) -> tuple[float, ...]:
    return tuple(rng.gauss(0., settings.initial_weight_sd)
                 for _ in range(genome_size(settings.hidden)))


def mutate(genome: tuple[float, ...], rng: random.Random,
           settings: Settings) -> tuple[float, ...]:
    return tuple(max(-5., min(5., w + rng.gauss(0., settings.mutation_sd)))
                 for w in genome)


class RecurrentController:
    def __init__(self, genome: tuple[float, ...], hidden: int, no_odor=False):
        if len(genome) != genome_size(hidden):
            raise ValueError("Genome length does not match architecture")
        self.genome = genome
        self.hidden = [0.] * hidden
        self.width = hidden
        self.no_odor = no_odor
        self.offset = hidden * (INPUTS + hidden + 1)

    def choose(self, obs: tuple[float, ...]) -> int:
        if len(obs) != INPUTS:
            raise ValueError("Sensor shape incorrect")
        x = (obs[:4] + (0.,) * 4 + obs[8:]) if self.no_odor else obs
        new_hidden = []
        stride = INPUTS + self.width + 1
        for i in range(self.width):
            offset = i * stride
            weighted = sum(
                w * v for w, v in zip(
                    self.genome[offset:offset + INPUTS], x
                )
            )
            recurrent = sum(
                w * v for w, v in zip(
                    self.genome[offset + INPUTS:offset + INPUTS + self.width],
                    self.hidden,
                )
            )
            bias = self.genome[offset + INPUTS + self.width]
            new_hidden.append(math.tanh(weighted + recurrent + bias))
        self.hidden = new_hidden
        output = []
        for action in range(ACTIONS):
            start = self.offset + action * (self.width + 1)
            score = sum(
                w * h for w, h in zip(
                    self.genome[start:start + self.width], self.hidden
                )
            ) + self.genome[start + self.width]
            output.append(score)
        return max(range(ACTIONS), key=lambda a: (output[a], -a))


def decide_baseline(obs: tuple[float, ...], kind: str,
                    rng: random.Random) -> int:
    open_actions = [i for i in range(ACTIONS) if obs[i] == 0.]
    if not open_actions:
        return rng.randrange(ACTIONS)
    if kind == "wall_avoiding_random":
        return rng.choice(open_actions)
    if kind == "greedy_odor":
        best = max(obs[4 + i] for i in open_actions)
        choices = [i for i in open_actions if
                   abs(obs[4 + i] - best) < 1e-12]
        return rng.choice(choices)
    raise ValueError("Unknown fixed control")


def simulate(world: World, settings: Settings, arm: str,
             genome: tuple[float, ...] | None = None,
             trace: bool = False) -> dict:
    if arm not in ARMS:
        raise ValueError(arm)
    if arm in ("evolved", "random_neural", "evolved_no_odor") and genome is None:
        raise ValueError("Neural arm requires inherited weight vector")
    policy = (
        RecurrentController(genome, settings.hidden, arm == "evolved_no_odor")
        if genome is not None else None
    )
    rng = random.Random(world.seed * 1717 + ARMS.index(arm) * 97 + 7)
    state = initial_state(world, settings)
    trajectory = [world.start] if trace else []
    events = []
    odor_positive_steps = 0
    while state.energy > 0 and state.steps < settings.horizon:
        obs = observation(world, state, settings)
        odor_positive_steps += int(any(v > 0 for v in obs[4:8]))
        action = (
            policy.choose(obs)
            if arm in ("evolved", "random_neural", "evolved_no_odor")
            else decide_baseline(obs, arm, rng)
        )
        old_energy = state.energy
        apply_action(world, state, action, settings)
        if trace:
            trajectory.append(state.position)
            events.append({
                "step": state.steps, "position": list(state.position),
                "energy": round(state.energy, 5),
                "eaten": state.eaten,
                "collision": state.last_collision,
                "odor_detected": int(any(v > 0 for v in obs[4:8])),
            })
            if state.last_food:
                events[-1]["food_event"] = True
    return {
        "seed": world.seed, "shifted": world.shifted, "arm": arm,
        "food_collected": state.eaten,
        "steps_alive": state.steps,
        "wall_collisions": state.collisions,
        "unique_visited": len(state.visited),
        "end_energy": state.energy,
        "odor_detected_steps": odor_positive_steps,
        "terminated_by_starvation": int(state.energy <= 0),
        "fitness": (2 * state.eaten + .03 * state.steps -
                    .05 * state.collisions),
        "path": trajectory, "events": events,
    }


def evolve(settings: Settings) -> tuple[tuple[float, ...], list[dict], dict]:
    seed_rng = random.Random(settings.evolution_seed)
    train = [make_world(i, settings) for i in
             range(settings.train_start, settings.train_start +
                   settings.train_worlds)]
    validation = [make_world(i, settings) for i in
                  range(settings.validation_start, settings.validation_start +
                        settings.validation_worlds)]
    population = [fresh_genome(seed_rng, settings)
                  for _ in range(settings.population)]
    record = []
    finalists = []
    evaluated = 0
    for generation in range(settings.generations):
        ranked = []
        for g in population:
            results = [simulate(world, settings, "evolved", g)
                       for world in train]
            evaluated += len(results)
            ranked.append((statistics.mean(r["fitness"] for r in results),
                           g, statistics.mean(r["food_collected"] for r in results)))
        ranked.sort(key=lambda x: x[0], reverse=True)
        leader = ranked[0]
        validation_runs = [simulate(world, settings, "evolved", leader[1])
                           for world in validation]
        val_fitness = statistics.mean(r["fitness"] for r in validation_runs)
        finalists.append((val_fitness, generation, leader[1]))
        record.append({
            "generation": generation,
            "best_training_fitness": leader[0],
            "mean_training_fitness": statistics.mean(x[0] for x in ranked),
            "best_training_food": leader[2],
            "validation_fitness": val_fitness,
            "validation_food": statistics.mean(r["food_collected"]
                                               for r in validation_runs),
            "candidate_world_evaluations_so_far": evaluated,
        })
        print(f"M4.0 evolution gen={generation} "
              f"best_food={leader[2]:.3f} "
              f"train_fitness={leader[0]:.3f} "
              f"validation_fitness={val_fitness:.3f}", flush=True)
        elite = [g for _, g, _ in ranked[:settings.elites]]
        population = list(elite)
        while len(population) < settings.population - settings.immigrants:
            parent = seed_rng.choice(elite)
            population.append(mutate(parent, seed_rng, settings))
        while len(population) < settings.population:
            population.append(fresh_genome(seed_rng, settings))
    chosen_val, chosen_generation, chosen_genome = max(
        finalists, key=lambda item: (item[0], -item[1])
    )
    if evaluated != settings.generations * settings.population * settings.train_worlds:
        raise AssertionError("Evolution evaluation budget violated")
    selection = {
        "champion_generation": chosen_generation,
        "champion_validation_fitness": chosen_val,
        "genome_parameters": len(chosen_genome),
        "train_candidate_world_evaluations": evaluated,
        "validation_leader_world_evaluations":
            settings.validation_worlds * settings.generations,
    }
    return chosen_genome, record, selection


def render_world_svg(world: World, route: list[Point],
                     settings: Settings) -> str:
    """Artifact only: a deterministic 2D visualization of ONE recorded route."""
    cell = 23
    width = world.size * cell
    bg = []
    for y in range(world.size):
        for x in range(world.size):
            point = (x, y)
            fill = "#475569" if point in world.walls else "#e9f2e9"
            bg.append(
                f'<rect x="{x*cell}" y="{y*cell}" width="{cell-1}" '
                f'height="{cell-1}" rx="2" fill="{fill}"/>'
            )
    for x, y in world.initial_food:
        bg.append(f'<circle cx="{x*cell+cell/2}" cy="{y*cell+cell/2}" '
                  f'r="5" fill="#22a045"/>')
    points = " ".join(
        f"{x*cell+cell/2:.1f},{y*cell+cell/2:.1f}"
        for x, y in route
    )
    if points:
        bg.append(f'<polyline points="{points}" stroke="#f97316" '
                  'stroke-width="2.2" stroke-opacity=".8" '
                  'stroke-linecap="round" stroke-linejoin="round" fill="none"/>')
    x0, y0 = world.start
    bg.append(f'<circle cx="{x0*cell+cell/2}" cy="{y0*cell+cell/2}" '
              'r="7" fill="#2563eb" stroke="#ffffff" stroke-width="2"/>')
    if route:
        xe, ye = route[-1]
        bg.append(f'<circle cx="{xe*cell+cell/2}" cy="{ye*cell+cell/2}" '
                  'r="5.3" fill="#ea580c" stroke="#fff" stroke-width="1.5"/>')
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{width}" viewBox="0 0 {width} {width}">'
        + "".join(bg) + "</svg>\n"
    )


def evaluate_split(settings: Settings, genome: tuple[float, ...],
                   initial: tuple[float, ...], shifted: bool) -> tuple[list[dict],dict]:
    first, count = ((settings.shifted_start, settings.shifted_worlds)
                    if shifted else (settings.test_start, settings.test_worlds))
    records = []
    for seed in range(first, first + count):
        world = make_world(seed, settings, shifted)
        if (any(point in world.walls for point in world.initial_food) or
                any(point not in world.reachable()
                    for point in world.initial_food)):
            raise AssertionError("Food resource not reachable")
        genomes = {
            "evolved": genome, "random_neural": initial,
            "evolved_no_odor": genome,
        }
        for arm in ARMS:
            result = simulate(world, settings, arm, genomes.get(arm))
            records.append({k: v for k, v in result.items()
                            if k not in ("path", "events")})
        print(f"M4.0 holdout {'shifted' if shifted else 'main'} "
              f"seed={seed} evolved={records[-5]['food_collected']} "
              f"random={records[-3]['food_collected']} "
              f"greedy={records[-2]['food_collected']}", flush=True)
    by_arm = {
        arm: {r["seed"]: r for r in records if r["arm"] == arm}
        for arm in ARMS
    }
    mean_metrics = {
        arm: {key: statistics.mean(row[key] for row in group.values())
              for key in ("food_collected", "steps_alive", "wall_collisions",
                          "unique_visited", "end_energy",
                          "odor_detected_steps", "terminated_by_starvation")}
        for arm, group in by_arm.items()
    }
    comparisons = {}
    for competitor in ARMS[1:]:
        diff = [by_arm["evolved"][seed]["food_collected"] -
                by_arm[competitor][seed]["food_collected"]
                for seed in sorted(by_arm["evolved"])]
        comparisons[f"evolved_minus_{competitor}"] = {
            "mean_food_difference": statistics.mean(diff),
            "paired_bootstrap_95": list(paired_bootstrap(diff, first + 117)),
            "positive_worlds": sum(x > 0 for x in diff),
            "negative_worlds": sum(x < 0 for x in diff),
            "tied_worlds": sum(x == 0 for x in diff),
        }
    result = {
        "independent_worlds": count,
        "world_type": "shifted" if shifted else "main",
        "food_per_world": (settings.shifted_food_count if shifted
                           else settings.food_count),
        "mean": mean_metrics,
        "comparisons": comparisons,
        "evolved_food_positive_fraction": sum(
            r["food_collected"] > 0
            for r in by_arm["evolved"].values()
        ) / count,
    }
    return records, result


def run(settings: Settings, out: Path) -> dict:
    settings.validate()
    out.mkdir(parents=True, exist_ok=True)
    chosen, history, selection = evolve(settings)
    random_reference = fresh_genome(
        random.Random(settings.random_neural_seed), settings
    )
    main_records, main_summary = evaluate_split(
        settings, chosen, random_reference, False
    )
    shifted_records, shifted_summary = evaluate_split(
        settings, chosen, random_reference, True
    )
    report = {
        "protocol": "M4.0 preregistered first embodied food-seeking world",
        "settings": asdict(settings),
        "selection": selection,
        "main": main_summary, "shifted": shifted_summary,
        "primary": "main paired food collected evolved minus wall_avoiding_random",
        "necessary_secondary": "main evolved food vs random_neural",
        "minimum_nonzero_food_fraction_for_positive_claim": 0.70,
        "historical_comparison": "M1-M3.5 untouched; M4.0 distinct embodiment line",
        "limits": [
            "Directional food odor is designed and visible to the greedy heuristic",
            "Evolution is between lifetimes only; no online synaptic learning",
            "World contains no predators, resource reproduction or within-world reproduction",
            "Fitness is assigned by researcher; agent has no independent objectives",
            "Fixed-topology numerical mutation, not autonomous program rewriting",
            "Selection uses only training and validation seeds; holdouts are independent",
            "Wall avoidance baseline uses local geometry directly",
            "Single evolved champion, small training set; no strong claims of generality",
        ],
    }
    (out / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out / "champion.json").write_text(
        json.dumps({"architecture": {"inputs": INPUTS, "hidden": settings.hidden,
                                      "actions": ACTIONS},
                    "parameters": list(chosen),
                    "validation_selection": selection}, indent=2) + "\n",
        encoding="utf-8",
    )
    with (out / "history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    with (out / "test_seed_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        columns = ("seed", "shifted", "arm", "food_collected",
                   "steps_alive", "wall_collisions", "unique_visited",
                   "end_energy", "odor_detected_steps",
                   "terminated_by_starvation", "fitness")
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows({k: row[k] for k in columns}
                         for row in main_records + shifted_records)
    for seed, shifted, name in (
        (settings.test_start, False, "main"),
        (settings.shifted_start, True, "shifted"),
    ):
        world = make_world(seed, settings, shifted)
        detail = simulate(world, settings, "evolved", chosen, trace=True)
        (out / f"example_{name}.svg").write_text(
            render_world_svg(world, detail["path"], settings), encoding="utf-8"
        )
        (out / f"example_{name}_trace.json").write_text(
            json.dumps({
                "seed": seed,
                "initial_food": [list(x) for x in world.initial_food],
                "walls": [list(x) for x in sorted(world.walls)],
                "metrics": {k: v for k, v in detail.items()
                            if k not in ("path", "events")},
                "events": detail["events"],
            }, indent=2) + "\n", encoding="utf-8"
        )
    print("M4.0 MAIN PRIMARY",
          main_summary["comparisons"]["evolved_minus_wall_avoiding_random"],
          flush=True)
    print("M4.0 MAIN GREEDY",
          main_summary["comparisons"]["evolved_minus_greedy_odor"],
          flush=True)
    print("M4.0 SHIFTED PRIMARY",
          shifted_summary["comparisons"]["evolved_minus_wall_avoiding_random"],
          flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = Settings()
    for fld in fields(Settings):
        default = getattr(defaults, fld.name)
        parser.add_argument("--" + fld.name.replace("_", "-"),
                            type=type(default), default=default)
    parser.add_argument("--out", type=Path, default=Path("results/m4_0"))
    args = parser.parse_args()
    settings = Settings(**{fld.name: getattr(args, fld.name)
                           for fld in fields(Settings)})
    run(settings, args.out)


if __name__ == "__main__":
    main()
