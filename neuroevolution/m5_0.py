"""M5.0: persistent energy-limited 2D artificial ecology with real in-world births.

No external fitness ranking, elite cloning or episodic champion selection.
Neural controllers decide among locomotor actions and an explicit reproduction
action. Parents' numerical weights can be inherited and mutated; a random-
offspring control severs phenotypic inheritance without changing birth physics.
Standard library only; never grants simulated organisms any tool access.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from dataclasses import asdict, dataclass, fields, field
from pathlib import Path

from .m4_0 import DIRECTIONS, World as GridWorld, Settings as WorldSettings, make_world

Point = tuple[int, int]
INPUTS = 15
ACTIONS = 5
REPRODUCE = 4
ARMS = ("heritable_mutation", "random_offspring", "heritable_no_mutation")
CHECKPOINTS = (0, 50, 100, 200, 300, 400)


@dataclass(frozen=True)
class Settings:
    seed_start: int = 71000
    seeds: int = 12
    shifted_start: int = 72000
    shifted_seeds: int = 8
    ticks: int = 400
    size: int = 15
    obstacle_density: float = 0.12
    shifted_obstacle_density: float = 0.20
    initial_food: int = 20
    food_cap: int = 32
    regeneration_attempts: int = 2
    regeneration_probability: float = 0.55
    shifted_regeneration_probability: float = 0.40
    odor_radius: int = 5
    shifted_odor_radius: int = 2
    initial_population: int = 20
    max_population: int = 64
    initial_energy: float = 36.
    max_energy: float = 54.
    food_energy: float = 18.
    movement_cost: float = 1.
    newborn_energy: float = 12.
    reproduction_energy_cost: float = 20.
    min_reproduction_energy: float = 34.
    min_reproduction_age: int = 4
    max_age: int = 100
    hidden: int = 12
    initial_weight_sd: float = 0.45
    mutation_sd: float = 0.12
    checkpoint_every: int = 50
    no_food_controls: int = 4

    def validate(self) -> None:
        if self.size < 11 or self.size % 2 == 0:
            raise ValueError("World size must be odd and >=11")
        positive = (self.seeds, self.shifted_seeds, self.ticks,
                    self.hidden, self.initial_population, self.max_population,
                    self.food_cap, self.regeneration_attempts, self.max_age,
                    self.checkpoint_every, self.min_reproduction_age)
        if min(positive) < 1:
            raise ValueError("Counts must be positive")
        if self.max_population <= self.initial_population:
            raise ValueError("Population capacity must allow reproduction")
        if not (self.initial_food >= 1 and
                self.initial_food < self.food_cap and
                self.food_cap <= (self.size - 2) ** 2):
            raise ValueError("Food occupancy/cap invalid")
        if not 0 <= self.obstacle_density < .6 or not 0 <= self.shifted_obstacle_density < .6:
            raise ValueError("Obstacle rates invalid")
        if not 0 <= self.regeneration_probability <= 1 or not 0 <= self.shifted_regeneration_probability <= 1:
            raise ValueError("Regeneration probability invalid")
        if self.min_reproduction_energy < self.reproduction_energy_cost + self.movement_cost:
            raise ValueError("Minimum reproductive energy cannot be less than its cost")
        if self.max_energy < self.initial_energy or self.initial_energy <= 0:
            raise ValueError("Energy initial/max inconsistency")
        if self.reproduction_energy_cost < self.newborn_energy or self.newborn_energy <= 0:
            raise ValueError("Birth must dissipate rather than create energy")
        if self.movement_cost <= 0 or self.food_energy <= 0:
            raise ValueError("Positive movement/food energetics required")
        if self.mutation_sd < 0 or self.initial_weight_sd <= 0:
            raise ValueError("Weight scales invalid")
        if self.no_food_controls > self.seeds:
            raise ValueError("No-food control count exceeds main seeds")
        if set(range(self.seed_start, self.seed_start + self.seeds)).intersection(
            range(self.shifted_start, self.shifted_start + self.shifted_seeds)
        ):
            raise ValueError("Main/shifted world seeds must be disjoint")


def genome_size(hidden: int) -> int:
    return hidden * (INPUTS + hidden + 1) + ACTIONS * (hidden + 1)


def initial_genome(rng: random.Random, settings: Settings) -> tuple[float, ...]:
    return tuple(rng.gauss(0., settings.initial_weight_sd)
                 for _ in range(genome_size(settings.hidden)))


def make_child_genome(parent: tuple[float, ...], settings: Settings,
                      arm: str, rng: random.Random) -> tuple[float, ...]:
    if arm == "heritable_mutation":
        return tuple(max(-5., min(5., w + rng.gauss(0., settings.mutation_sd)))
                     for w in parent)
    if arm == "heritable_no_mutation":
        return parent
    if arm == "random_offspring":
        return initial_genome(rng, settings)
    raise ValueError(f"Unknown arm {arm!r}")


def genome_digest(genome: tuple[float, ...]) -> str:
    data = ",".join(format(x, ".10g") for x in genome)
    return hashlib.sha256(data.encode("ascii")).hexdigest()[:16]


def genome_cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    aa = math.fsum(x * x for x in a)
    bb = math.fsum(x * x for x in b)
    if aa == 0 or bb == 0:
        return 0.
    return math.fsum(x * y for x, y in zip(a, b)) / math.sqrt(aa * bb)


@dataclass
class Agent:
    id: int
    founder: int
    parent: int | None
    generation: int
    position: Point
    energy: float
    genome: tuple[float, ...]
    hidden: list[float]
    age: int = 0
    last_failed: int = 0
    last_food: int = 0
    eaten: int = 0
    births: int = 0


@dataclass
class Ecology:
    seed: int
    shifted: bool
    arm: str
    world: GridWorld
    settings: Settings
    agents: dict[int, Agent]
    food: set[Point]
    accessible: tuple[Point, ...]
    initial_genome_digests: tuple[str, ...]
    next_id: int
    ticks_elapsed: int = 0
    births: int = 0
    deaths: int = 0
    deaths_starvation: int = 0
    deaths_age: int = 0
    food_eaten: int = 0
    food_spawned: int = 0
    move_actions: int = 0
    birth_actions: int = 0
    corrected_actions: int = 0
    skipped_moves: int = 0
    reproduction_ineligible_wishes: int = 0
    max_generation: int = 0
    extinct_at: int | None = None
    cumulative_food_energy: float = 0.
    cumulative_cost_basal: float = 0.
    cumulative_birth_dissipation: float = 0.
    cumulative_death_discard: float = 0.
    cosine_births: list[float] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)


def _world(seed: int, settings: Settings, shifted: bool) -> GridWorld:
    base = WorldSettings(
        size=settings.size, food_count=settings.initial_food,
        shifted_food_count=settings.initial_food,
        obstacle_density=settings.obstacle_density,
        shifted_obstacle_density=settings.shifted_obstacle_density,
        odor_radius=settings.odor_radius,
        shifted_odor_radius=settings.shifted_odor_radius,
    )
    return make_world(seed, base, shifted)


def create_ecology(seed: int, settings: Settings, arm: str,
                   *, shifted: bool = False, no_food: bool = False) -> Ecology:
    settings.validate()
    if arm not in ARMS:
        raise ValueError(arm)
    world = _world(seed, settings, shifted)
    reachable = sorted(world.reachable())
    occupied_free = [p for p in reachable if p not in world.initial_food]
    if len(occupied_free) < settings.initial_population:
        raise ValueError("Not enough reachable cells for founders")
    position_rng = random.Random(seed * 4243 + 193)
    positions = position_rng.sample(occupied_free, settings.initial_population)
    genome_rng = random.Random(seed * 1559 + 997)
    genomes = [initial_genome(genome_rng, settings)
               for _ in range(settings.initial_population)]
    agents = {}
    for i, (position, genome) in enumerate(zip(positions, genomes)):
        agents[i] = Agent(
            id=i, founder=i, parent=None, generation=0,
            position=position, energy=settings.initial_energy,
            genome=genome, hidden=[0.] * settings.hidden,
        )
    return Ecology(
        seed=seed, shifted=shifted, arm=arm, world=world, settings=settings,
        agents=agents, food=set() if no_food else set(world.initial_food),
        accessible=tuple(reachable),
        initial_genome_digests=tuple(genome_digest(x) for x in genomes),
        next_id=settings.initial_population,
        food_spawned=0 if no_food else len(world.initial_food),
    )


def blocked_and_free(eco: Ecology, agent: Agent) -> tuple[list[bool], list[Point]]:
    occupied = {a.position for a in eco.agents.values() if a.id != agent.id}
    x, y = agent.position
    neighbours = [(x + dx, y + dy) for dx, dy in DIRECTIONS]
    blocked = [p in eco.world.walls or p in occupied for p in neighbours]
    return blocked, [p for p, b in zip(neighbours, blocked) if not b]


def observe(eco: Ecology, agent: Agent, blocked: list[bool]) -> tuple[float, ...]:
    """15 local channels; no ID, position, map, age or parentage is exposed."""
    x, y = agent.position
    adjacent = []
    for dx, dy in DIRECTIONS:
        adjacent.append(float((x + dx, y + dy) in eco.food))
    odor = [0.] * 4
    radius = (eco.settings.shifted_odor_radius if eco.shifted
              else eco.settings.odor_radius)
    for fx, fy in eco.food:
        vx, vy = fx - x, fy - y
        distance = abs(vx) + abs(vy)
        if not 0 < distance <= radius:
            continue
        intensity = 1. / (1. + distance)
        for a, (dx, dy) in enumerate(DIRECTIONS):
            forward = max(0, dx * vx + dy * vy)
            if forward:
                odor[a] = max(odor[a], intensity * forward / distance)
    features = (
        [float(b) for b in blocked] + odor + adjacent +
        [min(1., agent.energy / eco.settings.max_energy),
         float(agent.last_failed), float(agent.last_food)]
    )
    if len(features) != INPUTS or any(not 0 <= x <= 1 for x in features):
        raise AssertionError("Observation features must remain bounded")
    return tuple(features)


def network_logits(agent: Agent, inputs: tuple[float, ...],
                   settings: Settings) -> list[float]:
    n = settings.hidden
    if len(agent.genome) != genome_size(n) or len(agent.hidden) != n:
        raise ValueError("Genome/hidden shape mismatch")
    stride = INPUTS + n + 1
    new_hidden = []
    for i in range(n):
        off = i * stride
        s = sum(w*x for w,x in zip(agent.genome[off:off+INPUTS], inputs))
        s += sum(w*h for w,h in zip(
            agent.genome[off+INPUTS:off+INPUTS+n], agent.hidden
        ))
        s += agent.genome[off+INPUTS+n]
        new_hidden.append(math.tanh(s))
    agent.hidden = new_hidden
    head = n * stride
    return [
        sum(w*h for w,h in zip(
            agent.genome[head + j*(n+1):head + j*(n+1)+n],
            new_hidden
        )) + agent.genome[head + j*(n+1)+n]
        for j in range(ACTIONS)
    ]


def _local_rng(seed: int, tick: int, agent_id: int, tag: int) -> random.Random:
    return random.Random(
        seed * 1_000_003 + tick * 1_009_001
        + agent_id * 110_503 + tag * 47_753
    )


def can_reproduce(eco: Ecology, agent: Agent, free: list[Point]) -> bool:
    s = eco.settings
    return bool(
        len(eco.agents) < s.max_population and
        agent.age >= s.min_reproduction_age and
        agent.energy >= s.min_reproduction_energy and
        free
    )


def _charge(eco: Ecology, agent: Agent) -> None:
    loss = min(agent.energy, eco.settings.movement_cost)
    agent.energy -= loss
    eco.cumulative_cost_basal += loss
    agent.age += 1


def _remove_dead(eco: Ecology, agent: Agent) -> None:
    if agent.energy > 0 and agent.age < eco.settings.max_age:
        return
    if agent.id not in eco.agents:
        raise AssertionError("Double death")
    if agent.energy <= 0:
        eco.deaths_starvation += 1
    else:
        eco.deaths_age += 1
        eco.cumulative_death_discard += agent.energy
    eco.deaths += 1
    del eco.agents[agent.id]


def act(eco: Ecology, agent_id: int, tick: int) -> None:
    """Exactly one energy-costed action for an agent alive at start of tick."""
    agent = eco.agents.get(agent_id)
    if agent is None:
        return
    blocked, free = blocked_and_free(eco, agent)
    inputs = observe(eco, agent, blocked)
    logits = network_logits(agent, inputs, eco.settings)
    preferred = max(range(ACTIONS), key=lambda a: (logits[a], -a))
    reproduction_legal = can_reproduce(eco, agent, free)
    if preferred == REPRODUCE and not reproduction_legal:
        eco.reproduction_ineligible_wishes += 1
    legal = [i for i, b in enumerate(blocked) if not b]
    if reproduction_legal:
        legal.append(REPRODUCE)
    chosen = (max(legal, key=lambda a: (logits[a], -a))
              if legal else None)
    eco.corrected_actions += int(chosen != preferred)
    _charge(eco, agent)
    agent.last_failed = 0
    agent.last_food = 0
    if chosen is None:
        eco.skipped_moves += 1
    elif chosen == REPRODUCE:
        # Eligibility was checked on energy BEFORE the basal action cost;
        # minimum energy >= reproductive transfer + basal cost.
        if len(eco.agents) >= eco.settings.max_population:
            raise AssertionError("Birth exceeded population cap")
        if agent.energy < eco.settings.reproduction_energy_cost:
            raise AssertionError("Birth would require negative parent energy")
        # Spatial choice may differ after earlier agents move but this
        # particular agent's current free list has not changed during action.
        child_pos = _local_rng(eco.seed, tick, agent.id, 1).choice(free)
        if child_pos in {a.position for a in eco.agents.values()}:
            raise AssertionError("Birth location must be unoccupied")
        new_id = eco.next_id
        eco.next_id += 1
        offspring_rng = _local_rng(eco.seed, tick, agent.id, 2)
        child_genome = make_child_genome(
            agent.genome, eco.settings, eco.arm, offspring_rng
        )
        eco.cosine_births.append(genome_cosine(agent.genome, child_genome))
        eco.cumulative_birth_dissipation += (
            eco.settings.reproduction_energy_cost - eco.settings.newborn_energy
        )
        agent.energy -= eco.settings.reproduction_energy_cost
        child = Agent(
            id=new_id, founder=agent.founder, parent=agent.id,
            generation=agent.generation + 1, position=child_pos,
            energy=eco.settings.newborn_energy, genome=child_genome,
            hidden=[0.] * eco.settings.hidden,
        )
        eco.agents[child.id] = child
        agent.births += 1
        eco.birth_actions += 1
        eco.births += 1
        eco.max_generation = max(eco.max_generation, child.generation)
        eco.events.append({
            "tick": tick, "type": "birth", "parent": agent.id,
            "child": child.id, "founder": child.founder,
            "generation": child.generation, "location": child.position,
            "genome": genome_digest(child.genome),
        })
    else:
        eco.move_actions += 1
        dx, dy = DIRECTIONS[chosen]
        target = (agent.position[0] + dx, agent.position[1] + dy)
        if target in eco.world.walls or target in {
            a.position for a in eco.agents.values() if a.id != agent.id
        }:
            raise AssertionError("Legal movement attempted an occupied wall/cell")
        agent.position = target
        if target in eco.food:
            eco.food.remove(target)
            eco.food_eaten += 1
            agent.eaten += 1
            agent.last_food = 1
            added = min(eco.settings.food_energy,
                        eco.settings.max_energy - agent.energy)
            agent.energy += added
            eco.cumulative_food_energy += added
            eco.events.append({
                "tick": tick, "type": "food", "agent": agent.id,
                "founder": agent.founder, "location": target,
                "energy_gain": added,
            })
    _remove_dead(eco, agent)


def _food_schedule(seed: int, settings: Settings, eco: Ecology,
                   no_food: bool) -> tuple[list[tuple[bool, Point]], ...]:
    """Generate all candidate food events PRIOR to any organism actions."""
    rng = random.Random(seed * 9_191 + 499)
    probability = (settings.shifted_regeneration_probability
                   if eco.shifted else settings.regeneration_probability)
    return tuple([
        (rng.random() < probability and not no_food,
         rng.choice(eco.accessible))
        for _ in range(settings.regeneration_attempts)
    ] for _ in range(settings.ticks))


def _food_regeneration(eco: Ecology, attempts: list[tuple[bool, Point]]) -> None:
    occupied = {a.position for a in eco.agents.values()}
    for want, cell in attempts:
        if (want and len(eco.food) < eco.settings.food_cap
                and cell not in eco.food and cell not in occupied
                and cell not in eco.world.walls):
            eco.food.add(cell)
            eco.food_spawned += 1


def energy_balance(eco: Ecology) -> float:
    initial = eco.settings.initial_population * eco.settings.initial_energy
    final = sum(a.energy for a in eco.agents.values())
    # Food inputs are capped actually acquired calories, not nominal 18 units.
    return (initial + eco.cumulative_food_energy -
            eco.cumulative_cost_basal - eco.cumulative_birth_dissipation -
            eco.cumulative_death_discard - final)


def snapshot(eco: Ecology, tick: int) -> dict:
    agents = list(eco.agents.values())
    founders_alive = len({a.founder for a in agents})
    return {
        "tick": tick, "population": len(agents),
        "food_stock": len(eco.food), "births": eco.births,
        "deaths": eco.deaths, "food_eaten": eco.food_eaten,
        "food_spawned": eco.food_spawned,
        "mean_generation": (statistics.mean(a.generation for a in agents)
                            if agents else 0.),
        "max_generation": eco.max_generation,
        "lineages_alive": founders_alive,
        "total_living_energy": sum(a.energy for a in agents),
        "energy_ledger_residual": energy_balance(eco),
    }


def simulate(seed: int, settings: Settings, arm: str,
             *, shifted: bool = False, no_food: bool = False,
             detailed: bool = False) -> tuple[dict, list[dict], dict | None]:
    eco = create_ecology(seed, settings, arm, shifted=shifted, no_food=no_food)
    food_schedule = _food_schedule(seed, settings, eco, no_food)
    checkpoints = [snapshot(eco, 0)]
    initial_genomes = eco.initial_genome_digests
    # Genome snapshots are hashes, not source or executable code.
    for tick in range(1, settings.ticks + 1):
        ids = list(eco.agents)
        _local_rng(seed, tick, 0, 0).shuffle(ids)
        for agent_id in ids:
            act(eco, agent_id, tick)
        _food_regeneration(eco, food_schedule[tick - 1])
        eco.ticks_elapsed = tick
        if not eco.agents and eco.extinct_at is None:
            eco.extinct_at = tick
        if tick % settings.checkpoint_every == 0 or tick == settings.ticks:
            checkpoints.append(snapshot(eco, tick))
            if abs(energy_balance(eco)) > 1e-6:
                raise AssertionError("Energy created or destroyed outside ledger")
    founders = {}
    for event in eco.events:
        if event["type"] == "birth":
            root = event["founder"]
            founders[root] = founders.get(root, 0) + 1
    survivor_founders = len({a.founder for a in eco.agents.values()})
    summary = {
        "seed": seed,
        "split": "no_food" if no_food else ("shifted" if shifted else "main"),
        "arm": arm,
        "births": eco.births,
        "deaths": eco.deaths,
        "deaths_starvation": eco.deaths_starvation,
        "deaths_age": eco.deaths_age,
        "alive_final": len(eco.agents),
        "extinct_at": eco.extinct_at,
        "food_eaten": eco.food_eaten,
        "food_spawned": eco.food_spawned,
        "food_remaining": len(eco.food),
        "mean_energy_final": (statistics.mean(a.energy for a in eco.agents.values())
                              if eco.agents else 0.),
        "max_generation": eco.max_generation,
        "has_generation_3": int(eco.max_generation >= 3),
        "surviving_founder_lineages": survivor_founders,
        "top_lineage_birth_share": (
            max(founders.values()) / eco.births if eco.births else 0.
        ),
        "mean_parent_child_cosine": (
            statistics.mean(eco.cosine_births) if eco.cosine_births else None
        ),
        "move_actions": eco.move_actions,
        "birth_actions": eco.birth_actions,
        "motor_mask_corrections": eco.corrected_actions,
        "ineligible_reproduction_preferences":
            eco.reproduction_ineligible_wishes,
        "total_food_energy_gained": eco.cumulative_food_energy,
        "energy_basal_cost": eco.cumulative_cost_basal,
        "energy_birth_dissipation": eco.cumulative_birth_dissipation,
        "energy_death_discard": eco.cumulative_death_discard,
        "final_energy_balance_residual": energy_balance(eco),
    }
    trace = None
    if detailed:
        trace = {
            "world_seed": seed, "split": summary["split"],
            "arm": arm, "walls": [list(p) for p in sorted(eco.world.walls)],
            "initial_food": ([] if no_food
                             else [list(p) for p in eco.world.initial_food]),
            "initial_founder_digests": list(initial_genomes),
            "event_log": eco.events,
            "final_agents": [{
                "id": a.id, "parent": a.parent, "founder": a.founder,
                "generation": a.generation, "energy": a.energy,
                "age": a.age, "pos": list(a.position),
                "genome_hash": genome_digest(a.genome),
            } for a in sorted(eco.agents.values(), key=lambda x: x.id)],
            "checkpoints": checkpoints,
        }
    return summary, checkpoints, trace


def paired_bootstrap(values: list[float], seed: int,
                     draws: int = 4000) -> tuple[float, float]:
    if not values:
        raise ValueError("No independent seed pairs")
    rng = random.Random(seed)
    n = len(values)
    sample_means = sorted(
        sum(values[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(draws)
    )
    return sample_means[int(draws * .025)], sample_means[int(draws * .975)]


def comparison(rows: list[dict], split: str, a: str, b: str,
               seed: int) -> dict:
    primary = {r["seed"]: r for r in rows
               if r["split"] == split and r["arm"] == a}
    rival = {r["seed"]: r for r in rows
             if r["split"] == split and r["arm"] == b}
    if primary.keys() != rival.keys():
        raise AssertionError("Mismatch in matched ecosystem seeds")
    diffs = [primary[s]["births"] - rival[s]["births"]
             for s in sorted(primary)]
    ci = paired_bootstrap(diffs, seed)
    return {
        "paired_worlds": len(diffs),
        "mean_birth_advantage": statistics.mean(diffs),
        "paired_bootstrap_95": list(ci),
        "positive_worlds": sum(v > 0 for v in diffs),
        "negative_worlds": sum(v < 0 for v in diffs),
        "tied_worlds": sum(v == 0 for v in diffs),
        "by_world": dict(zip(sorted(primary), diffs)),
    }


def render_ecosystem_svg(trace: dict, size: int) -> str:
    cell = 23
    walls = {tuple(x) for x in trace["walls"]}
    food = {tuple(x) for x in trace["initial_food"]}
    individuals = trace["final_agents"]
    shapes = []
    for y in range(size):
        for x in range(size):
            p = (x, y)
            fill = "#64748b" if p in walls else "#e4eddf"
            shapes.append(
                f'<rect x="{cell*x}" y="{cell*y}" width="{cell-1}" '
                f'height="{cell-1}" rx="2" fill="{fill}"/>'
            )
            if p in food:
                shapes.append(
                    f'<circle cx="{cell*x+cell/2}" cy="{cell*y+cell/2}" '
                    f'r="5" fill="#3c9c54"/>'
                )
    for a in individuals:
        x, y = a["pos"]
        shapes.append(
            f'<circle cx="{cell*x+cell/2}" cy="{cell*y+cell/2}" '
            f'r="7" fill="#2563eb" stroke="white" stroke-width="1.4"/>'
        )
    return (f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{size*cell}" height="{size*cell}" '
            f'viewBox="0 0 {size*cell} {size*cell}">'
            + "".join(shapes) + "</svg>\n")


def run(settings: Settings, out: Path) -> dict:
    settings.validate()
    out.mkdir(parents=True, exist_ok=True)
    rows, checkpoints, traces = [], [], []
    for name, first, count, shifted in (
        ("main", settings.seed_start, settings.seeds, False),
        ("shifted", settings.shifted_start, settings.shifted_seeds, True),
    ):
        for seed in range(first, first + count):
            initial_hash = None
            for arm in ARMS:
                capture = seed == first  # One full event/lineage trace per split/arm.
                result, cks, trace = simulate(
                    seed, settings, arm, shifted=shifted, detailed=capture
                )
                if trace:
                    if initial_hash is None:
                        initial_hash = trace["initial_founder_digests"]
                    elif trace["initial_founder_digests"] != initial_hash:
                        raise AssertionError("Matched arms start with unequal founders")
                    traces.append(trace)
                rows.append(result)
                checkpoints.extend(
                    {"seed": seed, "split": name, "arm": arm, **ck}
                    for ck in cks
                )
            print(f"M5.0 ecosystem split={name} seed={seed} "
                  + " ".join(f"{r['arm']}={r['births']}"
                    for r in rows[-len(ARMS):]), flush=True)
    no_food_checks = []
    for seed in range(settings.seed_start,
                      settings.seed_start + settings.no_food_controls):
        row, cks, _ = simulate(
            seed, settings, "heritable_mutation", no_food=True
        )
        if row["food_eaten"] != 0 or row["births"] != 0:
            # With initial E36, age>=4 and basal cost 1, a birth cannot
            # happen before food is consumed (energy<=32 at age four).
            raise AssertionError("Food-free organisms reproduced or ate food")
        if row["alive_final"] != 0:
            raise AssertionError("Energy-free environment failed to cause starvation")
        no_food_checks.append(row)
    rows += no_food_checks
    means = {
        split: {
            arm: {
                key: statistics.mean(
                    r[key] for r in rows
                    if r["split"] == split and r["arm"] == arm
                )
                for key in (
                    "births", "deaths", "alive_final", "food_eaten",
                    "food_spawned", "max_generation", "has_generation_3",
                    "surviving_founder_lineages", "motor_mask_corrections",
                    "ineligible_reproduction_preferences",
                )
            } for arm in ARMS
        } for split in ("main", "shifted")
    }
    main = comparison(
        rows, "main", "heritable_mutation", "random_offspring",
        settings.seed_start + 71
    )
    without_mutation = comparison(
        rows, "main", "heritable_mutation", "heritable_no_mutation",
        settings.seed_start + 97
    )
    shifted = comparison(
        rows, "shifted", "heritable_mutation", "random_offspring",
        settings.shifted_start + 71
    )
    summary = {
        "protocol": "M5.0 preregistered continuous ecological reproduction",
        "settings": asdict(settings),
        "world_seed_is_independent_unit": True,
        "no_external_fitness_or_champion_selection": True,
        "mean": means,
        "primary_heritable_minus_random_offspring": main,
        "mutation_contribution_vs_heritable_clone": without_mutation,
        "shifted_heritable_minus_random_offspring": shifted,
        "main_gen3_world_fraction": means["main"]["heritable_mutation"][
            "has_generation_3"],
        "gates": {
            "primary_births": bool(
                main["mean_birth_advantage"] > 0 and
                main["paired_bootstrap_95"][0] > 0 and
                main["positive_worlds"] >= math.ceil(.75*settings.seeds)
            ),
            "generation_depth": bool(
                means["main"]["heritable_mutation"]["has_generation_3"]
                >= .75
            ),
        },
        "no_food": {
            "tested_worlds": len(no_food_checks),
            "all_extinct": all(r["alive_final"] == 0 for r in no_food_checks),
            "all_no_births": all(r["births"] == 0 for r in no_food_checks),
            "all_no_food": all(r["food_eaten"] == 0 for r in no_food_checks),
        },
        "limits": [
            "Ecology/reproduction/action-energy laws externally designed",
            "No external genotype ranking; no fitness chosen by researcher",
            "Birth count differences alone do not prove cognitive adaptation",
            "Initial food odor is designed by researcher",
            "Genome fixed within life; mutates only in newborn offspring",
            "Founder and descendant count effects share the same world resources",
            "Small worlds and limited epochs may show only extinction",
            "The shifted worlds restart their own evolution, not transferred adults",
            "No predator, organism code execution, online SGD or dynamic topology",
        ],
    }
    with (out/"summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    with (out/"seed_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        keys = list(rows[0])
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    with (out/"population_checkpoints.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        keys = list(checkpoints[0])
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(checkpoints)
    for trace in traces:
        label = trace["split"] + "_" + trace["arm"]
        (out/f"lineages_{label}.json").write_text(
            json.dumps(trace, indent=2, sort_keys=True)+"\n",
            encoding="utf-8"
        )
        if trace["arm"] == "heritable_mutation":
            (out/f"ecosystem_{trace['split']}.svg").write_text(
                render_ecosystem_svg(trace, settings.size),
                encoding="utf-8"
            )
    print("M5.0 MAIN PRIMARY", summary["primary_heritable_minus_random_offspring"],
          flush=True)
    print("M5.0 MAIN MUTATION", summary["mutation_contribution_vs_heritable_clone"],
          flush=True)
    print("M5.0 SHIFTED", summary["shifted_heritable_minus_random_offspring"],
          flush=True)
    print("M5.0 GATES", summary["gates"], flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = Settings()
    for f in fields(Settings):
        v = getattr(defaults, f.name)
        parser.add_argument("--"+f.name.replace("_", "-"),
                            type=type(v), default=v)
    parser.add_argument("--out", type=Path, default=Path("results/m5_0"))
    args = parser.parse_args()
    run(Settings(**{f.name: getattr(args, f.name) for f in fields(Settings)}),
        args.out)


if __name__ == "__main__":
    main()
