"""M5.5B: isolated physically embodied enemy feasibility study.

Adds externally programmed one-step predators and a ninth heritable fear gain.
Does not modify M5.5A or claim spontaneous enemy recognition or adaptation.
The protocol was committed first: docs/M5_5B_PROTOCOL.md.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from . import m5_0 as base
from . import m5_1 as actuator
from . import m5_5a as innate
from . import m5_4b0 as stats

PROTOCOL_COMMIT = "bac95063d9c0a32fcec2380cd87f36d35b081790"
N_GENES = 9
PREDATOR_LEVELS = (0, 1, 3)
ARMS = (
    "innate_clone", "innate_sparse_mutation", "innate_dense_mutation",
    "innate_random_newborn", "fear_blind_clone",
)
CLONE, SPARSE, DENSE, RANDOM_CHILD, BLIND = ARMS
CONTRASTS = (
    ("sparse_minus_clone", SPARSE, CLONE),
    ("dense_minus_clone", DENSE, CLONE),
    ("random_newborn_minus_clone", RANDOM_CHILD, CLONE),
    ("clone_minus_fear_blind", CLONE, BLIND),
)
BOOT_REPS = 20000


@dataclass(frozen=True)
class Settings(base.Settings):
    seed_start: int = 94000
    seeds: int = 8
    shifted_start: int = 94100
    shifted_seeds: int = 4
    odor_radius: int = 2
    shifted_odor_radius: int = 2
    no_food_controls: int = 0
    innate_initial_gene_sd: float = .65
    innate_mutation_sd: float = .22
    innate_gene_limit: float = 2.5

    def validate(self) -> None:
        super().validate()
        if (self.odor_radius != 2 or self.shifted_odor_radius != 2 or
                self.innate_initial_gene_sd != .65 or
                self.innate_mutation_sd != .22 or
                self.innate_gene_limit != 2.5):
            raise ValueError("M5.5B previously fixed sensory/genetic constants")
        main = set(range(self.seed_start, self.seed_start + self.seeds))
        hard = set(range(self.shifted_start, self.shifted_start + self.shifted_seeds))
        historic = set(range(93000, 93112)) | set(range(94000, 94400))
        allowed = set(range(94000, 94008)) | set(range(94100, 94104))
        if main & hard or (main | hard) & (historic - allowed):
            raise ValueError("Do not leak historical/reserved predator world seeds")
        if self.ticks >= 400:
            canonical = Settings()
            if any(getattr(self, f.name) != getattr(canonical, f.name)
                   for f in fields(Settings)):
                raise ValueError("Full M5.5B exploratory trial is frozen")


def distance(a: base.Point, b: base.Point) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def founder_genes(seed: int, settings: Settings) -> tuple[tuple[float, ...], ...]:
    # The first eight genes are byte-identical to the ORIGINAL random
    # M5.5A founder bank for this NEW independent world seed.
    first_eight = innate.founder_genes(seed, settings)
    rng = random.Random(seed * 9929 + 77213)
    return tuple(g + (rng.gauss(0., settings.innate_initial_gene_sd),)
                 for g in first_eight)


def gene_hash(g: tuple[float, ...]) -> str:
    if len(g) != N_GENES or not all(math.isfinite(x) for x in g):
        raise ValueError("Exactly nine finite numeric genes required")
    return hashlib.sha256(",".join(format(x, ".17g") for x in g)
                          .encode("ascii")).hexdigest()[:24]


def child_genes(g: tuple[float, ...], seed: int, tick: int, aid: int,
                arm: str, settings: Settings) -> tuple[tuple[float, ...], list[int]]:
    gene_hash(g)
    if arm in (CLONE, BLIND):
        return g, []
    rng = base._local_rng(seed, tick, aid, 85)
    if arm == RANDOM_CHILD:
        return (tuple(rng.gauss(0., settings.innate_initial_gene_sd)
                      for _ in range(N_GENES)), list(range(N_GENES)))
    if arm == SPARSE:
        indices = sorted(rng.sample(range(N_GENES), 2))
    elif arm == DENSE:
        indices = list(range(N_GENES))
    else:
        raise ValueError(arm)
    new = list(g)
    for i in indices:
        new[i] = max(-settings.innate_gene_limit,
                     min(settings.innate_gene_limit,
                         new[i] + rng.gauss(0., settings.innate_mutation_sd)))
    return tuple(new), indices


def initial_predators(eco: base.Ecology, count: int) -> list[base.Point]:
    if count not in PREDATOR_LEVELS:
        raise ValueError("Unregistered predator level")
    occupied = {a.position for a in eco.agents.values()}
    candidates = [p for p in eco.accessible if p not in occupied]
    return random.Random(eco.seed * 37613 + 4141).sample(candidates, count)


def choose_action(eco: base.Ecology, agent: base.Agent, tick: int,
                  visits: dict[base.Point, int], heading: int | None,
                  predators: list[base.Point], arm: str) -> tuple[int | None, dict]:
    if arm not in ARMS:
        raise ValueError(arm)
    gene_hash(agent.genome)
    blocked, free = base.blocked_and_free(eco, agent)
    senses = base.observe(eco, agent, blocked)
    legal = [d for d, blocked_d in enumerate(blocked) if not blocked_d]
    params = innate.instinct_parameters(agent.genome[:8], agent.energy, eco.settings)
    can_birth = base.can_reproduce(eco, agent, free)
    will_birth = bool(can_birth and
                      agent.energy >= params["birth_energy_threshold"] and
                      base._local_rng(eco.seed, tick, agent.id, 84).random()
                      < params["birth_probability"])
    visible = [p for p in predators if distance(p, agent.position) <= 2]
    obs = {"enemies_visible": len(visible), "birth_requested": int(will_birth)}
    if will_birth:
        return base.REPRODUCE, obs
    if not legal:
        return None, obs
    fear = (0. if arm == BLIND else
            2.5 + 2.5 * math.tanh(agent.genome[8]))
    noise_rng = base._local_rng(eco.seed, tick, agent.id, 83)
    noise = [noise_rng.uniform(-1., 1.) for _ in range(4)]
    x, y = agent.position
    score = {}
    for d in legal:
        dx, dy = base.DIRECTIONS[d]
        target = (x + dx, y + dy)
        risk = sum(max(0., (3. - distance(target, p)) / 3.)
                   for p in visible)
        score[d] = (
            params["gain"] * (
                params["adjacent_food_weight"] * senses[8 + d] +
                params["odor_weight"] * senses[4 + d]
            ) +
            params["novelty_weight"] / (1. + visits.get(target, 0)) +
            params["persistence_weight"] * int(d == heading) +
            params["random_motor_amplitude"] * noise[d] -
            fear * risk
        )
    top = max(score.values())
    tied = [d for d in legal if abs(score[d] - top) < 1e-12]
    return base._local_rng(eco.seed, tick, agent.id, 37).choice(tied), obs


def step_predators(eco: base.Ecology, predators: list[base.Point],
                   tick: int) -> int:
    """Exactly one possible physical capture per predator per tick."""
    attacks = 0

    def capture(enemy_id: int, pos: base.Point) -> bool:
        nonlocal attacks
        victims = [a for a in eco.agents.values() if a.position == pos]
        if not victims:
            return False
        if len(victims) != 1:
            raise AssertionError("Original physical prey share a cell")
        victim = victims[0]
        if victim.energy <= 0:
            raise AssertionError("Predator met an unremoved dead prey")
        eco.cumulative_death_discard += victim.energy
        eco.deaths += 1
        attacks += 1
        eco.events.append({
            "type": "predation", "tick": tick, "predator": enemy_id,
            "agent": victim.id, "founder": victim.founder,
            "generation": victim.generation, "location": pos,
            "discarded_energy": victim.energy,
            "gene_sha24": gene_hash(victim.genome),
        })
        del eco.agents[victim.id]
        return True

    for index, pos in enumerate(predators):
        if capture(index, pos):
            continue
        others = {p for j, p in enumerate(predators) if j != index}
        legal = [(pos[0] + dx, pos[1] + dy)
                 for dx, dy in base.DIRECTIONS
                 if (pos[0] + dx, pos[1] + dy) not in eco.world.walls
                 and (pos[0] + dx, pos[1] + dy) not in others]
        if not legal:
            continue
        rng = base._local_rng(eco.seed, tick, index, 96)
        nearest = min((distance(pos, a.position) for a in eco.agents.values()),
                      default=10**9)
        if nearest <= 4:
            def target_distance(cell: base.Point) -> int:
                return min((distance(cell, a.position)
                            for a in eco.agents.values()), default=10**9)
            best = min(target_distance(p) for p in legal)
            candidates = [p for p in legal if target_distance(p) == best]
        else:
            candidates = legal
        predators[index] = rng.choice(candidates)
        capture(index, predators[index])
    return attacks


def simulate(seed: int, settings: Settings, arm: str, predator_count: int,
             *, shifted: bool = False, detailed: bool = False
             ) -> tuple[dict, list[dict], dict | None]:
    settings.validate()
    if arm not in ARMS or predator_count not in PREDATOR_LEVELS:
        raise ValueError("Unregistered arm or enemy intensity")
    eco = base.create_ecology(seed, settings, "heritable_no_mutation",
                              shifted=shifted)
    source = founder_genes(seed, settings)
    source_positions = tuple(a.position for a in eco.agents.values())
    for a, g in zip(eco.agents.values(), source):
        a.genome = g
        a.hidden = [0.] * settings.hidden
    eco.initial_genome_digests = tuple(base.genome_digest(g) for g in source)
    food_schedule = base._food_schedule(seed, settings, eco, False)
    predators = initial_predators(eco, predator_count)
    predator_starts = tuple(predators)
    original_world_sha = innate._geometry_sha(eco, food_schedule, source_positions)
    world_sha = hashlib.sha256(json.dumps({
        "original_world_sha": original_world_sha, "predator_count": predator_count,
        "predator_starts": predator_starts
    }, sort_keys=True).encode("utf-8")).hexdigest()
    source_sha = hashlib.sha256(
        json.dumps(source, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    visits = {a.id: {a.position: 1} for a in eco.agents.values()}
    heading = {a.id: None for a in eco.agents.values()}
    checkpoint_rows = []
    genealogy = []
    danger_events = []
    path = []
    kills = 0
    proximity_exposures = 0
    actual_mutations = 0
    fear_mutations = 0
    for tick in range(settings.ticks + 1):
        if tick:
            agent_ids = list(eco.agents)
            base._local_rng(seed, tick, 0, 0).shuffle(agent_ids)
            for aid in agent_ids:
                a = eco.agents.get(aid)
                if a is None:
                    continue
                before_pos, before_g = a.position, a.genome
                old_births, old_moves, newborn_id = (
                    eco.births, eco.move_actions, eco.next_id)
                action, obs = choose_action(eco, a, tick, visits[aid],
                                            heading[aid], predators, arm)
                proximity_exposures += int(obs["enemies_visible"] > 0)
                actuator._act_scripted(eco, aid, tick, action)
                born = eco.births != old_births
                moved = eco.move_actions != old_moves
                if born != (action == base.REPRODUCE):
                    raise AssertionError("Birth request and physical birth differ")
                if moved != (action in range(4) if action is not None else False):
                    raise AssertionError("Movement did not match original actuator")
                if born:
                    child = eco.agents[newborn_id]
                    if child.genome != before_g:
                        raise AssertionError("M5.0 reproduction ceased to clone")
                    g, indices = child_genes(
                        before_g, seed, tick, aid, arm, settings)
                    child.genome = g
                    eco.cosine_births[-1] = base.genome_cosine(before_g, g)
                    event = eco.events[-1]
                    if event["type"] != "birth" or event["child"] != newborn_id:
                        raise AssertionError("Original genealogy lost")
                    event.update(gene_sha24=gene_hash(g),
                                 parent_gene_sha24=gene_hash(before_g),
                                 changed_gene_indices=indices,
                                 genetic_control_arm=arm,
                                 genome=base.genome_digest(g))
                    actual_mutations += int(
                        arm in (SPARSE, DENSE) and g != before_g)
                    fear_mutations += int(8 in indices and g[8] != before_g[8]
                                          and arm in (SPARSE, DENSE))
                    genealogy.append({
                        "seed": seed, "tick": tick, "arm": arm,
                        "parent": aid, "child": newborn_id,
                        "founder": child.founder, "generation": child.generation,
                        "parent_genes": list(before_g),
                        "newborn_genes": list(g), "changed_indices": indices
                    })
                    visits[newborn_id] = {child.position: 1}
                    heading[newborn_id] = None
                if moved:
                    dx, dy = base.DIRECTIONS[action]
                    dest = (before_pos[0] + dx, before_pos[1] + dy)
                    if aid in eco.agents and eco.agents[aid].position != dest:
                        raise AssertionError("Illegal prey move")
                    visits[aid][dest] = visits[aid].get(dest, 0) + 1
                    heading[aid] = action

            before_kills = len(eco.agents)
            tick_kills = step_predators(eco, predators, tick)
            if before_kills - len(eco.agents) != tick_kills:
                raise AssertionError("Predator kill count disagrees with prey deaths")
            kills += tick_kills
            path.append({"tick": tick, "positions": list(predators),
                         "kills": tick_kills})
            if tick_kills:
                danger_events.extend(x for x in eco.events
                                     if x["type"] == "predation" and x["tick"] == tick)
            base._food_regeneration(eco, food_schedule[tick - 1])
            eco.ticks_elapsed = tick
            if not eco.agents and eco.extinct_at is None:
                eco.extinct_at = tick
        if tick == 0 or tick % settings.checkpoint_every == 0 or tick == settings.ticks:
            ck = base.snapshot(eco, tick)
            ck.update({
                "seed": seed, "split": "hard" if shifted else "main",
                "arm": arm, "predators": predator_count,
                "world_sha": world_sha, "founder_sha": source_sha,
                "predation_deaths": kills,
                "mean_fear_gene_alive": (
                    statistics.mean(a.genome[8] for a in eco.agents.values())
                    if eco.agents else None),
            })
            if abs(base.energy_balance(eco)) > 1e-6:
                raise AssertionError("Predation removed unaccounted energy")
            checkpoint_rows.append(ck)
    if len(genealogy) != eco.births or eco.birth_actions != eco.births:
        raise AssertionError("Missing actual hereditary births")
    if kills != len(danger_events):
        raise AssertionError("Missing predation events")
    if predator_count == 0 and kills:
        raise AssertionError("Predation in a zero-enemy control")
    row = {
        "seed": seed, "split": "hard" if shifted else "main",
        "predators": predator_count, "arm": arm,
        "world_sha": world_sha, "original_world_sha": original_world_sha,
        "founder_sha": source_sha,
        "predator_initial_positions": json.dumps(predator_starts),
        "survived": int(bool(eco.agents)), "alive_final": len(eco.agents),
        "births": eco.births, "deaths": eco.deaths,
        "predation_deaths": kills,
        "food_eaten": eco.food_eaten,
        "proximity_exposures": proximity_exposures,
        "mutated_births": actual_mutations,
        "mutated_fear_births": fear_mutations,
        "max_generation": eco.max_generation,
        "mean_fear_gene_alive": (
            statistics.mean(a.genome[8] for a in eco.agents.values())
            if eco.agents else None),
        "population_auc": statistics.mean(c["population"]
                                          for c in checkpoint_rows),
        "energy_ledger_residual": base.energy_balance(eco),
        "original_costed_birth_actions": eco.birth_actions,
    }
    trace = None
    if detailed:
        trace = {
            "seed": seed, "split": row["split"], "arm": arm,
            "predators": predator_count, "world_sha": world_sha,
            "original_world_sha": original_world_sha,
            "founder_sha": source_sha, "source_founder_genes": source,
            "source_founder_positions": source_positions,
            "predator_initial_positions": predator_starts,
            "predator_path": path,
            "predation_events": danger_events,
            "full_birth_genealogy": genealogy,
            "full_original_physical_events": eco.events,
            "checkpoints": checkpoint_rows,
        }
    return row, checkpoint_rows, trace


def summarize(rows: list[dict], settings: Settings,
              bootstrap_reps: int = BOOT_REPS) -> dict:
    result = {}
    for split, start, count in (("main", settings.seed_start, settings.seeds),
                                ("hard", settings.shifted_start,
                                 settings.shifted_seeds)):
        result[split] = {}
        seeds = list(range(start, start + count))
        for pressure in PREDATOR_LEVELS:
            subset = [r for r in rows
                      if r["split"] == split and r["predators"] == pressure]
            if len(subset) != count * len(ARMS):
                raise AssertionError("Missing world/arm trials")
            by = {arm: {r["seed"]: r for r in subset if r["arm"] == arm}
                  for arm in ARMS}
            if any(sorted(by[arm]) != seeds for arm in ARMS):
                raise AssertionError("Missing paired seed/arm")
            for seed in seeds:
                source = [by[arm][seed] for arm in ARMS]
                for key in ("world_sha", "founder_sha",
                            "predator_initial_positions", "original_world_sha"):
                    if len({r[key] for r in source}) != 1:
                        raise AssertionError("Unmatched source "+key)
            arms = {
                arm: {
                    "survivor_worlds": sum(by[arm][seed]["survived"]
                                           for seed in seeds),
                    "mean_population_auc": statistics.mean(
                        by[arm][seed]["population_auc"] for seed in seeds),
                    "mean_predation_deaths": statistics.mean(
                        by[arm][seed]["predation_deaths"] for seed in seeds),
                    "worlds_with_predation": sum(
                        by[arm][seed]["predation_deaths"] > 0 for seed in seeds),
                    "worlds_with_mutated_births": sum(
                        by[arm][seed]["mutated_births"] > 0 for seed in seeds),
                    "mean_births": statistics.mean(
                        by[arm][seed]["births"] for seed in seeds),
                } for arm in ARMS
            }
            comparisons = {}
            for i, (name, left, right) in enumerate(CONTRASTS):
                for measure, offset in (("survived", 0), ("population_auc", 1000)):
                    delta = [by[left][seed][measure] -
                             by[right][seed][measure] for seed in seeds]
                    lo, hi = stats.paired_bootstrap(
                        delta, 95501 + (start - 94000) * 100 +
                        pressure * 10000 + i * 2000 + offset, bootstrap_reps)
                    comparisons[name + "_" + measure] = {
                        "mean_paired_difference": statistics.mean(delta),
                        "world_paired_bootstrap_95": [lo, hi],
                        "per_world": dict(zip(map(str, seeds), delta)),
                    }
            result[split][str(pressure)] = {
                "worlds": count, "arms": arms, "comparisons": comparisons}
    normal = result["main"]
    viable = [
        p for p in (1, 3)
        if normal[str(p)]["arms"][CLONE]["worlds_with_predation"]
           >= math.ceil(settings.seeds / 2)
        and 2 <= normal[str(p)]["arms"][CLONE]["survivor_worlds"] <= 6
        and normal[str(p)]["arms"][SPARSE]["worlds_with_mutated_births"]
           >= math.ceil(settings.seeds / 2)
    ]
    return {
        **result,
        "feasible_pressure_levels_diagnostic_only": viable,
        "adaptation_proven": False,
        "programmed_fear_not_learned": True,
        "causal_lineage_transplant_needed": True,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    fields_ = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields_)
        writer.writeheader()
        writer.writerows(rows)


def run(settings: Settings, output: Path,
        bootstrap_reps: int = BOOT_REPS) -> dict:
    settings.validate()
    output.mkdir(parents=True, exist_ok=True)
    rows, checks, traces = [], [], []
    for split, start, count, hard in (
        ("main", settings.seed_start, settings.seeds, False),
        ("hard", settings.shifted_start, settings.shifted_seeds, True)
    ):
        for seed in range(start, start + count):
            for predator_count in PREDATOR_LEVELS:
                for arm in ARMS:
                    row, ck, detail = simulate(
                        seed, settings, arm, predator_count,
                        shifted=hard, detailed=True)
                    rows.append(row)
                    checks.extend(ck)
                    traces.append(detail)
                    print("M5.5B WORLD " + json.dumps(row, sort_keys=True),
                          flush=True)
    summary = {
        "protocol_commit": PROTOCOL_COMMIT,
        "settings": asdict(settings),
        "worlds": settings.seeds + settings.shifted_seeds,
        "physical_trials": len(rows),
        "checkpoints": len(checks),
        **summarize(rows, settings, bootstrap_reps),
    }
    if settings.ticks >= 400 and len(rows) != 180:
        raise AssertionError("Frozen 180-world exploratory design altered")
    write_csv(output / "worlds.csv", rows)
    write_csv(output / "checkpoints.csv", checks)
    (output / "full_predator_paths_and_genetic_lineages.json").write_text(
        json.dumps(traces, sort_keys=True) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print("M5.5B SUMMARY " + json.dumps({
        "main": summary["main"], "hard": summary["hard"],
        "feasible_pressure_levels": summary["feasible_pressure_levels_diagnostic_only"]
    }, sort_keys=True), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for f in fields(Settings):
        parser.add_argument("--"+f.name.replace("_", "-"),
                            default=f.default, type=type(f.default))
    parser.add_argument("--out", default=Path("results/m5-5b"), type=Path)
    parser.add_argument("--bootstrap-reps", default=BOOT_REPS, type=int)
    kw = vars(parser.parse_args())
    output, reps = kw.pop("out"), kw.pop("bootstrap_reps")
    if reps < 100 or (kw["ticks"] >= 400 and reps != BOOT_REPS):
        raise ValueError("Preregistered full runs need 20k world bootstraps")
    run(Settings(**kw), output, reps)


if __name__ == "__main__":
    main()
