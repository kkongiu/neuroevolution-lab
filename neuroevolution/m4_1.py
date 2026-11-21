"""M4.1: independently replicated evolution with a local legal-action decoder.

A factorial comparison holds the RNN genotype fixed while switching the
wall-action mask at deployment. Eight independent genetic initializations face
rotated mazes and variable starts. No within-life synaptic learning.
M4.0 is imported unchanged and its results remain frozen.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from .m4_0 import (
    ACTIONS, ARMS as OLD_ARMS, DIRECTIONS, Settings as BaseSettings,
    World, RecurrentController, apply_action, decide_baseline,
    fresh_genome, initial_state, make_world, mutate, observation,
    render_world_svg,
)

ARMS = (
    "evolved_masked",
    "evolved_masked_unmasked",
    "evolved_unmasked",
    "evolved_unmasked_masked",
    "random_neural_masked",
    "wall_avoiding_random",
    "greedy_odor",
    "evolved_masked_no_odor",
)
NEURAL_ARMS = frozenset(ARMS[:5] + ARMS[7:])
MASKED_ARMS = frozenset((
    "evolved_masked", "evolved_unmasked_masked",
    "random_neural_masked", "evolved_masked_no_odor",
))
CONTRASTS = (
    ("evolved_masked", "wall_avoiding_random", "primary_vs_walker"),
    ("evolved_masked", "random_neural_masked", "secondary_vs_random_neural"),
    ("evolved_masked", "greedy_odor", "versus_greedy"),
    ("evolved_masked", "evolved_masked_unmasked", "mask_effect_on_mask_trained"),
    ("evolved_unmasked_masked", "evolved_unmasked", "mask_effect_on_unmasked_trained"),
    ("evolved_masked", "evolved_unmasked_masked", "training_mask_effect"),
    ("evolved_masked", "evolved_masked_no_odor", "odor_ablation"),
)


@dataclass(frozen=True)
class Settings(BaseSettings):
    train_start: int = 45000
    train_worlds: int = 6
    validation_start: int = 46000
    validation_worlds: int = 4
    test_start: int = 47000
    test_worlds: int = 24
    shifted_start: int = 48000
    shifted_worlds: int = 12
    population: int = 10
    generations: int = 6
    elites: int = 2
    immigrants: int = 2
    genetic_seed_start: int = 81000
    genetic_runs: int = 8

    def validate(self) -> None:
        super().validate()
        if self.genetic_seed_start < 0 or self.genetic_runs < 2:
            raise ValueError("Require independent, distinct genetic runs")
        groups = (
            (self.train_start, self.train_worlds * self.generations),
            (self.validation_start, self.validation_worlds),
            (self.test_start, self.test_worlds),
            (self.shifted_start, self.shifted_worlds),
        )
        for i, (start, length) in enumerate(groups):
            a = set(range(start, start + length))
            for nxt, other in groups[i + 1:]:
                if a.intersection(range(nxt, nxt + other)):
                    raise ValueError("Training generations overlap sealed world seeds")


def rotate(p: tuple[int, int], size: int, quarter_turns: int) -> tuple[int, int]:
    x, y = p
    for _ in range(quarter_turns):
        x, y = size - 1 - y, x
    return x, y


def make_diverse_world(seed: int, settings: Settings,
                       shifted: bool = False) -> World:
    """Rotate the entire map and randomize a reachable starting location.

    Food is regenerated only by the original M4.0 world generator, and is
    reachable before and after rotation and relocation.
    """
    base = make_world(seed, settings, shifted)
    rng = random.Random(seed * 2111 + 887)
    quarter_turns = rng.randrange(4)
    walls = frozenset(rotate(p, base.size, quarter_turns) for p in base.walls)
    food = tuple(rotate(p, base.size, quarter_turns) for p in base.initial_food)
    reachable = base.reachable()
    candidates = [
        rotate(p, base.size, quarter_turns)
        for p in sorted(reachable)
        if rotate(p, base.size, quarter_turns) not in food
        and sum(
            (rotate(p, base.size, quarter_turns)[0] + dx,
             rotate(p, base.size, quarter_turns)[1] + dy) not in walls
            for dx, dy in DIRECTIONS
        ) >= 2
    ]
    if not candidates:
        raise AssertionError("No reachable unoccupied starting cell")
    start = rng.choice(candidates)
    world = World(seed, base.size, start, walls, food,
                  base.odor_radius, shifted)
    if any(f not in world.reachable() for f in food):
        raise AssertionError("Food unreachable after start randomization")
    return world


class MaskAwareRNN(RecurrentController):
    """Compute the SAME RNN state and logits, then mask blocked directions."""

    def __init__(self, genome: tuple[float, ...], hidden: int,
                 masked: bool, no_odor: bool = False):
        super().__init__(genome, hidden, no_odor=no_odor)
        self.masked = masked
        self.illegal_preferences = 0
        self.corrected_preferences = 0

    def choose(self, obs: tuple[float, ...]) -> int:
        original = super().choose(obs)
        if obs[original] < 0.5:
            return original
        self.illegal_preferences += 1
        if not self.masked:
            return original
        allowed = [i for i in range(ACTIONS) if obs[i] < 0.5]
        if not allowed:
            return original
        logits = []
        for i in range(ACTIONS):
            offset = self.offset + i * (self.width + 1)
            logits.append(
                sum(w * h for w, h in zip(
                    self.genome[offset:offset + self.width], self.hidden
                )) + self.genome[offset + self.width]
            )
        self.corrected_preferences += 1
        return max(allowed, key=lambda i: (logits[i], -i))


def simulate(world: World, settings: Settings, arm: str,
             genome: tuple[float, ...] | None = None,
             trace: bool = False) -> dict:
    if arm not in ARMS:
        raise ValueError(arm)
    if arm in NEURAL_ARMS and genome is None:
        raise ValueError("Neural policy requires a numeric genome")
    neural = None
    if arm in NEURAL_ARMS:
        neural = MaskAwareRNN(
            genome, settings.hidden,
            masked=arm in MASKED_ARMS,
            no_odor=arm == "evolved_masked_no_odor",
        )
    rng = random.Random(world.seed * 1717 + (0 if arm == "wall_avoiding_random"
                                             else 97) + 7)
    state = initial_state(world, settings)
    path = [world.start] if trace else []
    events = []
    odor_steps = 0
    while state.energy > 0 and state.steps < settings.horizon:
        senses = observation(world, state, settings)
        odor_steps += int(any(s > 0 for s in senses[4:8]))
        if neural is None:
            decision = decide_baseline(
                senses,
                "wall_avoiding_random" if arm == "wall_avoiding_random"
                else "greedy_odor",
                rng
            )
        else:
            decision = neural.choose(senses)
        apply_action(world, state, decision, settings)
        if trace:
            path.append(state.position)
            events.append({
                "step": state.steps,
                "position": list(state.position),
                "action": decision,
                "energy": round(state.energy, 4),
                "food": state.eaten,
                "collision": state.last_collision,
                "collected": state.last_food,
            })
    return {
        "seed": world.seed,
        "arm": arm,
        "shifted": world.shifted,
        "food": state.eaten,
        "steps": state.steps,
        "collisions": state.collisions,
        "corrected_preferences": neural.corrected_preferences if neural else 0,
        "illegal_preferences": neural.illegal_preferences if neural else 0,
        "visited": len(state.visited),
        "energy": state.energy,
        "starved": int(state.energy <= 0),
        "odor_steps": odor_steps,
        "fitness": 2 * state.eaten + 0.03 * state.steps - 0.05 * state.collisions,
        "path": path,
        "events": events,
    }


def evolve_one(settings: Settings, genetic_seed: int,
               masked_training: bool) -> tuple[tuple[float, ...], list[dict],dict]:
    """Matched initial populations and identical world/search budgets."""
    rng = random.Random(genetic_seed)
    population = [fresh_genome(rng, settings)
                  for _ in range(settings.population)]
    fixed_validation = [
        make_diverse_world(seed, settings)
        for seed in range(settings.validation_start,
                          settings.validation_start + settings.validation_worlds)
    ]
    history = []
    finalists = []
    evaluations = 0
    arm = "evolved_masked" if masked_training else "evolved_unmasked"
    for generation in range(settings.generations):
        training = [
            make_diverse_world(
                settings.train_start + generation * settings.train_worlds + i,
                settings,
            )
            for i in range(settings.train_worlds)
        ]
        ranked = []
        for genome in population:
            runs = [simulate(world, settings, arm, genome) for world in training]
            evaluations += len(runs)
            ranked.append((
                statistics.mean(r["fitness"] for r in runs),
                genome,
                statistics.mean(r["food"] for r in runs),
            ))
        ranked.sort(key=lambda r: r[0], reverse=True)
        candidate = ranked[0][1]
        validation = [simulate(w, settings, arm, candidate)
                      for w in fixed_validation]
        val_score = statistics.mean(r["fitness"] for r in validation)
        finalists.append((val_score, generation, candidate))
        history.append({
            "genetic_seed": genetic_seed,
            "training_mode": "masked" if masked_training else "unmasked",
            "generation": generation,
            "best_train_fitness": ranked[0][0],
            "mean_train_fitness": statistics.mean(r[0] for r in ranked),
            "best_train_food": ranked[0][2],
            "validation_fitness": val_score,
            "validation_food": statistics.mean(r["food"] for r in validation),
            "candidate_world_evaluations": evaluations,
        })
        print(
            f"M4.1 evolution gene={genetic_seed} mode={arm}"
            f" gen={generation} best_food={ranked[0][2]:.3f}"
            f" val={val_score:.3f}", flush=True
        )
        elites = [r[1] for r in ranked[:settings.elites]]
        population = list(elites)
        while len(population) < settings.population - settings.immigrants:
            population.append(mutate(rng.choice(elites), rng, settings))
        while len(population) < settings.population:
            population.append(fresh_genome(rng, settings))
    best_score, best_generation, best_genome = max(
        finalists, key=lambda r: (r[0], -r[1])
    )
    expected = settings.generations * settings.population * settings.train_worlds
    if evaluations != expected:
        raise AssertionError("Evolutionary evaluation budget exceeded")
    return best_genome, history, {
        "genetic_seed": genetic_seed,
        "mode": "masked" if masked_training else "unmasked",
        "validation_fitness": best_score,
        "selected_generation": best_generation,
        "candidate_world_evaluations": evaluations,
        "validation_world_evaluations":
            settings.generations * settings.validation_worlds,
    }


def hierarchical_bootstrap(matrix: list[list[float]], seed: int,
                           draws: int = 1600) -> tuple[float, float]:
    """Two-level resampling: independent genetic seeds and shared world seeds.

    Resample genetic-run rows and world columns, preserving every paired
    observation at the selected intersection. Do not treat n_runs*n_worlds as
    independent evolutionary replicates.
    """
    if not matrix or not matrix[0] or any(len(r) != len(matrix[0]) for r in matrix):
        raise ValueError("Expected rectangular nonempty matrix")
    rng = random.Random(seed)
    nrun, nworld = len(matrix), len(matrix[0])
    samples = []
    for _ in range(draws):
        irun = [rng.randrange(nrun) for _ in range(nrun)]
        iworld = [rng.randrange(nworld) for _ in range(nworld)]
        samples.append(
            sum(matrix[i][j] for i in irun for j in iworld) / (nrun * nworld)
        )
    samples.sort()
    return samples[int(draws * 0.025)], samples[int(draws * 0.975)]


def summarize(records: list[dict], genetic_seeds: list[int],
              world_seeds: list[int], label: str) -> dict:
    index = {(r["genetic_seed"], r["seed"], r["arm"]): r for r in records}
    metrics = ("food", "steps", "collisions", "corrected_preferences",
               "illegal_preferences", "visited", "energy", "starved", "odor_steps")
    means = {
        arm: {key: statistics.mean(index[g, w, arm][key]
                                  for g in genetic_seeds for w in world_seeds)
              for key in metrics}
        for arm in ARMS
    }
    comparisons = {}
    for a, b, name in CONTRASTS:
        matrix = [
            [index[g, w, a]["food"] - index[g, w, b]["food"]
             for w in world_seeds]
            for g in genetic_seeds
        ]
        per_run = [statistics.mean(row) for row in matrix]
        per_world = [
            statistics.mean(matrix[i][j] for i in range(len(genetic_seeds)))
            for j in range(len(world_seeds))
        ]
        ci = hierarchical_bootstrap(
            matrix, min(genetic_seeds) * 131 + min(world_seeds)
            + list(x[2] for x in CONTRASTS).index(name)
        )
        comparisons[name] = {
            "delta_food": statistics.mean(per_run),
            "hierarchical_bootstrap_95": list(ci),
            "per_genetic_run_deltas": per_run,
            "per_world_deltas": per_world,
            "positive_genetic_runs": sum(x > 0 for x in per_run),
            "negative_genetic_runs": sum(x < 0 for x in per_run),
            "tied_genetic_runs": sum(x == 0 for x in per_run),
            "positive_world_means": sum(x > 0 for x in per_world),
            "negative_world_means": sum(x < 0 for x in per_world),
        }
    return {
        "split": label,
        "independent_genetic_runs": len(genetic_seeds),
        "shared_independent_worlds": len(world_seeds),
        "descriptive_observations": len(genetic_seeds) * len(world_seeds),
        "means": means,
        "contrasts": comparisons,
        "evolved_masked_ate_food_fraction": sum(
            index[g, w, "evolved_masked"]["food"] > 0
            for g in genetic_seeds for w in world_seeds
        ) / (len(genetic_seeds) * len(world_seeds)),
        "primary_gate": (
            comparisons["primary_vs_walker"]["delta_food"] > 0
            and comparisons["primary_vs_walker"]["hierarchical_bootstrap_95"][0] > 0
            and comparisons["primary_vs_walker"]["positive_genetic_runs"] >= 6
        ),
    }


def run(settings: Settings, output: Path) -> dict:
    settings.validate()
    output.mkdir(parents=True, exist_ok=True)
    genetics = [settings.genetic_seed_start + i
                for i in range(settings.genetic_runs)]
    main_seeds = list(range(settings.test_start,
                            settings.test_start + settings.test_worlds))
    shifted_seeds = list(range(settings.shifted_start,
                               settings.shifted_start + settings.shifted_worlds))
    main_worlds = [make_diverse_world(s, settings) for s in main_seeds]
    shifted_worlds = [make_diverse_world(s, settings, True) for s in shifted_seeds]
    genomes = {}
    histories = []
    selections = []
    results = []
    null_results = []
    for genetic_seed in genetics:
        masked, mask_hist, mask_selection = evolve_one(settings, genetic_seed, True)
        unmasked, unmask_hist, unmask_selection = evolve_one(settings, genetic_seed, False)
        histories.extend(mask_hist + unmask_hist)
        selections.extend((mask_selection, unmask_selection))
        genomes[genetic_seed] = {"masked": list(masked), "unmasked": list(unmasked)}
        reference = fresh_genome(
            random.Random(settings.random_neural_seed + genetic_seed * 1973),
            settings
        )
        policies = {
            "evolved_masked": masked,
            "evolved_masked_unmasked": masked,
            "evolved_unmasked": unmasked,
            "evolved_unmasked_masked": unmasked,
            "random_neural_masked": reference,
            "evolved_masked_no_odor": masked,
        }
        for split, worlds in (("main", main_worlds),
                              ("shifted", shifted_worlds)):
            for world in worlds:
                for arm in ARMS:
                    r = simulate(world, settings, arm, policies.get(arm))
                    results.append({
                        **{k: v for k, v in r.items() if k not in ("path", "events")},
                        "split": split, "genetic_seed": genetic_seed,
                    })
            print(
                f"M4.1 evaluation seed={genetic_seed} split={split}"
                f" complete worlds={len(worlds)}", flush=True
            )
        # Negative control on the first eight main worlds; exactly zero food.
        for world in main_worlds[:8]:
            empty_world = World(
                world.seed, world.size, world.start,
                world.walls, (), world.odor_radius, world.shifted
            )
            null = simulate(empty_world, settings, "evolved_masked", masked)
            if null["food"] != 0 or null["energy"] > settings.initial_energy:
                raise AssertionError("Null world created impossible food or energy")
            null_results.append({
                "genetic_seed": genetic_seed,
                "world_seed": world.seed,
                "steps": null["steps"],
                "food": null["food"],
                "corrected": null["corrected_preferences"],
                "collisions": null["collisions"],
            })
    summary = {
        "protocol": "M4.1 preregistered motor mask and independent genetic searches",
        "settings": asdict(settings),
        "evolution_selection": selections,
        "main": summarize(results, genetics, main_seeds, "main"),
        "shifted": summarize(results, genetics, shifted_seeds, "shifted"),
        "null_food": {
            "observations": len(null_results),
            "all_zero_food": all(r["food"] == 0 for r in null_results),
            "mean_steps": statistics.mean(r["steps"] for r in null_results),
            "mean_corrected_preferences": statistics.mean(
                r["corrected"] for r in null_results),
        },
        "interpretation": [
            "Legal-action decoder is researcher-provided motor protection, not learning",
            "Genomes selected on validation, no final test-seed tuning",
            "Two-factor crossing separates mask at training from mask at evaluation",
            "Eight independent genetic searches share fixed world seeds; hierarchical bootstrap resamples both",
            "Odor is a designed navigation cue, available to the greedy controller",
            "No within-life learning, predator, within-world reproduction or self-generated objectives",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8"
    )
    (output / "champions.json").write_text(
        json.dumps({
            "genetic_replicates": genetics, "genomes": genomes,
            "selection": selections,
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8"
    )
    with (output / "train_validation.csv").open("w", newline="",
                                                encoding="utf-8") as f:
        cols = list(histories[0])
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(histories)
    with (output / "test_world_metrics.csv").open("w", newline="",
                                                  encoding="utf-8") as f:
        cols = ("split", "genetic_seed", "seed", "arm", "food", "steps",
                "collisions", "corrected_preferences", "illegal_preferences",
                "visited", "energy", "starved", "odor_steps", "fitness")
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows({k: r[k] for k in cols} for r in results)
    with (output / "null_food.csv").open("w", newline="",
                                        encoding="utf-8") as f:
        cols = list(null_results[0])
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(null_results)

    for world, label in ((main_worlds[0], "main"),
                         (shifted_worlds[0], "shifted")):
        detail = simulate(
            world, settings, "evolved_masked",
            tuple(genomes[genetics[0]]["masked"]), trace=True
        )
        (output / f"example_{label}.svg").write_text(
            render_world_svg(world, detail["path"], settings), encoding="utf-8"
        )
        (output / f"example_{label}_trace.json").write_text(
            json.dumps({
                "world_seed": world.seed,
                "genetic_seed": genetics[0],
                "start": list(world.start),
                "food": [list(x) for x in world.initial_food],
                "metrics": {k: v for k, v in detail.items()
                            if k not in ("path", "events")},
                "events": detail["events"],
            }, indent=2) + "\n", encoding="utf-8"
        )
    print("M4.1 MAIN PRIMARY",
          summary["main"]["contrasts"]["primary_vs_walker"], flush=True)
    print("M4.1 MAIN GREEDY",
          summary["main"]["contrasts"]["versus_greedy"], flush=True)
    print("M4.1 SHIFTED",
          summary["shifted"]["contrasts"]["primary_vs_walker"], flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = Settings()
    for field in fields(Settings):
        val = getattr(defaults, field.name)
        parser.add_argument("--" + field.name.replace("_", "-"),
                            type=type(val), default=val)
    parser.add_argument("--out", type=Path, default=Path("results/m4_1"))
    args = parser.parse_args()
    settings = Settings(**{
        fld.name: getattr(args, fld.name) for fld in fields(Settings)
    })
    run(settings, args.out)


if __name__ == "__main__":
    main()
