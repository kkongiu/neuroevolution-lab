"""M3: unpredictable task recurrence, novelty, and causal state ablations.

This is a fixed, supervised *world simulation*, not an autonomous agent.
Meta-controllers evolve bounded numerical timing policies. All final testing
uses disjoint seed ranges, and no model can modify source code or GitHub.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from .meta import Policy
from .meta_budget import (
    ARMS as M2_ARMS, BudgetSettings, simulate_budget, uniform_plan,
)
from .plasticity import Task, examples, paired_bootstrap


ARMS = ("adaptive", "time_only", "state_blind", "uniform", "permuted",
        "random_units", "no_reset", "fixed_max", "fresh")
MATCHED = ("adaptive", "time_only", "state_blind", "uniform", "permuted",
           "random_units")


@dataclass(frozen=True)
class M3Settings(BudgetSettings):
    train_start: int = 8000
    train_seeds: int = 4
    validation_start: int = 9000
    validation_seeds: int = 3
    test_start: int = 10000
    test_seeds: int = 12
    shifted_start: int = 11000
    shifted_seeds: int = 12
    tasks: int = 10
    steps: int = 48
    width: int = 16
    eval_size: int = 48
    report_every: int = 12
    generations: int = 5
    population: int = 8
    budget_fraction: float = 0.45

    def validate(self) -> None:
        super().validate()
        if self.shifted_start < 0 or self.shifted_seeds < 1:
            raise ValueError("Shifted seeds and start must be valid")
        shifted = set(range(self.shifted_start,
                            self.shifted_start + self.shifted_seeds))
        for start, count in ((self.train_start, self.train_seeds),
                             (self.validation_start, self.validation_seeds),
                             (self.test_start, self.test_seeds)):
            if shifted & set(range(start, start + count)):
                raise ValueError("Shifted seeds overlap an earlier split")
        if self.tasks < 8:
            raise ValueError("M3 recurrence comparison requires >= 8 tasks")


@dataclass(frozen=True)
class WorldTask:
    kind: str
    angle: float
    shift: float
    phase: float

    def label(self, xy: tuple[float, float]) -> int:
        x, y = xy
        if self.kind in ("line", "xor", "circle", "stripe"):
            return Task(self.kind, self.angle, self.shift, self.phase).label(xy)
        if self.kind == "checker":
            return int(math.sin(5 * x + self.angle) *
                       math.sin(5 * y + self.phase) > 0)
        if self.kind == "wedge":
            return int(abs(y - self.phase * 0.1) <
                       0.13 + 0.55 * abs(x - self.shift))
        raise ValueError(self.kind)


def prepare_world(seed: int, settings: M3Settings, *, shifted: bool = False) -> list[dict]:
    """Unpredictable return of exact latent tasks, with new observations.

    Task ID and the novel/repeat status are recorded for analysis only.
    The controller only receives training error/activity and elapsed index.
    """
    rng = random.Random(seed * 37_013 + 47)
    bank: dict[int, WorldTask] = {}
    if shifted:
        kinds = ("line", "checker", "wedge", "stripe")
    else:
        kinds = ("line", "xor", "circle", "stripe")
    sequence = []
    for index in range(settings.tasks):
        returning = index >= 2 and rng.random() < 0.48
        if returning:
            task_id = rng.choice(list(bank))
            status = "return"
        else:
            task_id = len(bank)
            bank[task_id] = WorldTask(
                kind=rng.choice(kinds),
                angle=rng.uniform(-math.pi, math.pi),
                shift=rng.uniform(-0.23, 0.23),
                phase=rng.uniform(-math.pi, math.pi),
            )
            status = "novel"
        task = bank[task_id]
        training = examples(
            task, seed * 2_000_003 + index * 31 + 111,
            settings.steps * settings.batch_size,
        )
        holdout = examples(task, seed * 2_000_003 + index * 31 + 999_777,
                           settings.eval_size)
        sequence.append({
            "train": training, "eval": holdout, "kind": task.kind,
            "status": status, "task_id": task_id,
        })
    return sequence


def restricted(policy: Policy, blind: bool) -> Policy:
    if blind:
        return Policy((policy.weights[0], 0., 0., 0., policy.weights[4]))
    return policy


def search(settings: M3Settings, *, time_only: bool) -> tuple[Policy, list[dict]]:
    train = list(range(settings.train_start,
                       settings.train_start + settings.train_seeds))
    val = list(range(settings.validation_start,
                     settings.validation_start + settings.validation_seeds))
    cache = {seed: prepare_world(seed, settings) for seed in train + val}
    rng = random.Random(settings.search_seed + (7 if time_only else 0))
    arm = "time_only" if time_only else "adaptive"

    def fitness(policy: Policy, seeds: list[int]) -> float:
        scores = []
        for seed in seeds:
            result = simulate_budget(seed, settings, cache[seed], arm, policy)
            scores.append(result["auc"] -
                          settings.retention_cost * result["forgetting"])
        return statistics.mean(scores)

    population = [
        restricted(Policy(weights), time_only)
        for weights in (
            (-3., 0., 0., 0., 0.),
            (0., 0., 0., 0., 0.),
            (3., 0., 0., 0., 0.),
            (-1., 2., 0., 0., -1.),
        )
    ]
    while len(population) < settings.population:
        population.append(restricted(
            Policy(tuple(rng.uniform(-1.5, 1.5) for _ in range(5))),
            time_only,
        ))
    finalists = []
    trace = []
    for generation in range(settings.generations):
        ranked = sorted(
            ((fitness(p, train), p) for p in population),
            key=lambda item: item[0], reverse=True,
        )
        champ = ranked[0][1]
        valid_score = fitness(champ, val)
        finalists.append((valid_score, champ))
        trace.append({
            "generation": generation, "mode": arm,
            "train_objective": ranked[0][0],
            "validation_objective": valid_score,
            "weights": list(champ.weights),
        })
        print(f"M3 meta {arm} gen={generation} train={ranked[0][0]:.4f} "
              f"val={valid_score:.4f}", flush=True)
        elite = [p for _, p in ranked[:max(2, settings.population // 4)]]
        population = list(elite)
        while len(population) < settings.population:
            parent = rng.choice(elite)
            weights = tuple(max(-5, min(5, w + rng.gauss(0, 0.55)))
                            for w in parent.weights)
            population.append(restricted(Policy(weights), time_only))
    return max(finalists, key=lambda item: item[0])[1], trace


def _as_arm(result: dict, name: str) -> dict:
    # Normalizes the diagnostic counterfactual from M2B's time_only arm.
    result = {**result, "arm": name}
    result["rows"] = [{**r, "arm": name} for r in result["rows"]]
    return result


def evaluate_test(seed: int, settings: M3Settings, adaptive: Policy,
                  time_only: Policy, *, shifted: bool) -> list[dict]:
    data = prepare_world(seed, settings, shifted=shifted)
    learned = simulate_budget(seed, settings, data, "adaptive", adaptive)
    clock = simulate_budget(seed, settings, data, "time_only", time_only)
    blind = _as_arm(simulate_budget(seed, settings, data, "time_only", adaptive),
                    "state_blind")
    uniform = uniform_plan(settings)
    permuted = list(learned["actions"])
    random.Random(seed * 417 + 17).shuffle(permuted)
    results = {
        "adaptive": learned,
        "time_only": clock,
        "state_blind": blind,
        "uniform": simulate_budget(seed, settings, data, "uniform",
                                   plan=uniform),
        "permuted": simulate_budget(seed, settings, data, "permuted",
                                    plan=permuted),
        "random_units": simulate_budget(seed, settings, data, "random_units",
                                        plan=learned["actions"]),
        "no_reset": simulate_budget(seed, settings, data, "no_reset"),
        "fixed_max": simulate_budget(seed, settings, data, "fixed_max"),
        "fresh": simulate_budget(seed, settings, data, "fresh"),
    }
    for name in MATCHED:
        assert results[name]["total_resets"] == settings.budget, (name, seed)
    for result in results.values():
        result["shifted"] = shifted
        for row, task in zip(result["rows"], data):
            row["status"] = task["status"]
            row["task_id"] = task["task_id"]
    return [results[name] for name in ARMS]


def seed_metrics(results: list[dict]) -> dict:
    means = {}
    for arm in ARMS:
        rs = [r for r in results if r["arm"] == arm]
        means[arm] = {
            k: statistics.mean(r[k] for r in rs)
            for k in ("auc", "end_accuracy", "forgetting", "total_resets")
        }
        means[arm]["novel_auc"] = statistics.mean(
            row["auc"] for r in rs for row in r["rows"][1:]
            if row["status"] == "novel"
        )
        means[arm]["return_auc"] = statistics.mean(
            row["auc"] for r in rs for row in r["rows"][1:]
            if row["status"] == "return"
        )
    lookup = {arm: {r["seed"]: r for r in results if r["arm"] == arm}
              for arm in ARMS}
    comparisons = {}
    for other in ARMS:
        if other == "adaptive":
            continue
        differences = [
            lookup["adaptive"][seed]["auc"] - lookup[other][seed]["auc"]
            for seed in sorted(lookup["adaptive"])
        ]
        comparisons["adaptive_minus_" + other] = {
            "mean_auc_diff": statistics.mean(differences),
            "paired_bootstrap_95": list(
                paired_bootstrap(differences, min(lookup["adaptive"]) + 21)
            ),
            "positives": sum(d > 0 for d in differences),
            "negatives": sum(d < 0 for d in differences),
            "zero": sum(d == 0 for d in differences),
        }
    adaptive = list(lookup["adaptive"].values())
    all_requested = [a for r in adaptive for a in r["requested"]]
    all_actions = [a for r in adaptive for a in r["actions"]]
    return {
        "means": means, "comparisons": comparisons,
        "state_sensitive_requests": sum(r["state_sensitive_requests"]
                                        for r in adaptive),
        "forced_action_count": sum(r["forced_actions"] for r in adaptive),
        "requests_histogram": {
            str(v): all_requested.count(v) for v in sorted(set(all_requested))
        },
        "actual_histogram": {
            str(v): all_actions.count(v) for v in sorted(set(all_actions))
        },
        "opportunities": len(all_requested),
        "novel_events": sum(
            row["status"] == "novel"
            for result in adaptive for row in result["rows"][1:]
        ),
        "return_events": sum(
            row["status"] == "return"
            for result in adaptive for row in result["rows"][1:]
        ),
    }


def run(settings: M3Settings, output: Path) -> dict:
    settings.validate()
    output.mkdir(parents=True, exist_ok=True)
    adaptive, trace_adaptive = search(settings, time_only=False)
    clock, trace_clock = search(settings, time_only=True)
    print("M3 selected", adaptive.weights, clock.weights, flush=True)
    all_results = []
    for shifted, start, count in (
        (False, settings.test_start, settings.test_seeds),
        (True, settings.shifted_start, settings.shifted_seeds),
    ):
        for seed in range(start, start + count):
            runs = evaluate_test(seed, settings, adaptive, clock, shifted=shifted)
            all_results.extend(runs)
            print(f"M3 heldout seed={seed} shifted={shifted} "
                  f"adaptive_auc={runs[0]['auc']:.4f} "
                  f"clock_auc={runs[1]['auc']:.4f}", flush=True)
    regular = [r for r in all_results if not r["shifted"]]
    shifted_results = [r for r in all_results if r["shifted"]]
    summary = {
        "protocol": "M3 unpredictable recurrence and novel-family causal ablation",
        "settings": asdict(settings),
        "controller_adaptive": list(adaptive.weights),
        "controller_time_only": list(clock.weights),
        "meta_history": trace_adaptive + trace_clock,
        "regular": seed_metrics(regular),
        "shifted": seed_metrics(shifted_results),
        "primary_metric": "shifted test paired AUC: adaptive minus independently trained time_only",
        "evidence_rule": "positive paired AUC and nonzero non-forced sensitivity required; otherwise null",
        "limitations": [
            "Controller coefficients evolved externally, not self-written",
            "Synthetic 2D classification and a fixed 0/half/max intervention grammar",
            "Ablation blinds state but changes controller decisions and schedule",
            "All matched arms use the same total renewal count, not the same CPU-time",
            "Shifted tasks add checker/wedge but retain line/stripe from meta-training",
            "The retained-task metric is aggregate forgetting, not lifelong knowledge",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "meta_history.jsonl").write_text(
        "".join(json.dumps(x, sort_keys=True) + "\n"
                for x in trace_adaptive + trace_clock), encoding="utf-8")

    with (output / "test_seed_metrics.csv").open("w", newline="",
                                                 encoding="utf-8") as f:
        cols = ("seed", "shifted", "arm", "auc", "end_accuracy", "forgetting",
                "total_resets", "actions", "requested", "forced_actions",
                "state_sensitive_requests")
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for result in all_results:
            writer.writerow({k: json.dumps(result[k]) if k in ("actions", "requested")
                             else result[k] for k in cols})
    with (output / "task_metrics.csv").open("w", newline="",
                                            encoding="utf-8") as f:
        cols = ("seed", "shifted", "task_index", "task_id", "status", "kind",
                "arm", "auc", "start_accuracy", "end_accuracy",
                "forgetting", "dead_fraction", "reset_units")
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for result in all_results:
            writer.writerows({**row, "shifted": result["shifted"],
                              "task_index": row["task_index"]}
                             for row in result["rows"])
    print("M3 regular primary",
          summary["regular"]["comparisons"]["adaptive_minus_time_only"],
          flush=True)
    print("M3 shifted primary",
          summary["shifted"]["comparisons"]["adaptive_minus_time_only"],
          flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = M3Settings()
    for field in fields(M3Settings):
        parser.add_argument("--" + field.name.replace("_", "-"),
                            default=getattr(defaults, field.name),
                            type=type(getattr(defaults, field.name)))
    parser.add_argument("--out", type=Path, default=Path("results/m3"))
    args = parser.parse_args()
    settings = M3Settings(**{f.name: getattr(args, f.name)
                              for f in fields(M3Settings)})
    run(settings, args.out)


if __name__ == "__main__":
    main()
