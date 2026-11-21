"""M2B: learn the timing of fixed-budget neural renewal.

All competitive policies renew exactly the same number of units per sequence.
An external evolution strategy selects a bounded controller on disjoint meta
train/validation seeds; final seeds are unseen until policy selection is frozen.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from .meta import MetaSettings, Policy, features, prepare
from .plasticity import Network, accuracy, curve_auc, dead_fraction, paired_bootstrap


ARMS = ("adaptive", "time_only", "uniform", "permuted", "random_units",
        "no_reset", "fixed_max", "fresh")


@dataclass(frozen=True)
class BudgetSettings(MetaSettings):
    train_start: int = 4000
    validation_start: int = 5000
    test_start: int = 6000
    budget_fraction: float = 0.50

    def validate(self) -> None:
        super().validate()
        if not 0 < self.budget_fraction < 1:
            raise ValueError("budget_fraction must be in (0, 1)")
        if not 1 <= self.budget < self.max_units * (self.tasks - 1):
            raise ValueError("Budget must permit some scheduling choices")

    @property
    def budget(self) -> int:
        return round(self.max_units * (self.tasks - 1) * self.budget_fraction)


def uniform_plan(settings: BudgetSettings) -> list[int]:
    """Spread the exact budget as evenly as possible over transitions."""
    intervals = settings.tasks - 1
    return [
        ((i + 1) * settings.budget) // intervals -
        (i * settings.budget) // intervals
        for i in range(intervals)
    ]


def network_initialization(seed: int, width: int) -> Network:
    return Network.create(random.Random(seed * 13_019 + 31), width)


def simulate_budget(seed: int, settings: BudgetSettings, dataset: list[dict],
                    arm: str, policy: Policy | None = None,
                    plan: list[int] | None = None) -> dict:
    if arm not in ARMS:
        raise ValueError(arm)
    if arm in ("adaptive", "time_only") and policy is None:
        raise ValueError("A policy is required for adaptive arms")
    if arm in ("uniform", "permuted", "random_units"):
        if plan is None or len(plan) != settings.tasks - 1:
            raise ValueError("Matched arms require full action plan")
        if sum(plan) != settings.budget or any(n < 0 or n > settings.max_units
                                                 for n in plan):
            raise ValueError("Matched action plan has wrong reset budget")

    net = network_initialization(seed, settings.width)
    earlier_evaluations = []
    earlier_end = []
    actions = []
    requested = []
    forced = 0
    different_from_time_only = 0
    remaining = settings.budget
    rows = []
    for task_index, item in enumerate(dataset):
        count = 0
        if task_index:
            if arm in ("adaptive", "time_only"):
                observed = features(net, item["train"],
                                    dataset[task_index - 1]["train"],
                                    task_index, settings)
                if arm == "time_only":
                    observed = (observed[0], 0., 0., 0., observed[4])
                desired = policy.action(observed, settings.max_units)
                if arm == "adaptive":
                    time_observed = (observed[0], 0., 0., 0., observed[4])
                    different_from_time_only += int(
                        desired != policy.action(time_observed, settings.max_units)
                    )
                transitions_left = settings.tasks - task_index
                lower = max(0, remaining - settings.max_units *
                            (transitions_left - 1))
                upper = min(settings.max_units, remaining)
                count = max(lower, min(upper, desired))
                forced += int(count != desired)
                remaining -= count
                requested.append(desired)
            elif arm in ("uniform", "permuted", "random_units"):
                count = plan[task_index - 1]
            elif arm == "fixed_max":
                count = settings.max_units
            elif arm == "fresh":
                net = Network.create(
                    random.Random(seed * 13_019 + task_index * 997 + 31),
                    settings.width,
                )
            if count:
                past_x = [xy for xy, _ in dataset[task_index - 1]["train"][-64:]]
                net.renew(
                    count,
                    random.Random(seed * 349_019 + task_index * 701 + 23),
                    past_x,
                    low_activity=(arm != "random_units"),
                )
            actions.append(count)
        measured = [(0, accuracy(net, item["eval"]))]
        for step in range(1, settings.steps + 1):
            start = (step - 1) * settings.batch_size
            net.train_batch(item["train"][start:start + settings.batch_size],
                            settings.learning_rate)
            if step % settings.report_every == 0:
                measured.append((step, accuracy(net, item["eval"])))
        end = measured[-1][1]
        forgetting = (statistics.mean(
            max(0.0, old_end - accuracy(net, old_eval))
            for old_end, old_eval in zip(earlier_end, earlier_evaluations)
        ) if earlier_end else 0.0)
        rows.append({
            "seed": seed, "task_index": task_index, "kind": item["kind"],
            "arm": arm, "auc": curve_auc(measured, settings.steps),
            "start_accuracy": measured[0][1], "end_accuracy": end,
            "forgetting": forgetting,
            "dead_fraction": dead_fraction(net, item["eval"]),
            "reset_units": count,
        })
        earlier_evaluations.append(item["eval"])
        earlier_end.append(end)
    if arm in ("adaptive", "time_only") and remaining:
        raise AssertionError("Budget was not exhausted")

    selected = rows[1:]
    return {
        "seed": seed, "arm": arm, "actions": actions, "requested": requested,
        "forced_actions": forced,
        "state_sensitive_requests": different_from_time_only,
        "total_resets": sum(actions),
        "auc": statistics.mean(r["auc"] for r in selected),
        "end_accuracy": statistics.mean(r["end_accuracy"] for r in selected),
        "forgetting": statistics.mean(r["forgetting"] for r in selected),
        "rows": rows,
    }


def restricted(policy: Policy, time_only: bool) -> Policy:
    if not time_only:
        return policy
    return Policy((policy.weights[0], 0., 0., 0., policy.weights[4]))


def select_policy(settings: BudgetSettings, time_only: bool) -> tuple[Policy, list[dict]]:
    train = list(range(settings.train_start,
                       settings.train_start + settings.train_seeds))
    valid = list(range(settings.validation_start,
                       settings.validation_start + settings.validation_seeds))
    cached = {seed: prepare(seed, settings) for seed in train + valid}
    rng = random.Random(settings.search_seed + (7 if time_only else 0))

    def objective(policy: Policy, seeds: list[int]) -> float:
        values = []
        for seed in seeds:
            result = simulate_budget(seed, settings, cached[seed],
                                     "time_only" if time_only else "adaptive",
                                     policy)
            # Renewal cost is constant across eligible candidates by design.
            values.append(result["auc"] -
                          settings.retention_cost * result["forgetting"])
        return statistics.mean(values)

    initial = [
        Policy((-3., 0., 0., 0., 0.)),
        Policy((0., 0., 0., 0., 0.)),
        Policy((3., 0., 0., 0., 0.)),
        Policy((-1., 2., 0., 0., -1.)),
    ]
    population = [restricted(p, time_only) for p in initial]
    population.extend(restricted(Policy(tuple(rng.uniform(-1.5, 1.5)
                                              for _ in range(5))), time_only)
                      for _ in range(settings.population - len(initial)))
    finalists = []
    trace = []
    for generation in range(settings.generations):
        ranked = [(objective(p, train), p) for p in population]
        ranked.sort(key=lambda pair: pair[0], reverse=True)
        best = ranked[0][1]
        valid_score = objective(best, valid)
        finalists.append((valid_score, best))
        trace.append({
            "generation": generation, "mode": "time_only" if time_only else "adaptive",
            "training_fitness": ranked[0][0], "validation_fitness": valid_score,
            "weights": list(best.weights),
        })
        print(f"{trace[-1]['mode']} generation={generation} "
              f"train={ranked[0][0]:.4f} validation={valid_score:.4f}",
              flush=True)
        elites = [p for _, p in ranked[:max(2, settings.population // 4)]]
        population = list(elites)
        while len(population) < settings.population:
            parent = rng.choice(elites)
            weights = tuple(max(-5., min(5., w + rng.gauss(0., .55)))
                            for w in parent.weights)
            population.append(restricted(Policy(weights), time_only))
    return max(finalists, key=lambda item: item[0])[1], trace


def summarize(results: list[dict], settings: BudgetSettings,
              adaptive: Policy, time_only: Policy,
              traces: list[dict]) -> dict:
    lookups = {
        arm: {row["seed"]: row for row in results if row["arm"] == arm}
        for arm in ARMS
    }
    metrics = ("auc", "end_accuracy", "forgetting", "total_resets")
    means = {
        arm: {key: statistics.mean(row[key] for row in records.values())
              for key in metrics}
        for arm, records in lookups.items()
    }
    comparisons = {}
    for other in ("time_only", "uniform", "permuted", "random_units",
                  "no_reset", "fixed_max", "fresh"):
        diffs = [
            lookups["adaptive"][seed]["auc"] - lookups[other][seed]["auc"]
            for seed in sorted(lookups["adaptive"])
        ]
        comparisons["adaptive_minus_" + other] = {
            "mean_auc_difference": statistics.mean(diffs),
            "paired_bootstrap_95_percentile_interval": list(
                paired_bootstrap(diffs, settings.test_start + 17)
            ),
            "positive_seeds": sum(value > 0 for value in diffs),
            "negative_seeds": sum(value < 0 for value in diffs),
            "zero_seeds": sum(value == 0 for value in diffs),
        }
    adaptive_results = list(lookups["adaptive"].values())
    action_counts = [a for row in adaptive_results for a in row["actions"]]
    requested_counts = [a for row in adaptive_results for a in row["requested"]]
    return {
        "settings": asdict(settings),
        "budget_units_per_sequence": settings.budget,
        "policy_adaptive_weights": list(adaptive.weights),
        "policy_time_only_weights": list(time_only.weights),
        "meta_search_trace": traces,
        "arm_means": means,
        "paired_test_auc_differences": comparisons,
        "adaptive_actual_action_histogram": {
            str(v): action_counts.count(v) for v in range(settings.max_units + 1)
        },
        "adaptive_requested_action_histogram": {
            str(v): requested_counts.count(v) for v in range(settings.max_units + 1)
        },
        "adaptive_forced_boundaries": sum(r["forced_actions"]
                                         for r in adaptive_results),
        "adaptive_requests_sensitive_to_training_state": sum(
            r["state_sensitive_requests"] for r in adaptive_results
        ),
        "primary": "paired seed-level held-out AUC: adaptive minus time_only",
        "interpretation_limits": [
            "Constant reset budget controls unit counts but not neural evaluation time",
            "Meta-trained controller is externally evolved, not self-rewriting",
            "Known task families; seeds are new, task distributions are not",
            "Budget forcing at late transitions must not be mistaken for adaptation",
            "Meta-validation chose between generation winners, never using final test",
        ],
    }


def run(settings: BudgetSettings, output: Path) -> dict:
    settings.validate()
    output.mkdir(parents=True, exist_ok=True)
    adaptive_policy, trace1 = select_policy(settings, False)
    time_policy, trace2 = select_policy(settings, True)
    results = []
    for seed in range(settings.test_start,
                      settings.test_start + settings.test_seeds):
        dataset = prepare(seed, settings)
        adaptive = simulate_budget(seed, settings, dataset, "adaptive",
                                   adaptive_policy)
        timed = simulate_budget(seed, settings, dataset, "time_only",
                                time_policy)
        equal = uniform_plan(settings)
        permutation = list(adaptive["actions"])
        random.Random(seed * 417 + 17).shuffle(permutation)
        arms = {
            "adaptive": adaptive,
            "time_only": timed,
            "uniform": simulate_budget(seed, settings, dataset, "uniform",
                                       plan=equal),
            "permuted": simulate_budget(seed, settings, dataset, "permuted",
                                        plan=permutation),
            "random_units": simulate_budget(seed, settings, dataset,
                                             "random_units",
                                             plan=adaptive["actions"]),
            "no_reset": simulate_budget(seed, settings, dataset, "no_reset"),
            "fixed_max": simulate_budget(seed, settings, dataset, "fixed_max"),
            "fresh": simulate_budget(seed, settings, dataset, "fresh"),
        }
        if any(arms[key]["total_resets"] != settings.budget for key in (
                "adaptive", "time_only", "uniform", "permuted", "random_units")):
            raise AssertionError("Dose-matched comparisons must use exact budget")
        results.extend(arms[key] for key in ARMS)
        print(f"test seed={seed} adaptive_auc={adaptive['auc']:.4f} "
              f"action_counts={adaptive['actions']}", flush=True)
    summary = summarize(results, settings, adaptive_policy, time_policy,
                        trace1 + trace2)
    summary["search_notes"] = "Meta-search on train, winner selection on validation"
    (output / "summary.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    (output / "meta_training.jsonl").write_text(
        "".join(json.dumps(t, sort_keys=True) + "\n" for t in trace1 + trace2),
        encoding="utf-8"
    )
    with (output / "test_seed_metrics.csv").open("w", newline="",
                                                 encoding="utf-8") as handle:
        columns = ["seed", "arm", "auc", "end_accuracy", "forgetting",
                   "total_resets", "actions", "requested",
                   "forced_actions", "state_sensitive_requests"]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for result in results:
            row = {key: result[key] for key in columns}
            row["actions"] = json.dumps(row["actions"])
            row["requested"] = json.dumps(row["requested"])
            writer.writerow(row)
    with (output / "task_metrics.csv").open("w", newline="",
                                            encoding="utf-8") as handle:
        columns = ["seed", "task_index", "kind", "arm", "auc", "start_accuracy",
                   "end_accuracy", "forgetting", "dead_fraction", "reset_units"]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for result in results:
            writer.writerows(result["rows"])
    print("heldout comparison", json.dumps(summary["paired_test_auc_differences"],
                                           sort_keys=True), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = BudgetSettings()
    for field in fields(BudgetSettings):
        default = getattr(defaults, field.name)
        parser.add_argument("--" + field.name.replace("_", "-"),
                            type=type(default), default=default)
    parser.add_argument("--out", type=Path, default=Path("results/m2b"))
    args = parser.parse_args()
    settings = BudgetSettings(**{f.name: getattr(args, f.name)
                                  for f in fields(BudgetSettings)})
    run(settings, args.out)


if __name__ == "__main__":
    main()
