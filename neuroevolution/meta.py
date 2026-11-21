"""M2: bounded, evolved neural-unit renewal policy.

An external evolutionary search trains a tiny controller which chooses how many
units to renew at task boundaries. The controller has no code-writing ability,
no access to held-out test data, and no access to source files or secrets.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

from .plasticity import (
    Network, accuracy, curve_auc, dead_fraction, examples, make_task,
    paired_bootstrap,
)

ARMS = ("learned", "continual", "fixed_low_activity", "random_matched",
        "shuffled_matched", "fresh")


@dataclass(frozen=True)
class MetaSettings:
    train_start: int = 1000
    train_seeds: int = 4
    validation_start: int = 2000
    validation_seeds: int = 3
    test_start: int = 3000
    test_seeds: int = 16
    tasks: int = 8
    steps: int = 64
    batch_size: int = 2
    width: int = 16
    eval_size: int = 48
    report_every: int = 16
    learning_rate: float = 0.04
    max_reset_fraction: float = 0.20
    generations: int = 5
    population: int = 8
    reset_cost: float = 0.025
    retention_cost: float = 0.05
    search_seed: int = 911

    def validate(self) -> None:
        values = (self.train_seeds, self.validation_seeds, self.test_seeds,
                  self.tasks, self.steps, self.batch_size, self.width,
                  self.eval_size, self.report_every, self.generations,
                  self.population)
        if min(values) < 1 or self.tasks < 2 or self.width < 3:
            raise ValueError("Counts must be positive; tasks>=2 and width>=3")
        if self.population < 4 or self.steps % self.report_every:
            raise ValueError("population>=4 and steps divisible by report_every")
        if not (0 < self.max_reset_fraction <= 0.5):
            raise ValueError("max_reset_fraction must be in (0, 0.5]")
        if not (0 < self.learning_rate <= 0.5):
            raise ValueError("learning_rate must be in (0, 0.5]")
        if min(self.reset_cost, self.retention_cost, self.search_seed) < 0:
            raise ValueError("Costs and search seed must be nonnegative")
        regions = (
            set(range(self.train_start, self.train_start + self.train_seeds)),
            set(range(self.validation_start, self.validation_start + self.validation_seeds)),
            set(range(self.test_start, self.test_start + self.test_seeds)),
        )
        if any(a & b for i, a in enumerate(regions) for b in regions[i+1:]):
            raise ValueError("Train, validation and test seed sets must be disjoint")

    @property
    def max_units(self) -> int:
        return max(1, min(self.width - 1,
                          round(self.width * self.max_reset_fraction)))


@dataclass(frozen=True)
class Policy:
    weights: tuple[float, float, float, float, float]

    def action(self, features: tuple[float, ...], max_units: int) -> int:
        """Discrete 0/half/max action; bounded by the policy-independent limit."""
        if len(features) != len(self.weights):
            raise ValueError("Unexpected observation size")
        score = sum(w * f for w, f in zip(self.weights, features))
        if score < -0.5:
            return 0
        if score < 0.5:
            return max(1, round(max_units / 2))
        return max_units


def prepare(seed: int, settings: MetaSettings) -> list[dict]:
    result = []
    for task_index in range(settings.tasks):
        task = make_task(seed, task_index)
        train = examples(task, seed * 2_000_003 + task_index * 31 + 111,
                         settings.steps * settings.batch_size)
        validation = examples(task, seed * 2_000_003 +
                              task_index * 31 + 999_777, settings.eval_size)
        result.append({"train": train, "eval": validation, "kind": task.kind})
    return result


def features(net: Network, current_training: list, previous_training: list,
             index: int, settings: MetaSettings) -> tuple[float, ...]:
    """Only labelled *training* examples and existing weights are visible."""
    new_probe = current_training[:min(32, len(current_training))]
    previous_probe = previous_training[-min(64, len(previous_training)):]
    new_error = 1 - accuracy(net, new_probe)
    old_error = 1 - accuracy(net, previous_probe)
    activity = [0.0] * len(net.w_out)
    for x, _ in previous_probe:
        for unit, value in enumerate(net.hidden_values(x)):
            activity[unit] += value / len(previous_probe)
    almost_inactive = sum(v < 0.05 for v in activity) / len(activity)
    return (1.0, 2 * new_error - 1, 2 * old_error - 1,
            2 * almost_inactive - 1, index / settings.tasks)


def simulate(seed: int, settings: MetaSettings, tasks: list[dict], arm: str,
             policy: Policy | None = None, plan: list[int] | None = None) -> dict:
    if arm not in ARMS:
        raise ValueError(f"Unknown arm: {arm}")
    net = Network.create(random.Random(seed * 13_019 + 31), settings.width)
    past_validation = []
    past_accuracy = []
    rows = []
    actions = []
    for index, item in enumerate(tasks):
        resets = 0
        if index:
            if arm == "fresh":
                net = Network.create(random.Random(seed * 13_019 + index * 997 + 31),
                                     settings.width)
            elif arm == "fixed_low_activity":
                resets = settings.max_units
            elif arm in ("random_matched", "shuffled_matched"):
                if plan is None or len(plan) != settings.tasks - 1:
                    raise ValueError("Matched controls require a full action plan")
                resets = plan[index - 1]
            elif arm == "learned":
                if policy is None:
                    raise ValueError("Learned arm requires a policy")
                observation = features(net, item["train"], tasks[index-1]["train"],
                                       index, settings)
                resets = policy.action(observation, settings.max_units)
            if resets:
                if not 0 < resets <= settings.max_units:
                    raise ValueError("Unbounded action")
                probe = [xy for xy, _ in tasks[index-1]["train"][-64:]]
                net.renew(resets, random.Random(seed * 349_019 + index * 701 + 23),
                          probe, low_activity=(arm != "random_matched"))
            actions.append(resets)
        start = accuracy(net, item["eval"])
        curve = [(0, start)]
        for step in range(1, settings.steps + 1):
            i = (step - 1) * settings.batch_size
            net.train_batch(item["train"][i:i + settings.batch_size],
                            settings.learning_rate)
            if step % settings.report_every == 0:
                curve.append((step, accuracy(net, item["eval"])))
        end = curve[-1][1]
        forgetting = (statistics.mean(
            max(0.0, before - accuracy(net, old_data))
            for before, old_data in zip(past_accuracy, past_validation)
        ) if past_validation else 0.0)
        rows.append({
            "seed": seed, "task": index, "kind": item["kind"], "arm": arm,
            "start_accuracy": start, "end_accuracy": end,
            "auc": curve_auc(curve, settings.steps),
            "forgetting": forgetting,
            "dead_fraction": dead_fraction(net, item["eval"]),
            "resets": resets,
        })
        past_accuracy.append(end)
        past_validation.append(item["eval"])
    after_first = rows[1:]
    total = sum(actions)
    return {
        "seed": seed, "arm": arm, "rows": rows, "actions": actions,
        "auc": statistics.mean(r["auc"] for r in after_first),
        "end_accuracy": statistics.mean(r["end_accuracy"] for r in after_first),
        "forgetting": statistics.mean(r["forgetting"] for r in after_first),
        "total_resets": total,
        "normalized_resets": total / (settings.max_units * (settings.tasks - 1)),
    }


def fitness(result: dict, settings: MetaSettings) -> float:
    return (result["auc"] -
            settings.reset_cost * result["normalized_resets"] -
            settings.retention_cost * result["forgetting"])


def evaluate_policy(policy: Policy, seeds: list[int], settings: MetaSettings,
                    cache: dict[int, list[dict]]) -> float:
    return statistics.mean(
        fitness(simulate(seed, settings, cache[seed], "learned", policy), settings)
        for seed in seeds
    )


def search(settings: MetaSettings) -> tuple[Policy, list[dict]]:
    """Evolution on meta-train; candidate selection on separate meta-validation."""
    rng = random.Random(settings.search_seed)
    train = list(range(settings.train_start, settings.train_start + settings.train_seeds))
    valid = list(range(settings.validation_start,
                       settings.validation_start + settings.validation_seeds))
    cache = {seed: prepare(seed, settings) for seed in train + valid}
    initial = [
        Policy((-3.0, 0.0, 0.0, 0.0, 0.0)),  # never renew
        Policy((0.0, 0.0, 0.0, 0.0, 0.0)),   # half-size renewal every switch
        Policy((3.0, 0.0, 0.0, 0.0, 0.0)),   # maximum renewal every switch
        Policy((-1.0, 2.0, 0.0, 0.0, 0.0)),
    ]
    population = initial + [
        Policy(tuple(rng.uniform(-1.5, 1.5) for _ in range(5)))
        for _ in range(settings.population - len(initial))
    ]
    candidates = []
    history = []
    for generation in range(settings.generations):
        scored = [(evaluate_policy(p, train, settings, cache), p) for p in population]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        winner = scored[0][1]
        validation_score = evaluate_policy(winner, valid, settings, cache)
        candidates.append((validation_score, winner))
        history.append({
            "generation": generation,
            "best_training_fitness": scored[0][0],
            "validation_fitness": validation_score,
            "winner_weights": list(winner.weights),
        })
        print(f"meta_generation={generation} train={scored[0][0]:.5f} "
              f"validation={validation_score:.5f}", flush=True)
        elite = [p for _, p in scored[:max(2, settings.population // 4)]]
        population = list(elite)
        while len(population) < settings.population:
            source = rng.choice(elite)
            evolved = tuple(
                max(-5.0, min(5.0, w + rng.gauss(0, 0.55)))
                for w in source.weights
            )
            population.append(Policy(evolved))
    return max(candidates, key=lambda x: x[0])[1], history


def summarize(results: list[dict], settings: MetaSettings, policy: Policy,
              history: list[dict]) -> dict:
    grouped = {
        arm: sorted((r for r in results if r["arm"] == arm),
                    key=lambda r: r["seed"]) for arm in ARMS
    }
    by_seed = {arm: {r["seed"]: r for r in rows}
               for arm, rows in grouped.items()}
    metrics = ("auc", "end_accuracy", "forgetting", "normalized_resets", "total_resets")
    arm_means = {
        arm: {metric: statistics.mean(r[metric] for r in rows) for metric in metrics}
        for arm, rows in grouped.items()
    }
    comparisons = {}
    for reference in ("continual", "fixed_low_activity", "random_matched",
                      "shuffled_matched", "fresh"):
        differences = [
            by_seed["learned"][seed]["auc"] - by_seed[reference][seed]["auc"]
            for seed in sorted(by_seed["learned"])
        ]
        comparisons[f"learned_minus_{reference}_auc"] = {
            "mean": statistics.mean(differences),
            "bootstrap_95_percentile_interval": list(
                paired_bootstrap(differences, settings.test_start + 17)
            ),
            "positive_seeds": sum(d > 0 for d in differences),
            "zero_seeds": sum(d == 0 for d in differences),
            "negative_seeds": sum(d < 0 for d in differences),
        }
    actions = [a for result in grouped["learned"] for a in result["actions"]]
    return {
        "protocol": "M2 bounded renewal, seed-isolated meta-train/selection/test",
        "settings": asdict(settings),
        "policy_weights": list(policy.weights),
        "meta_training": history,
        "test_arm_means": arm_means,
        "test_paired_auc_comparisons": comparisons,
        "learned_action_histogram": {
            str(v): sum(a == v for a in actions) for v in sorted(set(actions))
        },
        "primary_metric": "paired heldout learning-curve AUC (learned - shuffled matched)",
        "limitations": [
            "Controller is evolved externally and only chooses bounded reset counts",
            "Validation seeds choose meta-policy; independent test seeds remain sealed",
            "Task family is the same as M1-P; no out-of-distribution generalization established",
            "Matched controls have equal reset counts; not identical inference/selection CPU cost",
            "Selection and per-task renewal diagnostics only inspect training examples",
            "Meta-training and meta-validation curves are not independent final evidence",
        ],
    }


def graph(results: list[dict], settings: MetaSettings) -> str:
    chosen = ("continual", "fixed_low_activity", "learned",
              "shuffled_matched", "fresh")
    colors = ("#666", "#bf6727", "#167d5a", "#884bb3", "#2767a4")
    lines = []
    for arm, color in zip(chosen, colors):
        points = []
        for index in range(settings.tasks):
            values = [row["auc"] for r in results if r["arm"] == arm
                      for row in r["rows"] if row["task"] == index]
            x = 48 + 620 * index / (settings.tasks - 1)
            y = 294 - 245 * statistics.mean(values)
            points.append(f"{x:.2f},{y:.2f}")
        lines.append(f'<polyline fill="none" stroke="{color}" '
                     f'stroke-width="2.5" points="{" ".join(points)}"/>')
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 740 410">'
        '<rect width="740" height="410" fill="white"/>'
        '<text x="48" y="25" font-size="17">M2 held-out learning AUC by task</text>'
        '<path d="M48 49 V294 H670" fill="none" stroke="#777"/>'
        '<text x="10" y="54" font-size="12">1.0</text>'
        '<text x="10" y="296" font-size="12">0.0</text>'
        + "".join(lines)
        + "".join(
            f'<text x="{50+(i%2)*340}" y="{326+(i//2)*23}" '
            f'fill="{colors[i]}" font-size="12">{name}</text>'
            for i, name in enumerate(chosen)
        ) + "</svg>\n"
    )


def run(settings: MetaSettings, output: Path) -> dict:
    settings.validate()
    output.mkdir(parents=True, exist_ok=True)
    policy, history = search(settings)
    print("selected policy:", policy.weights, flush=True)
    results = []
    for seed in range(settings.test_start, settings.test_start + settings.test_seeds):
        dataset = prepare(seed, settings)
        learned = simulate(seed, settings, dataset, "learned", policy)
        plan = learned["actions"]
        shuffled = list(plan)
        random.Random(seed * 417 + 17).shuffle(shuffled)
        arms = {
            "learned": learned,
            "continual": simulate(seed, settings, dataset, "continual"),
            "fixed_low_activity": simulate(seed, settings, dataset,
                                           "fixed_low_activity"),
            "random_matched": simulate(seed, settings, dataset,
                                       "random_matched", plan=plan),
            "shuffled_matched": simulate(seed, settings, dataset,
                                         "shuffled_matched", plan=shuffled),
            "fresh": simulate(seed, settings, dataset, "fresh"),
        }
        assert (arms["random_matched"]["total_resets"] ==
                arms["shuffled_matched"]["total_resets"] ==
                learned["total_resets"])
        results.extend(arms[name] for name in ARMS)
        print(f"sealed_test_seed={seed} "
              f"learned_auc={learned['auc']:.4f}", flush=True)
    summary = summarize(results, settings, policy, history)

    fields = ("seed", "task", "kind", "arm", "start_accuracy",
              "end_accuracy", "auc", "forgetting", "dead_fraction", "resets")
    with (output / "task_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerows(result["rows"])
    with (output / "test_seed_metrics.csv").open("w", newline="",
                                                 encoding="utf-8") as file:
        fieldnames = ("seed", "arm", "auc", "end_accuracy", "forgetting",
                      "total_resets", "normalized_resets")
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({k: result[k] for k in fieldnames})
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "meta_training.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in history),
        encoding="utf-8"
    )
    (output / "auc.svg").write_text(graph(results, settings), encoding="utf-8")
    print(json.dumps(summary["test_paired_auc_comparisons"], sort_keys=True),
          flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name, field in MetaSettings.__dataclass_fields__.items():
        default = getattr(MetaSettings(), name)
        parser.add_argument("--" + name.replace("_", "-"),
                            type=type(default), default=default)
    parser.add_argument("--out", type=Path, default=Path("results/m2"))
    args = parser.parse_args()
    settings = MetaSettings(**{
        name: getattr(args, name) for name in MetaSettings.__dataclass_fields__
    })
    run(settings, args.out)


if __name__ == "__main__":
    main()
