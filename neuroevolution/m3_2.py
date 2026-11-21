"""M3.2: natural (non-damaged) lifelong SGD interference and counterfactual resets.

Agents retain one network across a stochastic sequence of supervised tasks.
A fixed budget of neuron renewals is allocated by externally evolved, bounded
policies. Test-time one-step counterfactuals are analysis only, never training.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from .m3 import WorldTask
from .plasticity import Network, accuracy, curve_auc, examples, paired_bootstrap


ARMS = ("adaptive", "accuracy_only", "time_only", "uniform", "shuffled",
        "random_units", "no_reset", "always_reset")
MATCHED = ARMS[:6]
MODES = ("adaptive", "accuracy_only", "time_only")


@dataclass(frozen=True)
class Settings:
    train_start: int = 16000
    train_seeds: int = 6
    validation_start: int = 17000
    validation_seeds: int = 3
    test_start: int = 18000
    test_seeds: int = 20
    shifted_start: int = 19000
    shifted_seeds: int = 12
    tasks: int = 12
    steps: int = 48
    batch_size: int = 2
    width: int = 16
    eval_size: int = 64
    report_every: int = 12
    learning_rate: float = 0.04
    renew_units: int = 3
    quota: int = 4
    generations: int = 4
    population: int = 8
    search_seed: int = 931
    forgetting_cost: float = 0.08

    def validate(self) -> None:
        sizes = (self.train_seeds, self.validation_seeds, self.test_seeds,
                 self.shifted_seeds, self.tasks, self.steps, self.batch_size,
                 self.width, self.eval_size, self.report_every,
                 self.renew_units, self.quota, self.generations, self.population)
        if min(sizes) < 1 or self.population < 4 or self.tasks < 4:
            raise ValueError("Positive counts, population>=4, tasks>=4 required")
        if self.tasks - 1 <= self.quota or self.width <= self.renew_units:
            raise ValueError("quota must allow choices and renewal must be bounded")
        if self.steps % self.report_every:
            raise ValueError("steps must be divisible by report_every")
        if not 0 < self.learning_rate <= 0.5 or not 0 <= self.forgetting_cost <= 1:
            raise ValueError("Invalid learning rate or forgetting cost")
        groups = (
            (self.train_start, self.train_seeds),
            (self.validation_start, self.validation_seeds),
            (self.test_start, self.test_seeds),
            (self.shifted_start, self.shifted_seeds),
        )
        for i, (start, size) in enumerate(groups):
            if start < 0:
                raise ValueError("Seed starts must be nonnegative")
            a = set(range(start, start + size))
            for other_start, other_size in groups[i + 1:]:
                if a.intersection(range(other_start, other_start + other_size)):
                    raise ValueError("Meta train/validation/final seed overlap")


@dataclass(frozen=True)
class Policy:
    weights: tuple[float, float, float, float, float, float]

    def decide(self, obs: tuple[float, ...], mode: str) -> bool:
        if len(obs) != 6 or mode not in MODES:
            raise ValueError("Unexpected policy mode/observation")
        active = (
            (0, 1, 2, 3, 4, 5) if mode == "adaptive" else
            (0, 1, 5) if mode == "accuracy_only" else (0, 5)
        )
        return sum(self.weights[i] * obs[i] for i in active) > 0.0


def make_tasks(seed: int, settings: Settings, shifted: bool) -> list[dict]:
    rng = random.Random(seed * 37_013 + 47)
    kinds = ("line", "checker", "wedge", "stripe") if shifted else (
        "line", "xor", "circle", "stripe")
    bank: dict[int, WorldTask] = {}
    sequence = []
    for task_index in range(settings.tasks):
        returning = task_index >= 2 and rng.random() < 0.45
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
        base = seed * 2_000_003 + task_index * 31
        train = examples(task, base + 111, settings.steps * settings.batch_size)
        evaluation = examples(task, base + 999_777, settings.eval_size)
        sequence.append({
            "train": train, "eval": evaluation,
            "task_id": task_id, "status": status, "kind": task.kind,
        })
    return sequence


def output_gradient(net: Network, data: list) -> list[float]:
    """Cross-entropy gradient of OUTPUT weights on labelled TRAINING points."""
    grad = [0.0] * (len(net.w_out) + 1)
    for x, y in data:
        h = net.hidden_values(x)
        error = net.predict(x) - y
        for i, value in enumerate(h):
            grad[i] += error * value / len(data)
        grad[-1] += error / len(data)
    return grad


def observe(net: Network, new_training: list, past_training: list,
            task_index: int, settings: Settings) -> tuple[float, ...]:
    """Every observation is training-only: no eval labels or task identity."""
    new_probe = new_training[:min(32, len(new_training))]
    past_probe = past_training[-min(32, len(past_training)):]
    a = output_gradient(net, new_probe)
    b = output_gradient(net, past_probe)
    dot = sum(v * w for v, w in zip(a, b))
    norm = math.sqrt(sum(v * v for v in a) * sum(v * v for v in b))
    cosine = dot / norm if norm > 1e-12 else 0.0
    activity = [0.0] * len(net.w_out)
    for x, _ in past_probe:
        for i, value in enumerate(net.hidden_values(x)):
            activity[i] += value / len(past_probe)
    inactive = sum(v < 0.025 for v in activity) / len(activity)
    return (
        1.0,
        accuracy(net, new_probe),
        accuracy(net, past_probe),
        max(-1.0, min(1.0, cosine)),
        inactive,
        task_index / (settings.tasks - 1),
    )


def evenly_spaced(settings: Settings) -> list[bool]:
    n = settings.tasks - 1
    return [((i + 1) * settings.quota // n >
             i * settings.quota // n) for i in range(n)]


def learn_task(net: Network, item: dict, settings: Settings,
               memory: list[tuple[list, float]]) -> tuple[Network, dict]:
    curve = [(0, accuracy(net, item["eval"]))]
    for step in range(1, settings.steps + 1):
        index = (step - 1) * settings.batch_size
        net.train_batch(
            item["train"][index:index + settings.batch_size],
            settings.learning_rate,
        )
        if step % settings.report_every == 0:
            curve.append((step, accuracy(net, item["eval"])))
    forgetting = (
        statistics.mean(max(0.0, before - accuracy(net, old))
                        for old, before in memory)
        if memory else 0.0
    )
    return net, {
        "auc": curve_auc(curve, settings.steps),
        "start_accuracy": curve[0][1],
        "end_accuracy": curve[-1][1],
        "forgetting": forgetting,
    }


def intervention(net: Network, item: dict, previous_train: list,
                 settings: Settings, seed: int, index: int,
                 renew: bool, random_units: bool) -> tuple[Network, list[int]]:
    # Use deepcopy to guarantee identical pre-action potential states.
    import copy
    copy_net = copy.deepcopy(net)
    if not renew:
        return copy_net, []
    probe = [x for x, _ in previous_train[-64:]]
    selected = copy_net.renew(
        settings.renew_units,
        random.Random(seed * 349_019 + index * 701 + 23),
        probe,
        low_activity=not random_units,
    )
    return copy_net, selected


def simulate(seed: int, settings: Settings, data: list[dict], arm: str,
             policy: Policy | None = None, plan: list[bool] | None = None,
             counterfactual: bool = False) -> dict:
    if arm not in ARMS:
        raise ValueError(arm)
    if arm in MODES and policy is None:
        raise ValueError("Learned arms require a trained policy")
    if arm in ("uniform", "shuffled", "random_units"):
        if plan is None or len(plan) != settings.tasks - 1 or sum(plan) != settings.quota:
            raise ValueError("Matched plan must contain exactly the fixed quota")
    if counterfactual and arm != "adaptive":
        raise ValueError("Counterfactual instrumentation is for the adaptive arm")

    net = Network.create(random.Random(seed * 13_019 + 31), settings.width)
    memory = []
    rows = []
    actions, requested = [], []
    counterfactual_rows = []
    remaining = settings.quota
    forced = 0
    sensitivity = {"accuracy_blind": 0, "time_blind": 0, "unforced_state_sensitive": 0}
    for index, item in enumerate(data):
        chosen = False
        wish = False
        is_forced = False
        if index:
            if arm in MODES:
                obs = observe(net, item["train"], data[index - 1]["train"],
                              index, settings)
                wish = policy.decide(obs, arm)
                steps_left = settings.tasks - index
                lower = remaining >= steps_left  # last chances require renewing
                upper = remaining > 0
                chosen = bool(upper and (wish or lower))
                is_forced = chosen != wish
                forced += int(is_forced)
                if chosen:
                    remaining -= 1
                requested.append(int(wish))
                if arm == "adaptive":
                    masked_accuracy = (obs[0], obs[1], 0., 0., 0., obs[5])
                    masked_clock = (obs[0], 0., 0., 0., 0., obs[5])
                    accuracy_sensitive = wish != policy.decide(
                        masked_accuracy, "adaptive")
                    clock_sensitive = wish != policy.decide(
                        masked_clock, "adaptive")
                    sensitivity["accuracy_blind"] += int(accuracy_sensitive)
                    sensitivity["time_blind"] += int(clock_sensitive)
                    sensitivity["unforced_state_sensitive"] += int(
                        clock_sensitive and not is_forced
                    )
            elif arm == "no_reset":
                chosen = False
            elif arm == "always_reset":
                chosen = True
            else:
                chosen = plan[index - 1]
            actions.append(int(chosen))

        before = net
        if counterfactual and index > 0:
            options = {}
            for alt in (False, True):
                alt_net, ids = intervention(
                    before, item, data[index - 1]["train"], settings,
                    seed, index, alt, random_units=False,
                )
                after, metric = learn_task(alt_net, item, settings, memory)
                options[alt] = (after, metric, ids)
            target_net, metric, selected = options[chosen]
            preserve = options[False][1]
            renew_score = options[True][1]
            value_false = preserve["auc"] - settings.forgetting_cost * preserve["forgetting"]
            value_true = renew_score["auc"] - settings.forgetting_cost * renew_score["forgetting"]
            oracle = value_true > value_false
            counterfactual_rows.append({
                "seed": seed, "index": index, "kind": item["kind"],
                "status": item["status"],
                "requested": int(wish), "chosen": int(chosen),
                "forced": int(is_forced),
                "renew_auc_delta": renew_score["auc"] - preserve["auc"],
                "renew_forgetting_delta": renew_score["forgetting"] - preserve["forgetting"],
                "renew_value_delta": value_true - value_false,
                "oracle_renew": int(oracle),
                "correct_local_action": int(oracle == chosen),
                "selected_units": len(selected),
            })
            net = target_net
        else:
            candidate, selected = intervention(
                before, item,
                data[index - 1]["train"] if index else item["train"],
                settings, seed, index, chosen,
                random_units=(arm == "random_units"),
            )
            net, metric = learn_task(candidate, item, settings, memory)
        rows.append({
            "seed": seed, "task_index": index, "kind": item["kind"],
            "status": item["status"], "task_id": item["task_id"],
            "arm": arm, "renewal": int(chosen), "renewed_units": len(selected),
            **metric,
        })
        memory.append((item["eval"], metric["end_accuracy"]))
    if arm in MODES and remaining != 0:
        raise AssertionError("Learned policy did not use all renewal quota")
    if arm in MATCHED and sum(actions) != settings.quota:
        raise AssertionError("Mismatch in intervention budgets")
    selected_rows = rows[1:]
    return {
        "seed": seed, "arm": arm,
        "auc": statistics.mean(x["auc"] for x in selected_rows),
        "forgetting": statistics.mean(x["forgetting"] for x in selected_rows),
        "end_accuracy": statistics.mean(x["end_accuracy"] for x in selected_rows),
        "renewal_actions": sum(actions),
        "renewed_units": sum(r["renewed_units"] for r in rows),
        "forced": forced,
        "actions": actions, "requested": requested,
        **sensitivity,
        "rows": rows, "counterfactuals": counterfactual_rows,
    }


def project(weights: tuple[float, ...], mode: str) -> Policy:
    if mode == "time_only":
        return Policy((weights[0], 0., 0., 0., 0., weights[5]))
    if mode == "accuracy_only":
        return Policy((weights[0], weights[1], 0., 0., 0., weights[5]))
    if mode == "adaptive":
        return Policy(tuple(weights))
    raise ValueError(mode)


def search(settings: Settings, mode: str, cache: dict[int, list[dict]]) -> tuple[Policy, list[dict]]:
    train_seeds = range(settings.train_start, settings.train_start + settings.train_seeds)
    val_seeds = range(settings.validation_start,
                      settings.validation_start + settings.validation_seeds)
    rng = random.Random(settings.search_seed + MODES.index(mode) * 71)
    population = [
        project((-3., 0., 0., 0., 0., 0.), mode),
        project((3., 0., 0., 0., 0., 0.), mode),
        project((-0.3, 1.4, 0., -0.9, .8, -.4), mode),
        project((.2, -1.5, .6, .9, -.5, .3), mode),
    ]
    while len(population) < settings.population:
        population.append(project(tuple(rng.uniform(-1.5, 1.5)
                                        for _ in range(6)), mode))

    def evaluate(policy: Policy, seeds) -> float:
        outcome = [
            simulate(seed, settings, cache[seed], mode, policy)
            for seed in seeds
        ]
        return statistics.mean(
            r["auc"] - settings.forgetting_cost * r["forgetting"]
            for r in outcome
        )

    history = []
    candidates = []
    for gen in range(settings.generations):
        ranked = sorted(
            ((evaluate(p, train_seeds), p) for p in population),
            key=lambda pair: pair[0], reverse=True,
        )
        champ = ranked[0][1]
        val_score = evaluate(champ, val_seeds)
        candidates.append((val_score, champ))
        history.append({
            "generation": gen, "mode": mode,
            "train_objective": ranked[0][0],
            "validation_objective": val_score,
            "coefficients": list(champ.weights),
        })
        print(f"M3.2 meta {mode} gen={gen} train={ranked[0][0]:.5f}"
              f" validation={val_score:.5f}", flush=True)
        elite = [p for _, p in ranked[:max(2, settings.population // 4)]]
        population = list(elite)
        while len(population) < settings.population:
            parent = rng.choice(elite)
            mutated = tuple(max(-5., min(5., w + rng.gauss(0., 0.55)))
                            for w in parent.weights)
            population.append(project(mutated, mode))
    # Top valid wins; no access to test seeds.
    return max(candidates, key=lambda item: item[0])[1], history


def summarize_split(results: list[dict], settings: Settings, label: str) -> dict:
    lookup = {
        arm: {r["seed"]: r for r in results if r["arm"] == arm}
        for arm in ARMS
    }
    means = {
        arm: {
            metric: statistics.mean(r[metric] for r in seed_results.values())
            for metric in ("auc", "forgetting", "end_accuracy", "renewal_actions",
                           "renewed_units")
        }
        for arm, seed_results in lookup.items()
    }
    comparisons = {}
    for other in ARMS[1:]:
        diff = [
            lookup["adaptive"][seed]["auc"] - lookup[other][seed]["auc"]
            for seed in sorted(lookup["adaptive"])
        ]
        comparisons["adaptive_minus_" + other] = {
            "mean_auc_diff": statistics.mean(diff),
            "paired_bootstrap_95": list(paired_bootstrap(diff, min(lookup["adaptive"]) + 35)),
            "positive_seeds": sum(x > 0 for x in diff),
            "negative_seeds": sum(x < 0 for x in diff),
            "zero_seeds": sum(x == 0 for x in diff),
        }
    adaptive = list(lookup["adaptive"].values())
    rows = [r for item in adaptive for r in item["rows"][1:]]
    counterfactuals = [cf for item in adaptive for cf in item["counterfactuals"]]
    return {
        "label": label,
        "independent_seed_count": len(adaptive),
        "mean_metrics": means,
        "comparisons": comparisons,
        "adaptive_requested_total": sum(len(item["requested"]) for item in adaptive),
        "adaptive_renewal_actions": sum(item["renewal_actions"] for item in adaptive),
        "adaptive_forced_actions": sum(item["forced"] for item in adaptive),
        "adaptive_unforced_state_sensitive": sum(item["unforced_state_sensitive"]
                                                  for item in adaptive),
        "mean_auc_novel": statistics.mean(r["auc"] for r in rows if r["status"] == "novel"),
        "mean_auc_return": statistics.mean(r["auc"] for r in rows if r["status"] == "return"),
        "local_counterfactual": {
            "observations": len(counterfactuals),
            "mean_renew_effect_auc": statistics.mean(
                x["renew_auc_delta"] for x in counterfactuals),
            "mean_renew_effect_forgetting": statistics.mean(
                x["renew_forgetting_delta"] for x in counterfactuals),
            "oracle_agreement": statistics.mean(
                x["correct_local_action"] for x in counterfactuals),
            "positive_renew_effect_count": sum(
                x["renew_value_delta"] > 0 for x in counterfactuals),
            "negative_renew_effect_count": sum(
                x["renew_value_delta"] < 0 for x in counterfactuals),
        },
    }


def run(settings: Settings, output: Path) -> dict:
    settings.validate()
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    cache = {seed: make_tasks(seed, settings, shifted=False) for seed in
             list(range(settings.train_start, settings.train_start + settings.train_seeds)) +
             list(range(settings.validation_start,
                        settings.validation_start + settings.validation_seeds))}
    policies = {}
    history = []
    for mode in MODES:
        policies[mode], trace = search(settings, mode, cache)
        history.extend(trace)
        print(f"SELECTED {mode}: {policies[mode].weights}", flush=True)

    results = []
    for label, start, count, shifted in (
        ("main", settings.test_start, settings.test_seeds, False),
        ("shifted", settings.shifted_start, settings.shifted_seeds, True),
    ):
        split_results = []
        for seed in range(start, start + count):
            data = make_tasks(seed, settings, shifted)
            adaptive = simulate(seed, settings, data, "adaptive",
                                policies["adaptive"], counterfactual=True)
            shuffled = list(adaptive["actions"])
            random.Random(seed * 417 + 17).shuffle(shuffled)
            by_arm = {
                "adaptive": adaptive,
                "accuracy_only": simulate(seed, settings, data, "accuracy_only",
                                          policies["accuracy_only"]),
                "time_only": simulate(seed, settings, data, "time_only",
                                      policies["time_only"]),
                "uniform": simulate(seed, settings, data, "uniform",
                                    plan=evenly_spaced(settings)),
                "shuffled": simulate(seed, settings, data, "shuffled", plan=shuffled),
                "random_units": simulate(seed, settings, data, "random_units",
                                         plan=adaptive["actions"]),
                "no_reset": simulate(seed, settings, data, "no_reset"),
                "always_reset": simulate(seed, settings, data, "always_reset"),
            }
            if any(by_arm[a]["renewal_actions"] != settings.quota for a in MATCHED):
                raise AssertionError("Budget mismatch in dose-matched arms")
            split_results.extend(by_arm[a] for a in ARMS)
            print(f"FINAL {label} seed={seed} adaptive_auc={adaptive['auc']:.4f}"
                  f" clock={by_arm['time_only']['auc']:.4f}", flush=True)
        results.append((label, split_results))
    summary = {
        "settings": asdict(settings),
        "protocol": "M3.2 preregistered natural-interference lifelong SGD",
        "selected_policy_weights": {k: list(v.weights) for k, v in policies.items()},
        "meta_history": history,
        "main": summarize_split(results[0][1], settings, "main"),
        "shifted": summarize_split(results[1][1], settings, "shifted"),
        "primary": "main paired AUC adaptive - independently evolved time_only",
        "necessary_secondary": "adaptive - independently evolved accuracy_only",
        "wall_seconds_including_instrumentation": time.perf_counter() - started,
        "limitations": [
            "Task rules may conflict on the same 2D inputs without context identifiers",
            "Local counterfactual effects are not entire alternative lifetime trajectories",
            "Agents are externally evolved numerical threshold policies, not autonomous goals",
            "Matched reset and SGD budgets do not guarantee equal wall time per policy",
            "Task-class shift reuses two of the four families; not general intelligence",
            "Evaluation data appears only in meta-search fitness for meta train/validation seeds",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    with (output / "seed_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        cols = ("split", "seed", "arm", "auc", "forgetting",
                "end_accuracy", "renewal_actions", "renewed_units",
                "forced", "unforced_state_sensitive", "actions", "requested")
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for label, rows in results:
            for row in rows:
                writer.writerow({
                    k: json.dumps(row[k]) if k in ("actions", "requested") else
                    label if k == "split" else row[k]
                    for k in cols
                })
    with (output / "task_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        cols = ("split", "seed", "arm", "task_index", "kind", "task_id",
                "status", "auc", "start_accuracy", "end_accuracy",
                "forgetting", "renewal", "renewed_units")
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for label, rows in results:
            for row in rows:
                writer.writerows({**record, "split": label} for record in row["rows"])
    with (output / "local_counterfactuals.csv").open("w", newline="",
                                                    encoding="utf-8") as f:
        cols = ("split", "seed", "index", "kind", "status", "requested",
                "chosen", "forced", "renew_auc_delta", "renew_forgetting_delta",
                "renew_value_delta", "oracle_renew", "correct_local_action",
                "selected_units")
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for label, rows in results:
            for row in rows:
                if row["arm"] == "adaptive":
                    writer.writerows({**r, "split": label}
                                     for r in row["counterfactuals"])
    print("PRIMARY main:", summary["main"]["comparisons"]["adaptive_minus_time_only"],
          flush=True)
    print("REQUIRED secondary:",
          summary["main"]["comparisons"]["adaptive_minus_accuracy_only"], flush=True)
    print("SHIFTED:", summary["shifted"]["comparisons"]["adaptive_minus_time_only"],
          flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = Settings()
    for field in fields(Settings):
        default = getattr(defaults, field.name)
        parser.add_argument("--" + field.name.replace("_", "-"),
                            type=type(default), default=default)
    parser.add_argument("--out", type=Path, default=Path("results/m3_2"))
    args = parser.parse_args()
    settings = Settings(**{field.name: getattr(args, field.name)
                           for field in fields(Settings)})
    run(settings, args.out)


if __name__ == "__main__":
    main()
