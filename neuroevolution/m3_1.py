"""M3.1: causal positive-control benchmark for preserving vs renewing neurons.

The environment deliberately injects a controlled representation defect in
half of paired cases. Both actions are evaluated from identical starting
states with identical data. A controller sees only labelled training probes.
No arbitrary source modification or network access is possible.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import statistics
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from .plasticity import Network, Task, accuracy, curve_auc, examples, paired_bootstrap


@dataclass(frozen=True)
class Settings:
    train_start: int = 12000
    train_seeds: int = 8
    validation_start: int = 13000
    validation_seeds: int = 4
    test_start: int = 14000
    test_seeds: int = 24
    shifted_start: int = 15000
    shifted_seeds: int = 16
    trials: int = 4
    width: int = 12
    reset_units: int = 3
    pretrain_steps: int = 96
    adaptation_steps: int = 48
    batch_size: int = 4
    probe_size: int = 48
    eval_size: int = 96
    report_every: int = 12
    learning_rate: float = 0.05
    shock_multiplier: float = 5.0

    def validate(self) -> None:
        ranges = (
            (self.train_start, self.train_seeds),
            (self.validation_start, self.validation_seeds),
            (self.test_start, self.test_seeds),
            (self.shifted_start, self.shifted_seeds),
        )
        if any(start < 0 or count < 1 for start, count in ranges):
            raise ValueError("All seed partitions must be nonempty and nonnegative")
        for index, (start, count) in enumerate(ranges):
            current = set(range(start, start + count))
            for other_start, other_count in ranges[index + 1:]:
                if current.intersection(range(other_start, other_start + other_count)):
                    raise ValueError("Seed partitions overlap")
        if (self.trials < 2 or self.width < 4 or
                not 0 < self.reset_units < self.width):
            raise ValueError("Need 2+ trials and 0 < reset_units < width")
        if min(self.pretrain_steps, self.adaptation_steps, self.batch_size,
               self.probe_size, self.eval_size, self.report_every) < 1:
            raise ValueError("Training and evaluation dimensions must be positive")
        if self.adaptation_steps % self.report_every:
            raise ValueError("adaptation_steps must be divisible by report_every")
        if not 0 < self.learning_rate <= 0.5 or self.shock_multiplier <= 1:
            raise ValueError("Invalid learning rate or shock strength")


@dataclass(frozen=True)
class Rule:
    """Policy with one degree of freedom. Variant is selected before test."""

    kind: str  # threshold_accuracy / threshold_clock / always_preserve / always_renew
    cutoff: float

    def decide(self, item: dict, opportunities: int) -> bool:
        if self.kind == "always_preserve":
            return False
        if self.kind == "always_renew":
            return True
        if self.kind == "threshold_accuracy":
            return item["probe_accuracy"] < self.cutoff
        if self.kind == "threshold_clock":
            return item["index"] / max(1, opportunities - 1) < self.cutoff
        raise ValueError(f"Unknown rule kind {self.kind}")


def rules(clock: bool) -> list[Rule]:
    candidates = [
        Rule("threshold_clock" if clock else "threshold_accuracy",
             j / 20) for j in range(8, 21)
    ]
    return candidates + [Rule("always_preserve", -1.0),
                         Rule("always_renew", 2.0)]


def make_task(seed: int, trial: int, shifted: bool) -> Task:
    rng = random.Random(seed * 100_003 + trial * 109 + 17)
    kind = ("xor" if trial % 2 == 0 else "circle") if shifted else "line"
    return Task(
        kind=kind, angle=rng.uniform(-3.14, 3.14),
        shift=rng.uniform(-0.15, 0.15),
        phase=rng.uniform(-0.15, 0.15),
    )


def influence(net: Network, probe: list) -> list[float]:
    """Activation contribution per hidden unit, using inputs ONLY."""
    width = len(net.w_out)
    totals = [0.0] * width
    for x, _ in probe:
        for i, value in enumerate(net.hidden_values(x)):
            totals[i] += abs(net.w_out[i]) * value
    return [value / len(probe) for value in totals]


def influential_ids(net: Network, probe: list, count: int) -> list[int]:
    scores = influence(net, probe)
    return sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:count]


def shock(net: Network, probe: list, settings: Settings) -> list[int]:
    """Known exogenous damage for the experiment, NOT visible to policies."""
    ids = influential_ids(net, probe, settings.reset_units)
    for i in ids:
        net.w_out[i] *= -settings.shock_multiplier
    return ids


def renew(net: Network, probe: list, settings: Settings, seed: int) -> list[int]:
    """Fixed action grammar: reinitialize high-contribution units, not oracle ids."""
    rng = random.Random(seed)
    ids = influential_ids(net, probe, settings.reset_units)
    for i in ids:
        net.w_in[i] = [rng.gauss(0, 0.85), rng.gauss(0, 0.85)]
        net.b_hidden[i] = 0.0
        net.w_out[i] = rng.gauss(0, 0.28)
    return ids


def train(net: Network, samples: list, steps: int, batch_size: int,
          learning_rate: float) -> None:
    for index in range(steps):
        start = index * batch_size
        net.train_batch(samples[start:start + batch_size], learning_rate)


def potential(net: Network, probe: list, post_train: list, holdout: list,
              settings: Settings, *, reset: bool, reset_seed: int) -> dict:
    clone = copy.deepcopy(net)
    selected = renew(clone, probe, settings, reset_seed) if reset else []
    points = [(0, accuracy(clone, holdout))]
    for step in range(1, settings.adaptation_steps + 1):
        start = (step - 1) * settings.batch_size
        clone.train_batch(post_train[start:start + settings.batch_size],
                          settings.learning_rate)
        if step % settings.report_every == 0:
            points.append((step, accuracy(clone, holdout)))
    return {
        "auc": curve_auc(points, settings.adaptation_steps),
        "start": points[0][1],
        "end": points[-1][1],
        "selected": selected,
    }


def make_seed_episodes(seed: int, settings: Settings, *,
                       shifted: bool = False) -> list[dict]:
    """Each underlying pretrained net yields intact and damaged contexts.

    Context order is shuffled by a seed-independent-of-task RNG, never
    disclosed to learned policies. No final test episodes enter meta-search.
    """
    rng = random.Random(seed * 73_021 + 89)
    episodes = []
    for trial in range(settings.trials):
        task = make_task(seed, trial, shifted)
        pretrain = examples(task, seed * 97_001 + trial * 13 + 101,
                            settings.pretrain_steps * settings.batch_size)
        probe = examples(task, seed * 97_001 + trial * 13 + 202,
                         settings.probe_size)
        post_train = examples(task, seed * 97_001 + trial * 13 + 303,
                              settings.adaptation_steps * settings.batch_size)
        holdout = examples(task, seed * 97_001 + trial * 13 + 404,
                           settings.eval_size)
        base = Network.create(random.Random(seed * 13_019 + trial * 997 + 31),
                              settings.width)
        train(base, pretrain, settings.pretrain_steps, settings.batch_size,
              settings.learning_rate)
        contexts = ["intact", "disrupted"]
        rng.shuffle(contexts)
        for context in contexts:
            state = copy.deepcopy(base)
            damaged_ids = shock(state, probe, settings) if context == "disrupted" else []
            probe_accuracy = accuracy(state, probe)
            both = {
                action: potential(
                    state, probe, post_train, holdout, settings,
                    reset=bool(action),
                    reset_seed=seed * 349_019 + trial * 701 + 23,
                ) for action in (0, 1)
            }
            episodes.append({
                "seed": seed, "trial": trial,
                "index": len(episodes),
                "context": context, "kind": task.kind,
                "probe_accuracy": probe_accuracy,
                "preserve": both[0], "renew": both[1],
                "potential_difference": both[1]["auc"] - both[0]["auc"],
                "damage_units": len(damaged_ids),
            })
    return episodes


def predicted_actions(policy: Rule, episodes: list[dict]) -> list[bool]:
    return [policy.decide(item, len(episodes)) for item in episodes]


def policy_score(episodes: list[dict], actions: list[bool]) -> float:
    return statistics.mean(
        episode["renew" if action else "preserve"]["auc"]
        for episode, action in zip(episodes, actions)
    )


def choose_policy(cache: dict[int, list[dict]], settings: Settings,
                  clock: bool) -> tuple[Rule, list[dict]]:
    train_seeds = list(range(settings.train_start,
                             settings.train_start + settings.train_seeds))
    val_seeds = list(range(settings.validation_start,
                           settings.validation_start + settings.validation_seeds))
    candidates = []
    for policy in rules(clock):
        train_value = statistics.mean(
            policy_score(cache[seed], predicted_actions(policy, cache[seed]))
            for seed in train_seeds
        )
        candidates.append((train_value, policy))
    # Stable tie-break by frozen grid order.
    order = {policy: i for i, policy in enumerate(rules(clock))}
    ranked = sorted(candidates, key=lambda pair: (-pair[0], order[pair[1]]))
    shortlist = ranked[:4]
    validation = [
        (statistics.mean(
            policy_score(cache[seed], predicted_actions(policy, cache[seed]))
            for seed in val_seeds
        ), train_value, policy)
        for train_value, policy in shortlist
    ]
    selected = max(validation, key=lambda item: (item[0], item[1],
                                                 -order[item[2]]))[2]
    trace = [{
        "kind": policy.kind, "cutoff": policy.cutoff,
        "train_auc": train_value,
        "validation_auc": next(
            (result[0] for result in validation if result[2] == policy), None
        ), "shortlisted": policy in [p for _, p in shortlist],
    } for train_value, policy in ranked]
    return selected, trace


def context_effects(episodes_by_seed: dict[int, list[dict]]) -> dict:
    per_context = {}
    for context in ("intact", "disrupted"):
        differences = [
            statistics.mean(ep["potential_difference"] for ep in episodes
                            if ep["context"] == context)
            for episodes in episodes_by_seed.values()
        ]
        per_context[context] = {
            "mean_effect": statistics.mean(differences),
            "paired_seed_effects": differences,
            "bootstrap_95": list(paired_bootstrap(differences, 990)),
            "positive_seeds": sum(v > 0 for v in differences),
        }
    return per_context


def evaluate_split(episodes_by_seed: dict[int, list[dict]],
                   state_rule: Rule, clock_rule: Rule,
                   settings: Settings, label: str) -> tuple[dict, list[dict], list[dict]]:
    rows = []
    detail = []
    for seed, episodes in sorted(episodes_by_seed.items()):
        state_actions = predicted_actions(state_rule, episodes)
        clock_actions = predicted_actions(clock_rule, episodes)
        shuffled = list(state_actions)
        random.Random(seed * 417 + 17).shuffle(shuffled)
        controls = {
            "learned": state_actions,
            "time_only": clock_actions,
            "dose_matched_shuffled": shuffled,
            "preserve_all": [False] * len(episodes),
            "renew_all": [True] * len(episodes),
            "oracle": [ep["potential_difference"] > 0 for ep in episodes],
        }
        outcomes = {
            name: policy_score(episodes, actions)
            for name, actions in controls.items()
        }
        row = {"seed": seed, "split": label, **outcomes,
               "learned_renewals": sum(state_actions),
               "shuffled_renewals": sum(shuffled),
               "time_only_renewals": sum(clock_actions),
               "oracle_renewals": sum(controls["oracle"]),
               "correct_action_fraction": statistics.mean(
                   chosen == ideal
                   for chosen, ideal in zip(state_actions, controls["oracle"])
               )}
        assert row["learned_renewals"] == row["shuffled_renewals"]
        rows.append(row)
        for ep, state_action, clock_action, shuffled_action in zip(
            episodes, state_actions, clock_actions, shuffled
        ):
            detail.append({
                "seed": seed, "split": label, "index": ep["index"],
                "trial": ep["trial"], "context": ep["context"],
                "kind": ep["kind"],
                "probe_accuracy": ep["probe_accuracy"],
                "preserve_auc": ep["preserve"]["auc"],
                "renew_auc": ep["renew"]["auc"],
                "effect": ep["potential_difference"],
                "learned_action": int(state_action),
                "time_only_action": int(clock_action),
                "shuffled_action": int(shuffled_action),
            })
    mean = {name: statistics.mean(row[name] for row in rows)
            for name in ("learned", "time_only", "dose_matched_shuffled",
                         "preserve_all", "renew_all", "oracle",
                         "learned_renewals", "time_only_renewals",
                         "correct_action_fraction")}
    comparisons = {}
    for other in ("dose_matched_shuffled", "time_only",
                  "preserve_all", "renew_all"):
        differences = [row["learned"] - row[other] for row in rows]
        comparisons[f"learned_minus_{other}"] = {
            "mean": statistics.mean(differences),
            "paired_bootstrap_95": list(
                paired_bootstrap(differences, min(episodes_by_seed) + 17)
            ),
            "positive_seeds": sum(d > 0 for d in differences),
            "negative_seeds": sum(d < 0 for d in differences),
            "zero_seeds": sum(d == 0 for d in differences),
        }
    return {
        "mean": mean,
        "contrasts": comparisons,
        "potential_outcomes_by_context": context_effects(episodes_by_seed),
        "independent_seeds": len(rows),
    }, rows, detail


def run(settings: Settings, output: Path) -> dict:
    settings.validate()
    output.mkdir(parents=True, exist_ok=True)
    train_seeds = list(range(settings.train_start,
                             settings.train_start + settings.train_seeds))
    validation_seeds = list(range(settings.validation_start,
                                  settings.validation_start + settings.validation_seeds))
    training_cache = {
        seed: make_seed_episodes(seed, settings)
        for seed in train_seeds + validation_seeds
    }
    cal = context_effects({seed: training_cache[seed] for seed in train_seeds})
    calibration_passed = (
        cal["intact"]["mean_effect"] < 0
        and cal["disrupted"]["mean_effect"] > 0
    )
    print(f"CALIBRATION train-only intact_effect="
          f"{cal['intact']['mean_effect']:.5f} "
          f"disrupted_effect={cal['disrupted']['mean_effect']:.5f} "
          f"passed={calibration_passed}", flush=True)

    state_rule, state_trace = choose_policy(training_cache, settings, False)
    clock_rule, clock_trace = choose_policy(training_cache, settings, True)
    print(f"SELECTED state={state_rule} time={clock_rule}", flush=True)

    results = {}
    all_rows = []
    all_details = []
    for split, start, count, shifted in (
        ("main", settings.test_start, settings.test_seeds, False),
        ("shifted", settings.shifted_start, settings.shifted_seeds, True),
    ):
        test_cache = {
            seed: make_seed_episodes(seed, settings, shifted=shifted)
            for seed in range(start, start + count)
        }
        result, rows, detail = evaluate_split(test_cache, state_rule,
                                             clock_rule, settings, split)
        results[split] = result
        all_rows.extend(rows)
        all_details.extend(detail)
        print(f"FINAL {split} learned={result['mean']['learned']:.5f} "
              f"clock={result['mean']['time_only']:.5f} "
              f"matched={result['mean']['dose_matched_shuffled']:.5f}", flush=True)

    summary = {
        "protocol": "M3.1 preregistered causal preserve-versus-renew benchmark",
        "settings": asdict(settings),
        "calibration_train_only": cal,
        "calibration_passed": calibration_passed,
        "selected_state_rule": asdict(state_rule),
        "selected_time_only_rule": asdict(clock_rule),
        "meta_state_selection": state_trace,
        "meta_time_only_selection": clock_trace,
        "main": results["main"],
        "shifted": results["shifted"],
        "primary_metric": "main paired seed AUC: learned minus dose_matched_shuffled",
        "required_secondary": "main paired seed AUC: learned minus independent time_only",
        "limitations": [
            "Deliberately engineered damage and renewal operator: positive control only",
            "Two contexts per pretrained model are correlated counterfactual pairs",
            "Single threshold learned offline, not self-written plasticity rules",
            "Independent snapshots, not a lifelong continually learning agent",
            "Shifted XOR/circle tasks differ from line tasks in both complexity and task family",
            "Oracle uses heldout potential outcomes only as an evaluation ceiling",
            "Meta-train/meta-validation use their own evaluation outcomes; final seeds sealed",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output / "seed_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    with (output / "counterfactuals.csv").open("w", newline="",
                                              encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_details[0]))
        writer.writeheader()
        writer.writerows(all_details)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = Settings()
    for field in fields(Settings):
        default = getattr(defaults, field.name)
        parser.add_argument("--" + field.name.replace("_", "-"),
                            default=default, type=type(default))
    parser.add_argument("--out", type=Path, default=Path("results/m3_1"))
    args = parser.parse_args()
    settings = Settings(**{f.name: getattr(args, f.name) for f in fields(Settings)})
    run(settings, args.out)


if __name__ == "__main__":
    main()
