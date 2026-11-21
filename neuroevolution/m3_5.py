"""M3.5: request extra labelled feedback when context attribution is uncertain.

All matched active policies spend the SAME global query budget (4 requests of
2 fresh labelled observations per 12-episode lifetime). Requests are decided
from the initial noisy support and prior training prototypes only. The true
latent context ID is never sent to the resolver or to the neural predictor.
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
from .m3_4 import (
    Settings as BaseSettings, Resolver, balanced_pairwise_accuracy,
    task_world, train_arm,
)
from .plasticity import paired_bootstrap


ARMS = ("margin_uncertainty", "entropy_uncertainty", "random",
        "periodic", "no_query", "query_all", "privileged_reference")
MATCHED = ARMS[:4]
MARGIN_GRID = (0.25, 0.40, 0.55, 0.70, 0.85)
ENTROPY_GRID = (0.30, 0.45, 0.60, 0.75, 0.90)


@dataclass(frozen=True)
class Settings(BaseSettings):
    train_start: int = 29000
    train_seeds: int = 6
    validation_start: int = 29500
    validation_seeds: int = 3
    test_start: int = 30000
    test_seeds: int = 24
    shifted_start: int = 31000
    shifted_seeds: int = 12
    few_start: int = 32000  # M3.4's additional same-family split -> corrupted-extra test
    few_seeds: int = 12
    main_support: int = 6
    hard_support: int = 6
    hard_flips: int = 1
    extra_per_request: int = 2
    request_quota: int = 4
    reservoir_threshold: float = 0.84  # M3.4 acceptance threshold, frozen

    def validate(self) -> None:
        super().validate()
        if not 0 < self.extra_per_request < 8:
            raise ValueError("Query group size must be small and nonzero")
        if not 0 < self.request_quota < self.tasks:
            raise ValueError("Query quota must allow decisions, not all/no requests")
        if not 0.0 <= self.reservoir_threshold <= 1.0:
            raise ValueError("Invalid frozen context admission threshold")


def _latent_rules(seed: int, shifted: bool) -> list[WorldTask]:
    """Reproduce M3.4's *environment* labels for a disjoint query stream."""
    rng = random.Random(seed * 10_019 + 91)
    kinds = ("stripe", "checker") if shifted else ("line", "xor")
    return [
        WorldTask(
            kind=kind,
            angle=rng.uniform(-math.pi, math.pi),
            shift=rng.uniform(-.18, .18),
            phase=rng.uniform(-math.pi, math.pi)
            if kind in ("stripe", "checker") else rng.uniform(-.18, .18),
        )
        for kind in kinds
    ]


def query_pool(seed: int, settings: Settings, data: dict,
               *, shifted: bool, corrupt_extra: bool) -> list[list]:
    """Only the generator knows the latent label function.

    The policy NEVER receives this pool before deciding to query.
    Extra x,y is independent of existing base/training/eval streams.
    """
    rules = _latent_rules(seed, shifted)
    extra = []
    for idx, episode in enumerate(data["episodes"]):
        c = episode["latent"]  # Generator only. Never passed to decision function.
        rng = random.Random(seed * 99_000_017 + idx * 199 + 83)
        values = []
        for _ in range(settings.extra_per_request):
            point = (rng.uniform(-1., 1.), rng.uniform(-1., 1.))
            y = rules[c // 2].label(point) ^ (c % 2)
            values.append((point, y))
        if corrupt_extra:
            chosen = rng.randrange(len(values))
            point, y = values[chosen]
            values[chosen] = (point, 1 - y)
        extra.append(values)
    return extra


def uncertainty_scores(resolver: Resolver, support: list) -> dict:
    """Features use ONLY existing prototypes and INITIAL labelled support.

    No oracle ID, future labels, training stream or evaluation panel.
    """
    scores = [resolver.match_score(slot, support)
              for slot in sorted(resolver.prototypes)
              if resolver.prototypes[slot]]
    if not scores:
        return {"margin": 0.0, "entropy": 0.0, "best": 0.0,
                "second": 0.0, "candidates": 0}
    ranked = sorted(scores, reverse=True)
    best = ranked[0]
    second = ranked[1] if len(ranked) > 1 else 0.0
    margin = (
        1.0 - (best - second) if len(ranked) > 1
        else 1.0 - best
    )
    if len(scores) == 1:
        entropy = 0.0
    else:
        weights = [math.exp(6.0 * (v - best)) for v in scores]
        scale = sum(weights)
        probabilities = [v / scale for v in weights]
        entropy = -sum(p * math.log(p) for p in probabilities) / math.log(len(scores))
    return {
        "margin": max(0.0, min(1.0, margin)),
        "entropy": max(0.0, min(1.0, entropy)),
        "best": best,
        "second": second,
        "candidates": len(scores),
    }


def scheduled_queries(seed: int, settings: Settings, arm: str) -> set[int]:
    if arm == "random":
        return set(random.Random(seed * 417 + 173).sample(
            range(settings.tasks), settings.request_quota
        ))
    if arm == "periodic":
        n, budget = settings.tasks, settings.request_quota
        return {i for i in range(n)
                if (i + 1) * budget // n > i * budget // n}
    return set()


def decide_query(scores: dict, arm: str, threshold: float,
                 *, index: int, remaining: int, settings: Settings,
                 planned: set[int]) -> tuple[bool, bool, bool]:
    """Decide BEFORE any extra query label is accessed.

    Returns (actual, original_request, forced_or_blocked). Four matched
    requests are guaranteed by the remaining-opportunities constraint.
    """
    if arm == "margin_uncertainty":
        want = scores["margin"] >= threshold
    elif arm == "entropy_uncertainty":
        want = scores["entropy"] >= threshold
    elif arm in ("random", "periodic"):
        want = index in planned
    elif arm == "no_query":
        want = False
    elif arm == "query_all":
        want = True
    else:
        raise ValueError(f"Invalid request mode {arm}")
    if arm in MATCHED:
        must = remaining >= settings.tasks - index
        actual = (want or must) and remaining > 0
        return actual, want, actual != want
    return want, want, False


def resolve_seed(seed: int, settings: Settings, data: dict,
                 extra: list[list], arm: str, threshold: float) -> dict:
    """Online decisions: support -> query decision -> extra if requested -> infer.

    The returned latent-keyed diagnostics are calculated only after each action.
    They are NEVER supplied back to the resolver or policy.
    """
    if arm not in ARMS[:-1]:
        raise ValueError("Privileged reference cannot query")
    if len(extra) != len(data["episodes"]):
        raise ValueError("Every episode needs a sealed extra-query pool")
    resolver = Resolver(settings, settings.reservoir_threshold)
    plan = scheduled_queries(seed, settings, arm)
    remaining = settings.request_quota
    episode_slots = []
    log = []
    for index, episode in enumerate(data["episodes"]):
        initial = episode["support"]
        observed = uncertainty_scores(resolver, initial)
        query, want, overridden = decide_query(
            observed, arm, threshold,
            index=index, remaining=remaining, settings=settings,
            planned=plan,
        )
        # The pool is read ONLY after the decision is final.
        support = initial + extra[index] if query else initial
        if query and arm in MATCHED:
            remaining -= 1
        slot = resolver.infer(support, allow_new=True)
        episode_slots.append(slot)
        resolver.observe_training(slot, episode["training"])
        log.append({
            "index": index, "slot": slot, "query": int(query),
            "requested": int(want), "forced_or_blocked": int(overridden),
            "uncertainty_margin": observed["margin"],
            "uncertainty_entropy": observed["entropy"],
            "top_score": observed["best"],
            "candidate_count": observed["candidates"],
            "extra_examples_consumed": len(extra[index]) if query else 0,
        })
    if arm in MATCHED and remaining:
        raise AssertionError("Matched policy left query quota unspent")
    expected = settings.request_quota if arm in MATCHED else (
        0 if arm == "no_query" else settings.tasks
    )
    if sum(item["query"] for item in log) != expected:
        raise AssertionError("Query count violates the predetermined policy budget")

    # At final evaluation, ALL arms receive identical noisy final support
    # (no further paid query); no neural holdout labels enter slot inference.
    final_slots = {
        c: resolver.infer(data["final_support"][c], allow_new=False)
        for c in range(settings.contexts)
    }
    shuffled_final = {
        c: resolver.infer(
            data["final_support"][(c + 1) % settings.contexts], allow_new=False
        )
        for c in range(settings.contexts)
    }
    first = {}
    for ep, slot in zip(data["episodes"], episode_slots):
        first.setdefault(ep["latent"], slot)
    returning = [(ep["latent"], slot)
                 for ep, slot in zip(data["episodes"], episode_slots)
                 if not ep["novel"]]
    trace = {
        "episode_slots": episode_slots,
        "first_slots": first,
        "final_slots": final_slots,
        "shuffled_final_slots": shuffled_final,
        "first_unique": len(set(episode_slots[:settings.contexts])),
        "return_consistency": statistics.mean(
            slot == first[c] for c, slot in returning
        ),
        "final_support_consistency": statistics.mean(
            final_slots[c] == first[c] for c in range(settings.contexts)
        ),
        "pairwise_accuracy": balanced_pairwise_accuracy(
            data["episodes"], episode_slots
        ),
        "prototype_samples": sum(map(len, resolver.prototypes.values())),
        "knn_votes": resolver.votes,
        "query_events": sum(item["query"] for item in log),
        "extra_labels": sum(item["extra_examples_consumed"] for item in log),
        "forced_or_blocked_count": sum(item["forced_or_blocked"] for item in log),
        "unforced_uncertainty_requests": sum(
            item["query"] and item["requested"] and
            not item["forced_or_blocked"] for item in log
        ) if arm in MATCHED[:2] else 0,
        "log": log,
    }
    return trace


def select_threshold(settings: Settings, arm: str) -> tuple[float, list[dict]]:
    if arm not in MATCHED[:2]:
        raise ValueError(arm)
    grid = MARGIN_GRID if arm == "margin_uncertainty" else ENTROPY_GRID
    data = {}
    for seed in list(range(settings.train_start,
                           settings.train_start + settings.train_seeds)) + list(
        range(settings.validation_start,
              settings.validation_start + settings.validation_seeds)
    ):
        world = task_world(seed, settings, "few")
        extra = query_pool(seed, settings, world, shifted=False,
                           corrupt_extra=False)
        data[seed] = (world, extra)
    history = []
    for threshold in grid:
        train = statistics.mean(
            resolve_seed(seed, settings, *data[seed], arm, threshold)["pairwise_accuracy"]
            for seed in range(settings.train_start,
                              settings.train_start + settings.train_seeds)
        )
        history.append({
            "policy": arm, "threshold": threshold,
            "train_pairwise": train, "validation_pairwise": None,
        })
    order = {threshold: i for i, threshold in enumerate(grid)}
    ranked = sorted(history, key=lambda item:
                    (-item["train_pairwise"], order[item["threshold"]]))
    for row in ranked[:2]:
        threshold = row["threshold"]
        row["validation_pairwise"] = statistics.mean(
            resolve_seed(seed, settings, *data[seed], arm, threshold)["pairwise_accuracy"]
            for seed in range(settings.validation_start,
                              settings.validation_start + settings.validation_seeds)
        )
    winner = max(ranked[:2], key=lambda item: (
        item["validation_pairwise"], item["train_pairwise"],
        -order[item["threshold"]]
    ))
    print(f"M3.5 SELECTED {arm} threshold={winner['threshold']}"
          f" val={winner['validation_pairwise']:.4f}", flush=True)
    return winner["threshold"], history


def summarize(records: list[dict], settings: Settings, label: str) -> dict:
    lookup = {arm: {r["seed"]: r for r in records if r["arm"] == arm}
              for arm in ARMS}
    metrics = (
        "final_balanced_accuracy", "worst_context_accuracy",
        "final_peak_forgetting", "mean_learning_auc", "mean_forgetting",
        "return_auc", "replay_updates", "memory_stored",
        "pairwise_accuracy", "first_unique", "return_consistency",
        "final_support_consistency", "query_events", "extra_labels",
        "forced_or_blocked_count", "unforced_uncertainty_requests",
        "knn_votes",
    )
    means = {arm: {
        key: statistics.mean(row[key] for row in results.values())
        for key in metrics
    } for arm, results in lookup.items()}
    pairs = (
        ("margin_uncertainty", "random", "margin_vs_random"),
        ("margin_uncertainty", "periodic", "margin_vs_periodic"),
        ("margin_uncertainty", "no_query", "margin_vs_no_query"),
        ("margin_uncertainty", "query_all", "margin_vs_query_all"),
        ("margin_uncertainty", "entropy_uncertainty", "margin_vs_entropy"),
        ("entropy_uncertainty", "random", "entropy_vs_random"),
    )
    contrasts = {}
    for a, b, name in pairs:
        diff = [lookup[a][seed]["final_balanced_accuracy"] -
                lookup[b][seed]["final_balanced_accuracy"]
                for seed in sorted(lookup[a])]
        contrasts[name] = {
            "mean_accuracy_diff": statistics.mean(diff),
            "paired_bootstrap_95": list(paired_bootstrap(
                diff, min(lookup[a]) + 47
            )),
            "positive_seeds": sum(d > 0 for d in diff),
            "negative_seeds": sum(d < 0 for d in diff),
            "zero_seeds": sum(d == 0 for d in diff),
            "gain_per_extra_label_at_equal_budget":
                statistics.mean(diff) / (settings.request_quota *
                                          settings.extra_per_request)
                if a in MATCHED and b in MATCHED else None,
        }
    if any(
        row["query_events"] != settings.request_quota or
        row["extra_labels"] != settings.request_quota *
                               settings.extra_per_request
        for arm in MATCHED for row in lookup[arm].values()
    ):
        raise AssertionError("Matched query-count or extra-label mismatch")
    return {
        "split": label, "independent_seeds": len(lookup["margin_uncertainty"]),
        "means": means, "comparisons": contrasts,
        "all_matched_policies_used_exact_equal_extra_labels": True,
        "notes": [
            "Historical latent IDs are used only to audit clustering, not in queries",
            "Different inferred slots cause legitimately different replay groupings",
            "Uncertainty request count includes forced quota completion separately",
        ],
    }


def run(settings: Settings, output: Path) -> dict:
    settings.validate()
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    margin, margin_history = select_threshold(settings, "margin_uncertainty")
    entropy, entropy_history = select_threshold(settings, "entropy_uncertainty")
    all_records = []
    all_decisions = []
    splits = []
    for label, start, count, family, corrupt_extra in (
        ("main", settings.test_start, settings.test_seeds, "few", False),
        ("shifted", settings.shifted_start, settings.shifted_seeds, "shifted", False),
        ("corrupted_extra", settings.few_start, settings.few_seeds, "few", True),
    ):
        records = []
        for seed in range(start, start + count):
            world = task_world(seed, settings, family)
            extra = query_pool(seed, settings, world,
                               shifted=(family == "shifted"),
                               corrupt_extra=corrupt_extra)
            # The heldout world is unchanged for all policies.
            fallback_trace = None
            for arm in ARMS:
                if arm == "privileged_reference":
                    if fallback_trace is None:
                        raise AssertionError("Privileged comparator must run last")
                    metrics = train_arm(seed, settings, world, fallback_trace,
                                        "oracle_context")
                    trace = fallback_trace
                else:
                    threshold = margin if arm == "margin_uncertainty" else (
                        entropy if arm == "entropy_uncertainty" else 0.0
                    )
                    trace = resolve_seed(seed, settings, world, extra, arm, threshold)
                    metrics = train_arm(seed, settings, world, trace, "inferred_fifo")
                    if arm == "margin_uncertainty":
                        fallback_trace = trace
                    for log in trace["log"]:
                        all_decisions.append({
                            "split": label, "seed": seed, "arm": arm, **log,
                        })
                record = {
                    "split": label, "seed": seed, "arm": arm,
                    **{key: metrics[key] for key in (
                        "final_balanced_accuracy", "worst_context_accuracy",
                        "final_peak_forgetting", "mean_learning_auc",
                        "mean_forgetting", "return_auc", "replay_updates",
                        "memory_stored",
                    )},
                    **{key: trace[key] for key in (
                        "pairwise_accuracy", "first_unique",
                        "return_consistency", "final_support_consistency",
                        "prototype_samples", "knn_votes",
                    )},
                    "query_events": trace["query_events"] if arm !=
                        "privileged_reference" else 0,
                    "extra_labels": trace["extra_labels"] if arm !=
                        "privileged_reference" else 0,
                    "forced_or_blocked_count": trace["forced_or_blocked_count"]
                        if arm != "privileged_reference" else 0,
                    "unforced_uncertainty_requests":
                        trace["unforced_uncertainty_requests"]
                        if arm != "privileged_reference" else 0,
                    "episode_slots": trace["episode_slots"]
                        if arm != "privileged_reference" else [],
                }
                records.append(record)
            margin_row = records[0]
            random_row = records[2]
            print(f"M3.5 FINAL {label} seed={seed}"
                  f" margin={margin_row['final_balanced_accuracy']:.4f}"
                  f" random={random_row['final_balanced_accuracy']:.4f}"
                  f" return={margin_row['return_consistency']:.3f}", flush=True)
        all_records.extend(records)
        splits.append((label, records))
    summary = {
        "protocol": "M3.5 preregistered budget-matched active feedback querying",
        "settings": asdict(settings),
        "selected_margin_threshold": margin,
        "selected_entropy_threshold": entropy,
        "meta_history": margin_history + entropy_history,
        "main": summarize(splits[0][1], settings, "main"),
        "shifted": summarize(splits[1][1], settings, "shifted"),
        "corrupted_extra": summarize(splits[2][1], settings, "corrupted_extra"),
        "primary_metric": "main final balanced context accuracy: margin vs random",
        "necessary_secondary": "main margin vs periodic at identical 8-label budget",
        "wall_seconds_including_meta_selection": time.perf_counter() - started,
        "limitations": [
            "Query policy is externally programmed, not learned by neural weights",
            "True latent ID used only in test harness and offline meta-fitness",
            "All methods receive initial 6 labels with exactly one corruption",
            "Extra queries supply IID points, not optimally selected spatial coordinates",
            "Forced last-chance requests are not autonomous uncertainty decisions",
            "Meta-scoring uses training/validation true IDs only, sealed tests isolated",
            "Equal SGD and query counts do not mean equal CPU/kNN processing",
            "Extra-query noise is injected artificially in corrupted-extra split",
            "Opposing rule pairs make no-context blind accuracy exactly 50%",
            "Privileged reference knows true IDs and is not query-cost-matched",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output / "seed_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        cols = (
            "split", "seed", "arm", "final_balanced_accuracy",
            "worst_context_accuracy", "final_peak_forgetting",
            "mean_learning_auc", "mean_forgetting", "return_auc",
            "pairwise_accuracy", "first_unique", "return_consistency",
            "final_support_consistency", "query_events", "extra_labels",
            "forced_or_blocked_count", "unforced_uncertainty_requests",
            "replay_updates", "memory_stored", "prototype_samples",
            "knn_votes", "episode_slots",
        )
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in all_records:
            writer.writerow({key: json.dumps(row[key]) if key == "episode_slots"
                             else row[key] for key in cols})
    with (output / "request_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        cols = ("split", "seed", "arm", "index", "slot", "query", "requested",
                "forced_or_blocked", "uncertainty_margin",
                "uncertainty_entropy", "top_score",
                "candidate_count", "extra_examples_consumed")
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows({key: row[key] for key in cols}
                         for row in all_decisions)
    print("M3.5 PRIMARY", summary["main"]["comparisons"]["margin_vs_random"],
          flush=True)
    print("M3.5 PERIODIC", summary["main"]["comparisons"]["margin_vs_periodic"],
          flush=True)
    print("M3.5 SHIFTED", summary["shifted"]["comparisons"]["margin_vs_random"],
          flush=True)
    print("M3.5 NOISY EXTRA",
          summary["corrupted_extra"]["comparisons"]["margin_vs_random"],
          flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default = Settings()
    for field in fields(Settings):
        v = getattr(default, field.name)
        parser.add_argument("--" + field.name.replace("_", "-"),
                            type=type(v), default=v)
    parser.add_argument("--out", type=Path, default=Path("results/m3_5"))
    args = parser.parse_args()
    run(Settings(**{field.name: getattr(args, field.name)
                    for field in fields(Settings)}), args.out)


if __name__ == "__main__":
    main()
