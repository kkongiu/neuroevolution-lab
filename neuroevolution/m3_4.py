"""M3.4 — Online latent-context inference from labelled support observations.

A bounded, externally specified kNN resolver allocates anonymous context slots
from training-only samples; a persistent ReLU MLP uses inferred one-hot slots.
Latent IDs are never passed to the resolver or deployable neural input.
Counterfactual support ablations share EXACTLY the same learned network.
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
from .m3_3 import ContextNetwork, acc, encode
from .plasticity import curve_auc, paired_bootstrap


Sample = tuple[tuple[float, float], int]
ARMS = ("inferred_fifo", "inferred_reservoir", "inferred_no_replay",
        "oracle_context", "no_context_replay")
THRESHOLDS = (0.60, 0.68, 0.76, 0.84)


@dataclass(frozen=True)
class Settings:
    train_start: int = 24000
    train_seeds: int = 6
    validation_start: int = 25000
    validation_seeds: int = 3
    test_start: int = 26000
    test_seeds: int = 24
    few_start: int = 27000
    few_seeds: int = 12
    shifted_start: int = 28000
    shifted_seeds: int = 12
    contexts: int = 4
    tasks: int = 12
    width: int = 24
    steps: int = 48
    batch_size: int = 4
    eval_size: int = 64
    report_every: int = 12
    learning_rate: float = 0.04
    main_support: int = 16
    hard_support: int = 6
    hard_flips: int = 1
    prototype_size: int = 64
    buffer_size: int = 24
    replay_every: int = 4
    knn_k: int = 7

    def validate(self) -> None:
        values = (self.train_seeds, self.validation_seeds, self.test_seeds,
                  self.few_seeds, self.shifted_seeds, self.contexts,
                  self.tasks, self.width, self.steps, self.batch_size,
                  self.eval_size, self.report_every, self.main_support,
                  self.hard_support, self.prototype_size, self.buffer_size,
                  self.replay_every, self.knn_k)
        if min(values) < 1 or self.contexts != 4 or self.tasks < 8:
            raise ValueError("All counts positive, contexts=4, tasks>=8")
        if self.steps % self.report_every != 0 or self.replay_every > self.steps:
            raise ValueError("Invalid SGD evaluation/replay schedule")
        if not (0 < self.learning_rate <= .5):
            raise ValueError("Invalid learning rate")
        if not (0 <= self.hard_flips < self.hard_support):
            raise ValueError("Invalid fixed support-label corruption")
        ranges = ((self.train_start, self.train_seeds),
                  (self.validation_start, self.validation_seeds),
                  (self.test_start, self.test_seeds),
                  (self.few_start, self.few_seeds),
                  (self.shifted_start, self.shifted_seeds))
        for i, (start, size) in enumerate(ranges):
            if start < 0:
                raise ValueError("Seed start nonnegative")
            this = set(range(start, start + size))
            for next_start, next_size in ranges[i + 1:]:
                if this.intersection(range(next_start, next_start + next_size)):
                    raise ValueError("Seed partitions overlap")


def task_world(seed: int, settings: Settings, kind: str) -> dict:
    """The true latent ID is used for label generation/audit only."""
    if kind not in ("main", "few", "shifted"):
        raise ValueError(kind)
    harder = kind != "main"
    shifted = kind == "shifted"
    support_count = settings.hard_support if harder else settings.main_support
    rng = random.Random(seed * 10_019 + 91)
    families = ("stripe", "checker") if shifted else ("line", "xor")
    base_rules = [
        WorldTask(
            kind=family,
            angle=rng.uniform(-math.pi, math.pi),
            shift=rng.uniform(-.18, .18),
            phase=rng.uniform(-math.pi, math.pi)
            if family in ("stripe", "checker") else rng.uniform(-.18, .18),
        )
        for family in families
    ]

    def label(c: int, xy: tuple[float, float]) -> int:
        return base_rules[c // 2].label(xy) ^ (c % 2)

    def sample(seed_salt: int, context: int, count: int,
               flips: int = 0) -> list[Sample]:
        local = random.Random(seed * 7_000_001 + seed_salt * 10_007 + context * 101)
        observations = []
        for _ in range(count):
            xy = (local.uniform(-1., 1.), local.uniform(-1., 1.))
            observations.append((xy, label(context, xy)))
        if flips:
            for i in local.sample(range(count), flips):
                xy, y = observations[i]
                observations[i] = (xy, 1 - y)
        return observations

    panel = {}
    for pair in (0, 1):
        panel_rng = random.Random(seed * 1_000_003 + pair * 847 + 909)
        xy = [(panel_rng.uniform(-1., 1.), panel_rng.uniform(-1., 1.))
              for _ in range(settings.eval_size)]
        for c in (pair * 2, pair * 2 + 1):
            panel[c] = [(point, label(c, point)) for point in xy]

    contexts = list(range(settings.contexts))
    rng.shuffle(contexts)
    contexts.extend(rng.randrange(settings.contexts)
                    for _ in range(settings.tasks - settings.contexts))
    episodes = []
    for i, c in enumerate(contexts):
        episodes.append({
            "latent": c,  # inaccessible to the resolver (audit / oracle only)
            "training": sample(101 + 13 * i, c,
                               settings.steps * settings.batch_size),
            "support": sample(301 + 19 * i, c, support_count,
                              settings.hard_flips if harder else 0),
            "novel": c not in contexts[:i],
        })
    final_support = {
        c: sample(801 + 31 * c, c, support_count,
                  settings.hard_flips if harder else 0)
        for c in range(settings.contexts)
    }
    return {"episodes": episodes, "panel": panel,
            "final_support": final_support, "condition": kind}


class Resolver:
    """Fixed kNN classifier plus externally meta-selected admission threshold."""

    def __init__(self, settings: Settings, threshold: float):
        self.settings = settings
        self.threshold = threshold
        self.prototypes: dict[int, list[Sample]] = {}
        self.votes = 0

    def _predict(self, slot: int, xy: tuple[float, float]) -> int:
        reference = self.prototypes[slot]
        close = sorted(reference, key=lambda sample:
                       (sample[0][0] - xy[0]) ** 2 +
                       (sample[0][1] - xy[1]) ** 2)
        neighbours = close[:self.settings.knn_k]
        self.votes += len(neighbours)
        return int(sum(y for _, y in neighbours) * 2 >= len(neighbours))

    def match_score(self, slot: int, support: list[Sample]) -> float:
        return sum(self._predict(slot, xy) == label
                   for xy, label in support) / len(support)

    def infer(self, support: list[Sample], *, allow_new: bool) -> int:
        if not support:
            raise ValueError("Context inference requires nonempty labelled support")
        if not self.prototypes:
            if not allow_new:
                raise ValueError("Resolver was never trained")
            self.prototypes[0] = []
            return 0
        scored = sorted(
            ((self.match_score(slot, support), slot)
             for slot in sorted(self.prototypes)),
            key=lambda p: (-p[0], p[1])
        )
        top_score, top_slot = scored[0]
        if (allow_new and top_score < self.threshold
                and len(self.prototypes) < self.settings.contexts):
            new_slot = min(set(range(self.settings.contexts)) -
                           set(self.prototypes))
            self.prototypes[new_slot] = []
            return new_slot
        return top_slot

    def observe_training(self, slot: int, training: list[Sample]) -> None:
        reference = self.prototypes[slot]
        reference.extend(training[:self.settings.prototype_size])
        del reference[:-self.settings.prototype_size]


def balanced_pairwise_accuracy(episodes: list[dict],
                               slots: list[int]) -> float:
    yes, no = [], []
    for i in range(len(episodes)):
        for j in range(i):
            same_truth = episodes[i]["latent"] == episodes[j]["latent"]
            same_slot = slots[i] == slots[j]
            (yes if same_truth else no).append(same_truth == same_slot)
    return 0.5 * (statistics.mean(yes) + statistics.mean(no))


def trace_resolver(world: dict, settings: Settings, threshold: float) -> dict:
    """Only support/training are sent to the resolver; truth used AFTER choices."""
    resolver = Resolver(settings, threshold)
    slots = []
    for episode in world["episodes"]:
        slot = resolver.infer(episode["support"], allow_new=True)
        slots.append(slot)
        resolver.observe_training(slot, episode["training"])
    final = {
        c: resolver.infer(world["final_support"][c], allow_new=False)
        for c in range(settings.contexts)
    }
    shuffled = {
        c: resolver.infer(world["final_support"][(c + 1) % settings.contexts],
                          allow_new=False)
        for c in range(settings.contexts)
    }
    first = {}
    for ep, slot in zip(world["episodes"], slots):
        first.setdefault(ep["latent"], slot)
    returning = [(ep["latent"], slot) for ep, slot in zip(world["episodes"], slots)
                 if not ep["novel"]]
    return {
        "episode_slots": slots, "final_slots": final,
        "shuffled_final_slots": shuffled,
        "first_slots": first,
        "first_unique": len(set(slots[:settings.contexts])),
        "return_consistency": statistics.mean(
            slot == first[c] for c, slot in returning
        ),
        "final_support_consistency": statistics.mean(
            final[c] == first[c] for c in range(settings.contexts)
        ),
        "pairwise_accuracy": balanced_pairwise_accuracy(world["episodes"], slots),
        "prototype_samples": sum(map(len, resolver.prototypes.values())),
        "knn_votes": resolver.votes,
    }


def select_threshold(settings: Settings) -> tuple[float, list[dict]]:
    cache = {seed: task_world(seed, settings, "main") for seed in
             list(range(settings.train_start,
                        settings.train_start + settings.train_seeds)) +
             list(range(settings.validation_start,
                        settings.validation_start + settings.validation_seeds))}
    history = []
    for threshold in THRESHOLDS:
        training = statistics.mean(
            trace_resolver(cache[seed], settings, threshold)["pairwise_accuracy"]
            for seed in range(settings.train_start,
                              settings.train_start + settings.train_seeds)
        )
        history.append({"threshold": threshold,
                        "meta_train_pairwise_accuracy": training,
                        "meta_validation_pairwise_accuracy": None})
    indexed = {t: i for i, t in enumerate(THRESHOLDS)}
    ranked = sorted(history, key=lambda row: (
        -row["meta_train_pairwise_accuracy"], indexed[row["threshold"]]
    ))
    for row in ranked[:2]:
        threshold = row["threshold"]
        row["meta_validation_pairwise_accuracy"] = statistics.mean(
            trace_resolver(cache[seed], settings, threshold)["pairwise_accuracy"]
            for seed in range(settings.validation_start,
                              settings.validation_start + settings.validation_seeds)
        )
    winner = max(ranked[:2], key=lambda row: (
        row["meta_validation_pairwise_accuracy"],
        row["meta_train_pairwise_accuracy"],
        -indexed[row["threshold"]]
    ))
    print("M3.4 SELECTED threshold", winner["threshold"],
          "meta_validation", winner["meta_validation_pairwise_accuracy"], flush=True)
    return winner["threshold"], history


class ReplayMemory:
    """Fixed capacity per inferred (or oracle) slot, no test samples."""

    def __init__(self, settings: Settings, seed: int, kind: str):
        if kind not in ("fifo", "reservoir"):
            raise ValueError(kind)
        self.kind = kind
        self.settings = settings
        self.stored: dict[int, list[Sample]] = {}
        self.seen: dict[int, int] = {}
        self.rngs: dict[int, random.Random] = {}
        self.seed = seed

    def update(self, slot: int, samples: list[Sample]) -> None:
        bank = self.stored.setdefault(slot, [])
        if slot not in self.rngs:
            self.rngs[slot] = random.Random(self.seed * 1901 + slot * 499 + 11)
            self.seen[slot] = 0
        for sample in samples:
            self.seen[slot] += 1
            if self.kind == "fifo":
                bank.append(sample)
                if len(bank) > self.settings.buffer_size:
                    bank.pop(0)
            elif len(bank) < self.settings.buffer_size:
                bank.append(sample)
            else:
                index = self.rngs[slot].randrange(self.seen[slot])
                if index < self.settings.buffer_size:
                    bank[index] = sample

    def replay(self, current_slot: int, seed: int, episode: int,
               step: int, batch_size: int) -> tuple[int, list[Sample]] | None:
        available = [slot for slot in sorted(self.stored)
                     if slot != current_slot and self.stored[slot]]
        if not available:
            return None
        slot = available[(seed + episode * 7 + step * 13) % len(available)]
        bank = self.stored[slot]
        return slot, [bank[(episode * 43 + step * batch_size + j) % len(bank)]
                      for j in range(batch_size)]

    @property
    def size(self) -> int:
        return sum(len(bank) for bank in self.stored.values())


def _encoded(samples: list[Sample], slot: int | None) -> list:
    if slot is None:
        return [(xy + (0., 0., 0., 0.), label) for xy, label in samples]
    return [(encode(xy[0], xy[1], slot), label) for xy, label in samples]


def _score(net: ContextNetwork, samples: list[Sample],
           slot: int | None) -> float:
    return acc(net, _encoded(samples, slot))


def train_arm(seed: int, settings: Settings, world: dict,
              trace: dict, arm: str) -> dict:
    if arm not in ARMS:
        raise ValueError(arm)
    net = ContextNetwork.create(seed, settings)
    inferred = arm not in ("oracle_context", "no_context_replay")
    no_context = arm == "no_context_replay"
    replay = arm != "inferred_no_replay"
    memory_kind = "reservoir" if arm == "inferred_reservoir" else "fifo"
    bank = ReplayMemory(settings, seed, memory_kind)

    all_rows = []
    peaks = {}
    replay_updates = 0
    for i, ep in enumerate(world["episodes"]):
        latent = ep["latent"]  # audit / oracle only
        slot = trace["episode_slots"][i] if arm != "oracle_context" else latent
        cue = None if no_context else slot
        curve = [(0, _score(net, world["panel"][latent], cue))]
        current = _encoded(ep["training"], cue)
        for step in range(1, settings.steps + 1):
            start = (step - 1) * settings.batch_size
            batch = current[start:start + settings.batch_size]
            if replay and step % settings.replay_every == 0:
                choice = bank.replay(slot, seed, i, step, settings.batch_size)
                if choice is not None:
                    old_slot, replay_samples = choice
                    batch = _encoded(replay_samples, None if no_context else old_slot)
                    replay_updates += 1
            net.train_batch(batch, settings.learning_rate)
            if step % settings.report_every == 0:
                curve.append((step, _score(net, world["panel"][latent], cue)))
        bank.update(slot, ep["training"])
        remembered = {}
        for context in dict.fromkeys(e["latent"] for e in world["episodes"][:i + 1]):
            old_slot = (
                context if arm == "oracle_context"
                else trace["first_slots"][context]
            )
            remembered[context] = _score(
                net, world["panel"][context], None if no_context else old_slot
            )
        forgetting = (statistics.mean(
            max(0., peaks[context] - score)
            for context, score in remembered.items() if context in peaks
        ) if peaks else 0.)
        for context, score in remembered.items():
            peaks[context] = max(peaks.get(context, 0.), score)
        all_rows.append({
            "seed": seed, "arm": arm, "episode": i,
            "latent_audit_only": latent, "novel": ep["novel"],
            "assigned_slot": slot if not no_context else -1,
            "auc": curve_auc(curve, settings.steps),
            "final_episode_accuracy": curve[-1][1],
            "seen_accuracy": statistics.mean(remembered.values()),
            "forgetting": forgetting,
            "replay_updates_cumulative": replay_updates,
        })

    final_accuracies = []
    for c in range(settings.contexts):
        slot = (
            c if arm == "oracle_context" else
            trace["final_slots"][c]
        )
        final_accuracies.append(
            _score(net, world["panel"][c], None if no_context else slot)
        )

    # Paired final-support interventions on EXACT SAME inferred+FIFO trained net.
    shuffled = None
    no_support = None
    if arm == "inferred_fifo":
        shuffled = statistics.mean(
            _score(net, world["panel"][c],
                   trace["shuffled_final_slots"][c])
            for c in range(settings.contexts)
        )
        no_support = statistics.mean(
            _score(net, world["panel"][c], None)
            for c in range(settings.contexts)
        )
        if abs(no_support - .5) > 1e-12:
            raise AssertionError("No-support balanced mathematical ceiling was broken")

    return {
        "seed": seed, "arm": arm,
        "final_balanced_accuracy": statistics.mean(final_accuracies),
        "worst_context_accuracy": min(final_accuracies),
        "final_peak_forgetting": statistics.mean(
            max(0., peaks[c] - final_accuracies[c]) for c in range(settings.contexts)
        ),
        "mean_learning_auc": statistics.mean(x["auc"] for x in all_rows[1:]),
        "mean_seen_accuracy": statistics.mean(x["seen_accuracy"] for x in all_rows[1:]),
        "mean_forgetting": statistics.mean(x["forgetting"] for x in all_rows[1:]),
        "return_auc": statistics.mean(
            x["auc"] for x in all_rows if not x["novel"]
        ),
        "replay_updates": replay_updates,
        "memory_stored": bank.size,
        "final_accuracies": final_accuracies,
        "shuffled_support_accuracy": shuffled,
        "no_support_accuracy": no_support,
        "episodes": all_rows,
    }


def summarize(records: list[dict], traces: list[dict],
              settings: Settings, label: str) -> dict:
    lookup = {arm: {r["seed"]: r for r in records if r["arm"] == arm}
              for arm in ARMS}
    seed_order = sorted(lookup["inferred_fifo"])
    comparisons = (
        ("inferred_fifo", "shuffled_support", "support_vs_shuffled",
         "final_balanced_accuracy", "shuffled_support_accuracy"),
        ("inferred_fifo", "no_support", "support_vs_no_support",
         "final_balanced_accuracy", "no_support_accuracy"),
        ("inferred_fifo", "inferred_no_replay", "fifo_vs_no_replay",
         "final_balanced_accuracy", "final_balanced_accuracy"),
        ("inferred_fifo", "inferred_reservoir", "fifo_vs_reservoir",
         "final_balanced_accuracy", "final_balanced_accuracy"),
        ("inferred_fifo", "oracle_context", "inferred_vs_oracle",
         "final_balanced_accuracy", "final_balanced_accuracy"),
        ("inferred_fifo", "no_context_replay", "inferred_vs_no_context",
         "final_balanced_accuracy", "final_balanced_accuracy"),
    )
    contrasts = {}
    for source, target, name, field, target_field in comparisons:
        differences = [
            lookup[source][seed][field] -
            (lookup[source][seed][target_field] if target in
             ("shuffled_support", "no_support") else
             lookup[target][seed][target_field])
            for seed in seed_order
        ]
        contrasts[name] = {
            "mean_difference": statistics.mean(differences),
            "bootstrap_95": list(paired_bootstrap(differences, seed_order[0] + 31)),
            "positive_seeds": sum(d > 0 for d in differences),
            "negative_seeds": sum(d < 0 for d in differences),
            "zero_seeds": sum(d == 0 for d in differences),
        }
    metrics = ("final_balanced_accuracy", "worst_context_accuracy",
               "final_peak_forgetting", "mean_learning_auc",
               "mean_seen_accuracy", "mean_forgetting", "return_auc",
               "replay_updates", "memory_stored")
    means = {arm: {name: statistics.mean(r[name] for r in results.values())
                   for name in metrics} for arm, results in lookup.items()}
    return {
        "split": label, "independent_seeds": len(seed_order),
        "means": means,
        "comparisons": contrasts,
        "shuffled_support_mean_accuracy": statistics.mean(
            lookup["inferred_fifo"][seed]["shuffled_support_accuracy"]
            for seed in seed_order
        ),
        "no_support_mean_accuracy": statistics.mean(
            lookup["inferred_fifo"][seed]["no_support_accuracy"]
            for seed in seed_order
        ),
        "resolver_mean_pairwise_accuracy": statistics.mean(
            t["pairwise_accuracy"] for t in traces
        ),
        "resolver_mean_first_unique": statistics.mean(
            t["first_unique"] for t in traces
        ),
        "resolver_mean_return_consistency": statistics.mean(
            t["return_consistency"] for t in traces
        ),
        "resolver_mean_final_support_consistency": statistics.mean(
            t["final_support_consistency"] for t in traces
        ),
        "resolver_mean_prototype_samples": statistics.mean(
            t["prototype_samples"] for t in traces
        ),
        "resolver_mean_knn_votes": statistics.mean(
            t["knn_votes"] for t in traces
        ),
        "blind_50_percent_exact": all(
            abs(r["final_balanced_accuracy"] - .5) < 1e-12
            for r in lookup["no_context_replay"].values()
        ),
    }


def run(settings: Settings, output: Path) -> dict:
    settings.validate()
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    selected, history = select_threshold(settings)
    split_records = []
    all_traces = []
    for label, start, count in (
        ("main", settings.test_start, settings.test_seeds),
        ("few", settings.few_start, settings.few_seeds),
        ("shifted", settings.shifted_start, settings.shifted_seeds),
    ):
        rows, traces = [], []
        for seed in range(start, start + count):
            data = task_world(seed, settings, label)
            trace = trace_resolver(data, settings, selected)
            traces.append(trace)
            all_traces.append((label, seed, trace))
            arms = [train_arm(seed, settings, data, trace, arm) for arm in ARMS]
            rows.extend(arms)
            print(f"M3.4 FINAL {label} seed={seed} "
                  f"inference={trace['pairwise_accuracy']:.3f} "
                  f"memory={arms[0]['final_balanced_accuracy']:.3f} "
                  f"shuffled={arms[0]['shuffled_support_accuracy']:.3f}", flush=True)
        split_records.append((label, rows, traces))
    summary = {
        "protocol": "M3.4 preregistered support-observation inferred contexts",
        "settings": asdict(settings),
        "selected_threshold": selected,
        "meta_search": history,
        "main": summarize(split_records[0][1], split_records[0][2],
                          settings, "main"),
        "few": summarize(split_records[1][1], split_records[1][2],
                         settings, "few"),
        "shifted": summarize(split_records[2][1], split_records[2][2],
                             settings, "shifted"),
        "primary": "main paired final balanced AUC inference FIFO vs same-model shuffled final support",
        "required_checks": "main >0.5 no-support, return consistency >0.65, first-visit distinct mean>=3.5",
        "wall_seconds_including_meta_search": time.perf_counter() - started,
        "limitations": [
            "Online context inference is an externally coded kNN resolver, not neural discovery",
            "Labelled support is available; zero-support context is logically ambiguous",
            "Paired opposite labels guarantee exactly 50% zero-context accuracy",
            "The oracle arm sees privileged latent IDs; it is a reference, not deployable",
            "Replay replaces 1/4 of SGD batches; not equal exposure to current task",
            "Prototype and replay memory add compute/storage absent from oracle",
            "Final support permutation is an information intervention, not a trained policy",
            "Few-shot/shifted support includes one deliberately flipped label",
            "Shifted task families differ; not open-ended generalization",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output / "seed_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        cols = ("split", "seed", "arm", "final_balanced_accuracy",
                "worst_context_accuracy", "final_peak_forgetting",
                "mean_learning_auc", "mean_seen_accuracy", "mean_forgetting",
                "return_auc", "replay_updates", "memory_stored",
                "shuffled_support_accuracy", "no_support_accuracy",
                "final_accuracies")
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for label, rows, _ in split_records:
            for row in rows:
                writer.writerow({k: label if k == "split" else
                                 json.dumps(row[k]) if k == "final_accuracies"
                                 else row[k] for k in cols})
    with (output / "episode_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        cols = ("split", "seed", "arm", "episode", "latent_audit_only",
                "novel", "assigned_slot", "auc", "final_episode_accuracy",
                "seen_accuracy", "forgetting", "replay_updates_cumulative")
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for label, rows, _ in split_records:
            for row in rows:
                writer.writerows({**item, "split": label} for item in row["episodes"])
    with (output / "inference_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        cols = ("split", "seed", "first_unique", "return_consistency",
                "final_support_consistency", "pairwise_accuracy",
                "prototype_samples", "knn_votes", "episode_slots",
                "final_slots", "shuffled_final_slots")
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for label, seed, trace in all_traces:
            writer.writerow({
                k: label if k == "split" else seed if k == "seed" else
                json.dumps(trace[k]) if k in (
                    "episode_slots", "final_slots", "shuffled_final_slots"
                ) else trace[k]
                for k in cols
            })
    print("M3.4 PRIMARY", summary["main"]["comparisons"]["support_vs_shuffled"],
          flush=True)
    print("M3.4 REPLAY", summary["main"]["comparisons"]["fifo_vs_no_replay"],
          flush=True)
    print("M3.4 FEWSHOT", summary["few"]["comparisons"]["support_vs_shuffled"],
          flush=True)
    print("M3.4 SHIFTED", summary["shifted"]["comparisons"]["support_vs_shuffled"],
          flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = Settings()
    for field in fields(Settings):
        default = getattr(defaults, field.name)
        parser.add_argument("--" + field.name.replace("_", "-"),
                            type=type(default), default=default)
    parser.add_argument("--out", type=Path, default=Path("results/m3_4"))
    args = parser.parse_args()
    settings = Settings(**{field.name: getattr(args, field.name)
                           for field in fields(Settings)})
    run(settings, args.out)


if __name__ == "__main__":
    main()
