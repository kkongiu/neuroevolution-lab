"""M3.3 — Contextually compatible continual learning, replay, renewal.

Four observable contexts provide logically opposing labels for identical x,y
points. A context-blind classifier has an exact 50% balanced-accuracy ceiling
on paired holdout panels. Policies are externally evolved numeric thresholds,
not autonomous self-rewriting programs.
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
from .plasticity import paired_bootstrap, sigmoid, curve_auc

ARMS = (
    "cue_replay_no_reset", "blind_replay_no_reset", "cue_no_replay_no_reset",
    "adaptive", "accuracy_only", "time_only", "uniform", "shuffled",
    "random_units",
)
MATCHED = ARMS[3:]
POLICIES = ("adaptive", "accuracy_only", "time_only")
Sample = tuple[tuple[float, ...], int]


@dataclass(frozen=True)
class Settings:
    train_start: int = 20000
    train_seeds: int = 4
    validation_start: int = 21000
    validation_seeds: int = 2
    test_start: int = 22000
    test_seeds: int = 24
    shifted_start: int = 23000
    shifted_seeds: int = 12
    contexts: int = 4
    tasks: int = 12
    steps: int = 64
    batch_size: int = 4
    width: int = 20
    eval_size: int = 64
    report_every: int = 16
    learning_rate: float = 0.04
    buffer_size: int = 24
    replay_every: int = 4
    renew_units: int = 3
    quota: int = 3
    generations: int = 3
    population: int = 6
    search_seed: int = 971
    memory_weight: float = 0.20

    def validate(self) -> None:
        values = (self.train_seeds, self.validation_seeds, self.test_seeds,
                  self.shifted_seeds, self.contexts, self.tasks, self.steps,
                  self.batch_size, self.width, self.eval_size, self.report_every,
                  self.buffer_size, self.replay_every, self.renew_units,
                  self.quota, self.generations, self.population)
        if min(values) < 1 or self.contexts != 4 or self.tasks < 6:
            raise ValueError("Need four contexts, at least six tasks, positive sizes")
        if self.population < 4 or self.width <= self.renew_units:
            raise ValueError("Population too small or invalid renewal count")
        if self.tasks - 1 <= self.quota or self.steps % self.report_every:
            raise ValueError("Need optional renewal timing and measurable SGD curve")
        if self.replay_every > self.steps or not 0 < self.learning_rate <= 0.5:
            raise ValueError("Invalid replay frequency or learning rate")
        if not 0 <= self.memory_weight <= 1:
            raise ValueError("Invalid memory weight")
        groups = (
            (self.train_start, self.train_seeds),
            (self.validation_start, self.validation_seeds),
            (self.test_start, self.test_seeds),
            (self.shifted_start, self.shifted_seeds),
        )
        for i, (start, count) in enumerate(groups):
            if start < 0:
                raise ValueError("Seed starts must be nonnegative")
            for next_start, next_count in groups[i + 1:]:
                if set(range(start, start + count)) & set(range(next_start, next_start + next_count)):
                    raise ValueError("Overlap between meta-train/validation/final seeds")


@dataclass
class ContextNetwork:
    w_in: list[list[float]]
    b_hidden: list[float]
    w_out: list[float]
    b_out: float

    @classmethod
    def create(cls, seed: int, settings: Settings) -> ContextNetwork:
        rng = random.Random(seed * 13_019 + 31)
        return cls(
            [[rng.gauss(0, 0.55) for _ in range(2 + settings.contexts)]
             for _ in range(settings.width)],
            [0.] * settings.width,
            [rng.gauss(0, 0.28) for _ in range(settings.width)],
            0.,
        )

    def hidden_values(self, x: tuple[float, ...]) -> list[float]:
        return [max(0., sum(w * v for w, v in zip(row, x)) + bias)
                for row, bias in zip(self.w_in, self.b_hidden)]

    def predict(self, x: tuple[float, ...]) -> float:
        h = self.hidden_values(x)
        return sigmoid(sum(w * v for w, v in zip(self.w_out, h)) + self.b_out)

    def train_batch(self, batch: list[Sample], rate: float) -> None:
        dim = len(self.w_in[0])
        width = len(self.w_out)
        din = [[0.] * dim for _ in range(width)]
        dh = [0.] * width
        dout = [0.] * width
        db = 0.
        for x, label in batch:
            h = self.hidden_values(x)
            err = sigmoid(sum(w * v for w, v in zip(self.w_out, h)) +
                          self.b_out) - label
            db += err
            for i in range(width):
                dout[i] += err * h[i]
                if h[i] > 0.:
                    d = err * self.w_out[i]
                    for k in range(dim):
                        din[i][k] += d * x[k]
                    dh[i] += d
        scale = rate / len(batch)
        self.b_out -= scale * max(-3., min(3., db))
        for i in range(width):
            self.w_out[i] -= scale * max(-3., min(3., dout[i]))
            self.b_hidden[i] -= scale * max(-3., min(3., dh[i]))
            for k in range(dim):
                self.w_in[i][k] -= scale * max(-3., min(3., din[i][k]))

    def renew(self, count: int, inputs: list[Sample], seed: int,
              random_units: bool = False) -> list[int]:
        if not inputs or not 0 < count < len(self.w_out):
            raise ValueError("Renewal needs bounded size and training inputs")
        rng = random.Random(seed)
        if random_units:
            ids = rng.sample(range(len(self.w_out)), count)
        else:
            activation = [0.] * len(self.w_out)
            for x, _ in inputs:
                for i, val in enumerate(self.hidden_values(x)):
                    activation[i] += val
            ids = sorted(range(len(activation)), key=lambda i: (activation[i], i))[:count]
        for i in ids:
            self.w_in[i] = [rng.gauss(0, 0.55) for _ in self.w_in[i]]
            self.b_hidden[i] = 0.
            self.w_out[i] = rng.gauss(0, 0.28)
        return ids


def mask(samples: list[Sample], blind: bool) -> list[Sample]:
    if not blind:
        return samples
    return [(x[:2] + (0., 0., 0., 0.), label) for x, label in samples]


def acc(net: ContextNetwork, samples: list[Sample]) -> float:
    return sum(int((net.predict(x) >= 0.5) == label) for x, label in samples) / len(samples)


def encode(x: float, y: float, context: int) -> tuple[float, ...]:
    return (x, y) + tuple(float(i == context) for i in range(4))


def world(seed: int, settings: Settings, shifted: bool = False) -> dict:
    """Paired context panels use identical x,y and complementary labels."""
    rng = random.Random(seed * 10_019 + 91)
    task_types = ("stripe", "checker") if shifted else ("line", "xor")
    rules = []
    for kind in task_types:
        rules.append(WorldTask(
            kind=kind, angle=rng.uniform(-math.pi, math.pi),
            shift=rng.uniform(-0.18, 0.18),
            phase=rng.uniform(-math.pi, math.pi) if kind in ("stripe", "checker")
                  else rng.uniform(-0.18, 0.18),
        ))

    def label(context: int, xy: tuple[float, float]) -> int:
        base = rules[context // 2].label(xy)
        return base ^ (context % 2)

    panel: dict[int, list[Sample]] = {}
    for pair in range(2):
        panel_rng = random.Random(seed * 1_000_003 + 847 * pair + 909)
        xy_points = [(panel_rng.uniform(-1., 1.), panel_rng.uniform(-1., 1.))
                     for _ in range(settings.eval_size)]
        for context in (2 * pair, 2 * pair + 1):
            panel[context] = [(encode(*xy, context), label(context, xy))
                              for xy in xy_points]

    contexts = list(range(4))
    rng.shuffle(contexts)
    contexts.extend(rng.randrange(4) for _ in range(settings.tasks - 4))
    episodes = []
    for idx, context in enumerate(contexts):
        samples_rng = random.Random(seed * 2_000_003 + idx * 31 + 111)
        training = []
        for _ in range(settings.steps * settings.batch_size):
            x, y = samples_rng.uniform(-1., 1.), samples_rng.uniform(-1., 1.)
            training.append((encode(x, y, context), label(context, (x, y))))
        episodes.append({
            "context": context, "novel": context not in contexts[:idx],
            "train": training,
        })
    return {"episodes": episodes, "panel": panel}


@dataclass(frozen=True)
class Policy:
    weights: tuple[float, float, float, float, float]

    def decide(self, features: tuple[float, ...], mode: str) -> bool:
        if len(features) != 5 or mode not in POLICIES:
            raise ValueError("Bad policy observation or mode")
        active = (
            (0, 1, 2, 3, 4) if mode == "adaptive" else
            (0, 1, 4) if mode == "accuracy_only" else (0, 4)
        )
        return sum(self.weights[i] * features[i] for i in active) > 0.


def features(net: ContextNetwork, current: list[Sample],
             buffer: dict[int, list[Sample]], previous: list[Sample],
             index: int, settings: Settings) -> tuple[float, ...]:
    """Only training examples, never test panels or hidden novelty flags."""
    new_probe = current[:32]
    other_probes = [sample for samples in buffer.values()
                    for sample in samples]
    retention = acc(net, other_probes) if other_probes else 0.5
    activity = [0.] * len(net.w_out)
    probe = previous[-min(32, len(previous)):]
    for x, _ in probe:
        for j, v in enumerate(net.hidden_values(x)):
            activity[j] += v / len(probe)
    inactive = sum(v < 0.025 for v in activity) / len(activity)
    return (1., 1. - acc(net, new_probe), 1. - retention,
            inactive, index / (settings.tasks - 1))


def uniform_schedule(settings: Settings) -> list[bool]:
    n = settings.tasks - 1
    return [((i + 1) * settings.quota // n > i * settings.quota // n)
            for i in range(n)]


def simulate(seed: int, settings: Settings, data: dict, arm: str,
             policy: Policy | None = None,
             plan: list[bool] | None = None) -> dict:
    if arm not in ARMS:
        raise ValueError(arm)
    if arm in POLICIES and policy is None:
        raise ValueError("Learned policy missing")
    if arm in ("uniform", "shuffled", "random_units"):
        if plan is None or len(plan) != settings.tasks - 1 or sum(plan) != settings.quota:
            raise ValueError("Invalid fixed quota intervention plan")

    net = ContextNetwork.create(seed, settings)
    blind = arm == "blind_replay_no_reset"
    replay = arm != "cue_no_replay_no_reset"
    panel = {ctx: mask(samples, blind) for ctx, samples in data["panel"].items()}
    buffer: dict[int, list[Sample]] = {}
    peak: dict[int, float] = {}
    seen: set[int] = set()
    rows = []
    actions, requests = [], []
    remaining = settings.quota
    forced = 0
    sensitive = 0
    replay_batches = 0
    previous = []
    for index, episode in enumerate(data["episodes"]):
        context = episode["context"]
        current = mask(episode["train"], blind)
        renewed = 0
        request = False
        chosen = False
        forced_action = False
        if index:
            if arm in POLICIES:
                obs = features(net, current, buffer, previous, index, settings)
                request = policy.decide(obs, arm)
                remaining_boundaries = settings.tasks - index
                must = remaining >= remaining_boundaries
                chosen = (request or must) and remaining > 0
                forced_action = chosen != request
                forced += int(forced_action)
                if arm == "adaptive":
                    blind_obs = (obs[0], 0., 0., 0., obs[4])
                    sensitive += int(request != policy.decide(blind_obs, "adaptive")
                                     and not forced_action)
                if chosen:
                    remaining -= 1
                requests.append(int(request))
            elif arm in ("uniform", "shuffled", "random_units"):
                chosen = plan[index - 1]
            actions.append(int(chosen))
            if chosen:
                ids = net.renew(
                    settings.renew_units, previous[-min(64, len(previous)):],
                    seed * 349_019 + index * 701 + 23,
                    random_units=(arm == "random_units"),
                )
                renewed = len(ids)

        curve = [(0, acc(net, panel[context]))]
        for step in range(1, settings.steps + 1):
            start = (step - 1) * settings.batch_size
            batch = current[start:start + settings.batch_size]
            alternatives = [c for c in sorted(buffer) if c != context]
            if replay and alternatives and step % settings.replay_every == 0:
                chosen_context = alternatives[(seed + index * 7 + step * 13) %
                                              len(alternatives)]
                stored = buffer[chosen_context]
                batch = [stored[(index * 43 + step * settings.batch_size + j) %
                                len(stored)] for j in range(settings.batch_size)]
                replay_batches += 1
            net.train_batch(batch, settings.learning_rate)
            if step % settings.report_every == 0:
                curve.append((step, acc(net, panel[context])))

        seen.add(context)
        after = {ctx: acc(net, panel[ctx]) for ctx in sorted(seen)}
        forgetting = statistics.mean(
            max(0., peak[ctx] - after[ctx]) for ctx in peak
        ) if peak else 0.
        mean_seen = statistics.mean(after.values())
        for ctx, score in after.items():
            peak[ctx] = max(peak.get(ctx, 0.), score)
        rows.append({
            "seed": seed, "arm": arm, "episode": index, "context": context,
            "novel": episode["novel"],
            "learning_auc": curve_auc(curve, settings.steps),
            "start_accuracy": curve[0][1],
            "end_accuracy": curve[-1][1],
            "seen_accuracy": mean_seen,
            "forgetting": forgetting,
            "renewal": int(chosen), "renewed_units": renewed,
            "replay_batches": replay_batches,
        })
        # Captures training data only; identical across arms for this seed.
        buffer[context] = current[:settings.buffer_size]
        previous = current

    if arm in MATCHED and (sum(actions) != settings.quota or
                            sum(r["renewed_units"] for r in rows) !=
                            settings.quota * settings.renew_units):
        raise AssertionError("Not an exactly matched renewal budget")
    if arm in POLICIES and remaining:
        raise AssertionError("Learned policy did not spend intervention quota")
    if len(seen) != 4:
        raise AssertionError("All contexts must have been presented")

    final_context = [acc(net, panel[ctx]) for ctx in range(4)]
    return {
        "seed": seed, "arm": arm,
        "final_balanced_accuracy": statistics.mean(final_context),
        "final_worst_context_accuracy": min(final_context),
        "final_context_accuracies": final_context,
        "final_peak_forgetting": statistics.mean(
            max(0., peak[ctx] - final_context[ctx]) for ctx in range(4)
        ),
        "learning_auc": statistics.mean(r["learning_auc"] for r in rows[1:]),
        "mean_seen_accuracy": statistics.mean(r["seen_accuracy"] for r in rows[1:]),
        "mean_forgetting": statistics.mean(r["forgetting"] for r in rows[1:]),
        "renewal_actions": sum(actions),
        "renewed_units": sum(r["renewed_units"] for r in rows),
        "forced": forced, "unforced_state_sensitive": sensitive,
        "requests": requests, "actions": actions,
        "replay_batches": replay_batches, "rows": rows,
    }


def project(weights: tuple[float, ...], mode: str) -> Policy:
    if mode == "time_only":
        return Policy((weights[0], 0., 0., 0., weights[4]))
    if mode == "accuracy_only":
        return Policy((weights[0], weights[1], 0., 0., weights[4]))
    return Policy(tuple(weights))


def search(settings: Settings, mode: str, cache: dict[int, dict]) -> tuple[Policy, list[dict]]:
    train_seeds = list(range(settings.train_start,
                             settings.train_start + settings.train_seeds))
    validation_seeds = list(range(settings.validation_start,
                                  settings.validation_start + settings.validation_seeds))
    rng = random.Random(settings.search_seed + POLICIES.index(mode) * 73)
    population = [
        project((-3., 0., 0., 0., 0.), mode),
        project((3., 0., 0., 0., 0.), mode),
        project((-.2, 2., 1., 0., -.4), mode),
        project((.2, -1.5, 0., 1.2, .3), mode),
    ]
    while len(population) < settings.population:
        population.append(project(tuple(rng.uniform(-1.5, 1.5) for _ in range(5)), mode))

    def score(p: Policy, seeds: list[int]) -> float:
        outcomes = [simulate(seed, settings, cache[seed], mode, p) for seed in seeds]
        return statistics.mean(
            result["learning_auc"] + settings.memory_weight * result["mean_seen_accuracy"]
            for result in outcomes
        )
    history, finalists = [], []
    for generation in range(settings.generations):
        ranked = sorted(((score(p, train_seeds), p) for p in population),
                        key=lambda x: x[0], reverse=True)
        champion = ranked[0][1]
        validation = score(champion, validation_seeds)
        finalists.append((validation, champion))
        history.append({
            "mode": mode, "generation": generation,
            "train_score": ranked[0][0], "validation_score": validation,
            "weights": list(champion.weights),
        })
        print(f"M3.3 meta {mode} gen={generation} "
              f"train={ranked[0][0]:.5f} val={validation:.5f}", flush=True)
        elite = [p for _, p in ranked[:max(2, settings.population // 3)]]
        population = list(elite)
        while len(population) < settings.population:
            parent = rng.choice(elite)
            weights = tuple(max(-5., min(5., w + rng.gauss(0., .55)))
                            for w in parent.weights)
            population.append(project(weights, mode))
    return max(finalists, key=lambda x: x[0])[1], history


def summarize(results: list[dict], settings: Settings, label: str) -> dict:
    lookup = {arm: {r["seed"]: r for r in results if r["arm"] == arm}
              for arm in ARMS}
    metrics = ("final_balanced_accuracy", "final_worst_context_accuracy",
               "final_peak_forgetting", "learning_auc",
               "mean_seen_accuracy", "mean_forgetting", "renewal_actions",
               "replay_batches")
    means = {arm: {
        key: statistics.mean(r[key] for r in records.values())
        for key in metrics
    } for arm, records in lookup.items()}
    comparisons = {}
    contrasts = (
        ("cue_replay_no_reset", "blind_replay_no_reset", "context_vs_blind"),
        ("cue_replay_no_reset", "cue_no_replay_no_reset", "replay_vs_no_replay"),
        ("adaptive", "time_only", "adaptive_vs_time_only"),
        ("adaptive", "accuracy_only", "adaptive_vs_accuracy_only"),
        ("adaptive", "shuffled", "adaptive_vs_shuffled"),
        ("adaptive", "uniform", "adaptive_vs_uniform"),
        ("adaptive", "random_units", "adaptive_vs_random_units"),
    )
    for a, b, name in contrasts:
        metric = "final_balanced_accuracy" if name in (
            "context_vs_blind", "replay_vs_no_replay") else "learning_auc"
        diffs = [lookup[a][seed][metric] - lookup[b][seed][metric]
                 for seed in sorted(lookup[a])]
        comparisons[name] = {
            "metric": metric,
            "mean_difference": statistics.mean(diffs),
            "paired_bootstrap_95": list(paired_bootstrap(diffs, min(lookup[a]) + 71)),
            "positive_seeds": sum(x > 0 for x in diffs),
            "negative_seeds": sum(x < 0 for x in diffs),
            "zero_seeds": sum(x == 0 for x in diffs),
        }
    adaptive = list(lookup["adaptive"].values())
    return {
        "label": label,
        "seeds": len(adaptive),
        "means": means,
        "comparisons": comparisons,
        "blind_accuracy_50_percent_for_all_seeds":
            all(abs(r["final_balanced_accuracy"] - 0.5) < 1e-12
                for r in lookup["blind_replay_no_reset"].values()),
        "adaptive_forced_total": sum(r["forced"] for r in adaptive),
        "adaptive_unforced_state_sensitive": sum(
            r["unforced_state_sensitive"] for r in adaptive
        ),
        "adaptive_decision_opportunities": len(adaptive) * (settings.tasks - 1),
    }


def run(settings: Settings, output: Path) -> dict:
    settings.validate()
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    cache = {
        seed: world(seed, settings)
        for seed in list(range(settings.train_start,
                               settings.train_start + settings.train_seeds)) +
                    list(range(settings.validation_start,
                               settings.validation_start + settings.validation_seeds))
    }
    policies, training_history = {}, []
    for mode in POLICIES:
        p, trace = search(settings, mode, cache)
        policies[mode] = p
        training_history.extend(trace)
        print(f"SELECTED {mode}: {p.weights}", flush=True)
    all_results: list[tuple[str, list[dict]]] = []
    for label, first, count, shifted in (
        ("main", settings.test_start, settings.test_seeds, False),
        ("shifted", settings.shifted_start, settings.shifted_seeds, True),
    ):
        records = []
        for seed in range(first, first + count):
            data = world(seed, settings, shifted)
            adaptive = simulate(seed, settings, data, "adaptive", policies["adaptive"])
            actions = list(adaptive["actions"])
            shuffled = list(actions)
            random.Random(seed * 417 + 17).shuffle(shuffled)
            by_arm = {
                "cue_replay_no_reset": simulate(seed, settings, data, "cue_replay_no_reset"),
                "blind_replay_no_reset": simulate(seed, settings, data, "blind_replay_no_reset"),
                "cue_no_replay_no_reset": simulate(seed, settings, data, "cue_no_replay_no_reset"),
                "adaptive": adaptive,
                "accuracy_only": simulate(seed, settings, data, "accuracy_only",
                                          policies["accuracy_only"]),
                "time_only": simulate(seed, settings, data, "time_only",
                                      policies["time_only"]),
                "uniform": simulate(seed, settings, data, "uniform",
                                    plan=uniform_schedule(settings)),
                "shuffled": simulate(seed, settings, data, "shuffled", plan=shuffled),
                "random_units": simulate(seed, settings, data, "random_units",
                                         plan=actions),
            }
            if any(by_arm[a]["renewal_actions"] != settings.quota for a in MATCHED):
                raise AssertionError("Dose mismatch")
            if abs(by_arm["blind_replay_no_reset"]["final_balanced_accuracy"] - 0.5) > 1e-12:
                raise AssertionError("Context-blind mathematical ceiling violated")
            records.extend(by_arm[a] for a in ARMS)
            print(f"M3.3 TEST {label} seed={seed} cue={by_arm['cue_replay_no_reset']['final_balanced_accuracy']:.4f} "
                  f"blind={by_arm['blind_replay_no_reset']['final_balanced_accuracy']:.4f} "
                  f"adaptive={adaptive['learning_auc']:.4f}", flush=True)
        all_results.append((label, records))

    summary = {
        "protocol": "M3.3 contextual compatible lifelong tasks, replay and equal-dose renewal",
        "settings": asdict(settings),
        "policies": {mode: list(p.weights) for mode, p in policies.items()},
        "meta_history": training_history,
        "main": summarize(all_results[0][1], settings, "main"),
        "shifted": summarize(all_results[1][1], settings, "shifted"),
        "preregistered_primary": "main final balanced context accuracy: cue+replay-noreset vs blind+replay-noreset",
        "adaptive_secondary": "main episode AUC: adaptive vs independent time-only (and independent accuracy-only)",
        "wall_seconds_including_meta_search": time.perf_counter() - start,
        "interpretation_limits": [
            "Explicit one-hot context ID supplied by the environment, not discovered",
            "Opposing labels make 50% balanced context-blind limit tautological",
            "Replay replaces current-task SGD batches: equal compute not equal current sample exposure",
            "Numeric renewal policies externally evolved, not self-modifying code",
            "Shifted family requires relearning different tasks, not new universal intelligence",
            "Context-specific learning may remain imperfect with a small ReLU network",
            "Heldout panels are only used for meta-fitness on train/validation seed splits",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output / "seed_metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        cols = ("split", "seed", "arm", "final_balanced_accuracy",
                "final_worst_context_accuracy", "final_peak_forgetting",
                "learning_auc", "mean_seen_accuracy", "mean_forgetting",
                "renewal_actions", "renewed_units", "replay_batches",
                "forced", "unforced_state_sensitive", "final_context_accuracies",
                "actions")
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for label, records in all_results:
            for rec in records:
                writer.writerow({
                    k: (json.dumps(rec[k]) if k in ("final_context_accuracies", "actions")
                        else label if k == "split" else rec[k])
                    for k in cols
                })
    with (output / "episode_metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        cols = ("split", "seed", "arm", "episode", "context", "novel",
                "learning_auc", "start_accuracy", "end_accuracy",
                "seen_accuracy", "forgetting", "renewal", "renewed_units",
                "replay_batches")
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for label, records in all_results:
            for rec in records:
                writer.writerows({**r, "split": label} for r in rec["rows"])
    print("M3.3 PRIMARY", summary["main"]["comparisons"]["context_vs_blind"], flush=True)
    print("M3.3 REPLAY", summary["main"]["comparisons"]["replay_vs_no_replay"], flush=True)
    print("M3.3 ADAPTIVE", summary["main"]["comparisons"]["adaptive_vs_time_only"], flush=True)
    print("M3.3 SHIFTED", summary["shifted"]["comparisons"]["context_vs_blind"], flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = Settings()
    for fld in fields(Settings):
        default = getattr(defaults, fld.name)
        parser.add_argument("--" + fld.name.replace("_", "-"),
                            type=type(default), default=default)
    parser.add_argument("--out", type=Path, default=Path("results/m3_3"))
    opts = parser.parse_args()
    settings = Settings(**{fld.name: getattr(opts, fld.name) for fld in fields(Settings)})
    run(settings, opts.out)


if __name__ == "__main__":
    main()
