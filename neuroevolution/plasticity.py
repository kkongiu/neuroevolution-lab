"""M1-P: CPU-only continual-learning plasticity experiment.

This experiment modifies weights within each agent's lifetime (unlike M1's
between-generation neuroevolution). It is not agent-directed source rewriting.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

ARMS = ("continual", "fresh", "low_activity_reset", "random_reset")


@dataclass(frozen=True)
class Settings:
    seed_start: int = 100
    seeds: int = 8
    tasks: int = 8
    steps: int = 128
    batch_size: int = 2
    width: int = 24
    eval_size: int = 96
    report_every: int = 32
    learning_rate: float = 0.04
    reset_fraction: float = 0.20

    def validate(self) -> None:
        if self.seed_start < 0 or min(self.seeds, self.tasks, self.steps, self.batch_size,
                                      self.width, self.eval_size, self.report_every) < 1:
            raise ValueError("Seed start must be nonnegative; counts must be positive")
        if self.tasks < 2 or self.width < 2 or self.steps % self.report_every:
            raise ValueError("Need >=2 tasks, >=2 hidden units and divisible evaluation steps")
        if not (0 < self.learning_rate <= 0.5) or not (0 < self.reset_fraction < 1):
            raise ValueError("Learning rate must be in (0, 0.5], reset fraction in (0, 1)")


@dataclass(frozen=True)
class Task:
    kind: str
    angle: float
    shift: float
    phase: float

    def label(self, xy: tuple[float, float]) -> int:
        x, y = xy
        if self.kind == "line":
            return int(x * math.cos(self.angle) + y * math.sin(self.angle) > self.shift)
        if self.kind == "xor":
            return int((x > self.shift) != (y > self.phase))
        if self.kind == "circle":
            return int((x - self.shift) ** 2 + (y - self.phase) ** 2 > 0.56 ** 2)
        if self.kind == "stripe":
            return int(math.sin(3.0 * (x * math.cos(self.angle) +
                                       y * math.sin(self.angle)) + self.phase) > 0)
        raise ValueError(self.kind)


def make_task(seed: int, index: int) -> Task:
    rng = random.Random(100_003 * seed + 109 * index + 17)
    return Task(
        kind=("line", "xor", "circle", "stripe")[index % 4],
        angle=rng.uniform(-math.pi, math.pi),
        shift=rng.uniform(-0.23, 0.23),
        phase=rng.uniform(-0.22, 0.22) if index % 4 in (1, 2)
              else rng.uniform(-math.pi, math.pi),
    )


def examples(task: Task, seed: int, count: int) -> list[tuple[tuple[float, float], int]]:
    rng = random.Random(seed)
    return [
        ((x, y), task.label((x, y)))
        for x, y in ((rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in range(count))
    ]


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1 + exp_value)


@dataclass
class Network:
    """One-hidden-layer ReLU classifier with explicit, inspectable SGD updates."""

    w_in: list[list[float]]
    b_hidden: list[float]
    w_out: list[float]
    b_out: float

    @classmethod
    def create(cls, rng: random.Random, width: int) -> Network:
        return cls(
            w_in=[[rng.gauss(0, 0.85), rng.gauss(0, 0.85)]
                  for _ in range(width)],
            b_hidden=[0.0] * width,
            w_out=[rng.gauss(0, 0.28) for _ in range(width)],
            b_out=0.0,
        )

    def hidden_values(self, x: tuple[float, float]) -> list[float]:
        return [
            max(0.0, weights[0] * x[0] + weights[1] * x[1] + bias)
            for weights, bias in zip(self.w_in, self.b_hidden)
        ]

    def predict(self, x: tuple[float, float]) -> float:
        h = self.hidden_values(x)
        return sigmoid(sum(w * v for w, v in zip(self.w_out, h)) + self.b_out)

    def train_batch(self, batch: list[tuple[tuple[float, float], int]], rate: float) -> None:
        width = len(self.w_out)
        d_in = [[0.0, 0.0] for _ in range(width)]
        d_hidden = [0.0] * width
        d_out = [0.0] * width
        d_bias = 0.0
        for x, target in batch:
            hidden = self.hidden_values(x)
            error = sigmoid(sum(w * v for w, v in zip(self.w_out, hidden)) +
                            self.b_out) - target
            d_bias += error
            for i in range(width):
                d_out[i] += error * hidden[i]
                if hidden[i] > 0:
                    delta = error * self.w_out[i]
                    d_in[i][0] += delta * x[0]
                    d_in[i][1] += delta * x[1]
                    d_hidden[i] += delta
        scale = rate / len(batch)
        self.b_out -= scale * max(-3, min(3, d_bias))
        for i in range(width):
            self.w_out[i] -= scale * max(-3, min(3, d_out[i]))
            self.w_in[i][0] -= scale * max(-3, min(3, d_in[i][0]))
            self.w_in[i][1] -= scale * max(-3, min(3, d_in[i][1]))
            self.b_hidden[i] -= scale * max(-3, min(3, d_hidden[i]))

    def renew(self, count: int, rng: random.Random,
              probe: list[tuple[float, float]], low_activity: bool) -> list[int]:
        """Replace exactly count units, chosen from *past training inputs* only.

        A random-reset control replaces the same number of units.
        """
        width = len(self.w_out)
        if not (0 < count < width) or not probe:
            raise ValueError("Invalid renewal count or missing probe")
        if low_activity:
            avg = [0.0] * width
            for x in probe:
                for i, v in enumerate(self.hidden_values(x)):
                    avg[i] += v
            selected = sorted(range(width), key=lambda i: (avg[i], i))[:count]
        else:
            selected = rng.sample(range(width), count)
        for i in selected:
            self.w_in[i] = [rng.gauss(0, 0.85), rng.gauss(0, 0.85)]
            self.b_hidden[i] = 0.0
            self.w_out[i] = rng.gauss(0, 0.28)
        return selected


def accuracy(net: Network, samples: list[tuple[tuple[float, float], int]]) -> float:
    return sum(int((net.predict(x) >= 0.5) == y) for x, y in samples) / len(samples)


def dead_fraction(net: Network, samples: list[tuple[tuple[float, float], int]]) -> float:
    totals = [0.0] * len(net.w_out)
    for x, _ in samples:
        for i, v in enumerate(net.hidden_values(x)):
            totals[i] += v
    return sum(v <= 1e-9 for v in totals) / len(totals)


def curve_auc(points: list[tuple[int, float]], steps: int) -> float:
    return sum(
        (b[0] - a[0]) * (a[1] + b[1]) / 2.0
        for a, b in zip(points, points[1:])
    ) / steps


def paired_bootstrap(diffs: list[float], seed: int) -> tuple[float, float]:
    """Descriptive percentile interval; seed is fixed for reproducibility."""
    if not diffs:
        raise ValueError("No paired differences")
    if len(diffs) == 1:
        return diffs[0], diffs[0]
    rng = random.Random(seed + 9_172)
    samples = []
    for _ in range(2000):
        samples.append(statistics.mean(rng.choice(diffs) for _ in diffs))
    samples.sort()
    return samples[49], samples[1950]


def svg_chart(rows: list[dict], task_count: int) -> str:
    colors = {
        "continual": "#446cae",
        "fresh": "#666666",
        "low_activity_reset": "#21936e",
        "random_reset": "#bb6a24",
    }
    paths = []
    for arm in ARMS:
        pts = []
        for task_index in range(task_count):
            values = [row["auc"] for row in rows
                      if row["arm"] == arm and row["task_index"] == task_index]
            val = statistics.mean(values)
            x = 60 + 580 * task_index / max(1, task_count - 1)
            y = 305 - 245 * val
            pts.append(f"{x:.1f},{y:.1f}")
        paths.append(
            f'<polyline fill="none" stroke="{colors[arm]}" stroke-width="2.5" '
            f'points="{" ".join(pts)}"/>'
        )
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 410">'
        '<rect width="720" height="410" fill="white"/>'
        '<text x="40" y="30" font-size="18">Mean within-task learning AUC</text>'
        '<line x1="60" y1="60" x2="60" y2="305" stroke="#777"/>'
        '<line x1="60" y1="305" x2="650" y2="305" stroke="#777"/>'
        '<text x="20" y="65" font-size="12">1.0</text>'
        '<text x="20" y="185" font-size="12">0.5</text>'
        '<text x="20" y="310" font-size="12">0.0</text>'
        '<text x="255" y="340" font-size="13">Task index</text>'
        + "".join(paths)
        + "".join(
            f'<text x="{35 + (i % 2) * 325}" y="{365 + (i // 2) * 20}" '
            f'fill="{colors[arm]}" font-size="12">{arm}</text>'
            for i, arm in enumerate(ARMS)
        )
        + "</svg>\n"
    )


def summarize(rows: list[dict], settings: Settings) -> dict:
    per_seed: dict[int, dict[str, dict[str, float]]] = {}
    for seed in range(settings.seed_start, settings.seed_start + settings.seeds):
        per_seed[seed] = {}
        for arm in ARMS:
            selected = [r for r in rows
                        if r["seed"] == seed and r["arm"] == arm
                        and r["task_index"] > 0]
            if len(selected) != settings.tasks - 1:
                raise AssertionError("Missing one or more paired tasks")
            per_seed[seed][arm] = {
                key: statistics.mean(row[key] for row in selected)
                for key in ("auc", "end_accuracy", "learning_gain",
                            "forgetting", "dead_fraction")
            }
    means = {
        arm: {
            metric: statistics.mean(per_seed[s][arm][metric] for s in per_seed)
            for metric in ("auc", "end_accuracy", "learning_gain", "forgetting",
                           "dead_fraction")
        }
        for arm in ARMS
    }
    comparisons = {}
    for arm in ("continual", "low_activity_reset", "random_reset"):
        diffs = [per_seed[s][arm]["auc"] - per_seed[s]["fresh"]["auc"]
                 for s in per_seed]
        comparisons[f"{arm}_minus_fresh_auc"] = {
            "mean_difference": statistics.mean(diffs),
            "bootstrap_95_percentile_interval": list(
                paired_bootstrap(diffs, settings.seed_start)
            ),
            "paired_seed_differences": diffs,
        }
    reset_diffs = [
        per_seed[s]["low_activity_reset"]["auc"] -
        per_seed[s]["random_reset"]["auc"]
        for s in per_seed
    ]
    comparisons["low_activity_minus_random_auc"] = {
        "mean_difference": statistics.mean(reset_diffs),
        "bootstrap_95_percentile_interval": list(
            paired_bootstrap(reset_diffs, settings.seed_start + 71)
        ),
        "paired_seed_differences": reset_diffs,
    }
    return {
        "settings": asdict(settings),
        "arms": list(ARMS),
        "units": "accuracy fractions; AUC normalized 0..1",
        "primary_metric": "per-seed mean AUC over tasks 1..N-1",
        "mean_metrics_over_seed_means": means,
        "comparisons": comparisons,
        "limitations": [
            "Task family and learning-rate settings chosen by experimenter",
            "Bootstrap intervals are exploratory and not validation of a published claim",
            "Fresh control starts new weights each task; continual arms retain their weights",
            "Unit renewal is selected externally, not decided by the neural network",
            "Learning tasks share the same 2D input space; not general intelligence",
        ],
    }


def run(settings: Settings, output: Path) -> dict:
    settings.validate()
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in range(settings.seed_start, settings.seed_start + settings.seeds):
        base = Network.create(random.Random(seed * 13_019 + 31), settings.width)
        networks = {arm: copy.deepcopy(base) for arm in ARMS}
        past_evaluations: list[list[tuple[tuple[float, float], int]]] = []
        past_end: dict[str, list[float]] = {arm: [] for arm in ARMS}
        past_probe: list[tuple[float, float]] = []
        for task_index in range(settings.tasks):
            task = make_task(seed, task_index)
            training = examples(task, seed * 2_000_003 + task_index * 31 + 111,
                                settings.steps * settings.batch_size)
            validation = examples(task, seed * 2_000_003 + task_index * 31 + 999_777,
                                  settings.eval_size)
            batches = [
                training[i * settings.batch_size:(i + 1) * settings.batch_size]
                for i in range(settings.steps)
            ]
            resets = {arm: 0 for arm in ARMS}
            if task_index:
                networks["fresh"] = Network.create(
                    random.Random(seed * 13_019 + task_index * 997 + 31),
                    settings.width,
                )
                count = max(1, min(settings.width - 1,
                                   round(settings.reset_fraction * settings.width)))
                for arm, low_activity in (
                    ("low_activity_reset", True), ("random_reset", False)
                ):
                    chosen = networks[arm].renew(
                        count, random.Random(seed * 349_019 + task_index * 701 + 23),
                        past_probe, low_activity=low_activity,
                    )
                    resets[arm] = len(chosen)
            for arm in ARMS:
                net = networks[arm]
                points = [(0, accuracy(net, validation))]
                for step_index, batch in enumerate(batches, 1):
                    net.train_batch(batch, settings.learning_rate)
                    if step_index % settings.report_every == 0:
                        points.append((step_index, accuracy(net, validation)))
                end_acc = points[-1][1]
                previous = [
                    max(0.0, past - accuracy(net, samples))
                    for past, samples in zip(past_end[arm], past_evaluations)
                ]
                rows.append({
                    "seed": seed, "task_index": task_index, "kind": task.kind,
                    "arm": arm, "start_accuracy": points[0][1],
                    "end_accuracy": end_acc,
                    "learning_gain": end_acc - points[0][1],
                    "auc": curve_auc(points, settings.steps),
                    "forgetting": statistics.mean(previous) if previous else 0.0,
                    "dead_fraction": dead_fraction(net, validation),
                    "reset_units": resets[arm],
                })
                past_end[arm].append(end_acc)
            past_evaluations.append(validation)
            # Only past *training* features can influence renewal choice.
            past_probe = [x for x, _ in training[-min(64, len(training)):]]
        print(f"completed seed={seed}", flush=True)
    columns = ["seed", "task_index", "kind", "arm", "start_accuracy",
               "end_accuracy", "learning_gain", "auc", "forgetting",
               "dead_fraction", "reset_units"]
    with (output / "measurements.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows, settings)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "curves.svg").write_text(svg_chart(rows, settings.tasks), encoding="utf-8")
    print("mean AUC after first task:",
          {arm: round(summary["mean_metrics_over_seed_means"][arm]["auc"], 4)
           for arm in ARMS}, flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag, typ, default in (
        ("seed-start", int, 100), ("seeds", int, 8), ("tasks", int, 8),
        ("steps", int, 128), ("batch-size", int, 2), ("width", int, 24),
        ("eval-size", int, 96), ("report-every", int, 32),
        ("learning-rate", float, 0.04), ("reset-fraction", float, 0.20),
    ):
        parser.add_argument(f"--{flag}", type=typ, default=default)
    parser.add_argument("--out", type=Path, default=Path("results/plasticity"))
    args = parser.parse_args()
    settings = Settings(**{
        name: getattr(args, name) for name in Settings.__dataclass_fields__
    })
    run(settings, args.out)


if __name__ == "__main__":
    main()
