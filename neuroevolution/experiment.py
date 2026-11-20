"""Reproducible, CPU-only neuroevolution experiment. No runtime code execution."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path

INPUTS = 4
OUTPUTS = 2


@dataclass(frozen=True)
class Config:
    seed: int = 42
    generations: int = 12
    population: int = 12
    episodes: int = 3
    steps: int = 120
    hidden: int = 8
    flip_interval: int = 20
    mutations: int = 3

    def validate(self) -> None:
        if self.seed < 0 or self.generations < 1 or self.population < 4:
            raise ValueError("seed >= 0, generations >= 1, population >= 4 required")
        if self.episodes < 1 or self.hidden < 2 or self.mutations < 1:
            raise ValueError("episodes >= 1, hidden >= 2, mutations >= 1 required")
        if not (3 <= self.flip_interval < self.steps):
            raise ValueError("3 <= flip_interval < steps required")


@dataclass
class Genome:
    hidden: int
    wi: list[float]
    wr: list[float]
    wo: list[float]
    bh: list[float]
    bo: list[float]
    mi: list[bool]
    mr: list[bool]
    mo: list[bool]
    active: list[bool]

    @classmethod
    def fresh(cls, rng: random.Random, n: int) -> "Genome":
        def weights(count: int) -> list[float]:
            return [rng.gauss(0.0, 0.5) for _ in range(count)]

        def mask(count: int) -> list[bool]:
            return [rng.random() < 0.70 for _ in range(count)]

        return cls(
            hidden=n,
            wi=weights(n * INPUTS),
            wr=weights(n * n),
            wo=weights(OUTPUTS * n),
            bh=weights(n),
            bo=weights(OUTPUTS),
            mi=mask(n * INPUTS),
            mr=mask(n * n),
            mo=mask(OUTPUTS * n),
            active=[True] * n,
        )

    def forward(self, observation: tuple[float, ...], state: list[float]) -> tuple[int, list[float]]:
        n = self.hidden
        new_state = [0.0] * n
        for i in range(n):
            if not self.active[i]:
                continue
            z = self.bh[i]
            for j in range(INPUTS):
                index = i * INPUTS + j
                if self.mi[index]:
                    z += self.wi[index] * observation[j]
            for j in range(n):
                index = i * n + j
                if self.active[j] and self.mr[index]:
                    z += self.wr[index] * state[j]
            new_state[i] = math.tanh(z)

        logits = list(self.bo)
        for k in range(OUTPUTS):
            for i in range(n):
                index = k * n + i
                if self.active[i] and self.mo[index]:
                    logits[k] += self.wo[index] * new_state[i]
        return int(logits[1] > logits[0]), new_state

    def active_connections(self) -> int:
        n = self.hidden
        incoming = sum(
            self.mi[i * INPUTS + j]
            for i in range(n) if self.active[i] for j in range(INPUTS)
        )
        recurrent = sum(
            self.mr[i * n + j]
            for i in range(n) if self.active[i]
            for j in range(n) if self.active[j]
        )
        outgoing = sum(
            self.mo[k * n + i]
            for k in range(OUTPUTS)
            for i in range(n) if self.active[i]
        )
        return incoming + recurrent + outgoing


def mutate(parent: Genome, rng: random.Random, topology: bool, count: int) -> Genome:
    child = copy.deepcopy(parent)
    weight_groups = [child.wi, child.wr, child.wo, child.bh, child.bo]
    mask_groups = [child.mi, child.mr, child.mo]
    for _ in range(count):
        if topology and rng.random() < 0.35:
            if rng.random() < 0.15:
                index = rng.randrange(child.hidden)
                if not child.active[index] or sum(child.active) > 1:
                    child.active[index] = not child.active[index]
            else:
                group = rng.choice(mask_groups)
                index = rng.randrange(len(group))
                group[index] = not group[index]
        else:
            group = rng.choice(weight_groups)
            index = rng.randrange(len(group))
            group[index] += rng.gauss(0, 0.25)
    return child


def evaluate(genome: Genome, seeds: list[int], config: Config) -> dict[str, float]:
    """Episode scores, with mapping reversals and no regime signal in observations."""
    correct = total = near_correct = near_total = 0
    for seed in seeds:
        rng = random.Random(seed)
        regime = rng.randrange(2)
        state = [0.0] * genome.hidden
        prev_cue = prev_action = prev_reward = 0.0
        for step in range(config.steps):
            if step > 0 and step % config.flip_interval == 0:
                regime ^= 1
            cue = rng.randrange(2)
            observation = (float(2 * cue - 1), prev_cue, prev_action, prev_reward)
            action, state = genome.forward(observation, state)
            reward = int(action == (cue ^ regime))
            correct += reward
            total += 1
            if step >= config.flip_interval and step % config.flip_interval < 4:
                near_correct += reward
                near_total += 1
            prev_cue = float(2 * cue - 1)
            prev_action = float(2 * action - 1)
            prev_reward = float(2 * reward - 1)
    return {
        "accuracy": correct / total,
        "post_flip_accuracy": near_correct / near_total if near_total else 0.0,
    }


def _tuple_nested(value: object) -> object:
    if isinstance(value, list):
        return tuple(_tuple_nested(v) for v in value)
    return value


def _save_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_outputs(out: Path, state: dict) -> None:
    _save_json(out / "checkpoint.json", state)
    history = state["history"]
    (out / "history.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in history),
        encoding="utf-8",
    )
    if history:
        _save_json(
            out / "summary.json",
            {"config": state["config"], "generations_completed": len(history),
             "last_generation": history[-1],
             "interpretation": "Exploratory comparison; no claim of autonomy or consciousness."},
        )
        (out / "scores.svg").write_text(plot_svg(history), encoding="utf-8")


def plot_svg(history: list[dict]) -> str:
    """Self-contained SVG, no matplotlib dependency."""
    width, height = 680, 360
    lines = []
    for field, color in [("fixed_heldout", "#3465a4"), ("topology_heldout", "#c35035")]:
        points = []
        for index, row in enumerate(history):
            x = 55 + (width - 85) * index / max(1, len(history) - 1)
            y = 305 - 250 * row[field]
            points.append(f"{x:.1f},{y:.1f}")
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{" ".join(points)}"/>')
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 680 360">'
        '<rect width="680" height="360" fill="white"/>'
        '<text x="55" y="25" font-size="18">Held-out accuracy by generation</text>'
        '<line x1="55" y1="55" x2="55" y2="305" stroke="#777"/>'
        '<line x1="55" y1="305" x2="655" y2="305" stroke="#777"/>'
        '<text x="12" y="60" font-size="13">1.0</text>'
        '<text x="12" y="185" font-size="13">0.5</text>'
        '<text x="12" y="307" font-size="13">0.0</text>'
        '<text x="230" y="342" fill="#3465a4">Fixed topology</text>'
        '<text x="400" y="342" fill="#c35035">Mutable topology</text>'
        + "".join(lines) + "</svg>\n"
    )


def run(config: Config, out: Path, resume: Path | None = None) -> list[dict]:
    config.validate()
    out.mkdir(parents=True, exist_ok=True)
    if resume is None:
        rng = random.Random(config.seed)
        initial = [Genome.fresh(rng, config.hidden) for _ in range(config.population)]
        populations = {
            "fixed": copy.deepcopy(initial),
            "topology": copy.deepcopy(initial),
        }
        history: list[dict] = []
        start = 0
    else:
        state = json.loads(resume.read_text(encoding="utf-8"))
        former = Config(**state["config"])
        former.validate()
        if replace(former, generations=config.generations) != config:
            raise ValueError("Checkpoint config mismatch (only total generations may change)")
        start = state["next_generation"]
        if config.generations < start:
            raise ValueError("Requested generations below checkpoint progress")
        rng = random.Random()
        rng.setstate(_tuple_nested(state["rng_state"]))
        populations = {
            name: [Genome(**item) for item in state["populations"][name]]
            for name in ("fixed", "topology")
        }
        history = state["history"]

    heldout_seeds = [1_000_000_000 + config.seed * 101 + i for i in range(8)]
    for generation in range(start, config.generations):
        train_seeds = [100_000 + config.seed * 10_001 + generation * 100 + i
                       for i in range(config.episodes)]
        record: dict = {"generation": generation, "train_episodes": config.episodes}
        next_populations = {}
        for mode in ("fixed", "topology"):
            population = populations[mode]
            scores = [evaluate(genome, train_seeds, config)["accuracy"] for genome in population]
            ranking = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            best = population[ranking[0]]
            heldout = evaluate(best, heldout_seeds, config)
            record[f"{mode}_train"] = scores[ranking[0]]
            record[f"{mode}_heldout"] = heldout["accuracy"]
            record[f"{mode}_post_flip"] = heldout["post_flip_accuracy"]
            record[f"{mode}_active_connections"] = best.active_connections()
            elite_count = max(2, config.population // 4)
            elites = [copy.deepcopy(population[i]) for i in ranking[:elite_count]]
            children = [copy.deepcopy(elite) for elite in elites]
            while len(children) < config.population:
                parent = rng.choice(elites)
                children.append(mutate(parent, rng, mode == "topology", config.mutations))
            next_populations[mode] = children
        history.append(record)
        populations = next_populations
        state = {
            "format_version": 1,
            "config": asdict(config),
            "next_generation": generation + 1,
            "rng_state": rng.getstate(),
            "populations": {k: [asdict(g) for g in v] for k, v in populations.items()},
            "history": history,
        }
        _write_outputs(out, state)
        print(
            f"generation={generation:03d} "
            f"fixed={record['fixed_heldout']:.3f} "
            f"topology={record['topology_heldout']:.3f}",
            flush=True,
        )
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generations", type=int)
    parser.add_argument("--population", type=int, default=12)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--hidden", type=int, default=8)
    parser.add_argument("--flip-interval", type=int, default=20)
    parser.add_argument("--mutations", type=int, default=3)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.resume:
        saved = json.loads(args.resume.read_text(encoding="utf-8"))
        old = Config(**saved["config"])
        config = replace(old, generations=args.generations if args.generations is not None
                         else old.generations + 10)
        out = args.out or args.resume.parent
    else:
        config = Config(seed=args.seed, generations=args.generations or 12,
                        population=args.population, episodes=args.episodes,
                        steps=args.steps, hidden=args.hidden,
                        flip_interval=args.flip_interval, mutations=args.mutations)
        out = args.out or Path("results/local")
    run(config, out, args.resume)


if __name__ == "__main__":
    main()
