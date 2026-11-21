"""M5.3A: positive neural representation controls, NOT autonomous evolution.

- A deliberately ENGINEERED numerical forager, with directions and birth
  preferences expressed as RNN weights; never call it evolved.
- Two numerical genomes externally supervised to imitate M5.1's script
  on disjoint development worlds. No scripted action during ecological tests.
- Fixed random and scrambled-genotype controls, exact clonal births.
Every ecological action is the unchanged neural m5_0.act function.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

from . import m5_0 as base
from . import m5_1 as teacher

ARMS = (
    "engineered_12", "imitation_12", "imitation_2",
    "random_12", "random_2", "scrambled_12"
)
N_HIDDEN = {"engineered_12": 12, "imitation_12": 12,
            "imitation_2": 2, "random_12": 12,
            "random_2": 2, "scrambled_12": 12}
CONTRASTS = (
    ("engineered_12", "random_12"),
    ("engineered_12", "scrambled_12"),
    ("imitation_12", "random_12"),
    ("imitation_2", "random_2"),
)


@dataclass(frozen=True)
class Settings(teacher.Settings):
    seed_start: int = 80000
    seeds: int = 24
    shifted_start: int = 81000
    shifted_seeds: int = 12
    hidden: int = 12
    no_food_controls: int = 0
    negative_control_worlds: int = 1
    teacher_seed_start: int = 79000
    teacher_seed_count: int = 6
    validation_teacher_start: int = 79100
    validation_teacher_count: int = 2
    teacher_ticks: int = 200
    train_cap: int = 4500
    validation_cap: int = 1500
    epochs: int = 12
    learning_rate: float = 0.03

    def validate(self) -> None:
        super().validate()
        if min(
            self.teacher_seed_count, self.validation_teacher_count,
            self.teacher_ticks, self.train_cap, self.validation_cap,
            self.epochs
        ) <= 0 or self.learning_rate <= 0:
            raise ValueError("Training budget must be positive")
        splits = (
            set(range(self.teacher_seed_start,
                      self.teacher_seed_start+self.teacher_seed_count)),
            set(range(self.validation_teacher_start,
                      self.validation_teacher_start +
                      self.validation_teacher_count)),
            set(range(self.seed_start, self.seed_start + self.seeds)),
            set(range(self.shifted_start,
                      self.shifted_start + self.shifted_seeds)),
        )
        for i, one in enumerate(splits):
            if any(one & x for x in splits[i + 1:]):
                raise ValueError("M5.3A training/validation/test seed leakage")
        if self.ticks >= 400 and (
            self.hidden != 12 or self.teacher_ticks != 200 or
            self.teacher_seed_count != 6 or
            self.validation_teacher_count != 2 or
            self.train_cap != 4500 or self.validation_cap != 1500 or
            self.epochs != 12 or self.learning_rate != .03
        ):
            raise ValueError("Sealed M5.3A parameters are frozen")


def witness_12() -> tuple[float, ...]:
    """A HUMAN-DESIGNED locally reactive numeric neural positive witness.

    Hidden[0:4] are direction detectors, hidden[4] energy-sensitive birth.
    No recurrence. Runtime decisions are numerical m5_0 network outputs.
    """
    n = 12
    stride = base.INPUTS + n + 1
    head = n * stride
    w = [0.] * base.genome_size(n)
    for a in range(4):
        offset = a * stride
        w[offset + a] = -1.0           # blocked direction
        w[offset + 4 + a] = 2.0       # local odor
        w[offset + 8 + a] = 8.0       # adjacent food
        w[head + a * (n + 1) + a] = 1.
    w[4 * stride + 12] = 9.0          # normalized energy
    w[4 * stride + base.INPUTS + n] = -8.0
    w[head + 4 * (n + 1) + 4] = 2.5
    w[head + 4 * (n + 1) + n] = .85
    return tuple(w)


def initial_student(n: int) -> list[float]:
    rng = random.Random(92000 + n)
    stride = base.INPUTS + n + 1
    w = [0.] * base.genome_size(n)
    for i in range(n):
        off = i * stride
        for j in range(base.INPUTS):
            w[off + j] = rng.gauss(0., .075)
        w[off + base.INPUTS + n] = rng.gauss(0., .075)
    head = n * stride
    for a in range(base.ACTIONS):
        off = head + a * (n + 1)
        for i in range(n + 1):
            w[off + i] = rng.gauss(0., .075)
    return w


def teacher_examples(settings: Settings, *, validation: bool
                     ) -> list[tuple[tuple[float, ...], tuple[int, ...], int]]:
    """Generate ONLY labelled local observation/action tuples from the script.

    The scripted teacher supplies labels in development worlds. The fixed
    trained weights, not the teacher, determine every final-world action.
    """
    first = (settings.validation_teacher_start if validation
             else settings.teacher_seed_start)
    nseeds = (settings.validation_teacher_count if validation
              else settings.teacher_seed_count)
    sample_cap = settings.validation_cap if validation else settings.train_cap
    out = []
    for seed in range(first, first + nseeds):
        eco = base.create_ecology(seed, settings, "heritable_no_mutation")
        schedule = base._food_schedule(seed, settings, eco, False)
        for tick in range(1, settings.teacher_ticks + 1):
            ids = list(eco.agents)
            base._local_rng(seed, tick, 0, 0).shuffle(ids)
            for agent_id in ids:
                agent = eco.agents.get(agent_id)
                if agent is None:
                    continue
                _, _, sensors, legal_motors, can_birth = (
                    teacher._candidate_actions(eco, agent)
                )
                legal = legal_motors + (
                    [base.REPRODUCE] if can_birth else []
                )
                action = teacher.scripted_choice(
                    eco, agent, "greedy_odor", tick, {}
                )
                if action is not None:
                    if action not in legal:
                        raise AssertionError("Teacher labelled an illegal action")
                    out.append((sensors, tuple(legal), action))
                teacher._act_scripted(eco, agent_id, tick, action)
            base._food_regeneration(eco, schedule[tick - 1])
            eco.ticks_elapsed = tick
            if abs(base.energy_balance(eco)) > 1e-6:
                raise AssertionError("Teacher world violates physical ledger")
    if len(out) > sample_cap:
        rng = random.Random(90513 if validation else 90512)
        out = [out[i] for i in sorted(rng.sample(range(len(out)), sample_cap))]
    if not out:
        raise AssertionError("No teacher samples")
    return out


def predict(w: list[float] | tuple[float, ...],
            n: int, observation: tuple[float, ...],
            legal: tuple[int, ...]) -> int:
    # Use the exact same original network evaluation primitive as ecology.
    agent = base.Agent(
        id=0, founder=0, parent=None, generation=0,
        position=(1, 1), energy=1, genome=tuple(w), hidden=[0.] * n
    )
    logits = base.network_logits(agent, observation, replace(
        base.Settings(), hidden=n
    ))
    return max(legal, key=lambda a: (logits[a], -a))


def accuracy(w, n, examples) -> float:
    return sum(predict(w, n, obs, legal) == label
               for obs, legal, label in examples) / len(examples)


def train_student(n: int, settings: Settings,
                  training, validation
                  ) -> tuple[tuple[float, ...], dict]:
    """Predeclared 12-epoch SGD, legal-action softmax cross-entropy.

    No recurrent weights are trained: initialized to zero and frozen.
    The recurrent network is restricted to its memoryless numeric subspace.
    """
    w = initial_student(n)
    stride = base.INPUTS + n + 1
    head = n * stride
    best = tuple(w)
    best_acc = -1.
    best_epoch = 0
    history = []
    rng = random.Random(93000 + n)
    items = list(training)
    for epoch in range(1, settings.epochs + 1):
        rng.shuffle(items)
        for observation, legal, label in items:
            hidden = []
            for i in range(n):
                offset = i * stride
                act = sum(
                    w[offset + j] * observation[j]
                    for j in range(base.INPUTS)
                ) + w[offset + base.INPUTS + n]
                hidden.append(math.tanh(act))
            logits = [
                sum(
                    w[head + a*(n+1) + i]*hidden[i]
                    for i in range(n)
                ) + w[head + a*(n+1) + n]
                for a in range(base.ACTIONS)
            ]
            maximum = max(logits[a] for a in legal)
            weights = {a: math.exp(logits[a] - maximum) for a in legal}
            normalizer = sum(weights.values())
            grad_logits = {
                a: weights[a] / normalizer - (1. if a == label else 0.)
                for a in legal
            }
            grad_h = [
                (1. - hidden[i]**2) * sum(
                    grad_logits[a] * w[head + a*(n+1) + i]
                    for a in legal
                ) for i in range(n)
            ]
            for a in legal:
                off = head + a*(n+1)
                d = settings.learning_rate * grad_logits[a]
                for i in range(n):
                    w[off + i] -= d * hidden[i]
                w[off + n] -= d
            for i in range(n):
                off = i*stride
                d = settings.learning_rate * grad_h[i]
                for j in range(base.INPUTS):
                    w[off + j] -= d * observation[j]
                w[off + base.INPUTS + n] -= d
        measured = accuracy(w, n, validation)
        history.append({
            "epoch": epoch, "validation_accuracy": measured
        })
        if measured > best_acc:
            best, best_acc, best_epoch = tuple(w), measured, epoch
    return best, {
        "hidden_units": n, "parameters": len(best),
        "train_examples": len(training),
        "validation_examples": len(validation),
        "selected_epoch": best_epoch,
        "best_validation_accuracy": best_acc,
        "validation_history": history,
        "selected_train_accuracy": accuracy(best, n, training),
        "note": "External teacher imitation, NOT in-world evolution",
    }


def genotypes(settings: Settings) -> tuple[dict, dict]:
    train = teacher_examples(settings, validation=False)
    valid = teacher_examples(settings, validation=True)
    w12, m12 = train_student(12, settings, train, valid)
    w2, m2 = train_student(2, settings, train, valid)
    witness = witness_12()
    scrambled = list(witness)
    random.Random(90212).shuffle(scrambled)
    result = {
        "engineered_12": witness,
        "imitation_12": w12,
        "imitation_2": w2,
        "random_12": base.initial_genome(
            random.Random(90112), replace(settings, hidden=12)
        ),
        "random_2": base.initial_genome(
            random.Random(90102), replace(settings, hidden=2)
        ),
        "scrambled_12": tuple(scrambled),
    }
    if set(result) != set(ARMS) or any(
        len(result[a]) != base.genome_size(N_HIDDEN[a]) for a in ARMS
    ):
        raise AssertionError("Fixed genotype bank is inconsistent")
    meta = {
        "imitation_12": m12, "imitation_2": m2,
        "training_seed_range": [settings.teacher_seed_start,
                                settings.teacher_seed_start +
                                settings.teacher_seed_count - 1],
        "validation_seed_range": [settings.validation_teacher_start,
                                  settings.validation_teacher_start +
                                  settings.validation_teacher_count - 1],
        "genome_digests": {
            a: base.genome_digest(result[a]) for a in ARMS
        },
        "engineered_policy_notice":
            "Hand-designed directional/energy weights, not emerged or evolved",
    }
    return result, meta


def simulate(seed: int, settings: Settings, arm: str,
             genotype: tuple[float, ...], *,
             shifted: bool = False, detailed: bool = False
             ) -> tuple[dict, list[dict], dict | None]:
    settings.validate()
    if arm not in ARMS:
        raise ValueError(arm)
    n = N_HIDDEN[arm]
    if len(genotype) != base.genome_size(n):
        raise ValueError("Transplant genotype has wrong numerical shape")
    actual = replace(settings, hidden=n)
    # Original physical M5.0 world and original neural action primitive:
    eco = base.create_ecology(
        seed, actual, "heritable_no_mutation", shifted=shifted
    )
    for agent in eco.agents.values():
        agent.genome = genotype
        agent.hidden = [0.] * n
    eco.initial_genome_digests = tuple(
        base.genome_digest(genotype)
        for _ in range(actual.initial_population)
    )
    schedule = base._food_schedule(seed, actual, eco, False)
    checkpoints = []
    for tick in range(settings.ticks + 1):
        if tick:
            ids = list(eco.agents)
            base._local_rng(seed, tick, 0, 0).shuffle(ids)
            for aid in ids:
                base.act(eco, aid, tick)
            base._food_regeneration(eco, schedule[tick - 1])
            eco.ticks_elapsed = tick
            if not eco.agents and eco.extinct_at is None:
                eco.extinct_at = tick
        if (tick == 0 or tick % settings.checkpoint_every == 0 or
                tick == settings.ticks):
            checkpoint = base.snapshot(eco, tick)
            checkpoint.update({
                "split": "shifted" if shifted else "main",
                "seed": seed, "arm": arm,
            })
            checkpoints.append(checkpoint)
            if abs(base.energy_balance(eco)) > 1e-6:
                raise AssertionError("M5.3A energy ledger not closed")
    if any(event["genome"] != base.genome_digest(genotype)
           for event in eco.events if event["type"] == "birth"):
        raise AssertionError("Clonal offspring genome differs from transplant")
    row = {
        "seed": seed, "split": "shifted" if shifted else "main",
        "arm": arm, "genome_digest": base.genome_digest(genotype),
        "parameters": len(genotype),
        "survived": int(bool(eco.agents)),
        "alive_final": len(eco.agents), "births": eco.births,
        "deaths": eco.deaths,
        "max_generation": eco.max_generation,
        "has_generation3": int(eco.max_generation >= 3),
        "food_eaten": eco.food_eaten,
        "food_remaining": len(eco.food),
        "extinct_at": eco.extinct_at,
        "motor_mask_corrections": eco.corrected_actions,
        "ineligible_reproduction_preferences":
            eco.reproduction_ineligible_wishes,
        "total_food_energy_gained": eco.cumulative_food_energy,
        "energy_ledger_residual": base.energy_balance(eco),
        "policy_type": (
            "human_engineered_numeric" if arm == "engineered_12"
            else "externally_supervised_numeric" if arm.startswith("imitation")
            else "negative_genotype_control"
        ),
        "all_actions_original_neural": 1,
        "inheritance": "clone_no_mutation",
    }
    trace = None
    if detailed:
        trace = {
            "seed": seed, "arm": arm, "split": row["split"],
            "walls": [list(pos) for pos in sorted(eco.world.walls)],
            "initial_food": [list(pos) for pos in eco.world.initial_food],
            "fixed_genome": list(genotype),
            "events": eco.events, "checkpoints": checkpoints,
            "final_agents": [
                {"id": a.id, "parent": a.parent, "founder": a.founder,
                 "generation": a.generation, "energy": a.energy,
                 "position": list(a.position)}
                for a in sorted(eco.agents.values(), key=lambda a: a.id)
            ],
        }
    return row, checkpoints, trace


def contrasts(rows: list[dict], split: str, metric: str, boot_seed: int) -> dict:
    tab = {
        a: {r["seed"]: r[metric] for r in rows
            if r["split"] == split and r["arm"] == a}
        for a in ARMS
    }
    keys = sorted(tab[ARMS[0]])
    if not keys or any(sorted(tab[a]) != keys for a in ARMS):
        raise AssertionError("Unpaired world seeds")
    result = {}
    for i, (a, b) in enumerate(CONTRASTS):
        delta = [tab[a][k] - tab[b][k] for k in keys]
        lo, hi = base.paired_bootstrap(delta, boot_seed + i * 103)
        pos, neg = sum(v > 0 for v in delta), sum(v < 0 for v in delta)
        mean = statistics.mean(delta)
        result[a + "_minus_" + b] = {
            "mean_delta": mean,
            "bootstrap_95_descriptive": [lo, hi],
            "positive_worlds": pos, "negative_worlds": neg,
            "ties": len(delta) - pos - neg,
            "supported": bool(mean > 0 and lo > 0 and pos > neg),
            "by_world": {str(k): d for k, d in zip(keys, delta)},
        }
    return result


def population_svg(checkpoints: list[dict], settings: Settings) -> str:
    width, height, left, top, bottom = 820, 415, 65, 42, 320
    ticks = sorted({c["tick"] for c in checkpoints if c["split"] == "main"})
    max_population = max(settings.initial_population, max(
        c["population"] for c in checkpoints if c["split"] == "main"
    ))
    palette = ("#0d9488", "#2563eb", "#9333ea",
               "#d97706", "#64748b", "#dc2626")
    curves = []
    for arm, color in zip(ARMS, palette):
        vals = [
            statistics.mean(c["population"] for c in checkpoints
                            if c["split"] == "main" and c["arm"] == arm
                            and c["tick"] == tick)
            for tick in ticks
        ]
        points = " ".join(
            f"{left+(width-left-15)*tick/settings.ticks:.1f},"
            f"{bottom-(bottom-top)*v/max_population:.1f}"
            for tick, v in zip(ticks, vals)
        )
        curves.append(
            f'<polyline points="{points}" fill="none" '
            f'stroke="{color}" stroke-width="2.5"/>'
        )
    labels = "".join(
        f'<text x="{left+(i%3)*245}" y="{365+(i//3)*24}" '
        f'font-family="sans-serif" font-size="12" fill="{palette[i]}">'
        f'{arm}</text>' for i, arm in enumerate(ARMS)
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="white"/>'
        '<text x="65" y="24" font-size="17" font-family="sans-serif">'
        'M5.3A: fixed neural genotypes, main-world population</text>'
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" '
        'stroke="black"/>'
        f'<line x1="{left}" y1="{bottom}" x2="{width-20}" y2="{bottom}" '
        'stroke="black"/>'
        + "".join(curves) + labels + '</svg>\n'
    )


def run(settings: Settings, out: Path) -> dict:
    settings.validate()
    out.mkdir(parents=True, exist_ok=True)
    genome_bank, development = genotypes(settings)
    rows, checkpoints, traces = [], [], []
    for split, start, count, shift in (
        ("main", settings.seed_start, settings.seeds, False),
        ("shifted", settings.shifted_start, settings.shifted_seeds, True),
    ):
        for seed in range(start, start + count):
            for arm in ARMS:
                row, ck, trace = simulate(
                    seed, settings, arm, genome_bank[arm],
                    shifted=shift, detailed=(seed == start)
                )
                rows.append(row)
                checkpoints.extend(ck)
                if trace:
                    traces.append(trace)
            print(f"M5.3A {split} {seed} " + " ".join(
                f"{r['arm']}={r['alive_final']}/{r['births']}"
                for r in rows[-len(ARMS):]
            ), flush=True)
    means = {
        split: {
            arm: {
                key: statistics.mean(
                    row[key] for row in rows
                    if row["split"] == split and row["arm"] == arm
                )
                for key in (
                    "survived", "alive_final", "births", "food_eaten",
                    "max_generation", "has_generation3", "food_remaining",
                    "motor_mask_corrections",
                )
            } for arm in ARMS
        } for split in ("main", "shifted")
    }
    cmp = contrasts(rows, "main", "survived", settings.seed_start + 17)
    shifted = contrasts(
        rows, "shifted", "survived", settings.shifted_start + 17
    )
    eng = means["main"]["engineered_12"]
    sup12 = means["main"]["imitation_12"]
    sup2 = means["main"]["imitation_2"]
    summary = {
        "protocol": "M5.3A; docs/M5_3A_PROTOCOL.md",
        "interpretation": "Representational positive control, never evolved competence",
        "settings": asdict(settings),
        "development": development,
        "means": means,
        "main_survival_comparisons": cmp,
        "shifted_survival_comparisons": shifted,
        "main_exploratory_food": contrasts(
            rows, "main", "food_eaten", settings.seed_start + 211
        ),
        "preregistered_gates": {
            "engineered_12_viable_worlds":
                int(round(eng["survived"] * settings.seeds)),
            "engineered_12_generation3_worlds":
                int(round(eng["has_generation3"] * settings.seeds)),
            "engineered_12_primary_pass": bool(
                eng["survived"] >= .5 and
                eng["has_generation3"] >= .5 and
                cmp["engineered_12_minus_random_12"]["supported"]
            ),
            "engineered_12_causal_scramble_pass":
                cmp["engineered_12_minus_scrambled_12"]["supported"],
            "imitation_12_secondary_pass": bool(
                sup12["survived"] >= .5 and
                sup12["has_generation3"] >= .5 and
                cmp["imitation_12_minus_random_12"]["supported"]
            ),
            "imitation_2_exploratory_pass": bool(
                sup2["survived"] >= .5 and
                sup2["has_generation3"] >= .5 and
                cmp["imitation_2_minus_random_2"]["supported"]
            ),
            "engineered_12_shifted_viability":
                means["shifted"]["engineered_12"]["survived"] >= .5,
            "imitation_12_shifted_viability":
                means["shifted"]["imitation_12"]["survived"] >= .5,
        },
        "limits": [
            "Engineered_12 weights explicitly encode researcher-designed odor following",
            "Imitation weights are externally trained on scripted-teacher data",
            "Neither treatment evolves behavior via in-world natural selection",
            "Fixed clonal genotypes exclude mutation-driven adaptation by design",
            "Student teacher age is not directly exposed as a neural sensor",
            "One world seed, not a birth or action, is the replication unit",
            "No ecological test world used for gradient updates or selection",
        ],
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8"
    )
    for filename, data in (
        ("world_metrics.csv", rows),
        ("population_checkpoints.csv", checkpoints),
    ):
        with (out / filename).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    (out / "genotypes.json").write_text(
        json.dumps({a: list(genome_bank[a]) for a in ARMS},
                   indent=2, sort_keys=True) + "\n",
        encoding="utf-8"
    )
    for trace in traces:
        (out / f"events_{trace['split']}_{trace['arm']}.json").write_text(
            json.dumps(trace, indent=2, sort_keys=True) + "\n",
            encoding="utf-8"
        )
    (out / "population_curves.svg").write_text(
        population_svg(checkpoints, settings), encoding="utf-8"
    )
    print("M5.3A GATES", summary["preregistered_gates"], flush=True)
    print("M5.3A MAIN", summary["main_survival_comparisons"], flush=True)
    print("M5.3A MEANS", summary["means"], flush=True)
    print("M5.3A LEARNING", {
        k: (v["best_validation_accuracy"], v["selected_epoch"])
        for k, v in development.items() if k.startswith("imitation")
    }, flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for f in fields(Settings):
        parser.add_argument("--" + f.name.replace("_", "-"),
                            type=type(f.default), default=f.default)
    parser.add_argument("--out", type=Path, default=Path("results/m5-3a"))
    args = vars(parser.parse_args())
    output = args.pop("out")
    run(Settings(**args), output)


if __name__ == "__main__":
    main()
