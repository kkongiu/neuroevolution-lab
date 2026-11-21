"""M5.2B: genuinely genotype-controlled ecology with compact 51-weight RNN.

Every organism chooses all five actions via unchanged M5.0 act; no scripted
motor or birth policy is used. A sparse newborn mutation adapter changes only
the child's numeric genome after a valid M5.0 clone birth, before child acts.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from . import m5_0 as base

ARMS = ("sparse_inherited", "dense_inherited",
        "clone_inherited", "random_offspring")
BASE_ARM = {
    "sparse_inherited": "heritable_no_mutation",
    "dense_inherited": "heritable_mutation",
    "clone_inherited": "heritable_no_mutation",
    "random_offspring": "random_offspring",
}
CONTRASTS = (("sparse_inherited", "random_offspring"),
             ("sparse_inherited", "clone_inherited"),
             ("sparse_inherited", "dense_inherited"),
             ("dense_inherited", "random_offspring"))


@dataclass(frozen=True)
class Settings(base.Settings):
    seed_start: int = 77000
    seeds: int = 24
    shifted_start: int = 78000
    shifted_seeds: int = 12
    hidden: int = 2
    no_food_controls: int = 0
    sparse_mutated_weights: int = 3

    def validate(self) -> None:
        super().validate()
        if self.sparse_mutated_weights < 1 or self.sparse_mutated_weights > base.genome_size(self.hidden):
            raise ValueError("Sparse mutation count outside genome")
        if self.ticks >= 400 and (self.hidden != 2 or self.sparse_mutated_weights != 3):
            raise ValueError("Full M5.2B protocol requires 2 hidden and three sparse mutations")


def mutate_sparse(parent: tuple[float, ...], settings: Settings,
                  seed: int, tick: int, parent_id: int
                  ) -> tuple[tuple[float, ...], tuple[int, ...]]:
    """Exactly k heritable changes at a legal in-world birth, no elsewhere."""
    if len(parent) != base.genome_size(settings.hidden):
        raise ValueError("Parent genotype has wrong architecture")
    rng = base._local_rng(seed, tick, parent_id, 77)
    indices = tuple(sorted(rng.sample(
        range(len(parent)), settings.sparse_mutated_weights
    )))
    child = list(parent)
    for idx in indices:
        child[idx] += rng.gauss(0., settings.mutation_sd)
    return tuple(child), indices


def simulate(seed: int, settings: Settings, arm: str, *,
             shifted: bool = False, detailed: bool = False
             ) -> tuple[dict, list[dict], dict | None]:
    settings.validate()
    if arm not in ARMS:
        raise ValueError(arm)
    eco = base.create_ecology(seed, settings, BASE_ARM[arm], shifted=shifted)
    schedule = base._food_schedule(seed, settings, eco, False)
    checks = []
    sparse_births = 0
    for tick in range(settings.ticks + 1):
        if tick:
            ids = list(eco.agents)
            base._local_rng(seed, tick, 0, 0).shuffle(ids)
            for aid in ids:
                agent = eco.agents.get(aid)
                if agent is None:
                    continue
                parent_genome = agent.genome if arm == "sparse_inherited" else None
                next_id = eco.next_id
                # All motor choices, reproductive wishes, recurrent state,
                # physical birth, energy and motor masking are original M5.0.
                base.act(eco, aid, tick)
                if arm == "sparse_inherited" and eco.next_id != next_id:
                    if eco.next_id != next_id + 1:
                        raise AssertionError("One actor born multiple times")
                    child = eco.agents[next_id]
                    if child.genome != parent_genome:
                        raise AssertionError("Sparse adapter expected clone newborn")
                    new_genome, mutated = mutate_sparse(
                        parent_genome, settings, seed, tick, aid
                    )
                    # A Gaussian draw is practically never exactly zero, but
                    # the index list is the authoritative mutation count.
                    if len(mutated) != settings.sparse_mutated_weights:
                        raise AssertionError("Sparse mutation scope changed")
                    child.genome = new_genome
                    eco.cosine_births[-1] = base.genome_cosine(
                        parent_genome, new_genome
                    )
                    if (not eco.events or eco.events[-1]["type"] != "birth"
                            or eco.events[-1]["child"] != child.id):
                        raise AssertionError("Birth log mismatch")
                    eco.events[-1]["genome"] = base.genome_digest(new_genome)
                    eco.events[-1]["sparse_mutated_indices"] = list(mutated)
                    sparse_births += 1
            base._food_regeneration(eco, schedule[tick - 1])
            eco.ticks_elapsed = tick
            if not eco.agents and eco.extinct_at is None:
                eco.extinct_at = tick
        if tick == 0 or tick % settings.checkpoint_every == 0 or tick == settings.ticks:
            entry = base.snapshot(eco, tick)
            entry.update({
                "seed": seed, "arm": arm,
                "split": "shifted" if shifted else "main",
            })
            checks.append(entry)
            if abs(base.energy_balance(eco)) > 1e-6:
                raise AssertionError("Ecological energy ledger not closed")
    births_by_founder = {}
    for event in eco.events:
        if event["type"] == "birth":
            founder = event["founder"]
            births_by_founder[founder] = births_by_founder.get(founder, 0) + 1
    living = list(eco.agents.values())
    row = {
        "seed": seed, "split": "shifted" if shifted else "main",
        "arm": arm, "births": eco.births, "deaths": eco.deaths,
        "deaths_starvation": eco.deaths_starvation,
        "deaths_age": eco.deaths_age, "alive_final": len(living),
        "survived": int(bool(living)), "extinct_at": eco.extinct_at,
        "food_eaten": eco.food_eaten, "food_spawned": eco.food_spawned,
        "food_remaining": len(eco.food),
        "max_generation": eco.max_generation,
        "has_generation3": int(eco.max_generation >= 3),
        "surviving_founder_lineages": len({a.founder for a in living}),
        "top_lineage_birth_share": (
            max(births_by_founder.values()) / eco.births
            if eco.births else 0.
        ),
        "mean_parent_child_cosine": (
            statistics.mean(eco.cosine_births) if eco.cosine_births else None
        ),
        "motor_mask_corrections": eco.corrected_actions,
        "ineligible_reproduction_preferences":
            eco.reproduction_ineligible_wishes,
        "move_actions": eco.move_actions,
        "birth_actions": eco.birth_actions,
        "total_food_energy_gained": eco.cumulative_food_energy,
        "energy_basal_cost": eco.cumulative_cost_basal,
        "energy_birth_dissipation": eco.cumulative_birth_dissipation,
        "energy_death_discard": eco.cumulative_death_discard,
        "energy_ledger_residual": base.energy_balance(eco),
        "sparse_births": sparse_births,
        "genome_parameters": base.genome_size(settings.hidden),
        "all_actions_neural": 1,
    }
    if arm == "sparse_inherited" and sparse_births != eco.births:
        raise AssertionError("Every sparse-arm child must be mutated")
    trace = None
    if detailed:
        trace = {
            "seed": seed, "split": row["split"], "arm": arm,
            "walls": [list(p) for p in sorted(eco.world.walls)],
            "initial_food": [list(p) for p in eco.world.initial_food],
            "initial_founder_genomes": list(eco.initial_genome_digests),
            "events": eco.events, "checkpoints": checks,
            "final_agents": [
                {"id": a.id, "parent": a.parent, "founder": a.founder,
                 "generation": a.generation, "position": list(a.position),
                 "energy": a.energy, "genome_digest": base.genome_digest(a.genome)}
                for a in sorted(living, key=lambda x: x.id)
            ],
        }
    return row, checks, trace


def contrast(rows: list[dict], split: str, measure: str, bootstrap_seed: int) -> dict:
    table = {
        arm: {r["seed"]: r[measure] for r in rows
              if r["split"] == split and r["arm"] == arm}
        for arm in ARMS
    }
    expected = sorted(table[ARMS[0]])
    if not expected or any(sorted(table[arm]) != expected for arm in ARMS):
        raise AssertionError("Unmatched world-seed/genotype-arm table")
    out = {}
    for i, (a, b) in enumerate(CONTRASTS):
        differences = [table[a][s] - table[b][s] for s in expected]
        low, high = base.paired_bootstrap(
            differences, bootstrap_seed + i * 101
        )
        mean = statistics.mean(differences)
        pos = sum(d > 0 for d in differences)
        neg = sum(d < 0 for d in differences)
        out[a + "_minus_" + b] = {
            "mean_delta": mean,
            "bootstrap_95_descriptive": [low, high],
            "positive_worlds": pos,
            "negative_worlds": neg,
            "tied_worlds": len(differences) - pos - neg,
            "supported": bool(mean > 0 and low > 0 and pos > neg),
            "per_world": {str(s): d for s, d in zip(expected, differences)},
        }
    return out


def population_svg(checks: list[dict], settings: Settings) -> str:
    width, height = 760, 405
    left, top, bottom = 64, 44, 320
    ticks = sorted({r["tick"] for r in checks if r["split"] == "main"})
    highest = max(settings.initial_population, max(
        r["population"] for r in checks if r["split"] == "main"
    ))
    palette = ("#7c3aed", "#0f766e", "#ea580c", "#2563eb")
    paths = []
    for arm, color in zip(ARMS, palette):
        positions = []
        for tick in ticks:
            avg = statistics.mean(
                r["population"] for r in checks
                if r["split"] == "main" and r["arm"] == arm
                and r["tick"] == tick
            )
            positions.append(
                f"{left+(width-left-25)*tick/settings.ticks:.1f},"
                f"{bottom-(bottom-top)*avg/highest:.1f}"
            )
        paths.append(
            f'<polyline points="{" ".join(positions)}" '
            f'fill="none" stroke="{color}" stroke-width="2.8"/>'
        )
    labels = "".join(
        f'<text x="{left + i*177}" y="370" fill="{palette[i]}" '
        f'font-size="12" font-family="sans-serif">{a}</text>'
        for i, a in enumerate(ARMS)
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="white"/>'
        '<text x="64" y="23" font-family="sans-serif" font-size="15">'
        'M5.2B population mean, 24 paired main worlds</text>'
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" '
        'stroke="black"/>'
        f'<line x1="{left}" y1="{bottom}" x2="{width-25}" y2="{bottom}" '
        'stroke="black"/>'
        + "".join(paths) + labels + '</svg>\n'
    )


def run(settings: Settings, output: Path) -> dict:
    settings.validate()
    output.mkdir(parents=True, exist_ok=True)
    rows, checks, traces = [], [], []
    for split, start, count, shifted in (
        ("main", settings.seed_start, settings.seeds, False),
        ("shifted", settings.shifted_start, settings.shifted_seeds, True),
    ):
        for seed in range(start, start + count):
            genomes = None
            for arm in ARMS:
                row, checkpoints, trace = simulate(
                    seed, settings, arm, shifted=shifted,
                    detailed=(seed == start)
                )
                rows.append(row)
                checks.extend(checkpoints)
                if trace is not None:
                    if genomes is None:
                        genomes = trace["initial_founder_genomes"]
                    elif genomes != trace["initial_founder_genomes"]:
                        raise AssertionError("Genetic founder mismatch across arms")
                    traces.append(trace)
            print(f"M5.2B {split} {seed}: " + " ".join(
                f"{row['arm']}={row['alive_final']}/{row['births']}"
                for row in rows[-4:]
            ), flush=True)
    means = {
        split: {
            arm: {
                metric: statistics.mean(
                    r[metric] for r in rows
                    if r["split"] == split and r["arm"] == arm
                ) for metric in (
                    "survived", "alive_final", "births",
                    "food_eaten", "max_generation", "has_generation3",
                    "food_remaining", "motor_mask_corrections"
                )
            } for arm in ARMS
        } for split in ("main", "shifted")
    }
    main_survival = contrast(
        rows, "main", "survived", settings.seed_start + 19
    )
    shifted_survival = contrast(
        rows, "shifted", "survived", settings.shifted_start + 19
    )
    sp = means["main"]["sparse_inherited"]
    sh = means["shifted"]["sparse_inherited"]
    sparse_random = main_survival["sparse_inherited_minus_random_offspring"]
    sparse_clone = main_survival["sparse_inherited_minus_clone_inherited"]
    shifted_random = shifted_survival[
        "sparse_inherited_minus_random_offspring"
    ]
    summary = {
        "protocol": "M5.2B; docs/M5_2B_PROTOCOL.md",
        "settings": asdict(settings), "means": means,
        "main_primary_survival": main_survival,
        "shifted_survival": shifted_survival,
        "main_exploratory_food": contrast(
            rows, "main", "food_eaten", settings.seed_start + 341
        ),
        "preregistered_gates": {
            "main_viable_worlds": int(round(sp["survived"]*settings.seeds)),
            "main_generation3_worlds":
                int(round(sp["has_generation3"]*settings.seeds)),
            "main_viability_pass": sp["survived"] >= .5,
            "main_generation3_pass": sp["has_generation3"] >= .5,
            "main_heritable_vs_random_pass": sparse_random["supported"],
            "main_all_primary_pass": bool(
                sp["survived"] >= .5 and
                sp["has_generation3"] >= .5 and
                sparse_random["supported"]
            ),
            "novel_mutation_vs_clone_pass": sparse_clone["supported"],
            "shifted_viable_worlds": int(round(
                sh["survived"]*settings.shifted_seeds
            )),
            "shifted_transfer_pass": bool(
                sh["survived"] >= .5 and shifted_random["supported"]
            ),
        },
        "limits": [
            "Every single motor and birth action is decided by genotype-dependent RNN",
            "A different 51-parameter architecture has lower expressive capacity",
            "An inherited advantage versus random does not prove novel mutations",
            "A novel mutation advantage also requires independent genotype transplant",
            "Independent units are whole world seeds, not actions or births",
            "Numeric heredity is a simulated mechanism, not open-ended life",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8"
    )
    for name, data in (("world_metrics.csv", rows),
                       ("population_checkpoints.csv", checks)):
        with (output / name).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    for trace in traces:
        (output / f"events_{trace['split']}_{trace['arm']}.json").write_text(
            json.dumps(trace, indent=2, sort_keys=True) + "\n",
            encoding="utf-8"
        )
    (output / "population_curves.svg").write_text(
        population_svg(checks, settings), encoding="utf-8"
    )
    print("M5.2B GATES", summary["preregistered_gates"], flush=True)
    print("M5.2B MAIN", summary["main_primary_survival"], flush=True)
    print("M5.2B SHIFTED", summary["shifted_survival"], flush=True)
    print("M5.2B MEANS", summary["means"], flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for field in fields(Settings):
        parser.add_argument(
            "--" + field.name.replace("_", "-"),
            type=type(field.default), default=field.default
        )
    parser.add_argument("--out", type=Path, default=Path("results/m5-2b"))
    kwargs = vars(parser.parse_args())
    out = kwargs.pop("out")
    run(Settings(**kwargs), out)


if __name__ == "__main__":
    main()
