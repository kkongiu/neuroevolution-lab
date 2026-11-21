"""M5.3C: prospectively preregistered genetic ancestor/descendant transplant.

Source agents were selected *only from five already published successful M5.3B
worlds* by a fixed generation/ID rule. Test worlds are entirely independent.
Fixed neural genotypes are transplanted, cloned on birth, and evaluated by the
unchanged original M5.0 RNN action/physical ecology. This test measures numeric
phenotype transfer, not a new evolutionary search or emergent architecture.
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
from . import m5_3b as evolution

SOURCES = (82006, 82015, 82018, 82019, 82023)
ARMS = ("descendant", "ancestor", "last_step_reverted",
        "scrambled", "independent_random")
CONTRASTS = ("ancestor", "scrambled", "last_step_reverted",
             "independent_random")
BOOT_REPS = 20_000
BOOT_SEED = 87001


@dataclass(frozen=True)
class Settings(evolution.Settings):
    seed_start: int = 84000
    seeds: int = 24
    shifted_start: int = 85000
    shifted_seeds: int = 12
    no_food_controls: int = 0

    def validate(self) -> None:
        super().validate()
        src = set(SOURCES)
        main = set(range(self.seed_start, self.seed_start + self.seeds))
        hard = set(range(self.shifted_start,
                         self.shifted_start + self.shifted_seeds))
        historic = set(range(79000, 79006)) | set(range(79100, 79102))
        historic |= set(range(80000, 80024)) | set(range(81000, 81012))
        historic |= set(range(82000, 82024)) | set(range(83000, 83012))
        if main & hard or main & (src | historic) or hard & (src | historic):
            raise ValueError("M5.3C seed leakage into prior worlds")
        if self.ticks >= 400:
            canonical = Settings()
            for name in (
                "seed_start", "seeds", "shifted_start", "shifted_seeds",
                "ticks", "size", "obstacle_density",
                "shifted_obstacle_density", "initial_food", "food_cap",
                "regeneration_attempts", "regeneration_probability",
                "shifted_regeneration_probability", "odor_radius",
                "shifted_odor_radius", "initial_population", "max_population",
                "initial_energy", "max_energy", "food_energy",
                "movement_cost", "newborn_energy",
                "reproduction_energy_cost", "min_reproduction_energy",
                "min_reproduction_age", "max_age", "hidden",
                "checkpoint_every", "no_food_controls",
                "initial_gene_sd", "gene_mutation_sd",
                "sparse_gene_count",
            ):
                if getattr(self, name) != getattr(canonical, name):
                    raise ValueError(
                        "M5.3C 400-tick campaign parameter frozen: " + name
                    )


def original_settings() -> evolution.Settings:
    # The source run is EXACT M5.3B's already reported ordinary ecology.
    return evolution.Settings(no_food_controls=0)


def recover_one(source_seed: int) -> dict:
    """Deterministically reconstruct exact selected genotype and parent chain."""
    if source_seed not in SOURCES:
        raise ValueError("Unregistered source world")
    record, checkpoints, trace = evolution.simulate(
        source_seed, original_settings(), "sparse_2",
        shifted=False, detailed=True
    )
    if record["survived"] != 1 or not trace or not trace["final_agents"]:
        raise AssertionError("Frozen M5.3B source no longer survives")
    selected = min(
        trace["final_agents"], key=lambda a: (-a["generation"], a["id"])
    )
    founders = [tuple(g) for g in trace["initial_founder_genes"]]
    genes = {i: g for i, g in enumerate(founders)}
    birth_index = {}
    for event in trace["events"]:
        if event["type"] != "birth":
            continue
        parent_id, child_id = event["parent"], event["child"]
        if parent_id not in genes or child_id in genes:
            raise AssertionError("Invalid historical parent-child genealogy")
        parent = genes[parent_id]
        proposed, changed = evolution.child_genes(
            parent, original_settings(), source_seed,
            event["tick"], parent_id, "sparse_2"
        )
        inherited = tuple(event["gene_values"])
        if (inherited != proposed or
                tuple(event["changed_gene_indices"]) != changed or
                event["gene_digest"] != evolution.gene_digest(inherited) or
                event["genome"] != base.genome_digest(
                    evolution.to_weights(inherited)
                ) or len(changed) != 2):
            raise AssertionError("Source mutation/digest was not replayed exactly")
        genes[child_id] = inherited
        birth_index[child_id] = event
    chosen_id = selected["id"]
    child_genes = tuple(selected["gene_values"])
    if genes.get(chosen_id) != child_genes or (
        base.genome_digest(evolution.to_weights(child_genes)) !=
        selected["genome_digest"]
    ):
        raise AssertionError("Chosen final genotype inconsistent with M5.3B")
    lineage = []
    cursor = chosen_id
    while cursor >= len(founders):
        event = birth_index.get(cursor)
        if event is None:
            raise AssertionError("Missing historical birth in genealogy")
        lineage.append(event)
        cursor = event["parent"]
    lineage.reverse()
    if (cursor != selected["founder"] or
            len(lineage) != selected["generation"] or
            selected["generation"] <= 0):
        raise AssertionError("Selected descendant genealogy not valid")
    ancestor = founders[cursor]
    final = birth_index[chosen_id]
    immediate_parent = genes[final["parent"]]
    reverted = list(child_genes)
    for ix in final["changed_gene_indices"]:
        reverted[ix] = immediate_parent[ix]
    reverted = tuple(reverted)
    if reverted == child_genes or reverted != immediate_parent:
        # Only the two specified genes change at a sparse birth, so reverting
        # exactly those two must recreate the full immediate-parent genotype.
        raise AssertionError("Last-step mutation ablation not exact")
    scramble = list(child_genes)
    random.Random(95000 + source_seed).shuffle(scramble)
    scrambled = tuple(scramble)
    if scrambled == child_genes:
        raise AssertionError("Randomly scrambled genome stayed unchanged")
    rng = random.Random(94000 + source_seed)
    independent_random = tuple(
        rng.gauss(0., original_settings().initial_gene_sd)
        for _ in range(evolution.N_GENES)
    )
    bank = {
        "descendant": child_genes, "ancestor": ancestor,
        "last_step_reverted": reverted,
        "scrambled": scrambled,
        "independent_random": independent_random,
    }
    metadata = {
        "source_seed": source_seed,
        "original_source_arm": "sparse_2",
        "selected_final_id": chosen_id,
        "selected_final_generation": selected["generation"],
        "selected_final_founder_id": cursor,
        "selection_rule": "max final living generation, then lowest ID",
        "original_source_alive_final": record["alive_final"],
        "original_source_births": record["births"],
        "genealogy": [
            {
                "tick": e["tick"], "parent": e["parent"],
                "child": e["child"], "founder": e["founder"],
                "generation": e["generation"],
                "mutated_indices": e["changed_gene_indices"],
                "gene_values": e["gene_values"],
                "gene_digest": e["gene_digest"],
                "expanded_genome_digest": e["genome"],
            }
            for e in lineage
        ],
        "last_step_mutated_indices": final["changed_gene_indices"],
        "genotypes": {
            a: {
                "genes": list(bank[a]),
                "gene_digest": evolution.gene_digest(bank[a]),
                "native_401_genome_digest": base.genome_digest(
                    evolution.to_weights(bank[a])
                ),
            }
            for a in ARMS
        },
        "source_checkpoints": checkpoints,
        "note": (
            "Five families selected conditionally on M5.3B successful sources; "
            "NO held-out ecological outcome used to choose or alter candidates"
        ),
    }
    return metadata


def recover_candidates() -> list[dict]:
    return [recover_one(seed) for seed in SOURCES]


def simulate_transplant(seed: int, settings: Settings, source: dict,
                        arm: str, *, shifted: bool = False,
                        detailed: bool = False
                        ) -> tuple[dict, list[dict], dict | None]:
    settings.validate()
    if source["source_seed"] not in SOURCES or arm not in ARMS:
        raise ValueError("Source or genotype arm not in frozen protocol")
    stored = source["genotypes"][arm]
    genes = tuple(stored["genes"])
    genome = evolution.to_weights(genes)
    if (evolution.gene_digest(genes) != stored["gene_digest"] or
            base.genome_digest(genome) != stored["native_401_genome_digest"]):
        raise AssertionError("Genetic transplant provenance corrupt")
    eco = base.create_ecology(
        seed, settings, "heritable_no_mutation", shifted=shifted
    )
    for agent in eco.agents.values():
        agent.genome = genome
        agent.hidden = [0.] * settings.hidden
    eco.initial_genome_digests = tuple(
        base.genome_digest(genome) for _ in eco.agents
    )
    schedule = base._food_schedule(seed, settings, eco, False)
    checks = []
    for tick in range(settings.ticks + 1):
        if tick:
            ids = list(eco.agents)
            base._local_rng(seed, tick, 0, 0).shuffle(ids)
            for agent_id in ids:
                base.act(eco, agent_id, tick)
            base._food_regeneration(eco, schedule[tick - 1])
            eco.ticks_elapsed = tick
            if not eco.agents and eco.extinct_at is None:
                eco.extinct_at = tick
        if tick == 0 or tick % settings.checkpoint_every == 0 or tick == settings.ticks:
            ck = base.snapshot(eco, tick)
            ck.update({
                "source_seed": source["source_seed"],
                "split": "shifted" if shifted else "main",
                "seed": seed, "arm": arm,
            })
            checks.append(ck)
            if abs(base.energy_balance(eco)) > 1e-6:
                raise AssertionError("Unchanged M5.0 physical energy ledger broken")
    births = [e for e in eco.events if e["type"] == "birth"]
    if (len(births) != eco.births or any(
        e["genome"] != base.genome_digest(genome)
        for e in births
    )):
        raise AssertionError("M5.3C offspring must clone numeric genome exactly")
    if any(a.genome != genome for a in eco.agents.values()):
        raise AssertionError("A living genotype changed after transplant")
    row = {
        "source_seed": source["source_seed"],
        "split": "shifted" if shifted else "main",
        "seed": seed, "arm": arm,
        "genome_digest": stored["native_401_genome_digest"],
        "gene_digest": stored["gene_digest"],
        "survived": int(bool(eco.agents)),
        "alive_final": len(eco.agents),
        "births": eco.births,
        "deaths": eco.deaths,
        "food_eaten": eco.food_eaten,
        "food_remaining": len(eco.food),
        "max_generation": eco.max_generation,
        "has_generation3": int(eco.max_generation >= 3),
        "extinct_at": eco.extinct_at,
        "motor_mask_corrections": eco.corrected_actions,
        "total_food_energy_gained": eco.cumulative_food_energy,
        "energy_ledger_residual": base.energy_balance(eco),
        "births_are_fixed_clones": 1,
        "all_actions_original_numeric_neural": 1,
        "mutations_during_transplant": 0,
    }
    trace = None
    if detailed:
        trace = {
            "source_seed": source["source_seed"],
            "split": row["split"], "seed": seed, "arm": arm,
            "genotype": list(genes),
            "original_genome_digest": stored["native_401_genome_digest"],
            "walls": [list(p) for p in sorted(eco.world.walls)],
            "initial_food": [list(p) for p in eco.world.initial_food],
            "checkpoints": checks, "events": eco.events,
            "final_agents": [
                {
                    "id": a.id, "parent": a.parent,
                    "founder": a.founder, "generation": a.generation,
                    "position": list(a.position), "energy": a.energy,
                }
                for a in sorted(eco.agents.values(), key=lambda a: a.id)
            ],
        }
    return row, checks, trace


def paired_crossed_cluster_contrast(rows: list[dict], split: str, other: str,
                                  bootstrap_seed: int, replicates: int
                                  ) -> dict:
    """Bootstrap 5 independent selected families AND common heldout worlds."""
    family_ids = list(SOURCES)
    ids = sorted({
        r["seed"] for r in rows if r["split"] == split
    })
    lookup = {
        (r["source_seed"], r["seed"], r["arm"]): r["survived"]
        for r in rows if r["split"] == split
    }
    if not ids or len(lookup) != len(family_ids)*len(ids)*len(ARMS):
        raise AssertionError("Five complete source families x heldout worlds missing")
    if len(ids) != len({
        k[1] for k in lookup if k[2] == "descendant"
    }):
        raise AssertionError("Duplicate heldout trial keys")
    grid = [[
        lookup[(family, seed, "descendant")]-
        lookup[(family, seed, other)]
        for seed in ids
    ] for family in family_ids]
    family_delta = [
        statistics.mean(d) for d in grid
    ]
    mean_delta = statistics.mean(family_delta)
    rng = random.Random(bootstrap_seed)
    n_family, n_world = len(family_ids), len(ids)
    samples = []
    for _ in range(replicates):
        fi = [rng.randrange(n_family) for _ in family_ids]
        wi = [rng.randrange(n_world) for _ in ids]
        samples.append(sum(
            grid[f][w] for f in fi for w in wi
        )/(n_family*n_world))
    samples.sort()
    lower = samples[int(.025*(replicates-1))]
    upper = samples[int(.975*(replicates-1))]
    positive = sum(x > 0 for x in family_delta)
    negative = sum(x < 0 for x in family_delta)
    return {
        "contrast": "descendant_minus_"+other,
        "split": split,
        "mean_delta": mean_delta,
        "crossed_cluster_bootstrap_95_descriptive": [lower,upper],
        "source_families_positive": positive,
        "source_families_negative": negative,
        "source_families_tied": n_family-positive-negative,
        "family_deltas": {
            str(f): family_delta[i] for i,f in enumerate(family_ids)
        },
        "source_family_count": n_family,
        "heldout_world_count": n_world,
        "bootstraps": replicates,
        "bootstrap_seed": bootstrap_seed,
        "all_families_individually_selected_from_prior_successes": True,
        "supported_with_four_families": (
            mean_delta > 0 and lower > 0 and positive >= 4 and
            positive > negative
        ),
    }


def analyze(rows: list[dict], settings: Settings,
            bootstrap_reps: int = BOOT_REPS) -> dict:
    by = {}
    comparison = {}
    for split, world_count in (
        ("main", settings.seeds),
        ("shifted", settings.shifted_seeds),
    ):
        by[split] = {}
        for family in SOURCES:
            by[split][str(family)] = {}
            for arm in ARMS:
                arm_rows = [
                    r for r in rows if r["split"] == split
                    and r["source_seed"] == family and r["arm"] == arm
                ]
                if len(arm_rows) != world_count:
                    raise AssertionError("Incomplete family/arm/world outcomes")
                by[split][str(family)][arm] = {
                    "survival_worlds": sum(r["survived"] for r in arm_rows),
                    "survival_fraction": statistics.mean(
                        r["survived"] for r in arm_rows
                    ),
                    "mean_births": statistics.mean(r["births"] for r in arm_rows),
                    "mean_food_eaten": statistics.mean(
                        r["food_eaten"] for r in arm_rows
                    ),
                    "mean_alive_final": statistics.mean(
                        r["alive_final"] for r in arm_rows
                    ),
                }
        comparison[split] = {
            other: paired_crossed_cluster_contrast(
                rows, split, other,
                BOOT_SEED + (1000 if split == "shifted" else 0)
                + CONTRASTS.index(other)*101, bootstrap_reps
            )
            for other in CONTRASTS
        }
    md = comparison["main"]["ancestor"]
    scramble = comparison["main"]["scrambled"]
    last = comparison["main"]["last_step_reverted"]
    hard = comparison["shifted"]["ancestor"]
    descendants_viable = sum(
        by["main"][str(f)]["descendant"]["survival_worlds"] >= 12
        for f in SOURCES
    )
    hard_viable = sum(
        by["shifted"][str(f)]["descendant"]["survival_worlds"] >= 6
        for f in SOURCES
    )
    return {
        "by_family": by,
        "paired_crossed_cluster_comparisons": comparison,
        "gates": {
            "main_descendant_viable_families_ge12of24":
                descendants_viable,
            "main_primary_ancestor_transfer": bool(
                md["supported_with_four_families"] and
                descendants_viable >= 3
            ),
            "numeric_scramble_specificity": bool(
                scramble["supported_with_four_families"]
            ),
            "last_birth_mutation_causal_ablation": bool(
                last["supported_with_four_families"]
            ),
            "shifted_descendant_viable_families_ge6of12":
                hard_viable,
            "shifted_transfer": bool(
                hard_viable >= 3 and
                hard["supported_with_four_families"]
            ),
        },
        "interpretation": (
            "All comparisons condition on ONLY FIVE previously successful "
            "M5.3B sparse source worlds; hierarchical bootstrap across origin "
            "families and shared new test world seeds. Transplant offspring are "
            "EXACT CLONES: this is a causally controlled numerical phenotype "
            "test, NOT new within-world evolution or an estimate of prevalence "
            "of success in unbiased evolutionary founder populations."
        )
    }


def run(settings: Settings, out: Path,
        bootstrap_reps: int = BOOT_REPS) -> dict:
    settings.validate()
    out.mkdir(parents=True, exist_ok=True)
    # Freeze EVERY candidate using only original prior successful source worlds
    # before evaluating a SINGLE heldout ecological seed.
    selected = recover_candidates()
    if len(selected) != len(SOURCES):
        raise AssertionError("Candidate family count altered")
    bank_path = out/"frozen_source_candidate_bank.json"
    bank_content = json.dumps(selected, indent=2, sort_keys=True)+"\n"
    bank_path.write_text(bank_content, encoding="utf-8")
    for source in selected:
        print("M5.3C SOURCE "+json.dumps({
            k:source[k] for k in (
                "source_seed","selected_final_id",
                "selected_final_generation","selected_final_founder_id",
                "original_source_alive_final","original_source_births",
                "last_step_mutated_indices","genotypes"
            )
        },sort_keys=True),flush=True)
    print("M5.3C SOURCE BANK FROZEN; begin new heldout worlds",flush=True)
    rows, checkpoints, traces = [], [], []
    for split,start,n_worlds,shift in (
        ("main",settings.seed_start,settings.seeds,False),
        ("shifted",settings.shifted_start,settings.shifted_seeds,True),
    ):
        for world_seed in range(start,start+n_worlds):
            for source in selected:
                subset = []
                for arm in ARMS:
                    row,ck,trace = simulate_transplant(
                        world_seed,settings,source,arm,shifted=shift,
                        detailed=(world_seed==start)
                    )
                    rows.append(row);subset.append(row);checkpoints.extend(ck)
                    if trace is not None:
                        traces.append(trace)
                print(
                    f"M5.3C WORLD {split} {world_seed} FAMILY {source['source_seed']} "+
                    " ".join(
                        f"{r['arm']}={r['alive_final']}/{r['births']}"
                        for r in subset
                    ),flush=True
                )
    observed=analyze(rows,settings,bootstrap_reps)
    report={
        "protocol": "docs/M5_3C_PROTOCOL.md; frozen commit 235fa05894c573b76f097073100069c25e172e9b",
        "settings": asdict(settings),
        "origins": list(SOURCES),
        "treatments": list(ARMS),
        "simulation_count": len(rows),
        "candidate_genome_digests": {
            str(s["source_seed"]): {
                arm:s["genotypes"][arm]["native_401_genome_digest"]
                for arm in ARMS
            } for s in selected
        },
        **observed,
    }
    (out/"summary.json").write_text(
        json.dumps(report,indent=2,sort_keys=True)+"\n",
        encoding="utf-8"
    )
    for filename,data in (
        ("world_metrics.csv",rows),
        ("population_checkpoints.csv",checkpoints),
    ):
        with (out/filename).open("w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    for trace in traces:
        (out/f"events_{trace['split']}_{trace['source_seed']}_{trace['arm']}.json"
        ).write_text(json.dumps(trace,indent=2,sort_keys=True)+"\n",
                     encoding="utf-8")
    print("M5.3C GATES "+json.dumps(report["gates"],sort_keys=True),flush=True)
    print("M5.3C COMPARISONS "+json.dumps(
        report["paired_crossed_cluster_comparisons"],sort_keys=True
    ),flush=True)
    print("M5.3C FAMILY OUTCOMES "+json.dumps(
        report["by_family"],sort_keys=True
    ),flush=True)
    return report


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    for f in fields(Settings):
        parser.add_argument("--"+f.name.replace("_","-"),
                            type=type(f.default),default=f.default)
    parser.add_argument("--out",type=Path,default=Path("results/m5-3c"))
    parser.add_argument("--bootstrap-reps",type=int,default=BOOT_REPS)
    options=vars(parser.parse_args())
    out=options.pop("out")
    reps=options.pop("bootstrap_reps")
    if reps<100:
        raise ValueError("Bootstrap requires >=100 repetitions")
    if options["ticks"]>=400 and reps!=BOOT_REPS:
        raise ValueError("Final sealed campaign must use 20,000 resamples")
    run(Settings(**options),out,reps)


if __name__=="__main__":
    main()
