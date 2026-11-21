"""M5.5C — independent transplant of actual born mutations vs their OWN founder.

The source bank is frozen and committed before any heldout transfer execution.
M5.5A and M5.5B remain unchanged. Fear is programmer-supplied, not learned.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from . import m5_0 as base
from . import m5_1 as physics
from . import m5_5a as original_innate
from . import m5_5b as predator
from . import m5_4b0 as stats

PROTOCOL_COMMIT = "5f8d2b803b3a30143027483c4e071265b5c7851e"
SOURCE_STUDY_COMMIT = "b9d0f4bf00b667dae68249214e6db4616db33d37"
SOURCE_SEEDS = tuple(range(94000, 94008))
TRANSFER_MAIN = tuple(range(94200, 94224))
TRANSFER_HARD = tuple(range(94300, 94312))
ARMS = (
    "mutant_descendant", "own_original_founder", "full_gene_reversion",
    "fear_gene_reversion", "fear_gene_only", "matched_random_genome",
)
MUTANT, FOUNDER, FULL_REVERT, FEAR_REVERT, FEAR_ONLY, RANDOM = ARMS
CONTRASTS = (
    ("mutant_minus_own_founder", MUTANT, FOUNDER),
    ("mutant_minus_fear_reverted", MUTANT, FEAR_REVERT),
    ("fear_only_minus_own_founder", FEAR_ONLY, FOUNDER),
    ("mutant_minus_matched_random", MUTANT, RANDOM),
    ("full_reversion_minus_own_founder", FULL_REVERT, FOUNDER),
)
BOOT_REPS = 20000


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def _canonical_source_rows(archive: Path) -> dict[int, dict]:
    with archive.open(newline="", encoding="utf-8") as f:
        source = list(csv.DictReader(f))
    rows = {}
    for row in source:
        if (row["split"] == "main"
                and row["arm"] == predator.SPARSE
                and int(row["predators"]) == 3
                and int(row["seed"]) in SOURCE_SEEDS):
            seed = int(row["seed"])
            if seed in rows:
                raise AssertionError("Duplicate original source trial")
            rows[seed] = row
    if set(rows) != set(SOURCE_SEEDS):
        raise AssertionError("Eight original source predator worlds missing")
    return rows


def freeze_source(archive: Path) -> dict:
    """Replay ONLY ORIGINAL 94000..94007 source worlds; no heldout transfers."""
    old = predator.Settings()
    old.validate()
    known = _canonical_source_rows(archive)
    candidates = []
    for seed in SOURCE_SEEDS:
        row, checkpoints, trace = predator.simulate(
            seed, old, predator.SPARSE, 3, detailed=True)
        previous = known[seed]
        for key in ("world_sha", "founder_sha", "original_world_sha"):
            if row[key] != previous[key]:
                raise AssertionError(f"Source {seed} original {key} mismatch")
        for key in ("births", "predation_deaths", "alive_final", "mutated_births"):
            if row[key] != int(previous[key]):
                raise AssertionError(f"Source {seed} original {key} mismatch")
        if abs(row["population_auc"] - float(previous["population_auc"])) > 1e-12:
            raise AssertionError("Original source ecology population AUC drift")
        events = trace["full_birth_genealogy"]
        births_by_parent = Counter(int(e["parent"]) for e in events)
        choices = []
        for event in events:
            g = tuple(event["newborn_genes"])
            parent_genes = tuple(event["parent_genes"])
            founder_id = int(event["founder"])
            founder = tuple(trace["source_founder_genes"][founder_id])
            if (8 not in event["changed_indices"]
                    or g[8] == parent_genes[8]
                    or g[8] == founder[8]
                    or g == founder):
                continue
            child = int(event["child"])
            n_children = births_by_parent[child]
            choices.append(((-n_children, int(event["tick"]), child),
                            event, founder, n_children))
        if not choices:
            raise AssertionError(
                f"Source {seed} has no actually born g8 mutant; no fallback")
        _, selected, founder, n_children = min(choices, key=lambda x: x[0])
        source_mutant = tuple(selected["newborn_genes"])
        if len(source_mutant) != 9 or len(founder) != 9:
            raise AssertionError("Corrupted exact source nine-gene lineage")
        random_rng = random.Random(seed * 71917 + 55551)
        random_control = tuple(random_rng.gauss(0., .65) for _ in range(9))
        candidates.append({
            "source_seed": seed,
            "source_world_sha": row["world_sha"],
            "source_initial_founder_bank_sha": row["founder_sha"],
            "source_original_world_sha": row["original_world_sha"],
            "source_original_trial_births": row["births"],
            "source_original_trial_predation_deaths": row["predation_deaths"],
            "source_original_trial_population_auc": row["population_auc"],
            "own_founder_id": int(selected["founder"]),
            "newborn_child_id": int(selected["child"]),
            "newborn_birth_tick": int(selected["tick"]),
            "newborn_generation": int(selected["generation"]),
            "source_direct_offspring_count": n_children,
            "source_changed_gene_indices": selected["changed_indices"],
            "own_original_founder_genes": list(founder),
            "actual_mutant_descendant_genes": list(source_mutant),
            "matched_random_control_genes": list(random_control),
            "founder_mutant_pair_sha256": digest({
                "founder": founder, "mutant": source_mutant
            }),
        })
    if [x["source_seed"] for x in candidates] != list(SOURCE_SEEDS):
        raise AssertionError("Independent source families lost")
    return {
        "protocol_commit": PROTOCOL_COMMIT,
        "M5_5B_source_run_commit": SOURCE_STUDY_COMMIT,
        "source_arm": predator.SPARSE,
        "source_predators": 3,
        "original_source_world_seeds": list(SOURCE_SEEDS),
        "heldout_transfer_worlds_examined": 0,
        "source_selection_rule": (
            "One physically born g8-mutated child per source world; "
            "maximize number of later actual own offspring, then "
            "minimize newborn birth tick and child id. Source-only."
        ),
        "candidate_bank_sha256": digest(candidates),
        "candidates": candidates,
    }


def genotypes(source: dict) -> dict[str, tuple[float, ...]]:
    founder = tuple(source["own_original_founder_genes"])
    mutant = tuple(source["actual_mutant_descendant_genes"])
    control = tuple(source["matched_random_control_genes"])
    for g in (founder, mutant, control):
        predator.gene_hash(g)
    if mutant == founder or mutant[8] == founder[8]:
        raise AssertionError("No genuine mutated fear trait in chosen family")
    if source["founder_mutant_pair_sha256"] != digest({
            "founder": founder, "mutant": mutant}):
        raise AssertionError("Frozen source genetic pair was modified")
    mapping = {
        MUTANT: mutant,
        FOUNDER: founder,
        FULL_REVERT: tuple(founder),
        FEAR_REVERT: mutant[:8] + (founder[8],),
        FEAR_ONLY: founder[:8] + (mutant[8],),
        RANDOM: control,
    }
    if mapping[FULL_REVERT] != mapping[FOUNDER]:
        raise AssertionError("Full gene reversion is not the original founder")
    return mapping


def load_frozen_bank(bank_path: Path) -> dict:
    with bank_path.open(encoding="utf-8") as f:
        bank = json.load(f)
    if (bank.get("protocol_commit") != PROTOCOL_COMMIT or
            bank.get("M5_5B_source_run_commit") != SOURCE_STUDY_COMMIT or
            bank.get("original_source_world_seeds") != list(SOURCE_SEEDS) or
            bank.get("heldout_transfer_worlds_examined") != 0 or
            len(bank.get("candidates", [])) != len(SOURCE_SEEDS) or
            bank.get("candidate_bank_sha256") != digest(bank["candidates"])):
        raise AssertionError("Only previously committed exact source bank allowed")
    if [c["source_seed"] for c in bank["candidates"]] != list(SOURCE_SEEDS):
        raise AssertionError("Missing independent source source-family")
    for c in bank["candidates"]:
        genotypes(c)
    return bank


def transfer_world(seed: int, source: dict, arm: str,
                   *, hard: bool, ticks: int = 400,
                   checkpoint_every: int = 50) -> tuple[dict, list[dict]]:
    """One original-physics clone transplant with three real mobile enemies."""
    if arm not in ARMS:
        raise ValueError("Unregistered intervention")
    if ticks == 400:
        permitted = TRANSFER_HARD if hard else TRANSFER_MAIN
        if seed not in permitted:
            raise ValueError("Transfer must use prospectively reserved worlds")
    settings = predator.Settings()
    genes = genotypes(source)[arm]
    eco = base.create_ecology(seed, settings, "heritable_no_mutation",
                              shifted=hard)
    founder_positions = tuple(a.position for a in eco.agents.values())
    for agent in eco.agents.values():
        agent.genome = genes
        agent.hidden = [0.] * settings.hidden
    eco.initial_genome_digests = tuple(base.genome_digest(genes)
                                       for _ in eco.agents)
    schedule = base._food_schedule(seed, settings, eco, False)
    enemies = predator.initial_predators(eco, 3)
    starts = tuple(enemies)
    world_geom = original_innate._geometry_sha(
        eco, schedule, founder_positions)
    world_sha = digest({"original_world_sha": world_geom,
                        "predator_count": 3, "predator_starts": starts})
    visits = {a.id: {a.position: 1} for a in eco.agents.values()}
    heading = {a.id: None for a in eco.agents.values()}
    killed = 0
    observed_enemies = 0
    checkpoints = []
    for tick in range(ticks + 1):
        if tick:
            agent_ids = list(eco.agents)
            base._local_rng(seed, tick, 0, 0).shuffle(agent_ids)
            for aid in agent_ids:
                a = eco.agents.get(aid)
                if a is None:
                    continue
                previous = a.position
                old_births, old_moves, new_id = (
                    eco.births, eco.move_actions, eco.next_id)
                action, obs = predator.choose_action(
                    eco, a, tick, visits[aid], heading[aid], enemies,
                    predator.CLONE)
                observed_enemies += int(obs["enemies_visible"] > 0)
                physics._act_scripted(eco, aid, tick, action)
                born = eco.births > old_births
                moved = eco.move_actions > old_moves
                if born != (action == base.REPRODUCE):
                    raise AssertionError("Native birth action mismatch")
                if moved != (action in range(4) if action is not None else False):
                    raise AssertionError("Native movement action mismatch")
                if born:
                    child = eco.agents[new_id]
                    if child.genome != genes:
                        raise AssertionError("Hidden adaptation during transplant")
                    visits[new_id] = {child.position: 1}
                    heading[new_id] = None
                if moved:
                    dx, dy = base.DIRECTIONS[action]
                    target = (previous[0] + dx, previous[1] + dy)
                    if aid in eco.agents and eco.agents[aid].position != target:
                        raise AssertionError("Incorrect original native move")
                    visits[aid][target] = visits[aid].get(target, 0) + 1
                    heading[aid] = action
            before = len(eco.agents)
            kills_tick = predator.step_predators(eco, enemies, tick)
            if before - len(eco.agents) != kills_tick:
                raise AssertionError("Predation count is not physical")
            killed += kills_tick
            base._food_regeneration(eco, schedule[tick - 1])
            eco.ticks_elapsed = tick
            if not eco.agents and eco.extinct_at is None:
                eco.extinct_at = tick
        if (tick == 0 or tick % checkpoint_every == 0 or tick == ticks):
            snap = base.snapshot(eco, tick)
            if abs(snap["energy_ledger_residual"]) > 1e-6:
                raise AssertionError("Transfer energy conservation violation")
            snap.update({
                "source_seed": source["source_seed"],
                "transfer_seed": seed, "split": "hard" if hard else "main",
                "arm": arm, "predators": 3, "world_sha": world_sha,
                "source_genotype_pair_sha": source["founder_mutant_pair_sha256"],
                "genotype_sha": predator.gene_hash(genes),
                "physical_predation_deaths": killed,
            })
            checkpoints.append(snap)
    if eco.births != eco.birth_actions:
        raise AssertionError("Reproductive physical cost missing")
    if len([e for e in eco.events if e["type"] == "predation"]) != killed:
        raise AssertionError("Physical predator loss records missing")
    if len([e for e in eco.events if e["type"] == "birth"]) != eco.births:
        raise AssertionError("Clonal physical births missing")
    if any(a.genome != genes for a in eco.agents.values()):
        raise AssertionError("In-transplant evolutionary gene change")
    row = {
        "source_seed": source["source_seed"],
        "original_founder_id": source["own_founder_id"],
        "source_mutant_child_id": source["newborn_child_id"],
        "transfer_seed": seed, "split": "hard" if hard else "main",
        "arm": arm, "predators": 3,
        "world_sha": world_sha,
        "original_world_sha": world_geom,
        "source_genotype_pair_sha": source["founder_mutant_pair_sha256"],
        "genotype_sha": predator.gene_hash(genes),
        "predator_initial_positions": json.dumps(starts),
        "survived": int(bool(eco.agents)),
        "alive_final": len(eco.agents),
        "births": eco.births, "food_eaten": eco.food_eaten,
        "deaths": eco.deaths, "predation_deaths": killed,
        "observed_enemies": observed_enemies,
        "population_auc": statistics.mean(x["population"]
                                          for x in checkpoints),
        "energy_ledger_residual": base.energy_balance(eco),
        "original_costed_births": eco.birth_actions,
        "transfer_birth_mutations": 0,
    }
    return row, checkpoints


def analyze(rows: list[dict], bank: dict,
            main: tuple[int, ...], hard: tuple[int, ...],
            bootstrap_reps: int = BOOT_REPS) -> dict:
    family_ids = [x["source_seed"] for x in bank["candidates"]]
    result = {}
    for split, seeds, bootstrap_seed in (
        ("main", main, 95701), ("hard", hard, 95702)
    ):
        chosen = [x for x in rows if x["split"] == split]
        if len(chosen) != len(family_ids) * len(seeds) * len(ARMS):
            raise AssertionError("Missing independent transplant factorial")
        by = {(r["source_seed"], r["transfer_seed"], r["arm"]): r
              for r in chosen}
        if len(by) != len(chosen):
            raise AssertionError("Duplicate transplant")
        for source_seed in family_ids:
            for seed in seeds:
                trials = [by[(source_seed, seed, arm)] for arm in ARMS]
                if len({r["world_sha"] for r in trials}) != 1:
                    raise AssertionError("Different world geometry between arms")
                if len({r["predator_initial_positions"] for r in trials}) != 1:
                    raise AssertionError("Different predator starts between arms")
                if len({r["source_genotype_pair_sha"] for r in trials}) != 1:
                    raise AssertionError("Different original source family")
                if trials[ARMS.index(FOUNDER)]["genotype_sha"] != trials[
                        ARMS.index(FULL_REVERT)]["genotype_sha"]:
                    raise AssertionError("Full reversion does not match founder")
                for metric in ("survived", "alive_final", "births",
                               "food_eaten", "deaths", "predation_deaths",
                               "observed_enemies", "population_auc"):
                    if by[(source_seed, seed, FOUNDER)][metric] != by[
                            (source_seed, seed, FULL_REVERT)][metric]:
                        raise AssertionError("Full reversion audit failed: "+metric)
        averages = {
            arm: {
                "survival_worlds": sum(
                    by[(family, seed, arm)]["survived"]
                    for family in family_ids for seed in seeds),
                "mean_population_auc": statistics.mean(
                    by[(family, seed, arm)]["population_auc"]
                    for family in family_ids for seed in seeds),
                "mean_births": statistics.mean(
                    by[(family, seed, arm)]["births"]
                    for family in family_ids for seed in seeds),
                "mean_predation_deaths": statistics.mean(
                    by[(family, seed, arm)]["predation_deaths"]
                    for family in family_ids for seed in seeds),
            } for arm in ARMS
        }
        comparisons = {}
        for i, (label, left, right) in enumerate(CONTRASTS):
            for measure, m_offset in (("survived", 0), ("population_auc", 10000)):
                # Fixed source family is the resampling unit; nested transfer
                # worlds only improve each family's estimate.
                deltas = [
                    statistics.mean(
                        by[(family, seed, left)][measure] -
                        by[(family, seed, right)][measure]
                        for seed in seeds)
                    for family in family_ids
                ]
                lo, hi = stats.paired_bootstrap(
                    deltas, bootstrap_seed + i * 1000 + m_offset,
                    bootstrap_reps)
                comparisons[label+"_"+measure] = {
                    "mean_family_paired_effect": statistics.mean(deltas),
                    "source_family_bootstrap_95": [lo, hi],
                    "per_independent_source_family": dict(zip(
                        map(str, family_ids), deltas)),
                    "positive_source_families": sum(d > 0 for d in deltas),
                    "strictly_positive_lower95": bool(lo > 0),
                }
        result[split] = {
            "independent_source_families": len(family_ids),
            "nested_transfer_seeds_per_source": len(seeds),
            "arms": averages,
            "comparisons": comparisons,
        }
    main_pairs = result["main"]["comparisons"]
    hard_pairs = result["hard"]["comparisons"]
    gate = (
        main_pairs["mutant_minus_own_founder_survived"]
        ["strictly_positive_lower95"]
        and main_pairs["mutant_minus_own_founder_population_auc"]
        ["strictly_positive_lower95"]
        and hard_pairs["mutant_minus_own_founder_population_auc"]
        ["strictly_positive_lower95"]
        and main_pairs["mutant_minus_matched_random_population_auc"]
        ["strictly_positive_lower95"]
        and main_pairs["mutant_minus_own_founder_population_auc"]
        ["positive_source_families"] >= 6
    )
    return {
        **result, "gates": {
            "prospective_strong_mutation_advantage_all_requirements_met": bool(gate),
            "fear_gene_gain_mechanism_supported": bool(
                main_pairs["mutant_minus_fear_reverted_population_auc"]
                ["strictly_positive_lower95"]
            ),
            "full_genotype_gene_reversion_exact_identity": True,
            "all_transfer_births_exact_clones": True,
            "new_avoidance_behavior_evolved": False,
            "source_family_selection_conditioning_reported": True,
        },
        "inference_warning": (
            "Eight independently selected source families, not independent "
            "world/family combinations. All behavior modules are human designed; "
            "a positive transplant effect would establish genotype causation "
            "under artificial predator ecology, not an invented sensory rule."
        ),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    headers = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def run(bank_path: Path, out: Path, *, bootstrap_reps: int = BOOT_REPS,
        main: tuple[int, ...] = TRANSFER_MAIN,
        hard: tuple[int, ...] = TRANSFER_HARD,
        ticks: int = 400) -> dict:
    if ticks == 400 and (main != TRANSFER_MAIN or hard != TRANSFER_HARD
                         or bootstrap_reps != BOOT_REPS):
        raise ValueError("Full transfer is prospectively frozen")
    if ticks < 1 or bootstrap_reps < 100:
        raise ValueError("Invalid developmental transfer or bootstrap")
    if not set(main).isdisjoint(hard):
        raise ValueError("Shared main/hard worlds")
    bank = load_frozen_bank(bank_path)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    checks = []
    for source in bank["candidates"]:
        for split, seed_set, harder in (
            ("main", main, False), ("hard", hard, True)
        ):
            for seed in seed_set:
                for arm in ARMS:
                    row, ck = transfer_world(
                        seed, source, arm, hard=harder, ticks=ticks,
                        checkpoint_every=50 if ticks >= 400 else 10)
                    rows.append(row)
                    checks.extend(ck)
                    print("M5.5C WORLD "+json.dumps(row, sort_keys=True),
                          flush=True)
    outcome = {
        "protocol_commit": PROTOCOL_COMMIT,
        "frozen_bank_sha256": bank["candidate_bank_sha256"],
        "source_worlds": len(bank["candidates"]),
        "independent_transfer_main_worlds": len(main),
        "independent_transfer_hard_worlds": len(hard),
        "full_transfer_trials": len(rows),
        "population_checkpoints": len(checks),
        "physical_ticks_per_world": ticks,
        "bootstrap_reps": bootstrap_reps,
        **analyze(rows, bank, main, hard, bootstrap_reps),
    }
    if ticks == 400 and (
            len(rows) != 1728 or len(checks) != 15552):
        raise AssertionError("Frozen complete transplant counts altered")
    write_csv(out/"all_original_transplants.csv", rows)
    write_csv(out/"all_original_checkpoints.csv", checks)
    (out/"summary.json").write_text(
        json.dumps(outcome, sort_keys=True, indent=2)+"\n",
        encoding="utf-8")
    print("M5.5C SUMMARY "+json.dumps(outcome, sort_keys=True), flush=True)
    return outcome


def main_cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-source", action="store_true")
    parser.add_argument("--archive", type=Path, default=Path(
        "reports/M5_5B_all_180_original_predator_world_outcomes_2026-11-21.csv"))
    parser.add_argument("--bank", type=Path, default=Path(
        "reports/M5_5C_FROZEN_SOURCE_CANDIDATES_2026-11-21.json"))
    parser.add_argument("--out", type=Path, default=Path("results/m5-5c"))
    parser.add_argument("--bootstrap-reps", type=int, default=BOOT_REPS)
    args = parser.parse_args()
    if args.freeze_source:
        result = freeze_source(args.archive)
        # Source-only mode does not look at any transfer worlds or outcomes.
        print("M5.5C FROZEN SOURCE BANK "+json.dumps(
            result, sort_keys=True, separators=(",", ":")), flush=True)
        return
    run(args.bank, args.out, bootstrap_reps=args.bootstrap_reps)


if __name__ == "__main__":
    main_cli()
