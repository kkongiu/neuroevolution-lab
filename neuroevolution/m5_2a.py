"""M5.2A: preregistered 2x2 factorial movement x reproduction intervention.

Diagnostic experiment only: NN is the unmodified M5.0 actor; GT is the
M5.1 scripted positive control. NT and GN splice one decision component
while retaining original M5.0 physical costs and heritable neural genomes.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from . import m5_0 as base
from . import m5_1 as controls

ARMS = ("NN", "NT", "GN", "GT")
CONTRASTS = (("NT", "NN"), ("GN", "NN"), ("GT", "NT"),
             ("GT", "GN"), ("GT", "NN"))


@dataclass(frozen=True)
class Settings(controls.Settings):
    seed_start: int = 75000
    seeds: int = 20
    shifted_start: int = 76000
    shifted_seeds: int = 10
    negative_control_worlds: int = 1
    no_food_controls: int = 0

    def validate(self) -> None:
        super().validate()
        if self.ticks >= 400 and self.hidden != 12:
            raise ValueError("Full M5.2A campaign must use hidden=12")


def choose(eco: base.Ecology, agent: base.Agent, tick: int,
           arm: str, visits: dict) -> tuple[int | None, dict]:
    """Advance the RNN once if neural decisions are used; return legal action.

    NN is handled by original base.act in simulate, rather than by this adapter.
    """
    if arm not in ("NT", "GN", "GT"):
        raise ValueError(arm)
    blocked, free, senses, motors, can_birth = controls._candidate_actions(
        eco, agent
    )
    threshold_birth = bool(
        can_birth and agent.energy >= eco.settings.scripted_reproduction_threshold
    )
    meta = {"neural_birth_wish": 0,
            "neural_birth_eligible_wish": 0,
            "neural_illegal_preference": 0}
    if arm == "GT":
        return controls.scripted_choice(
            eco, agent, "greedy_odor", tick, visits
        ), meta
    logits = base.network_logits(agent, senses, eco.settings)
    preferred = max(range(base.ACTIONS), key=lambda a: (logits[a], -a))
    legal = motors + ([base.REPRODUCE] if can_birth else [])
    neural_choice = max(legal, key=lambda a: (logits[a], -a)) if legal else None
    meta["neural_birth_wish"] = int(preferred == base.REPRODUCE)
    meta["neural_birth_eligible_wish"] = int(
        neural_choice == base.REPRODUCE
    )
    meta["neural_illegal_preference"] = int(preferred not in legal)
    # In both interventions count the masking correction of the neural actor,
    # distinct from the deliberate script replacement.
    eco.corrected_actions += int(neural_choice != preferred)
    if arm == "NT":
        if threshold_birth:
            return base.REPRODUCE, meta
        return (max(motors, key=lambda a: (logits[a], -a))
                if motors else None), meta
    if neural_choice == base.REPRODUCE:
        return base.REPRODUCE, meta
    return controls.scripted_choice(
        eco, agent, "greedy_no_reproduction", tick, visits
    ), meta


def simulate(seed: int, settings: Settings, arm: str, *,
             shifted: bool = False,
             detailed: bool = False) -> tuple[dict, list[dict], dict | None]:
    settings.validate()
    if arm not in ARMS:
        raise ValueError(arm)
    eco = base.create_ecology(
        seed, settings, "heritable_mutation", shifted=shifted
    )
    schedule = base._food_schedule(seed, settings, eco, False)
    diag = controls.Diagnostics()
    diag.record_initial_visits(eco)
    neural_birth_wish = neural_birth_eligible_wish = 0
    neural_illegal_preference = 0
    threshold_birth_opportunities = 0
    checkpoints = []
    for tick in range(settings.ticks + 1):
        if tick:
            ids = list(eco.agents)
            base._local_rng(seed, tick, 0, 0).shuffle(ids)
            for aid in ids:
                agent = eco.agents.get(aid)
                if agent is None:
                    continue
                _, _, senses, motors, can_birth = controls._candidate_actions(
                    eco, agent
                )
                diag.actions_observed += 1
                diag.energy_before_total += agent.energy
                diag.food_adjacent_opportunities += int(
                    any(senses[8 + a] > .5 for a in motors)
                )
                diag.odor_present_actions += int(
                    any(senses[4 + a] > 0 for a in motors)
                )
                diag.eligible_reproduction_opportunities += int(can_birth)
                threshold_birth_opportunities += int(
                    can_birth and
                    agent.energy >= settings.scripted_reproduction_threshold
                )
                old_pos, old_moves = agent.position, eco.move_actions
                old_births, old_food = eco.births, eco.food_eaten
                if arm == "NN":
                    # Crucial identity control: literally original M5.0 act.
                    preference = controls._neural_preference(eco, agent, senses)
                    neural_birth_wish += int(preference == base.REPRODUCE)
                    neural_illegal_preference += int(
                        preference not in (
                            motors + ([base.REPRODUCE] if can_birth else [])
                        )
                    )
                    base.act(eco, aid, tick)
                    neural_birth_eligible_wish += eco.births - old_births
                else:
                    choice, meta = choose(
                        eco, agent, tick, arm, diag.agent_visits[aid]
                    )
                    neural_birth_wish += meta["neural_birth_wish"]
                    neural_birth_eligible_wish += meta[
                        "neural_birth_eligible_wish"
                    ]
                    neural_illegal_preference += meta[
                        "neural_illegal_preference"
                    ]
                    controls._act_scripted(eco, aid, tick, choice)
                diag.action_births += eco.births - old_births
                diag.action_food_consumed += eco.food_eaten - old_food
                diag.motor_actions += eco.move_actions - old_moves
                survivor = eco.agents.get(aid)
                moved_to = survivor.position if survivor is not None else None
                if moved_to is not None and moved_to != old_pos:
                    visits = diag.agent_visits[aid]
                    previous = visits.get(moved_to, 0)
                    diag.traversed_previously_visited_cells += int(previous > 0)
                    diag.same_cell_revisits += int(previous > 0)
                    visits[moved_to] = previous + 1
                    if motors and any(senses[4 + a] > 0 for a in motors):
                        diag.odor_alignable_moves += 1
                        odor_max = max(senses[4 + a] for a in motors)
                        direction = (moved_to[0] - old_pos[0],
                                     moved_to[1] - old_pos[1])
                        actual = base.DIRECTIONS.index(direction)
                        diag.odor_aligned_moves += int(
                            abs(senses[4 + actual] - odor_max) < 1e-12
                        )
                diag.record_newborns(eco)
            base._food_regeneration(eco, schedule[tick - 1])
            eco.ticks_elapsed = tick
            if not eco.agents and eco.extinct_at is None:
                eco.extinct_at = tick
        if (tick == 0 or tick % settings.checkpoint_every == 0
                or tick == settings.ticks):
            entry = base.snapshot(eco, tick)
            entry.update({"seed": seed, "arm": arm,
                          "split": "shifted" if shifted else "main"})
            checkpoints.append(entry)
            if abs(base.energy_balance(eco)) > 1e-6:
                raise AssertionError("M5.2A physical energy ledger not closed")
    diag_values = diag.as_metrics()
    row = {
        "seed": seed, "split": "shifted" if shifted else "main",
        "arm": arm, "survived": int(bool(eco.agents)),
        "alive_final": len(eco.agents), "births": eco.births,
        "deaths": eco.deaths, "max_generation": eco.max_generation,
        "has_generation3": int(eco.max_generation >= 3),
        "food_eaten": eco.food_eaten, "food_remaining": len(eco.food),
        "extinct_at": eco.extinct_at,
        "motor_mask_corrections": eco.corrected_actions,
        "ineligible_reproduction_preferences":
            eco.reproduction_ineligible_wishes if arm == "NN"
            else neural_birth_wish - neural_birth_eligible_wish,
        "neural_birth_wish": neural_birth_wish,
        "neural_birth_eligible_wish": neural_birth_eligible_wish,
        "neural_illegal_preference": neural_illegal_preference,
        "threshold_birth_opportunities": threshold_birth_opportunities,
        "neural_motor_used": int(arm in ("NN", "NT")),
        "neural_reproduction_used": int(arm in ("NN", "GN")),
        "energy_ledger_residual": base.energy_balance(eco),
        **diag_values,
    }
    if diag.action_births != eco.births or diag.action_food_consumed != eco.food_eaten:
        raise AssertionError("Action diagnostics do not match physical events")
    trace = None
    if detailed:
        trace = {
            "seed": seed, "split": row["split"], "arm": arm,
            "walls": [list(p) for p in sorted(eco.world.walls)],
            "initial_food": [list(p) for p in eco.world.initial_food],
            "initial_founder_genomes": list(eco.initial_genome_digests),
            "events": eco.events, "checkpoints": checkpoints,
            "final_agents": [
                {"id": a.id, "parent": a.parent, "founder": a.founder,
                 "generation": a.generation, "position": list(a.position),
                 "energy": a.energy, "genome_digest": base.genome_digest(a.genome)}
                for a in sorted(eco.agents.values(), key=lambda x: x.id)
            ],
        }
    return row, checkpoints, trace


def comparisons(rows: list[dict], split: str, metric: str,
                seed: int) -> dict:
    by = {
        arm: {r["seed"]: r[metric] for r in rows
              if r["split"] == split and r["arm"] == arm}
        for arm in ARMS
    }
    if any(not by[a] for a in ARMS):
        raise AssertionError("Missing factorial arm")
    keys = sorted(by["NN"])
    if any(sorted(by[arm]) != keys for arm in ARMS):
        raise AssertionError("Unpaired factorial worlds")
    results = {}
    for i, (a, b) in enumerate(CONTRASTS):
        vals = [by[a][k] - by[b][k] for k in keys]
        lo, hi = base.paired_bootstrap(vals, seed + i * 101)
        results[a + "_minus_" + b] = {
            "mean_delta": statistics.mean(vals),
            "bootstrap_95_descriptive": [lo, hi],
            "positive_worlds": sum(v > 0 for v in vals),
            "negative_worlds": sum(v < 0 for v in vals),
            "tied_worlds": sum(v == 0 for v in vals),
            "supported": bool(
                statistics.mean(vals) > 0 and lo > 0 and
                sum(v > 0 for v in vals) > sum(v < 0 for v in vals)
            ),
        }
    interaction = [
        (by["GT"][k] - by["GN"][k]) -
        (by["NT"][k] - by["NN"][k])
        for k in keys
    ]
    results["interaction_threshold_by_motor"] = {
        "mean_delta": statistics.mean(interaction),
        "bootstrap_95_descriptive": list(
            base.paired_bootstrap(interaction, seed + 977)
        ),
        "interpretation": "Difference-in-differences on chosen endpoint scale",
    }
    return results


def render_population_svg(checkpoints: list[dict], settings: Settings) -> str:
    # Export self-contained plot, based only on real world-level trajectories.
    width, height, left, top, bottom = 740, 410, 60, 45, 315
    ticks = sorted({
        ck["tick"] for ck in checkpoints if ck["split"] == "main"
    })
    maximum = max(settings.initial_population, max(
        ck["population"] for ck in checkpoints if ck["split"] == "main"
    ))
    palette = {"NN": "#7c3aed", "NT": "#2563eb",
               "GN": "#ea580c", "GT": "#0f766e"}
    lines = []
    for arm in ARMS:
        vals = [
            statistics.mean(
                ck["population"] for ck in checkpoints
                if ck["split"] == "main" and ck["tick"] == tick
                and ck["arm"] == arm
            ) for tick in ticks
        ]
        xy = " ".join(
            f"{left + t / settings.ticks * (width - left - 30):.1f},"
            f"{bottom - v / maximum * (bottom - top):.1f}"
            for t, v in zip(ticks, vals)
        )
        lines.append(
            f'<polyline points="{xy}" fill="none" '
            f'stroke="{palette[arm]}" stroke-width="3"/>'
        )
    legend = "".join(
        f'<text x="{left + i * 150}" y="368" fill="{palette[arm]}" '
        f'font-family="sans-serif" font-size="13">{arm}</text>'
        for i, arm in enumerate(ARMS)
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="white"/>'
        '<text x="60" y="25" font-family="sans-serif" font-size="16">'
        'M5.2A population trajectories (main worlds)</text>'
        f'<line x1="{left}" x2="{left}" y1="{top}" y2="{bottom}" '
        'stroke="black"/>'
        f'<line x1="{left}" x2="{width - 30}" y1="{bottom}" '
        f'y2="{bottom}" stroke="black"/>'
        + "".join(lines) + legend + '</svg>\n'
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
            for arm in ARMS:
                row, check, trace = simulate(
                    seed, settings, arm, shifted=shifted,
                    detailed=(seed == start)
                )
                rows.append(row)
                checks.extend(check)
                if trace is not None:
                    traces.append(trace)
            print(f"M5.2A {split} {seed}: " + " ".join(
                f"{r['arm']}={r['alive_final']}/{r['births']}"
                for r in rows[-len(ARMS):]
            ), flush=True)
    mean = {
        split: {
            arm: {
                measure: statistics.mean(
                    r[measure] for r in rows
                    if r["split"] == split and r["arm"] == arm
                )
                for measure in (
                    "survived", "alive_final", "births", "food_eaten",
                    "max_generation", "food_remaining",
                    "food_per_100_movement_actions",
                    "neural_illegal_preference"
                )
            } for arm in ARMS
        } for split in ("main", "shifted")
    }
    summary = {
        "protocol": "M5.2A; docs/M5_2A_PROTOCOL.md",
        "settings": asdict(settings),
        "means": mean,
        "main_primary_survival": comparisons(
            rows, "main", "survived", settings.seed_start + 13
        ),
        "shifted_survival": comparisons(
            rows, "shifted", "survived", settings.shifted_start + 13
        ),
        "exploratory_main_food": comparisons(
            rows, "main", "food_eaten", settings.seed_start + 213
        ),
        "interpretation_limits": [
            "A scripted movement improvement is not evolved behavior",
            "Threshold reproduction uses privileged experimenter age/energy",
            "No autonomous neural-genome fitness advantage inferred",
            "Whole worlds, not organism actions, are independent units",
            "GT and GN's scripted motor does not depend on neural genotype",
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
        name = trace["split"] + "_" + trace["arm"]
        (output / f"events_{name}.json").write_text(
            json.dumps(trace, indent=2, sort_keys=True) + "\n",
            encoding="utf-8"
        )
    (output / "population_curves.svg").write_text(
        render_population_svg(checks, settings), encoding="utf-8"
    )
    print("M5.2A PRIMARY", summary["main_primary_survival"], flush=True)
    print("M5.2A MEANS", summary["means"], flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for f in fields(Settings):
        option = "--" + f.name.replace("_", "-")
        parser.add_argument(option, type=type(f.default), default=f.default)
    parser.add_argument("--out", type=Path, default=Path("results/m5-2a"))
    args = vars(parser.parse_args())
    output = args.pop("out")
    run(Settings(**args), output)


if __name__ == "__main__":
    main()
