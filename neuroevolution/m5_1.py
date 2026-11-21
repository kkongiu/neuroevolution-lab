"""M5.1: diagnose food-search and population viability under frozen M5.0 physics.

Neural heredity remains unchanged. Separate *scripted* controllers test whether
the existing ecology can be inhabited and forage enough; no scripted behavior
is injected into a neural genome. Uses M5.0 world, births, food schedule,
reproduction energetics, energy conservation and physical controls.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from . import m5_0 as base

ACTIVE_ARMS = ("neural_heritable", "greedy_odor", "greedy_novelty", "random_legal")
NEGATIVE_ARMS = ("greedy_no_reproduction", "greedy_no_regrowth")
ALL_ARMS = ACTIVE_ARMS + NEGATIVE_ARMS
DIAGNOSTICS = (
    "actions_observed", "mean_energy_at_action", "motor_actions",
    "food_adjacent_opportunities", "odor_present_actions",
    "odor_aligned_moves", "odor_alignable_moves",
    "traversed_previously_visited_cells", "same_cell_revisits",
    "eligible_reproduction_opportunities", "requested_reproduction",
)


@dataclass(frozen=True)
class Settings(base.Settings):
    seed_start: int = 73000
    seeds: int = 16
    shifted_start: int = 74000
    shifted_seeds: int = 8
    no_food_controls: int = 0
    ticks: int = 400
    scripted_reproduction_threshold: float = 46.0
    negative_control_worlds: int = 4

    def validate(self) -> None:
        super().validate()
        if self.scripted_reproduction_threshold < self.min_reproduction_energy:
            raise ValueError("Scripted threshold cannot bypass physical eligibility")
        if self.scripted_reproduction_threshold > self.max_energy:
            raise ValueError("Scripted threshold exceeds energy cap")
        if self.negative_control_worlds < 1 or self.negative_control_worlds > self.seeds:
            raise ValueError("Negative controls must be subset of main seeds")


@dataclass
class Diagnostics:
    actions_observed: int = 0
    energy_before_total: float = 0.0
    motor_actions: int = 0
    food_adjacent_opportunities: int = 0
    odor_present_actions: int = 0
    odor_aligned_moves: int = 0
    odor_alignable_moves: int = 0
    traversed_previously_visited_cells: int = 0
    same_cell_revisits: int = 0
    eligible_reproduction_opportunities: int = 0
    requested_reproduction: int = 0
    action_births: int = 0
    action_food_consumed: int = 0
    agent_visits: dict[int, dict[base.Point, int]] = field(default_factory=dict)

    def record_initial_visits(self, eco: base.Ecology) -> None:
        self.agent_visits = {a.id: {a.position: 1} for a in eco.agents.values()}

    def record_newborns(self, eco: base.Ecology) -> None:
        for agent in eco.agents.values():
            if agent.id not in self.agent_visits:
                self.agent_visits[agent.id] = {agent.position: 1}

    def as_metrics(self) -> dict:
        count = self.actions_observed
        return {
            "actions_observed": count,
            "mean_energy_at_action": self.energy_before_total / count if count else 0.,
            "motor_actions": self.motor_actions,
            "food_adjacent_opportunities": self.food_adjacent_opportunities,
            "odor_present_actions": self.odor_present_actions,
            "odor_aligned_moves": self.odor_aligned_moves,
            "odor_alignable_moves": self.odor_alignable_moves,
            "odor_alignment_rate": (
                self.odor_aligned_moves / self.odor_alignable_moves
                if self.odor_alignable_moves else 0.
            ),
            "traversed_previously_visited_cells":
                self.traversed_previously_visited_cells,
            "same_cell_revisits": self.same_cell_revisits,
            "eligible_reproduction_opportunities":
                self.eligible_reproduction_opportunities,
            "requested_reproduction": self.requested_reproduction,
            "action_births": self.action_births,
            "action_food_consumed": self.action_food_consumed,
            "food_per_100_movement_actions": (
                100. * self.action_food_consumed / self.motor_actions
                if self.motor_actions else 0.
            ),
            "odorless_action_fraction": (
                1. - self.odor_present_actions / count if count else 0.
            ),
        }


def _candidate_actions(eco: base.Ecology, agent: base.Agent):
    blocked, free = base.blocked_and_free(eco, agent)
    sensors = base.observe(eco, agent, blocked)
    legal = [action for action, b in enumerate(blocked) if not b]
    reproduction_ok = base.can_reproduce(eco, agent, free)
    return blocked, free, sensors, legal, reproduction_ok


def scripted_choice(eco: base.Ecology, agent: base.Agent, arm: str,
                    tick: int, visits: dict[base.Point, int]) -> int | None:
    """Scripted local food search; never inspect a global food coordinate."""
    if arm not in ALL_ARMS or arm == "neural_heritable":
        raise ValueError("Scripted choice requires a scripted arm")
    blocked, free, sensors, legal, reproduction_ok = _candidate_actions(eco, agent)
    if (arm != "greedy_no_reproduction" and reproduction_ok and
            agent.energy >= eco.settings.scripted_reproduction_threshold):
        return base.REPRODUCE
    if not legal:
        return None
    rng = base._local_rng(eco.seed, tick, agent.id, 37)
    if arm == "random_legal":
        return rng.choice(legal)
    if arm == "greedy_novelty":
        x, y = agent.position
        scores = {}
        for action in legal:
            dx, dy = base.DIRECTIONS[action]
            p = (x + dx, y + dy)
            scores[action] = (
                4. * sensors[8 + action] + sensors[4 + action]
                + .35 / (1. + visits.get(p, 0))
            )
        top = max(scores.values())
        return rng.choice([a for a in legal if abs(scores[a] - top) < 1e-12])
    edible = [a for a in legal if sensors[8 + a] > 0.5]
    if edible:
        return rng.choice(edible)
    best = max(sensors[4 + a] for a in legal)
    if best == 0:
        return rng.choice(legal)
    return rng.choice([
        a for a in legal if abs(sensors[4 + a] - best) < 1e-12
    ])


def _neural_preference(eco: base.Ecology, agent: base.Agent,
                       sensors: tuple[float, ...]) -> int:
    """Diagnostic shadow forward pass, restoring state before real M5.0 act."""
    saved = list(agent.hidden)
    try:
        logits = base.network_logits(agent, sensors, eco.settings)
        return max(range(base.ACTIONS), key=lambda a: (logits[a], -a))
    finally:
        agent.hidden = saved


def _act_scripted(eco: base.Ecology, agent_id: int, tick: int,
                  action: int | None) -> None:
    """Apply exactly M5.0's physics with action choice from scripted control.

    Deliberately retained as a separate adapter: published M5.0 act remains
    unchanged and this is audited against the same energy ledger.
    """
    agent = eco.agents[agent_id]
    _, free, _, legal, reproduce_ok = _candidate_actions(eco, agent)
    if action is not None and action not in legal and not (
        action == base.REPRODUCE and reproduce_ok
    ):
        raise AssertionError("Policy selected physically impossible move/birth")
    base._charge(eco, agent)
    agent.last_failed = 0
    agent.last_food = 0
    if action is None:
        eco.skipped_moves += 1
    elif action == base.REPRODUCE:
        if agent.energy < eco.settings.reproduction_energy_cost:
            raise AssertionError("Reproduction energetics violated")
        child_pos = base._local_rng(eco.seed, tick, agent.id, 1).choice(free)
        if child_pos in {a.position for a in eco.agents.values()}:
            raise AssertionError("Newborn placed on another organism")
        child_id = eco.next_id
        eco.next_id += 1
        offspring_rng = base._local_rng(eco.seed, tick, agent.id, 2)
        child_genome = base.make_child_genome(
            agent.genome, eco.settings, eco.arm, offspring_rng
        )
        eco.cosine_births.append(base.genome_cosine(agent.genome, child_genome))
        eco.cumulative_birth_dissipation += (
            eco.settings.reproduction_energy_cost - eco.settings.newborn_energy
        )
        agent.energy -= eco.settings.reproduction_energy_cost
        child = base.Agent(
            id=child_id, founder=agent.founder, parent=agent.id,
            generation=agent.generation + 1, position=child_pos,
            energy=eco.settings.newborn_energy, genome=child_genome,
            hidden=[0.] * eco.settings.hidden,
        )
        eco.agents[child_id] = child
        agent.births += 1
        eco.birth_actions += 1
        eco.births += 1
        eco.max_generation = max(eco.max_generation, child.generation)
        eco.events.append({
            "tick": tick, "type": "birth", "parent": agent.id,
            "child": child_id, "founder": child.founder,
            "generation": child.generation, "location": child_pos,
            "genome": base.genome_digest(child.genome),
        })
    else:
        eco.move_actions += 1
        dx, dy = base.DIRECTIONS[action]
        dest = (agent.position[0] + dx, agent.position[1] + dy)
        if dest in eco.world.walls or dest in {
            a.position for a in eco.agents.values() if a.id != agent.id
        }:
            raise AssertionError("Scripted movement walked into wall/agent")
        agent.position = dest
        if dest in eco.food:
            eco.food.remove(dest)
            eco.food_eaten += 1
            agent.eaten += 1
            agent.last_food = 1
            added = min(eco.settings.food_energy,
                        eco.settings.max_energy - agent.energy)
            agent.energy += added
            eco.cumulative_food_energy += added
            eco.events.append({
                "tick": tick, "type": "food", "agent": agent.id,
                "founder": agent.founder, "location": dest,
                "energy_gain": added,
            })
    base._remove_dead(eco, agent)


def _diagnosed_action(eco: base.Ecology, agent_id: int, tick: int,
                      arm: str, diagnostic: Diagnostics) -> None:
    agent = eco.agents.get(agent_id)
    if agent is None:
        return
    _, _, sensors, legal, reproduction_ok = _candidate_actions(eco, agent)
    diagnostic.actions_observed += 1
    diagnostic.energy_before_total += agent.energy
    diagnostic.food_adjacent_opportunities += int(any(
        sensors[8 + a] > .5 for a in legal
    ))
    diagnostic.odor_present_actions += int(any(
        sensors[4 + a] > 0 for a in legal
    ))
    diagnostic.eligible_reproduction_opportunities += int(reproduction_ok)
    old_pos = agent.position
    old_moves = eco.move_actions
    old_births = eco.births
    old_food = eco.food_eaten
    if arm == "neural_heritable":
        preferred = _neural_preference(eco, agent, sensors)
        diagnostic.requested_reproduction += int(
            preferred == base.REPRODUCE
        )
        base.act(eco, agent_id, tick)
        new_agent = eco.agents.get(agent_id)
        moved_to = new_agent.position if new_agent is not None else None
        # Starved agents may have moved before dying; movement count inferred
        # from the actual motor action ledger below.
    else:
        choice = scripted_choice(
            eco, agent, arm, tick, diagnostic.agent_visits[agent.id]
        )
        diagnostic.requested_reproduction += int(choice == base.REPRODUCE)
        _act_scripted(eco, agent_id, tick, choice)
        moved_to = (eco.agents[agent_id].position
                    if agent_id in eco.agents else None)
    made_birth = eco.births - old_births
    made_food = eco.food_eaten - old_food
    diagnostic.action_births += made_birth
    diagnostic.action_food_consumed += made_food
    diagnostic.motor_actions += eco.move_actions - old_moves
    if moved_to is not None and moved_to != old_pos:
        visits = diagnostic.agent_visits[agent.id]
        previous = visits.get(moved_to, 0)
        diagnostic.traversed_previously_visited_cells += int(previous > 0)
        visits[moved_to] = previous + 1
        diagnostic.same_cell_revisits += int(previous > 0)
        if legal and any(sensors[4 + a] > 0 for a in legal):
            diagnostic.odor_alignable_moves += 1
            max_odor = max(sensors[4 + a] for a in legal)
            dx, dy = moved_to[0] - old_pos[0], moved_to[1] - old_pos[1]
            actual = base.DIRECTIONS.index((dx, dy))
            diagnostic.odor_aligned_moves += int(
                abs(sensors[4 + actual] - max_odor) < 1e-12
            )
    # Count movements by actual game ledger even for agents that die on move.
    # All diagnostic action counters are separate from game event accounting.
    diagnostic.record_newborns(eco)


def simulate(seed: int, settings: Settings, arm: str, *,
             shifted: bool = False, detailed: bool = False) -> tuple[dict, list[dict], dict | None]:
    settings.validate()
    if arm not in ALL_ARMS:
        raise ValueError(arm)
    # Scripted policies have inherited genomes only for genealogical bookkeeping:
    # genotype NEVER controls their movement or reproduction.
    eco = base.create_ecology(
        seed, settings, "heritable_mutation", shifted=shifted
    )
    diagnostic = Diagnostics()
    diagnostic.record_initial_visits(eco)
    no_regrowth = arm == "greedy_no_regrowth"
    schedule = base._food_schedule(seed, settings, eco, no_regrowth)
    checkpoints = []
    for tick in range(0, settings.ticks + 1):
        if tick:
            ids = list(eco.agents)
            base._local_rng(seed, tick, 0, 0).shuffle(ids)
            for agent_id in ids:
                _diagnosed_action(eco, agent_id, tick, arm, diagnostic)
            base._food_regeneration(eco, schedule[tick - 1])
            eco.ticks_elapsed = tick
            if not eco.agents and eco.extinct_at is None:
                eco.extinct_at = tick
        if tick == 0 or tick % settings.checkpoint_every == 0 or tick == settings.ticks:
            entry = base.snapshot(eco, tick)
            entry.update({
                "arm": arm,
                "seed": seed,
                "split": "shifted" if shifted else "main",
                "food_consumed_per_100_actions": diagnostic.as_metrics()[
                    "food_per_100_movement_actions"
                ],
                "odor_alignment_rate": diagnostic.as_metrics()[
                    "odor_alignment_rate"
                ],
                "observed_actions": diagnostic.actions_observed,
            })
            checkpoints.append(entry)
            if abs(base.energy_balance(eco)) > 1e-6:
                raise AssertionError("Energy ledger not balanced")
    agents = list(eco.agents.values())
    stats = diagnostic.as_metrics()
    row = {
        "seed": seed,
        "split": "shifted" if shifted else "main",
        "arm": arm,
        "alive_final": len(agents),
        "survived": int(bool(agents)),
        "extinct_at": eco.extinct_at,
        "births": eco.births,
        "deaths": eco.deaths,
        "max_generation": eco.max_generation,
        "has_generation3": int(eco.max_generation >= 3),
        "food_eaten": eco.food_eaten,
        "food_spawned": eco.food_spawned,
        "food_remaining": len(eco.food),
        "total_food_energy_gained": eco.cumulative_food_energy,
        "energy_basal_cost": eco.cumulative_cost_basal,
        "energy_birth_dissipation": eco.cumulative_birth_dissipation,
        "energy_ledger_residual": base.energy_balance(eco),
        "motor_mask_corrections": eco.corrected_actions,
        "ineligible_reproduction_preferences": eco.reproduction_ineligible_wishes,
        "policy_genome_used": int(arm == "neural_heritable"),
        "reproduction_policy": (
            "neural" if arm == "neural_heritable"
            else ("disabled" if arm == "greedy_no_reproduction"
                  else "scripted_energy_threshold")
        ),
        **stats,
    }
    if row["action_births"] != row["births"] or row["action_food_consumed"] != row["food_eaten"]:
        raise AssertionError("Diagnostics must reconcile with physical events")
    if arm == "greedy_no_reproduction" and (eco.births or (settings.ticks >= settings.max_age and agents)):
        raise AssertionError("No-birth control must lose all founders by age 100")
    if arm == "greedy_no_regrowth" and eco.food_spawned != settings.initial_food:
        raise AssertionError("No-regrowth arm unexpectedly spawned food")
    trace = None
    if detailed:
        trace = {
            "seed": seed, "arm": arm, "split": row["split"],
            "walls": [list(x) for x in sorted(eco.world.walls)],
            "initial_food": [list(x) for x in eco.world.initial_food],
            "events": eco.events,
            "final_agents": [{
                "id": a.id, "founder": a.founder, "parent": a.parent,
                "position": list(a.position), "energy": a.energy,
                "age": a.age, "generation": a.generation,
                "genome_digest": base.genome_digest(a.genome),
            } for a in sorted(agents, key=lambda a: a.id)],
            "initial_founder_genomes": list(eco.initial_genome_digests),
            "checkpoints": checkpoints,
            "diagnostics": stats,
        }
    return row, checkpoints, trace


def _compare(rows: list[dict], split: str, a: str, b: str, measure: str,
             bootstrap_seed: int) -> dict:
    aa = {x["seed"]:x for x in rows if x["split"] == split and x["arm"] == a}
    bb = {x["seed"]:x for x in rows if x["split"] == split and x["arm"] == b}
    if not aa or aa.keys() != bb.keys():
        raise AssertionError("Paired world seed mismatch")
    vals = [aa[seed][measure] - bb[seed][measure] for seed in sorted(aa)]
    return {
        "measure": measure, "mean_delta": statistics.mean(vals),
        "bootstrap_95": list(base.paired_bootstrap(vals, bootstrap_seed)),
        "positive_worlds": sum(x>0 for x in vals),
        "negative_worlds": sum(x<0 for x in vals),
        "tied_worlds": sum(x==0 for x in vals),
        "by_world": {str(s):v for s,v in zip(sorted(aa), vals)},
    }


def _means(rows: list[dict], split: str) -> dict:
    arms = ACTIVE_ARMS if split == "shifted" else ALL_ARMS
    metrics = (
        "alive_final", "survived", "births", "max_generation",
        "has_generation3", "food_eaten", "food_remaining",
        "food_per_100_movement_actions", "odor_alignment_rate",
        "odorless_action_fraction", "motor_mask_corrections",
        "ineligible_reproduction_preferences",
    )
    return {
        arm: {key: statistics.mean(
            r[key] for r in rows if r["split"] == split and r["arm"] == arm
        ) for key in metrics}
        for arm in arms
    }


def render_population_svg(checkpoints: list[dict], settings: Settings) -> str:
    """Static SVG comparison with explicit axes/tick legends, no external libs."""
    top, width, height = 40, 720, 400
    left, bottom = 70, 335
    series = {
        arm: [
            statistics.mean(
                row["population"] for row in checkpoints
                if row["split"] == "main" and row["arm"] == arm
                and row["tick"] == tick
            )
            for tick in range(0, settings.ticks + 1, settings.checkpoint_every)
        ] for arm in ACTIVE_ARMS
    }
    colors = {
        "neural_heritable": "#8b5cf6",
        "greedy_odor": "#0d9488",
        "greedy_novelty": "#ea580c",
        "random_legal": "#2563eb",
    }
    max_pop = max(settings.initial_population, max(max(s) for s in series.values()))
    xaxis = lambda x: left + (width-left-20)*x/settings.ticks
    yaxis = lambda y: bottom - (bottom-top)*y/max_pop
    paths=[]
    for arm,vals in series.items():
        points=" ".join(
            f"{xaxis(t):.1f},{yaxis(v):.1f}" for t,v in zip(
                range(0,settings.ticks+1,settings.checkpoint_every),vals
            )
        )
        paths.append(f'<polyline points="{points}" fill="none" '
                     f'stroke="{colors[arm]}" stroke-width="2.8"/>')
    keys=[]
    for i,arm in enumerate(ACTIVE_ARMS):
        x=left+i*159
        keys.append(f'<circle cx="{x}" cy="378" r="5" fill="{colors[arm]}"/>'
                    f'<text x="{x+9}" y="382" fill="#263238" '
                    'font-family="sans-serif" font-size="11">'
                    f'{arm}</text>')
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">'
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>'
        f'<text x="{left}" y="22" font-family="sans-serif" font-size="17">'
        'M5.1 — Population per world (mean, main 16 seeds)</text>'
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" '
        'stroke="#334155" stroke-width="1"/>'
        f'<line x1="{left}" y1="{bottom}" x2="{width-20}" y2="{bottom}" '
        'stroke="#334155" stroke-width="1"/>'
        f'<text x="5" y="{top+7}" font-family="sans-serif" font-size="12">'
        f'{max_pop:.0f}</text>'
        f'<text x="42" y="{bottom+4}" font-family="sans-serif" font-size="12">0</text>'
        f'<text x="{left}" y="{bottom+22}" font-family="sans-serif" '
        'font-size="12">0</text>'
        f'<text x="{width-65}" y="{bottom+22}" font-family="sans-serif" '
        f'font-size="12">{settings.ticks} ticks</text>'
        + "".join(paths+keys) + '</svg>\n'
    )


def run(settings: Settings, output: Path) -> dict:
    settings.validate()
    output.mkdir(parents=True, exist_ok=True)
    rows, checks, traces = [], [], []
    for split, start, count, shift, arms in (
        ("main",settings.seed_start,settings.seeds,False,ACTIVE_ARMS),
        ("shifted",settings.shifted_start,settings.shifted_seeds,True,ACTIVE_ARMS),
    ):
        for seed in range(start, start+count):
            founder_signature = None
            for arm in arms:
                r, ck, trace = simulate(seed, settings, arm, shifted=shift,
                                         detailed=(seed == start))
                rows.append(r)
                checks.extend(ck)
                if trace:
                    if founder_signature is None:
                        founder_signature=trace["initial_founder_genomes"]
                    elif founder_signature!=trace["initial_founder_genomes"]:
                        raise AssertionError("Arms do not start with same genotypes")
                    traces.append(trace)
            print(f"M5.1 {split} seed={seed} "
                  + " ".join(f"{x['arm']}={x['alive_final']}/"
                             f"{x['births']}births"
                             for x in rows[-len(arms):]), flush=True)
    for seed in range(settings.seed_start,
                      settings.seed_start+settings.negative_control_worlds):
        for arm in NEGATIVE_ARMS:
            r,ck,trace=simulate(seed,settings,arm,detailed=(seed==settings.seed_start))
            rows.append(r)
            checks.extend(ck)
            if trace:
                traces.append(trace)
    means={split:_means(rows,split) for split in ("main","shifted")}
    survive=means["main"]["greedy_novelty"]["survived"]
    deep=means["main"]["greedy_novelty"]["has_generation3"]
    forage=_compare(rows,"main","greedy_odor","random_legal","food_eaten",
                    settings.seed_start+317)
    summary={
        "protocol":"M5.1 preregistered ecological viability and programmed-control diagnosis",
        "settings":asdict(settings),
        "mean":means,
        "primary_viability": {
            "greedy_novelty_survived_worlds":int(round(survive*settings.seeds)),
            "greedy_novelty_generation3_worlds":int(round(deep*settings.seeds)),
            "survival_gate":survive>=.5,
            "generation_gate":deep>=.5,
            "both_pass":bool(survive>=.5 and deep>=.5),
        },
        "necessary_forage_greedy_minus_random": {
            **forage,
            "pass":bool(
                forage["mean_delta"]>0 and forage["bootstrap_95"][0]>0
                and forage["positive_worlds"]>=math.ceil(.75*settings.seeds)
            ),
        },
        "exploratory": {
            "novelty_minus_reactive_food":_compare(
                rows,"main","greedy_novelty","greedy_odor","food_eaten",
                settings.seed_start+401
            ),
            "reactive_minus_neural_food":_compare(
                rows,"main","greedy_odor","neural_heritable","food_eaten",
                settings.seed_start+421
            ),
            "novelty_minus_neural_end_population":_compare(
                rows,"main","greedy_novelty","neural_heritable",
                "alive_final",settings.seed_start+451
            ),
            "shifted_novelty_minus_neural_end_population":_compare(
                rows,"shifted","greedy_novelty","neural_heritable",
                "alive_final",settings.shifted_start+479
            ),
        },
        "negative_controls":{
            "no_reproduction_all_extinct":all(
                row["alive_final"]==0 and row["births"]==0
                for row in rows if row["arm"]=="greedy_no_reproduction"
            ),
            "no_regrowth_initial_only":all(
                row["food_spawned"]==settings.initial_food
                for row in rows if row["arm"]=="greedy_no_regrowth"
            ),
        },
        "interpretation_limits":[
            "Scripted controllers are deliberately more privileged than a neural controller, including access to age and personal visit memory in novelty policy",
            "Genotype bookkeeping in scripted arms has NO causal effect on behavior",
            "No neural weights trained or changed by the M5.1 controls",
            "Identical food-event candidate schedule does not imply identical realized food occupancy",
            "The independent experimental unit is the world seed, not births or actions",
            "Physical reproduction rules and food-odor sensor are researcher-specified",
            "No open-ended evolution, general cognition, or newly invented objectives inferred",
        ],
    }
    (output/"summary.json").write_text(
        json.dumps(summary,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    with (output/"world_metrics.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    with (output/"population_checkpoints.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(checks[0]))
        writer.writeheader();writer.writerows(checks)
    for trace in traces:
        name=f"{trace['split']}_{trace['arm']}"
        (output/f"events_{name}.json").write_text(
            json.dumps(trace,indent=2,sort_keys=True)+"\n",encoding="utf-8"
        )
        if trace["arm"] in ("greedy_novelty","neural_heritable"):
            # Same M5.0 visualization primitive for observed physical bodies.
            compatible={
                "walls":trace["walls"],
                "initial_food":trace["initial_food"],
                "final_agents":[{
                    "pos":a["position"] for a in trace["final_agents"]
                }],
            }
            (output/f"ecosystem_{name}.svg").write_text(
                base.render_ecosystem_svg(compatible,settings.size),
                encoding="utf-8"
            )
    (output/"population_curves.svg").write_text(
        render_population_svg(checks,settings),encoding="utf-8"
    )
    print("M5.1 PRIMARY",summary["primary_viability"],flush=True)
    print("M5.1 FORAGE",summary["necessary_forage_greedy_minus_random"],flush=True)
    print("M5.1 MEANS",summary["mean"],flush=True)
    return summary


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    defaults=Settings()
    for fld in fields(Settings):
        v=getattr(defaults,fld.name)
        parser.add_argument("--"+fld.name.replace("_","-"),
                            type=type(v),default=v)
    parser.add_argument("--out",type=Path,default=Path("results/m5_1"))
    args=parser.parse_args()
    s=Settings(**{f.name:getattr(args,f.name) for f in fields(Settings)})
    run(s,args.out)


if __name__=="__main__":
    main()
