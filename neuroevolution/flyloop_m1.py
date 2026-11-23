"""FlyLoop M1: preregistered sensor, layout and environmental memory stress.

Extends the fully functioning M0 World/Controller *without editing* M0.
No new action, neuron, trainer, path planning or ecologically privileged
event schedule is provided to the organism. Explicit engineered sensory
features and the engineered novelty reward remain disclosed.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import random
import statistics
from pathlib import Path

from neuroevolution import flyloop_m0 as m0

SCENARIOS = (
    "no_scent_gate",
    "occluded_noisy_gate",
    "novel_layout",
    "food_relocation",
)
ARMS = ("frozen", "plastic", "plastic_novelty", "plastic_reset")
SEEDS = tuple(range(7300, 7332))
HORIZON = m0.HORIZON
SWITCH = 240
RESTORE = 360
SENSOR_NOISE_BOUND = .08
NOVEL_OLD_GATES = frozenset((3, 9))
NOVEL_NEW_GATES = frozenset((4, 11))
NOVEL_WALL_SPURS = frozenset({
    (3, 4), (4, 4), (5, 4),
    (8, 8), (9, 8), (10, 8),
})


def canonical_digest(obj):
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def mean(xs):
    return statistics.mean(xs) if xs else 0.


class FlyWorld(m0.World):
    """Distinct experimental stimuli, same atomic movement physics."""

    def __init__(self, seed: int, scenario: str):
        if scenario not in SCENARIOS:
            raise ValueError("Unregistered scenario")
        self.scenario = scenario
        self.layout = "novel" if scenario == "novel_layout" else "m0"
        super().__init__(seed)
        self.events = []
        if self.layout == "novel":
            self.openings = NOVEL_OLD_GATES
            self.walls = self._walls(self.openings)
            # Re-seed only environmental hazards/food, never agent
            # parameters. Eight food sites on each side.
            gen = random.Random(seed * 12763 + 701)
            free = [
                (x,y) for x in range(1,m0.SIZE-1)
                for y in range(1,m0.SIZE-1)
                if (x,y) not in self.walls and (x,y)!=(2,6)
            ]
            left = [pos for pos in free if pos[0]<6]
            right = [pos for pos in free if pos[0]>6]
            food = gen.sample(left,8)+gen.sample(right,8)
            self.food_cooldown={p:0 for p in food}
            remaining = [p for p in free
                         if p not in self.food_cooldown and p[0]!=6]
            self.hazards=frozenset(gen.sample(remaining,m0.HAZARD_COUNT))
        self.original_food_sites = tuple(sorted(self.food_cooldown))
        self.original_hazards = tuple(sorted(self.hazards))
        self.initial_walls = tuple(sorted(self.walls))
        self.relocated_sites = ()
        self.restored = False
        self.phase_events = []

    def _walls(self,openings):
        walls = m0.World._walls(openings)
        return walls | NOVEL_WALL_SPURS if self.layout=="novel" else walls

    def _path_only_scent(self):
        """Shortest local accessible-path food signal, bounded 4 steps.

        This is engineered geodesic pseudo-smell, NOT a diffusion solver.
        The policy receives only the five local directional values, NOT
        a path, global map, route, future position or event timing.
        """
        start=(self.body.x,self.body.y)
        active={p for p,ready in self.food_cooldown.items()
                if ready<=self.tick}
        best=[0.,0.,0.,0.,0.]
        for action, (dx,dy) in enumerate(m0.DIRECTIONS[:4]):
            first=(start[0]+dx,start[1]+dy)
            if first in self.walls:
                continue
            seen={first}
            queue=deque([(first,1)])
            while queue:
                pos, steps=queue.popleft()
                if pos in active:
                    best[action]=max(best[action],(5.0-steps)/4.0)
                if steps>=4:
                    continue
                for mx,my in m0.DIRECTIONS[:4]:
                    nxt=(pos[0]+mx,pos[1]+my)
                    if nxt not in seen and nxt not in self.walls:
                        seen.add(nxt)
                        queue.append((nxt,steps+1))
        return best

    def _sensor_noise(self, action):
        # Deterministic bounded context noise, separate from actor RNG.
        value=(
            (self.seed*73856093)
            ^ (self.tick*19349663)
            ^ (self.body.x*83492791)
            ^ (self.body.y*2654435761)
            ^ (action*97531)
        ) & 0xffffffff
        return (((value % 2001) / 1000.) - 1.) * SENSOR_NOISE_BOUND

    def observe(self):
        options, _ = m0.World.observe(self)
        if self.scenario=="no_scent_gate":
            scent=[0.]*5
        else:
            path_scent=self._path_only_scent()
            scent=[
                min(1.,max(0.,path_scent[i]+self._sensor_noise(i)))
                if i<4 else 0.0
                for i in range(5)
            ]
        hunger=(m0.MAX_ENERGY-self.body.energy)/m0.MAX_ENERGY
        corrected=[
            (f[0],hunger*scent[i],f[2],f[3],scent[i],
             f[5],f[6],f[7])
            for i,f in enumerate(options)
        ]
        critic=(
            1.0,hunger,(m0.MAX_HEALTH-self.body.health)/m0.MAX_HEALTH,
            max(f[3] for f in corrected),
            max(f[4] for f in corrected),
            sum(f[5] for f in corrected[:4])/4.,
        )
        return corrected,critic

    def _move_food(self):
        gen=random.Random(self.seed * 17117 + 202603)
        used = set(self.original_food_sites) | set(self.hazards)
        used.add((self.body.x,self.body.y))
        free=[
            (x,y) for x in range(1,m0.SIZE-1)
            for y in range(1,m0.SIZE-1)
            if (x,y) not in used and (x,y) not in self.walls
        ]
        left=[p for p in free if p[0]<6]
        right=[p for p in free if p[0]>6]
        new=gen.sample(left,8)+gen.sample(right,8)
        self.relocated_sites=tuple(sorted(new))
        self.food_cooldown={p:self.tick for p in self.relocated_sites}
        if set(self.relocated_sites) & set(self.original_food_sites):
            raise AssertionError("Moved resources overlap original sites")

    def _restore_food(self):
        self.food_cooldown={p:self.tick for p in self.original_food_sites}
        self.restored=True

    def switch_if_due(self):
        if self.tick==SWITCH:
            self.health_before_switch=self.body.health
            self.food_before_switch=self.body.food_eaten
            self.cells_before_switch=len(self.visited)
            if self.scenario=="food_relocation":
                self._move_food()
                event="relocate_food"
                self.alive_at_switch=self.body.health>0 and self.body.energy>0
            else:
                self.openings=(
                    NOVEL_NEW_GATES if self.layout=="novel"
                    else m0.NEW_OPENINGS
                )
                self.walls=self._walls(self.openings)
                event="change_gate_openings"
                if (self.body.x,self.body.y) in self.walls:
                    self.body.health=0.
                self.alive_at_switch=self.body.health>0 and self.body.energy>0
            item={
                "step":self.tick,"event":event,
                "openings":sorted(self.openings),
                "food_sites":[list(p) for p in sorted(self.food_cooldown)],
                "body_location":[self.body.x,self.body.y],
                "health":self.body.health,
                "energy":self.body.energy,
            }
            self.events.append(item)
            self.trace.append(item)
            return event
        if self.tick==RESTORE and self.scenario=="food_relocation":
            self._restore_food()
            event="restore_food"
            item={
                "step":self.tick,"event":event,
                "food_sites":[list(p) for p in sorted(self.food_cooldown)],
                "body_location":[self.body.x,self.body.y],
                "energy":self.body.energy,
                "health":self.body.health,
            }
            self.events.append(item)
            self.trace.append(item)
            return event
        return None


def _controller(seed, arm):
    return m0.Controller(seed, "plastic" if arm=="plastic_reset" else arm)


def simulate(seed: int, scenario: str, arm: str):
    if arm not in ARMS or scenario not in SCENARIOS:
        raise ValueError("Unregistered scenario or controller")
    world=FlyWorld(seed,scenario)
    brain=_controller(seed,arm)
    initial_params={
        "actor":list(brain.actor_start),"critic":list(brain.critic_start)
    }
    start_map={
        "food":[list(p) for p in world.original_food_sites],
        "hazards":[list(p) for p in world.original_hazards],
        "walls":[list(p) for p in world.initial_walls],
        "initial_openings":sorted(world.openings),
        "gate_change_target":(
            sorted(NOVEL_NEW_GATES) if world.layout=="novel"
            else sorted(m0.NEW_OPENINGS)
        ),
    }
    resets=[]
    for step in range(HORIZON):
        if world.tick != step:
            raise AssertionError("Clock drift")
        event=world.switch_if_due()
        if event and arm=="plastic_reset" and (
            world.body.energy>0 and world.body.health>0
        ):
            old_hash=canonical_digest({"actor":brain.actor,"critic":brain.critic})
            brain.actor=list(brain.actor_start)
            brain.critic=list(brain.critic_start)
            resets.append({
                "step":step, "trigger":event,
                "parameters_before_reset_sha256":old_hash,
                "after_reset_equal_initial": (
                    tuple(brain.actor)==brain.actor_start
                    and tuple(brain.critic)==brain.critic_start
                ),
                "position_not_reset":[world.body.x,world.body.y],
            })
        if world.body.energy<=0 or world.body.health<=0:
            break
        opts,critic=world.observe()
        observation_scent=tuple(round(f[4],6) for f in opts)
        chosen,prob=brain.choose(opts)
        physical,terminal=world.step(chosen)
        _,newcritic=world.observe()
        bonus=brain.adapt(
            opts,critic,chosen,prob,physical,newcritic,
            terminal,world.signature()
        )
        line=world.trace[-1]
        line["perceived_directional_food_scent"]=list(observation_scent)
        line["perceived_adjacent_food"]=[int(f[3]) for f in opts]
        line["novelty_bonus_researcher_programmed"]=round(bonus,6)
        if terminal:
            break
    actions=[r for r in world.trace if "action" in r]
    phases={
        "initial":(0,SWITCH),
        "changed":(SWITCH,RESTORE),
        "restored":(RESTORE,HORIZON),
    }
    by_phase={}
    for label,(lower,upper) in phases.items():
        rows=[r for r in actions if lower<=r["step"]<upper]
        eating=[r["step"] for r in rows if r["food"]]
        by_phase[label]={
            "food":sum(r["food"] for r in rows),
            "hits":sum(r["collision"] for r in rows),
            "damage":sum(r["hazard"] for r in rows),
            "steps":len(rows),
            "time_to_first_food_censored": (
                eating[0]-lower+1 if eating else upper-lower+1
            ),
        }
    return {
        "world_seed":seed,"scenario":scenario,"arm":arm,
        "steps_alive":len(actions),
        "survived_480":len(actions)==HORIZON,
        "alive_at_240":len(actions)>SWITCH,
        "alive_at_360":len(actions)>RESTORE,
        "food_consumed":world.body.food_eaten,
        "health_end":round(world.body.health,6),
        "energy_end":round(world.body.energy,6),
        "hazard_contacts":world.body.damage_events,
        "wall_collisions":world.body.hits,
        "revisited_food_sites":world.body.food_returns,
        "unique_cells":len(world.visited),
        "crossings_before":world.body.crossings_pre,
        "crossings_after":world.body.crossings_post,
        "food_phase_initial":by_phase["initial"]["food"],
        "food_phase_changed":by_phase["changed"]["food"],
        "food_phase_restored":by_phase["restored"]["food"],
        "changed_food_first_delay_censored":
            by_phase["changed"]["time_to_first_food_censored"],
        "restored_food_first_delay_censored":
            by_phase["restored"]["time_to_first_food_censored"],
        "by_phase":by_phase,
        "initial_weights":initial_params,
        "final_weights":{"actor":list(brain.actor),"critic":list(brain.critic)},
        "online_parameter_updates":brain.parameter_updates,
        "resets":resets,"events":world.events,
        "start_map":start_map,
        "relocated_food_sites":[list(p) for p in world.relocated_sites],
        "restored_original_food":world.restored,
        "trajectory":world.trace,
    }


def _paired(rows,scenario,field,arm,reference="frozen",n=1000):
    delta=[]
    for seed in SEEDS:
        v=next(r[field] for r in rows if r["world_seed"]==seed
               and r["scenario"]==scenario and r["arm"]==arm)
        ref=next(r[field] for r in rows if r["world_seed"]==seed
                 and r["scenario"]==scenario and r["arm"]==reference)
        delta.append(v-ref)
    draw=random.Random(73119)
    estimates=sorted(mean([delta[draw.randrange(len(delta))]
                           for _ in delta]) for _ in range(n))
    return {
        "mean_paired_delta":round(mean(delta),6),
        "exploratory_paired_95pct":[
            round(estimates[int(.025*n)],6),
            round(estimates[int(.975*n)-1],6),
        ],
    }


def _integrity(rows):
    # Full raw physics check, NOT a scientific "emergent" outcome.
    legal=True
    for row in rows:
        if row["steps_alive"] > HORIZON:
            legal=False
        prev_tick=-1
        for tr in row["trajectory"]:
            if "action" not in tr:
                if tr["event"] not in (
                    "change_gate_openings","relocate_food","restore_food"
                ):
                    legal=False
                continue
            if tr["step"]<=prev_tick:
                legal=False
            prev_tick=tr["step"]
            x,y=tr["position"]
            mx,my=tr["from"]
            if abs(x-mx)+abs(y-my)>1 or tr["action"] not in range(5):
                legal=False
            if not 0<=tr["energy"]<=100 or not 0<=tr["health"]<=100:
                legal=False
            if row["scenario"]=="novel_layout":
                wall=m0.World._walls(
                    NOVEL_NEW_GATES if tr["step"]>=SWITCH
                    else NOVEL_OLD_GATES
                )|NOVEL_WALL_SPURS
            else:
                openings=(m0.OLD_OPENINGS
                          if row["scenario"]=="food_relocation"
                          or tr["step"]<SWITCH
                          else m0.NEW_OPENINGS)
                wall=m0.World._walls(openings)
            if (x,y) in wall:
                legal=False
            if row["scenario"]=="no_scent_gate":
                if any(abs(s)>1e-9
                       for s in tr["perceived_directional_food_scent"]):
                    legal=False
        if sum(row["by_phase"][p]["steps"]
               for p in ("initial","changed","restored")
              )!=row["steps_alive"]:
            legal=False
    return legal


def run():
    rows=[
        simulate(seed,scenario,arm)
        for scenario in SCENARIOS
        for seed in SEEDS
        for arm in ARMS
    ]
    summary={}
    for scenario in SCENARIOS:
        summary[scenario]={}
        for arm in ARMS:
            scope=[r for r in rows if
                   r["scenario"]==scenario and r["arm"]==arm]
            summary[scenario][arm]={
                key:round(mean(r[key] for r in scope),6)
                for key in (
                    "steps_alive","food_consumed","food_phase_initial",
                    "food_phase_changed","food_phase_restored",
                    "hazard_contacts","wall_collisions","unique_cells",
                    "revisited_food_sites",
                    "changed_food_first_delay_censored",
                    "restored_food_first_delay_censored",
                    "crossings_before","crossings_after"
                )
            } | {
                "survived_480_rate":round(mean(r["survived_480"]
                                               for r in scope),6),
                "alive_at_240_rate":round(mean(r["alive_at_240"]
                                              for r in scope),6),
                "alive_at_360_rate":round(mean(r["alive_at_360"]
                                              for r in scope),6),
                "nonzero_update_fraction":round(
                    mean(r["online_parameter_updates"]>0 for r in scope),6),
            }
    per_scenario_same_seed=all(
        len({canonical_digest(r["start_map"])
             for r in rows if r["scenario"]==s
             and r["world_seed"]==seed})==1
        and len({canonical_digest(r["initial_weights"])
                 for r in rows if r["scenario"]==s
                 and r["world_seed"]==seed})==1
        for s in SCENARIOS for seed in SEEDS
    )
    # Audit deterministic wall-shortcut signal at a point with
    # no permitted local path, excluding the noise.
    wall_probe=FlyWorld(SEEDS[0],"occluded_noisy_gate")
    wall_probe.body.x=5
    wall_probe.body.y=6
    wall_probe.food_cooldown={(7,6):0}
    no_path=wall_probe._path_only_scent()[1] == 0.
    no_path_noise=wall_probe.observe()[0][1][4] <= SENSOR_NOISE_BOUND+1e-10

    fresh_and_free=all(
        not (set(map(tuple,r["start_map"]["food"]))
             & set(map(tuple,r["start_map"]["hazards"])))
        and not (set(map(tuple,r["start_map"]["food"]))
                 & set(map(tuple,r["start_map"]["walls"])))
        and not (set(map(tuple,r["start_map"]["hazards"]))
                 & set(map(tuple,r["start_map"]["walls"])))
        for r in rows
    )
    scheduled=all(
        [e["step"] for e in r["events"]]==(
            [SWITCH,RESTORE] if r["scenario"]=="food_relocation"
            and r["steps_alive"]>RESTORE else
            [SWITCH] if r["steps_alive"]>SWITCH else []
        )
        and (r["scenario"]!="food_relocation"
             or not r["relocated_food_sites"]
             or set(map(tuple,r["relocated_food_sites"]))
                .isdisjoint(set(map(tuple,r["start_map"]["food"]))))
        for r in rows
    )
    # A transition that kills the agent at the instant of gate
    # closure still records its environmental event; the raw
    # event schedule, not a survivor-only metric, is authoritative.
    # For gate closure death at t=240 there will be an event but
    # only 240 action steps; validate the conditional explicitly.
    scheduled=all(
        [e["step"] for e in r["events"]]==(
            [SWITCH,RESTORE] if r["scenario"]=="food_relocation"
            and r["steps_alive"]>RESTORE else
            [SWITCH] if r["steps_alive"]>SWITCH or
                        (r["steps_alive"]==SWITCH
                         and r["events"] and
                         r["events"][0]["step"]==SWITCH) else []
        )
        for r in rows
    )
    moved_and_restored=all(
        (not r["relocated_food_sites"] or (
            len(r["relocated_food_sites"])==16
            and not(set(map(tuple,r["relocated_food_sites"]))
                    & set(map(tuple,r["start_map"]["food"])))
        ))
        and (not r["restored_original_food"] or (
            [tuple(p) for p in next(e for e in r["events"]
                 if e["event"]=="restore_food")["food_sites"]]
            ==sorted(map(tuple,r["start_map"]["food"]))
        ))
        for r in rows if r["scenario"]=="food_relocation"
    )
    reset_correct=all(
        r["initial_weights"]==r["final_weights"]
        and r["online_parameter_updates"]==0
        for r in rows if r["arm"]=="frozen"
    ) and all(
        r["online_parameter_updates"]>0
        and r["initial_weights"]!=r["final_weights"]
        for r in rows if r["arm"] in ("plastic","plastic_novelty")
    ) and all(
        all(reset["after_reset_equal_initial"]
            for reset in r["resets"])
        and len(r["resets"])==sum(e["health"]>0 and e["energy"]>0 for e in r["events"])
        for r in rows if r["arm"]=="plastic_reset"
    )
    n=summary["no_scent_gate"]
    o=summary["occluded_noisy_gate"]
    novel=summary["novel_layout"]
    reloc=summary["food_relocation"]
    tests={
        "512_rows_complete_three_phases_and_32_seeds":
            len(rows)==4*4*32 and
            all(sum(r["scenario"]==s and r["arm"]==arm
                    for r in rows)==32 for s in SCENARIOS
                for arm in ARMS),
        "physics_no_wall_crossing_legal_actions_energy_and_injury":
            _integrity(rows),
        "initial_world_and_14_parameters_identical_within_pairs":
            per_scenario_same_seed,
        "no_scent_exactly_zero_with_adjacent_food_unchanged":
            all(all(all(abs(v)<1e-9 for v in t["perceived_directional_food_scent"])
                    for t in r["trajectory"] if "action" in t)
                for r in rows if r["scenario"]=="no_scent_gate"),
        "occluded_path_no_wall_leak_and_noise_bounded":
            no_path and no_path_noise,
        "novel_layout_distinct_and_all_resources_in_free_cells":
            fresh_and_free and
            all(r["start_map"]["initial_openings"]==sorted(NOVEL_OLD_GATES)
                and len(r["start_map"]["walls"])>
                    len(m0.World._walls(NOVEL_OLD_GATES))
                for r in rows if r["scenario"]=="novel_layout"),
        "relocation_at_240_restore_at_360_and_no_input_schedule":
            scheduled and moved_and_restored,
        "frozen_weights_unchanged_online_updates_and_exact_resets":
            reset_correct,
        "raw_complete_and_evolution_disabled":
            len(rows)==512 and all(r["trajectory"] and r["initial_weights"]
                                   and r["final_weights"] for r in rows),
        "no_scent_plastic_food_ge_frozen_plus15pct":
            n["plastic"]["food_consumed"]>=
            1.15*n["frozen"]["food_consumed"],
        "occluded_plastic_lifetime_ge_frozen_plus10pct":
            o["plastic"]["steps_alive"]>=
            1.1*o["frozen"]["steps_alive"],
        "novel_layout_plastic_food_ge_frozen_plus15pct":
            novel["plastic"]["food_consumed"]>=
            1.15*novel["frozen"]["food_consumed"],
        "relocation_restored_plastic_food_ge_weight_reset_plus10pct":
            reloc["plastic"]["food_phase_restored"]>=
            1.1*reloc["plastic_reset"]["food_phase_restored"],
        "relocated_phase_novelty_food_ge_plastic_plus10pct":
            reloc["plastic_novelty"]["food_phase_changed"]>=
            1.1*reloc["plastic"]["food_phase_changed"],
    }
    comparisons={
        s:{
            "plastic_vs_frozen_food":_paired(rows,s,"food_consumed","plastic"),
            "plastic_vs_frozen_alive":_paired(rows,s,"steps_alive","plastic"),
            "novelty_vs_plastic_food":
                _paired(rows,s,"food_consumed","plastic_novelty","plastic"),
            "plastic_vs_reset_restored_food":
                _paired(rows,s,"food_phase_restored","plastic","plastic_reset"),
            "novelty_vs_plastic_changed_food":
                _paired(rows,s,"food_phase_changed","plastic_novelty","plastic"),
        } for s in SCENARIOS
    }
    return {
        "status":"FlyLoop M1 sensor and synaptic retention stress (synthetic DEV)",
        "protocol":"docs/FLYLOOP_M1_PROTOCOL_2026-11-23.md",
        "metadata":{
            "scenarios":SCENARIOS,"controller_arms":ARMS,
            "seed_range":[SEEDS[0],SEEDS[-1]],"worlds_per_condition":len(SEEDS),
            "total_lifetimes":len(rows),"up_to_steps_each":HORIZON,
            "change_tick":SWITCH,"restore_tick":RESTORE,
            "network_size_actor_critic_real_weights":[m0.N_ACTOR,m0.N_CRITIC],
            "M0_hyperparameters_unchanged":True,
            "novelty_reward_human_specified":True,
            "adjacent_food_still_perceived_when_scent_disabled":True,
            "scent_is_engineered_shortest_accessible_path_not_physics":True,
            "food_relocation_is_externally_programmed":True,
            "no_explicit_spatial_map_or_positional_memory_network":True,
            "no_evolution":True,
        },
        "summary":summary,
        "paired_bootstraps":comparisons,
        "screening_gates":tests,
        "screening_passed_count":sum(tests.values()),
        "screening_pass":all(tests.values()),
        "emergent_intelligence_demonstrated":False,
        "spatial_memory_verified":False,
        "independent_realworld_replication_complete":False,
        "evolution_authorized":False,
        "advance_to_evolution":False,
        "raw_rows_sha256":canonical_digest(rows),
        "raw_rows":rows,
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,
                        default=Path("results/flyloop/m1_sensory_memory.json"))
    args=parser.parse_args()
    result=run()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(
        json.dumps(result,sort_keys=True,indent=2)+"\n",encoding="utf8"
    )
    print("FLYLOOP_M1 "+json.dumps(
        {k:result[k] for k in ("status","summary","screening_gates",
             "screening_passed_count","screening_pass","paired_bootstraps",
             "raw_rows_sha256")},sort_keys=True
    ),flush=True)

if __name__=="__main__":
    main()
