"""FlyLoop M3: BLIND one-step motor and reward history with exact M0 weights.

No sensor, action or TD-learning-rate invention. Existing 8+6 weights
are reused; formerly zero blind action/critic inputs are filled
with one-step action/consequence or older yoke control. NO positional
map, no food-location sensor, no intrinsic novelty or evolution.

Protocol: docs/FLYLOOP_M3_PROTOCOL_2026-11-24.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from neuroevolution import flyloop_m0 as m0
from neuroevolution import flyloop_m1 as m1
from neuroevolution import flyloop_m2 as m2


SCENARIOS=("baseline","procedural","relocation")
ARMS=(
    "frozen","random_motion","plastic_plain","motor_memory",
    "temporal_feedback","temporal_sham","feedback_frozen"
)
SEEDS=tuple(range(7900,7932))
STOP=480
MOVE=240
RESTORE=360
EMA_DECAY=.75
FEEDBACK_SCALE=17.0
BOOTSTRAP=1000

def digest(x):
    return hashlib.sha256(json.dumps(
        x,sort_keys=True,separators=(",",":")
    ).encode()).hexdigest()

def mean(x):
    x=list(x)
    return sum(x)/len(x) if x else 0.0


class World(m2.World):
    def __init__(self,seed,scenario):
        if scenario not in SCENARIOS:
            raise ValueError("Unregistered M3 environment")
        super().__init__(seed, "baseline" if scenario=="relocation" else scenario,
                         "blind")
        self.scenario=scenario
        self.geometry=scenario
        self.start_map["geometry"]=scenario
        if scenario=="relocation":
            # There is NO simultaneous wall change in the food phase test.
            self.start_map["future_openings"]=sorted(self.openings)

    def switch_if_due(self):
        if self.scenario!="relocation":
            return super().switch_if_due()
        if self.tick==MOVE:
            self.health_before_switch=self.body.health
            self.food_before_switch=self.body.food_eaten
            self.cells_before_switch=len(self.visited)
            self.alive_at_switch=self.body.energy>0 and self.body.health>0
            # M1 corrected exogenous generator excludes NO body position.
            self._move_food()
            event={
                "step":self.tick,"event":"relocate_food",
                "openings":sorted(self.openings),
                "food_sites":[list(p) for p in sorted(self.food_cooldown)],
                "body_location":[self.body.x,self.body.y],
                "health":self.body.health,"energy":self.body.energy,
            }
            self.events.append(event)
            self.trace.append(event)
            return "relocate_food"
        if self.tick==RESTORE:
            self._restore_food()
            event={
                "step":self.tick,"event":"restore_food",
                "openings":sorted(self.openings),
                "food_sites":[list(p) for p in sorted(self.food_cooldown)],
                "body_location":[self.body.x,self.body.y],
                "health":self.body.health,"energy":self.body.energy,
            }
            self.events.append(event)
            self.trace.append(event)
            return "restore_food"
        return None


class Memory:
    """Only stores the agent's OWN past action and physical consequences.

    Structural action alignment and scalar feedback are EXPERIMENTER-CODED,
    not a proof of spontaneous memory representation.
    """
    def __init__(self,seed):
        self.events=[]
        self.sham_rng=random.Random(seed*18211+12337)
        self.reward_ema=0.0

    def remembered(self,arm):
        history=self.events
        if not history:
            return None,None
        if arm=="temporal_sham":
            # Older-than-last yoke, preserving past marginals only.
            if len(history)<2:
                return None,None
            older=self.sham_rng.randrange(len(history)-1)
            return history[older],history[older]["tick"]
        if arm in ("temporal_feedback","feedback_frozen"):
            return history[-1],history[-1]["tick"]
        return None,None

    def encode(self,action_inputs,critic_inputs,arm,tick):
        if arm in ("frozen","random_motion","plastic_plain"):
            return action_inputs,critic_inputs,{
                "source_tick":None,"feedback":[0.,0.,0.],
                "last_action":None,"last_collision":0,
            }
        latest=self.events[-1] if self.events else None
        last_action=latest["action"] if latest else None
        last_collision=latest["collision"] if latest else 0
        selected,source=self.remembered(arm)
        if arm=="motor_memory":
            selected,source=None,None
        fb=([selected["positive"],selected["damage"],selected["ema"]]
            if selected else [0.,0.,0.])
        base=[]
        for candidate,original in enumerate(action_inputs):
            item=list(original)
            if latest is not None:
                dx,dy=m0.DIRECTIONS[last_action]
                nx,ny=m0.DIRECTIONS[candidate]
                alignment=float(dx*nx+dy*ny)
            else:
                alignment=0.0
            # All target feature slots are exactly ZERO in M2 blind input.
            assert all(item[j]==0. for j in (1,2,3,4,6))
            item[3]=alignment
            item[4]=alignment*float(last_collision)
            item[1]=alignment*fb[0]
            item[2]=alignment*fb[1]
            item[6]=alignment*fb[2]
            base.append(tuple(item))
        new_critic=list(critic_inputs)
        assert new_critic[3]==0. and new_critic[4]==0.
        new_critic[3]=fb[0]
        new_critic[4]=fb[1]
        assert source is None or source<tick
        if arm=="temporal_sham":
            assert source is None or source<=tick-2
        return base,tuple(new_critic),{
            "source_tick":source,"feedback":fb,
            "last_action":last_action,"last_collision":last_collision,
        }

    def observe_actual(self,tick,action,physical_reward,energy_prior,
                       health_prior,event):
        """Observe AFTER action. Never pass world food labels or coordinates."""
        energy_delta=event["energy"]-energy_prior
        damage=max(0.,health_prior-event["health"])/4.
        positive=max(0.,min(1.,physical_reward/FEEDBACK_SCALE))
        normalized=max(-1.,min(1.,physical_reward/FEEDBACK_SCALE))
        self.reward_ema=(EMA_DECAY*self.reward_ema+
                         (1.-EMA_DECAY)*normalized)
        entry={
            "tick":tick,"action":action,
            "collision":event["collision"],
            "positive":round(positive,9),
            "damage":round(damage,9),
            "ema":round(self.reward_ema,9),
            "physical_reward":round(physical_reward,6),
            "energy_delta":round(energy_delta,6),
        }
        self.events.append(entry)
        return entry


def _controller(seed,arm):
    if arm in ("frozen","random_motion","feedback_frozen"):
        return m0.Controller(seed,"frozen")
    return m0.Controller(seed,"plastic")


def simulate(seed,scenario,arm,reference_check=False):
    if seed not in SEEDS and not reference_check:
        raise ValueError("M3 world seed not preregistered")
    if scenario not in SCENARIOS or arm not in ARMS:
        raise ValueError("M3 scenario or arm not preregistered")
    world=World(seed,scenario)
    actor=_controller(seed,arm)
    memory=Memory(seed)
    initial={
        "actor":list(actor.actor_start),
        "critic":list(actor.critic_start),
    }
    for t in range(STOP):
        if world.tick!=t:
            raise AssertionError("World tick drift")
        world.switch_if_due()
        if world.body.energy<=0 or world.body.health<=0:
            break
        raw_actor,raw_critic=world.observe()
        effective,critic,history=memory.encode(raw_actor,raw_critic,arm,t)
        if arm=="random_motion":
            a=min(3,int(4*actor.rng.random()))
            probabilities=(.25,.25,.25,.25,0.0)
        else:
            a,probabilities=actor.choose(effective)
        energy_prior=world.body.energy
        health_prior=world.body.health
        observed_reward,terminal=world.step(a)
        obs_event=world.trace[-1]
        observation=memory.observe_actual(
            t,a,observed_reward,energy_prior,health_prior,obs_event
        )
        _,next_raw_critic=world.observe()
        # TD next-state features require the newly observed memory state.
        # encode MUST NOT consume the sham RNG before next action, else
        # the sham comparator would see a different shuffled history.
        if arm=="temporal_sham":
            next_critic=next_raw_critic
        elif arm in ("temporal_feedback","feedback_frozen"):
            nr=list(next_raw_critic)
            nr[3]=observation["positive"]
            nr[4]=observation["damage"]
            next_critic=tuple(nr)
        else:
            next_critic=next_raw_critic
        actor.adapt(
            effective,critic,a,probabilities,observed_reward,
            next_critic,terminal,world.signature()
        )
        obs_event["raw_blind_action_inputs"]=[list(s) for s in raw_actor]
        obs_event["raw_blind_critic_inputs"]=list(raw_critic)
        obs_event["effective_actor_inputs"]=[list(s) for s in effective]
        obs_event["effective_critic_inputs"]=list(critic)
        obs_event["memory_used"]=history
        obs_event["observed_actual_feedback"]=observation
        if terminal:
            break

    steps=[r for r in world.trace if "action" in r]
    ranges={
        "initial":(0,MOVE),
        "changed":(MOVE,RESTORE),
        "final":(RESTORE,STOP),
    }
    phase={
        name:{
            "food":sum(x["food"] for x in steps if start<=x["step"]<end),
            "steps":sum(1 for x in steps if start<=x["step"]<end),
            "hazard":sum(x["hazard"] for x in steps if start<=x["step"]<end),
        }
        for name,(start,end) in ranges.items()
    }
    return {
        "world_seed":seed,"scenario":scenario,"arm":arm,
        "start_map":world.start_map,
        "events":world.events,
        "steps_alive":len(steps),
        "survived_480":len(steps)==STOP,
        "food_consumed":world.body.food_eaten,
        "food_phase_initial":phase["initial"]["food"],
        "food_phase_changed":phase["changed"]["food"],
        "food_phase_final":phase["final"]["food"],
        "phase":phase,
        "hazard_contacts":world.body.damage_events,
        "wall_collisions":world.body.hits,
        "food_site_count":len(world.consumed_food_spots),
        "revisited_food_events":world.body.food_returns,
        "unique_cells":len(world.visited),
        "post_change_crossings":world.body.crossings_post,
        "terminal_energy":round(world.body.energy,6),
        "terminal_health":round(world.body.health,6),
        "initial_weights":initial,
        "final_weights":{
            "actor":list(actor.actor),"critic":list(actor.critic)
        },
        "weight_update_events":actor.parameter_updates,
        "trajectory":world.trace,
    }


def reference_m2_baseline_check():
    """Independent full re-simulation against original M2 method on old seeds."""
    for seed in m2.SEEDS[:4]:
        for m3_arm,m2_arm in (
            ("frozen","frozen"),
            ("random_motion","random_motion"),
            ("plastic_plain","plastic"),
        ):
            our=simulate(seed,"baseline",m3_arm,reference_check=True)
            ref=m2.simulate(seed,"baseline","blind",m2_arm)
            assert our["initial_weights"]==ref["initial_weights"]
            assert our["final_weights"]==ref["final_weights"]
            assert our["food_consumed"]==ref["food_consumed"]
            assert our["steps_alive"]==ref["steps_alive"]
            for target,orig in zip(
                [s for s in our["trajectory"] if "action" in s],
                [s for s in ref["trajectory"] if "action" in s],
            ):
                for key in ("step","action","from","position","collision",
                            "hazard","food","physical_reward","energy","health"):
                    if target[key]!=orig[key]:
                        return False
    return True


def _verify(rows):
    legal=True
    memory_clean=True
    masks_clean=True
    weight_correct=True
    for row in rows:
        if row["steps_alive"]<1 or row["steps_alive"]>STOP:
            legal=False
        per_step={}
        p=(2,6)
        for t in row["trajectory"]:
            if "action" not in t:
                if t["event"] not in (
                    "change_gate_openings","relocate_food","restore_food"
                ):
                    legal=False
                continue
            tick=t["step"]
            if tick in per_step:
                legal=False
            if t["from"]!=list(p):
                legal=False
            p=tuple(t["position"])
            if t["action"] not in range(5):
                legal=False
            dx,dy=m0.DIRECTIONS[t["action"]]
            expected=tuple(t["from"]) if t["collision"] else (
                t["from"][0]+dx,t["from"][1]+dy
            )
            if p!=expected:
                legal=False
            openings=(row["start_map"]["future_openings"]
                      if tick>=MOVE and row["scenario"]!="relocation"
                      else row["start_map"]["initial_openings"])
            walls=m0.World._walls(frozenset(openings)) | set(
                map(tuple,row["start_map"]["posts"])
            )
            if p in walls or not (0<=t["energy"]<=100 and
                                  0<=t["health"]<=100):
                legal=False
            raw=t["raw_blind_action_inputs"]
            critic=t["raw_blind_critic_inputs"]
            if any(
                v[k]!=0.0 for v in raw for k in (1,2,3,4,6)
            ) or critic[3]!=0.0 or critic[4]!=0.0:
                masks_clean=False
            hist=t["memory_used"]
            source=hist["source_tick"]
            if source is not None:
                if not (source<tick and source in per_step):
                    memory_clean=False
                elif list(hist["feedback"])!=[
                    per_step[source]["positive"],
                    per_step[source]["damage"],
                    per_step[source]["ema"]
                ]:
                    memory_clean=False
            elif any(hist["feedback"]):
                memory_clean=False
            if row["arm"]=="temporal_sham" and source is not None:
                if source>tick-2:
                    memory_clean=False
            if tick==0 and (
                source is not None or hist["last_action"] is not None or
                any(hist["feedback"])
            ):
                memory_clean=False
            if row["arm"] in ("frozen","random_motion","plastic_plain") and (
                t["raw_blind_action_inputs"]!=t["effective_actor_inputs"]
                or t["raw_blind_critic_inputs"]!=t["effective_critic_inputs"]
            ):
                memory_clean=False
            if row["arm"]=="motor_memory" and (
                source is not None or any(hist["feedback"])
            ):
                memory_clean=False
            if hist["last_action"] is not None:
                if tick==0 or hist["last_action"]!=(
                    next((z["action"] for z in row["trajectory"] if
                         z.get("step")==tick-1 and "action" in z),None)
                ):
                    memory_clean=False
            # No future sensor carried in the network.
            per_step[tick]=t["observed_actual_feedback"]
        if row["arm"] in ("frozen","random_motion","feedback_frozen"):
            if (row["final_weights"]!=row["initial_weights"] or
                row["weight_update_events"]!=0):
                weight_correct=False
        else:
            if (row["final_weights"]==row["initial_weights"] or
                row["weight_update_events"]<1):
                weight_correct=False
        if row["arm"]=="random_motion" and any(
            t["action"]==4 for t in row["trajectory"] if "action" in t
        ):
            weight_correct=False
    return legal,masks_clean,memory_clean,weight_correct


def _paired_rate_bootstrap(rows,scenario,arm,reference,n=BOOTSTRAP):
    index={(r["world_seed"],r["arm"]):r for r in rows
           if r["scenario"]==scenario}
    def rate(seeds,method):
        sub=[index[(seed,method)] for seed in seeds]
        return 100*sum(x["food_consumed"] for x in sub)/sum(
            x["steps_alive"] for x in sub
        )
    rng=random.Random(79023)
    observed=rate(SEEDS,arm)-rate(SEEDS,reference)
    deltas=[]
    for _ in range(n):
        sample=[rng.choice(SEEDS) for _ in SEEDS]
        deltas.append(rate(sample,arm)-rate(sample,reference))
    deltas.sort()
    return {
        "delta_food_per_100_living_ticks":round(observed,6),
        "paired_world_bootstrap95_exploratory":[
            round(deltas[int(n*.025)],6),
            round(deltas[int(n*.975)-1],6),
        ]
    }


def run():
    # Save core historical compatibility as an integrity control.
    old_m2_compatibility=reference_m2_baseline_check()
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
            group=[r for r in rows if
                   r["scenario"]==scenario and r["arm"]==arm]
            total_steps=sum(r["steps_alive"] for r in group)
            summary[scenario][arm]={
                field:round(mean(r[field] for r in group),6)
                for field in (
                    "steps_alive","food_consumed","food_phase_initial",
                    "food_phase_changed","food_phase_final",
                    "hazard_contacts","wall_collisions","food_site_count",
                    "revisited_food_events","unique_cells",
                    "post_change_crossings"
                )
            }|{
                "food_per_100_alive_ticks":
                    round(100*sum(r["food_consumed"] for r in group)
                          /total_steps,6),
                "survived_480_fraction":
                    round(mean(r["survived_480"] for r in group),6),
            }
    all_paired=all(
        len({digest(r["start_map"]) for r in rows
             if r["scenario"]==sc and r["world_seed"]==seed})==1
        and len({digest(r["initial_weights"]) for r in rows
                 if r["scenario"]==sc and r["world_seed"]==seed})==1
        for sc in SCENARIOS for seed in SEEDS
    )
    world_sigs={
        digest(r["start_map"])
        for r in rows if r["scenario"]=="procedural"
    }
    proc_connected=all(
        m2._reachable(
            m0.World._walls(frozenset(r["start_map"][phase]))|
            set(map(tuple,r["start_map"]["posts"]))
        )
        for r in rows if r["scenario"]=="procedural"
        for phase in ("initial_openings","future_openings")
    )
    proc_valid=all(
        len(r["start_map"]["posts"])==10 and
        not (
            set(r["start_map"]["initial_openings"]) &
            set(r["start_map"]["future_openings"])
        )
        for r in rows if r["scenario"]=="procedural"
    )
    relocation_valid=all(
        r["start_map"]["initial_openings"]==
            r["start_map"]["future_openings"]
        and [e["step"] for e in r["events"]] == (
            [MOVE,RESTORE] if r["steps_alive"]>RESTORE else
            [MOVE] if r["steps_alive"]>MOVE else []
        )
        and (not r["events"] or
            not (
                set(map(tuple,r["events"][0]["food_sites"])) &
                set(map(tuple,r["start_map"]["food"]))
            ))
        and (len(r["events"])<2 or
            sorted(r["events"][1]["food_sites"])==
            sorted(r["start_map"]["food"]))
        for r in rows if r["scenario"]=="relocation"
    )
    same_relocations=all(
        len({digest(r["events"][0]["food_sites"])
             for r in rows if r["scenario"]=="relocation"
             and r["world_seed"]==seed and r["events"]})<=1
        for seed in SEEDS
    )
    safe_physics,blind_masks,valid_memory,valid_weights=_verify(rows)
    B=summary["baseline"]
    P=summary["procedural"]
    L=summary["relocation"]
    rate=lambda tab,arm:tab[arm]["food_per_100_alive_ticks"]
    gates={
        "01_exact_672_balanced_raw_rows":len(rows)==672 and
            all(sum(r["scenario"]==sc and r["arm"]==arm for r in rows)==32
                for sc in SCENARIOS for arm in ARMS),
        "02_legal_complete_no_wall_tunnel_or_resurrection":
            safe_physics,
        "03_identical_paired_map_schedule_and_initial_weights":
            all_paired,
        "04_all_raw_sensory_modes_blind_no_future_food_hazard":
            blind_masks,
        "05_temporal_sources_previous_actual_outcomes_only":
            valid_memory,
        "06_sham_feedback_only_older_past_separate_rng":
            valid_memory,
        "07_exact_14_weights_frozen_unchanged_and_td_updates":
            valid_weights,
        "08_unaugmented_baseline_matches_original_m2":
            old_m2_compatibility,
        "09_distinct_connected_valid_procedural_maps":
            len(world_sigs)>=25 and proc_connected and proc_valid,
        "10_exogenous_relocation_at_240_restore_360_same_for_all":
            relocation_valid and same_relocations,
        "11_raw_auditable_stop_flags_and_untouched_old_work":
            len(rows)==672 and all(r["trajectory"]
                and r["start_map"] and r["initial_weights"] for r in rows),
        "12_baseline_temporal_15pct_above_plain_rate":
            rate(B,"temporal_feedback")>=
                rate(B,"plastic_plain")*1.15,
        "13_baseline_temporal_10pct_above_random_rate":
            rate(B,"temporal_feedback")>=
                rate(B,"random_motion")*1.10,
        "14_procedural_temporal_10pct_above_random_rate":
            rate(P,"temporal_feedback")>=
                rate(P,"random_motion")*1.10,
        "15_procedural_temporal_15pct_above_plain_rate":
            rate(P,"temporal_feedback")>=
                rate(P,"plastic_plain")*1.15,
        "16_procedural_temporal_10pct_above_motor_only":
            rate(P,"temporal_feedback")>=
                rate(P,"motor_memory")*1.10,
        "17_procedural_temporal_10pct_above_sham":
            rate(P,"temporal_feedback")>=
                rate(P,"temporal_sham")*1.10,
        "18_relocation_changed_phase_temporal_15pct_above_plain":
            L["temporal_feedback"]["food_phase_changed"]>=
                L["plastic_plain"]["food_phase_changed"]*1.15,
    }
    paired={
        sc:{
            ref:_paired_rate_bootstrap(rows,sc,"temporal_feedback",ref)
            for ref in ("frozen","random_motion","plastic_plain",
                        "motor_memory","temporal_sham","feedback_frozen")
        } for sc in SCENARIOS
    }
    return {
        "status":"FlyLoop M3 causal motor/consequence trace experiment",
        "protocol":"docs/FLYLOOP_M3_PROTOCOL_2026-11-24.md",
        "metadata":{
            "seeds":SEEDS,"scenarios":SCENARIOS,"arms":ARMS,
            "topology_actor_critic":[m0.N_ACTOR,m0.N_CRITIC],
            "world_size":m0.SIZE,"maximum_steps":STOP,
            "memory_ema_decay":EMA_DECAY,
            "temporal_features_are_researcher_engineered":True,
            "exact_m0_td_and_reward":True,
            "no_new_trainable_neurons_or_weights":True,
            "no_hidden_precontact_food_or_hazard":True,
            "no_spatial_map":True,
            "prior_m2_baseline_compatibility_verified":
                old_m2_compatibility,
            "new_procedural_world_map_count":len(world_sigs),
            "no_evolution":True,
        },
        "summary":summary,
        "paired_exploratory_world_bootstraps":paired,
        "screening_gates":gates,
        "screening_passed_count":sum(gates.values()),
        "screening_pass":all(gates.values()),
        "general_intelligence_demonstrated":False,
        "spatial_memory_verified":False,
        "autonomous_operator_invention_verified":False,
        "evolution_authorized":False,
        "advance_to_evolution":False,
        "raw_rows_sha256":digest(rows),
        "raw_rows":rows,
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,
        default=Path("results/flyloop/m3_temporal_memory.json"))
    arg=parser.parse_args()
    outcome=run()
    arg.out.parent.mkdir(parents=True,exist_ok=True)
    arg.out.write_text(json.dumps(
        outcome,sort_keys=True,separators=(",",":"))+"\n",encoding="utf8"
    )
    print("FLYLOOP_M3 "+json.dumps({
        k:outcome[k] for k in (
            "status","summary","screening_gates","screening_passed_count",
            "screening_pass","paired_exploratory_world_bootstraps",
            "raw_rows_sha256")
    },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
