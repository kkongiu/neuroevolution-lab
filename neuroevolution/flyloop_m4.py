"""FlyLoop M4: single-cell odor samples, causal temporal differences, sham controls.

Only the *environment* computes odor from food and accessible paths.
An animal sees its own current scalar, not adjacent source locations,
map, route, or directional gradient. It can compare its own successive
samples; this alignment representation is researcher ENGINEERED.
Uses exactly M0's existing eight actor + six critic real weights and
original TD learning rate/physical reward; not a reconstructed fly.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import random
from pathlib import Path

from neuroevolution import flyloop_m0 as m0
from neuroevolution import flyloop_m2 as m2
from neuroevolution import flyloop_m3 as m3

SCENARIOS=("baseline","procedural","relocation")
ARMS=("frozen","random_motion","plastic_plain","motor_only",
      "chemo_true","chemo_sham","chemo_frozen")
SEEDS=tuple(range(8300,8332))
HORIZON=480
SWITCH=240
RESTORE=360
DISTANCE_CAP=7
DECAY=.65
NOISE_BOUND=.015
SCENT_SCALE=6.0
EMA_DECAY=.75

def digest(data):
    return hashlib.sha256(
        json.dumps(data,sort_keys=True,separators=(",",":")).encode()
    ).hexdigest()

def mean(xs):
    return sum(xs)/len(xs) if xs else 0.0

def clamp(value,lo=-1.,hi=1.):
    return max(lo,min(hi,value))


class World(m3.World):
    """M3 physics, externally produced scalar olfactory field."""
    def __init__(self,seed,scenario):
        super().__init__(seed,scenario)
        self._distance_cache={}

    def _distances_from(self,at):
        wall_id=tuple(sorted(self.walls))
        key=(wall_id,at)
        if key not in self._distance_cache:
            distances={at:0}
            queue=deque([at])
            while queue:
                x,y=queue.popleft()
                d=distances[(x,y)]
                if d>=DISTANCE_CAP:
                    continue
                for dx,dy in m0.DIRECTIONS[:4]:
                    nxt=(x+dx,y+dy)
                    if nxt not in self.walls and nxt not in distances:
                        distances[nxt]=d+1
                        queue.append(nxt)
            self._distance_cache[key]=distances
        return self._distance_cache[key]

    def odor_here(self):
        """Only current-position scalar sampled once before action.

        Shortest legal path in an accessible-cell graph is an ENGINEERED
        stationary diffusion proxy, *not* directional oracle or airflow.
        """
        at=(self.body.x,self.body.y)
        distances=self._distances_from(at)
        signal=sum(
            DECAY**distances[pos]
            for pos,ready in self.food_cooldown.items()
            if ready<=self.tick and pos in distances
        )/4.
        # Deterministic bounded measurement noise; no actor RNG usage.
        value=(
            (self.seed*73856093)
            ^ (self.tick*19349663)
            ^ (self.body.x*83492791)
            ^ (self.body.y*2654435761)
        ) & 0xffffffff
        noise=(((value%2001)/1000.)-1.)*NOISE_BOUND
        return round(clamp(signal+noise,0.,1.),9),round(noise,9)


class SensorMemory:
    """Causal self-observed scalar differences plus movement persistence."""

    def __init__(self,seed):
        self.last_odor=None
        self.last_action=None
        self.last_collision=0
        self.ema=0.
        self.samples=[]
        self.sham_rng=random.Random(seed*18737+40071)

    def _sham(self,action_tick):
        older=[v for v in self.samples
               if 1<=v["tick"]<=action_tick-2]
        if action_tick<3 or not older:
            return None
        return older[self.sham_rng.randrange(len(older))]

    def encode(self,raw_actions,raw_critic,arm,tick,odor):
        if self.last_odor is None:
            real_delta=0.
        else:
            real_delta=round(clamp(
                SCENT_SCALE*(odor-self.last_odor)),9)
        self.ema=round(
            EMA_DECAY*self.ema+(1.-EMA_DECAY)*real_delta,9)
        current={"tick":tick,"odor":odor,"delta":real_delta,
                 "ema":self.ema}
        self.samples.append(current)

        if arm=="chemo_sham":
            sampled=self._sham(tick)
            delta=sampled["delta"] if sampled else 0.
            ema=sampled["ema"] if sampled else 0.
            source=sampled["tick"] if sampled else None
        elif arm in ("chemo_true","chemo_frozen"):
            delta,ema=real_delta,self.ema
            source=tick if tick>=1 else None
        else:
            delta,ema,source=0.,0.,None

        if arm in ("frozen","random_motion","plastic_plain"):
            actual=[tuple(v) for v in raw_actions]
            critic=tuple(raw_critic)
        else:
            actual=[]
            for a,feat in enumerate(raw_actions):
                f=list(feat)
                assert all(f[k]==0. for k in (1,2,3,4,6))
                if self.last_action is None:
                    alignment=0.
                else:
                    prev=m0.DIRECTIONS[self.last_action]
                    cur=m0.DIRECTIONS[a]
                    alignment=float(prev[0]*cur[0]+prev[1]*cur[1])
                f[3]=alignment
                f[4]=alignment*self.last_collision
                f[1]=alignment*delta
                f[2]=alignment*ema
                actual.append(tuple(f))
            critic=list(raw_critic)
            assert critic[3]==critic[4]==0.
            if arm in ("chemo_true","chemo_sham","chemo_frozen"):
                critic[3]=odor  # scalar current smell, not direction
                critic[4]=delta
            critic=tuple(critic)
        log={
            "source_tick":source,
            "actual_last_action":self.last_action,
            "actual_last_collision":self.last_collision,
            "odor_now":odor,
            "odor_previous":self.last_odor,
            "true_delta":real_delta,
            "true_delta_ema":self.ema,
            "used_delta":delta,
            "used_ema":ema,
            "sham_from_older_sample":arm=="chemo_sham" and source is not None,
        }
        return actual,critic,log

    def next_critic(self,raw_next,arm,current_odor,next_odor,next_tick):
        if arm not in ("chemo_true","chemo_sham","chemo_frozen"):
            return raw_next
        nextdelta=round(clamp(
            SCENT_SCALE*(next_odor-current_odor)),9)
        if arm=="chemo_sham":
            # Look up the upcoming shuffled feedback WITHOUT consuming
            # its separate PRNG ahead of the actual next action.
            saved=self.sham_rng.getstate()
            older=self._sham(next_tick)
            self.sham_rng.setstate(saved)
            nextdelta=older["delta"] if older else 0.
        critic=list(raw_next)
        critic[3]=next_odor
        critic[4]=nextdelta
        return tuple(critic)

    def commit_action(self,action,collision,odor):
        self.last_action=action
        self.last_collision=collision
        self.last_odor=odor


def controller(seed,arm):
    if arm in ("frozen","random_motion","chemo_frozen"):
        return m0.Controller(seed,"frozen")
    return m0.Controller(seed,"plastic")


def simulate(seed,scenario,arm,reference=False):
    if seed not in SEEDS and not reference:
        raise ValueError("M4 seed unregistered")
    if scenario not in SCENARIOS or arm not in ARMS:
        raise ValueError("M4 condition unregistered")
    w=World(seed,scenario)
    mem=SensorMemory(seed)
    policy=controller(seed,arm)
    before={
        "actor":list(policy.actor_start),
        "critic":list(policy.critic_start),
    }
    for tick in range(HORIZON):
        assert w.tick==tick
        w.switch_if_due()
        if w.body.energy<=0. or w.body.health<=0.:
            break
        raw_actions,raw_critic=w.observe()
        odor,noise=w.odor_here()
        effective,critic,hist=mem.encode(
            raw_actions,raw_critic,arm,tick,odor)
        if arm=="random_motion":
            action=min(3,int(4*policy.rng.random()))
            probs=(.25,.25,.25,.25,0.)
        else:
            action,probs=policy.choose(effective)
        real_reward,terminal=w.step(action)
        record=w.trace[-1]
        _,next_raw=w.observe()
        next_odor,_=w.odor_here()
        nextcritic=mem.next_critic(
            next_raw,arm,odor,next_odor,w.tick)
        policy.adapt(effective,critic,action,probs,
                     real_reward,nextcritic,terminal,w.signature())
        record["raw_blind_actor_inputs"]=[
            list(v) for v in raw_actions
        ]
        record["raw_blind_critic_inputs"]=list(raw_critic)
        record["effective_actor_inputs"]=[
            list(v) for v in effective
        ]
        record["effective_critic_inputs"]=list(critic)
        record["measured_odor_scalar"]=odor
        record["odor_measurement_noise"]=noise
        record["memory"]=hist
        mem.commit_action(action,record["collision"],odor)
        if terminal:
            break
    actual=[x for x in w.trace if "action" in x]
    phases={
        "initial":(0,240),
        "changed":(240,360),
        "final":(360,480),
    }
    phase={
        name:{
            "food":sum(x["food"] for x in actual if a<=x["step"]<b),
            "steps":sum(1 for x in actual if a<=x["step"]<b),
            "hazards":sum(x["hazard"] for x in actual if a<=x["step"]<b),
        }
        for name,(a,b) in phases.items()
    }
    return {
        "world_seed":seed,
        "scenario":scenario,
        "arm":arm,
        "start_map":w.start_map,
        "events":w.events,
        "steps_alive":len(actual),
        "survived_480":len(actual)==HORIZON,
        "food_consumed":w.body.food_eaten,
        "food_initial":phase["initial"]["food"],
        "food_changed":phase["changed"]["food"],
        "food_final":phase["final"]["food"],
        "phase":phase,
        "hazard_contacts":w.body.damage_events,
        "wall_collisions":w.body.hits,
        "unique_food_sites":len(w.consumed_food_spots),
        "revisited_food_events":w.body.food_returns,
        "unique_cells":len(w.visited),
        "completed_post_switch_crossings":w.body.crossings_post,
        "initial_weights":before,
        "final_weights":{
            "actor":list(policy.actor),
            "critic":list(policy.critic),
        },
        "weight_updates":policy.parameter_updates,
        "trajectory":w.trace,
    }


def reference_history_compatibility():
    """Exact unchanged M2 and M3 baseline trajectory replay on old seeds."""
    for seed in m3.SEEDS[:4]:
        for new,old in (
            ("frozen","frozen"),
            ("random_motion","random_motion"),
            ("plastic_plain","plastic_plain"),
            ("motor_only","motor_memory"),
        ):
            a=simulate(seed,"baseline",new,reference=True)
            b=m3.simulate(seed,"baseline",old)
            for key,original in (
                ("steps_alive","steps_alive"),
                ("food_consumed","food_consumed"),
                ("initial_weights","initial_weights"),
                ("final_weights","final_weights"),
            ):
                if a[key]!=b[original]:
                    return False
            moves_a=[v for v in a["trajectory"] if "action" in v]
            moves_b=[v for v in b["trajectory"] if "action" in v]
            if len(moves_a)!=len(moves_b):
                return False
            for x,y in zip(moves_a,moves_b):
                if any(x[k]!=y[k] for k in (
                    "step","action","from","position","physical_reward",
                    "food","hazard","energy","health",
                )):
                    return False
    return True


def _validate(rows):
    physics=True
    causal=True
    raw_blind=True
    weights=True
    field_bounded=True
    for row in rows:
        oldpos=(2,6)
        past_scent={}
        past_action=None
        past_collision=0
        last_tick=-1
        for t in row["trajectory"]:
            if "action" not in t:
                if t["event"] not in (
                    "change_gate_openings","relocate_food","restore_food"
                ):
                    physics=False
                continue
            tick=t["step"]
            if tick<=last_tick or tuple(t["from"])!=oldpos:
                physics=False
            last_tick=tick
            oldpos=tuple(t["position"])
            delta=tuple(t["from"][i] for i in range(2))
            move=m0.DIRECTIONS[t["action"]]
            expected=delta if t["collision"] else (
                delta[0]+move[0],delta[1]+move[1]
            )
            if oldpos!=expected:
                physics=False
            gates=(row["start_map"]["future_openings"]
                   if tick>=SWITCH and row["scenario"]!="relocation"
                   else row["start_map"]["initial_openings"])
            walls=m0.World._walls(frozenset(gates))|set(
                map(tuple,row["start_map"]["posts"])
            )
            if oldpos in walls or not (
                0<=t["energy"]<=100 and 0<=t["health"]<=100
            ):
                physics=False
            raw=t["raw_blind_actor_inputs"]
            rc=t["raw_blind_critic_inputs"]
            if (any(f[i]!=0. for f in raw
                    for i in (1,2,3,4,6))
                or rc[3]!=0. or rc[4]!=0.):
                raw_blind=False
            odor=t["measured_odor_scalar"]
            noise=t["odor_measurement_noise"]
            hist=t["memory"]
            if not (0<=odor<=1 and -NOISE_BOUND-1e-6<=noise<=NOISE_BOUND+1e-6):
                field_bounded=False
            if hist["odor_now"]!=odor or (
                hist["odor_previous"]!=past_scent.get(tick-1)
            ):
                causal=False
            true_delta=(round(clamp(
                SCENT_SCALE*(odor-past_scent[tick-1])),9)
                if tick-1 in past_scent else 0.)
            if hist["true_delta"]!=true_delta:
                causal=False
            if (hist["actual_last_action"]!=past_action or
                hist["actual_last_collision"]!=past_collision):
                causal=False
            source=hist["source_tick"]
            if row["arm"]=="chemo_sham":
                if source is not None:
                    if source>tick-2 or source<1 or source not in past_scent:
                        causal=False
                elif tick>=3:
                    causal=False
                if tick<3 and hist["used_delta"]!=0.:
                    causal=False
            elif row["arm"] in ("chemo_true","chemo_frozen"):
                if source!=(tick if tick>=1 else None):
                    causal=False
                if hist["used_delta"]!=true_delta:
                    causal=False
            elif source is not None:
                causal=False
            applied=t["effective_actor_inputs"]
            ec=t["effective_critic_inputs"]
            for i,base in enumerate(raw):
                f=applied[i]
                if any(f[j]!=base[j] for j in (0,5,7)):
                    causal=False
                if row["arm"] in (
                    "frozen","random_motion","plastic_plain"
                ):
                    if f!=base:
                        causal=False
                    continue
                if past_action is None:
                    align=0.
                else:
                    d1=m0.DIRECTIONS[past_action]
                    d2=m0.DIRECTIONS[i]
                    align=float(d1[0]*d2[0]+d1[1]*d2[1])
                if (f[3]!=align or f[4]!=align*past_collision
                    or abs(f[1]-align*hist["used_delta"])>1e-8
                    or abs(f[2]-align*hist["used_ema"])>1e-8):
                    causal=False
            if row["arm"] in (
                "frozen","random_motion","plastic_plain","motor_only"
            ) and ec!=rc:
                causal=False
            if row["arm"] in ("chemo_true","chemo_sham","chemo_frozen"):
                if ec[3]!=odor or ec[4]!=hist["used_delta"]:
                    causal=False
            past_action=t["action"]
            past_collision=t["collision"]
            past_scent[tick]=odor
        if (sum(v["food"] for v in row["trajectory"] if "action" in v)
            !=row["food_consumed"]):
            physics=False
        if row["arm"] in ("frozen","random_motion","chemo_frozen"):
            if (row["initial_weights"]!=row["final_weights"]
                or row["weight_updates"]!=0):
                weights=False
        else:
            if (row["initial_weights"]==row["final_weights"]
                or row["weight_updates"]<1):
                weights=False
        if row["arm"]=="random_motion" and any(
            x["action"]==4 for x in row["trajectory"] if "action" in x
        ):
            weights=False
    return physics,raw_blind,causal,weights,field_bounded


def bootstrap(rows,scenario,arm,reference,n=1000):
    index={(r["world_seed"],r["arm"]):r for r in rows
           if r["scenario"]==scenario}
    def pooled(seeds,which):
        select=[index[(s,which)] for s in seeds]
        return (100*sum(r["food_consumed"] for r in select)/
                sum(r["steps_alive"] for r in select))
    rng=random.Random(83114)
    delta=pooled(SEEDS,arm)-pooled(SEEDS,reference)
    sample=sorted(
        pooled([rng.choice(SEEDS) for _ in SEEDS],arm)
        for _ in [] # no hidden individual-step resampling
    )
    # Paired sampling: use the SAME sampled worlds on both sides.
    series=[]
    for _ in range(n):
        seeds=[rng.choice(SEEDS) for _ in SEEDS]
        series.append(pooled(seeds,arm)-pooled(seeds,reference))
    series.sort()
    return {
        "difference_food_per_100_lived_ticks":round(delta,6),
        "paired_world_seed_bootstrap95_exploratory":[
            round(series[int(n*.025)],6),
            round(series[int(n*.975)-1],6),
        ],
    }


def run():
    prior_controls=reference_history_compatibility()
    rows=[
        simulate(seed,scenario,arm)
        for scenario in SCENARIOS
        for seed in SEEDS
        for arm in ARMS
    ]
    summary={}
    for sc in SCENARIOS:
        summary[sc]={}
        for arm in ARMS:
            group=[r for r in rows if r["scenario"]==sc and r["arm"]==arm]
            denom=sum(r["steps_alive"] for r in group)
            summary[sc][arm]={
                key:round(mean([r[key] for r in group]),6)
                for key in (
                    "steps_alive","food_consumed","food_initial",
                    "food_changed","food_final",
                    "hazard_contacts","wall_collisions",
                    "unique_food_sites","revisited_food_events",
                    "unique_cells","completed_post_switch_crossings"
                )
            }|{
                "food_per_100_living_ticks":round(
                    100*sum(r["food_consumed"] for r in group)/denom,6),
                "survived_480_rate":round(
                    mean([r["survived_480"] for r in group]),6),
            }
    paired=all(
        len({digest(r["start_map"]) for r in rows
             if r["scenario"]==sc and r["world_seed"]==seed})==1
        and len({digest(r["initial_weights"]) for r in rows
                 if r["scenario"]==sc and r["world_seed"]==seed})==1
        for sc in SCENARIOS for seed in SEEDS
    )
    geometry_rows=[r for r in rows if r["scenario"]=="procedural"]
    unique_maps=len({digest(r["start_map"]) for r in geometry_rows})
    geometry_valid=all(
        len(r["start_map"]["posts"])==10
        and not (
            set(r["start_map"]["initial_openings"])
            & set(r["start_map"]["future_openings"])
        )
        and all(
            m2._reachable(
                m0.World._walls(
                    frozenset(r["start_map"][gates])
                )|set(map(tuple,r["start_map"]["posts"]))
            )
            for gates in ("initial_openings","future_openings")
        )
        for r in geometry_rows
    )
    relocation_valid=all(
        r["start_map"]["initial_openings"]==
            r["start_map"]["future_openings"]
        and [e["step"] for e in r["events"]]==(
            [SWITCH,RESTORE] if r["steps_alive"]>RESTORE
            else [SWITCH] if r["steps_alive"]>SWITCH else []
        )
        and (len(r["events"])<2 or
             sorted(r["events"][1]["food_sites"])==
                 sorted(r["start_map"]["food"]))
        and (not r["events"] or
             not(set(map(tuple,r["events"][0]["food_sites"])) &
                 set(map(tuple,r["start_map"]["food"]))))
        for r in rows if r["scenario"]=="relocation"
    )
    relocated_same=all(
        len({
            digest(r["events"][0]["food_sites"]) for r in rows
            if r["scenario"]=="relocation" and
            r["world_seed"]==seed and r["events"]
        })<=1 for seed in SEEDS
    )
    physics,raw_blind,causal,weights,bounded=_validate(rows)
    # Structural/no-leak local chemical field test: the body is
    # enclosed and only food behind the disconnected wall emits.
    probe=World(SEEDS[0],"baseline")
    probe.body.x,probe.body.y=5,6
    probe.food_cooldown={(7,6):0}
    probe._distance_cache={}
    no_path=(7,6) not in probe._distances_from((5,6))
    only_noise=probe.odor_here()[0] <= NOISE_BOUND+1e-9
    base=summary["baseline"];proc=summary["procedural"];rel=summary["relocation"]
    rate=lambda sc,a:sc[a]["food_per_100_living_ticks"]
    gates={
        "01_672_balanced_episodes_repeat_required":len(rows)==672 and
            all(sum(r["scenario"]==sc and r["arm"]==arm for r in rows)==32
                for sc in SCENARIOS for arm in ARMS),
        "02_legal_physical_continuous_motion":physics,
        "03_exact_paired_world_schedule_and_initial_14_weights":paired,
        "04_all_raw_neural_sensors_blind_no_food_hazard_lookahead":raw_blind,
        "05_local_scalar_odor_bounded_wall_occluded_noise_bounded":
            bounded and no_path and only_noise,
        "06_true_consecutive_odor_samples_no_future_and_actor_causality":causal,
        "07_sham_strictly_older_samples_and_independent_rng":causal,
        "08_exact_prior_m2_m3_control_compatibility_same_14_weights":
            prior_controls and weights,
        "09_procedural_connected_and_exogenous_relocation":
            unique_maps>=25 and geometry_valid
            and relocation_valid and relocated_same,
        "10_all_raw_traces_archived_previous_studies_untouched":
            len(rows)==672 and all(r["trajectory"] and
                r["initial_weights"] and r["final_weights"] for r in rows),
        "11_baseline_true_odor_15pct_above_plain_plastic":
            rate(base,"chemo_true")>=1.15*rate(base,"plastic_plain"),
        "12_baseline_true_odor_10pct_above_random":
            rate(base,"chemo_true")>=1.10*rate(base,"random_motion"),
        "13_baseline_true_odor_10pct_above_motor":
            rate(base,"chemo_true")>=1.10*rate(base,"motor_only"),
        "14_procedural_true_odor_10pct_above_random":
            rate(proc,"chemo_true")>=1.10*rate(proc,"random_motion"),
        "15_procedural_true_odor_10pct_above_motor":
            rate(proc,"chemo_true")>=1.10*rate(proc,"motor_only"),
        "16_procedural_true_odor_10pct_above_sham":
            rate(proc,"chemo_true")>=1.10*rate(proc,"chemo_sham"),
        "17_relocation_phase_true_odor_15pct_above_motor":
            rel["chemo_true"]["food_changed"]>=
            1.15*rel["motor_only"]["food_changed"],
        "18_procedural_true_odor_15pct_above_chemo_frozen":
            rate(proc,"chemo_true")>=
            1.15*rate(proc,"chemo_frozen"),
    }
    comparisons={
        sc:{
            ref:bootstrap(rows,sc,"chemo_true",ref)
            for ref in (
                "frozen","random_motion","plastic_plain",
                "motor_only","chemo_sham","chemo_frozen"
            )
        } for sc in SCENARIOS
    }
    return {
        "status":"FlyLoop M4 true local scalar chemosensing",
        "protocol":"docs/FLYLOOP_M4_PROTOCOL_2026-11-24.md",
        "metadata":{
            "world_seeds":SEEDS,"scenarios":SCENARIOS,"arms":ARMS,
            "max_steps":HORIZON,"gate_switch":SWITCH,
            "relocation_restore":RESTORE,
            "neural_parameter_counts":[m0.N_ACTOR,m0.N_CRITIC],
            "scent_proxy":"sum 0.65^graph_distance for available food at distance <=7 /4",
            "scent_noise_abs_max":NOISE_BOUND,
            "scent_delta_scale":SCENT_SCALE,
            "scent_memory_ema":EMA_DECAY,
            "odor_direction_not_directly_supplied":True,
            "only_one_local_scent_scalar_per_actual_location":True,
            "movement_alignment_feature_human_engineered":True,
            "original_m0_learning_rule_unchanged":True,
            "previous_m2_m3_control_exactness":prior_controls,
            "number_unique_procedural_world_maps":unique_maps,
            "novelty_bonus_removed":True,
            "old_m0_m1_m2_m3_m56_unchanged":True,
        },
        "summary":summary,
        "paired_world_seed_exploratory_bootstrap":comparisons,
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
        default=Path("results/flyloop/m4_local_odor.json"))
    args=parser.parse_args()
    result=run()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(
        json.dumps(result,sort_keys=True,separators=(",",":"))+"\n",
        encoding="utf8",
    )
    print("FLYLOOP_M4 "+json.dumps({
        k:result[k] for k in (
            "status","summary","screening_gates","screening_passed_count",
            "screening_pass","paired_world_seed_exploratory_bootstrap",
            "raw_rows_sha256",
        )
    },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
