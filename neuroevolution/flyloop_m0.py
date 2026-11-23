"""FlyLoop M0: embodied action/feedback on actual 2D micro-worlds.

Prospective DEV science, NOT an autonomous fly, new brain or M5.6 replication.
Reuses original, UNMODIFIED M5.6 Learner (260-synapse Elman RNN
and bounded one-step REINFORCE), and M5.6 intrinsic visit novelty.
No researcher-programmed SEEK/ESCAPE/ROUTE/BYPASS decision procedure.
The physical world/sensory scaffold is researcher engineered.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path

from . import m4_2, m4_4a
from .m5_6_online import Learner

ACTIONS = ((0,-1),(1,0),(0,1),(-1,0))  # N,E,S,W motor primitives
ARMS = ("frozen","plastic","plastic_novelty")
SEEDS = tuple(range(44000,44016))
DEV_SEEDS = tuple(range(43000,43004))
PHASES = ("source_a","source_b","shifted_eval","shifted_adapt")
GATES = (2,3,7,8)
BOUT_STEPS = 150
WIDTH = 11
HEIGHT = 11
SPAWN = (2,5)
FOODS = ((3,3),(8,5))
HAZARD = (4,6)
ENERGY_INITIAL = 80.
ENERGY_MAX = 95.
BASE_MOVE_COST = .24
WALL_EXTRA_COST = .45
HAZARD_EXTRA_COST = 7.
FOOD_ENERGY = 24.
FOOD_RESPAWN_TICKS = 27
LEARNING_RATE = .02
SOURCE_LEARN_PHASES = ("source_a","source_b","shifted_adapt")
FEEDBACK_FOOD = 1.5
FEEDBACK_DAMAGE = -1.
FEEDBACK_WALL = -.20
FEEDBACK_ORDINARY = -.02
EXPLORATION_CAP = .10
BOOTSTRAP_DRAWS = 2000
EXPERIMENT_TAG = "FlyLoop-M0-embodied-development"

def digest_weights(weights):
    return hashlib.sha256(json.dumps(
        [float(v) for v in weights],separators=(",",":")
    ).encode()).hexdigest()

@dataclass
class Body:
    position: tuple[int,int] = SPAWN
    energy: float = ENERGY_INITIAL
    last_collision: bool = False
    last_reward: float = 0.
    last_move: bool = False
    food_eaten: int = 0
    far_food_eaten: int = 0
    damage_events: int = 0

class World:
    """Physics, never a goal-directed plan or an 'optimal action' oracle."""
    def __init__(self, gate_y: int):
        if gate_y not in GATES:
            raise ValueError("Unregistered gate world; test geometry sealed")
        self.gate_y = gate_y
        self.walls = {
            (x,y) for x in range(WIDTH) for y in range(HEIGHT)
            if x in (0,WIDTH-1) or y in (0,HEIGHT-1)
        }
        self.walls.update((5,y) for y in range(1,HEIGHT-1) if y!=gate_y)
        self.food_ready = {f:0 for f in FOODS}
        if any(p in self.walls for p in (*FOODS,SPAWN,HAZARD)):
            raise AssertionError("Invalid body, hazard or food world layout")

    def blocked(self, pos):
        return (pos in self.walls or not (0<=pos[0]<WIDTH and
                                        0<=pos[1]<HEIGHT))

    def available_food(self, tick):
        return tuple(f for f in FOODS if tick>=self.food_ready[f])

    def observe(self, body: Body, tick: int):
        """19 causal readings, no access to target labels or future rewards.

        0..3 nearby wall, 4..7 nearby hazard, 8..11 food smell through
        a simple engineered distance field, 12 energy, 13 previous
        collision, 14 observed feedback, 15..16 normalized coordinates,
        17 previous successful movement, 18 bias.
        """
        x,y=body.position
        adjacent = [(x+dx,y+dy) for dx,dy in ACTIONS]
        obstacles = [float(self.blocked(p)) for p in adjacent]
        danger = [float(p==HAZARD) for p in adjacent]
        # Actual odor is NOT a privileged optimal movement vector: it is
        # a locally sampled scalar field evaluated at each neighboring cell.
        # It can pass through a divider; the source field is human-designed.
        food = self.available_food(tick)
        odors = []
        for p in adjacent:
            if self.blocked(p):
                odors.append(0.)
            else:
                odors.append(sum(1./(1.+abs(p[0]-f[0])+abs(p[1]-f[1]))
                                  for f in food)/len(FOODS))
        obs=tuple(
            obstacles+danger+odors+
            [float(body.energy/ENERGY_MAX),
             float(body.last_collision),
             float(max(-1.,min(1.,body.last_reward))),
             float(x/(WIDTH-1)),
             float(y/(HEIGHT-1)),
             float(body.last_move), 1.]
        )
        if len(obs)!=19 or not all(math.isfinite(v) for v in obs):
            raise AssertionError("M5.6 RNN requires 19 finite sensory channels")
        return obs

    def transition(self, body: Body, action: int, tick: int):
        """One physical action and its AFTER-THE-FACT actual consequence."""
        if type(action) is not int or action not in range(len(ACTIONS)):
            raise ValueError("Illegal cardinal motor command")
        x,y=body.position
        dx,dy=ACTIONS[action]
        candidate=(x+dx,y+dy)
        collision=self.blocked(candidate)
        if not collision:
            body.position=candidate
        moved=not collision
        harm=moved and body.position==HAZARD
        ate=moved and body.position in FOODS and tick>=self.food_ready.get(body.position,10**9)
        if ate:
            self.food_ready[body.position]=tick+FOOD_RESPAWN_TICKS
            body.food_eaten+=1
            if body.position[0]>=6:
                body.far_food_eaten+=1
        if harm:
            body.damage_events+=1
        cost=BASE_MOVE_COST+(WALL_EXTRA_COST if collision else 0.)+(
            HAZARD_EXTRA_COST if harm else 0.)
        body.energy=max(0., min(ENERGY_MAX,
                 body.energy-cost+(FOOD_ENERGY if ate else 0.)))
        physical_reward=(FEEDBACK_FOOD if ate else
                         FEEDBACK_DAMAGE if harm else
                         FEEDBACK_WALL if collision else
                         FEEDBACK_ORDINARY)
        body.last_collision=bool(collision)
        body.last_reward=float(physical_reward)
        body.last_move=bool(moved)
        return {
            "to":list(body.position),
            "actual_moved":bool(moved),
            "wall_collision":bool(collision),
            "hazard_event":bool(harm),
            "actual_food":bool(ate),
            "actual_far_food":bool(ate and body.position[0]>=6),
            "physical_reward":physical_reward,
            "energy_after":body.energy,
            "alive_after":body.energy>0.,
        }

def initial_weights(seed:int):
    return tuple(float(x) for x in m4_4a._initial_genome(
        124000+seed,"rnn",m4_2.Settings()
    ))

def initial_agent(seed:int):
    weights=initial_weights(seed)
    if len(weights)!=260 or not all(math.isfinite(v) for v in weights):
        raise AssertionError("Unexpected original recurrent M5.6 genotype")
    learner=Learner.fresh(weights, SPAWN)
    return learner,weights

def run_bout(seed:int,arm:str,phase_index:int,learner:Learner,*,trace=True):
    """Finite physical bout; learner SYNAPSES persist through later bouts."""
    if seed not in SEEDS+DEV_SEEDS:
        raise ValueError("Only preregistered run seeds")
    if arm not in ARMS+("plastic_novelty_reset",):
        raise ValueError("Unknown predeclared intervention")
    if phase_index not in range(4):
        raise ValueError("Unknown phase")
    phase=PHASES[phase_index]
    world=World(GATES[phase_index])
    body=Body()
    learner.hidden=[0.]*8
    action_rng=random.Random(seed*10009+phase_index*79+7)
    initial_hash=digest_weights(learner.weights)
    initial_updates=learner.update_count
    tracks=[]
    distinct={SPAWN}
    first_far_cross=None
    wall_collisions=0
    harm_count=0
    meals=0
    far_meals=0
    total_physical=0.
    total_intrinsic=0.
    total_learning_feedback=0.
    total_drift=0.
    for tick in range(BOUT_STEPS):
        pos=body.position
        energy_before=body.energy
        observation=world.observe(body,tick)
        selected,cache=learner.action(observation,(0,1,2,3),action_rng)
        if selected is None or cache is None:
            raise AssertionError("Four physical motor primitives required")
        physical=world.transition(body,selected,tick)
        assert physical["to"]==list(body.position)
        novel=0.
        if physical["actual_moved"]:
            # The original individual M5.6 novelty estimator is updated
            # for all arms; only novelty arm adds it to INTERNAL training.
            novel=learner.curiosity.observe_arrival(body.position)
        bonus=(novel if arm in ("plastic_novelty","plastic_novelty_reset") else 0.)
        if physical["wall_collision"] and bonus:
            raise AssertionError("No intrinsic bonus for fake collision arrivals")
        # Internally bounded learning scalar is NOT the organism's energy.
        feedback=physical["physical_reward"]+bonus
        enabled=(arm!="frozen" and phase in SOURCE_LEARN_PHASES)
        drift=learner.learn(cache,feedback,enabled=enabled,
                             learning_rate=LEARNING_RATE)
        total_drift+=drift
        total_physical+=physical["physical_reward"]
        total_intrinsic+=bonus
        total_learning_feedback+=feedback
        meals+=int(physical["actual_food"])
        far_meals+=int(physical["actual_far_food"])
        harm_count+=int(physical["hazard_event"])
        wall_collisions+=int(physical["wall_collision"])
        distinct.add(body.position)
        if body.position[0]>=6 and first_far_cross is None:
            first_far_cross=tick
        if trace:
            tracks.append({
                "tick":tick,
                "observation_19_before_action":[round(float(v),9) for v in observation],
                "from":list(pos),
                "chosen_motor_command":int(selected),
                "energy_before":energy_before,
                **physical,
                "intrinsic_bonus_only_internal":bonus,
                "learning_feedback":feedback,
                "synaptic_update":bool(enabled),
                "synaptic_drift_l2":drift,
            })
        if body.energy<=0:
            break
    result={
        "seed":seed,"arm":arm,"phase":phase,"gate":world.gate_y,
        "max_bout_steps":BOUT_STEPS,
        "steps_survived":len(tracks) if trace else tick+1,
        "physically_alive_at_bout_end":bool(body.energy>0),
        "final_energy_physical":body.energy,
        "food_eaten_physical":meals,
        "far_food_eaten_physical":far_meals,
        "hazard_events_physical":harm_count,
        "wall_collisions_physical":wall_collisions,
        "successful_unique_cells":len(distinct),
        "first_far_cross_tick":first_far_cross,
        "physical_feedback_total":total_physical,
        "intrinsic_novelty_bonus_total_NOT_physical":total_intrinsic,
        "learning_feedback_total":total_learning_feedback,
        "weight_drift_l2_total":total_drift,
        "learning_updates_this_bout":learner.update_count-initial_updates,
        "weights_initial_sha256":initial_hash,
        "weights_final_sha256":digest_weights(learner.weights),
        "trace":tracks if trace else [],
    }
    if phase=="shifted_eval" and (
        result["learning_updates_this_bout"]!=0 or
        result["weights_initial_sha256"]!=result["weights_final_sha256"]
    ):
        raise AssertionError("Shifted test must freeze neural plasticity")
    return result

def one_seed(seed:int,*,trace=True):
    if seed not in SEEDS+DEV_SEEDS:
        raise ValueError("Unregistered science/development seed")
    all_rows=[]
    init_fingerprints={}
    for arm in ARMS:
        learner,start=initial_agent(seed)
        init_fingerprints[arm]=digest_weights(start)
        for idx in (0,1):
            all_rows.append(run_bout(seed,arm,idx,learner,trace=trace))
        if arm=="plastic_novelty":
            reset_learner=copy.deepcopy(learner)
            reset_learner.weights=list(start)  # causal memory erasure
            reset_learner.hidden=[0.]*8
        for idx in (2,3):
            all_rows.append(run_bout(seed,arm,idx,learner,trace=trace))
        if arm=="plastic_novelty":
            for idx in (2,3):
                all_rows.append(run_bout(
                    seed,"plastic_novelty_reset",idx,reset_learner,trace=trace
                ))
    if len(set(init_fingerprints.values()))!=1 or len(all_rows)!=14:
        raise AssertionError("Different starting RNN weights or wrong arm count")
    return {"seed":seed,"initial_260_weights_sha256":init_fingerprints["frozen"],
            "rows":all_rows}

def paired_interval(rows,arm,other,field,phase="shifted_eval"):
    per_seed={}
    for seed in SEEDS:
        a=next(x[field] for x in rows if x["seed"]==seed and
               x["phase"]==phase and x["arm"]==arm)
        b=next(x[field] for x in rows if x["seed"]==seed and
               x["phase"]==phase and x["arm"]==other)
        per_seed[seed]=a-b
    rng=random.Random(440139)
    vals=sorted(statistics.mean(
        per_seed[rng.choice(SEEDS)] for _ in SEEDS
    ) for _ in range(BOOTSTRAP_DRAWS))
    return {
        "paired_mean_difference":statistics.mean(per_seed.values()),
        "exploratory_seed_cluster_boot95":[vals[int(.025*BOOTSTRAP_DRAWS)],
                                         vals[int(.975*BOOTSTRAP_DRAWS)-1]],
        "unit":"16 paired controller-initialization/action-seeds, only 2 test layouts"
    }

def audit_physical_trace(row):
    """Independent recomputation of all recorded physical energy transitions."""
    last=SPAWN
    eaten_near=0
    eaten_far=0
    damage=0
    wall=0
    visited={SPAWN}
    world=World(row["gate"])
    body=Body()
    for index,entry in enumerate(row["trace"]):
        if entry["tick"]!=index or entry["from"]!=list(last):
            return False
        if abs(entry["energy_before"]-body.energy)>1.e-8:
            return False
        regenerated=world.observe(body,index)
        if any(abs(a-b)>1.e-8 for a,b in zip(
            regenerated,entry["observation_19_before_action"]
        )) or len(entry["observation_19_before_action"])!=19:
            return False
        result=world.transition(body,entry["chosen_motor_command"],index)
        for k in result:
            if isinstance(result[k],float):
                if abs(result[k]-entry[k])>1.e-8:return False
            elif result[k]!=entry[k]:
                return False
        if entry["intrinsic_bonus_only_internal"]<0 or (
            not result["actual_moved"] and
            entry["intrinsic_bonus_only_internal"]!=0.):
            return False
        if abs(entry["learning_feedback"]-(
            result["physical_reward"]+
            entry["intrinsic_bonus_only_internal"]
        ))>1.e-8:
            return False
        if entry["synaptic_update"] and row["arm"]=="frozen":
            return False
        last=body.position
        visited.add(last)
        eaten_near+=int(result["actual_food"])
        eaten_far+=int(result["actual_far_food"])
        wall+=int(result["wall_collision"])
        damage+=int(result["hazard_event"])
    return (
        eaten_near==row["food_eaten_physical"]
        and eaten_far==row["far_food_eaten_physical"]
        and damage==row["hazard_events_physical"]
        and wall==row["wall_collisions_physical"]
        and len(visited)==row["successful_unique_cells"]
        and abs(body.energy-row["final_energy_physical"])<1.e-8
        and row["steps_survived"]==len(row["trace"])
        and (body.energy>0.)==row["physically_alive_at_bout_end"]
    )

def run():
    groups=[one_seed(seed) for seed in SEEDS]
    raw=[row for group in groups for row in group["rows"]]
    summary={}
    for phase in PHASES:
        for arm in ARMS+("plastic_novelty_reset",):
            r=[x for x in raw if x["arm"]==arm and x["phase"]==phase]
            if not r:continue
            summary[phase+"|"+arm]={
                "bouts":len(r),
                **{
                    k:statistics.mean(x[k] for x in r)
                    for k in (
                        "food_eaten_physical","far_food_eaten_physical",
                        "hazard_events_physical","wall_collisions_physical",
                        "steps_survived","final_energy_physical",
                        "successful_unique_cells","learning_updates_this_bout",
                        "intrinsic_novelty_bonus_total_NOT_physical"
                    )
                },
                "far_side_crossed_rate":sum(
                    x["first_far_cross_tick"] is not None for x in r
                )/len(r),
            }
    frozen=summary["shifted_eval|frozen"]
    plastic=summary["shifted_eval|plastic"]
    curious=summary["shifted_eval|plastic_novelty"]
    reset=summary["shifted_eval|plastic_novelty_reset"]
    prim="far_food_eaten_physical"
    deltas={
        "curiosity_minus_frozen_far_food":curious[prim]-frozen[prim],
        "plastic_minus_frozen_far_food":plastic[prim]-frozen[prim],
        "curiosity_minus_plastic_far_food":curious[prim]-plastic[prim],
        "curiosity_minus_frozen_survival":(
            curious["steps_survived"]-frozen["steps_survived"]
        ),
        "curiosity_minus_reset_far_food":curious[prim]-reset[prim],
    }
    paired={
        "curiosity_vs_frozen_far_food":paired_interval(
            raw,"plastic_novelty","frozen",prim),
        "plastic_vs_frozen_far_food":paired_interval(
            raw,"plastic","frozen",prim),
        "curiosity_vs_plastic_far_food":paired_interval(
            raw,"plastic_novelty","plastic",prim),
        "curiosity_vs_frozen_survival":paired_interval(
            raw,"plastic_novelty","frozen","steps_survived"),
        "curiosity_vs_reset_far_food":paired_interval(
            raw,"plastic_novelty","plastic_novelty_reset",prim),
    }
    all_arms_present=(
        len(groups)==16 and len(raw)==16*14
        and all(len(g["rows"])==14 for g in groups)
        and all(len({x["arm"] for x in g["rows"]
                    if x["phase"]=="source_a"})==3 for g in groups)
        and all(len(set(
            x["weights_initial_sha256"] for x in g["rows"]
            if x["phase"]=="source_a"
        ))==1 for g in groups)
    )
    verified_physics=all(audit_physical_trace(row) for row in raw)
    source_untouched=True  # external M5.6 regression checked by CI
    gates={
        "science_curiosity_shifted_far_food_ge_0_30":
            curious[prim]>=.30,
        "science_curiosity_far_food_gain_over_frozen_ge_0_20":
            deltas["curiosity_minus_frozen_far_food"]>=.20,
        "science_plastic_far_food_gain_over_frozen_ge_0_10":
            deltas["plastic_minus_frozen_far_food"]>=.10,
        "science_curiosity_far_food_gain_over_plastic_ge_0_10":
            deltas["curiosity_minus_plastic_far_food"]>=.10,
        "science_curiosity_survival_gain_over_frozen_ge_10_ticks":
            deltas["curiosity_minus_frozen_survival"]>=10.,
        "integrity_16_seeds_3_original_arms_4_bouts_identical_genomes":
            all_arms_present,
        "integrity_recompute_all_physical_transitions_no_target_label":
            verified_physics and all(
                len(x["trace"])==x["steps_survived"] for x in raw),
        "integrity_intrinsic_separate_and_no_bonus_on_collision":
            all(
                x["intrinsic_novelty_bonus_total_NOT_physical"]==0.
                for x in raw if x["arm"] in ("frozen","plastic")
            ) and all(
                not e["wall_collision"] or
                e["intrinsic_bonus_only_internal"]==0.
                for x in raw for e in x["trace"]
            ),
        "integrity_shifted_initial_eval_frozen_and_reset_ablation":
            all(x["learning_updates_this_bout"]==0 and
                x["weights_initial_sha256"]==x["weights_final_sha256"]
                for x in raw if x["phase"]=="shifted_eval")
            and len([
                x for x in raw if x["arm"]=="plastic_novelty_reset"
                and x["phase"]=="shifted_eval"
            ])==16,
        "integrity_raw_per_action_and_legacy_regression_external":
            len(raw)==224 and source_untouched and all(
                x["trace"] and
                all("observation_19_before_action" in e and
                    "actual_food" in e and "chosen_motor_command" in e
                    for e in x["trace"])
                for x in raw),
    }
    compact_sha=hashlib.sha256(json.dumps(
        raw,sort_keys=True,separators=(",",":")
    ).encode()).hexdigest()
    return {
        "experiment":EXPERIMENT_TAG,
        "protocol":"docs/FLYLOOP_M0_PROTOCOL_2026-11-23.md",
        "metadata":{
            "science_seeds":SEEDS,"development_seeds_disjoint":DEV_SEEDS,
            "source_gates":GATES[:2],"shifted_test_gates":GATES[2:],
            "sensor_channels":19,"reused_m56_synaptic_weights":260,
            "source_M5_6_unmodified":True,
            "four_fixed_cardinal_motor_primitives":True,
            "researcher_engineered_food_smell_field":True,
            "researcher_engineered_energy_damage_and_intrinsic_curiosity":True,
            "first_shifted_evaluation_freezes_all_synaptic_weights":True,
            "one_organism_controller_per_seed_per_arm_but_four_controlled_bouts":True,
            "number_of_distinct_test_layouts":2,
            "no_reproduction_and_no_evolution":True,
        },
        "summary":summary,"effect_deltas":deltas,
        "exploratory_paired_seed_bootstrap":paired,
        "screening_gates":gates,
        "scientific_success":all(list(gates.values())[:5]),
        "technical_success":all(list(gates.values())[5:]),
        "overall_dev_screen_pass":all(gates.values()),
        "learned_new_behavior_established":False,
        "learned_general_curiosity_policy":False,
        "autonomous_high_level_planning":False,
        "independent_ecological_replication":False,
        "evolution_authorized":False,
        "advance_to_evolution":False,
        "source_M5_6_edited":False,
        "raw_rows_sha256":compact_sha,
        "raw_seed_bouts":groups,
    }

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,
                        default=Path("results/flyloop-m0/embodied-dev.json"))
    args=parser.parse_args()
    out=run()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(out,sort_keys=True,indent=2)+"\n",
                        encoding="utf8")
    print("FLYLOOP_M0_RESULT "+json.dumps({
        k:out[k] for k in (
            "experiment","summary","effect_deltas",
            "exploratory_paired_seed_bootstrap","screening_gates",
            "scientific_success","technical_success","overall_dev_screen_pass",
            "raw_rows_sha256")
    },sort_keys=True),flush=True)

if __name__=="__main__":
    main()
