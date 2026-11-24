"""FlyLoop R4 — action-conditioned local sensor prediction vs passive prediction.

Fixed prospective protocol: docs/FLYLOOP_R4_PROTOCOL_2026-11-24.md
R1/R2/R3 modules unchanged. NO food, map, search/flee planner or
evolution. Motor influence reward, predictor features and LMS are
RESEARCHER-SPECIFIED, NOT self-invented animal objectives.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
from pathlib import Path

from neuroevolution import flyloop_r1 as r1
from neuroevolution import flyloop_r2 as r2
from neuroevolution import flyloop_r3 as r3

SEEDS=tuple(range(9950,9974))
SCENES=("obstacles","nearfield")
ARMS=(
    "recurrent_frozen","r2_progress","passive_probes",
    "agency_gain","sham_gain","prediction_error","inertial_random",
)
HORIZON=r1.HORIZON
LATE_START=300
FEATURES=r1.N+3+2+6+1
HEAD_PARAMS=3*FEATURES
HEAD_LR=.014
HEAD_BOUND=2.
YOKED_LAG=6
HEAD_LEARNING_ARMS=frozenset((
    "passive_probes","agency_gain","sham_gain","prediction_error",
))
MOTOR_LEARNING_ARMS=frozenset((
    "agency_gain","sham_gain","prediction_error",
))
PREDICTOR_NAMES=("true","blind","yoked")


def sha(obj):
    return hashlib.sha256(json.dumps(
        obj,sort_keys=True,separators=(",",":")
    ).encode()).hexdigest()


def mean(values):
    values=list(values)
    return sum(values)/len(values) if values else 0.


def mse(pred,observed):
    return sum((a-b)*(a-b) for a,b in zip(pred,observed))/3.


class World(r1.World):
    def __init__(self,seed,scene):
        if seed not in SEEDS or scene not in SCENES:
            raise ValueError("R4 world seed or scene unregistered")
        self.seed=seed
        self.scenario=scene
        self.rectangles=(
            r1.procedural_rectangles(seed)
            if scene=="obstacles" else r2.nearfield_rectangles(seed)
        )
        self.body=r1.Body(
            heading=random.Random(seed*11113+51).uniform(-math.pi,math.pi)
        )
        self.frame=0
        self.visited={(int(self.body.x),int(self.body.y))}
        self.collisions=0
        self.distance=0.
        self.trace=[]


class ActionSensorHeads:
    """Three predictive heads, same *real* next sensory targets.

    Only the executed-vs-disabled-vs-old motor-command feature differs.
    This is a small self-supervised action-conditioning experiment.
    """
    def __init__(self,seed):
        self.weights={
            name:[[0.]*FEATURES for _ in range(3)]
            for name in PREDICTOR_NAMES
        }
        self.motor_history=[]
        self.yoked_rng=random.Random(seed*32549+63071)
        self.updates={name:0 for name in PREDICTOR_NAMES}

    @staticmethod
    def make_features(hidden,rays,command):
        return (
            list(hidden)+list(rays)+list(command)
            +[v*c for v in rays for c in command]+[1.]
        )

    def command_from_old_motor(self,tick):
        if tick<YOKED_LAG:
            return (0.,0.),None
        source=self.yoked_rng.randrange(tick-YOKED_LAG+1)
        return tuple(self.motor_history[source]),source

    def inputs(self,tick,hidden,rays,executed):
        yoked,source=self.command_from_old_motor(tick)
        phi={
            "true":self.make_features(hidden,rays,executed),
            "blind":self.make_features(hidden,rays,(0.,0.)),
            "yoked":self.make_features(hidden,rays,yoked),
        }
        assert all(len(v)==FEATURES for v in phi.values())
        return phi,yoked,source

    def predict(self,name,rays,features):
        weights=self.weights[name]
        return [
            round(r1.clamp(
                rays[i]+r1.dot(weights[i],features),0.,1.
            ),9)
            for i in range(3)
        ]

    def train_all(self,features,predictions,actual):
        for name in PREDICTOR_NAMES:
            for i in range(3):
                residual=HEAD_LR*(actual[i]-predictions[name][i])
                for j in range(FEATURES):
                    old=self.weights[name][i][j]
                    self.weights[name][i][j]=r1.clamp(
                        old+residual*features[name][j],
                        -HEAD_BOUND,HEAD_BOUND
                    )
            self.updates[name]+=1


def _brain(seed,arm):
    mode=(
        "inertial_random" if arm=="inertial_random"
        else "recurrent_plastic" if arm=="r2_progress"
        else "recurrent_frozen"
    )
    return r2.MotorLearner(seed,mode)


def simulate(seed,scene,arm):
    if seed not in SEEDS or scene not in SCENES or arm not in ARMS:
        raise ValueError("Unregistered R4 cell")
    world=World(seed,scene)
    brain=_brain(seed,arm)
    heads=ActionSensorHeads(seed)
    motor=r3.MotorErrorLearner()
    initial=copy.deepcopy(brain.circuit.weights)
    heading=world.body.heading
    positions=[(world.body.x,world.body.y)]

    for t in range(HORIZON):
        if world.body.energy<=0 or world.body.integrity<=0:
            break
        current=tuple(world.perceive())
        rays=tuple(current[1:4])
        raw,hidden,input_actual,noise,executed=brain.command(current)
        phis,yoked_command,yoked_tick=heads.inputs(
            t,hidden,rays,executed
        )
        preds={
            name:heads.predict(name,rays,phis[name])
            for name in PREDICTOR_NAMES
        }
        # Same trained TRUE head, before it updates, but with an old command.
        true_head_on_yoked=heads.predict("true",rays,phis["yoked"])
        # Physics is executed exactly once; no hypothetical outcome is fed.
        action=world.integrate(executed)
        positions.append((world.body.x,world.body.y))
        next_rays=tuple(world.perceive()[1:4])
        error={name:mse(preds[name],next_rays)
               for name in PREDICTOR_NAMES}
        eval_swap_error=mse(true_head_on_yoked,next_rays)
        actual_gain=r1.clamp(
            (error["blind"]-error["true"])/(error["blind"]+.01),
            -1.,1.,
        )
        sham_gain=r1.clamp(
            (error["blind"]-error["yoked"])/(error["blind"]+.01),
            -1.,1.,
        )
        # ALL learning comes strictly AFTER this real physical outcome.
        if arm in HEAD_LEARNING_ARMS:
            heads.train_all(phis,preds,next_rays)
        heads.motor_history.append(tuple(executed))
        feedback=0.
        if arm=="r2_progress":
            record=brain.adapt(
                t,action["moved"],action["collision"],
                positions,hidden,noise,
            )
            feedback=record["training_feedback"]
        elif arm in MOTOR_LEARNING_ARMS:
            scalar=(
                actual_gain if arm=="agency_gain"
                else sham_gain if arm=="sham_gain"
                else -error["true"]
            )
            # R3 original motor learner transforms LOSS into feedback=-loss.
            feedback=motor.update(brain,hidden,noise,-scalar)
        action.update({
            "tick":t,
            "eight_local_sensor_inputs":[round(v,9) for v in input_actual],
            "hidden_neural_state":[round(v,9) for v in hidden],
            "base_motor_command":[round(v,9) for v in raw],
            "injected_motor_noise":[round(v,9) for v in noise],
            "executed_motor_command":[round(v,9) for v in executed],
            "pre_wall_rays":[round(v,9) for v in rays],
            "yoked_motor_command":[round(v,9) for v in yoked_command],
            "yoked_motor_source_tick":yoked_tick,
            "post_wall_rays":[round(v,9) for v in next_rays],
            "prediction_true":preds["true"],
            "prediction_blind":preds["blind"],
            "prediction_yoked":preds["yoked"],
            "true_head_prediction_with_yoked_motor":true_head_on_yoked,
            "mse_true":round(error["true"],12),
            "mse_blind":round(error["blind"],12),
            "mse_yoked":round(error["yoked"],12),
            "mse_true_head_swapped_motor":round(eval_swap_error,12),
            "agency_feedback_true":round(actual_gain,9),
            "agency_feedback_sham":round(sham_gain,9),
            "motor_plasticity_feedback":round(feedback,9),
        })
        world.trace.append(action)

    final=copy.deepcopy(brain.circuit.weights)
    centroid=(mean(p[0] for p in positions),
              mean(p[1] for p in positions))
    rms=math.sqrt(mean(
        (x-centroid[0])**2+(y-centroid[1])**2
        for x,y in positions
    ))
    late=[f for f in world.trace if f["tick"]>=LATE_START]
    return {
        "seed":seed,"scene":scene,"arm":arm,
        "rectangles":[list(rect) for rect in world.rectangles],
        "initial_pose":[2.2,2.2,round(heading,9)],
        "initial_coefficients":initial,
        "final_coefficients":final,
        "initial_coefficients_sha256":sha(initial),
        "final_coefficients_sha256":sha(final),
        "initial_heads_all_zero":True,
        "final_heads":heads.weights,
        "head_updates":heads.updates,
        "motor_update_events":(
            brain.updates if arm=="r2_progress" else motor.update_events
        ),
        "steps_alive":len(world.trace),
        "distance":round(world.distance,6),
        "unique_grid_cells":len(world.visited),
        "collisions":world.collisions,
        "trajectory_rms_radius":round(rms,6),
        "terminal_energy":round(world.body.energy,6),
        "terminal_integrity":round(world.body.integrity,6),
        "late_steps":len(late),
        "late_mse_sums":{
            key:round(sum(f[key] for f in late),10)
            for key in (
                "mse_true","mse_blind","mse_yoked",
                "mse_true_head_swapped_motor"
            )
        },
        "trace":world.trace,
    }


def _paired_bootstrap(rows,scene,arm,ref,metric,n=1000):
    idx={(row["seed"],row["arm"]):row for row in rows
         if row["scene"]==scene}
    differences=[
        idx[(s,arm)][metric]-idx[(s,ref)][metric]
        for s in SEEDS
    ]
    rng=random.Random(99507)
    draws=sorted(
        mean(differences[rng.randrange(len(differences))]
             for _ in differences)
        for _ in range(n)
    )
    return {
        "mean_paired_delta":round(mean(differences),6),
        "exploratory_world_bootstrap95":[
            round(draws[25],6),round(draws[974],6)
        ],
    }


def _audit(rows):
    physical=True
    input_causality=True
    source_past=True
    legal_weights=True
    prediction_heads=True
    for row in rows:
        initial=row["initial_coefficients"]
        final=row["final_coefficients"]
        if any(initial[field]!=final[field]
               for field in ("w_input","w_recurrent","bias")):
            legal_weights=False
        frozen_motor=row["arm"] in (
            "recurrent_frozen","passive_probes","inertial_random"
        )
        if frozen_motor and initial!=final:
            legal_weights=False
        if not frozen_motor and row["motor_update_events"]!=row["steps_alive"]:
            legal_weights=False
        should_train=row["arm"] in HEAD_LEARNING_ARMS
        for name in PREDICTOR_NAMES:
            if row["head_updates"][name]!=(
                row["steps_alive"] if should_train else 0
            ):
                prediction_heads=False
            w=row["final_heads"][name]
            if len(w)!=3 or any(len(v)!=FEATURES for v in w):
                prediction_heads=False
            if not should_train and any(value!=0 for v in w for value in v):
                prediction_heads=False
        prev=(2.2,2.2)
        motor_past=[]
        for i,t in enumerate(row["trace"]):
            if t["tick"]!=i or any(
                abs(a-b)>2e-7 for a,b in zip(t["before_position"],prev)
            ):
                physical=False
            actual=tuple(t["position"])
            if t["collision"] and any(
                abs(a-b)>1e-7 for a,b in zip(actual,prev)
            ):
                physical=False
            if not (
                r1.RADIUS-1e-7<=actual[0]<=r1.SIDE-r1.RADIUS+1e-7 and
                r1.RADIUS-1e-7<=actual[1]<=r1.SIDE-r1.RADIUS+1e-7 and
                0<=t["energy"]<=100 and 0<=t["integrity"]<=100
            ):
                physical=False
            if any(
                x1-r1.RADIUS<actual[0]<x2+r1.RADIUS and
                y1-r1.RADIUS<actual[1]<y2+r1.RADIUS
                for x1,y1,x2,y2 in row["rectangles"]
            ):
                physical=False
            prev=actual
            rays=t["pre_wall_rays"]
            next_rays=t["post_wall_rays"]
            if (
                len(t["eight_local_sensor_inputs"])!=r1.INPUTS or
                len(t["hidden_neural_state"])!=r1.N or
                rays!=t["eight_local_sensor_inputs"][1:4] or
                len(t["executed_motor_command"])!=2
            ):
                input_causality=False
            if i+1<len(row["trace"]) and next_rays!=(
                row["trace"][i+1]["pre_wall_rays"]
            ):
                input_causality=False
            if any(not 0<=v<=1 for v in (
                rays+next_rays+t["prediction_true"]
                +t["prediction_blind"]+t["prediction_yoked"]
            )):
                input_causality=False
            for k in ("true","blind","yoked"):
                if abs(mse(
                    t["prediction_"+k],next_rays
                )-t["mse_"+k])>1e-7:
                    input_causality=False
            src=t["yoked_motor_source_tick"]
            old=t["yoked_motor_command"]
            if src is None:
                if i>=YOKED_LAG or old!=[0.,0.]:
                    source_past=False
            elif not (
                0<=src<=i-YOKED_LAG and
                old==motor_past[src]
            ):
                source_past=False
            if abs(mse(
                t["true_head_prediction_with_yoked_motor"],next_rays
            )-t["mse_true_head_swapped_motor"])>1e-7:
                input_causality=False
            motor_past.append(t["executed_motor_command"])
        if row["steps_alive"]!=len(row["trace"]):
            physical=False
    return {
        "physical_no_wall_or_life_violation":physical,
        "three_head_prediction_preaction_postaction_causality":input_causality,
        "older_motor_command_is_own_past_6_step_minimum":source_past,
        "upstream_434_coefficient_substrate_intact":legal_weights,
        "84_parameter_heads_only_train_in_permitted_arms":prediction_heads,
    }


def run():
    rows=[
        simulate(seed,scene,arm)
        for scene in SCENES
        for seed in SEEDS
        for arm in ARMS
    ]
    summary={}
    for scene in SCENES:
        summary[scene]={}
        for arm in ARMS:
            group=[r for r in rows if r["scene"]==scene and r["arm"]==arm]
            late_steps=sum(r["late_steps"] for r in group)
            meters=sum(r["distance"] for r in group)
            summary[scene][arm]={
                key:round(mean(r[key] for r in group),6)
                for key in (
                    "steps_alive","distance","unique_grid_cells",
                    "collisions","trajectory_rms_radius",
                    "terminal_integrity","terminal_energy",
                )
            }|{
                "contacts_per_meter":round(
                    sum(r["collisions"] for r in group)/
                    max(meters,1e-12),6
                ),
                "late_mse":{
                    key:round(sum(
                        r["late_mse_sums"][key] for r in group
                    )/max(late_steps,1),9)
                    for key in (
                        "mse_true","mse_blind","mse_yoked",
                        "mse_true_head_swapped_motor"
                    )
                },
                "late_steps":late_steps,
            }
    checks=_audit(rows)
    paired=all(
        len({sha(r["rectangles"]) for r in rows
             if r["scene"]==scene and r["seed"]==seed})==1 and
        len({sha(r["initial_pose"]) for r in rows
             if r["scene"]==scene and r["seed"]==seed})==1 and
        len({r["initial_coefficients_sha256"] for r in rows
             if r["scene"]==scene and r["seed"]==seed})==1
        for scene in SCENES for seed in SEEDS
    )
    noise_paired=all(
        all(
            [t["injected_motor_noise"] for t in r["trace"][:limit]]
            ==[t["injected_motor_noise"] for t in group[0]["trace"][:limit]]
            for r in group
        )
        for scene in SCENES for seed in SEEDS
        for group in [[r for r in rows
                       if r["scene"]==scene and r["seed"]==seed
                       and r["arm"]!="inertial_random"]]
        for limit in [min(r["steps_alive"] for r in group)]
    )
    maps={
        scene:len({
            sha(r["rectangles"]) for r in rows if r["scene"]==scene
        }) for scene in SCENES
    }
    valid_maps=all(
        len(r["rectangles"])==r1.OBSTACLES
        and all(not(x1-1.1<2.2<x2+1.1 and
                    y1-1.1<2.2<y2+1.1)
                for x1,y1,x2,y2 in r["rectangles"])
        for r in rows
    )
    frozen_heads=all(
        r["arm"]!="recurrent_frozen" or all(
            t["prediction_true"]==t["pre_wall_rays"]
            and t["prediction_blind"]==t["pre_wall_rays"]
            and t["prediction_yoked"]==t["pre_wall_rays"]
            for t in r["trace"]
        ) for r in rows
    )
    pass_s=summary["nearfield"]["passive_probes"]["late_mse"]
    near=summary["nearfield"]
    rate=lambda arm:near[arm]["unique_grid_cells"]
    collisions=lambda arm:near[arm]["contacts_per_meter"]
    gates={
        "01_exact_336_balanced_full_replay_byte_identical_required":
            len(rows)==336 and all(
                sum(r["scene"]==scene and r["arm"]==arm for r in rows)
                ==len(SEEDS) for scene in SCENES for arm in ARMS
            ),
        "02_original_continuous_physical_obstacle_boundary_and_death":
            checks["physical_no_wall_or_life_violation"],
        "03_paired_world_pose_weights_and_noise":
            paired and noise_paired,
        "04_16_hidden_8_local_sensor_values_no_future_map":
            checks["three_head_prediction_preaction_postaction_causality"],
        "05_exact_84_head_coefficients_and_only_34_motor_mutable":
            checks["84_parameter_heads_only_train_in_permitted_arms"]
            and checks["upstream_434_coefficient_substrate_intact"],
        "06_predictions_preaction_training_only_real_postaction_rays":
            checks["three_head_prediction_preaction_postaction_causality"],
        "07_actual_vs_zero_motor_features_reconstructable":
            checks["three_head_prediction_preaction_postaction_causality"],
        "08_yoked_motor_only_own_actual_history_at_least_6_old":
            checks["older_motor_command_is_own_past_6_step_minimum"],
        "09_passive_heads_same_real_trajectory_pre_update_swapped_eval":
            all(r["arm"]!="passive_probes" or
                r["initial_coefficients"]==r["final_coefficients"]
                for r in rows) and frozen_heads,
        "10_two_valid_exogenous_generators_min20_distinct_maps":
            valid_maps and all(maps[s]>=20 for s in SCENES),
        "11_all_raw_data_stop_flags_prior_studies_untouched":
            all(r["trace"] and r["final_heads"] for r in rows),
        "12_passive_true_head_late_mse_le_blind_85pct":
            pass_s["mse_blind"]>0 and pass_s["mse_true"]<=
                .85*pass_s["mse_blind"],
        "13_passive_true_head_late_mse_le_yoked_85pct":
            pass_s["mse_yoked"]>0 and pass_s["mse_true"]<=
                .85*pass_s["mse_yoked"],
        "14_same_true_head_actual_command_late_mse_le_swapped_85pct":
            pass_s["mse_true_head_swapped_motor"]>0
            and pass_s["mse_true"]<=
                .85*pass_s["mse_true_head_swapped_motor"],
        "15_agency_gain_coverage_ge_frozen_plus10pct":
            rate("agency_gain")>=1.10*rate("recurrent_frozen"),
        "16_agency_gain_collision_density_le_frozen_90pct":
            collisions("agency_gain")<=.90*collisions("recurrent_frozen"),
        "17_agency_gain_coverage_ge_sham_plus10pct":
            rate("agency_gain")>=1.10*rate("sham_gain"),
        "18_agency_gain_coverage_ge_minimum_surprise_plus10pct":
            rate("agency_gain")>=1.10*rate("prediction_error"),
    }
    bootstrap={
        scene:{
            ref:_paired_bootstrap(
                rows,scene,"agency_gain",ref,"unique_grid_cells"
            ) for ref in (
                "recurrent_frozen","sham_gain","prediction_error",
                "r2_progress","inertial_random"
            )
        } for scene in SCENES
    }
    return {
        "name":"FlyLoop R4 true vs motor-blind vs old-action predicted sensory changes",
        "protocol":"docs/FLYLOOP_R4_PROTOCOL_2026-11-24.md",
        "metadata":{
            "seeds":SEEDS,"world_families":SCENES,"arms":ARMS,
            "existing_recurrent_neurons":r1.N,
            "original_neural_coefficients":r1.PARAMS,
            "coefficients_per_predictor_head":HEAD_PARAMS,
            "predictor_head_count":len(PREDICTOR_NAMES),
            "learning_rate_head":HEAD_LR,
            "learning_rate_motor":r2.LR,
            "min_age_of_yoked_motor_command":YOKED_LAG,
            "motor_perturbation_sd":r2.EXPLORATION_SD,
            "late_phase_from_step":LATE_START,
            "no_food_or_evolution":True,
            "intrinsic_controllability_reward_is_engineered":True,
            "no_counterfactual_true_outcome_assumed":True,
            "old_R1_R2_R3_M0_M4_main_untouched":True,
            "number_distinct_maps":maps,
        },
        "summary":summary,
        "integrity":checks,
        "paired_world_seed_bootstrap_exploratory":bootstrap,
        "gates":gates,
        "gates_passed_count":sum(gates.values()),
        "screen_pass":all(gates.values()),
        "food_search_demonstrated":False,
        "escape_reflex_learned":False,
        "autonomous_goal_invention_demonstrated":False,
        "general_intelligence_demonstrated":False,
        "biological_fly_replication_demonstrated":False,
        "evolution_authorized":False,
        "advance_to_evolution":False,
        "raw_rows_sha256":sha(rows),
        "raw_rows":rows,
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument(
        "--out",type=Path,
        default=Path("results/flyloop_r4/r4_motor_controllability.json"),
    )
    args=parser.parse_args()
    result=run()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(
        json.dumps(result,sort_keys=True,separators=(",",":"))+"\n",
        encoding="utf8",
    )
    print("FLYLOOP_R4 "+json.dumps({
        k:result[k] for k in (
            "name","summary","integrity","gates","gates_passed_count",
            "screen_pass","paired_world_seed_bootstrap_exploratory",
            "raw_rows_sha256",
        )
    },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
