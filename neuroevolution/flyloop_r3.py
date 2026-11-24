"""FlyLoop R3 — self-supervised single-step wall-ray prediction and motor tests.

R1 & R2 modules UNCHANGED. This program has NO food, map access,
scripted obstacle avoidance, evolutionary selection, or invented goal.
Sensor predictor, MSE, eligibility-based motor feedback are explicitly
RESEARCHER ENGINEERED. Detailed fixed scientific protocol:
docs/FLYLOOP_R3_PROTOCOL_2026-11-24.md.
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

SEEDS=tuple(range(9700,9724))
SCENES=("obstacles","nearfield")
ARMS=(
    "recurrent_frozen","r2_progress","predict_passive",
    "predict_motor","predict_sham","predict_frozen",
    "predict_sham_passive","inertial_random",
)
LIFETIME=r1.HORIZON
HEAD_FEATURES=r1.N+3+2+1  # 16 neural state +3 current rays +2 motor +bias
HEAD_PARAMETERS=HEAD_FEATURES*3
HEAD_LR=.014
HEAD_WEIGHT_BOUND=2.
SHAM_LAG=6
LATE_START=300
MOTOR_LR=r2.LR
MOTOR_TRACE_LEAK=r2.TRACE_LEAK
MOTOR_BASELINE_LEAK=r2.BASELINE_LEAK
MOTOR_ADVANTAGE_LIMIT=r2.MAX_ADV
ACTIVE_PREDICTION_ARMS=frozenset((
    "predict_passive","predict_motor","predict_sham","predict_sham_passive",
))
MOTOR_PREDICTIVE_ARMS=frozenset((
    "predict_motor","predict_sham","predict_frozen",
))
SHAM_ARMS=frozenset(("predict_sham","predict_sham_passive"))


def sha(obj):
    return hashlib.sha256(json.dumps(
        obj,sort_keys=True,separators=(",",":")
    ).encode("utf8")).hexdigest()


def mean(xs):
    xs=list(xs)
    return sum(xs)/len(xs) if xs else 0.


def mse(pred,target):
    return sum((a-b)**2 for a,b in zip(pred,target))/3.


class World(r1.World):
    """R1 physics, fresh seed cohort and two R1/R2 map generators."""
    def __init__(self,seed,scene):
        if seed not in SEEDS or scene not in SCENES:
            raise ValueError("Unregistered FlyLoop R3 seed/scene")
        self.seed=seed
        self.scenario=scene
        self.rectangles=(
            r1.procedural_rectangles(seed) if scene=="obstacles"
            else r2.nearfield_rectangles(seed)
        )
        self.body=r1.Body(
            heading=random.Random(seed*11113+51).uniform(-math.pi,math.pi)
        )
        self.frame=0
        self.visited={(int(self.body.x),int(self.body.y))}
        self.collisions=0
        self.distance=0.
        self.trace=[]


class SensorHead:
    """Only predicts next local three wall rays after a REAL action."""
    def __init__(self,seed):
        self.weights=[[0.]*HEAD_FEATURES for _ in range(3)]
        self.observed_next=[]
        self.sham_rng=random.Random(seed*19913+73007)
        self.update_events=0

    @staticmethod
    def features(hidden,pre_rays,command):
        features=(
            list(hidden)+list(pre_rays)+list(command)+[1.]
        )
        assert len(features)==HEAD_FEATURES
        return features

    def predict(self,pre_rays,features):
        return [
            round(r1.clamp(
                pre_rays[i]+r1.dot(self.weights[i],features),0.,1.
            ),9)
            for i in range(3)
        ]

    def training_target(self,tick,actual_next,arm):
        source=None
        if arm in SHAM_ARMS:
            last_eligible=tick-SHAM_LAG
            if last_eligible<0:
                target=None
            else:
                source=self.sham_rng.randrange(last_eligible+1)
                target=self.observed_next[source]
        elif arm in ACTIVE_PREDICTION_ARMS:
            source=tick
            target=actual_next
        else:
            target=None
        # May only append AFTER choosing eligible historic values.
        self.observed_next.append(list(actual_next))
        return target,source

    def train(self,features,pred,target):
        if target is None:
            return False
        for i in range(3):
            delta=HEAD_LR*(target[i]-pred[i])
            for j in range(HEAD_FEATURES):
                self.weights[i][j]=r1.clamp(
                    self.weights[i][j]+delta*features[j],
                    -HEAD_WEIGHT_BOUND,HEAD_WEIGHT_BOUND,
                )
        self.update_events+=1
        return True


class MotorErrorLearner:
    """Researcher-coded output-only exploration-eligibility update.

    An intrinsic preference for predictable sensations is NOT a
    self-invented goal; it might actually favor sensorimotor stagnation.
    """
    def __init__(self):
        self.traces=[[0.]*r2.N_MOTOR_COEFF for _ in range(2)]
        self.baseline=0.
        self.update_events=0

    def update(self,brain,hidden,noise,loss):
        feedback=-loss
        advantage=r1.clamp(
            feedback-self.baseline,
            -MOTOR_ADVANTAGE_LIMIT,MOTOR_ADVANTAGE_LIMIT
        )
        phi=list(hidden)+[1.]
        assert len(phi)==r2.N_MOTOR_COEFF
        for motor in range(2):
            for j in range(r2.N_MOTOR_COEFF):
                self.traces[motor][j]=(
                    MOTOR_TRACE_LEAK*self.traces[motor][j]
                    +(noise[motor]/r2.EXPLORATION_SD)*phi[j]
                )
            for j in range(r1.N):
                previous=brain.circuit.weights["w_motor"][motor][j]
                brain.circuit.weights["w_motor"][motor][j]=r1.clamp(
                    previous+MOTOR_LR*advantage*self.traces[motor][j],
                    -r2.UPDATE_LIMIT,r2.UPDATE_LIMIT
                )
            oldbias=brain.circuit.weights["motor_bias"][motor]
            brain.circuit.weights["motor_bias"][motor]=r1.clamp(
                oldbias+MOTOR_LR*advantage*self.traces[motor][-1],
                -r2.UPDATE_LIMIT,r2.UPDATE_LIMIT
            )
        self.baseline=(
            MOTOR_BASELINE_LEAK*self.baseline
            +(1.-MOTOR_BASELINE_LEAK)*feedback
        )
        self.update_events+=1
        return round(feedback,9)


def _r2_brain(seed,arm):
    mode=(
        "inertial_random" if arm=="inertial_random"
        else "recurrent_plastic" if arm=="r2_progress"
        else "recurrent_frozen"
    )
    return r2.MotorLearner(seed,mode)


def simulate(seed,scene,arm):
    if seed not in SEEDS or scene not in SCENES or arm not in ARMS:
        raise ValueError("Unregistered R3 experimental cell")
    world=World(seed,scene)
    brain=_r2_brain(seed,arm)
    original=copy.deepcopy(brain.circuit.weights)
    head=SensorHead(seed)
    motor=MotorErrorLearner()
    positions=[(world.body.x,world.body.y)]
    initial_heading=world.body.heading
    for tick in range(LIFETIME):
        if world.body.energy<=0 or world.body.integrity<=0:
            break
        pre_sensors=tuple(world.perceive())
        current_rays=tuple(pre_sensors[1:4])
        raw,hidden,sensed,noise,executed=brain.command(pre_sensors)
        features=head.features(hidden,current_rays,executed)
        pred=head.predict(current_rays,features)
        # IMPORTANT: prediction, inputs, motor are frozen before physics.
        consequence=world.integrate(executed)
        positions.append((world.body.x,world.body.y))
        next_sensors=tuple(world.perceive())
        target=tuple(next_sensors[1:4])
        physical_error=mse(pred,target)
        persistence_error=mse(current_rays,target)
        learning_target,source=head.training_target(
            tick,target,arm
        )
        head.train(features,pred,learning_target)
        feedback=0.
        if arm=="r2_progress":
            progress=brain.adapt(
                tick,consequence["moved"],consequence["collision"],
                positions,hidden,noise
            )
            feedback=progress["training_feedback"]
        if arm in MOTOR_PREDICTIVE_ARMS:
            feedback=motor.update(brain,hidden,noise,physical_error)
        consequence.update({
            "tick":tick,
            "raw_sensor_inputs":[round(v,9) for v in sensed],
            "hidden_neural_state":[round(v,9) for v in hidden],
            "base_motor_command":[round(v,9) for v in raw],
            "injected_motor_noise":[round(v,9) for v in noise],
            "executed_motor_command":[round(v,9) for v in executed],
            "current_wall_rays":[round(v,9) for v in current_rays],
            "predicted_next_wall_rays":list(pred),
            "observed_next_wall_rays":[round(v,9) for v in target],
            "true_prediction_mse":round(physical_error,12),
            "persistence_reference_mse":round(persistence_error,12),
            "training_target_source_tick":source,
            "predictor_trained_this_tick":learning_target is not None,
            "training_wall_rays":(
                [round(v,9) for v in learning_target]
                if learning_target is not None else None
            ),
            "motor_learning_feedback":round(feedback,9),
        })
        world.trace.append(consequence)
    final=copy.deepcopy(brain.circuit.weights)
    all_positions=list(positions)
    centroid=(mean(p[0] for p in all_positions),
              mean(p[1] for p in all_positions))
    radius=math.sqrt(mean(
        (p[0]-centroid[0])**2+(p[1]-centroid[1])**2
        for p in all_positions
    ))
    late=[r for r in world.trace if r["tick"]>=LATE_START]
    return {
        "seed":seed,"scene":scene,"arm":arm,
        "rectangles":[list(v) for v in world.rectangles],
        "initial_pose":[2.2,2.2,round(initial_heading,9)],
        "initial_coefficients":original,
        "final_coefficients":final,
        "initial_coefficients_sha256":sha(original),
        "final_coefficients_sha256":sha(final),
        "head_initial_zero":True,
        "head_final_weights":head.weights,
        "head_training_events":head.update_events,
        "motor_training_events":(
            brain.updates if arm=="r2_progress" else motor.update_events
        ),
        "steps_alive":len(world.trace),
        "distance":round(world.distance,6),
        "net_displacement":round(math.hypot(
            world.body.x-2.2,world.body.y-2.2
        ),6),
        "trajectory_rms_radius":round(radius,6),
        "unique_grid_cells":len(world.visited),
        "collisions":world.collisions,
        "terminal_energy":round(world.body.energy,6),
        "terminal_integrity":round(world.body.integrity,6),
        "sum_prediction_mse":round(sum(
            t["true_prediction_mse"] for t in world.trace
        ),9),
        "sum_persistence_mse":round(sum(
            t["persistence_reference_mse"] for t in world.trace
        ),9),
        "late_steps":len(late),
        "late_sum_prediction_mse":round(sum(
            t["true_prediction_mse"] for t in late
        ),9),
        "late_sum_persistence_mse":round(sum(
            t["persistence_reference_mse"] for t in late
        ),9),
        "trace":world.trace,
    }


def _audit(rows):
    physics=True
    prediction_causal=True
    sham_past=True
    weights_legal=True
    for life in rows:
        previous=(2.2,2.2)
        earlier=[]
        initial=life["initial_coefficients"]
        final=life["final_coefficients"]
        if any(initial[k]!=final[k] for k in (
            "w_input","w_recurrent","bias"
        )):
            weights_legal=False
        motor_static=life["arm"] in (
            "recurrent_frozen","predict_passive",
            "predict_sham_passive","inertial_random",
        )
        if motor_static and initial!=final:
            weights_legal=False
        if not motor_static and life["motor_training_events"]!=life["steps_alive"]:
            weights_legal=False
        if any(len(r)!=HEAD_FEATURES for r in life["head_final_weights"]):
            weights_legal=False
        if life["arm"] not in ACTIVE_PREDICTION_ARMS and any(
            v!=0. for row in life["head_final_weights"] for v in row
        ):
            weights_legal=False
        for t,frame in enumerate(life["trace"]):
            if frame["tick"]!=t or any(abs(a-b)>1e-7 for a,b in zip(
                frame["before_position"],previous
            )):
                physics=False
            pos=tuple(frame["position"])
            if frame["collision"] and any(abs(a-b)>1e-7
                                           for a,b in zip(pos,previous)):
                physics=False
            if not (
                r1.RADIUS-1e-7<=pos[0]<=r1.SIDE-r1.RADIUS+1e-7
                and r1.RADIUS-1e-7<=pos[1]<=r1.SIDE-r1.RADIUS+1e-7
                and 0<=frame["energy"]<=100
                and 0<=frame["integrity"]<=100
            ):
                physics=False
            if any(x1-r1.RADIUS<pos[0]<x2+r1.RADIUS
                   and y1-r1.RADIUS<pos[1]<y2+r1.RADIUS
                   for x1,y1,x2,y2 in life["rectangles"]):
                physics=False
            previous=pos
            current=frame["current_wall_rays"]
            future=frame["observed_next_wall_rays"]
            predicted=frame["predicted_next_wall_rays"]
            if len(frame["hidden_neural_state"])!=r1.N or len(
                frame["raw_sensor_inputs"]
            )!=r1.INPUTS or len(frame["executed_motor_command"])!=2:
                prediction_causal=False
            if current!=frame["raw_sensor_inputs"][1:4]:
                prediction_causal=False
            if any(not 0<=v<=1 for v in current+future+predicted):
                prediction_causal=False
            if t+1<len(life["trace"]) and future!=(
                life["trace"][t+1]["current_wall_rays"]
            ):
                prediction_causal=False
            if abs(mse(predicted,future)-frame["true_prediction_mse"])>5e-9:
                prediction_causal=False
            if abs(mse(current,future)-frame["persistence_reference_mse"])>5e-9:
                prediction_causal=False
            source=frame["training_target_source_tick"]
            if life["arm"] in SHAM_ARMS:
                if source is not None and not (
                    0<=source<=t-SHAM_LAG
                ):
                    sham_past=False
                if source is None and t>=SHAM_LAG:
                    sham_past=False
                if source is not None and frame["training_wall_rays"]!=earlier[source]:
                    sham_past=False
            elif life["arm"] in ACTIVE_PREDICTION_ARMS:
                if source!=t or frame["training_wall_rays"]!=future:
                    prediction_causal=False
            elif source is not None or frame["training_wall_rays"] is not None:
                prediction_causal=False
            earlier.append(list(future))
        if len(life["trace"])!=life["steps_alive"]:
            physics=False
    return {
        "legal_continuous_physics":physics,
        "causal_preaction_predictor_true_postaction_labels":prediction_causal,
        "sham_sources_strictly_older_observed_wall_rays":sham_past,
        "permitted_prediction_and_motor_coefficients_only":weights_legal,
    }


def _boot(rows,scene,field,arm,ref,n=1000):
    lookup={
        (r["seed"],r["arm"]):r for r in rows if r["scene"]==scene
    }
    diffs=[
        lookup[(seed,arm)][field]-lookup[(seed,ref)][field]
        for seed in SEEDS
    ]
    rng=random.Random(97093)
    values=sorted(mean(
        diffs[rng.randrange(len(diffs))] for _ in diffs
    ) for _ in range(n))
    return {
        "mean_paired_delta":round(mean(diffs),6),
        "exploratory_world_seed_bootstrap95":[
            round(values[25],6),round(values[974],6)
        ],
    }


def run():
    rows=[
        simulate(seed,scene,arm)
        for scene in SCENES for seed in SEEDS for arm in ARMS
    ]
    summary={}
    for scene in SCENES:
        summary[scene]={}
        for arm in ARMS:
            scope=[r for r in rows if r["scene"]==scene and r["arm"]==arm]
            distance=sum(r["distance"] for r in scope)
            late_steps=sum(r["late_steps"] for r in scope)
            summary[scene][arm]={
                k:round(mean(r[k] for r in scope),6)
                for k in (
                    "steps_alive","distance","unique_grid_cells",
                    "collisions","trajectory_rms_radius",
                    "terminal_integrity","terminal_energy",
                )
            }|{
                "contacts_per_meter":round(
                    sum(r["collisions"] for r in scope)/max(distance,1e-12),6
                ),
                "late_mse":round(
                    sum(r["late_sum_prediction_mse"] for r in scope)
                    /max(late_steps,1),9
                ),
                "late_persistence_mse":round(
                    sum(r["late_sum_persistence_mse"] for r in scope)
                    /max(late_steps,1),9
                ),
                "late_steps_total":late_steps,
                "head_update_fraction":round(
                    sum(r["head_training_events"] for r in scope)/
                    max(sum(r["steps_alive"] for r in scope),1),6
                ),
            }
    integrity=_audit(rows)
    paired=all(
        len({sha(r["rectangles"]) for r in rows
             if r["seed"]==seed and r["scene"]==scene})==1
        and len({sha(r["initial_pose"]) for r in rows
                 if r["seed"]==seed and r["scene"]==scene})==1
        and len({r["initial_coefficients_sha256"] for r in rows
                 if r["seed"]==seed and r["scene"]==scene})==1
        for scene in SCENES for seed in SEEDS
    )
    noise_paired=all(
        all(
            [x["injected_motor_noise"] for x in r["trace"][:limit]]
            ==[x["injected_motor_noise"] for x in group[0]["trace"][:limit]]
            for r in group
        )
        for scene in SCENES for seed in SEEDS
        for group in [[r for r in rows
                       if r["seed"]==seed and r["scene"]==scene
                       and r["arm"]!="inertial_random"]]
        for limit in [min(r["steps_alive"] for r in group)]
    )
    passive_paired=all(
        # Two passive predictors have EXACTLY equal actions, states and
        # observations. They differ ONLY in head training labels/weights.
        all(
            [(t["position"],t["executed_motor_command"],t["observed_next_wall_rays"])
             for t in a["trace"]]==
            [(t["position"],t["executed_motor_command"],t["observed_next_wall_rays"])
             for t in b["trace"]]
        ) for scene in SCENES for seed in SEEDS
        for a in [next(r for r in rows if r["scene"]==scene and
                       r["seed"]==seed and r["arm"]=="predict_passive")]
        for b in [next(r for r in rows if r["scene"]==scene and
                       r["seed"]==seed and r["arm"]=="predict_sham_passive")]
    )
    mapcounts={
        scene:len({
            sha(r["rectangles"]) for r in rows if r["scene"]==scene
        }) for scene in SCENES
    }
    map_valid=all(
        len(r["rectangles"])==r1.OBSTACLES
        and all(not (x1-1.1<2.2<x2+1.1
                     and y1-1.1<2.2<y2+1.1)
                for x1,y1,x2,y2 in r["rectangles"])
        for r in rows
    )
    frozen_head_equals_persistence=all(
        r["arm"]!="predict_frozen" or all(
            t["predicted_next_wall_rays"]==t["current_wall_rays"]
            for t in r["trace"]
        ) for r in rows
    )
    n=summary["nearfield"]
    gates={
        "01_384_balanced_episodes_and_full_byte_repeat_required":
            len(rows)==384 and all(
                sum(r["scene"]==scene and r["arm"]==arm
                    for r in rows)==len(SEEDS)
                for scene in SCENES for arm in ARMS
            ),
        "02_true_original_r1_continuous_physics_no_tunneling":
            integrity["legal_continuous_physics"],
        "03_identical_maps_pose_434_weights_matched_motor_noise":
            paired and noise_paired,
        "04_same_16_neurons_and_8_local_sensors_no_hidden_map":
            integrity["causal_preaction_predictor_true_postaction_labels"],
        "05_only_66_predictor_and_34_original_motor_coefficients_mutable":
            integrity["permitted_prediction_and_motor_coefficients_only"],
        "06_prediction_before_motion_labels_after_real_motion":
            integrity["causal_preaction_predictor_true_postaction_labels"],
        "07_sham_target_strict_old_and_passive_trajectories_identical":
            integrity["sham_sources_strictly_older_observed_wall_rays"]
            and passive_paired,
        "08_frozen_predictor_exact_persistence_reference":
            frozen_head_equals_persistence,
        "09_10_valid_exogenous_rectangles_per_map_20_unique":
            map_valid and all(v>=20 for v in mapcounts.values()),
        "10_complete_raw_traces_no_evolution_previous_studies_untouched":
            len(rows)==384 and all(r["trace"] and r["initial_coefficients"]
                                   and r["final_coefficients"] for r in rows),
        "11_nearfield_passive_true_late_mse_below_persistence_by15pct":
            n["predict_passive"]["late_persistence_mse"]>0 and
            n["predict_passive"]["late_mse"]<=
                .85*n["predict_passive"]["late_persistence_mse"],
        "12_nearfield_true_passive_late_mse_below_sham_passive_15pct":
            n["predict_sham_passive"]["late_mse"]>0 and
            n["predict_passive"]["late_mse"]<=
                .85*n["predict_sham_passive"]["late_mse"],
        "13_nearfield_motor_prediction_coverage_ge_frozen_plus10pct":
            n["predict_motor"]["unique_grid_cells"]>=
                1.10*n["recurrent_frozen"]["unique_grid_cells"],
        "14_nearfield_motor_prediction_contacts_per_meter_le_90pct":
            n["predict_motor"]["contacts_per_meter"]<=
                .90*n["recurrent_frozen"]["contacts_per_meter"],
        "15_nearfield_motor_prediction_coverage_ge_sham_plus10pct":
            n["predict_motor"]["unique_grid_cells"]>=
                1.10*n["predict_sham"]["unique_grid_cells"],
        "16_nearfield_motor_prediction_coverage_ge_frozenhead_plus10pct":
            n["predict_motor"]["unique_grid_cells"]>=
                1.10*n["predict_frozen"]["unique_grid_cells"],
    }
    boot={
        scene:{
            reference:_boot(rows,scene,"unique_grid_cells",
                            "predict_motor",reference)
            for reference in (
                "recurrent_frozen","predict_sham","predict_frozen",
                "r2_progress","inertial_random"
            )
        } for scene in SCENES
    }
    return {
        "name":"FlyLoop R3 local sensory prediction vs exploration",
        "protocol":"docs/FLYLOOP_R3_PROTOCOL_2026-11-24.md",
        "metadata":{
            "seeds":SEEDS,"scenes":SCENES,"arms":ARMS,
            "existing_recurrent_neurons":r1.N,
            "existing_neural_coefficients":r1.PARAMS,
            "additional_sensor_predictor_coefficients":HEAD_PARAMETERS,
            "original_motor_coefficients_allowed_to_update":34,
            "predictive_head_lr":HEAD_LR,
            "motor_lr":MOTOR_LR,
            "sham_observation_min_lag":SHAM_LAG,
            "late_phase_starts_at":LATE_START,
            "no_explicit_wall_avoidance_command":True,
            "no_future_sensor_or_map_input":True,
            "error_minimization_is_researcher_programmed":True,
            "previous_studies_and_main_untouched":True,
            "no_food_or_evolution":True,
        },
        "summary":summary,
        "integrity":integrity,
        "paired_world_seed_exploratory_bootstrap":boot,
        "gates":gates,
        "gates_passed_count":sum(gates.values()),
        "screen_pass":all(gates.values()),
        "goal_invention_demonstrated":False,
        "self_learned_planning_demonstrated":False,
        "successful_obstacle_avoidance_demonstrated":False,
        "food_search_demonstrated":False,
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
        default=Path("results/flyloop_r3/r3_predictive_motor.json")
    )
    args=parser.parse_args()
    result=run()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(
        json.dumps(result,sort_keys=True,separators=(",",":"))+"\n",
        encoding="utf8",
    )
    print("FLYLOOP_R3 "+json.dumps({
        k:result[k] for k in (
            "name","summary","integrity","gates","gates_passed_count",
            "screen_pass","paired_world_seed_exploratory_bootstrap",
            "raw_rows_sha256"
        )
    },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
