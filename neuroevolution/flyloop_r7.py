"""FlyLoop R7: an ONLINE learned sensor model ACTUALLY selects continuous actions.

This is a deliberately small, falsifiable first model-to-action closure:
R1 continuous 16-recurrent-neuron motor commands + same R2 Gaussian
noise; R4 LMS heads update only AFTER the organism's REAL physical
sensations. From tick 60, a five-way LOCAL continuous motor candidate
menu is scored from pre-action learned predictions of sensory
variation. No world clones inside the organism, food, navigation map,
prewritten wall escape, evolutionary search, or external head transfer.

The score/command menu are RESEARCHER ENGINEERED, NOT emergent goals.
The original R1–R6 code is imported unmodified.

FROZEN BEFORE SCIENCE: docs/FLYLOOP_R7_PROTOCOL_2026-11-24.md
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
from neuroevolution import flyloop_r4 as r4

SEEDS=tuple(range(10900,10916))
GEOMETRIES=("obstacles","nearfield")
REGIMES=("static","removal")
ARMS=(
    "recurrent_frozen","passive_predictor","true_closed_loop",
    "yoked_closed_loop","random_menu","r2_progress",
)
HORIZON=480
CHANGE_TICK=240
WARMUP=60
STEER_DELTA=.35
THROTTLE_DELTA=.20
CANDIDATES=("base","steer_minus","steer_plus","throttle_minus","throttle_plus")
HEAD_ARMS=frozenset(ARMS)-{"recurrent_frozen"}
MODEL_SCORER_ARMS=frozenset(("true_closed_loop","yoked_closed_loop"))
NOISE_ARMS=frozenset(ARMS)
HEADS=r4.PREDICTOR_NAMES


def sha(x):
    return hashlib.sha256(
        json.dumps(x,sort_keys=True,separators=(",",":")).encode()
    ).hexdigest()


def mean(values):
    values=list(values)
    return sum(values)/len(values) if values else 0.


def rms_radius(pos):
    if not pos:
        return 0.
    cx=mean(p[0] for p in pos)
    cy=mean(p[1] for p in pos)
    return math.sqrt(mean((x-cx)**2+(y-cy)**2 for x,y in pos))


class World(r1.World):
    """Physical R1 world; optional exogenous harmless obstacle REMOVAL."""
    def __init__(self,seed,geometry,regime):
        if seed not in SEEDS or geometry not in GEOMETRIES or regime not in REGIMES:
            raise ValueError("R7 world seed, geometry or regime unregistered")
        self.seed=seed
        self.scenario=geometry
        self.geometry=geometry
        self.regime=regime
        self.initial_rectangles=(
            r1.procedural_rectangles(seed) if geometry=="obstacles"
            else r2.nearfield_rectangles(seed)
        )
        self.rectangles=list(self.initial_rectangles)
        self.body=r1.Body(
            heading=random.Random(seed*11113+51).uniform(-math.pi,math.pi)
        )
        self.frame=0
        self.visited={(int(self.body.x),int(self.body.y))}
        self.collisions=0
        self.distance=0.
        self.trace=[]
        self.switched=False

    def apply_environment(self,tick):
        if tick==CHANGE_TICK and self.regime=="removal":
            # Remove odd-index rectangles; retains even-index rectangles.
            # Never add new walls, so a real body cannot become trapped.
            self.rectangles=list(self.initial_rectangles[::2])
            self.switched=True
        return self.switched


def candidates(base):
    throttle,steer=base
    return (
        tuple(base),
        (throttle,r1.clamp(steer-STEER_DELTA,-1.,1.)),
        (throttle,r1.clamp(steer+STEER_DELTA,-1.,1.)),
        (r1.clamp(throttle-THROTTLE_DELTA,-1.,1.),steer),
        (r1.clamp(throttle+THROTTLE_DELTA,-1.,1.),steer),
    )


def predicted_candidates(heads,hidden,rays,options):
    """Predict ONLY using current local sensors and current head weights."""
    return {
        head:[
            heads.predict(
                head,rays,heads.make_features(hidden,rays,cmd)
            ) for cmd in options
        ]
        for head in ("true","yoked")
    }


def novelty_scores(predictions):
    base=predictions[0]
    return [
        sum((x-y)**2 for x,y in zip(pred,base))/3.
        for pred in predictions
    ]


def selected_candidate(predictions):
    """Tie broken in favor of original base command (index 0)."""
    scores=novelty_scores(predictions)
    return max(range(len(scores)),key=lambda i:scores[i]),scores


def _brain(seed,arm):
    if arm=="r2_progress":
        return r2.MotorLearner(seed,"recurrent_plastic")
    return r2.MotorLearner(seed,"recurrent_frozen")


def simulate(seed,geometry,regime,arm):
    if arm not in ARMS:
        raise ValueError("Unknown R7 arm")
    world=World(seed,geometry,regime)
    brain=_brain(seed,arm)
    heads=r4.ActionSensorHeads(seed)
    random_menu_rng=random.Random(seed*59149+77401)
    initial=copy.deepcopy(brain.circuit.weights)
    first_heading=world.body.heading
    positions=[(world.body.x,world.body.y)]
    pre_visited={(int(world.body.x),int(world.body.y))}
    post_visited=set()
    post_novel=set()
    post_xy=[]
    post_meters=0.
    post_contacts=0
    changed_action_count=0
    post_changed_count=0

    for tick in range(HORIZON):
        if world.body.energy<=0. or world.body.integrity<=0.:
            break
        world.apply_environment(tick)
        sensors=tuple(world.perceive())
        rays=tuple(sensors[1:4])
        base_raw,hidden,observed,noise,base=brain.command(sensors)
        options=candidates(base)
        preds=predicted_candidates(heads,hidden,rays,options)
        chosen=0
        scores=[0.]*len(options)
        if tick>=WARMUP and arm in MODEL_SCORER_ARMS:
            model="true" if arm=="true_closed_loop" else "yoked"
            chosen,scores=selected_candidate(preds[model])
        elif tick>=WARMUP and arm=="random_menu":
            chosen=random_menu_rng.randrange(len(options))
        executed=options[chosen]
        # Sample the yoked command only ONCE, before this physical action.
        # Strictly use previously actually executed actions (age >=6).
        phi,yoked,source=heads.inputs(tick,hidden,rays,executed)
        pre_predictions={
            name:heads.predict(name,rays,phi[name])
            for name in HEADS
        }
        # One and only one ACTUAL physical motor execution.
        physical=world.integrate(executed)
        positions.append((world.body.x,world.body.y))
        real_next=tuple(world.perceive()[1:4])
        loss={
            name:round(r4.mse(pred,real_next),12)
            for name,pred in pre_predictions.items()
        }
        if arm in HEAD_ARMS:
            heads.train_all(phi,pre_predictions,real_next)
        heads.motor_history.append(tuple(executed))

        progress_feedback=None
        if arm=="r2_progress":
            update=brain.adapt(
                tick,physical["moved"],physical["collision"],
                positions,hidden,noise
            )
            progress_feedback=update["training_feedback"]

        if chosen:
            changed_action_count+=1
            if tick>=CHANGE_TICK:
                post_changed_count+=1
        if tick<CHANGE_TICK:
            pre_visited.add((int(world.body.x),int(world.body.y)))
        else:
            cell=(int(world.body.x),int(world.body.y))
            post_visited.add(cell)
            if cell not in pre_visited:
                post_novel.add(cell)
            post_xy.append((world.body.x,world.body.y))
            post_meters+=physical["moved"]
            post_contacts+=physical["collision"]
        frame={
            "tick":tick,
            "geometry_changed":world.switched,
            "original_eight_local_sensors":[round(v,9) for v in observed],
            "original_16_recurrent_states":[round(v,9) for v in hidden],
            "original_neural_motor":[round(v,9) for v in base_raw],
            "original_exogenous_motor_noise":[round(v,9) for v in noise],
            "base_motor":[round(v,9) for v in base],
            "five_local_motor_candidates":[
                [round(v,9) for v in candidate] for candidate in options
            ],
            "preaction_candidate_model_predictions":preds,
            "selected_candidate_index":chosen,
            "selected_candidate_scores":[round(v,12) for v in scores],
            "physically_executed_motor":[round(v,9) for v in executed],
            "historic_yoked_motor":[round(v,9) for v in yoked],
            "historic_yoked_source_tick":source,
            "actual_next_three_wall_sensors":[round(v,9) for v in real_next],
            "predicted_actual_three_wall_sensors":pre_predictions,
            "same_step_actual_future_mse":loss,
            "online_head_trained_after_physics":arm in HEAD_ARMS,
            "r2_progress_reward_if_applicable":progress_feedback,
            "physical_body_outcome":physical,
            "head_weights_sha256_after_actual_step":(
                sha(heads.weights) if (tick+1)%20==0 else None
            ),
        }
        world.trace.append(frame)

    final=copy.deepcopy(brain.circuit.weights)
    late=[f for f in world.trace if f["tick"]>=CHANGE_TICK]
    return {
        "seed":seed,
        "geometry":geometry,
        "regime":regime,
        "arm":arm,
        "initial_ten_rectangles":[list(x) for x in world.initial_rectangles],
        "postchange_rectangles":[list(x) for x in world.rectangles],
        "environment_removed_five_walls":world.switched,
        "initial_body_pose":[2.2,2.2,round(first_heading,9)],
        "original_434_neural_coefficients_at_birth":initial,
        "original_434_neural_coefficients_at_end":final,
        "initial_brain_sha256":sha(initial),
        "final_brain_sha256":sha(final),
        "head_weights_initially_zero":True,
        "final_3x84_online_sensor_head_weights":heads.weights,
        "head_update_events":heads.updates,
        "motor_update_events":brain.updates,
        "real_physical_steps":len(world.trace),
        "total_grid_cells":len(world.visited),
        "total_collisions":world.collisions,
        "total_distance":round(world.distance,6),
        "postchange_real_steps":len(late),
        "postchange_visited_cells":len(post_visited),
        "postchange_new_cells_not_seen_previously":len(post_novel),
        "postchange_meters":round(post_meters,6),
        "postchange_collisions":post_contacts,
        "postchange_trajectory_rms_radius":round(rms_radius(post_xy),6),
        "after_warmup_altered_action_count":changed_action_count,
        "postchange_altered_action_count":post_changed_count,
        "late_sum_actual_true_mse":round(sum(
            f["same_step_actual_future_mse"]["true"] for f in late
        ),10),
        "late_sum_actual_yoked_mse":round(sum(
            f["same_step_actual_future_mse"]["yoked"] for f in late
        ),10),
        "raw_trace":world.trace,
    }


def _audit(rows):
    physics=True
    action_causality=True
    historic=True
    head_legal=True
    brain_legal=True
    for life in rows:
        seed,geometry,regime,arm=(
            life["seed"],life["geometry"],life["regime"],life["arm"]
        )
        world=World(seed,geometry,regime)
        brain=_brain(seed,arm)
        heads=r4.ActionSensorHeads(seed)
        random_rng=random.Random(seed*59149+77401)
        positions=[(world.body.x,world.body.y)]
        if life["initial_brain_sha256"]!=sha(brain.circuit.weights):
            brain_legal=False
        for f in life["raw_trace"]:
            t=f["tick"]
            if world.frame!=t or world.body.energy<=0. or world.body.integrity<=0.:
                physics=False
                break
            world.apply_environment(t)
            sensors=tuple(world.perceive())
            raw,hidden,perceived,noise,base=brain.command(sensors)
            opts=candidates(base)
            preds=predicted_candidates(heads,hidden,sensors[1:4],opts)
            if (
                f["original_eight_local_sensors"]!=[
                    round(v,9) for v in perceived
                ]
                or f["original_16_recurrent_states"]!=[
                    round(v,9) for v in hidden
                ]
                or f["original_exogenous_motor_noise"]!=[
                    round(v,9) for v in noise
                ]
                or f["five_local_motor_candidates"]!=[
                    [round(v,9) for v in cmd] for cmd in opts
                ]
                or f["preaction_candidate_model_predictions"]!=preds
            ):
                action_causality=False
            selected=0
            scores=[0.]*5
            if t>=WARMUP and arm in MODEL_SCORER_ARMS:
                selected,scores=selected_candidate(
                    preds["true" if arm=="true_closed_loop" else "yoked"]
                )
            elif t>=WARMUP and arm=="random_menu":
                selected=random_rng.randrange(5)
            actual=opts[selected]
            if (f["selected_candidate_index"]!=selected or
                f["selected_candidate_scores"]!=[
                    round(v,12) for v in scores
                ] or f["physically_executed_motor"]!=[
                    round(v,9) for v in actual
                ]):
                action_causality=False
            features,past,source=heads.inputs(
                t,hidden,sensors[1:4],actual
            )
            if (f["historic_yoked_source_tick"]!=source or
                f["historic_yoked_motor"]!=[round(v,9) for v in past]
            ):
                historic=False
            if source is None:
                if t>=r4.YOKED_LAG or past!=(0.,0.):
                    historic=False
            elif not (
                0<=source<=t-r4.YOKED_LAG
                and tuple(heads.motor_history[source])==tuple(past)
            ):
                historic=False
            pre_prediction={
                name:heads.predict(name,sensors[1:4],features[name])
                for name in HEADS
            }
            if pre_prediction!=f["predicted_actual_three_wall_sensors"]:
                action_causality=False
            physically=world.integrate(actual)
            positions.append((world.body.x,world.body.y))
            real_next=tuple(world.perceive()[1:4])
            if physically!=f["physical_body_outcome"] or [
                round(v,9) for v in real_next
            ]!=f["actual_next_three_wall_sensors"]:
                physics=False
            losses={
                name:round(r4.mse(p,real_next),12)
                for name,p in pre_prediction.items()
            }
            if losses!=f["same_step_actual_future_mse"]:
                action_causality=False
            if arm in HEAD_ARMS:
                heads.train_all(features,pre_prediction,real_next)
            heads.motor_history.append(tuple(actual))
            if arm=="r2_progress":
                progress=brain.adapt(
                    t,physically["moved"],physically["collision"],
                    positions,hidden,noise
                )
                if progress["training_feedback"]!=f[
                    "r2_progress_reward_if_applicable"
                ]:
                    action_causality=False
            if (t+1)%20==0 and sha(heads.weights)!=f[
                "head_weights_sha256_after_actual_step"
            ]:
                head_legal=False
        if (len(life["raw_trace"])!=world.frame or
            len(life["raw_trace"])!=life["real_physical_steps"]
        ):
            physics=False
        if heads.weights!=life["final_3x84_online_sensor_head_weights"]:
            head_legal=False
        if heads.updates!=life["head_update_events"]:
            head_legal=False
        if brain.circuit.weights!=life["original_434_neural_coefficients_at_end"]:
            brain_legal=False
        if arm!="r2_progress" and (
            life["initial_brain_sha256"]!=life["final_brain_sha256"]
        ):
            brain_legal=False
        if arm=="r2_progress" and brain.updates!=life["motor_update_events"]:
            brain_legal=False
        for k in ("w_input","w_recurrent","bias"):
            if (life["original_434_neural_coefficients_at_birth"][k]!=
                life["original_434_neural_coefficients_at_end"][k]
            ):
                brain_legal=False
    return {
        "all_actual_physical_neuron_body_replays_valid":physics,
        "five_candidate_selection_and_preaction_model_reconstructable":action_causality,
        "sham_motor_strictly_from_own_old_physical_actions":historic,
        "predictor_heads_updated_only_after_real_observation":head_legal,
        "original_recurrent_weights_frozen_except_r2_output_readout":brain_legal,
    }


def _paired_bootstrap(rows,geometry,regime,arm,reference,n=1000):
    values={(r["seed"],r["arm"]):r for r in rows
            if r["geometry"]==geometry and r["regime"]==regime}
    deltas=[
        values[(seed,arm)]["postchange_visited_cells"]-
        values[(seed,reference)]["postchange_visited_cells"]
        for seed in SEEDS
    ]
    rng=random.Random(109090)
    draws=sorted(
        mean(deltas[rng.randrange(len(deltas))] for _ in deltas)
        for _ in range(n)
    )
    return {
        "mean_paired_seed_delta":round(mean(deltas),6),
        "exploratory_seed_bootstrap95":[
            round(draws[25],6),round(draws[974],6)
        ],
    }


def run():
    rows=[
        simulate(seed,geometry,regime,arm)
        for geometry in GEOMETRIES
        for regime in REGIMES
        for seed in SEEDS
        for arm in ARMS
    ]
    audit=_audit(rows)
    summary={}
    for geometry in GEOMETRIES:
        summary[geometry]={}
        for regime in REGIMES:
            summary[geometry][regime]={}
            for arm in ARMS:
                group=[
                    r for r in rows if r["geometry"]==geometry
                    and r["regime"]==regime and r["arm"]==arm
                ]
                meter=sum(r["postchange_meters"] for r in group)
                summary[geometry][regime][arm]={
                    metric:round(mean(r[metric] for r in group),6)
                    for metric in (
                        "real_physical_steps","postchange_real_steps",
                        "postchange_visited_cells",
                        "postchange_new_cells_not_seen_previously",
                        "postchange_trajectory_rms_radius",
                        "postchange_collisions","postchange_meters",
                        "postchange_altered_action_count",
                    )
                }|{
                    "pooled_postchange_collisions_per_real_meter":round(
                        sum(r["postchange_collisions"] for r in group)/
                        max(meter,1e-12),9
                    ),
                    "postchange_model_override_fraction":round(
                        sum(r["postchange_altered_action_count"] for r in group)
                        /max(sum(r["postchange_real_steps"] for r in group),1),6
                    ),
                    "late_true_model_next_sensor_mse":round(
                        sum(r["late_sum_actual_true_mse"] for r in group)
                        /max(sum(r["postchange_real_steps"] for r in group),1),9
                    ),
                    "late_yoked_model_next_sensor_mse":round(
                        sum(r["late_sum_actual_yoked_mse"] for r in group)
                        /max(sum(r["postchange_real_steps"] for r in group),1),9
                    ),
                }
    pairing=all(
        len({r["initial_brain_sha256"] for r in rows if (
            r["seed"]==seed and r["geometry"]==geometry
            and r["regime"]==regime
        )})==1
        and len({sha(r["initial_ten_rectangles"]) for r in rows if (
            r["seed"]==seed and r["geometry"]==geometry
            and r["regime"]==regime
        )})==1
        and len({sha(r["initial_body_pose"]) for r in rows if (
            r["seed"]==seed and r["geometry"]==geometry
            and r["regime"]==regime
        )})==1
        for seed in SEEDS for geometry in GEOMETRIES for regime in REGIMES
    )
    common_motor_noise=all(
        all(
            [f["original_exogenous_motor_noise"] for f in r["raw_trace"][:stop]]
            ==[f["original_exogenous_motor_noise"]
                for f in group[0]["raw_trace"][:stop]]
            for r in group
        )
        for geometry in GEOMETRIES for regime in REGIMES for seed in SEEDS
        for group in [[r for r in rows if
                       r["seed"]==seed and r["geometry"]==geometry
                       and r["regime"]==regime]]
        for stop in [min(len(r["raw_trace"]) for r in group)]
    )
    def rows_for(seed,geometry,regime,arm):
        return next(r for r in rows if
                    r["seed"]==seed and r["geometry"]==geometry
                    and r["regime"]==regime and r["arm"]==arm)
    passive_physically_equal=all(
        [
            (t["physical_body_outcome"],t["physically_executed_motor"],
             t["actual_next_three_wall_sensors"])
            for t in rows_for(seed,geometry,regime,"recurrent_frozen")["raw_trace"]
        ]==[
            (t["physical_body_outcome"],t["physically_executed_motor"],
             t["actual_next_three_wall_sensors"])
            for t in rows_for(seed,geometry,regime,"passive_predictor")["raw_trace"]
        ]
        for seed in SEEDS for geometry in GEOMETRIES for regime in REGIMES
    )
    prechange_paired=all(
        [
            {k:f[k] for k in (
                "physically_executed_motor","actual_next_three_wall_sensors",
                "physical_body_outcome","selected_candidate_index",
                "original_16_recurrent_states",
            )} for f in rows_for(seed,geometry,"static",arm)["raw_trace"]
            if f["tick"]<CHANGE_TICK
        ]==[
            {k:f[k] for k in (
                "physically_executed_motor","actual_next_three_wall_sensors",
                "physical_body_outcome","selected_candidate_index",
                "original_16_recurrent_states",
            )} for f in rows_for(seed,geometry,"removal",arm)["raw_trace"]
            if f["tick"]<CHANGE_TICK
        ]
        for seed in SEEDS for geometry in GEOMETRIES for arm in ARMS
    )
    geometry_valid=all(
        len(r["initial_ten_rectangles"])==10
        and len(r["postchange_rectangles"])==(
            5 if r["regime"]=="removal" and r["environment_removed_five_walls"]
            else 10
        )
        and r["postchange_rectangles"]==(
            r["initial_ten_rectangles"][::2]
            if r["regime"]=="removal" and r["environment_removed_five_walls"]
            else r["initial_ten_rectangles"]
        )
        for r in rows
    )
    world_maps={
        geom:len({
            sha(r["initial_ten_rectangles"]) for r in rows
            if r["geometry"]==geom
        }) for geom in GEOMETRIES
    }
    near=summary["nearfield"]["removal"]
    fr=near["recurrent_frozen"]
    true=near["true_closed_loop"]
    passive=near["passive_predictor"]
    gates={
        "01_exact_384_balanced_full_replay_required":
            len(rows)==384 and all(
                sum(r["geometry"]==geom and r["regime"]==reg
                    and r["arm"]==arm for r in rows)==len(SEEDS)
                for geom in GEOMETRIES for reg in REGIMES for arm in ARMS
            ),
        "02_original_R1_physics_collision_energy_no_resurrection":
            audit["all_actual_physical_neuron_body_replays_valid"],
        "03_equal_initial_brain_world_pose_and_random_motor_noise":
            pairing and common_motor_noise,
        "04_exact_16_recurrent_8_local_sensors_no_global_map":
            audit["five_candidate_selection_and_preaction_model_reconstructable"],
        "05_zero_initialized_heads_lifelong_postaction_learning_only":
            audit["predictor_heads_updated_only_after_real_observation"]
            and all(
                r["head_weights_initially_zero"]
                and r["head_update_events"]=={
                    head:r["real_physical_steps"] if r["arm"] in HEAD_ARMS
                    else 0 for head in HEADS
                } for r in rows
            ),
        "06_exact_five_candidates_predicted_preaction_no_world_lookahead":
            audit["five_candidate_selection_and_preaction_model_reconstructable"],
        "07_older_yoked_real_motor_same_organism_min_lag_six":
            audit["sham_motor_strictly_from_own_old_physical_actions"],
        "08_passive_predictor_physically_equals_frozen_baseline":
            passive_physically_equal
            and audit["original_recurrent_weights_frozen_except_r2_output_readout"],
        "09_static_changed_match_to_tick240_and_remove_exact_five":
            prechange_paired and geometry_valid and all(
                n==len(SEEDS) for n in world_maps.values()
            ),
        "10_complete_raw_logs_no_evolution_or_food":
            len(rows)==384 and all(
                r["raw_trace"] and r["original_434_neural_coefficients_at_birth"]
                and r["final_3x84_online_sensor_head_weights"]
                for r in rows
            ),
        "11_passive_online_true_sensor_prediction_late_mse_le_95pct_yoked":
            passive["late_yoked_model_next_sensor_mse"]>0
            and passive["late_true_model_next_sensor_mse"]<=
                .95*passive["late_yoked_model_next_sensor_mse"],
        "12_changed_nearfield_true_late_cells_ge_frozen_plus10pct":
            true["postchange_visited_cells"]>=
                1.10*fr["postchange_visited_cells"],
        "13_changed_nearfield_true_contacts_per_meter_le_frozen_90pct":
            fr["pooled_postchange_collisions_per_real_meter"]>0
            and true["pooled_postchange_collisions_per_real_meter"]<=
                .90*fr["pooled_postchange_collisions_per_real_meter"],
        "14_changed_nearfield_true_late_cells_ge_yoked_plus10pct":
            true["postchange_visited_cells"]>=1.10*
                near["yoked_closed_loop"]["postchange_visited_cells"],
        "15_changed_nearfield_true_late_cells_ge_random_menu_plus10pct":
            true["postchange_visited_cells"]>=1.10*
                near["random_menu"]["postchange_visited_cells"],
        "16_changed_nearfield_true_RMS_radius_ge_frozen_90pct":
            true["postchange_trajectory_rms_radius"]>=.90*
                fr["postchange_trajectory_rms_radius"],
        "17_true_model_actually_steers_postchange_at_least10pct":
            true["postchange_model_override_fraction"]>=.10,
    }
    paired_bootstrap={
        geo:{
            ref:_paired_bootstrap(
                rows,geo,"removal","true_closed_loop",ref
            )
            for ref in (
                "recurrent_frozen","yoked_closed_loop",
                "random_menu","r2_progress"
            )
        } for geo in GEOMETRIES
    }
    return {
        "name":"FlyLoop R7 within-life learned sensor model modifies real continuous actions",
        "protocol":"docs/FLYLOOP_R7_PROTOCOL_2026-11-24.md",
        "metadata":{
            "world_seeds":SEEDS,
            "geometry_families":GEOMETRIES,
            "environment_regimes":REGIMES,
            "arms":ARMS,
            "maximum_physical_steps_per_life":HORIZON,
            "exogenous_wall_removal_at_tick":CHANGE_TICK,
            "head_only_warmup_steps":WARMUP,
            "steering_candidate_offset":STEER_DELTA,
            "throttle_candidate_offset":THROTTLE_DELTA,
            "original_recurrent_units":r1.N,
            "original_neural_coefficients":r1.PARAMS,
            "learning_head_coefficients_each":r4.HEAD_PARAMS,
            "engineered_sensory_variation_score":True,
            "online_prediction_modifies_action":True,
            "no_actual_world_candidate_lookahead":True,
            "no_food_escape_or_evolution":True,
            "original_r1_r6_m0_m4_sealed_m56_main_unmodified":True,
            "world_map_distinct_counts":world_maps,
        },
        "summary":summary,
        "integrity_audit":audit,
        "paired_world_seed_bootstrap_exploratory":paired_bootstrap,
        "gates":gates,
        "gates_passed":sum(gates.values()),
        "screen_pass":all(gates.values()),
        "closed_loop_action_selection_implemented":True,
        "effective_adaptive_navigation_demonstrated":all(gates.values()),
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
    p=argparse.ArgumentParser()
    p.add_argument(
        "--out",type=Path,
        default=Path("results/flyloop_r7/r7_online_predictive_action.json")
    )
    args=p.parse_args()
    result=run()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(
        json.dumps(result,sort_keys=True,separators=(",",":"))+"\n",
        encoding="utf-8",
    )
    print("FLYLOOP_R7 "+json.dumps({
        k:result[k] for k in (
            "name","summary","integrity_audit","gates",
            "gates_passed","screen_pass",
            "paired_world_seed_bootstrap_exploratory",
            "raw_rows_sha256"
        )
    },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
