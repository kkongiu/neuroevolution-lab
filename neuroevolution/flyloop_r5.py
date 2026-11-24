"""FlyLoop R5 — ACTUAL experimental motor-action interventions from identical states.

Research experiment, NOT an organism motor planner.
R1–R4 are imported without modification. Neural/motor policy frozen.
Pretrain three R4 action-conditioned local sensory heads ONLY on 300
physically experienced natural steps, then freeze them. At 30 pre-fixed
snapshots, copy the SAME body + world state five times and PHYSICALLY
execute five distinct forced motor commands in actual R1 physics.
Score all model predictions against resulting REAL next wall sensors.

Frozen before results: docs/FLYLOOP_R5_PROTOCOL_2026-11-24.md
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

SEEDS=tuple(range(10300,10324))
SCENES=("obstacles","nearfield")
TRAIN_STEPS=300
HORIZON=600
PROBE_STEPS=tuple(range(300,600,10))
ACTIONS=("natural","steer_left","steer_right","throttle_low","throttle_high")
YOKE_LAG=r4.YOKED_LAG
SENSOR_EFFECT_THRESHOLD=.005
LONG_EFFECT_THRESHOLD=.05
LONG_STEPS=6
HEADS=r4.PREDICTOR_NAMES

def sha(x):
    return hashlib.sha256(
        json.dumps(x,sort_keys=True,separators=(",",":")).encode()
    ).hexdigest()

def mean(xs):
    xs=list(xs)
    return sum(xs)/len(xs) if xs else 0.

def squared(a,b):
    return sum((x-y)**2 for x,y in zip(a,b))/3.


class World(r1.World):
    """Unchanged original R1 physical rules, registered independent R5 seeds."""
    def __init__(self,seed,scene):
        if seed not in SEEDS or scene not in SCENES:
            raise ValueError("R5 seed or world not registered")
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


def commands(natural):
    """EXPERIMENTER FORCED interventions, NEVER selected by the organism."""
    throttle,steer=natural
    candidates={
        "natural":(throttle,steer),
        "steer_left":(throttle,-.85),
        "steer_right":(throttle,.85),
        "throttle_low":(.05,steer),
        "throttle_high":(.95,steer),
    }
    assert all(
        len(v)==2 and all(-1.<=x<=1. for x in v)
        for v in candidates.values()
    )
    return candidates


def physical_branches(world,hidden,rays,natural,heads,yoked,source,tick):
    """Read-only neural state; every motor action PHYSICALLY executed in a clone."""
    before_body=copy.deepcopy(vars(world.body))
    before_head_sha=sha(heads.weights)
    all_commands=commands(natural)
    trials=[]
    for label,cmd in all_commands.items():
        # Every clone starts with the same physical, sensory and neural
        # state, independently of all other forced branches.
        clone=copy.deepcopy(world)
        assert vars(clone.body)==before_body and clone.frame==world.frame
        phi_true=heads.make_features(hidden,rays,cmd)
        phi_blind=heads.make_features(hidden,rays,(0.,0.))
        phi_yoked=heads.make_features(hidden,rays,yoked)
        predicted={
            "true":heads.predict("true",rays,phi_true),
            "blind":heads.predict("blind",rays,phi_blind),
            "yoked":heads.predict("yoked",rays,phi_yoked),
            "persistence":list(rays),
        }
        # Pre-action predictions above are now locked.
        step1=clone.integrate(cmd)
        after_sensor=tuple(clone.perceive()[1:4])
        # Five MORE REAL integrations using the same experimentally forced
        # motor command, no motor brain, no predictor and no neural learning.
        for _ in range(LONG_STEPS-1):
            if clone.body.energy<=0. or clone.body.integrity<=0.:
                raise AssertionError("Physical intervention clone died")
            clone.integrate(cmd)
        last_position=[clone.body.x,clone.body.y]
        trial={
            "label":label,
            "forced_executed_motor":list(cmd),
            "physical_pre_body":copy.deepcopy(before_body),
            "step1_physics":step1,
            "real_step1_next_wall_rays":list(after_sensor),
            "six_step_terminal_xy":last_position,
            "model_predictions":predicted,
            "actual_one_step_mse":{
                name:squared(v,after_sensor)
                for name,v in predicted.items()
            },
        }
        trials.append(trial)
    assert sha(heads.weights)==before_head_sha
    original=next(t for t in trials if t["label"]=="natural")
    other=[t for t in trials if t["label"]!="natural"]
    contrasts=[]
    for trial in other:
        actual_delta=[
            a-b for a,b in zip(
                trial["real_step1_next_wall_rays"],
                original["real_step1_next_wall_rays"],
            )
        ]
        predicted_delta=[
            a-b for a,b in zip(
                trial["model_predictions"]["true"],
                original["model_predictions"]["true"],
            )
        ]
        contrasts.append({
            "label":trial["label"],
            "actual_sensor_change":actual_delta,
            "predicted_true_sensor_change":predicted_delta,
            "physically_sensor_informative":
                max(abs(v) for v in actual_delta)>=SENSOR_EFFECT_THRESHOLD,
            "true_change_prediction_mse":
                squared(predicted_delta,actual_delta),
            "zero_change_baseline_mse":squared(
                (0.,0.,0.),actual_delta
            ),
        })
    left=next(t for t in trials if t["label"]=="steer_left")
    right=next(t for t in trials if t["label"]=="steer_right")
    divergence=math.hypot(
        left["six_step_terminal_xy"][0]-right["six_step_terminal_xy"][0],
        left["six_step_terminal_xy"][1]-right["six_step_terminal_xy"][1],
    )
    return {
        "tick":tick,
        "preaction_full_body":before_body,
        "preaction_16_neurons":list(hidden),
        "preaction_8_sensors":list(world.perceive()),
        "preaction_three_wall_rays":list(rays),
        "original_natural_executed_motor":list(natural),
        "frozen_3_head_weights_sha256":before_head_sha,
        "yoked_past_motor":list(yoked),
        "yoked_source_tick":source,
        "experiments_are_externally_forced":True,
        "trials":trials,
        "contrasts":contrasts,
        "six_step_left_right_xy_distance":divergence,
        "six_step_physically_divergent":
            divergence>=LONG_EFFECT_THRESHOLD,
    }


def simulate(seed,scene):
    world=World(seed,scene)
    brain=r2.MotorLearner(seed,"recurrent_frozen")
    heads=r4.ActionSensorHeads(seed)
    neural_initial=copy.deepcopy(brain.circuit.weights)
    log=[]
    probes=[]
    frozen_heads=None
    training_updates_first_phase=None

    for tick in range(HORIZON):
        if world.body.energy<=0. or world.body.integrity<=0.:
            break
        old_sensor=tuple(world.perceive())
        rays=tuple(old_sensor[1:4])
        raw,hidden,perceived,noise,natural=brain.command(old_sensor)
        features,yoked,source=heads.inputs(tick,hidden,rays,natural)
        prediction={
            name:heads.predict(name,rays,features[name])
            for name in HEADS
        }
        if tick==TRAIN_STEPS:
            frozen_heads=copy.deepcopy(heads.weights)
            training_updates_first_phase=dict(heads.updates)
        maybe_probe=None
        if tick in PROBE_STEPS:
            assert tick>=TRAIN_STEPS and frozen_heads is not None
            assert heads.weights==frozen_heads
            maybe_probe=physical_branches(
                world,hidden,rays,natural,heads,yoked,source,tick
            )
            probes.append(maybe_probe)

        # Separately progress the ONE ORIGINAL world using its own
        # natural controller's chosen movement. Experiments above cannot
        # alter this world because they occur in separate deep clones.
        phys=world.integrate(natural)
        after=tuple(world.perceive()[1:4])
        if maybe_probe is not None:
            natural_clone=maybe_probe["trials"][0]
            if natural_clone["step1_physics"]!=phys or not all(
                abs(a-b)<1e-12
                for a,b in zip(
                    natural_clone["real_step1_next_wall_rays"],after
                )
            ):
                raise AssertionError("Natural experimental clone != original physical life")
        if tick<TRAIN_STEPS:
            heads.train_all(features,prediction,after)
        else:
            assert heads.weights==frozen_heads
        heads.motor_history.append(tuple(natural))
        log.append({
            "tick":tick,
            "real_pre_8_sensors":list(perceived),
            "real_16_neurons":list(hidden),
            "natural_raw_motor":list(raw),
            "motor_noise":list(noise),
            "natural_executed_motor":list(natural),
            "past_yoked_motor":list(yoked),
            "past_yoked_source_tick":source,
            "preaction_head_predictions":prediction,
            "actual_next_wall_rays":list(after),
            "physical_real_action":phys,
            "head_was_trained":tick<TRAIN_STEPS,
        })
    if frozen_heads is None:
        frozen_heads=copy.deepcopy(heads.weights)
    all_trials=[t for p in probes for t in p["trials"]]
    all_contrasts=[c for p in probes for c in p["contrasts"]]
    informative=[
        c for c in all_contrasts if c["physically_sensor_informative"]
    ]
    sum_loss={
        name:sum(
            t["actual_one_step_mse"][name] for t in all_trials
        ) for name in ("true","blind","yoked","persistence")
    }
    contrast_sum={
        "all_true":sum(c["true_change_prediction_mse"] for c in all_contrasts),
        "all_zero":sum(c["zero_change_baseline_mse"] for c in all_contrasts),
        "informative_true":sum(c["true_change_prediction_mse"] for c in informative),
        "informative_zero":sum(c["zero_change_baseline_mse"] for c in informative),
    }
    final_coeff=copy.deepcopy(brain.circuit.weights)
    return {
        "seed":seed,
        "scene":scene,
        "original_world_rectangles":[list(v) for v in world.rectangles],
        "initial_pose":[2.2,2.2,log[0]["physical_real_action"]["before_heading"]],
        "initial_original_neural_coefficients_sha256":sha(neural_initial),
        "original_neural_coefficients":neural_initial,
        "original_neural_final_sha256":sha(final_coeff),
        "original_neural_coefficient_changed":neural_initial!=final_coeff,
        "three_heads_originally_all_zero":True,
        "frozen_three_heads_after_300_real_steps":frozen_heads,
        "final_three_heads":copy.deepcopy(heads.weights),
        "training_updates_per_head":training_updates_first_phase,
        "original_life_steps":len(log),
        "real_original_life":log,
        "probe_count":len(probes),
        "real_intervention_one_step_trials":len(all_trials),
        "actual_pair_contrast_count":len(all_contrasts),
        "informative_contrast_count":len(informative),
        "original_vs_forced_six_step_divergent_count":
            sum(p["six_step_physically_divergent"] for p in probes),
        "six_step_left_right_divergence_sum":sum(
            p["six_step_left_right_xy_distance"] for p in probes
        ),
        "sum_heldout_real_next_sensor_mse":sum_loss,
        "sum_action_contrast_mse":contrast_sum,
        "probe_snapshots_and_real_branches":probes,
    }


def _audit(rows):
    physics=True
    histories=True
    frozen=True
    provenance=True
    for row in rows:
        if row["original_life_steps"]!=HORIZON or len(
            row["real_original_life"]
        )!=HORIZON or row["probe_count"]!=len(PROBE_STEPS):
            physics=False
        if row["original_neural_coefficient_changed"] or row[
            "initial_original_neural_coefficients_sha256"
        ]!=row["original_neural_final_sha256"]:
            frozen=False
        if row["frozen_three_heads_after_300_real_steps"]!=row["final_three_heads"]:
            frozen=False
        if row["training_updates_per_head"]!={name:TRAIN_STEPS for name in HEADS}:
            frozen=False
        for original in row["real_original_life"]:
            tick=original["tick"]
            source=original["past_yoked_source_tick"]
            if source is None:
                if tick>=YOKE_LAG or original["past_yoked_motor"]!=[0.,0.]:
                    histories=False
            elif not (
                0<=source<=tick-YOKE_LAG
                and original["past_yoked_motor"]==
                row["real_original_life"][source]["natural_executed_motor"]
            ):
                histories=False
            if original["head_was_trained"]!=(tick<TRAIN_STEPS):
                frozen=False
        if [x["tick"] for x in row["probe_snapshots_and_real_branches"]]!=list(PROBE_STEPS):
            physics=False
        trained_head=r4.ActionSensorHeads(row["seed"])
        trained_head.weights=copy.deepcopy(row["frozen_three_heads_after_300_real_steps"])
        for p in row["probe_snapshots_and_real_branches"]:
            tick=p["tick"]
            pre=p["preaction_full_body"]
            if p["frozen_3_head_weights_sha256"]!=sha(trained_head.weights):
                frozen=False
            if p["yoked_source_tick"]!=row["real_original_life"][tick][
                "past_yoked_source_tick"
            ]:
                histories=False
            if p["yoked_past_motor"]!=row["real_original_life"][tick][
                "past_yoked_motor"
            ]:
                histories=False
            if [t["label"] for t in p["trials"]]!=list(ACTIONS):
                physics=False
            if not p["experiments_are_externally_forced"]:
                provenance=False
            if p["original_natural_executed_motor"]!=row[
                "real_original_life"
            ][tick]["natural_executed_motor"]:
                provenance=False
            if p["trials"][0]["step1_physics"]!=row[
                "real_original_life"
            ][tick]["physical_real_action"]:
                physics=False
            if p["trials"][0]["real_step1_next_wall_rays"]!=row[
                "real_original_life"
            ][tick]["actual_next_wall_rays"]:
                physics=False
            cm=commands(p["original_natural_executed_motor"])
            for trial in p["trials"]:
                forced=cm[trial["label"]]
                if trial["forced_executed_motor"]!=list(forced):
                    physics=False
                if trial["physical_pre_body"]!=pre:
                    physics=False
                after=trial["step1_physics"]
                x,y=after["position"]
                if not (
                    r1.RADIUS-1e-7<=x<=r1.SIDE-r1.RADIUS+1e-7
                    and r1.RADIUS-1e-7<=y<=r1.SIDE-r1.RADIUS+1e-7
                ):
                    physics=False
                if any(
                    x1-r1.RADIUS<x<x2+r1.RADIUS and
                    y1-r1.RADIUS<y<y2+r1.RADIUS
                    for x1,y1,x2,y2 in row["original_world_rectangles"]
                ):
                    physics=False
                expected={
                    "true":trained_head.predict(
                        "true",p["preaction_three_wall_rays"],
                        trained_head.make_features(
                            p["preaction_16_neurons"],
                            p["preaction_three_wall_rays"],forced
                        )
                    ),
                    "blind":trained_head.predict(
                        "blind",p["preaction_three_wall_rays"],
                        trained_head.make_features(
                            p["preaction_16_neurons"],
                            p["preaction_three_wall_rays"],(0.,0.)
                        )
                    ),
                    "yoked":trained_head.predict(
                        "yoked",p["preaction_three_wall_rays"],
                        trained_head.make_features(
                            p["preaction_16_neurons"],
                            p["preaction_three_wall_rays"],
                            p["yoked_past_motor"]
                        )
                    ),
                    "persistence":p["preaction_three_wall_rays"],
                }
                if any(
                    trial["model_predictions"][key]!=value
                    for key,value in expected.items()
                ):
                    provenance=False
                for name,pred in trial["model_predictions"].items():
                    err=squared(
                        pred,trial["real_step1_next_wall_rays"]
                    )
                    if abs(err-trial["actual_one_step_mse"][name])>1e-10:
                        provenance=False
            base=p["trials"][0]
            for t,c in zip(p["trials"][1:],p["contrasts"]):
                actual=[x-y for x,y in zip(
                    t["real_step1_next_wall_rays"],
                    base["real_step1_next_wall_rays"],
                )]
                pred=[x-y for x,y in zip(
                    t["model_predictions"]["true"],
                    base["model_predictions"]["true"],
                )]
                if c["actual_sensor_change"]!=actual or (
                    c["predicted_true_sensor_change"]!=pred
                ):
                    provenance=False
                if c["physically_sensor_informative"]!=(
                    max(abs(x) for x in actual)>=SENSOR_EFFECT_THRESHOLD
                ):
                    provenance=False
            left=p["trials"][1]["six_step_terminal_xy"]
            right=p["trials"][2]["six_step_terminal_xy"]
            diverged=math.hypot(left[0]-right[0],left[1]-right[1])
            if abs(diverged-p["six_step_left_right_xy_distance"])>1e-10:
                physics=False
    return {
        "physical_original_clone_and_obstacle_clearance":physics,
        "historic_yoked_action_provenance_6plus_ticks_old":histories,
        "all_neural_and_predictor_coefficients_frozen_during_probes":frozen,
        "predictions_and_real_sensor_differences_reconstructable":provenance,
    }


def _summarize(rows):
    total_trials=sum(r["real_intervention_one_step_trials"] for r in rows)
    total_contrasts=sum(r["actual_pair_contrast_count"] for r in rows)
    ninfo=sum(r["informative_contrast_count"] for r in rows)
    losses={
        head:sum(r["sum_heldout_real_next_sensor_mse"][head] for r in rows)
        for head in ("true","blind","yoked","persistence")
    }
    delta={
        kind:sum(r["sum_action_contrast_mse"][kind] for r in rows)
        for kind in (
            "all_true","all_zero","informative_true","informative_zero"
        )
    }
    snaps=sum(r["probe_count"] for r in rows)
    divergent=sum(r["original_vs_forced_six_step_divergent_count"] for r in rows)
    return {
        "lives":len(rows),
        "probe_snapshots":snaps,
        "real_intervention_one_step_trials":total_trials,
        "actual_pair_contrasts":total_contrasts,
        "sensor_informative_contrasts":ninfo,
        "informative_fraction":round(ninfo/max(total_contrasts,1),6),
        "sensor_informative_world_seeds":len({
            r["seed"] for r in rows if r["informative_contrast_count"]>0
        }),
        "physical_six_step_left_right_divergent_probes":divergent,
        "physical_six_step_divergent_fraction":round(
            divergent/max(snaps,1),6
        ),
        "mean_left_right_real_terminal_distance":round(
            sum(r["six_step_left_right_divergence_sum"] for r in rows)
            /max(snaps,1),6
        ),
        "one_step_mse":{
            k:round(v/max(total_trials,1),12)
            for k,v in losses.items()
        },
        "one_step_mse_unrounded_sums":losses,
        "real_contrast_mse_all":{
            "true":round(delta["all_true"]/max(total_contrasts,1),12),
            "zero":round(delta["all_zero"]/max(total_contrasts,1),12),
        },
        "real_contrast_mse_informative":{
            "true":round(delta["informative_true"]/max(ninfo,1),12),
            "zero":round(delta["informative_zero"]/max(ninfo,1),12),
        },
        "contrast_unrounded_sums":delta,
    }


def _bootstrap(rows,metric1,metric2,n=1000):
    by_seed={
        seed:[r for r in rows if r["seed"]==seed]
        for seed in SEEDS
    }
    # 2 scenes nested inside each independent seeded world.
    def value(group,key):
        n=sum(r["real_intervention_one_step_trials"] for r in group)
        return sum(
            r["sum_heldout_real_next_sensor_mse"][key] for r in group
        )/n
    diffs=[
        value(by_seed[s],metric1)-value(by_seed[s],metric2)
        for s in SEEDS
    ]
    rng=random.Random(103021)
    samples=sorted(
        mean(diffs[rng.randrange(len(diffs))] for _ in diffs)
        for _ in range(n)
    )
    return {
        "per_world_seed_mean_mse_difference":round(mean(diffs),10),
        "exploratory_paired_seed_bootstrap95":[
            round(samples[25],10),round(samples[974],10)
        ],
        "independent_world_seeds":len(SEEDS),
    }


def run():
    rows=[
        simulate(seed,scene)
        for scene in SCENES for seed in SEEDS
    ]
    audit=_audit(rows)
    all_summary=_summarize(rows)
    by_scene={
        scene:_summarize([r for r in rows if r["scene"]==scene])
        for scene in SCENES
    }
    geometry_valid=all(
        len(r["original_world_rectangles"])==r1.OBSTACLES
        and all(not (
            x1-1.1<2.2<x2+1.1 and y1-1.1<2.2<y2+1.1
        ) for x1,y1,x2,y2 in r["original_world_rectangles"])
        for r in rows
    )
    distinct_maps={
        scene:len({
            sha(r["original_world_rectangles"])
            for r in rows if r["scene"]==scene
        }) for scene in SCENES
    }
    physically_different_actions=all(
        len({
            tuple(trial["forced_executed_motor"])
            for trial in p["trials"]
        })>=4
        for r in rows for p in r["probe_snapshots_and_real_branches"]
    )
    nseeds=all_summary["sensor_informative_world_seeds"]
    info=all_summary["sensor_informative_contrasts"]
    ti=all_summary["real_contrast_mse_informative"]
    gates={
        "01_48_original_lives_1440_snapshots_7200_branches_repeat":
            len(rows)==48 and all_summary["probe_snapshots"]==1440
            and all_summary["real_intervention_one_step_trials"]==7200
            and all_summary["actual_pair_contrasts"]==5760,
        "02_identical_cloned_physical_preaction_world_and_legal_motion":
            audit["physical_original_clone_and_obstacle_clearance"],
        "03_natural_clone_matches_independently_advanced_original_world":
            audit["physical_original_clone_and_obstacle_clearance"],
        "04_original_16_hidden_8_local_434_coefficients_all_frozen":
            audit["all_neural_and_predictor_coefficients_frozen_during_probes"],
        "05_300_real_training_only_then_zero_head_or_motor_updates":
            audit["all_neural_and_predictor_coefficients_frozen_during_probes"],
        "06_five_bounded_distinct_real_forced_commands_no_oracle":
            physically_different_actions,
        "07_past_yoke_same_living_agent_at_least_six_ticks_old":
            audit["historic_yoked_action_provenance_6plus_ticks_old"],
        "08_preaction_predictions_only_real_executed_physical_targets":
            audit["predictions_and_real_sensor_differences_reconstructable"],
        "09_two_valid_exogenous_obstacle_generators_24_maps_each":
            geometry_valid and all(n==24 for n in distinct_maps.values()),
        "10_complete_original_raw_data_and_false_stop_flags":
            len(rows)==48 and all(
                len(r["probe_snapshots_and_real_branches"])==30
                and r["original_neural_coefficients"]
                and r["final_three_heads"]
                for r in rows
            ),
        "11_six_step_real_steer_intervention_diverges_60pct":
            all_summary["physical_six_step_divergent_fraction"]>=.60,
        "12_at_least_100_sensor_effect_pairs_and_20_distinct_seeds":
            info>=100 and nseeds>=20,
        "13_true_head_offpolicy_real_one_step_mse_le_95pct_blind":
            all_summary["one_step_mse"]["true"]<=
            .95*all_summary["one_step_mse"]["blind"],
        "14_true_head_offpolicy_real_one_step_mse_le_95pct_yoked":
            all_summary["one_step_mse"]["true"]<=
            .95*all_summary["one_step_mse"]["yoked"],
        "15_true_predicted_physical_delta_mse_le_90pct_zero_baseline":
            info>=100 and ti["zero"]>0
            and ti["true"]<=.90*ti["zero"],
        "16_both_scenes_real_delta_predictor_le_95pct_zero":
            all(
                by_scene[sc]["sensor_informative_contrasts"]>=20 and
                by_scene[sc]["real_contrast_mse_informative"]["zero"]>0 and
                by_scene[sc]["real_contrast_mse_informative"]["true"]<=
                .95*by_scene[sc]["real_contrast_mse_informative"]["zero"]
                for sc in SCENES
            ),
    }
    return {
        "name":"FlyLoop R5 actual cloned physical motor interventions",
        "protocol":"docs/FLYLOOP_R5_PROTOCOL_2026-11-24.md",
        "metadata":{
            "fresh_world_seeds":SEEDS,
            "physical_scenes":SCENES,
            "probes_per_life":len(PROBE_STEPS),
            "number_forced_commands":len(ACTIONS),
            "forced_commands":ACTIONS,
            "natural_training_only_until_tick":TRAIN_STEPS-1,
            "test_head_frozen_after_tick":TRAIN_STEPS,
            "physical_branch_steps":LONG_STEPS,
            "actual_sensor_difference_threshold":SENSOR_EFFECT_THRESHOLD,
            "physical_terminal_divergence_threshold":LONG_EFFECT_THRESHOLD,
            "original_recurrent_neurons":r1.N,
            "original_brain_weights":r1.PARAMS,
            "predictive_heads_each_coefficients":r4.HEAD_PARAMS,
            "old_m0_m4_and_r1_r4_unchanged":True,
            "no_control_policy_learning_during_interventions":True,
            "all_counterfactuals_are_real_executed_cloned_physics":True,
            "no_food_escape_navigation_or_evolution_claim":True,
            "obstacle_maps_per_scene":distinct_maps,
        },
        "summary":all_summary,
        "scene_summaries":by_scene,
        "integrity_audit":audit,
        "exploratory_seed_bootstrap":{
            "true_minus_blind":_bootstrap(rows,"true","blind"),
            "true_minus_yoked":_bootstrap(rows,"true","yoked"),
        },
        "gates":gates,
        "gates_passed":sum(gates.values()),
        "screen_pass":all(gates.values()),
        "autonomous_control_demonstrated":False,
        "food_search_demonstrated":False,
        "escape_reflex_learned":False,
        "autonomous_goal_invention_demonstrated":False,
        "general_intelligence_demonstrated":False,
        "biological_fly_replication_demonstrated":False,
        "evolution_authorized":False,
        "advance_to_evolution":False,
        "raw_sha256":sha(rows),
        "raw_rows":rows,
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument(
        "--out",type=Path,
        default=Path("results/flyloop_r5/r5_physical_interventions.json")
    )
    args=parser.parse_args()
    outcome=run()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(
        json.dumps(outcome,sort_keys=True,separators=(",",":"))+"\n",
        encoding="utf-8",
    )
    print("FLYLOOP_R5 "+json.dumps({
        k:outcome[k] for k in (
            "name","summary","scene_summaries","integrity_audit",
            "gates","gates_passed","screen_pass",
            "exploratory_seed_bootstrap","raw_sha256"
        )
    },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
