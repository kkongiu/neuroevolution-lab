"""FlyLoop R6: real intervention-trained heads tested on unseen worlds/actions.

All sensory training labels are PHYSICALLY EXECUTED three-step outcomes
from identical-world cloned states. Three 84-coefficient R4 prediction
heads learn across investigator-controlled TRAIN worlds only. No
training, motor learning, planning, or evolution in held-out TEST worlds.
This is EXTERNAL training/transfer, not organism-internal self-learning.

Frozen protocol: docs/FLYLOOP_R6_PROTOCOL_2026-11-24.md
R1-R5 original code unmodified.
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

TRAIN_SEEDS=tuple(range(10500,10524))
TEST_SEEDS=tuple(range(10600,10624))
SCENES=("obstacles","nearfield")
SNAPSHOT_TICKS=tuple(range(40,221,20))
HORIZON=240
BRANCH_STEPS=3
ACTIONS=("natural","steer_left","steer_right","throttle_low","throttle_high")
INFORMATIVE_THRESHOLD=.02
PHYSICAL_DIVERGENCE_THRESHOLD=.04
HEADS=("true","blind","yoked")


def sha(x):
    return hashlib.sha256(
        json.dumps(x,sort_keys=True,separators=(",",":")).encode()
    ).hexdigest()


def mean(x):
    x=list(x)
    return sum(x)/len(x) if x else 0.


def mse(a,b):
    return sum((x-y)**2 for x,y in zip(a,b))/3.


class World(r1.World):
    def __init__(self,seed,scene):
        if seed not in TRAIN_SEEDS+TEST_SEEDS or scene not in SCENES:
            raise ValueError("Unregistered R6 seed or scene")
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


def forced_commands(natural,phase):
    if phase not in ("train","test"):
        raise ValueError("Unregistered R6 phase")
    throttle,steer=natural
    if phase=="train":
        left,right,low,high=-.95,.95,.10,.90
    else:
        left,right,low,high=-.55,.55,.30,.70
    return {
        "natural":(throttle,steer),
        "steer_left":(throttle,left),
        "steer_right":(throttle,right),
        "throttle_low":(low,steer),
        "throttle_high":(high,steer),
    }


def predictor_features(hidden,rays,cmd):
    phi=r4.ActionSensorHeads.make_features(hidden,rays,cmd)
    assert len(phi)==r4.FEATURES==28
    return phi


def collect_snapshot(world,tick,hidden,sensors,natural,phase,heads):
    """5 actual bodily interventions, NEVER update the original world."""
    body=copy.deepcopy(vars(world.body))
    original_frame=world.frame
    rays=tuple(sensors[1:4])
    commands=forced_commands(natural,phase)
    motor_list=list(commands.values())
    trials=[]
    head_sha_before=sha(heads.weights)
    for i,(name,motor) in enumerate(commands.items()):
        # Sham command is rotated within the same five real experiments.
        # Same action distribution, same physical starting state, different
        # command-to-actual-label pairing. No fabricated sensory target.
        shuffled_motor=motor_list[(i+1)%len(motor_list)]
        phis={
            "true":predictor_features(hidden,rays,motor),
            "blind":predictor_features(hidden,rays,(0.,0.)),
            "yoked":predictor_features(hidden,rays,shuffled_motor if phase=="train" else motor),
        }
        preds={
            h:heads.predict(h,rays,phis[h]) for h in HEADS
        }|{"persistence":list(rays)}
        clone=copy.deepcopy(world)
        assert vars(clone.body)==body
        steps=[]
        for k in range(BRANCH_STEPS):
            if clone.body.energy<=0 or clone.body.integrity<=0:
                break
            physical=clone.integrate(motor)
            steps.append({
                "physics":physical,
                "observed_wall_rays":list(clone.perceive()[1:4]),
            })
        if not steps:
            raise AssertionError("Snapshot of dead original organism")
        real_end=steps[-1]["observed_wall_rays"]
        trial={
            "label":name,
            "forced_real_motor":list(motor),
            "same_state_rotated_training_motor":list(shuffled_motor),
            "actual_physical_steps":steps,
            "real_integrations":len(steps),
            "early_absorbing_death":len(steps)<BRANCH_STEPS,
            "terminal_body":copy.deepcopy(vars(clone.body)),
            "actual_three_step_terminal_wall_rays":real_end,
            "head_predictions_before_executing":preds,
            "heldout_actual_mse":{
                head:mse(pred,real_end) for head,pred in preds.items()
            },
        }
        if phase=="train":
            heads.train_all(phis,{h:preds[h] for h in HEADS},real_end)
        trials.append(trial)
    if vars(world.body)!=body or world.frame!=original_frame:
        raise AssertionError("Cloned experimental actions mutated original organism")
    if phase=="test" and sha(heads.weights)!=head_sha_before:
        raise AssertionError("Leak from test labels into frozen heads")
    # Natural first step of clone MUST match independently moving
    # original life; checked by calling simulation after this method.
    natural_trial=trials[0]
    contrasts=[]
    for trial in trials[1:]:
        real_delta=[
            x-y for x,y in zip(
                trial["actual_three_step_terminal_wall_rays"],
                natural_trial["actual_three_step_terminal_wall_rays"],
            )
        ]
        pred_delta=[
            x-y for x,y in zip(
                trial["head_predictions_before_executing"]["true"],
                natural_trial["head_predictions_before_executing"]["true"],
            )
        ]
        contrasts.append({
            "against_natural":trial["label"],
            "measured_actual_wall_delta":real_delta,
            "predicted_true_head_wall_delta":pred_delta,
            "informative_physical_effect":
                max(abs(x) for x in real_delta)>=INFORMATIVE_THRESHOLD,
            "true_delta_mse":mse(pred_delta,real_delta),
            "zero_delta_mse":mse((0.,0.,0.),real_delta),
        })
    left=trials[1]["terminal_body"]
    right=trials[2]["terminal_body"]
    separation=math.hypot(left["x"]-right["x"],left["y"]-right["y"])
    return {
        "tick":tick,
        "pre_world_frame":world.frame,
        "pre_body_state":body,
        "pre_eight_local_sensors":list(sensors),
        "pre_16_recurrent_neurons":list(hidden),
        "natural_executed_motor":list(natural),
        "head_hash_before_snapshot":head_sha_before,
        "head_hash_after_snapshot":sha(heads.weights),
        "all_actions_experimenter_forced":True,
        "trials":trials,
        "contrasts":contrasts,
        "three_step_left_right_physical_separation":separation,
        "physically_divergent":
            separation>=PHYSICAL_DIVERGENCE_THRESHOLD,
    }


def collect_life(seed,scene,phase,heads):
    world=World(seed,scene)
    brain=r2.MotorLearner(seed,"recurrent_frozen")
    initial=copy.deepcopy(brain.circuit.weights)
    original=[]
    snapshots=[]
    for tick in range(HORIZON):
        if world.body.energy<=0 or world.body.integrity<=0:
            break
        sensors=tuple(world.perceive())
        raw,hidden,perceived,noise,natural=brain.command(sensors)
        probe=None
        if tick in SNAPSHOT_TICKS:
            probe=collect_snapshot(
                world,tick,hidden,sensors,natural,phase,heads
            )
            snapshots.append(probe)
        outcome=world.integrate(natural)
        real_next=tuple(world.perceive()[1:4])
        if probe is not None:
            natural_first=probe["trials"][0]["actual_physical_steps"][0]
            if (natural_first["physics"]!=outcome or
                natural_first["observed_wall_rays"]!=list(real_next)):
                raise AssertionError("Natural action physical clone mismatch")
        original.append({
            "tick":tick,
            "before_local_eight_sensor_values":list(perceived),
            "before_16_neural_values":list(hidden),
            "raw_motor":list(raw),
            "seeded_motor_perturbation":list(noise),
            "executed_natural_motor":list(natural),
            "physical_outcome":outcome,
            "real_next_three_wall_rays":list(real_next),
        })
    all_trials=[t for p in snapshots for t in p["trials"]]
    all_contrasts=[c for p in snapshots for c in p["contrasts"]]
    informative=[c for c in all_contrasts if c["informative_physical_effect"]]
    return {
        "seed":seed,"scene":scene,"phase":phase,
        "exogenous_ten_rectangles":[list(r) for r in world.rectangles],
        "initial_body_heading":World(seed,scene).body.heading,
        "initial_neural_434_coefficients":initial,
        "final_neural_434_coefficients":copy.deepcopy(brain.circuit.weights),
        "natural_life_real_frames":len(original),
        "real_natural_life":original,
        "actual_snapshot_count":len(snapshots),
        "physically_executed_3step_trials":len(all_trials),
        "forced_vs_natural_contrasts":len(all_contrasts),
        "informative_actual_contrasts":len(informative),
        "divergent_forced_left_right_snapshots":sum(
            p["physically_divergent"] for p in snapshots
        ),
        "early_death_forced_clones":sum(
            t["early_absorbing_death"] for t in all_trials
        ),
        "sum_real_intervention_mse":{
            h:sum(t["heldout_actual_mse"][h] for t in all_trials)
            for h in ("true","blind","yoked","persistence")
        },
        "real_contrast_sums":{
            "all_true":sum(c["true_delta_mse"] for c in all_contrasts),
            "all_zero":sum(c["zero_delta_mse"] for c in all_contrasts),
            "informative_true":sum(c["true_delta_mse"] for c in informative),
            "informative_zero":sum(c["zero_delta_mse"] for c in informative),
        },
        "snapshots_with_five_physical_branches":snapshots,
    }


def summarize(rows):
    trials=sum(r["physically_executed_3step_trials"] for r in rows)
    snapshots=sum(r["actual_snapshot_count"] for r in rows)
    contrasts=sum(r["forced_vs_natural_contrasts"] for r in rows)
    ninfo=sum(r["informative_actual_contrasts"] for r in rows)
    errors={
        h:sum(r["sum_real_intervention_mse"][h] for r in rows)
        for h in ("true","blind","yoked","persistence")
    }
    effects={
        k:sum(r["real_contrast_sums"][k] for r in rows)
        for k in ("all_true","all_zero","informative_true","informative_zero")
    }
    return {
        "real_original_lives":len(rows),
        "natural_lives_ended_before_240":sum(
            r["natural_life_real_frames"]<HORIZON for r in rows
        ),
        "actual_natural_frames":sum(
            r["natural_life_real_frames"] for r in rows
        ),
        "real_fixed_snapshot_count":snapshots,
        "actual_three_step_forced_physical_branches":trials,
        "real_forced_vs_natural_contrasts":contrasts,
        "informative_real_action_contrasts":ninfo,
        "informative_ratio":round(ninfo/max(contrasts,1),6),
        "test_world_seeds_with_informative_contrasts":len({
            r["seed"] for r in rows if r["informative_actual_contrasts"]>0
        }),
        "physically_divergent_3step_steering_snapshots":sum(
            r["divergent_forced_left_right_snapshots"] for r in rows
        ),
        "three_step_divergence_ratio":round(
            sum(r["divergent_forced_left_right_snapshots"] for r in rows)
            /max(snapshots,1),6
        ),
        "actual_physically_early_dead_forced_clones":sum(
            r["early_death_forced_clones"] for r in rows
        ),
        "one_real_terminal_3step_sensor_mse":{
            h:round(v/max(trials,1),12) for h,v in errors.items()
        },
        "physical_sensor_delta_mse_all":{
            "true":round(effects["all_true"]/max(contrasts,1),12),
            "zero":round(effects["all_zero"]/max(contrasts,1),12),
        },
        "physical_sensor_delta_mse_informative":{
            "true":round(effects["informative_true"]/max(ninfo,1),12),
            "zero":round(effects["informative_zero"]/max(ninfo,1),12),
        },
    }


def _audit(train_rows,test_rows,trained_heads,head_hash_before_test,head_hash_after_test):
    physics=True
    pairing=True
    causal=True
    updates=True
    models=r4.ActionSensorHeads(10500)
    total_updates=0
    train_snapshots=0
    for group in (train_rows,test_rows):
        phase="train" if group is train_rows else "test"
        for life in group:
            world=World(life["seed"],life["scene"])
            brain=r2.MotorLearner(life["seed"],"recurrent_frozen")
            if brain.circuit.weights!=life["initial_neural_434_coefficients"]:
                pairing=False
            if life["initial_neural_434_coefficients"]!=life["final_neural_434_coefficients"]:
                updates=False
            saved={s["tick"]:s for s in life["snapshots_with_five_physical_branches"]}
            if len(saved)!=len(life["snapshots_with_five_physical_branches"]):
                causal=False
            for natural in life["real_natural_life"]:
                tick=natural["tick"]
                pre=tuple(world.perceive())
                raw,h,inp,noise,command=brain.command(pre)
                if not (
                    tick==world.frame and inp==tuple(natural[
                        "before_local_eight_sensor_values"
                    ]) and list(h)==natural["before_16_neural_values"]
                    and list(command)==natural["executed_natural_motor"]
                    and list(noise)==natural["seeded_motor_perturbation"]
                ):
                    causal=False
                if tick in saved:
                    snapshot=saved[tick]
                    train_snapshots+=int(phase=="train")
                    if not (
                        snapshot["pre_body_state"]==vars(world.body) and
                        snapshot["pre_16_recurrent_neurons"]==list(h) and
                        snapshot["pre_eight_local_sensors"]==list(pre) and
                        snapshot["natural_executed_motor"]==list(command)
                        and snapshot["all_actions_experimenter_forced"]
                    ):
                        causal=False
                    if snapshot["head_hash_before_snapshot"]!=sha(models.weights):
                        updates=False
                    candidate=forced_commands(command,phase)
                    trials=snapshot["trials"]
                    if len(trials)!=5 or [t["label"] for t in trials]!=list(ACTIONS):
                        physics=False
                    actual_motor=[candidate[action] for action in ACTIONS]
                    for i,trial in enumerate(trials):
                        motor=actual_motor[i]
                        shuffled=actual_motor[(i+1)%5]
                        if (trial["forced_real_motor"]!=list(motor)
                            or trial["same_state_rotated_training_motor"]!=list(shuffled)
                        ):
                            causal=False
                        rays=pre[1:4]
                        f={
                            "true":predictor_features(h,rays,motor),
                            "blind":predictor_features(h,rays,(0.,0.)),
                            "yoked":predictor_features(h,rays,(
                                shuffled if phase=="train" else motor
                            )),
                        }
                        preds={
                            k:models.predict(k,rays,f[k])
                            for k in HEADS
                        }|{"persistence":list(rays)}
                        if preds!=trial["head_predictions_before_executing"]:
                            causal=False
                        clone=copy.deepcopy(world)
                        steps=[]
                        for j in range(BRANCH_STEPS):
                            if clone.body.energy<=0 or clone.body.integrity<=0:
                                break
                            o=clone.integrate(motor)
                            steps.append({
                                "physics":o,
                                "observed_wall_rays":list(clone.perceive()[1:4]),
                            })
                        if (steps!=trial["actual_physical_steps"] or
                            trial["actual_three_step_terminal_wall_rays"]!=
                            steps[-1]["observed_wall_rays"] or
                            trial["real_integrations"]!=len(steps) or
                            trial["early_absorbing_death"]!=(len(steps)<BRANCH_STEPS)
                            or trial["terminal_body"]!=vars(clone.body)):
                            physics=False
                        for key,pred in preds.items():
                            if abs(mse(pred,steps[-1]["observed_wall_rays"])-
                                   trial["heldout_actual_mse"][key])>1e-10:
                                causal=False
                        if phase=="train":
                            models.train_all(f,{k:preds[k] for k in HEADS},
                                             steps[-1]["observed_wall_rays"])
                            total_updates+=1
                    if sha(models.weights)!=snapshot["head_hash_after_snapshot"]:
                        updates=False
                    base=trials[0]
                    for trial,contrast in zip(trials[1:],snapshot["contrasts"]):
                        delta=[
                            a-b for a,b in zip(
                                trial["actual_three_step_terminal_wall_rays"],
                                base["actual_three_step_terminal_wall_rays"],
                            )
                        ]
                        pred_delta=[
                            a-b for a,b in zip(
                                trial["head_predictions_before_executing"]["true"],
                                base["head_predictions_before_executing"]["true"],
                            )
                        ]
                        if not(
                            delta==contrast["measured_actual_wall_delta"]
                            and pred_delta==contrast["predicted_true_head_wall_delta"]
                            and contrast["informative_physical_effect"]==(
                                max(abs(v) for v in delta)>=INFORMATIVE_THRESHOLD
                            )
                            and abs(mse(delta,pred_delta)-contrast["true_delta_mse"])<1e-12
                            and abs(mse(delta,(0.,0.,0.))-contrast["zero_delta_mse"])<1e-12
                        ):
                            causal=False
                outcome=world.integrate(command)
                if (outcome!=natural["physical_outcome"]
                    or list(world.perceive()[1:4])!=natural["real_next_three_wall_rays"]):
                    physics=False
                if tick in saved and (
                    saved[tick]["trials"][0]["actual_physical_steps"][0]["physics"]!=outcome
                ):
                    physics=False
            if len(life["real_natural_life"])!=life["natural_life_real_frames"]:
                physics=False
    if models.weights!=trained_heads or sha(models.weights)!=head_hash_after_test:
        updates=False
    if head_hash_before_test!=head_hash_after_test:
        updates=False
    return {
        "independently_replayed_original_and_all_physical_clones":physics,
        "paired_initial_434_coefficients_and_world_physics":pairing,
        "preaction_16_neurons_eight_senses_and_true_blind_shuffled_inputs":causal,
        "exact_training_only_LMS_updates_and_test_weights_frozen":updates,
        "recomputed_training_trial_updates":total_updates,
        "train_snapshots_audited":train_snapshots,
    }


def _bootstrap(test_rows,a,b,n=1000):
    by_seed={
        s:[r for r in test_rows if r["seed"]==s]
        for s in TEST_SEEDS
    }
    diffs=[]
    for seed in TEST_SEEDS:
        group=by_seed[seed]
        ntrials=sum(r["physically_executed_3step_trials"] for r in group)
        if ntrials==0:
            return {"unavailable_due_to_dead_organism":True}
        diffs.append(
            sum(r["sum_real_intervention_mse"][a]-
                r["sum_real_intervention_mse"][b] for r in group)
            /ntrials
        )
    rng=random.Random(106070)
    draws=sorted(mean(
        diffs[rng.randrange(len(diffs))] for _ in diffs
    ) for _ in range(n))
    return {
        "heldout_paired_seed_mean_delta":round(mean(diffs),10),
        "exploratory_world_seed_bootstrap95":[
            round(draws[25],10),round(draws[974],10)
        ],
        "independent_world_seed_count":24,
    }


def run():
    heads=r4.ActionSensorHeads(TRAIN_SEEDS[0])
    zero=copy.deepcopy(heads.weights)
    train_rows=[
        collect_life(seed,scene,"train",heads)
        for scene in SCENES for seed in TRAIN_SEEDS
    ]
    trained=copy.deepcopy(heads.weights)
    before_test=sha(heads.weights)
    test_rows=[
        collect_life(seed,scene,"test",heads)
        for scene in SCENES for seed in TEST_SEEDS
    ]
    after_test=sha(heads.weights)
    train_summary=summarize(train_rows)
    test_summary=summarize(test_rows)
    test_scene={
        scene:summarize([r for r in test_rows if r["scene"]==scene])
        for scene in SCENES
    }
    audit=_audit(train_rows,test_rows,trained,before_test,after_test)
    all_rows=train_rows+test_rows
    maps={
        phase:{
            scene:len({
                sha(r["exogenous_ten_rectangles"])
                for r in rows if r["scene"]==scene
            })
            for scene in SCENES
        } for phase,rows in (("train",train_rows),("test",test_rows))
    }
    valid_maps=all(
        len(row["exogenous_ten_rectangles"])==10
        and all(not(x1-1.1<2.2<x2+1.1 and
                    y1-1.1<2.2<y2+1.1)
                for x1,y1,x2,y2 in row["exogenous_ten_rectangles"])
        for row in all_rows
    )
    prior_stop_flags={
        "autonomous_control_demonstrated":False,
        "food_search_demonstrated":False,
        "escape_reflex_learned":False,
        "autonomous_goal_invention_demonstrated":False,
        "general_intelligence_demonstrated":False,
        "biological_fly_replication_demonstrated":False,
        "evolution_authorized":False,
        "advance_to_evolution":False,
    }
    base=test_summary["one_real_terminal_3step_sensor_mse"]
    info=test_summary["physical_sensor_delta_mse_informative"]
    gates={
        "01_96_lives_960_fixed_snapshots_4800_real_forced_and_repeat":
            len(all_rows)==96 and
            train_summary["real_original_lives"]==48
            and test_summary["real_original_lives"]==48
            and train_summary["natural_lives_ended_before_240"]==0
            and test_summary["natural_lives_ended_before_240"]==0
            and train_summary["real_fixed_snapshot_count"]==480
            and test_summary["real_fixed_snapshot_count"]==480
            and train_summary["actual_three_step_forced_physical_branches"]==2400
            and test_summary["actual_three_step_forced_physical_branches"]==2400
            and train_summary["real_forced_vs_natural_contrasts"]==1920
            and test_summary["real_forced_vs_natural_contrasts"]==1920,
        "02_actual_R1_physical_trajectories_collision_death_and_cloning":
            audit["independently_replayed_original_and_all_physical_clones"],
        "03_natural_clone_first_step_matches_original_actual_world":
            audit["independently_replayed_original_and_all_physical_clones"],
        "04_16_recurrent_units_8_local_sensors_434_frozen_brain_slots":
            audit["paired_initial_434_coefficients_and_world_physics"],
        "05_3x84_zero_start_exact_train_LMS_and_never_motor_learn":
            zero=={h:[[0.]*r4.FEATURES for _ in range(3)] for h in HEADS}
            and audit["exact_training_only_LMS_updates_and_test_weights_frozen"]
            and all(r["initial_neural_434_coefficients"]==
                    r["final_neural_434_coefficients"] for r in all_rows)
            and all(heads.updates[h]==
                train_summary["actual_three_step_forced_physical_branches"]
                for h in HEADS),
        "06_blind_zero_motors_shuffled_real_same_state_action_distribution":
            audit["preaction_16_neurons_eight_senses_and_true_blind_shuffled_inputs"],
        "07_heldout_seed_and_action_magnitudes_disjoint_heads_frozen":
            set(TRAIN_SEEDS).isdisjoint(TEST_SEEDS)
            and forced_commands((.43,-.17),"train")!=
                forced_commands((.43,-.17),"test")
            and before_test==after_test,
        "08_five_real_forced_per_available_snapshot_predictions_preaction":
            audit["independently_replayed_original_and_all_physical_clones"]
            and audit["preaction_16_neurons_eight_senses_and_true_blind_shuffled_inputs"],
        "09_unchanged_valid_train_test_independent_obstacle_maps":
            valid_maps and all(
                maps[phase][scene]>=20
                for phase in ("train","test") for scene in SCENES
            ),
        "10_all_raw_models_physics_stop_flags_and_exact_repeat_required":
            len(all_rows)==96 and all(
                r["real_natural_life"] and r["snapshots_with_five_physical_branches"]
                for r in all_rows
            ) and all(v is False for v in prior_stop_flags.values()),
        "11_heldout_three_step_real_left_right_effect_ge_80pct":
            test_summary["three_step_divergence_ratio"]>=.80,
        "12_heldout_real_sensor_effect_min200_20_seeds_50_each_scene":
            test_summary["informative_real_action_contrasts"]>=200
            and test_summary["test_world_seeds_with_informative_contrasts"]>=20
            and all(test_scene[scene]["informative_real_action_contrasts"]>=50
                    for scene in SCENES),
        "13_heldout_true_mse_le_90pct_motor_blind":
            base["blind"]>0 and base["true"]<=.90*base["blind"],
        "14_heldout_true_mse_le_90pct_full_capacity_shuffled":
            base["yoked"]>0 and base["true"]<=.90*base["yoked"],
        "15_heldout_true_mse_le_85pct_persistence":
            base["persistence"]>0 and base["true"]<=.85*base["persistence"],
        "16_heldout_real_action_delta_true_mse_le_80pct_zero":
            test_summary["informative_real_action_contrasts"]>=200
            and info["zero"]>0 and info["true"]<=.80*info["zero"],
        "17_both_heldout_geometries_action_delta_true_mse_le_90pct_zero":
            all(
                test_scene[scene]["informative_real_action_contrasts"]>=50
                and test_scene[scene]["physical_sensor_delta_mse_informative"]["zero"]>0
                and test_scene[scene]["physical_sensor_delta_mse_informative"]["true"]<=
                    .90*test_scene[scene]["physical_sensor_delta_mse_informative"]["zero"]
                for scene in SCENES
            ),
    }
    return {
        "name":"FlyLoop R6 experimenter-trained real action effects, transfer to unseen worlds and motor amplitudes",
        "protocol":"docs/FLYLOOP_R6_PROTOCOL_2026-11-24.md",
        "metadata":{
            "train_seeds":TRAIN_SEEDS,
            "heldout_seeds":TEST_SEEDS,
            "scenes":SCENES,
            "snapshot_ticks":SNAPSHOT_TICKS,
            "forced_physical_frames_per_trial":BRANCH_STEPS,
            "training_candidate_commands":forced_commands((.43,-.17),"train"),
            "heldout_novel_candidate_commands":forced_commands((.43,-.17),"test"),
            "original_16_recurrent_neurons":r1.N,
            "original_neural_coefficients":r1.PARAMS,
            "external_predictor_heads":HEADS,
            "coefficients_per_head":r4.HEAD_PARAMS,
            "predictor_learning_rate":r4.HEAD_LR,
            "no_policy_or_motor_training":True,
            "predictor_weights_transferred_between_worlds_external_not_organic_learning":True,
            "no_food_map_or_evolution":True,
            "original_r1_r5_and_main_unchanged":True,
            "unique_world_maps":maps,
        },
        "training_summary":train_summary,
        "heldout_summary":test_summary,
        "heldout_scene_summaries":test_scene,
        "zero_initialized_predictor_heads":zero,
        "frozen_trained_predictor_heads":trained,
        "frozen_trained_heads_sha256":sha(trained),
        "after_heldout_test_heads_sha256":after_test,
        "independent_replay_and_learning_audit":audit,
        "exploratory_heldout_seed_bootstrap":{
            "actual_minus_motor_blind":_bootstrap(test_rows,"true","blind"),
            "actual_minus_shuffled":_bootstrap(test_rows,"true","yoked"),
        },
        "gates":gates,
        "gates_passed":sum(gates.values()),
        "screen_pass":all(gates.values()),
        **prior_stop_flags,
        "raw_rows_sha256":sha(all_rows),
        "raw_rows":all_rows,
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path,default=Path(
        "results/flyloop_r6/r6_intervention_trained_transfer.json"
    ))
    args=ap.parse_args()
    value=run()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(
        value,sort_keys=True,separators=(",",":"))+"\n",encoding="utf-8")
    print("FLYLOOP_R6 "+json.dumps({
        k:value[k] for k in (
            "name","training_summary","heldout_summary",
            "heldout_scene_summaries","independent_replay_and_learning_audit",
            "gates","gates_passed","screen_pass",
            "exploratory_heldout_seed_bootstrap","raw_rows_sha256"
        )
    },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
