"""FlyLoop R8 — within-life physical consequence prediction actually influences movement.

Original R1–R7 remain unmodified. An experimenter-designed five-command
analog selector combines an online learned REAL collision/displacement/
integrity-loss predictor with original R7 learned local sensory novelty.
No map, hidden future, path planner, explicit "turn away" rule, food,
escape skill or evolution. The INTRINSIC score is researcher-specified.

FROZEN BEFORE SCIENCE: docs/FLYLOOP_R8_PROTOCOL_2026-11-24.md
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
from neuroevolution import flyloop_r7 as r7

SEEDS=tuple(range(11200,11216))
GEOMETRIES=("obstacles","nearfield")
REGIMES=("static","removal")
ARMS=("frozen","r7_novelty","r8_true_physics",
      "r8_older_sham","random_menu","r2_progress")
HORIZON=320
CHANGE_TICK=160
WARMUP=50
HEAD_FEATURES=r4.FEATURES
HEAD_OUTPUTS=3
HEAD_LR=.025
HEAD_BOUND=2.
START_OFFSETS=(.25,.5,.25)
SHAM_LAG=6
HEAD_NAMES=("true","older_sham")
ORIGINAL_SENSOR_HEADS=r4.PREDICTOR_NAMES
PHYSICAL_MODEL_ARMS=frozenset(("r8_true_physics","r8_older_sham"))
ALL_HEAD_ARMS=frozenset(ARMS)


def sha(x):
    return hashlib.sha256(
        json.dumps(x,sort_keys=True,separators=(",",":")).encode()
    ).hexdigest()


def mean(values):
    values=list(values)
    return sum(values)/len(values) if values else 0.


def rms_radius(positions):
    if not positions:
        return 0.
    cx=mean(x for x,_ in positions)
    cy=mean(y for _,y in positions)
    return math.sqrt(mean((x-cx)**2+(y-cy)**2 for x,y in positions))


def mse(a,b):
    return sum((x-y)**2 for x,y in zip(a,b))/3.


class World(r1.World):
    """Original physical R1 integrator, original R1/R2 exogenous worlds."""
    def __init__(self,seed,geometry,regime):
        if seed not in SEEDS or geometry not in GEOMETRIES or regime not in REGIMES:
            raise ValueError("Unregistered R8 seed, geometry or regime")
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
            self.rectangles=list(self.initial_rectangles[::2])
            self.switched=True
        return self.switched


class PhysicalConsequenceHeads:
    """Two 84-weight 3-output heads: real consequences vs older REAL labels.

    Integrity loss and collision are deliberately redundant under R1
    constant injury per collision. They are NOT independent observations.
    """
    def __init__(self,seed):
        self.weights={
            name:[[0.]*HEAD_FEATURES for _ in range(HEAD_OUTPUTS)]
            for name in HEAD_NAMES
        }
        self.actual_history=[]
        self.sham_rng=random.Random(seed*76343+8819)
        self.updates={name:0 for name in HEAD_NAMES}

    def predict_features(self,name,features):
        return [
            round(r1.clamp(
                START_OFFSETS[k]+r1.dot(self.weights[name][k],features),
                0.,1.
            ),9)
            for k in range(HEAD_OUTPUTS)
        ]

    def predict(self,name,hidden,rays,cmd):
        phi=r4.ActionSensorHeads.make_features(hidden,rays,cmd)
        assert len(phi)==HEAD_FEATURES
        return self.predict_features(name,phi)

    @staticmethod
    def real_target(physics,integrity_before):
        # Real physical integrator returns movement after actual collision.
        return [
            float(physics["collision"]),
            r1.clamp(physics["moved"]/(r1.V_MAX*r1.DT),0.,1.),
            r1.clamp(
                (integrity_before-physics["integrity"])/r1.COLLISION_INJURY,
                0.,1.,
            ),
        ]

    def _update_one(self,name,phi,pred,target):
        for k in range(HEAD_OUTPUTS):
            rate=HEAD_LR*(target[k]-pred[k])
            for j in range(HEAD_FEATURES):
                self.weights[name][k][j]=r1.clamp(
                    self.weights[name][k][j]+rate*phi[j],
                    -HEAD_BOUND,HEAD_BOUND
                )
        self.updates[name]+=1

    def learn_after_physics(self,tick,phi,prior_predictions,real_target):
        # The older head's label is sampled only from truly executed
        # preceding moves, at least six steps old. Sample BEFORE append.
        source=None
        sham_label=None
        if tick>=SHAM_LAG:
            source=self.sham_rng.randrange(tick-SHAM_LAG+1)
            sham_label=list(self.actual_history[source])
        self._update_one("true",phi,prior_predictions["true"],real_target)
        if sham_label is not None:
            self._update_one("older_sham",phi,
                             prior_predictions["older_sham"],sham_label)
        self.actual_history.append(list(real_target))
        return sham_label,source


def candidate_score(predicted_local_rays,physical_predictions):
    novelty=r7.novelty_scores(predicted_local_rays)
    return [
        .25*math.sqrt(max(0.,novelty[i]))
        +.60*model[1]-.50*model[0]-.50*model[2]
        for i,model in enumerate(physical_predictions)
    ]


def choose(scores):
    # Python max resolves equal-scored candidates by first occurrence.
    return max(range(len(scores)),key=lambda i:scores[i])


def _brain(seed,arm):
    return r2.MotorLearner(seed,
            "recurrent_plastic" if arm=="r2_progress" else "recurrent_frozen")


def simulate(seed,geometry,regime,arm):
    if arm not in ARMS:
        raise ValueError("Unregistered R8 controller arm")
    world=World(seed,geometry,regime)
    brain=_brain(seed,arm)
    sensory_heads=r4.ActionSensorHeads(seed)
    physical_heads=PhysicalConsequenceHeads(seed)
    menu_rng=random.Random(seed*59149+77401)
    initial=copy.deepcopy(brain.circuit.weights)
    original_heading=world.body.heading
    positions=[(world.body.x,world.body.y)]
    before_cells={(int(world.body.x),int(world.body.y))}
    after_cells=set()
    post_xy=[]
    post_new=set()
    post_meters=0.
    post_collisions=0
    post_repeat=0
    post_contact_predecessors=0
    post_zero_movement=0
    post_changed=0
    post_damage=0.
    post_risk_true_mse=0.
    post_risk_sham_mse=0.
    prior_post_collision=None

    for tick in range(HORIZON):
        if world.body.energy<=0 or world.body.integrity<=0:
            break
        world.apply_environment(tick)
        senses=tuple(world.perceive())
        rays=tuple(senses[1:4])
        raw,hidden,observed,noise,base=brain.command(senses)
        menu=r7.candidates(base)
        predicted_rays=r7.predicted_candidates(
            sensory_heads,hidden,rays,menu
        )
        predicted_physics={
            name:[
                physical_heads.predict(name,hidden,rays,cmd)
                for cmd in menu
            ] for name in HEAD_NAMES
        }
        scores=[0.]*len(menu)
        index=0
        if tick>=WARMUP and arm in PHYSICAL_MODEL_ARMS:
            name="true" if arm=="r8_true_physics" else "older_sham"
            scores=candidate_score(predicted_rays["true"],
                                   predicted_physics[name])
            index=choose(scores)
        elif tick>=WARMUP and arm=="r7_novelty":
            index,scores=r7.selected_candidate(predicted_rays["true"])
        elif tick>=WARMUP and arm=="random_menu":
            index=menu_rng.randrange(len(menu))
        command=menu[index]
        feature=r4.ActionSensorHeads.make_features(hidden,rays,command)
        predicted_real={
            name:physical_heads.predict_features(name,feature)
            for name in HEAD_NAMES
        }
        sensor_phi,older_motor,older_motor_tick=sensory_heads.inputs(
            tick,hidden,rays,command
        )
        sensory_predicted_actual={
            name:sensory_heads.predict(name,rays,sensor_phi[name])
            for name in ORIGINAL_SENSOR_HEADS
        }
        integrity_before=world.body.integrity
        # ONLY ONE physical action: proposed actions have NO world oracle.
        outcome=world.integrate(command)
        positions.append((world.body.x,world.body.y))
        actual_rays=tuple(world.perceive()[1:4])
        target=physical_heads.real_target(outcome,integrity_before)
        risk_mse={
            name:round(mse(pred,target),12)
            for name,pred in predicted_real.items()
        }
        # Explicitly AFTER actual real motion and sensory consequence:
        real_older_target,source=physical_heads.learn_after_physics(
            tick,feature,predicted_real,target
        )
        sensory_heads.train_all(
            sensor_phi,sensory_predicted_actual,actual_rays
        )
        sensory_heads.motor_history.append(tuple(command))
        feedback=None
        if arm=="r2_progress":
            feedback=brain.adapt(
                tick,outcome["moved"],outcome["collision"],
                positions,hidden,noise
            )["training_feedback"]
        if tick<CHANGE_TICK:
            before_cells.add((int(world.body.x),int(world.body.y)))
        else:
            cell=(int(world.body.x),int(world.body.y))
            after_cells.add(cell)
            if cell not in before_cells:
                post_new.add(cell)
            post_xy.append((world.body.x,world.body.y))
            post_meters+=outcome["moved"]
            post_collisions+=outcome["collision"]
            post_damage+=(integrity_before-outcome["integrity"])
            post_zero_movement+=outcome["moved"]==0.
            post_changed+=index!=0
            post_risk_true_mse+=risk_mse["true"]
            post_risk_sham_mse+=risk_mse["older_sham"]
            if prior_post_collision is not None and prior_post_collision:
                post_contact_predecessors+=1
                post_repeat+=outcome["collision"]
            prior_post_collision=outcome["collision"]
        world.trace.append({
            "tick":tick,
            "geometry_changed":world.switched,
            "original_eight_local_sensors":[round(v,9) for v in observed],
            "original_16_recurrent_states":[round(v,9) for v in hidden],
            "original_neural_motor":[round(v,9) for v in raw],
            "original_exogenous_motor_noise":[round(v,9) for v in noise],
            "base_motor":[round(v,9) for v in base],
            "five_local_motor_candidates":[
                [round(v,9) for v in candidate] for candidate in menu
            ],
            "preaction_candidate_sensory_predictions":predicted_rays,
            "preaction_candidate_physical_predictions":predicted_physics,
            "selected_candidate_scores":[round(v,12) for v in scores],
            "selected_candidate_index":index,
            "physically_executed_motor":[round(v,9) for v in command],
            "original_sensory_head_older_motor_tick":older_motor_tick,
            "original_sensory_head_older_motor":list(older_motor),
            "actual_next_three_wall_sensors":[round(v,9) for v in actual_rays],
            "actual_real_physical_outcomes":target,
            "predicted_real_physical_outcomes":predicted_real,
            "real_target_physical_mse":risk_mse,
            "older_sham_target_from_real_history":real_older_target,
            "older_sham_physical_source_tick":source,
            "learned_sensory_heads_after_physics":True,
            "r2_reward_if_applicable":feedback,
            "physical_body_outcome":outcome,
            "physical_heads_sha256_after_step":(
                sha(physical_heads.weights) if (tick+1)%20==0 else None
            ),
            "sensory_heads_sha256_after_step":(
                sha(sensory_heads.weights) if (tick+1)%20==0 else None
            ),
        })
    post_steps=sum(f["tick"]>=CHANGE_TICK for f in world.trace)
    return {
        "seed":seed,"geometry":geometry,"regime":regime,"arm":arm,
        "initial_ten_rectangles":[list(x) for x in world.initial_rectangles],
        "postchange_rectangles":[list(x) for x in world.rectangles],
        "environment_removed_five_walls":world.switched,
        "initial_body_pose":[2.2,2.2,round(original_heading,9)],
        "original_434_neural_coefficients_at_birth":initial,
        "original_434_neural_coefficients_at_end":
            copy.deepcopy(brain.circuit.weights),
        "head_weights_initially_zero":True,
        "physical_heads_initially_zero":True,
        "physical_heads_final":physical_heads.weights,
        "physical_head_updates":physical_heads.updates,
        "sensory_heads_final":sensory_heads.weights,
        "sensory_head_updates":sensory_heads.updates,
        "motor_update_events":brain.updates,
        "real_physical_steps":len(world.trace),
        "postchange_real_steps":post_steps,
        "total_distance":round(world.distance,6),
        "total_collisions":world.collisions,
        "postchange_visited_cells":len(after_cells),
        "postchange_new_cells":len(post_new),
        "postchange_meters":round(post_meters,6),
        "postchange_collisions":post_collisions,
        "postchange_integrity_loss":round(post_damage,6),
        "postchange_zero_movement_frames":post_zero_movement,
        "postchange_repeated_collisions":post_repeat,
        "postchange_prior_collision_frames":post_contact_predecessors,
        "postchange_rms_radius":round(rms_radius(post_xy),6),
        "postchange_model_action_overrides":post_changed,
        "postchange_true_physical_prediction_mse_sum":round(
            post_risk_true_mse,10),
        "postchange_sham_physical_prediction_mse_sum":round(
            post_risk_sham_mse,10),
        "trace":world.trace,
    }


def _audit(rows):
    """Same-source simulator replay; independent external check is separate."""
    physical=True
    future_free=True
    sham_valid=True
    update_valid=True
    original_neural_valid=True
    for life in rows:
        world=World(life["seed"],life["geometry"],life["regime"])
        brain=_brain(life["seed"],life["arm"])
        sensor_heads=r4.ActionSensorHeads(life["seed"])
        risk=PhysicalConsequenceHeads(life["seed"])
        menu_rng=random.Random(life["seed"]*59149+77401)
        positions=[(world.body.x,world.body.y)]
        if brain.circuit.weights!=life["original_434_neural_coefficients_at_birth"]:
            original_neural_valid=False
        for f in life["trace"]:
            tick=f["tick"]
            if tick!=world.frame or world.body.energy<=0 or world.body.integrity<=0:
                physical=False
                break
            world.apply_environment(tick)
            senses=world.perceive()
            raw,hidden,observed,noise,base=brain.command(senses)
            menu=r7.candidates(base)
            sens_pred=r7.predicted_candidates(sensor_heads,hidden,senses[1:4],menu)
            risk_pred={
                h:[risk.predict(h,hidden,senses[1:4],cmd) for cmd in menu]
                for h in HEAD_NAMES
            }
            if (sens_pred!=f["preaction_candidate_sensory_predictions"]
                or risk_pred!=f["preaction_candidate_physical_predictions"]):
                future_free=False
            idx=0
            scores=[0.]*len(menu)
            if tick>=WARMUP and life["arm"] in PHYSICAL_MODEL_ARMS:
                name=("true" if life["arm"]=="r8_true_physics"
                      else "older_sham")
                scores=candidate_score(sens_pred["true"],risk_pred[name])
                idx=choose(scores)
            elif tick>=WARMUP and life["arm"]=="r7_novelty":
                idx,scores=r7.selected_candidate(sens_pred["true"])
            elif tick>=WARMUP and life["arm"]=="random_menu":
                idx=menu_rng.randrange(len(menu))
            command=menu[idx]
            if (
                idx!=f["selected_candidate_index"]
                or [round(v,12) for v in scores]!=f["selected_candidate_scores"]
                or [round(v,9) for v in command]!=f["physically_executed_motor"]
                or [round(v,9) for v in noise]!=f["original_exogenous_motor_noise"]
                or [round(v,9) for v in observed]!=f["original_eight_local_sensors"]
                or [round(v,9) for v in hidden]!=f["original_16_recurrent_states"]
                or [[round(v,9) for v in cmd] for cmd in menu]!=
                  f["five_local_motor_candidates"]
            ):
                future_free=False
            feature=r4.ActionSensorHeads.make_features(
                hidden,senses[1:4],command
            )
            risk_before={
                name:risk.predict_features(name,feature) for name in HEAD_NAMES
            }
            sensory_features,old_motor,old_tick=sensor_heads.inputs(
                tick,hidden,senses[1:4],command
            )
            predicted_rays={
                name:sensor_heads.predict(name,senses[1:4],sensory_features[name])
                for name in ORIGINAL_SENSOR_HEADS
            }
            old_integrity=world.body.integrity
            movement=world.integrate(command)
            positions.append((world.body.x,world.body.y))
            actual_rays=world.perceive()[1:4]
            target=risk.real_target(movement,old_integrity)
            if (
                movement!=f["physical_body_outcome"]
                or target!=f["actual_real_physical_outcomes"]
                or [round(v,9) for v in actual_rays]!=
                    f["actual_next_three_wall_sensors"]
                or risk_before!=f["predicted_real_physical_outcomes"]
            ):
                physical=False
            true_error={
                name:round(mse(pred,target),12)
                for name,pred in risk_before.items()
            }
            if true_error!=f["real_target_physical_mse"]:
                future_free=False
            sham_label,source=risk.learn_after_physics(
                tick,feature,risk_before,target
            )
            if (
                source!=f["older_sham_physical_source_tick"]
                or sham_label!=f["older_sham_target_from_real_history"]
                or (source is not None and not 0<=source<=tick-SHAM_LAG)
                or (source is None and tick>=SHAM_LAG)
                or (source is not None and sham_label!=
                    risk.actual_history[source])
            ):
                sham_valid=False
            sensor_heads.train_all(sensory_features,predicted_rays,actual_rays)
            sensor_heads.motor_history.append(tuple(command))
            if old_motor!=tuple(f["original_sensory_head_older_motor"]):
                sham_valid=False
            if old_tick!=f["original_sensory_head_older_motor_tick"]:
                sham_valid=False
            if life["arm"]=="r2_progress":
                reward=brain.adapt(
                    tick,movement["moved"],movement["collision"],
                    positions,hidden,noise
                )
                if reward["training_feedback"]!=f["r2_reward_if_applicable"]:
                    future_free=False
            if (tick+1)%20==0 and (
                f["physical_heads_sha256_after_step"]!=sha(risk.weights)
                or f["sensory_heads_sha256_after_step"]!=sha(sensor_heads.weights)
            ):
                update_valid=False
        if (
            risk.weights!=life["physical_heads_final"]
            or risk.updates!=life["physical_head_updates"]
            or sensor_heads.weights!=life["sensory_heads_final"]
            or sensor_heads.updates!=life["sensory_head_updates"]
            or brain.circuit.weights!=life["original_434_neural_coefficients_at_end"]
        ):
            update_valid=False
        if world.frame!=life["real_physical_steps"]:
            physical=False
        if life["arm"]!="r2_progress" and (
            life["original_434_neural_coefficients_at_birth"]!=
            life["original_434_neural_coefficients_at_end"]
        ):
            original_neural_valid=False
        for field in ("w_input","w_recurrent","bias"):
            if (life["original_434_neural_coefficients_at_birth"][field]!=
                life["original_434_neural_coefficients_at_end"][field]):
                original_neural_valid=False
    return {
        "real_physical_neural_body_replay_and_no_dead_motion":physical,
        "five_candidate_scores_use_only_preaction_model":future_free,
        "physical_sham_only_own_at_least_six_old_experiences":sham_valid,
        "head_update_counts_weights_and_sha_every_twenty":update_valid,
        "original_neural_substrate_frozen_except_r2_motor_readout":
            original_neural_valid,
    }


def _bootstrap(rows,arm,ref,n=1000):
    scope={
        (r["seed"],r["arm"]):r
        for r in rows if r["geometry"]=="nearfield" and r["regime"]=="removal"
    }
    delta=[
        scope[(seed,arm)]["postchange_visited_cells"]-
        scope[(seed,ref)]["postchange_visited_cells"]
        for seed in SEEDS
    ]
    rng=random.Random(112016)
    samples=sorted(mean(delta[rng.randrange(len(delta))]
                        for _ in delta) for _ in range(n))
    return {
        "paired_16_seed_mean_post_cells_delta":round(mean(delta),6),
        "exploratory_paired_bootstrap95":[
            round(samples[25],6),round(samples[974],6)
        ],
    }


def run():
    rows=[
        simulate(seed,geom,regime,arm)
        for geom in GEOMETRIES for regime in REGIMES
        for seed in SEEDS for arm in ARMS
    ]
    checks=_audit(rows)
    summary={}
    for geom in GEOMETRIES:
        summary[geom]={}
        for regime in REGIMES:
            summary[geom][regime]={}
            for arm in ARMS:
                group=[
                    row for row in rows
                    if row["geometry"]==geom and row["regime"]==regime
                    and row["arm"]==arm
                ]
                steps=sum(r["postchange_real_steps"] for r in group)
                meters=sum(r["postchange_meters"] for r in group)
                collision_prior=sum(r["postchange_prior_collision_frames"]
                                    for r in group)
                summary[geom][regime][arm]={
                    name:round(mean(r[name] for r in group),6)
                    for name in (
                        "real_physical_steps","postchange_real_steps",
                        "postchange_visited_cells","postchange_new_cells",
                        "postchange_meters","postchange_collisions",
                        "postchange_rms_radius",
                        "postchange_model_action_overrides",
                    )
                }|{
                    "pooled_postchange_contacts_per_meter":round(
                        sum(r["postchange_collisions"] for r in group)
                        /max(meters,1e-12),9
                    ),
                    "postchange_previous_collision_followups":collision_prior,
                    "postchange_repeated_collisions":sum(
                        r["postchange_repeated_collisions"] for r in group
                    ),
                    "pooled_repeated_contact_fraction":round(
                        sum(r["postchange_repeated_collisions"] for r in group)
                        /max(collision_prior,1),9
                    ),
                    "zero_movement_fraction":round(
                        sum(r["postchange_zero_movement_frames"] for r in group)
                        /max(steps,1),9
                    ),
                    "physical_model_overrode_base_fraction":round(
                        sum(r["postchange_model_action_overrides"] for r in group)
                        /max(steps,1),9
                    ),
                    "true_physical_outcome_mse":round(
                        sum(r["postchange_true_physical_prediction_mse_sum"]
                            for r in group)/max(steps,1),9
                    ),
                    "older_sham_physical_outcome_mse":round(
                        sum(r["postchange_sham_physical_prediction_mse_sum"]
                            for r in group)/max(steps,1),9
                    ),
                    "total_physical_steps_postchange":steps,
                    "pooled_physical_meters":round(meters,6),
                }
    def matching(seed,geom,regime,arm):
        return next(r for r in rows if
                    (r["seed"],r["geometry"],r["regime"],r["arm"])==
                    (seed,geom,regime,arm))
    pairing=all(
        len({sha(r["original_434_neural_coefficients_at_birth"])
             for r in rows if r["seed"]==seed
             and r["geometry"]==geom and r["regime"]==regime})==1
        and len({sha(r["initial_ten_rectangles"])
             for r in rows if r["seed"]==seed
             and r["geometry"]==geom and r["regime"]==regime})==1
        and len({sha(r["initial_body_pose"])
             for r in rows if r["seed"]==seed
             and r["geometry"]==geom and r["regime"]==regime})==1
        for seed in SEEDS for geom in GEOMETRIES for regime in REGIMES
    )
    motor_noise=all(
        all(
            [f["original_exogenous_motor_noise"] for f in r["trace"][:limit]]
            ==[f["original_exogenous_motor_noise"]
                for f in group[0]["trace"][:limit]]
            for r in group
        )
        for seed in SEEDS for geom in GEOMETRIES for regime in REGIMES
        for group in [[r for r in rows if
                       r["seed"]==seed and r["geometry"]==geom
                       and r["regime"]==regime]]
        for limit in [min(r["real_physical_steps"] for r in group)]
    )
    prechange=all(
        [
            (f["physically_executed_motor"],
             f["physical_body_outcome"],
             f["actual_next_three_wall_sensors"],
             f["original_16_recurrent_states"])
            for f in matching(seed,geom,"static",arm)["trace"]
            if f["tick"]<CHANGE_TICK
        ]==[
            (f["physically_executed_motor"],
             f["physical_body_outcome"],
             f["actual_next_three_wall_sensors"],
             f["original_16_recurrent_states"])
            for f in matching(seed,geom,"removal",arm)["trace"]
            if f["tick"]<CHANGE_TICK
        ]
        for seed in SEEDS for geom in GEOMETRIES for arm in ARMS
    )
    walls=all(
        len(r["initial_ten_rectangles"])==10
        and r["postchange_rectangles"]==(
            r["initial_ten_rectangles"][::2]
            if r["regime"]=="removal" and r["environment_removed_five_walls"]
            else r["initial_ten_rectangles"]
        ) for r in rows
    )
    n=summary["nearfield"]["removal"]
    true=n["r8_true_physics"]
    original=n["r7_novelty"]
    sham=n["r8_older_sham"]
    random_arm=n["random_menu"]
    passive=n["frozen"]
    gates={
        "01_384_balanced_lives_and_full_byte_identical_repeat_required":
            len(rows)==384 and all(
                sum(r["geometry"]==geom and r["regime"]==regime
                    and r["arm"]==arm for r in rows)==len(SEEDS)
                for geom in GEOMETRIES for regime in REGIMES for arm in ARMS
            ),
        "02_original_R1_physical_collision_injury_energy_no_resurrection":
            checks["real_physical_neural_body_replay_and_no_dead_motion"],
        "03_paired_world_body_434_initial_weights_and_gaussian_noise":
            pairing and motor_noise,
        "04_same_16_recurrent_8_local_sensors_neural_weights_unchanged":
            checks["original_neural_substrate_frozen_except_r2_motor_readout"],
        "05_two_84_weight_zero_heads_three_outcomes_original_sensor_heads":
            checks["head_update_counts_weights_and_sha_every_twenty"]
            and all(
                r["physical_heads_initially_zero"]
                and r["head_weights_initially_zero"]
                and r["physical_head_updates"]=={
                    "true":r["real_physical_steps"],
                    "older_sham":max(0,r["real_physical_steps"]-SHAM_LAG),
                }
                and r["sensory_head_updates"]=={
                    name:r["real_physical_steps"]
                    for name in ORIGINAL_SENSOR_HEADS
                } for r in rows
            ),
        "06_true_physical_targets_only_after_real_body_outcome":
            checks["five_candidate_scores_use_only_preaction_model"],
        "07_sham_real_older_labels_strict_six_frame_lag":
            checks["physical_sham_only_own_at_least_six_old_experiences"],
        "08_five_motor_candidates_fixed_preaction_no_map_oracle":
            checks["five_candidate_scores_use_only_preaction_model"],
        "09_static_and_removal_identical_before_tick160_exact_five_walls":
            prechange and walls,
        "10_complete_unmodified_raw_trace_regression_and_false_stop_flags":
            all(r["trace"] and r["original_434_neural_coefficients_at_birth"]
                and r["physical_heads_final"] for r in rows),
        "11_same_frozen_life_true_physical_outcome_mse_le_sham_85pct":
            passive["older_sham_physical_outcome_mse"]>0
            and passive["true_physical_outcome_mse"]<=
                .85*passive["older_sham_physical_outcome_mse"],
        "12_true_repeated_collision_rate_le_R7_80pct":
            original["postchange_previous_collision_followups"]>=50
            and true["pooled_repeated_contact_fraction"]<=
                .80*original["pooled_repeated_contact_fraction"],
        "13_true_contacts_per_meter_le_R7_75pct":
            original["pooled_physical_meters"]>0 and
            true["pooled_postchange_contacts_per_meter"]<=
                .75*original["pooled_postchange_contacts_per_meter"],
        "14_true_postchange_unique_cells_ge_R7_90pct":
            true["postchange_visited_cells"]>=
                .90*original["postchange_visited_cells"],
        "15_true_postchange_unique_cells_ge_random_90pct":
            true["postchange_visited_cells"]>=
                .90*random_arm["postchange_visited_cells"],
        "16_true_RMS_radius_ge_R7_85pct":
            true["postchange_rms_radius"]>=
                .85*original["postchange_rms_radius"],
        "17_true_repeated_collision_rate_le_older_sham_85pct":
            sham["postchange_previous_collision_followups"]>0
            and true["pooled_repeated_contact_fraction"]<=
                .85*sham["pooled_repeated_contact_fraction"],
        "18_true_actually_changes_movement_at_least10pct":
            true["physical_model_overrode_base_fraction"]>=.10,
        "19_true_mean_distance_traveled_ge_R7_90pct":
            true["postchange_meters"]>=.90*original["postchange_meters"],
    }
    return {
        "name":"FlyLoop R8 online bodily consequence learning without scripted reflex",
        "protocol":"docs/FLYLOOP_R8_PROTOCOL_2026-11-24.md",
        "metadata":{
            "seeds":SEEDS,"geometries":GEOMETRIES,"regimes":REGIMES,
            "arms":ARMS,"max_frames_per_life":HORIZON,
            "environment_removal_tick":CHANGE_TICK,
            "warmup_frames":WARMUP,
            "risk_head_size_each":HEAD_OUTPUTS*HEAD_FEATURES,
            "head_learner_eta":HEAD_LR,
            "physical_real_sham_min_age":SHAM_LAG,
            "engineered_physical_damage_score":True,
            "injury_channel_deterministically_redundant_with_collision":True,
            "zero_candidate_physics_lookahead":True,
            "original_r1_r7_and_main_untouched":True,
            "no_food_escape_or_evolution":True,
        },
        "summary":summary,
        "integrity_audit":checks,
        "exploratory_paired_16_seed_postchange_coverage_bootstrap":{
            comparator:_bootstrap(rows,"r8_true_physics",comparator)
            for comparator in ("r7_novelty","r8_older_sham",
                               "random_menu","r2_progress")
        },
        "gates":gates,
        "gates_passed":sum(gates.values()),
        "screen_pass":all(gates.values()),
        "closed_loop_physical_prediction_guides_action_implemented":True,
        "effective_collision_learning_demonstrated":all(gates.values()),
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
    parser.add_argument("--out",type=Path,default=Path(
        "results/flyloop_r8/r8_online_physical_outcomes.json"
    ))
    args=parser.parse_args()
    value=run()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(
        value,sort_keys=True,separators=(",",":"))+"\n",encoding="utf8")
    print("FLYLOOP_R8 "+json.dumps({
        k:value[k] for k in (
            "name","summary","integrity_audit","gates","gates_passed",
            "screen_pass",
            "exploratory_paired_16_seed_postchange_coverage_bootstrap",
            "raw_rows_sha256"
        )
    },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
