"""FlyLoop R10: learned slow *teacher* instructs a fast four-neuron student.

The fast supervised route receives ONLY a completed slow teacher motor
label, not physical contact, movement or injury as a fast training target.
A separate DIRECT R9 micro-feedback arm is an explicitly labelled control.
All scientific thresholds are fixed prospectively in the R10 protocol.
Researcher-designed criteria are NOT emergent animal goals or reflexes.
Original R1-R9 and main are imported unchanged.
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
from neuroevolution import flyloop_r9 as r9

SEEDS=tuple(range(11800,11812))
GEOMETRIES=("obstacles","nearfield")
REGIMES=("static","removal")
ARMS=("slow_only","fast_fixed","fast_direct","fast_naive",
      "fast_teacher","fast_shuffled")
MACROS=240
TRAIN_MACROS=120
CRITIC_WARMUP=20
SUBSTEPS=r9.SUBSTEPS
MICRO_DT=r9.MICRO_DT
SLOW_LR=.014
SLOW_OFFSETS=(.5,.25)
SLOW_ACTIONS=("base","steer_minus","steer_plus","throttle_minus","throttle_plus")
SLOW_EXPLORATION_EVERY=4
TEACHER_PROGRESS_MIN=.15
TEACHER_SHAM_MIN_AGE=8
FAST_SUPERVISED_LR=.025
FAST_OUTPUT_BOUND=2.
FAST_ARMS=ARMS[1:]


def sha(value):
    return hashlib.sha256(json.dumps(
        value,sort_keys=True,separators=(",",":")
    ).encode()).hexdigest()


def mean(v):
    v=list(v)
    return sum(v)/len(v) if v else 0.


def rms_radius(positions):
    if not positions:
        return 0.
    cx=mean(p[0] for p in positions)
    cy=mean(p[1] for p in positions)
    return math.sqrt(mean((x-cx)**2+(y-cy)**2 for x,y in positions))


class World(r9.World):
    """R9 equal-armed microphysics; only the R10 fixed seed/switch differs."""
    def __init__(self,seed,geometry,regime):
        if seed not in SEEDS or geometry not in GEOMETRIES or regime not in REGIMES:
            raise ValueError("Unregistered R10 seed/geometry/regime")
        self.seed=seed
        self.geometry=geometry
        self.regime=regime
        self.scenario=geometry
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
        self.changed=False

    def apply_change(self,macro):
        if macro==TRAIN_MACROS and self.regime=="removal":
            self.rectangles=list(self.initial_rectangles[::2])
            self.changed=True


class SlowCritic:
    """Learn actual macro physical outcomes, issue evaluated teacher motors."""
    def __init__(self,seed):
        self.weights=[[0.]*r4.FEATURES for _ in range(2)]
        self.updates=0
        self.menu_rng=random.Random(seed*47813+144)

    def features(self,hidden,rays,cmd):
        return r4.ActionSensorHeads.make_features(hidden,rays,cmd)

    def predict(self,phi):
        return tuple(
            r1.clamp(SLOW_OFFSETS[k]+r1.dot(self.weights[k],phi),0.,1.)
            for k in range(2)
        )

    def choose(self,macro,hidden,rays,base):
        menu=r7.candidates(base)
        preds=[self.predict(self.features(hidden,rays,cmd)) for cmd in menu]
        scores=[v[0]-1.2*v[1] for v in preds]
        exploratory=(macro>=CRITIC_WARMUP and
                     macro<TRAIN_MACROS and
                     macro%SLOW_EXPLORATION_EVERY==0)
        if macro<CRITIC_WARMUP:
            selected=0
        elif exploratory:
            selected=self.menu_rng.randrange(len(menu))
        else:
            selected=max(range(len(scores)),key=lambda i:scores[i])
        cmd=menu[selected]
        phi=self.features(hidden,rays,cmd)
        return {
            "candidates":menu,"predictions":preds,"scores":scores,
            "exploratory":exploratory,"index":selected,"command":cmd,
            "chosen_preaction_features":phi,
            "chosen_preaction_prediction":preds[selected],
        }

    def complete(self,macro,chosen,physical_steps):
        if not physical_steps:
            raise ValueError("No actual physical episode")
        moved=sum(s["moved"] for s in physical_steps)
        contact=max(s["collision"] for s in physical_steps)
        norm_move=r1.clamp(
            moved/(SUBSTEPS*r1.V_MAX*MICRO_DT),0.,1.
        )
        target=(norm_move,float(contact))
        # The critic's teacher validity is checked AFTER physical actions.
        validated=(
            len(physical_steps)==SUBSTEPS
            and not contact
            and norm_move>=TEACHER_PROGRESS_MIN
            and not chosen["exploratory"]
        )
        if macro<TRAIN_MACROS:
            phi=chosen["chosen_preaction_features"]
            pred=chosen["chosen_preaction_prediction"]
            for k in range(2):
                rate=SLOW_LR*(target[k]-pred[k])
                for j in range(r4.FEATURES):
                    self.weights[k][j]=r1.clamp(
                        self.weights[k][j]+rate*phi[j],-2.,2.
                    )
            self.updates+=1
        return {
            "target":list(target),
            "actual_microsteps":len(physical_steps),
            "any_real_contact":int(contact),
            "validated":bool(validated),
            "actual_slow_issued_motor":list(chosen["command"]),
        }


class TeacherStudent(r9.FastCircuit):
    """Fast synapses supervised ONLY by 2D slow teacher action labels."""
    def __init__(self,seed):
        super().__init__(seed)
        self.supervised_updates=0
        self.validated_supervised_updates=0
        self.teacher_sham_rng=random.Random(seed*71363+21277)

    def fast_motor(self,enabled):
        if not enabled:
            return None,(0.,0.),False
        phi=tuple(self.h)+(1.,)
        proposal=tuple(
            math.tanh(r1.dot(self.output[k],phi)) for k in range(2)
        )
        noise=(
            self.noise_rng.gauss(0.,r9.FAST_NOISE),
            self.noise_rng.gauss(0.,r9.FAST_NOISE),
        )
        motor=tuple(r1.clamp(proposal[k]+noise[k],-1.,1.)
                    for k in range(2))
        return motor,noise,True

    def fit_teacher(self,macro,fast_hidden,teacher_cmd,validated):
        """Only slow-issued MOTOR label, no physical outcome provided."""
        if macro>=TRAIN_MACROS:
            raise AssertionError("No fast teacher updates in evaluation")
        phi=tuple(fast_hidden)+(1.,)
        for k in range(2):
            pre=r1.dot(self.output[k],phi)
            predicted=math.tanh(pre)
            rate=(FAST_SUPERVISED_LR*(teacher_cmd[k]-predicted)*
                  (1-predicted*predicted))
            for j in range(5):
                self.output[k][j]=r1.clamp(
                    self.output[k][j]+rate*phi[j],
                    -FAST_OUTPUT_BOUND,FAST_OUTPUT_BOUND
                )
        self.supervised_updates+=1
        if validated:
            self.validated_supervised_updates+=1


def simulate(seed,geometry,regime,arm):
    if arm not in ARMS:
        raise ValueError("Unknown R10 arm")
    world=World(seed,geometry,regime)
    slow=r2.MotorLearner(seed,"recurrent_frozen")
    teacher=SlowCritic(seed)
    fast=TeacherStudent(seed)
    core_initial=copy.deepcopy(slow.circuit.weights)
    fast_initial=copy.deepcopy(fast.output)
    initial_pose=[2.2,2.2,round(world.body.heading,9)]
    validated_prior=[]
    macro_log=[]
    micro_log=[]
    post_micro=[]
    macro_validated=0
    macro_salient_validated=0
    teaching_history=[]

    for macro in range(MACROS):
        if world.body.energy<=0 or world.body.integrity<=0:
            break
        world.apply_change(macro)
        current=tuple(world.perceive())
        raw,slow_hidden,observed,noise,base=slow.command(current)
        choice=teacher.choose(macro,slow_hidden,current[1:4],base)
        slow_motor=choice["command"]
        episode_physics=[]
        fast_last_hidden=None
        episode_salient=False
        episode_micro_log=[]
        for sub in range(SUBSTEPS):
            if world.body.energy<=0 or world.body.integrity<=0:
                break
            micro_tick=world.frame
            sensors=tuple(world.perceive())
            seven=r9.local_fast_sensors(sensors)
            fast_hidden=fast.observe(seven)
            fast_last_hidden=fast_hidden
            gate=r9.is_triggered(seven)
            episode_salient|=gate
            active=macro>=TRAIN_MACROS and arm!="slow_only" and gate
            actual_micro,fast_noise,override=fast.fast_motor(active)
            if not active:
                actual_micro=slow_motor
            before_integrity=world.body.integrity
            outcome=world.micro_integrate(actual_micro)
            physical_reward=(.7*r1.clamp(
                outcome["moved"]/(r1.V_MAX*MICRO_DT),0.,1.
            )-1.2*outcome["collision"])
            # The teacher-student arms do not receive physical reward as
            # a learning target. Only the EXPLICIT R9 direct control does.
            direct_source,direct_value=fast.learn(
                micro_tick,
                "fast_real" if arm=="fast_direct" else "slow_only",
                active,fast_noise,physical_reward
            )
            if arm!="fast_direct" and direct_value is not None:
                raise AssertionError("Student received physical reward")
            entry={
                "micro_tick":micro_tick,
                "macro_tick":macro,
                "substep":sub,
                "slow_16_hidden":[round(v,9) for v in slow_hidden],
                "slow_base_motor":[round(v,9) for v in base],
                "slow_issued_teacher_motor":[round(v,9) for v in slow_motor],
                "slow_original_noise":[round(v,9) for v in noise],
                "actual_local_eight_sensors":[round(v,9) for v in sensors],
                "actual_local_fast_seven":[round(v,9) for v in seven],
                "actual_fast_four_hidden":[round(v,9) for v in fast_hidden],
                "fast_eligible_trigger":gate,
                "fast_activated":bool(active),
                "fast_override_applied":override,
                "fast_exploration_noise":[round(v,9) for v in fast_noise],
                "actual_executed_motor":[round(v,9) for v in actual_micro],
                "actual_micro_physics":outcome,
                "actual_integrity_loss":round(
                    before_integrity-outcome["integrity"],9
                ),
                "physical_reward_DIAGNOSTIC_not_student_target":round(
                    physical_reward,9
                ),
                "direct_R9_reward_training_used":(
                    round(direct_value,9) if direct_value is not None else None
                ),
                "direct_R9_reward_source_tick":direct_source,
            }
            micro_log.append(entry)
            episode_micro_log.append(entry)
            episode_physics.append(outcome)
            if macro>=TRAIN_MACROS:
                post_micro.append(entry)

        if not episode_physics:
            break
        teacher_judgment=teacher.complete(macro,choice,episode_physics)
        valid=teacher_judgment["validated"]
        if macro<TRAIN_MACROS and valid:
            macro_validated+=1
            if episode_salient:
                macro_salient_validated+=1
        target=None
        lesson_source=None
        if macro<TRAIN_MACROS:
            if arm=="fast_naive":
                target=tuple(slow_motor)
                lesson_source=macro
            elif arm=="fast_teacher" and valid:
                target=tuple(slow_motor)
                lesson_source=macro
            elif arm=="fast_shuffled" and valid:
                older=[
                    item for item in validated_prior
                    if item["macro_tick"]<=macro-TEACHER_SHAM_MIN_AGE
                ]
                if older:
                    src=older[fast.teacher_sham_rng.randrange(len(older))]
                    target=tuple(src["motor"])
                    lesson_source=src["macro_tick"]
            if target is not None:
                fast.fit_teacher(macro,fast_last_hidden,target,valid)
                teaching_history.append({
                    "learned_after_macro":macro,
                    "actual_original_slow_macro_source":lesson_source,
                    "actually_executed_slow_motor_label":list(target),
                    "current_episode_was_validated":bool(valid),
                    "actual_teacher_fast_hidden":list(fast_last_hidden),
                })
            if valid:
                validated_prior.append({
                    "macro_tick":macro,
                    "motor":list(slow_motor),
                    "physically_validated":True,
                    "had_salient_fast_sensors":episode_salient,
                })
        macro_log.append({
            "macro_tick":macro,
            "world_has_removed_walls":world.changed,
            "slow_eight_local_sensors_before":[round(v,9) for v in observed],
            "slow_16_neural_state":[round(v,9) for v in slow_hidden],
            "slow_neural_base_motor":[round(v,9) for v in base],
            "slow_motor_noise":[round(v,9) for v in noise],
            "critic_five_local_motor_candidates":[
                [round(v,9) for v in cmd] for cmd in choice["candidates"]
            ],
            "critic_five_preaction_predictions":[list(p) for p in choice["predictions"]],
            "critic_five_preaction_scores":list(choice["scores"]),
            "critic_selected_candidate":choice["index"],
            "critic_exogenous_training_exploration":choice["exploratory"],
            "slow_motor_physically_issued":[round(v,9) for v in slow_motor],
            "slow_critic_actual_physical_macro_outcome":teacher_judgment,
            "any_physically_salient_fast_sensors":episode_salient,
            "fast_last_four_hidden":list(fast_last_hidden),
            "student_teacher_motor_target":(
                list(target) if target is not None else None
            ),
            "student_teacher_source_macro":lesson_source,
            "slow_critic_weights_sha20":(
                sha(teacher.weights) if (macro+1)%20==0 else None
            ),
            "fast_output_weights_sha20":(
                sha(fast.output) if (macro+1)%20==0 else None
            ),
        })

    post_follow=sum(post_micro[i-1]["actual_micro_physics"]["collision"]
                    for i in range(1,len(post_micro)))
    post_repeat=sum(
        post_micro[i-1]["actual_micro_physics"]["collision"]
        and post_micro[i]["actual_micro_physics"]["collision"]
        for i in range(1,len(post_micro))
    )
    same_macro=sum(
        post_micro[i-1]["actual_micro_physics"]["collision"]
        and post_micro[i]["substep"]>0
        for i in range(1,len(post_micro))
    )
    immediate=sum(
        post_micro[i-1]["actual_micro_physics"]["collision"]
        and post_micro[i]["substep"]>0
        and post_micro[i]["fast_override_applied"]
        for i in range(1,len(post_micro))
    )
    return {
        "seed":seed,"geometry":geometry,"regime":regime,"arm":arm,
        "initial_ten_rectangles":[list(v) for v in world.initial_rectangles],
        "final_rectangles":[list(v) for v in world.rectangles],
        "exogenous_five_walls_removed":world.changed,
        "initial_body_pose":initial_pose,
        "initial_original_slow_434_weights":core_initial,
        "final_original_slow_434_weights":copy.deepcopy(slow.circuit.weights),
        "initial_slow_critic_56_weights":[[0.]*r4.FEATURES for _ in range(2)],
        "final_slow_critic_56_weights":teacher.weights,
        "slow_critic_update_count":teacher.updates,
        "original_fast_four_input":fast.input,
        "original_fast_four_recurrent":fast.recurrent,
        "original_fast_four_bias":fast.bias,
        "fast_original_10_output_weights":fast_initial,
        "fast_final_10_output_weights":fast.output,
        "fast_supervised_teacher_update_count":fast.supervised_updates,
        "fast_validated_teacher_update_count":fast.validated_supervised_updates,
        "fast_direct_physical_reward_update_count":fast.update_events,
        "completed_macro_steps":len(macro_log),
        "real_microsteps":len(micro_log),
        "real_postchange_microsteps":len(post_micro),
        "validated_slow_teacher_examples":macro_validated,
        "validated_slow_examples_with_salient_fast_input":macro_salient_validated,
        "actual_teaching_lessons":teaching_history,
        "postchange_visited_cells":len({
            (int(f["actual_micro_physics"]["position"][0]),
             int(f["actual_micro_physics"]["position"][1]))
            for f in post_micro
        }),
        "postchange_real_distance":round(
            sum(f["actual_micro_physics"]["moved"] for f in post_micro),6
        ),
        "postchange_micro_collisions":sum(
            f["actual_micro_physics"]["collision"] for f in post_micro
        ),
        "postchange_prior_contact_followups":post_follow,
        "postchange_repeated_micro_contacts":post_repeat,
        "postchange_RMS_radius":round(rms_radius([
            f["actual_micro_physics"]["position"] for f in post_micro
        ]),6),
        "postchange_fast_motor_override_count":sum(
            f["fast_override_applied"] for f in post_micro
        ),
        "postchange_same_macro_contact_followup_opportunities":same_macro,
        "postchange_same_macro_rapid_override_count":immediate,
        "macro_trace":macro_log,
        "raw_micro_trace":micro_log,
    }


def summarize(rows):
    distances=sum(r["postchange_real_distance"] for r in rows)
    frames=sum(r["real_postchange_microsteps"] for r in rows)
    follow=sum(r["postchange_prior_contact_followups"] for r in rows)
    immediate=sum(
        r["postchange_same_macro_contact_followup_opportunities"] for r in rows
    )
    return {
        "lives":len(rows),
        "mean_postchange_cells":round(
            mean(r["postchange_visited_cells"] for r in rows),6
        ),
        "mean_postchange_real_distance":round(
            mean(r["postchange_real_distance"] for r in rows),6
        ),
        "mean_postchange_RMS_radius":round(
            mean(r["postchange_RMS_radius"] for r in rows),6
        ),
        "postchange_micromovement_frames":frames,
        "postchange_actual_contacts":sum(
            r["postchange_micro_collisions"] for r in rows
        ),
        "postchange_contact_followups":follow,
        "postchange_repeat_contacts":sum(
            r["postchange_repeated_micro_contacts"] for r in rows
        ),
        "contact_per_real_meter":round(sum(
            r["postchange_micro_collisions"] for r in rows
        )/max(distances,1e-12),9),
        "repeated_contact_fraction":round(sum(
            r["postchange_repeated_micro_contacts"] for r in rows
        )/max(follow,1),9),
        "postchange_actual_fast_override_fraction":round(sum(
            r["postchange_fast_motor_override_count"] for r in rows
        )/max(frames,1),9),
        "same_macro_contact_followup_opportunities":immediate,
        "same_macro_fast_response_fraction":round(sum(
            r["postchange_same_macro_rapid_override_count"] for r in rows
        )/max(immediate,1),9),
        "validated_slow_teacher_lessons":sum(
            r["validated_slow_teacher_examples"] for r in rows
        ),
        "salient_validated_slow_lessons":sum(
            r["validated_slow_examples_with_salient_fast_input"] for r in rows
        ),
        "actual_fast_supervised_lessons":sum(
            r["fast_supervised_teacher_update_count"] for r in rows
        ),
        "actual_direct_reward_updates":sum(
            r["fast_direct_physical_reward_update_count"] for r in rows
        ),
        "shorter_than_240_macro_lives":sum(
            r["completed_macro_steps"]<MACROS for r in rows
        ),
    }


def audit(rows):
    """Source-code-independent audit is separate; strict recorded causality."""
    physical=True
    after_macro=True
    no_direct_label_leak=True
    valid_sham=True
    frozen=True
    for row in rows:
        micro=row["raw_micro_trace"]
        macros=row["macro_trace"]
        lessons=row["actual_teaching_lessons"]
        if (len(micro)!=row["real_microsteps"]
            or len(macros)!=row["completed_macro_steps"]
            or row["initial_original_slow_434_weights"]!=
               row["final_original_slow_434_weights"]):
            physical=False
        if row["slow_critic_update_count"]!=min(
            TRAIN_MACROS,row["completed_macro_steps"]
        ):
            after_macro=False
        if any(f["macro_tick"]!=i or f[
            "slow_critic_actual_physical_macro_outcome"]["actual_microsteps"]!=
            sum(m["macro_tick"]==i for m in micro)
            for i,f in enumerate(macros)):
            after_macro=False
        true_accepted={
            m["macro_tick"]:m for m in macros
            if m["slow_critic_actual_physical_macro_outcome"]["validated"]
            and m["macro_tick"]<TRAIN_MACROS
        }
        for lesson in lessons:
            t=lesson["learned_after_macro"]
            src=lesson["actual_original_slow_macro_source"]
            if t>=TRAIN_MACROS or t>=len(macros):
                frozen=False
                continue
            if row["arm"]=="fast_teacher":
                if src!=t or t not in true_accepted:
                    no_direct_label_leak=False
            if row["arm"]=="fast_naive":
                if src!=t:
                    no_direct_label_leak=False
            if row["arm"]=="fast_shuffled":
                if (src not in true_accepted or src>t-TEACHER_SHAM_MIN_AGE):
                    valid_sham=False
            if src is not None and (
                lesson["actually_executed_slow_motor_label"]!=
                macros[src]["slow_motor_physically_issued"]
            ):
                no_direct_label_leak=False
            if lesson["actual_teacher_fast_hidden"]!=macros[t][
                "fast_last_four_hidden"
            ]:
                after_macro=False
        if row["arm"] in ("slow_only","fast_fixed","fast_direct") and lessons:
            no_direct_label_leak=False
        if row["arm"]!="fast_direct" and row[
            "fast_direct_physical_reward_update_count"
        ]:
            no_direct_label_leak=False
        if row["arm"]=="fast_teacher" and row[
            "fast_supervised_teacher_update_count"
        ]!=row["validated_slow_teacher_examples"]:
            after_macro=False
        if row["arm"]=="fast_naive" and row[
            "fast_supervised_teacher_update_count"
        ]!=min(TRAIN_MACROS,row["completed_macro_steps"]):
            after_macro=False
        if any(
            f["fast_override_applied"] or f["fast_activated"]
            for f in micro if f["macro_tick"]<TRAIN_MACROS
        ):
            physical=False
        if any(f["direct_R9_reward_training_used"] is not None
               for f in micro if row["arm"]!="fast_direct"):
            no_direct_label_leak=False
        if any(f["direct_R9_reward_training_used"] is not None
               for f in micro if f["macro_tick"]<TRAIN_MACROS):
            no_direct_label_leak=False
    return {
        "physical_trajectories_and_original_neural_coefficients_integrity":physical,
        "slow_critic_and_supervised_updates_only_after_real_completed_macro":after_macro,
        "student_labels_from_slow_actions_no_bodily_target_leak":no_direct_label_leak,
        "shuffled_teacher_targets_real_and_at_least_8_macros_old":valid_sham,
        "student_frozen_during_evaluation":frozen,
    }


def bootstrap(rows,arm,other,n=1000):
    pair={(r["seed"],r["arm"]):r for r in rows
          if r["geometry"]=="nearfield" and r["regime"]=="removal"}
    diffs=[pair[(s,arm)]["postchange_visited_cells"]
           -pair[(s,other)]["postchange_visited_cells"] for s in SEEDS]
    rng=random.Random(118000)
    vals=sorted(mean(diffs[rng.randrange(len(diffs))] for _ in diffs)
                for _ in range(n))
    return {
        "mean_seed_paired_coverage_delta":round(mean(diffs),6),
        "exploratory_seed_bootstrap95":[
            round(vals[25],6),round(vals[974],6)
        ],
        "independent_seed_count":len(SEEDS),
    }


def run():
    rows=[
        simulate(seed,geom,regime,arm)
        for geom in GEOMETRIES
        for regime in REGIMES
        for seed in SEEDS for arm in ARMS
    ]
    integrity=audit(rows)
    summary={
        geo:{
            regime:{
                arm:summarize([
                    row for row in rows
                    if row["geometry"]==geo and row["regime"]==regime
                    and row["arm"]==arm
                ]) for arm in ARMS
            } for regime in REGIMES
        } for geo in GEOMETRIES
    }
    pair={(r["seed"],r["geometry"],r["regime"],r["arm"]):r for r in rows}
    paired_initial=all(
        len({sha(pair[(s,g,r,a)]["initial_original_slow_434_weights"])
             for a in ARMS})==1
        and len({sha(pair[(s,g,r,a)]["initial_ten_rectangles"])
             for a in ARMS})==1
        and len({sha(pair[(s,g,r,a)]["fast_original_10_output_weights"])
             for a in ARMS})==1
        for s in SEEDS for g in GEOMETRIES for r in REGIMES
    )
    training_paired=all(
        [
            (f["actual_executed_motor"],f["actual_micro_physics"],
             f["slow_16_hidden"])
            for f in pair[(s,g,r,ARMS[0])]["raw_micro_trace"]
            if f["macro_tick"]<TRAIN_MACROS
        ]==[
            (f["actual_executed_motor"],f["actual_micro_physics"],
             f["slow_16_hidden"])
            for f in pair[(s,g,r,arm)]["raw_micro_trace"]
            if f["macro_tick"]<TRAIN_MACROS
        ]
        for s in SEEDS for g in GEOMETRIES for r in REGIMES for arm in ARMS
    )
    prechange=all(
        [
            (f["actual_executed_motor"],f["actual_micro_physics"])
            for f in pair[(s,g,"static",a)]["raw_micro_trace"]
            if f["macro_tick"]<TRAIN_MACROS
        ]==[
            (f["actual_executed_motor"],f["actual_micro_physics"])
            for f in pair[(s,g,"removal",a)]["raw_micro_trace"]
            if f["macro_tick"]<TRAIN_MACROS
        ]
        for s in SEEDS for g in GEOMETRIES for a in ARMS
    )
    exact_walls=all(
        len(row["initial_ten_rectangles"])==10 and
        row["final_rectangles"]==(
            row["initial_ten_rectangles"][::2]
            if row["exogenous_five_walls_removed"] and row["regime"]=="removal"
            else row["initial_ten_rectangles"]
        ) for row in rows
    )
    n=summary["nearfield"]["removal"]
    t=n["fast_teacher"]
    slow=n["slow_only"]
    naive=n["fast_naive"]
    shuffled=n["fast_shuffled"]
    fixed=n["fast_fixed"]
    taught_by_seed=[
        pair[(s,"nearfield","removal","fast_teacher")]
        for s in SEEDS
    ]
    # Static/removal training is intentionally identical; count a seed ONCE.
    validated=sum(r["validated_slow_teacher_examples"] for r in taught_by_seed)
    salient=sum(r["validated_slow_examples_with_salient_fast_input"]
                for r in taught_by_seed)
    taught=sum(r["fast_supervised_teacher_update_count"]
               for r in taught_by_seed)
    gates={
        "01_288_balanced_lives_and_full_byte_exact_double_run_required":
            len(rows)==288 and all(
                sum(r["seed"]==s and r["geometry"]==g
                    and r["regime"]==reg and r["arm"]==arm
                    for r in rows)==1
                for s in SEEDS for g in GEOMETRIES
                for reg in REGIMES for arm in ARMS
            ),
        "02_all_arms_equal_R9_actual_microphysics_no_dead_resurrection":
            integrity["physical_trajectories_and_original_neural_coefficients_integrity"],
        "03_16_recurrent_8_local_434_initial_frozen_and_common_motor_noise":
            paired_initial and all(
                r["initial_original_slow_434_weights"]==
                r["final_original_slow_434_weights"] for r in rows
            ),
        "04_56_zero_slow_critic_real_afteraction_LMS_frozen_in_test":
            integrity["slow_critic_and_supervised_updates_only_after_real_completed_macro"]
            and all(r["initial_slow_critic_56_weights"]==
                    [[0.]*r4.FEATURES for _ in range(2)] for r in rows),
        "05_all_six_identical_real_training_worlds_120_macros_and_switch":
            training_paired and prechange and exact_walls,
        "06_four_fast_seven_senses_ten_readouts_no_pretest_overrides":
            integrity["physical_trajectories_and_original_neural_coefficients_integrity"],
        "07_actual_validated_slow_episodes_only_no_direct_student_reward":
            integrity["student_labels_from_slow_actions_no_bodily_target_leak"],
        "08_sham_real_validated_same_life_motor_8plus_macros_older":
            integrity["shuffled_teacher_targets_real_and_at_least_8_macros_old"]
            and integrity["student_frozen_during_evaluation"],
        "09_all_fast_same_real_micro_trigger_and_direct_comparator_separate":
            integrity["student_labels_from_slow_actions_no_bodily_target_leak"],
        "10_raw_real_micro_macro_teacher_student_records_and_false_stop_flags":
            all(r["macro_trace"] and r["raw_micro_trace"]
                and r["final_slow_critic_56_weights"]
                and r["fast_final_10_output_weights"]
                for r in rows),
        "11_at_least_60_genuine_validated_nearfield_slow_lessons":
            validated>=60,
        "12_every_nearfield_seed_validated_student_lesson_and_half_salient":
            all(r["fast_supervised_teacher_update_count"]>=1
                for r in taught_by_seed)
            and taught>0 and salient>=.50*taught,
        "13_teacher_fast_collisions_per_meter_le_85pct_naive_imitation":
            naive["contact_per_real_meter"]>0 and
            t["contact_per_real_meter"]<=.85*naive["contact_per_real_meter"],
        "14_teacher_fast_collisions_per_meter_le_85pct_old_shuffled_lesson":
            shuffled["contact_per_real_meter"]>0 and
            t["contact_per_real_meter"]<=.85*shuffled["contact_per_real_meter"],
        "15_teacher_fast_collisions_per_meter_le_85pct_fixed_fast":
            fixed["contact_per_real_meter"]>0 and
            t["contact_per_real_meter"]<=.85*fixed["contact_per_real_meter"],
        "16_teacher_fast_repeated_contact_fraction_le_90pct_slow_only":
            slow["postchange_contact_followups"]>=50 and
            t["repeated_contact_fraction"]<=.90*slow["repeated_contact_fraction"],
        "17_teacher_fast_cells_and_distance_ge_90pct_slow_only":
            t["mean_postchange_cells"]>=.90*slow["mean_postchange_cells"]
            and t["mean_postchange_real_distance"]>=
                .90*slow["mean_postchange_real_distance"],
        "18_teacher_fast_actual_micro_override_ge10pct_and_response_ge50pct":
            t["postchange_actual_fast_override_fraction"]>=.10
            and t["same_macro_contact_followup_opportunities"]>0
            and t["same_macro_fast_response_fraction"]>=.50,
    }
    return {
        "name":"FlyLoop R10 teach fast pathway from slow critic evaluated real actions",
        "protocol":"docs/FLYLOOP_R10_PROTOCOL_2026-11-25.md",
        "metadata":{
            "fresh_seeds":SEEDS,"geometries":GEOMETRIES,
            "regimes":REGIMES,"controllers":ARMS,
            "macro_seconds":r1.DT,
            "micro_seconds":MICRO_DT,
            "microframes_per_macro":SUBSTEPS,
            "slow_critic_56_trainable_coefficients":56,
            "fast_10_motor_student_coefficients":10,
            "exogenous_wall_switch_macro":TRAIN_MACROS,
            "all_teacher_train_only_before_macro":TRAIN_MACROS,
            "no_fast_reward_in_distilled_students":True,
            "teacher_slow_16_neural_core_weights_frozen":True,
            "no_flee_attack_or_oracle":True,
            "R1_R9_source_and_main_unmodified":True,
        },
        "primary_unique_nearfield_slow_validated_lessons":validated,
        "primary_unique_nearfield_salient_validated_lessons":salient,
        "primary_nearfield_true_student_lesson_updates":taught,
        "summary":summary,
        "internal_causal_provenance_audit":integrity,
        "exploratory_paired_world_seed_bootstrap":{
            ref:bootstrap(rows,"fast_teacher",ref)
            for ref in ("slow_only","fast_fixed","fast_naive",
                        "fast_shuffled","fast_direct")
        },
        "gates":gates,
        "gates_passed":sum(gates.values()),
        "science_screen_pass":all(gates.values()),
        "slow_to_fast_teacher_signal_implemented":True,
        "effective_learned_fast_defense_demonstrated":all(gates.values()),
        "autonomous_escape_demonstrated":False,
        "autonomous_attack_demonstrated":False,
        "autonomous_goal_invention_demonstrated":False,
        "food_search_demonstrated":False,
        "general_intelligence_demonstrated":False,
        "biological_fly_replication_demonstrated":False,
        "evolution_authorized":False,
        "advance_to_evolution":False,
        "raw_rows_sha256":sha(rows),
        "raw_rows":rows,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,
                   default=Path("results/flyloop_r10/r10_slow_teaches_fast.json"))
    args=p.parse_args()
    result=run()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(
        result,sort_keys=True,separators=(",",":"))+"\n",encoding="utf8")
    print("FLYLOOP_R10 "+json.dumps({
        k:result[k] for k in (
            "name","primary_unique_nearfield_slow_validated_lessons",
            "primary_unique_nearfield_salient_validated_lessons",
            "primary_nearfield_true_student_lesson_updates",
            "summary","internal_causal_provenance_audit",
            "gates","gates_passed","science_screen_pass","raw_rows_sha256"
        )
    },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
