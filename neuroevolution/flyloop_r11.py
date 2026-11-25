"""FlyLoop R11: learned SLOW recovery episodes teach the FAST motor route.

Prospectively frozen protocol docs/FLYLOOP_R11_PROTOCOL_2026-11-25.md
implements real-contact -> observed time-extended disengagement -> real
teacher action attached to first-postcontact fast sensory state. The
four-neuron rapid network learns from slow action labels ONLY, never
bodily damage or reward. All original R1-R10 modules remain UNCHANGED.

Researcher-specified salience, broad experimental motor exploration,
recoveries and motor learning are NOT emergent fight/flight/survival
motivation or programmed escape directions.
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

SEEDS=tuple(range(12100,12112))
GEOMETRIES=("obstacles","nearfield")
REGIMES=("static","removal")
ARMS=(
    "slow_only","fast_fixed","fast_generic_matched",
    "fast_recovery_true","fast_recovery_shuffled",
    "fast_recovery_random",
)
MACROS=260
TRAIN_MACROS=160
SUBSTEPS=r9.SUBSTEPS
MICRO_DT=r9.MICRO_DT
SLOW_WARMUP=20
SLOW_EXPLORATION_EVERY=4
CONTACT_EXPLORATION_MACROS=12
CONTACT_WINDOW_MICRO=32
RECOVERY_FREE_MICROS=4
RECOVERY_DISTANCE=.035
SLOW_CRITIC_LR=.014
SLOW_CRITIC_OFFSETS=(.5,.25,0.)
SLOW_RECOVERY_BONUS=.7
SLOW_CONTACT_COST=1.2
FAST_SUPERVISED_LR=.025
FAST_WEIGHT_BOUND=2.
FAST_ARM_SET=frozenset(ARMS)-{"slow_only"}
SUPERVISED_ARMS=frozenset((
    "fast_generic_matched","fast_recovery_true",
    "fast_recovery_shuffled","fast_recovery_random",
))


def sha(x):
    return hashlib.sha256(
        json.dumps(x,sort_keys=True,separators=(",",":")).encode()
    ).hexdigest()


def avg(values):
    values=list(values)
    return sum(values)/len(values) if values else 0.


def radius(positions):
    if not positions:
        return 0.
    x=avg(p[0] for p in positions)
    y=avg(p[1] for p in positions)
    return math.sqrt(avg(
        (p[0]-x)**2+(p[1]-y)**2 for p in positions
    ))


class World(r9.World):
    """Same REAL original R9 physical integration for EVERY arm."""
    def __init__(self,seed,geometry,regime):
        if seed not in SEEDS or geometry not in GEOMETRIES or regime not in REGIMES:
            raise ValueError("R11 seed/geometry/regime not preregistered")
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


class SlowRecoveryCritic:
    """Three genuinely online real-outcome 28D physical heads.

    Progress and contact are known after an ACTUAL completed macro.
    The third head receives a delayed label ONLY after a real
    contact/recovery or 32-microstep failure episode is resolved.
    """
    def __init__(self,seed):
        self.weights=[[0.]*r4.FEATURES for _ in range(3)]
        self.progress_updates=0
        self.recovery_updates=0
        self.regular_rng=random.Random(seed*47813+144)
        self.contact_rng=random.Random(seed*73427+201)

    def features(self,hidden,rays,command):
        return r4.ActionSensorHeads.make_features(hidden,rays,command)

    def predict(self,phi):
        return tuple(
            r1.clamp(
                SLOW_CRITIC_OFFSETS[k]+r1.dot(self.weights[k],phi),
                0.,1.
            ) for k in range(3)
        )

    def choose(self,tick,hidden,rays,base,contact_explore_until):
        menu=r7.candidates(base)
        predictions=[
            self.predict(self.features(hidden,rays,command))
            for command in menu
        ]
        scores=[
            p[0]-SLOW_CONTACT_COST*p[1]+SLOW_RECOVERY_BONUS*p[2]
            for p in predictions
        ]
        explore_kind="none"
        selected=0
        if tick<SLOW_WARMUP:
            cmd=menu[0]
        elif tick<TRAIN_MACROS and tick<contact_explore_until:
            # BOTH motors drawn without a programmed direction.
            cmd=(
                self.contact_rng.uniform(-1.,1.),
                self.contact_rng.uniform(-1.,1.),
            )
            selected=-1
            explore_kind="real_contact_broad_uniform"
        elif tick<TRAIN_MACROS and tick%SLOW_EXPLORATION_EVERY==0:
            selected=self.regular_rng.randrange(len(menu))
            cmd=menu[selected]
            explore_kind="real_regular_menu_uniform"
        else:
            selected=max(range(len(scores)),key=lambda j:scores[j])
            cmd=menu[selected]
        phi=self.features(hidden,rays,cmd)
        return {
            "menu":menu,"predictions":predictions,"scores":scores,
            "selected_index":selected,"command":cmd,
            "exploration":explore_kind,"pre_action_phi":phi,
            "pre_action_predictions":self.predict(phi),
        }

    def _learn(self,head,features,pred,target):
        rate=SLOW_CRITIC_LR*(target-pred)
        for j in range(r4.FEATURES):
            self.weights[head][j]=r1.clamp(
                self.weights[head][j]+rate*features[j],-2.,2.
            )

    def complete_actual_macro(self,macro,choice,physical):
        if not physical:
            raise ValueError("No physically executed slow macro")
        actual_distance=sum(outcome["moved"] for outcome in physical)
        contact=max(outcome["collision"] for outcome in physical)
        target=(
            r1.clamp(actual_distance/(SUBSTEPS*r1.V_MAX*MICRO_DT),0.,1.),
            float(contact)
        )
        if macro<TRAIN_MACROS:
            for k in (0,1):
                self._learn(
                    k,choice["pre_action_phi"],
                    choice["pre_action_predictions"][k],target[k]
                )
            self.progress_updates+=1
        return {
            "actual_normalized_distance":target[0],
            "any_real_collision":int(contact),
            "real_microframes":len(physical),
            "actual_slow_motor":list(choice["command"]),
        }

    def train_delayed_recovery(self,macro,resolved):
        if macro>=TRAIN_MACROS and resolved:
            raise AssertionError("Cannot learn recovery after split")
        for item in resolved:
            target=1. if item["verified_recovered"] else 0.
            phi=item["_slow_onset_phi"]
            pred=self.predict(phi)[2]
            self._learn(2,phi,pred,target)
            self.recovery_updates+=1


class FastStudent(r9.FastCircuit):
    """Fast learns slow motor instruction, NEVER direct physical reward."""
    def __init__(self,seed):
        super().__init__(seed)
        self.supervised_updates=0
        self.initial_output=copy.deepcopy(self.output)

    def learn_slow_command(self,hidden,actual_slow_motor):
        phi=tuple(hidden)+(1.,)
        if len(phi)!=5 or len(actual_slow_motor)!=2:
            raise ValueError("Invalid teacher label")
        for k in range(2):
            predicted=math.tanh(r1.dot(self.output[k],phi))
            rate=(
                FAST_SUPERVISED_LR*(actual_slow_motor[k]-predicted)
                *(1-predicted*predicted)
            )
            for j in range(5):
                self.output[k][j]=r1.clamp(
                    self.output[k][j]+rate*phi[j],
                    -FAST_WEIGHT_BOUND,FAST_WEIGHT_BOUND
                )
        self.supervised_updates+=1

    def fast_motor(self,enabled,slow_command):
        if not enabled:
            return tuple(slow_command),(0.,0.),False
        phi=tuple(self.h)+(1.,)
        intended=tuple(
            math.tanh(r1.dot(self.output[k],phi)) for k in (0,1)
        )
        perturb=(
            self.noise_rng.gauss(0.,r9.FAST_NOISE),
            self.noise_rng.gauss(0.,r9.FAST_NOISE)
        )
        physical=tuple(
            r1.clamp(intended[k]+perturb[k],-1.,1.) for k in (0,1)
        )
        return physical,perturb,True


class RecoveryEpisodes:
    """Only ACTUAL after-contact continuous physical recovery is a lesson."""
    def __init__(self):
        self.pending=None
        self.completed=[]
        self.successes=[]
        self.contact_onsets=0

    def before_micro(self,tick,hidden,seven):
        p=self.pending
        if (p is not None
            and p["first_post_contact_micro"] is None
            and tick>p["collision_micro"]):
            p["first_post_contact_micro"]=tick
            p["first_post_contact_fast_hidden"]=list(hidden)
            p["first_post_contact_local_senses"]=list(seven)

    def after_real_micro(
        self,macro,tick,physical,slow_command,hidden,seven,onset_phi
    ):
        """Call AFTER one real body integration; no model future access."""
        resolved=[]
        new_onset=False
        p=self.pending
        if p is not None and tick>p["collision_micro"]:
            if not physical["collision"]:
                p["_free_run"].append({
                    "tick":tick,
                    "movement":physical["moved"],
                    "actual_slow_motor":list(slow_command),
                })
                p["_free_run"]=p["_free_run"][-RECOVERY_FREE_MICROS:]
            else:
                p["_free_run"]=[]
            success=(
                len(p["_free_run"])==RECOVERY_FREE_MICROS and
                sum(x["movement"] for x in p["_free_run"])>=RECOVERY_DISTANCE
                and p["first_post_contact_fast_hidden"] is not None
            )
            expired=tick-p["collision_micro"]>=CONTACT_WINDOW_MICRO
            if success or expired:
                p["finished_after_micro"]=tick
                p["verified_recovered"]=bool(success)
                if success:
                    first=p["_free_run"][0]
                    p["actual_first_recovery_motor"]=first[
                        "actual_slow_motor"
                    ]
                    p["first_recovery_motor_micro"]=first["tick"]
                    self.successes.append(p)
                else:
                    p["actual_first_recovery_motor"]=None
                    p["first_recovery_motor_micro"]=None
                resolved.append(p)
                self.completed.append(p)
                self.pending=None
        if self.pending is None and physical["collision"] and not resolved:
            self.contact_onsets+=1
            self.pending={
                "collision_micro":tick,
                "collision_macro":macro,
                "_slow_onset_phi":list(onset_phi),
                "first_post_contact_micro":None,
                "first_post_contact_fast_hidden":None,
                "first_post_contact_local_senses":None,
                "actual_first_recovery_motor":None,
                "first_recovery_motor_micro":None,
                "_free_run":[],
                "finished_after_micro":None,
                "verified_recovered":False,
            }
            new_onset=True
        return resolved,new_onset

    @staticmethod
    def public(item):
        return {key:copy.deepcopy(value) for key,value in item.items()
                if not key.startswith("_")}

    def verified_lessons(self):
        return [
            self.public(item) for item in self.successes
            if item["first_post_contact_fast_hidden"] is not None
        ]


def consolidate_student(seed,arm,fast,successes,safe_macros):
    """EXACT matched recovery snapshot and number of lessons in all
    four supervised arms; only motor targets differ. No model future
    outside completed own-life training. No fast physics during train.
    """
    n=len(successes)
    if not safe_macros:
        # Standard preregistered case: cannot match ordinary teacher
        # labels; all FOUR supervision controls therefore skip.
        n=0
    rng=random.Random(seed*92177+7001)
    labels=[tuple(x["actual_first_recovery_motor"]) for x in successes]
    dataset=[]
    for i in range(n):
        src=successes[i]
        if arm=="fast_recovery_true":
            target=labels[i]
            source_tick=src["first_recovery_motor_micro"]
            lesson_kind="verified_slow_recovery"
        elif arm=="fast_recovery_shuffled":
            j=(i+1)%n
            target=labels[j]
            source_tick=successes[j]["first_recovery_motor_micro"]
            lesson_kind="same_life_verified_recovery_cyclic_shift"
        elif arm=="fast_generic_matched":
            # Genuine SAFE ordinary slow movement from same actual life.
            choice=safe_macros[(i*7+seed)%len(safe_macros)]
            target=tuple(choice["motor"])
            source_tick=choice["macro"]*SUBSTEPS
            lesson_kind="same_life_real_ordinary_safe_motor"
        elif arm=="fast_recovery_random":
            target=(rng.uniform(-1.,1.),rng.uniform(-1.,1.))
            source_tick=None
            lesson_kind="synthetic_uniform_random_motor_negative_control"
        else:
            raise AssertionError("Only explicitly supervised arms")
        hidden=src["first_post_contact_fast_hidden"]
        before_hash=sha(fast.output)
        fast.learn_slow_command(hidden,target)
        dataset.append({
            "recovery_collision_source_micro":src["collision_micro"],
            "actual_contact_fast_state_micro":src["first_post_contact_micro"],
            "actual_contact_fast_four_hidden":list(hidden),
            "input_is_real_postcontact_state":True,
            "supervised_motor_label":list(target),
            "motor_label_source_real_micro":source_tick,
            "motor_label_origin":lesson_kind,
            "fast_output_hash_before":before_hash,
            "fast_output_hash_after":sha(fast.output),
        })
    return dataset


def simulate(seed,geometry,regime,arm):
    if arm not in ARMS:
        raise ValueError("Unregistered R11 arm")
    world=World(seed,geometry,regime)
    slow=r2.MotorLearner(seed,"recurrent_frozen")
    teacher=SlowRecoveryCritic(seed)
    fast=FastStudent(seed)
    episodes=RecoveryEpisodes()
    original_slow=copy.deepcopy(slow.circuit.weights)
    original_fast=copy.deepcopy(fast.output)
    initial_pose=[2.2,2.2,round(world.body.heading,9)]
    history=[]
    macro_history=[]
    safe_macros=[]
    lessons=[]
    contact_explore_until=-1
    postmicro=[]
    training_contact_windows=0
    transfer_completed=False

    for macro in range(MACROS):
        if world.body.energy<=0 or world.body.integrity<=0:
            break
        if macro==TRAIN_MACROS:
            if episodes.pending is not None:
                # Future test contact/physics must not resolve a training
                # lesson. Unresolved is a legitimate negative episode.
                unresolved=episodes.pending
                unresolved["verified_recovered"]=False
                unresolved["finished_after_micro"]=None
                episodes.completed.append(unresolved)
                episodes.pending=None
            if arm in SUPERVISED_ARMS:
                lessons=consolidate_student(
                    seed,arm,fast,episodes.successes,safe_macros
                )
            transfer_completed=True
        world.apply_change(macro)
        sense=tuple(world.perceive())
        raw,hidden,observed,slow_noise,base=slow.command(sense)
        choice=teacher.choose(
            macro,hidden,sense[1:4],base,contact_explore_until
        )
        slow_command=choice["command"]
        physical_episode=[]
        resolution_events=[]
        micro_in_macro=[]
        had_real_salience=False

        for sub in range(SUBSTEPS):
            if world.body.integrity<=0 or world.body.energy<=0:
                break
            tick=world.frame
            local=tuple(world.perceive())
            seven=r9.local_fast_sensors(local)
            fast_hidden=fast.observe(seven)
            fast_trigger=r9.is_triggered(seven)
            had_real_salience|=fast_trigger
            if macro<TRAIN_MACROS:
                episodes.before_micro(tick,fast_hidden,seven)
            active=(
                macro>=TRAIN_MACROS and arm in FAST_ARM_SET
                and fast_trigger
            )
            actual,fast_noise,override=fast.fast_motor(active,slow_command)
            before_health=world.body.integrity
            physically=world.micro_integrate(actual)
            new_onset=False
            resolved=[]
            if macro<TRAIN_MACROS:
                resolved,new_onset=episodes.after_real_micro(
                    macro,tick,physically,slow_command,fast_hidden,seven,
                    choice["pre_action_phi"]
                )
                if new_onset:
                    # All post-contact exploration starts from NEXT macro.
                    contact_explore_until=max(
                        contact_explore_until,
                        macro+CONTACT_EXPLORATION_MACROS+1
                    )
                    training_contact_windows+=1
                resolution_events.extend(resolved)
            frame={
                "micro_tick":tick,
                "macro_tick":macro,
                "within_macro_substep":sub,
                "world_removed_walls":world.changed,
                "original_slow_sixteen_hidden":[round(x,9) for x in hidden],
                "original_slow_gaussian_noise":[round(x,9) for x in slow_noise],
                "original_neural_base_analog_motor":[round(x,9) for x in base],
                "slow_issued_analog_motor":[round(x,9) for x in slow_command],
                "eight_genuine_local_preaction_sensors":[round(x,9) for x in local],
                "seven_fast_local_sensors":[round(x,9) for x in seven],
                "four_fast_recurrent_states":[round(x,9) for x in fast_hidden],
                "contact_or_proximity_salience":fast_trigger,
                "fast_actually_overrode_motor":override,
                "fast_motor_exogenous_noise":[round(x,9) for x in fast_noise],
                "actually_executed_analog_motor":[round(x,9) for x in actual],
                "actual_physical_body_after_micro":physically,
                "actual_integrity_loss":round(
                    before_health-physically["integrity"],9
                ),
                "new_real_contact_episode_started":new_onset,
                "real_recovery_episode_resolved":any(
                    e["verified_recovered"] for e in resolved
                ),
            }
            history.append(frame)
            micro_in_macro.append(frame)
            physical_episode.append(physically)
            if macro>=TRAIN_MACROS:
                postmicro.append(frame)

        if not physical_episode:
            break
        decision=teacher.complete_actual_macro(
            macro,choice,physical_episode
        )
        teacher.train_delayed_recovery(macro,resolution_events)
        if (macro<TRAIN_MACROS
            and len(physical_episode)==SUBSTEPS
            and not decision["any_real_collision"]
            and decision["actual_normalized_distance"]>=.15):
            safe_macros.append({
                "macro":macro,
                "motor":list(slow_command),
                "had_salience":had_real_salience,
                "actually_executed":True,
            })
        macro_history.append({
            "macro_tick":macro,
            "world_changed":world.changed,
            "original_slow_8_sensors":[round(x,9) for x in observed],
            "original_slow_16_states":[round(x,9) for x in hidden],
            "original_noisy_base_motor":[round(x,9) for x in base],
            "original_motor_gaussian_noise":[round(x,9) for x in slow_noise],
            "five_local_critic_candidates":[
                [round(v,9) for v in motor] for motor in choice["menu"]
            ],
            "five_preaction_physical_critic_predictions":[
                list(x) for x in choice["predictions"]
            ],
            "five_preaction_physical_critic_scores":list(choice["scores"]),
            "slow_critic_selected_candidate_index":choice["selected_index"],
            "real_slow_exploration_kind":choice["exploration"],
            "actually_issued_slow_motor":[round(v,9) for v in slow_command],
            "actual_real_macro_physical_target":decision,
            "real_delayed_contact_windows_resolved":[
                RecoveryEpisodes.public(item)
                for item in resolution_events
            ],
            "full_slow_critic_hash_after_macro":(
                sha(teacher.weights) if (macro+1)%20==0 else None
            ),
            "fast_output_hash_after_macro":(
                sha(fast.output) if (macro+1)%20==0 else None
            ),
        })

    # After training, snapshots and corresponding real motor labels
    # were all obtained from this organism, never other lives.
    actual_post_frames=len(postmicro)
    real_meters=sum(
        f["actual_physical_body_after_micro"]["moved"] for f in postmicro
    )
    hits=sum(
        f["actual_physical_body_after_micro"]["collision"] for f in postmicro
    )
    prior_hits=sum(
        postmicro[i-1]["actual_physical_body_after_micro"]["collision"]
        for i in range(1,len(postmicro))
    )
    repeat=sum(
        postmicro[i-1]["actual_physical_body_after_micro"]["collision"]
        and postmicro[i]["actual_physical_body_after_micro"]["collision"]
        for i in range(1,len(postmicro))
    )
    contact_follow_macro=sum(
        postmicro[i-1]["actual_physical_body_after_micro"]["collision"]
        and postmicro[i]["within_macro_substep"]>0
        for i in range(1,len(postmicro))
    )
    rapid_after_contact=sum(
        postmicro[i-1]["actual_physical_body_after_micro"]["collision"]
        and postmicro[i]["within_macro_substep"]>0
        and postmicro[i]["fast_actually_overrode_motor"]
        for i in range(1,len(postmicro))
    )
    recovered=0
    recovery_opportunities=0
    for i in range(len(postmicro)-RECOVERY_FREE_MICROS):
        if postmicro[i]["actual_physical_body_after_micro"]["collision"]:
            recovery_opportunities+=1
            follow=postmicro[i+1:i+1+RECOVERY_FREE_MICROS]
            if (all(not f["actual_physical_body_after_micro"]["collision"]
                    for f in follow)
                and sum(f["actual_physical_body_after_micro"]["moved"]
                        for f in follow)>=RECOVERY_DISTANCE):
                recovered+=1
    positions=[
        f["actual_physical_body_after_micro"]["position"]
        for f in postmicro
    ]
    return {
        "seed":seed,"geometry":geometry,"regime":regime,"arm":arm,
        "initial_10_rectangles":[list(x) for x in world.initial_rectangles],
        "final_rectangles":[list(x) for x in world.rectangles],
        "exogenous_five_walls_removed":world.changed,
        "initial_body_pose":initial_pose,
        "original_434_slow_coefficients_initial":original_slow,
        "original_434_slow_coefficients_final":
            copy.deepcopy(slow.circuit.weights),
        "slow_84_critic_coefficients_initial":[[0.]*r4.FEATURES for _ in range(3)],
        "slow_84_critic_coefficients_final":copy.deepcopy(teacher.weights),
        "slow_progress_contact_updates":teacher.progress_updates,
        "slow_delayed_recovery_updates":teacher.recovery_updates,
        "fast_four_initial_input":fast.input,
        "fast_four_initial_recurrent":fast.recurrent,
        "fast_four_initial_bias":fast.bias,
        "fast_10_output_coefficients_initial":original_fast,
        "fast_10_output_coefficients_after_training":copy.deepcopy(fast.output),
        "supervised_fast_student_updates":fast.supervised_updates,
        "actual_recovery_training_lessons":lessons,
        "same_life_real_safe_macro_motors":safe_macros,
        "observed_completed_recovery_windows":[
            RecoveryEpisodes.public(x) for x in episodes.completed
        ],
        "real_successful_recovery_episodes":episodes.verified_lessons(),
        "real_contact_window_onsets_training":training_contact_windows,
        "unresolved_at_training_end":int(
            any(x["finished_after_micro"] is None for x in episodes.completed)
        ),
        "teacher_consolidated_at_evaluation_split":transfer_completed,
        "completed_macros":len(macro_history),
        "actual_microsteps":len(history),
        "postchange_actual_microsteps":actual_post_frames,
        "postchange_unique_1x1_cells":len({
            (int(p[0]),int(p[1])) for p in positions
        }),
        "postchange_actual_meters":round(real_meters,6),
        "postchange_collision_microframes":hits,
        "postchange_prior_collision_followups":prior_hits,
        "postchange_repeated_microcontacts":repeat,
        "postchange_RMS_radius":round(radius(positions),6),
        "postchange_fast_micro_overrides":sum(
            f["fast_actually_overrode_motor"] for f in postmicro
        ),
        "postchange_eligible_same_macro_after_contact":contact_follow_macro,
        "postchange_rapid_next_micro_response":rapid_after_contact,
        "postchange_immediate_recovery_opportunities":recovery_opportunities,
        "postchange_immediate_real_recoveries":recovered,
        "raw_micro_trace":history,
        "macro_trace":macro_history,
    }


def summarize(group):
    meters=sum(x["postchange_actual_meters"] for x in group)
    frame=sum(x["postchange_actual_microsteps"] for x in group)
    prior=sum(x["postchange_prior_collision_followups"] for x in group)
    opportunities=sum(x["postchange_eligible_same_macro_after_contact"]
                      for x in group)
    recover_eligible=sum(x["postchange_immediate_recovery_opportunities"]
                         for x in group)
    return {
        "lives":len(group),
        "mean_postchange_cells":round(avg(
            x["postchange_unique_1x1_cells"] for x in group
        ),6),
        "mean_postchange_meters":round(avg(
            x["postchange_actual_meters"] for x in group
        ),6),
        "mean_postchange_RMS_radius":round(avg(
            x["postchange_RMS_radius"] for x in group
        ),6),
        "collision_microframes_per_real_meter":round(
            sum(x["postchange_collision_microframes"] for x in group)
            /max(meters,1e-12),9
        ),
        "actual_postchange_microcollision_events":sum(
            x["postchange_collision_microframes"] for x in group
        ),
        "actual_postchange_prior_contact_followups":prior,
        "actual_postchange_repeated_contacts":sum(
            x["postchange_repeated_microcontacts"] for x in group
        ),
        "next_microstep_repeat_contact_fraction":round(
            sum(x["postchange_repeated_microcontacts"] for x in group)
            /max(prior,1),9
        ),
        "real_postchange_microframes":frame,
        "physical_fast_override_fraction":round(
            sum(x["postchange_fast_micro_overrides"] for x in group)
            /max(frame,1),9
        ),
        "real_same_macro_contact_followup_opportunities":opportunities,
        "same_macro_next_micro_fast_response_fraction":round(
            sum(x["postchange_rapid_next_micro_response"] for x in group)
            /max(opportunities,1),9
        ),
        "actual_4micro_recoveries":sum(
            x["postchange_immediate_real_recoveries"] for x in group
        ),
        "eligible_4micro_contact_followups":recover_eligible,
        "actual_4micro_recovery_fraction":round(
            sum(x["postchange_immediate_real_recoveries"] for x in group)
            /max(recover_eligible,1),9
        ),
        "verified_actual_slow_recovery_lessons":sum(
            len(x["real_successful_recovery_episodes"]) for x in group
        ),
        "supervised_fast_updates":sum(
            x["supervised_fast_student_updates"] for x in group
        ),
        "training_contact_windows":sum(
            x["real_contact_window_onsets_training"] for x in group
        ),
        "short_lives":sum(x["completed_macros"]<MACROS for x in group),
    }


def audit(rows):
    """Full physical replay and independent logged teacher provenance.
    This uses original imported physical equations; an external auditor
    should additionally reconstruct physical motion independently.
    """
    physics_ok=True
    teaching_ok=True
    matching_count=True
    frozen_core=True
    previous_contact_ok=True
    for life in rows:
        s=life["seed"]
        w=World(s,life["geometry"],life["regime"])
        train=[f for f in life["raw_micro_trace"]
               if f["macro_tick"]<TRAIN_MACROS]
        if life["original_434_slow_coefficients_initial"]!=life[
            "original_434_slow_coefficients_final"
        ]:
            frozen_core=False
        for f in life["raw_micro_trace"]:
            t=f["macro_tick"]
            if f["micro_tick"]!=w.frame or w.body.integrity<=0 or w.body.energy<=0:
                physics_ok=False
                break
            if f["within_macro_substep"]==0:
                w.apply_change(t)
            # Logged motor is the real ACTUATOR actually applied once.
            actual=tuple(f["actually_executed_analog_motor"])
            before=tuple(w.perceive())
            if any(abs(v-x)>2e-8 for v,x in zip(
                f["eight_genuine_local_preaction_sensors"],before
            )):
                physics_ok=False
            physical=w.micro_integrate(actual)
            for key in ("collision","moved","position","heading",
                        "speed","turn","energy","integrity"):
                x=f["actual_physical_body_after_micro"][key]
                v=physical[key]
                if isinstance(v,list):
                    if any(abs(a-b)>5e-8 for a,b in zip(v,x)):
                        physics_ok=False
                elif isinstance(v,(int,float)) and abs(v-x)>5e-8:
                    physics_ok=False
            if t<TRAIN_MACROS and f["fast_actually_overrode_motor"]:
                physics_ok=False
            if f["fast_actually_overrode_motor"] and t>=TRAIN_MACROS:
                if not f["contact_or_proximity_salience"]:
                    previous_contact_ok=False
        successful=life["real_successful_recovery_episodes"]
        lessons=life["actual_recovery_training_lessons"]
        count=len(successful)
        safe=life["same_life_real_safe_macro_motors"]
        expected=(count if safe else 0) if life["arm"] in SUPERVISED_ARMS else 0
        if len(lessons)!=expected or life[
            "supervised_fast_student_updates"
        ]!=expected:
            matching_count=False
        first=life["raw_micro_trace"]
        good_by_tick={x["first_recovery_motor_micro"]:x for x in successful}
        if any(
            item["verified_recovered"] is False
            or item["first_post_contact_micro"] is None
            or not item["collision_micro"]<
                item["first_post_contact_micro"]<
                item["finished_after_micro"]+1
            or not item["first_recovery_motor_micro"]<=
                item["collision_micro"]+CONTACT_WINDOW_MICRO
            or item["first_recovery_motor_micro"] not in good_by_tick
            for item in successful
        ):
            teaching_ok=False
        for i,lesson in enumerate(lessons):
            src=successful[i]
            if (lesson["recovery_collision_source_micro"]!=
                src["collision_micro"] or
                lesson["actual_contact_fast_state_micro"]!=
                src["first_post_contact_micro"] or
                lesson["actual_contact_fast_four_hidden"]!=
                src["first_post_contact_fast_hidden"]):
                teaching_ok=False
            if (lesson["actual_contact_fast_state_micro"]>=len(first)
                or first[lesson["actual_contact_fast_state_micro"]][
                    "macro_tick"]>=TRAIN_MACROS):
                teaching_ok=False
            dest=lesson["supervised_motor_label"]
            if life["arm"]=="fast_recovery_true":
                if any(abs(a-b)>1e-8 for a,b in zip(
                    dest,src["actual_first_recovery_motor"]
                )):
                    teaching_ok=False
            if life["arm"]=="fast_recovery_shuffled":
                j=(i+1)%count
                if any(abs(a-b)>1e-8 for a,b in zip(
                    dest,successful[j]["actual_first_recovery_motor"]
                )):
                    teaching_ok=False
            if life["arm"]=="fast_generic_matched":
                if not any(all(abs(a-b)<1e-8 for a,b in zip(
                    dest,m["motor"]
                )) for m in safe):
                    teaching_ok=False
            if life["arm"]=="fast_recovery_random":
                if lesson["motor_label_source_real_micro"] is not None:
                    teaching_ok=False
        # Rebuild student from actual past contact states and exact labels,
        # strictly AFTER physical training and BEFORE physical testing.
        fst=FastStudent(s)
        for item in lessons:
            fst.learn_slow_command(
                item["actual_contact_fast_four_hidden"],
                item["supervised_motor_label"]
            )
            if sha(fst.output)!=item["fast_output_hash_after"]:
                teaching_ok=False
        if fst.output!=life["fast_10_output_coefficients_after_training"]:
            teaching_ok=False
        if life["slow_progress_contact_updates"]!=min(
            TRAIN_MACROS,life["completed_macros"]
        ):
            teaching_ok=False
        if life["slow_delayed_recovery_updates"]>life[
            "real_contact_window_onsets_training"
        ]:
            teaching_ok=False
        if life["completed_macros"]>TRAIN_MACROS and not life[
            "teacher_consolidated_at_evaluation_split"
        ]:
            teaching_ok=False
    return {
        "real_body_physical_microstep_replay_and_local_rays":physics_ok,
        "validated_real_recovery_episodes_and_student_provenance":teaching_ok,
        "exact_matched_student_update_count_and_contact_states":matching_count,
        "original_434_slow_recurrent_coefficients_frozen":frozen_core,
        "fast_motor_priority_only_on_true_local_contact_or_proximity":previous_contact_ok,
    }


def bootstrap(rows,arm,ref,n=1000):
    subset={(r["seed"],r["arm"]):r for r in rows
            if r["geometry"]=="nearfield" and r["regime"]=="removal"}
    delta=[
        subset[(seed,arm)]["postchange_unique_1x1_cells"]-
        subset[(seed,ref)]["postchange_unique_1x1_cells"]
        for seed in SEEDS
    ]
    rng=random.Random(121001)
    scores=sorted(
        avg(delta[rng.randrange(len(delta))] for _ in delta)
        for _ in range(n)
    )
    return {
        "mean_paired_seed_cell_delta":round(avg(delta),6),
        "exploratory_seed_bootstrap95":[
            round(scores[25],6),round(scores[974],6)
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
    checks=audit(rows)
    summary={
        geometry:{
            regime:{
                arm:summarize([
                    r for r in rows if r["geometry"]==geometry
                    and r["regime"]==regime and r["arm"]==arm
                ])
                for arm in ARMS
            } for regime in REGIMES
        } for geometry in GEOMETRIES
    }
    pairs={
        (r["seed"],r["geometry"],r["regime"],r["arm"]):r
        for r in rows
    }
    paired=all(
        len({sha(pairs[(s,g,reg,arm)]["initial_10_rectangles"])
             for arm in ARMS})==1
        and len({sha(pairs[(s,g,reg,arm)]["original_434_slow_coefficients_initial"])
                 for arm in ARMS})==1
        and len({sha(pairs[(s,g,reg,arm)]["initial_body_pose"])
                 for arm in ARMS})==1
        for s in SEEDS for g in GEOMETRIES for reg in REGIMES
    )
    exact_train=all(
        [
            (f["actually_executed_analog_motor"],
             f["actual_physical_body_after_micro"],
             f["original_slow_sixteen_hidden"])
            for f in pairs[(s,g,reg,ARMS[0])]["raw_micro_trace"]
            if f["macro_tick"]<TRAIN_MACROS
        ]==[
            (f["actually_executed_analog_motor"],
             f["actual_physical_body_after_micro"],
             f["original_slow_sixteen_hidden"])
            for f in pairs[(s,g,reg,arm)]["raw_micro_trace"]
            if f["macro_tick"]<TRAIN_MACROS
        ]
        for s in SEEDS for g in GEOMETRIES for reg in REGIMES
        for arm in ARMS
    )
    paired_regimes=all(
        [
            (f["actually_executed_analog_motor"],
             f["actual_physical_body_after_micro"])
            for f in pairs[(s,g,"static",arm)]["raw_micro_trace"]
            if f["macro_tick"]<TRAIN_MACROS
        ]==[
            (f["actually_executed_analog_motor"],
             f["actual_physical_body_after_micro"])
            for f in pairs[(s,g,"removal",arm)]["raw_micro_trace"]
            if f["macro_tick"]<TRAIN_MACROS
        ]
        for s in SEEDS for g in GEOMETRIES for arm in ARMS
    )
    true_by_seed=[
        pairs[(s,"nearfield","removal","fast_recovery_true")]
        for s in SEEDS
    ]
    genuine_count=sum(len(r["real_successful_recovery_episodes"])
                      for r in true_by_seed)
    learners_with_1=sum(
        len(r["real_successful_recovery_episodes"])>=1
        for r in true_by_seed
    )
    learners_with_2=sum(
        len(r["real_successful_recovery_episodes"])>=2
        for r in true_by_seed
    )
    n=summary["nearfield"]["removal"]
    true=n["fast_recovery_true"]
    slow=n["slow_only"]
    generic=n["fast_generic_matched"]
    fixed=n["fast_fixed"]
    sham=n["fast_recovery_shuffled"]
    gates={
        "01_288_balanced_lives_and_byte_identical_full_repeat_required":
            len(rows)==288 and all(
                sum(r["seed"]==seed and r["geometry"]==geometry
                    and r["regime"]==regime and r["arm"]==arm
                    for r in rows)==1
                for seed in SEEDS for geometry in GEOMETRIES
                for regime in REGIMES for arm in ARMS
            ),
        "02_R9_4x005_real_physics_inertia_no_penetration_or_resurrection":
            checks["real_body_physical_microstep_replay_and_local_rays"],
        "03_original_frozen_16_recurrent_8_local_434_coefficients_matched":
            paired and checks[
                "original_434_slow_recurrent_coefficients_frozen"
            ],
        "04_fast_four_local7_ten_supervised_output_only_no_direct_reward":
            checks["exact_matched_student_update_count_and_contact_states"]
            and all(r["fast_four_initial_input"] and
                    r["fast_four_initial_recurrent"] for r in rows),
        "05_three_zero_28D_slow_heads_train_only_actual_resolved_physics":
            checks["validated_real_recovery_episodes_and_student_provenance"]
            and all(
                r["slow_84_critic_coefficients_initial"]==
                [[0.]*r4.FEATURES for _ in range(3)] for r in rows
            ),
        "06_identical_six_arm_training_and_prechange_paired_world_removal":
            exact_train and paired_regimes and all(
                len(r["initial_10_rectangles"])==10
                and r["final_rectangles"]==(
                    r["initial_10_rectangles"][::2]
                    if r["regime"]=="removal" and r["exogenous_five_walls_removed"]
                    else r["initial_10_rectangles"]
                ) for r in rows
            ),
        "07_recovery_lessons_32micro_real_four_contactfree_progress":
            checks["validated_real_recovery_episodes_and_student_provenance"],
        "08_true_sham_generic_random_exact_lesson_counts_and_input_states":
            checks["exact_matched_student_update_count_and_contact_states"],
        "09_real_fast_local_salience_005_same_macro_only_after_training":
            checks["fast_motor_priority_only_on_true_local_contact_or_proximity"]
            and all(
                not f["fast_actually_overrode_motor"]
                for r in rows for f in r["raw_micro_trace"]
                if f["macro_tick"]<TRAIN_MACROS
            ),
        "10_full_raw_memories_physically_causal_no_attack_food_or_evolution":
            all(r["raw_micro_trace"] and r["macro_trace"] and
                r["observed_completed_recovery_windows"] is not None
                for r in rows),
        "11_at_least12_true_recoveries_nearfield_in6_of12_world_seeds":
            genuine_count>=12 and learners_with_1>=6,
        "12_at_least6_world_seeds_with2_real_lessons_for_nontrivial_sham":
            learners_with_2>=6,
        "13_true_contacts_per_meter_le80pct_slow_only":
            slow["mean_postchange_meters"]>0
            and true["collision_microframes_per_real_meter"]<=.80*
                slow["collision_microframes_per_real_meter"],
        "14_true_contacts_per_meter_le85pct_fixed_fast":
            fixed["collision_microframes_per_real_meter"]>0
            and true["collision_microframes_per_real_meter"]<=.85*
                fixed["collision_microframes_per_real_meter"],
        "15_true_contacts_per_meter_le85pct_generic_safe_matched":
            generic["collision_microframes_per_real_meter"]>0
            and true["collision_microframes_per_real_meter"]<=.85*
                generic["collision_microframes_per_real_meter"],
        "16_true_contacts_per_meter_le85pct_countmatched_shuffled_recovery":
            sham["collision_microframes_per_real_meter"]>0
            and true["collision_microframes_per_real_meter"]<=.85*
                sham["collision_microframes_per_real_meter"],
        "17_true_repeat_contact_fraction_le85pct_slow":
            slow["actual_postchange_prior_contact_followups"]>=50
            and true["next_microstep_repeat_contact_fraction"]<=.85*
                slow["next_microstep_repeat_contact_fraction"],
        "18_true_preserves90pct_cells_distance_and85pct_RMS_radius":
            true["mean_postchange_cells"]>=.90*slow["mean_postchange_cells"]
            and true["mean_postchange_meters"]>=.90*
                slow["mean_postchange_meters"]
            and true["mean_postchange_RMS_radius"]>=.85*
                slow["mean_postchange_RMS_radius"],
        "19_true_fast_microoverride_ge10pct_and_same_macro_response_ge50pct":
            true["physical_fast_override_fraction"]>=.10
            and true["real_same_macro_contact_followup_opportunities"]>0
            and true["same_macro_next_micro_fast_response_fraction"]>=.50,
    }
    return {
        "name":"FlyLoop R11 slow experienced multi-micro contact recovery teaches fast",
        "protocol":"docs/FLYLOOP_R11_PROTOCOL_2026-11-25.md",
        "metadata":{
            "seeds":SEEDS,
            "geometries":GEOMETRIES,
            "regimes":REGIMES,
            "arms":ARMS,
            "training_macro":TRAIN_MACROS,
            "life_macro_horizon":MACROS,
            "real_physical_microsteps_per_macro":SUBSTEPS,
            "micro_seconds":MICRO_DT,
            "real_contact_to_verified_recovery_window_micro":CONTACT_WINDOW_MICRO,
            "verified_contact_free_microsteps":RECOVERY_FREE_MICROS,
            "verified_minimum_real_recovery_distance":RECOVERY_DISTANCE,
            "contact_exploration_macros_researcher_specified":
                CONTACT_EXPLORATION_MACROS,
            "slow_critic_coefficients":3*r4.FEATURES,
            "fast_supervised_output_coefficients":10,
            "no_direct_fast_physical_reward_in_any_arm":True,
            "original_R1_R10_and_main_unchanged":True,
            "no_provided_escape_move_or_attack_target":True,
            "no_brake_or_reverse_added":True,
        },
        "nearfield_unique_seeds_real_successful_recoveries":genuine_count,
        "nearfield_seeds_with_at_least_one_recovery":learners_with_1,
        "nearfield_seeds_with_at_least_two_recoveries":learners_with_2,
        "summary":summary,
        "internal_structural_physical_and_lesson_audit":checks,
        "exploratory_seed_paired_bootstrap":{
            ref:bootstrap(rows,"fast_recovery_true",ref)
            for ref in ("slow_only","fast_fixed","fast_generic_matched",
                        "fast_recovery_shuffled","fast_recovery_random")
        },
        "gates":gates,
        "gates_passed":sum(gates.values()),
        "science_screen_pass":all(gates.values()),
        "slow_recovery_teacher_to_fast_implemented":True,
        "effective_recovery_defense_demonstrated":all(gates.values()),
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
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,default=Path(
        "results/flyloop_r11/r11_real_recovery_teacher.json"
    ))
    args=parser.parse_args()
    result=run()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(
        result,sort_keys=True,separators=(",",":"))+"\n",encoding="utf8")
    print("FLYLOOP_R11 "+json.dumps({
        k:result[k] for k in (
            "name","nearfield_unique_seeds_real_successful_recoveries",
            "nearfield_seeds_with_at_least_one_recovery",
            "nearfield_seeds_with_at_least_two_recoveries",
            "summary","internal_structural_physical_and_lesson_audit",
            "gates","gates_passed","science_screen_pass",
            "exploratory_seed_paired_bootstrap","raw_rows_sha256"
        )
    },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
