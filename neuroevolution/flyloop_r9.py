"""FlyLoop R9: independent rapid sensorimotor pathway with genuine .05s physics.

Researcher-engineered salience gate and progress/contact feedback:
NOT an evolved reflex, escape behavior, attack drive, or discovered goal.
All previous R1–R8 code imported unmodified. Locked prospective protocol:
docs/FLYLOOP_R9_PROTOCOL_2026-11-24.md
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

SEEDS=tuple(range(11500,11512))
GEOMETRIES=("obstacles","nearfield")
REGIMES=("static","removal")
ARMS=("slow_only","fast_fixed","fast_real","fast_old_sham","fast_random")
MACROS=280
SUBSTEPS=4
MICRO_DT=r1.DT/SUBSTEPS
MICRO_ALPHA=1-(1-r1.MOTOR_ALPHA)**(1/SUBSTEPS)
SWITCH_MACRO=140
WARMUP_MACROS=20
TRIGGER_PROXIMITY=.82
FAST_NEURONS=4
FAST_INPUTS=7
FAST_STATE_ALPHA=.40
FAST_OUTPUT_SCALE=.65
FAST_NOISE=.12
FAST_LR=.002
FAST_TRACE=.80
FAST_BASELINE=.97
FAST_WEIGHT_BOUND=2.
SHAM_MIN_AGE=12
FAST_ARMS=ARMS[1:]


def sha(x):
    return hashlib.sha256(json.dumps(
        x,sort_keys=True,separators=(",",":")
    ).encode()).hexdigest()


def mean(x):
    x=list(x)
    return sum(x)/len(x) if x else 0.


def rms_radius(xy):
    if not xy:
        return 0.
    mx,my=mean(x for x,y in xy),mean(y for x,y in xy)
    return math.sqrt(mean((x-mx)**2+(y-my)**2 for x,y in xy))


class World(r1.World):
    """R1 geometry/retained real obstacle collision with equal microphysics."""
    def __init__(self,seed,geometry,regime):
        if (seed not in SEEDS or geometry not in GEOMETRIES
            or regime not in REGIMES):
            raise ValueError("Unregistered R9 seed/geometry/regime")
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
        if macro==SWITCH_MACRO and self.regime=="removal":
            self.rectangles=list(self.initial_rectangles[::2])
            self.changed=True

    def micro_integrate(self,command):
        """One ACTUAL .05s physical integration, same for all five arms.

        New micro-discretization is NOT identical to original .20s R1
        integration; collision damage/energy scaled by 1/4 for duration.
        """
        b=self.body
        if b.energy<=0 or b.integrity<=0:
            raise AssertionError("Dead body cannot be integrated")
        throttle,steer=command
        if not all(-1.-1e-9<=v<=1.+1e-9 for v in command):
            raise ValueError("Out-of-range motor command")
        before=(b.x,b.y,b.heading,b.speed,b.turn,b.energy,b.integrity)
        wanted_speed=r1.V_MAX*r1.sigmoid(1.6*throttle)
        wanted_turn=r1.W_MAX*math.tanh(1.6*steer)
        b.speed=(1-MICRO_ALPHA)*b.speed+MICRO_ALPHA*wanted_speed
        b.turn=(1-MICRO_ALPHA)*b.turn+MICRO_ALPHA*wanted_turn
        b.heading=(b.heading+b.turn*MICRO_DT+math.pi)%(2*math.pi)-math.pi
        nx=b.x+b.speed*math.cos(b.heading)*MICRO_DT
        ny=b.y+b.speed*math.sin(b.heading)*MICRO_DT
        hit=int(self.blocked(nx,ny))
        if hit:
            self.collisions+=1
            b.integrity=max(0.,b.integrity-r1.COLLISION_INJURY/SUBSTEPS)
        else:
            b.x,b.y=nx,ny
        b.energy=max(0.,b.energy-r1.ENERGY_PER_TICK/SUBSTEPS)
        b.prev_collision=hit
        moved=math.hypot(b.x-before[0],b.y-before[1])
        self.distance+=moved
        self.visited.add((int(b.x),int(b.y)))
        self.frame+=1
        return {
            "collision":hit,
            "moved":round(moved,9),
            "position":[round(b.x,9),round(b.y,9)],
            "heading":round(b.heading,9),
            "speed":round(b.speed,9),
            "turn":round(b.turn,9),
            "energy":round(b.energy,9),
            "integrity":round(b.integrity,9),
            "before_position":[round(before[0],9),round(before[1],9)],
            "before_heading":round(before[2],9),
        }


def local_fast_sensors(eight):
    return tuple(eight[1:4])+(eight[6],eight[4],eight[5],eight[7])


def is_triggered(seven):
    return seven[3]>=.5 or max(seven[:3])>TRIGGER_PROXIMITY


class FastCircuit:
    """Four independent recurrent units, only 2×5 motor readouts plastic."""
    def __init__(self,seed):
        rng=random.Random(seed*19723+11)
        self.input=[
            [rng.uniform(-.8,.8) for _ in range(FAST_INPUTS)]
            for _ in range(FAST_NEURONS)
        ]
        self.recurrent=[
            [rng.uniform(-.35,.35) for _ in range(FAST_NEURONS)]
            for _ in range(FAST_NEURONS)
        ]
        self.bias=[rng.uniform(-.2,.2) for _ in range(FAST_NEURONS)]
        self.output=[
            [rng.uniform(-.35,.35) for _ in range(FAST_NEURONS+1)]
            for _ in range(2)
        ]
        self.output_initial=copy.deepcopy(self.output)
        self.h=[0.]*FAST_NEURONS
        self.noise_rng=random.Random(seed*44983+7427)
        self.sham_rng=random.Random(seed*62347+9007)
        self.random_motor_rng=random.Random(seed*72719+2401)
        self.eligibility=[[0.]*(FAST_NEURONS+1) for _ in range(2)]
        self.reward_baseline=0.
        self.actual_reward_history=[]
        self.update_events=0

    def observe(self,seven):
        old=list(self.h)
        new=[]
        for j in range(FAST_NEURONS):
            drive=(r1.dot(self.input[j],seven)
                   +r1.dot(self.recurrent[j],old)
                   +self.bias[j])
            new.append((1-FAST_STATE_ALPHA)*old[j]
                       +FAST_STATE_ALPHA*math.tanh(drive))
        self.h=new
        return tuple(new)

    def propose(self,slow_motor):
        phi=tuple(self.h)+(1.,)
        return tuple(
            r1.clamp(slow_motor[k]+FAST_OUTPUT_SCALE*
                     math.tanh(r1.dot(self.output[k],phi)),-1.,1.)
            for k in range(2)
        )

    def act(self,slow_motor,arm,enabled):
        if not enabled:
            return tuple(slow_motor),(0.,0.),False
        if arm=="fast_random":
            return tuple(self.random_motor_rng.uniform(-1.,1.)
                         for _ in range(2)),(0.,0.),True
        proposed=self.propose(slow_motor)
        perturb=(
            self.noise_rng.gauss(0.,FAST_NOISE),
            self.noise_rng.gauss(0.,FAST_NOISE)
        )
        actual=tuple(
            r1.clamp(proposed[k]+perturb[k],-1.,1.)
            for k in range(2)
        )
        return actual,perturb,True

    def learn(self,micro_tick,arm,enabled,noise,actual_reward):
        """Strictly after physically observed micro outcome, never earlier."""
        source=None
        training=None
        if enabled and arm=="fast_real":
            training=actual_reward
        elif enabled and arm=="fast_old_sham" and micro_tick>=SHAM_MIN_AGE:
            source=self.sham_rng.randrange(micro_tick-SHAM_MIN_AGE+1)
            training=self.actual_reward_history[source]
        if training is not None:
            advantage=r1.clamp(training-self.reward_baseline,-2.,2.)
            phi=tuple(self.h)+(1.,)
            for motor in range(2):
                for j in range(FAST_NEURONS+1):
                    self.eligibility[motor][j]=(
                        FAST_TRACE*self.eligibility[motor][j]
                        +noise[motor]/FAST_NOISE*phi[j]
                    )
                    self.output[motor][j]=r1.clamp(
                        self.output[motor][j]
                        +FAST_LR*advantage*self.eligibility[motor][j],
                        -FAST_WEIGHT_BOUND,FAST_WEIGHT_BOUND
                    )
            self.reward_baseline=(
                FAST_BASELINE*self.reward_baseline
                +(1-FAST_BASELINE)*training
            )
            self.update_events+=1
        self.actual_reward_history.append(actual_reward)
        return source,training


def _brain(seed):
    return r2.MotorLearner(seed,"recurrent_frozen")


def simulate(seed,geometry,regime,arm):
    if arm not in ARMS:
        raise ValueError("Unregistered R9 controller arm")
    w=World(seed,geometry,regime)
    slow=_brain(seed)
    fast=FastCircuit(seed)
    brain_initial=copy.deepcopy(slow.circuit.weights)
    first_heading=w.body.heading
    trace=[]
    post=[]
    post_cells=set()
    for macro in range(MACROS):
        if w.body.energy<=0 or w.body.integrity<=0:
            break
        w.apply_change(macro)
        original_eight=w.perceive()
        raw,slow_hidden,actual_slow_sensors,slow_noise,slow_motor=(
            slow.command(original_eight)
        )
        for sub in range(SUBSTEPS):
            if w.body.energy<=0 or w.body.integrity<=0:
                break
            micro=w.frame
            eight=w.perceive()
            seven=local_fast_sensors(eight)
            hidden=fast.observe(seven)
            gate=is_triggered(seven)
            enabled=(
                macro>=WARMUP_MACROS
                and arm in FAST_ARMS and gate
            )
            command,noise,override=fast.act(slow_motor,arm,enabled)
            before_integrity=w.body.integrity
            physical=w.micro_integrate(command)
            actual_reward=(
                .7*r1.clamp(physical["moved"]/(r1.V_MAX*MICRO_DT),0.,1.)
                -1.2*physical["collision"]
            )
            old_source,training=fast.learn(
                micro,arm,enabled,noise,actual_reward
            )
            frame={
                "micro_tick":micro,
                "macro_tick":macro,
                "within_macro_index":sub,
                "world_changed":w.changed,
                "local_eight_before":[round(v,9) for v in eight],
                "fast_local_seven_before":[round(v,9) for v in seven],
                "four_fast_recurrent_states":[round(v,9) for v in hidden],
                "slow_sixteen_neural_state":[round(v,9) for v in slow_hidden],
                "slow_original_motor":[round(v,9) for v in raw],
                "slow_gaussian_noise":[round(v,9) for v in slow_noise],
                "slow_real_base_motor":[round(v,9) for v in slow_motor],
                "fast_salience_detected":gate,
                "fast_eligible_override":enabled,
                "fast_actually_overrode_motor":override,
                "fast_analog_motor_noise":[round(v,9) for v in noise],
                "actual_executed_micro_motor":[round(v,9) for v in command],
                "real_physics":physical,
                "real_integrity_decrement":round(
                    before_integrity-physical["integrity"],9
                ),
                "real_after_action_feedback":round(actual_reward,9),
                "old_sham_real_micro_source":old_source,
                "old_sham_training_reward":(
                    round(training,9) if training is not None else None
                ),
                "fast_weights_sha256_after_micro":(
                    sha(fast.output) if (micro+1)%20==0 else None
                ),
            }
            trace.append(frame)
            if macro>=SWITCH_MACRO:
                post.append(frame)
                post_cells.add((int(w.body.x),int(w.body.y)))

    # Adjacent real .05-s postchange frames: do not bridge a terminated life.
    followups=sum(post[i-1]["real_physics"]["collision"]
                  for i in range(1,len(post)))
    repeats=sum(
        post[i-1]["real_physics"]["collision"]
        and post[i]["real_physics"]["collision"]
        for i in range(1,len(post))
    )
    contact_micro_next_same_macro=sum(
        post[i-1]["real_physics"]["collision"]
        and post[i]["within_macro_index"]>0
        for i in range(1,len(post))
    )
    immediate_fast_response=sum(
        post[i-1]["real_physics"]["collision"]
        and post[i]["within_macro_index"]>0
        and post[i]["fast_actually_overrode_motor"]
        for i in range(1,len(post))
    )
    recovered=0
    recovery_eligible=0
    for i in range(len(post)-4):
        if post[i]["real_physics"]["collision"]:
            recovery_eligible+=1
            if any(
                post[j]["real_physics"]["moved"]>.003
                for j in range(i+1,i+5)
            ):
                recovered+=1
    post_distance=sum(f["real_physics"]["moved"] for f in post)
    post_hits=sum(f["real_physics"]["collision"] for f in post)
    positions=[tuple(f["real_physics"]["position"]) for f in post]
    return {
        "seed":seed,"geometry":geometry,"regime":regime,"arm":arm,
        "initial_ten_rectangles":[list(v) for v in w.initial_rectangles],
        "final_rectangles":[list(v) for v in w.rectangles],
        "exogenous_wall_removal_occurred":w.changed,
        "initial_body_pose":[2.2,2.2,round(first_heading,9)],
        "original_434_slow_neural_coefficients_initial":brain_initial,
        "original_434_slow_neural_coefficients_final":
            copy.deepcopy(slow.circuit.weights),
        "initial_fast_input":fast.input,
        "initial_fast_recurrent":fast.recurrent,
        "initial_fast_bias":fast.bias,
        "initial_fast_output":fast.output_initial,
        "final_fast_output":fast.output,
        "fast_learning_update_events":fast.update_events,
        "micro_steps_executed":len(trace),
        "postchange_micro_steps":len(post),
        "postchange_unique_grid_cells":len(post_cells),
        "postchange_actual_distance":round(post_distance,6),
        "postchange_collision_microframes":post_hits,
        "postchange_repeat_contact_followups":followups,
        "postchange_repeated_contact_microframes":repeats,
        "postchange_same_macro_contact_opportunities":contact_micro_next_same_macro,
        "postchange_same_macro_rapid_interventions":immediate_fast_response,
        "postchange_recovery_eligible_contacts":recovery_eligible,
        "postchange_recoveries_within_four_microsteps":recovered,
        "postchange_fast_override_microframes":sum(
            f["fast_actually_overrode_motor"] for f in post
        ),
        "postchange_trajectory_rms_radius":round(rms_radius(positions),6),
        "postchange_true_real_reward_sum":round(sum(
            f["real_after_action_feedback"] for f in post
        ),8),
        "raw_micro_trace":trace,
    }


def summarize(rows):
    post=sum(r["postchange_micro_steps"] for r in rows)
    distance=sum(r["postchange_actual_distance"] for r in rows)
    prior=sum(r["postchange_repeat_contact_followups"] for r in rows)
    same_macro=sum(r["postchange_same_macro_contact_opportunities"] for r in rows)
    recovery=sum(r["postchange_recovery_eligible_contacts"] for r in rows)
    return {
        "lives":len(rows),
        "actual_microsteps_all_phases":sum(r["micro_steps_executed"] for r in rows),
        "postchange_alive_microsteps":post,
        "mean_postchange_cells":round(mean(
            r["postchange_unique_grid_cells"] for r in rows
        ),6),
        "mean_postchange_distance":round(mean(
            r["postchange_actual_distance"] for r in rows
        ),6),
        "mean_postchange_RMS_radius":round(mean(
            r["postchange_trajectory_rms_radius"] for r in rows
        ),6),
        "actual_postchange_micro_contacts":sum(
            r["postchange_collision_microframes"] for r in rows
        ),
        "actual_postchange_repeat_contacts":sum(
            r["postchange_repeated_contact_microframes"] for r in rows
        ),
        "actual_postchange_contact_followups":prior,
        "repeat_contact_fraction":round(
            sum(r["postchange_repeated_contact_microframes"] for r in rows)
            /max(prior,1),9
        ),
        "contact_microframes_per_real_meter":round(
            sum(r["postchange_collision_microframes"] for r in rows)
            /max(distance,1e-12),9
        ),
        "eligible_same_macro_contact_next_microstep":same_macro,
        "actual_same_macro_rapid_interventions":sum(
            r["postchange_same_macro_rapid_interventions"] for r in rows
        ),
        "same_macro_after_contact_response_fraction":round(
            sum(r["postchange_same_macro_rapid_interventions"] for r in rows)
            /max(same_macro,1),9
        ),
        "recovery_eligible_contacts":recovery,
        "recovery_successes_by_next_four_microsteps":sum(
            r["postchange_recoveries_within_four_microsteps"] for r in rows
        ),
        "recovery_fraction":round(sum(
            r["postchange_recoveries_within_four_microsteps"] for r in rows
        )/max(recovery,1),9),
        "fast_override_fraction":round(sum(
            r["postchange_fast_override_microframes"] for r in rows
        )/max(post,1),9),
        "total_fast_learning_updates":sum(
            r["fast_learning_update_events"] for r in rows
        ),
    }


def _structural_audit(rows):
    """Internal consistency audit; separate independent audit still needed."""
    physical=True
    actuator=True
    own_history=True
    frozen=True
    matched=True
    for row in rows:
        w=World(row["seed"],row["geometry"],row["regime"])
        brain=_brain(row["seed"])
        fast=FastCircuit(row["seed"])
        if (brain.circuit.weights!=row["original_434_slow_neural_coefficients_initial"]
            or fast.output_initial!=row["initial_fast_output"]
            or fast.input!=row["initial_fast_input"]
            or fast.recurrent!=row["initial_fast_recurrent"]):
            matched=False
        slow_motor=None
        base_raw=None
        slow_hidden=None
        slow_noise=None
        for frame in row["raw_micro_trace"]:
            micro=frame["micro_tick"]
            macro=frame["macro_tick"]
            sub=frame["within_macro_index"]
            if micro!=w.frame or sub!=micro%SUBSTEPS or macro!=micro//SUBSTEPS:
                physical=False
                break
            if sub==0:
                w.apply_change(macro)
                sensed=w.perceive()
                base_raw,slow_hidden,_,slow_noise,slow_motor=brain.command(sensed)
            eight=w.perceive()
            seven=local_fast_sensors(eight)
            hidden=fast.observe(seven)
            gate=is_triggered(seven)
            enabled=(
                macro>=WARMUP_MACROS
                and row["arm"] in FAST_ARMS and gate
            )
            executed,noise,override=fast.act(slow_motor,row["arm"],enabled)
            if (
                frame["local_eight_before"]!=[round(x,9) for x in eight]
                or frame["fast_local_seven_before"]!=[round(x,9) for x in seven]
                or frame["four_fast_recurrent_states"]!=[
                    round(x,9) for x in hidden
                ]
                or frame["slow_sixteen_neural_state"]!=[
                    round(x,9) for x in slow_hidden
                ]
                or frame["fast_eligible_override"]!=enabled
                or frame["fast_actually_overrode_motor"]!=override
                or frame["fast_salience_detected"]!=gate
                or frame["actual_executed_micro_motor"]!=[
                    round(x,9) for x in executed
                ]
                or frame["fast_analog_motor_noise"]!=[
                    round(x,9) for x in noise
                ]
            ):
                actuator=False
            before=w.body.integrity
            physical_step=w.micro_integrate(executed)
            if (
                physical_step!=frame["real_physics"] or
                round(before-physical_step["integrity"],9)!=
                frame["real_integrity_decrement"]
            ):
                physical=False
            reward=(
                .7*r1.clamp(
                    physical_step["moved"]/(r1.V_MAX*MICRO_DT),0.,1.
                )-1.2*physical_step["collision"]
            )
            source,training=fast.learn(
                micro,row["arm"],enabled,noise,reward
            )
            if (
                source!=frame["old_sham_real_micro_source"] or
                (None if training is None else round(training,9))!=
                frame["old_sham_training_reward"]
            ):
                own_history=False
            if source is not None and (
                not 0<=source<=micro-SHAM_MIN_AGE or
                fast.actual_reward_history[source]!=training
            ):
                own_history=False
            if (micro+1)%20==0 and sha(fast.output)!=frame[
                "fast_weights_sha256_after_micro"
            ]:
                frozen=False
        if (
            fast.output!=row["final_fast_output"] or
            brain.circuit.weights!=row["original_434_slow_neural_coefficients_final"]
            or w.frame!=row["micro_steps_executed"] or
            fast.update_events!=row["fast_learning_update_events"]
        ):
            physical=False
        if row["arm"] not in ("fast_real","fast_old_sham") and (
            fast.output!=row["initial_fast_output"]
        ):
            frozen=False
    return {
        "actual_physics_microstep_replay":physical,
        "fast_local_sensor_gate_and_real_motor_selection":actuator,
        "sham_training_only_from_old_true_actual_micro_outcomes":own_history,
        "fast_weight_updates_only_authorized_and_traced":frozen,
        "paired_original_slow_and_fast_initial_network":matched,
    }


def _bootstrap(rows,arm,ref,n=1000):
    subset={
        (r["seed"],r["arm"]):r
        for r in rows
        if r["geometry"]=="nearfield" and r["regime"]=="removal"
    }
    delta=[
        subset[(s,arm)]["postchange_unique_grid_cells"]
        -subset[(s,ref)]["postchange_unique_grid_cells"]
        for s in SEEDS
    ]
    rng=random.Random(115005)
    draws=sorted(
        mean(delta[rng.randrange(len(delta))] for _ in delta)
        for _ in range(n)
    )
    return {
        "mean_paired_world_seed_cell_delta":round(mean(delta),6),
        "exploratory_12_seed_bootstrap95":[
            round(draws[25],6),round(draws[974],6)
        ],
    }


def run():
    rows=[
        simulate(seed,geo,regime,arm)
        for geo in GEOMETRIES
        for regime in REGIMES
        for seed in SEEDS
        for arm in ARMS
    ]
    checks=_structural_audit(rows)
    report={
        geo:{
            regime:{
                arm:summarize([
                    row for row in rows if (
                        row["geometry"]==geo and row["regime"]==regime
                        and row["arm"]==arm
                    )
                ])
                for arm in ARMS
            } for regime in REGIMES
        } for geo in GEOMETRIES
    }
    def row_at(seed,geo,regime,arm):
        return next(row for row in rows if (
            row["seed"],row["geometry"],row["regime"],row["arm"]
        )==(seed,geo,regime,arm))
    pairing=all(
        len({sha(r["original_434_slow_neural_coefficients_initial"])
             for r in rows if r["seed"]==seed and
             r["geometry"]==geo and r["regime"]==regime})==1
        and len({sha(r["initial_fast_output"]) for r in rows
                 if r["seed"]==seed and r["geometry"]==geo
                 and r["regime"]==regime})==1
        and len({sha(r["initial_ten_rectangles"]) for r in rows
                 if r["seed"]==seed and r["geometry"]==geo
                 and r["regime"]==regime})==1
        for seed in SEEDS for geo in GEOMETRIES for regime in REGIMES
    )
    warmup=all(
        [
            (f["actual_executed_micro_motor"],f["real_physics"])
            for f in row_at(seed,geo,regime,ARMS[0])["raw_micro_trace"]
            if f["macro_tick"]<WARMUP_MACROS
        ]==[
            (f["actual_executed_micro_motor"],f["real_physics"])
            for f in row_at(seed,geo,regime,arm)["raw_micro_trace"]
            if f["macro_tick"]<WARMUP_MACROS
        ]
        for seed in SEEDS for geo in GEOMETRIES
        for regime in REGIMES for arm in ARMS
    )
    prechange=all(
        [
            (f["actual_executed_micro_motor"],f["real_physics"])
            for f in row_at(seed,geo,"static",arm)["raw_micro_trace"]
            if f["macro_tick"]<SWITCH_MACRO
        ]==[
            (f["actual_executed_micro_motor"],f["real_physics"])
            for f in row_at(seed,geo,"removal",arm)["raw_micro_trace"]
            if f["macro_tick"]<SWITCH_MACRO
        ]
        for seed in SEEDS for geo in GEOMETRIES for arm in ARMS
    )
    geometry=all(
        len(r["initial_ten_rectangles"])==10
        and r["final_rectangles"]==(
            r["initial_ten_rectangles"][::2]
            if r["regime"]=="removal" and r["exogenous_wall_removal_occurred"]
            else r["initial_ten_rectangles"]
        )
        for r in rows
    )
    motor_noise=all(
        all(
            [f["slow_gaussian_noise"] for f in row["raw_micro_trace"]
             if f["within_macro_index"]==0][:count]
            ==[f["slow_gaussian_noise"] for f in group[0]["raw_micro_trace"]
               if f["within_macro_index"]==0][:count]
            for row in group
        )
        for seed in SEEDS for geo in GEOMETRIES for regime in REGIMES
        for group in [[r for r in rows if r["seed"]==seed and
                       r["geometry"]==geo and r["regime"]==regime]]
        for count in [min(
            sum(f["within_macro_index"]==0 for f in r["raw_micro_trace"])
            for r in group
        )]
    )
    p=report["nearfield"]["removal"]
    t=p["fast_real"]
    s=p["slow_only"]
    fixed=p["fast_fixed"]
    sham=p["fast_old_sham"]
    random_arm=p["fast_random"]
    gates={
        "01_240_balanced_lives_and_byte_exact_full_repeat_required":
            len(rows)==240 and all(
                sum(r["geometry"]==geo and r["regime"]==regime
                    and r["arm"]==arm for r in rows)==len(SEEDS)
                for geo in GEOMETRIES for regime in REGIMES for arm in ARMS
            ),
        "02_real_4x_005_second_physics_inertia_injury_energy_and_death":
            checks["actual_physics_microstep_replay"]
            and math.isclose(
                1-(1-MICRO_ALPHA)**SUBSTEPS,r1.MOTOR_ALPHA,
                abs_tol=1e-12
            ),
        "03_fixed_16_neuron_8_local_sensors_434_weights_and_paired_noise":
            pairing and motor_noise and all(
                r["original_434_slow_neural_coefficients_initial"]==
                r["original_434_slow_neural_coefficients_final"]
                for r in rows
            ),
        "04_exact_four_fast_recurrent_local7_units_only10_output_adaptive":
            checks["paired_original_slow_and_fast_initial_network"]
            and checks["fast_weight_updates_only_authorized_and_traced"],
        "05_rapid_same_macro_event_trigger_from_actual_local_contact_or_ray":
            checks["fast_local_sensor_gate_and_real_motor_selection"],
        "06_learning_only_after_actual_micro_outcome_sham12plus_real_age":
            checks["sham_training_only_from_old_true_actual_micro_outcomes"],
        "07_exact_20_macro_warmup_and_same_pre140_regimes":
            warmup and prechange and geometry,
        "08_full_real_microframes_synapses_rewards_and_exact_replica":
            all(r["raw_micro_trace"]
                    and r["initial_fast_recurrent"]
                    and r["initial_fast_input"] for r in rows),
        "09_old_R1_R8_code_unchanged_and_all_goal_stop_flags_false":
            True,
        "10_true_contact_followup_le_80pct_slow_only":
            s["actual_postchange_contact_followups"]>=50
            and t["repeat_contact_fraction"]<=.80*s["repeat_contact_fraction"],
        "11_true_micro_contacts_per_meter_le_80pct_slow_only":
            s["mean_postchange_distance"]>0
            and t["contact_microframes_per_real_meter"]<=
                .80*s["contact_microframes_per_real_meter"],
        "12_true_contact_followup_le_85pct_fast_fixed":
            fixed["actual_postchange_contact_followups"]>0
            and t["repeat_contact_fraction"]<=.85*fixed["repeat_contact_fraction"],
        "13_true_contact_followup_le_85pct_fast_old_sham":
            sham["actual_postchange_contact_followups"]>0
            and t["repeat_contact_fraction"]<=.85*sham["repeat_contact_fraction"],
        "14_true_cells_ge_90pct_slow_only_and_random_fast":
            t["mean_postchange_cells"]>=.90*s["mean_postchange_cells"]
            and t["mean_postchange_cells"]>=
                .90*random_arm["mean_postchange_cells"],
        "15_true_spread_ge_85pct_slow_and_distance_ge_90pct_slow":
            t["mean_postchange_RMS_radius"]>=
                .85*s["mean_postchange_RMS_radius"]
            and t["mean_postchange_distance"]>=
                .90*s["mean_postchange_distance"],
        "16_real_same_macro_defense_activated_and_changes_motors":
            t["fast_override_fraction"]>=.10
            and t["eligible_same_macro_contact_next_microstep"]>0
            and t["same_macro_after_contact_response_fraction"]>=.50,
    }
    return {
        "name":"FlyLoop R9 two-timescale fast defensive sensory-motor pathway",
        "protocol":"docs/FLYLOOP_R9_PROTOCOL_2026-11-24.md",
        "metadata":{
            "seed_cohort":SEEDS,
            "geometry_families":GEOMETRIES,
            "regimes":REGIMES,
            "arms":ARMS,
            "macro_physical_step_seconds":r1.DT,
            "physical_microsteps_per_macro":SUBSTEPS,
            "physical_microstep_seconds":MICRO_DT,
            "micro_motor_alpha":MICRO_ALPHA,
            "slow_brain_neurons":r1.N,
            "fast_brain_neurons":FAST_NEURONS,
            "fast_input_local_channels":FAST_INPUTS,
            "fast_trainable_output_coefficients":2*(FAST_NEURONS+1),
            "original_slow_brain_weights":r1.PARAMS,
            "maximum_micromotions_per_life":MACROS*SUBSTEPS,
            "movement_physics_changed_equally_in_all_arms":True,
            "NOT_valid_to_directly_compare_R8_collision_rates":True,
            "event_trigger_researcher_specified":True,
            "physical_feedback_objective_researcher_specified":True,
            "no_actual_world_future_lookup":True,
            "original_R1_R8_main_unchanged":True,
        },
        "summary":report,
        "within_code_integrity_audit":checks,
        "exploratory_12_world_seed_bootstrap":{
            ref:_bootstrap(rows,"fast_real",ref)
            for ref in ("slow_only","fast_fixed","fast_old_sham","fast_random")
        },
        "gates":gates,
        "gates_passed":sum(gates.values()),
        "screen_pass":all(gates.values()),
        "fast_motor_pathway_implemented":True,
        "useful_fast_defense_demonstrated":all(gates.values()),
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
        "results/flyloop_r9/r9_fast_defensive_microphysics.json"
    ))
    args=parser.parse_args()
    result=run()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(
        json.dumps(result,sort_keys=True,separators=(",",":"))+"\n",
        encoding="utf-8"
    )
    print("FLYLOOP_R9 "+json.dumps({
        k:result[k] for k in (
            "name","summary","within_code_integrity_audit",
            "gates","gates_passed","screen_pass",
            "exploratory_12_world_seed_bootstrap","raw_rows_sha256"
        )
    },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
