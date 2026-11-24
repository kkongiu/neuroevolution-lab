"""FlyLoop R2: motor-readout plasticity from ACTUAL physical displacement/contact.

FROZEN PROTOCOL: docs/FLYLOOP_R2_PROTOCOL_2026-11-24.md
R1 code and main stay unmodified. No food, prewritten navigation,
evolution or learned objective. THE REWARD AND ELIGIBILITY RULE
ARE RESEARCHER-DEFINED, NOT EMERGENT.
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

SEEDS=tuple(range(9400,9424))
SCENES=("obstacles","nearfield")
ARMS=(
    "recurrent_frozen","recurrent_plastic","recurrent_sham",
    "collision_only","leaky_plastic","inertial_random"
)
DURATION=r1.HORIZON
WINDOW=12
EXPLORATION_SD=.12
LR=.0018
TRACE_LEAK=.85
BASELINE_LEAK=.97
HIT_COST=1.2
UPDATE_LIMIT=3.0
MAX_ADV=2.0
N_MOTOR_COEFF=r1.N+1

def mean(x):
    x=list(x)
    return sum(x)/len(x) if x else 0.

def sha(x):
    return hashlib.sha256(
        json.dumps(x,sort_keys=True,separators=(",",":")).encode()
    ).hexdigest()


def nearfield_rectangles(seed):
    """Exogenous map generator concentrated nearer spawn, NOT reward-aware."""
    rng=random.Random(seed*83791+1731)
    rectangles=[]
    for i in range(r1.OBSTACLES):
        for _ in range(4000):
            cx=rng.uniform(3.3,11.8)
            cy=rng.uniform(3.3,11.8)
            width=rng.uniform(.8,1.55)
            height=rng.uniform(.8,1.55)
            rect=(
                round(cx-width/2,6),round(cy-height/2,6),
                round(cx+width/2,6),round(cy+height/2,6),
            )
            if (rect[0]-1.1<2.2<rect[2]+1.1 and
                rect[1]-1.1<2.2<rect[3]+1.1):
                continue
            if any(
                rect[0]<b[2]+.38 and rect[2]>b[0]-.38
                and rect[1]<b[3]+.38 and rect[3]>b[1]-.38
                for b in rectangles
            ):
                continue
            rectangles.append(rect)
            break
        else:
            raise RuntimeError("Failed to generate nearfield map")
    return rectangles


class World(r1.World):
    """R1 physical World methods, new seeds and independent map generator."""
    def __init__(self,seed,scene):
        if seed not in SEEDS or scene not in SCENES:
            raise ValueError("Unregistered R2 world seed or scene")
        self.seed=seed
        self.scenario=scene
        self.rectangles=(
            r1.procedural_rectangles(seed)
            if scene=="obstacles" else nearfield_rectangles(seed)
        )
        heading=random.Random(seed*11113+51).uniform(-math.pi,math.pi)
        self.body=r1.Body(heading=heading)
        self.frame=0
        self.visited={(int(self.body.x),int(self.body.y))}
        self.collisions=0
        self.distance=0.
        self.trace=[]


class MotorLearner:
    """Only motor synapses/biases adapt; recurrent core fixed for life."""

    def __init__(self,seed,arm):
        if arm not in ARMS:
            raise ValueError("Unregistered R2 controller")
        self.seed=seed
        self.arm=arm
        r1_arm=(
            "inertial_random" if arm=="inertial_random"
            else "leaky_no_recurrence" if arm=="leaky_plastic"
            else "recurrent"
        )
        self.circuit=r1.Circuit(seed,r1_arm)
        self.initial_coefficients=copy.deepcopy(self.circuit.weights)
        # Exogenous motor-noise stream: same draws for all five neural arms.
        self.noise_rng=random.Random(seed*27449+4501)
        # Separate sham sampler RNG; NEVER advances motor draws.
        self.sham_rng=random.Random(seed*11839+331)
        self.eligibility=[[0.]*N_MOTOR_COEFF for _ in range(2)]
        self.reward_baseline=0.
        self.real_feedback_history=[]
        self.updates=0

    def command(self,sensors):
        raw,h,perceived=self.circuit.forward(sensors)
        if self.arm=="inertial_random":
            perturb=(0.,0.)
            actuated=raw
        else:
            perturb=(
                self.noise_rng.gauss(0.,EXPLORATION_SD),
                self.noise_rng.gauss(0.,EXPLORATION_SD),
            )
            actuated=tuple(
                r1.clamp(raw[k]+perturb[k],-1.,1.)
                for k in range(2)
            )
        return raw,h,perceived,perturb,actuated

    def adapt(self,tick,physical,collision,positions,state,perturb):
        """Real feedback is NEVER known before World.integrate()."""
        lag=min(WINDOW,len(positions)-1)
        a=positions[-1-lag]
        b=positions[-1]
        displacement=math.hypot(b[0]-a[0],b[1]-a[1])
        # max speed bound times actual number of steps in the window
        progress=displacement/(r1.V_MAX*r1.DT*lag)
        actual=progress-HIT_COST*collision
        source=None
        if self.arm=="recurrent_sham":
            # Sample from the SAME life's older ACTUAL feedback,
            # excluding the most recent 4 observations, and current.
            eligible=tick-4
            if eligible>0:
                source=self.sham_rng.randrange(eligible)
                training=self.real_feedback_history[source]
            else:
                training=0.
        elif self.arm=="collision_only":
            training=-HIT_COST*collision
            source=tick
        else:
            training=actual
            if self.arm in ("recurrent_plastic","leaky_plastic"):
                source=tick

        update_enabled=self.arm in (
            "recurrent_plastic","recurrent_sham",
            "collision_only","leaky_plastic",
        )
        previous_baseline=self.reward_baseline
        advantage=r1.clamp(training-previous_baseline,-MAX_ADV,MAX_ADV)
        if update_enabled:
            # Simplified signed stochastic perturbation eligibility.
            # At clipped command limits this is a biased estimator.
            features=tuple(state)+(1.,)
            for motor in range(2):
                for j in range(N_MOTOR_COEFF):
                    self.eligibility[motor][j]=(
                        TRACE_LEAK*self.eligibility[motor][j]
                        +(perturb[motor]/EXPLORATION_SD)*features[j]
                    )
                for j in range(r1.N):
                    previous=self.circuit.weights["w_motor"][motor][j]
                    self.circuit.weights["w_motor"][motor][j]=r1.clamp(
                        previous+LR*advantage*self.eligibility[motor][j],
                        -UPDATE_LIMIT,UPDATE_LIMIT
                    )
                oldbias=self.circuit.weights["motor_bias"][motor]
                self.circuit.weights["motor_bias"][motor]=r1.clamp(
                    oldbias+LR*advantage*self.eligibility[motor][-1],
                    -UPDATE_LIMIT,UPDATE_LIMIT
                )
            self.updates+=1
            self.reward_baseline=(
                BASELINE_LEAK*previous_baseline
                +(1.-BASELINE_LEAK)*training
            )
        self.real_feedback_history.append(actual)
        return {
            "physical_progress":round(progress,9),
            "actual_feedback":round(actual,9),
            "training_feedback":round(training,9),
            "feedback_source_tick":source,
            "advantage_after_baseline":round(advantage,9),
            "previous_reward_baseline":round(previous_baseline,9),
            "eligibility_traces":[
                [round(x,8) for x in row] for row in self.eligibility
            ],
            "motor_coefficients_after_action":[
                [round(v,9) for v in row]+[
                    round(self.circuit.weights["motor_bias"][k],9)
                ] for k,row in enumerate(self.circuit.weights["w_motor"])
            ],
        }


def simulate(seed,scene,arm):
    w=World(seed,scene)
    brain=MotorLearner(seed,arm)
    positions=[(w.body.x,w.body.y)]
    original_heading=w.body.heading
    for tick in range(DURATION):
        if w.body.energy<=0 or w.body.integrity<=0:
            break
        sensory=w.perceive()
        before=brain.circuit.weights
        raw,neurons,perceived,noise,command=brain.command(sensory)
        outcome=w.integrate(command)
        positions.append((w.body.x,w.body.y))
        feedback=brain.adapt(
            tick,outcome["moved"],outcome["collision"],
            positions,neurons,noise,
        )
        outcome.update({
            "tick":tick,
            "sensor_inputs":[round(v,9) for v in perceived],
            "neural_state":[round(v,9) for v in neurons],
            "base_motor_command":[round(v,9) for v in raw],
            "injected_motor_noise":[round(v,9) for v in noise],
            "actual_motor_command":[round(v,9) for v in command],
            **feedback,
        })
        w.trace.append(outcome)
    final_coefficients=copy.deepcopy(brain.circuit.weights)
    path=list(positions)
    centroid=(mean(p[0] for p in path),mean(p[1] for p in path))
    rms_radius=math.sqrt(mean(
        (p[0]-centroid[0])**2+(p[1]-centroid[1])**2 for p in path
    ))
    return {
        "seed":seed,"scene":scene,"arm":arm,
        "rectangles":[list(v) for v in w.rectangles],
        "initial_pose":[2.2,2.2,round(original_heading,9)],
        "initial_coefficients":brain.initial_coefficients,
        "final_coefficients":final_coefficients,
        "initial_coefficients_sha256":sha(brain.initial_coefficients),
        "final_coefficients_sha256":sha(final_coefficients),
        "update_events":brain.updates,
        "steps_alive":len(w.trace),
        "distance":round(w.distance,6),
        "net_displacement":round(
            math.hypot(w.body.x-2.2,w.body.y-2.2),6
        ),
        "trajectory_rms_radius":round(rms_radius,6),
        "unique_grid_cells":len(w.visited),
        "collisions":w.collisions,
        "collisions_per_meter":round(
            w.collisions/max(w.distance,1.e-12),6
        ),
        "mean_absolute_turn":round(
            mean(abs(row["turn"]) for row in w.trace),6
        ),
        "mean_actual_reward":round(
            mean(row["actual_feedback"] for row in w.trace),6
        ),
        "terminal_energy":round(w.body.energy,6),
        "terminal_integrity":round(w.body.integrity,6),
        "trace":w.trace,
    }


def _paired_bootstrap(rows,scene,field,arm,ref,draws=1000):
    index={(row["seed"],row["arm"]):row for row in rows
           if row["scene"]==scene}
    d=[
        index[(s,arm)][field]-index[(s,ref)][field]
        for s in SEEDS
    ]
    rng=random.Random(94111)
    samples=sorted(mean(
        d[rng.randrange(len(d))] for _ in d
    ) for _ in range(draws))
    return {
        "mean_paired_delta":round(mean(d),6),
        "exploratory_seed_bootstrap95":[
            round(samples[25],6),round(samples[974],6),
        ],
    }


def _audit(rows):
    physics=True
    same_inputs=True
    learning=True
    correct_reward=True
    proper_sham=True
    legal_sensors=True
    for life in rows:
        position=(2.2,2.2)
        past_positions=[position]
        prev_energy=100.
        prev_integrity=100.
        prev_heading=life["initial_pose"][2]
        prev_speed=prev_turn=0.
        past_rewards=[]
        for t,row in enumerate(life["trace"]):
            if row["tick"]!=t or tuple(row["before_position"])!=(
                round(position[0],9),round(position[1],9)
            ):
                physics=False
            if not (len(row["sensor_inputs"])==r1.INPUTS
                and len(row["neural_state"])==r1.N
                and len(row["actual_motor_command"])==2):
                legal_sensors=False
            if any(not -1.-1e-8<=v<=1.+1e-8
                   for v in row["actual_motor_command"]):
                physics=False
            oldpos=position
            position=tuple(row["position"])
            if row["collision"] and math.hypot(
                position[0]-oldpos[0],position[1]-oldpos[1]
            )>1.e-7:
                physics=False
            if (not(r1.RADIUS-1.e-7<=position[0]<=
                    r1.SIDE-r1.RADIUS+1.e-7)
                or not(r1.RADIUS-1.e-7<=position[1]<=
                       r1.SIDE-r1.RADIUS+1.e-7)):
                physics=False
            for x1,y1,x2,y2 in life["rectangles"]:
                if x1-r1.RADIUS<position[0]<x2+r1.RADIUS and (
                    y1-r1.RADIUS<position[1]<y2+r1.RADIUS
                ):
                    physics=False
            expected_speed=(1.-r1.MOTOR_ALPHA)*prev_speed+(
                r1.MOTOR_ALPHA*r1.V_MAX*r1.sigmoid(
                    1.6*row["actual_motor_command"][0]
                )
            )
            expected_turn=(1.-r1.MOTOR_ALPHA)*prev_turn+(
                r1.MOTOR_ALPHA*r1.W_MAX*math.tanh(
                    1.6*row["actual_motor_command"][1]
                )
            )
            expected_heading=(
                prev_heading+expected_turn*r1.DT+math.pi
            )%(2*math.pi)-math.pi
            if any(abs(a-b)>2.e-7 for a,b in (
                (row["speed"],expected_speed),
                (row["turn"],expected_turn),
                (row["heading"],expected_heading),
            )):
                physics=False
            prev_speed,prev_turn,prev_heading=(
                expected_speed,expected_turn,expected_heading
            )
            prev_energy=max(0.,prev_energy-r1.ENERGY_PER_TICK)
            prev_integrity=max(
                0.,prev_integrity-row["collision"]*r1.COLLISION_INJURY
            )
            if (abs(prev_energy-row["energy"])>1.e-7 or
                abs(prev_integrity-row["integrity"])>1.e-7):
                physics=False
            past_positions.append(position)
            lag=min(WINDOW,len(past_positions)-1)
            prev=past_positions[-1-lag]
            physical_progress=math.hypot(
                position[0]-prev[0],position[1]-prev[1]
            )/(r1.V_MAX*r1.DT*lag)
            actual=physical_progress-HIT_COST*row["collision"]
            if (abs(physical_progress-row["physical_progress"])>1.e-7
                or abs(actual-row["actual_feedback"])>1.e-7):
                correct_reward=False
            if life["arm"]=="recurrent_sham":
                source=row["feedback_source_tick"]
                if source is None and t>=5:
                    proper_sham=False
                if source is not None and (
                    source>t-5 or source<0 or source>=len(past_rewards)
                ):
                    proper_sham=False
                elif source is not None and abs(
                    row["training_feedback"]-past_rewards[source]
                )>1.e-7:
                    proper_sham=False
            if life["arm"]=="collision_only" and abs(
                row["training_feedback"]+HIT_COST*row["collision"]
            )>1.e-7:
                correct_reward=False
            if life["arm"] in ("recurrent_plastic","leaky_plastic"
            ) and abs(row["training_feedback"]-actual)>1.e-7:
                correct_reward=False
            past_rewards.append(actual)
        orig=life["initial_coefficients"]
        final=life["final_coefficients"]
        for field in ("w_input","w_recurrent","bias"):
            if orig[field]!=final[field]:
                learning=False
        if (life["arm"] in ("recurrent_frozen","inertial_random")):
            if orig!=final or life["update_events"]!=0:
                learning=False
        elif life["update_events"]!=life["steps_alive"] or orig==final:
            learning=False
    return {
        "legal_continuous_physics":physics,
        "local_inputs_only":legal_sensors,
        "only_motor_coefficients_adapt":learning,
        "reward_from_actual_physical_progress":correct_reward,
        "older_than_four_sham_feedback":proper_sham,
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
            subset=[r for r in rows if r["scene"]==scene and r["arm"]==arm]
            summary[scene][arm]={
                field:round(mean(r[field] for r in subset),6)
                for field in (
                    "steps_alive","distance","net_displacement",
                    "trajectory_rms_radius","unique_grid_cells",
                    "collisions","mean_absolute_turn",
                    "mean_actual_reward","terminal_energy",
                    "terminal_integrity"
                )
            }|{
                "pooled_collisions_per_meter":round(
                    sum(r["collisions"] for r in subset)/
                    max(sum(r["distance"] for r in subset),1.e-12),6
                ),
                "total_distance_m":round(
                    sum(r["distance"] for r in subset),6
                ),
            }
    integrity=_audit(rows)
    initial_equal=all(
        len({sha(r["initial_coefficients"]) for r in rows
             if r["seed"]==s and r["scene"]==sc})==1
        and len({sha(r["initial_pose"]) for r in rows
                 if r["seed"]==s and r["scene"]==sc})==1
        and len({sha(r["rectangles"]) for r in rows
                 if r["seed"]==s and r["scene"]==sc})==1
        for s in SEEDS for sc in SCENES
    )
    map_counts={
        sc:len({sha(r["rectangles"]) for r in rows if r["scene"]==sc})
        for sc in SCENES
    }
    geometry_valid=all(
        len(r["rectangles"])==r1.OBSTACLES
        and all(not(
            x1-1.1<2.2<x2+1.1 and
            y1-1.1<2.2<y2+1.1
        ) for x1,y1,x2,y2 in r["rectangles"])
        for r in rows
    )
    # Each arm may DIE at a different time. Compare the shared LIVE
    # prefix of the exogenous noise stream, not unequal full lifetimes.
    noise_equal=all(
        all(
            [x["injected_motor_noise"] for x in r["trace"][:minimum]]
            ==[x["injected_motor_noise"] for x in group[0]["trace"][:minimum]]
            for r in group
        )
        for sc in SCENES for s in SEEDS
        for group in [[r for r in rows
                       if r["seed"]==s and r["scene"]==sc
                       and r["arm"]!="inertial_random"]]
        for minimum in [min(len(r["trace"]) for r in group)]
    )
    near=summary["nearfield"]
    gates={
        "01_288_balanced_lives_second_run_required":
            len(rows)==288 and all(
                sum(r["scene"]==sc and r["arm"]==arm for r in rows)
                ==len(SEEDS)
                for sc in SCENES for arm in ARMS
            ),
        "02_true_continuous_R1_physics_no_resurrection":
            integrity["legal_continuous_physics"],
        "03_identical_paired_maps_pose_and_434_initial_coefficients":
            initial_equal,
        "04_recurrent_leaky_8_local_sensors_without_map_hint":
            integrity["local_inputs_only"],
        "05_only_34_output_motor_coefficients_can_adapt":
            integrity["only_motor_coefficients_adapt"],
        "06_feedback_from_actual_past_12_steps_and_collision":
            integrity["reward_from_actual_physical_progress"],
        "07_identical_motor_noise_older_only_sham":
            noise_equal and integrity["older_than_four_sham_feedback"],
        "08_exogenous_valid_10_rectangles_20_distinct_maps":
            geometry_valid and all(n>=20 for n in map_counts.values()),
        "09_raw_audit_no_food_evolution_and_stop_flags":
            len(rows)==288 and all(r["trace"] and r["initial_coefficients"]
                                   and r["final_coefficients"] for r in rows),
        "10_nearfield_plastic_coverage_ge_frozen_plus10pct":
            near["recurrent_plastic"]["unique_grid_cells"]>=
            1.10*near["recurrent_frozen"]["unique_grid_cells"],
        "11_nearfield_plastic_collisions_per_meter_le_frozen_90pct":
            near["recurrent_plastic"]["pooled_collisions_per_meter"]<=
            .90*near["recurrent_frozen"]["pooled_collisions_per_meter"],
        "12_nearfield_plastic_coverage_ge_sham_plus10pct":
            near["recurrent_plastic"]["unique_grid_cells"]>=
            1.10*near["recurrent_sham"]["unique_grid_cells"],
        "13_nearfield_plastic_coverage_ge_collision_only_plus10pct":
            near["recurrent_plastic"]["unique_grid_cells"]>=
            1.10*near["collision_only"]["unique_grid_cells"],
    }
    boot={
        sc:{
            ref:_paired_bootstrap(rows,sc,"unique_grid_cells",
                                  "recurrent_plastic",ref)
            for ref in (
                "recurrent_frozen","recurrent_sham",
                "collision_only","leaky_plastic","inertial_random"
            )
        } for sc in SCENES
    }
    return {
        "name":"FlyLoop R2 online readout plasticity: physical movement vs old/sham feedback",
        "protocol":"docs/FLYLOOP_R2_PROTOCOL_2026-11-24.md",
        "metadata":{
            "world_seeds":SEEDS,"scenes":SCENES,"arms":ARMS,
            "network_slots":r1.PARAMS,"hidden_neurons":r1.N,
            "trainable_output_coefficients":2*N_MOTOR_COEFF,
            "progress_window":WINDOW,
            "exploration_sd":EXPLORATION_SD,
            "learning_rate":LR,
            "eligibility_leak":TRACE_LEAK,
            "baseline_leak":BASELINE_LEAK,
            "physical_collision_cost":HIT_COST,
            "weight_update_limit":UPDATE_LIMIT,
            "physical_progress_is_researcher_objective":True,
            "no_spatial_map_inputs":True,
            "no_food_or_evolution":True,
            "r1_and_main_untouched":True,
            "unique_maps_per_scene":map_counts,
        },
        "summary":summary,
        "integrity":integrity,
        "paired_seed_bootstrap_exploratory":boot,
        "gates":gates,
        "gates_passed_count":sum(gates.values()),
        "screen_pass":all(gates.values()),
        "food_search_demonstrated":False,
        "avoidance_reflex_learned":False,
        "unprogrammed_objective_emerged":False,
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
                   default=Path("results/flyloop_r2/r2_motor_plasticity.json"))
    args=p.parse_args()
    result=run()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(
        result,sort_keys=True,separators=(",",":"))+"\n",encoding="utf-8")
    print("FLYLOOP_R2 "+json.dumps({
        k:result[k] for k in (
            "name","summary","integrity","gates","gates_passed_count",
            "screen_pass","paired_seed_bootstrap_exploratory",
            "raw_rows_sha256"
        )
    },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
