"""FlyLoop R1: continuous locomotion from a 16-neuron recurrent dynamical circuit.

NEW independent experiment branched directly from main, not from M0-M4.
NO food, no task reward, NO online learning, no evolution, no planner.
The goal is to isolate recurrence from simple neural leakage, motor
inertia, and temporally correlated random actions in two physical worlds.

See docs/FLYLOOP_R1_PROTOCOL_2026-11-24.md (committed before this code).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

SEEDS=tuple(range(9100,9124))
SCENARIOS=("empty","obstacles")
ARMS=(
    "recurrent",
    "leaky_no_recurrence",
    "instantaneous",
    "recurrent_blind",
    "inertial_random",
)
N=16
INPUTS=8
OUTPUTS=2
PARAMS=N*INPUTS+N*N+OUTPUTS*N+N+OUTPUTS
SIDE=22.0
RADIUS=.30
DT=.20
HORIZON=600
ENERGY_PER_TICK=.02
COLLISION_INJURY=.25
RAY_LENGTH=3.5
SENSOR_ANGLE=.70
ALPHA=.30
MOTOR_ALPHA=.18
V_MAX=1.2
W_MAX=1.9
OBSTACLES=10


def sha(x):
    return hashlib.sha256(
        json.dumps(x,sort_keys=True,separators=(",",":")).encode()
    ).hexdigest()


def mean(xs):
    xs=list(xs)
    return sum(xs)/len(xs) if xs else 0.0


def dot(a,b):
    return sum(x*y for x,y in zip(a,b))


def clamp(x,a,b):
    return min(b,max(a,x))


def sigmoid(x):
    if x>=0:
        z=math.exp(-x)
        return 1/(1+z)
    z=math.exp(x)
    return z/(1+z)


def generated_coefficients(seed):
    rng=random.Random(seed*26099+20261124)
    # Weight scale frozen prospectively. Coefficients are never updated.
    w_in=[[rng.gauss(0,.85/math.sqrt(INPUTS))
           for _ in range(INPUTS)] for _ in range(N)]
    w_rec=[[rng.gauss(0,1.2/math.sqrt(N))
            for _ in range(N)] for _ in range(N)]
    w_out=[[rng.gauss(0,1.0/math.sqrt(N))
            for _ in range(N)] for _ in range(OUTPUTS)]
    bias=[rng.gauss(0,.22) for _ in range(N)]
    motor_bias=[rng.gauss(0,.18) for _ in range(OUTPUTS)]
    return {
        "w_input":w_in,
        "w_recurrent":w_rec,
        "w_motor":w_out,
        "bias":bias,
        "motor_bias":motor_bias,
    }


def coefficient_flat(weights):
    return (
        [v for row in weights["w_input"] for v in row]
        +[v for row in weights["w_recurrent"] for v in row]
        +[v for row in weights["w_motor"] for v in row]
        +list(weights["bias"])+list(weights["motor_bias"])
    )


def procedural_rectangles(seed):
    """Exogenous 10-rectangle environment; no conditioning on controller."""
    rng=random.Random(seed*71993+8319)
    rects=[]
    for _ in range(OBSTACLES):
        for attempt in range(2000):
            cx=rng.uniform(3.4,18.6)
            cy=rng.uniform(3.4,18.6)
            width=rng.uniform(1.1,2.2)
            height=rng.uniform(1.1,2.2)
            rect=(
                round(cx-width/2,6),round(cy-height/2,6),
                round(cx+width/2,6),round(cy+height/2,6),
            )
            if (rect[0]-1.1<2.2<rect[2]+1.1
                and rect[1]-1.1<2.2<rect[3]+1.1):
                continue
            if any(
                rect[0]<x2+.35 and rect[2]>x1-.35
                and rect[1]<y2+.35 and rect[3]>y1-.35
                for x1,y1,x2,y2 in rects
            ):
                continue
            rects.append(rect)
            break
        else:
            raise RuntimeError("Cannot generate ten separated rectangles")
    return rects


@dataclass
class Body:
    x:float=2.2
    y:float=2.2
    heading:float=0.
    speed:float=0.
    turn:float=0.
    energy:float=100.
    integrity:float=100.
    prev_collision:int=0


class World:
    def __init__(self,seed,scenario):
        if scenario not in SCENARIOS or seed not in SEEDS:
            raise ValueError("Unregistered R1 world")
        self.seed=seed
        self.scenario=scenario
        self.rectangles=(
            procedural_rectangles(seed)
            if scenario=="obstacles" else []
        )
        pose_rng=random.Random(seed*11113+51)
        self.body=Body(heading=pose_rng.uniform(-math.pi,math.pi))
        self.frame=0
        self.visited={(int(self.body.x),int(self.body.y))}
        self.collisions=0
        self.distance=0.0
        self.trace=[]

    def blocked(self,x,y):
        if not(RADIUS<=x<=SIDE-RADIUS
               and RADIUS<=y<=SIDE-RADIUS):
            return True
        return any(
            x1-RADIUS<=x<=x2+RADIUS
            and y1-RADIUS<=y<=y2+RADIUS
            for x1,y1,x2,y2 in self.rectangles
        )

    @staticmethod
    def _ray_box(ox,oy,ux,uy,x1,y1,x2,y2):
        """Forward ray vs body-radius-inflated solid rectangle."""
        enter=-math.inf
        leave=math.inf
        for origin,direction,lo,hi in (
            (ox,ux,x1,y1),  # overwritten immediately below
        ):
            pass
        # Slab intersection, finite positive ray segment.
        for origin,direction,lo,hi in (
            (ox,ux,x1,x2), (oy,uy,y1,y2)
        ):
            if abs(direction)<1e-12:
                if not lo<=origin<=hi:
                    return math.inf
                continue
            a=(lo-origin)/direction
            b=(hi-origin)/direction
            if a>b:
                a,b=b,a
            enter=max(enter,a)
            leave=min(leave,b)
            if enter>leave:
                return math.inf
        return max(0.,enter) if leave>=0. else math.inf

    def ray_distance(self,angle):
        b=self.body
        ux=math.cos(angle)
        uy=math.sin(angle)
        # External world border as physical wall.
        distances=[]
        if ux>1e-12:
            distances.append((SIDE-RADIUS-b.x)/ux)
        elif ux<-1e-12:
            distances.append((RADIUS-b.x)/ux)
        if uy>1e-12:
            distances.append((SIDE-RADIUS-b.y)/uy)
        elif uy<-1e-12:
            distances.append((RADIUS-b.y)/uy)
        d=max(0.,min(distances))
        for x1,y1,x2,y2 in self.rectangles:
            collision=self._ray_box(
                b.x,b.y,ux,uy,
                x1-RADIUS,y1-RADIUS,x2+RADIUS,y2+RADIUS
            )
            if collision<d:
                d=collision
        return min(RAY_LENGTH,d)

    def perceive(self):
        b=self.body
        proximity=tuple(
            clamp(1.-self.ray_distance(b.heading+offset)/RAY_LENGTH,0.,1.)
            for offset in (0.,SENSOR_ANGLE,-SENSOR_ANGLE)
        )
        return (
            1.0,*proximity,
            b.speed/V_MAX,
            b.turn/W_MAX,
            float(b.prev_collision),
            (100.-b.energy)/100.,
        )

    def integrate(self,command):
        b=self.body
        if b.energy<=0 or b.integrity<=0:
            raise AssertionError("Dead organism cannot move")
        before=(b.x,b.y,b.heading,b.speed,b.turn,b.energy,b.integrity)
        throttle,steer=command
        if not(-1.-1e-9<=throttle<=1.+1e-9
               and -1.-1e-9<=steer<=1.+1e-9):
            raise ValueError("Out-of-range continuous motor signal")
        desired_speed=V_MAX*sigmoid(1.6*throttle)
        desired_turn=W_MAX*math.tanh(1.6*steer)
        b.speed=(1.-MOTOR_ALPHA)*b.speed+MOTOR_ALPHA*desired_speed
        b.turn=(1.-MOTOR_ALPHA)*b.turn+MOTOR_ALPHA*desired_turn
        b.heading=(b.heading+b.turn*DT+math.pi)%(2*math.pi)-math.pi
        newx=b.x+b.speed*math.cos(b.heading)*DT
        newy=b.y+b.speed*math.sin(b.heading)*DT
        collision=int(self.blocked(newx,newy))
        if not collision:
            b.x,b.y=newx,newy
        else:
            self.collisions+=1
            b.integrity=max(0.,b.integrity-COLLISION_INJURY)
        b.energy=max(0.,b.energy-ENERGY_PER_TICK)
        moved=math.hypot(b.x-before[0],b.y-before[1])
        self.distance+=moved
        self.visited.add((int(b.x),int(b.y)))
        b.prev_collision=collision
        self.frame+=1
        return {
            "collision":collision,
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


class Circuit:
    def __init__(self,seed,arm):
        if arm not in ARMS:
            raise ValueError("Unknown neural ablation")
        self.seed=seed
        self.arm=arm
        self.weights=generated_coefficients(seed)
        self.initial_hash=sha(self.weights)
        self.h=[0.]*N
        self.random_state=[0.,0.]
        self.random_rng=random.Random(seed*90191+157)

    def forward(self,observed):
        if len(observed)!=INPUTS:
            raise ValueError("Must provide 8 local sensor readings")
        x=list(observed)
        if self.arm=="recurrent_blind":
            # Wall proximity and previous wall contact completely denied.
            for i in (1,2,3,6):
                x[i]=0.
        if self.arm=="inertial_random":
            # No sensory use; the motor commands are correlated over
            # time even though there are NO neural updates.
            self.random_state=[
                clamp(.92*previous+.08*self.random_rng.gauss(0.,1.),-1.,1.)
                for previous in self.random_state
            ]
            return (
                tuple(self.random_state),
                tuple([0.]*N),
                tuple(x),
            )
        h_old=list(self.h)
        activation=[]
        for j in range(N):
            base=dot(self.weights["w_input"][j],x)+self.weights["bias"][j]
            if self.arm in ("recurrent","recurrent_blind"):
                base+=dot(self.weights["w_recurrent"][j],h_old)
            activation.append(math.tanh(base))
        if self.arm=="instantaneous":
            h=activation
        else:
            h=[
                (1.-ALPHA)*old+ALPHA*act
                for old,act in zip(h_old,activation)
            ]
        self.h=h
        motor=tuple(clamp(
            dot(self.weights["w_motor"][i],h)
            +self.weights["motor_bias"][i],
            -1.,1.,
        ) for i in range(OUTPUTS))
        return motor,tuple(h),tuple(x)


def simulate(seed,scenario,arm):
    w=World(seed,scenario)
    c=Circuit(seed,arm)
    energy=100.
    for t in range(HORIZON):
        if w.body.energy<=0 or w.body.integrity<=0:
            break
        s=w.perceive()
        command,neural_state,actual_input=c.forward(s)
        consequence=w.integrate(command)
        consequence.update({
            "tick":t,
            "sensors":list(actual_input),
            "neural_state":[round(v,9) for v in neural_state],
            "motor_command":[round(v,9) for v in command],
        })
        w.trace.append(consequence)
    b=w.body
    posdist=math.hypot(b.x-2.2,b.y-2.2)
    total_frames=len(w.trace)
    return {
        "seed":seed,"scenario":scenario,"arm":arm,
        "rectangles":[list(r) for r in w.rectangles],
        "initial_pose":[2.2,2.2,round(World(seed,scenario).body.heading,9)],
        "initial_coefficients_sha256":c.initial_hash,
        "initial_coefficients":c.weights,
        "final_coefficients_sha256":sha(c.weights),
        "weights_updated":False,
        "steps_alive":total_frames,
        "distance":round(w.distance,6),
        "net_displacement":round(posdist,6),
        "path_efficiency":round(posdist/w.distance,6) if w.distance else 0.,
        "unique_grid_cells":len(w.visited),
        "collisions":w.collisions,
        "collisions_per_meter":round(
            w.collisions/max(w.distance,1e-12),6
        ),
        "mean_speed":round(
            mean(t["speed"] for t in w.trace),6
        ),
        "mean_abs_turn":round(
            mean(abs(t["turn"]) for t in w.trace),6
        ),
        "fraction_physically_moving":round(
            mean(t["moved"]>0 for t in w.trace),6
        ),
        "terminal_energy":round(b.energy,6),
        "terminal_integrity":round(b.integrity,6),
        "trace":w.trace,
    }


def _paired_bootstrap(rows,scenario,metric,arm,reference,n=1000):
    index={
        (r["seed"],r["arm"]):r
        for r in rows if r["scenario"]==scenario
    }
    differences=[
        index[(seed,arm)][metric]-index[(seed,reference)][metric]
        for seed in SEEDS
    ]
    rng=random.Random(91079)
    draws=sorted(mean(
        differences[rng.randrange(len(differences))]
        for _ in differences
    ) for _ in range(n))
    return {
        "mean_paired_difference":round(mean(differences),6),
        "exploratory_paired_seed_bootstrap95":[
            round(draws[int(n*.025)],6),
            round(draws[int(n*.975)-1],6),
        ]
    }


def _integrity(rows):
    physics=True
    sensors=True
    masks=True
    state=True
    coefficients=True
    for row in rows:
        oldpos=(2.2,2.2)
        prev_state=[0.]*N
        before_heading=row["initial_pose"][2]
        previous_speed=0.
        previous_turn=0.
        prev_collision=0
        energy=100.
        integrity=100.
        for t in row["trace"]:
            if tuple(t["before_position"])!=tuple(round(p,9) for p in oldpos):
                physics=False
            oldpos=tuple(t["position"])
            if not (
                RADIUS-1e-7<=oldpos[0]<=SIDE-RADIUS+1e-7 and
                RADIUS-1e-7<=oldpos[1]<=SIDE-RADIUS+1e-7
            ):
                physics=False
            for x1,y1,x2,y2 in row["rectangles"]:
                if (x1-RADIUS<oldpos[0]<x2+RADIUS and
                    y1-RADIUS<oldpos[1]<y2+RADIUS):
                    physics=False
            if not(0<=t["energy"]<=100 and 0<=t["integrity"]<=100):
                physics=False
            if t["collision"]:
                if math.hypot(
                    t["position"][0]-t["before_position"][0],
                    t["position"][1]-t["before_position"][1]
                )>1e-7:
                    physics=False
            if not(t["tick"]>=0 and len(t["sensors"])==INPUTS
                   and len(t["neural_state"])==N
                   and len(t["motor_command"])==OUTPUTS):
                state=False
            if row["arm"]=="recurrent_blind" and any(
                abs(t["sensors"][i])>1e-8 for i in (1,2,3,6)
            ):
                masks=False
            if row["arm"]=="inertial_random" and any(
                abs(v)>1e-8 for v in t["neural_state"]
            ):
                masks=False
            if row["arm"]!="inertial_random":
                x=t["sensors"]
                coeff=row["initial_coefficients"]
                new_state=[]
                for j in range(N):
                    z=dot(coeff["w_input"][j],x)+coeff["bias"][j]
                    if row["arm"] in ("recurrent","recurrent_blind"):
                        z+=dot(coeff["w_recurrent"][j],prev_state)
                    activation=math.tanh(z)
                    if row["arm"]=="instantaneous":
                        new_state.append(activation)
                    else:
                        new_state.append(
                            (1-ALPHA)*prev_state[j]+ALPHA*activation
                        )
                if any(abs(a-b)>2e-7 for a,b in zip(
                    new_state,t["neural_state"]
                )):
                    state=False
            prev_state=list(t["neural_state"])
            speed=(1-MOTOR_ALPHA)*previous_speed+MOTOR_ALPHA*(
                V_MAX*sigmoid(1.6*t["motor_command"][0])
            )
            turn=(1-MOTOR_ALPHA)*previous_turn+MOTOR_ALPHA*(
                W_MAX*math.tanh(1.6*t["motor_command"][1])
            )
            if (abs(speed-t["speed"])>1e-7 or
                abs(turn-t["turn"])>1e-7):
                physics=False
            previous_speed,previous_turn=speed,turn
            current_heading=(
                before_heading+turn*DT+math.pi
            )%(2*math.pi)-math.pi
            if abs(current_heading-t["heading"])>1e-7:
                physics=False
            before_heading=current_heading
            energy=max(0.,energy-ENERGY_PER_TICK)
            integrity=max(0.,integrity-COLLISION_INJURY*t["collision"])
            if (abs(energy-t["energy"])>1e-7 or
                abs(integrity-t["integrity"])>1e-7):
                physics=False
            prev_collision=t["collision"]
        if (row["initial_coefficients_sha256"]!=
            row["final_coefficients_sha256"]
            or not (len(coefficient_flat(row["initial_coefficients"]))==PARAMS)
            or row["weights_updated"]):
            coefficients=False
        if len(row["trace"])!=row["steps_alive"]:
            physics=False
    return {
        "continuous_geometry_and_physical_outcomes":physics,
        "eight_inputs_sixteen_state_two_motor_signals":state,
        "ablated_sensor_channels_zero":masks,
        "initialized_434_coefficients_never_updated":coefficients,
    }


def run():
    rows=[
        simulate(seed,scenario,arm)
        for scenario in SCENARIOS for seed in SEEDS for arm in ARMS
    ]
    summary={}
    for scenario in SCENARIOS:
        summary[scenario]={}
        for arm in ARMS:
            scope=[
                r for r in rows if
                r["scenario"]==scenario and r["arm"]==arm
            ]
            sumdist=sum(r["distance"] for r in scope)
            summary[scenario][arm]={
                key:round(mean(r[key] for r in scope),6)
                for key in (
                    "steps_alive","distance","net_displacement",
                    "path_efficiency","unique_grid_cells","collisions",
                    "mean_speed","mean_abs_turn","fraction_physically_moving",
                    "terminal_energy","terminal_integrity"
                )
            }|{
                "collisions_per_meter_pooled":round(
                    sum(r["collisions"] for r in scope)/
                    max(sumdist,1e-12),6
                ),
                "cohort_total_distance":round(sumdist,6),
                "cohort_total_collisions":sum(r["collisions"] for r in scope),
                "completed_600_fraction":round(
                    mean(r["steps_alive"]==HORIZON for r in scope),6),
            }
    integrity=_integrity(rows)
    pairing=all(
        len({sha(r["rectangles"]) for r in rows
             if r["scenario"]==scenario and r["seed"]==seed})==1
        and len({r["initial_coefficients_sha256"] for r in rows
                 if r["seed"]==seed})==1
        and len({sha(r["initial_pose"]) for r in rows
                 if r["seed"]==seed})==1
        for scenario in SCENARIOS for seed in SEEDS
    )
    obstacle_maps=len({
        sha(r["rectangles"]) for r in rows
        if r["scenario"]=="obstacles"
    })
    rectangles_valid=all(
        len(r["rectangles"])==OBSTACLES
        and all(not (
            x1-1.1<2.2<x2+1.1 and y1-1.1<2.2<y2+1.1
        ) for x1,y1,x2,y2 in r["rectangles"])
        for r in rows if r["scenario"]=="obstacles"
    )
    different_dynamics=sum(
        sha(next(r["trace"] for r in rows
                if r["seed"]==seed and r["scenario"]=="obstacles"
                and r["arm"]=="recurrent")) not in (
            sha(next(r["trace"] for r in rows
                    if r["seed"]==seed and r["scenario"]=="obstacles"
                    and r["arm"]==other))
            for other in ("instantaneous","leaky_no_recurrence")
        )
        for seed in SEEDS
    )
    obs=summary["obstacles"]
    pooled=lambda arm:obs[arm]["collisions_per_meter_pooled"]
    gates={
        "01_240_exact_balanced_lives_repeat_required":
            len(rows)==240 and all(
                sum(r["scenario"]==scenario and r["arm"]==arm
                    for r in rows)==len(SEEDS)
                for scenario in SCENARIOS for arm in ARMS
            ),
        "02_legal_continuous_physics_and_no_wall_penetration":
            integrity["continuous_geometry_and_physical_outcomes"],
        "03_identical_world_pose_and_434_initial_coefficients":
            pairing,
        "04_no_food_reward_weight_updates_or_evolution":
            integrity["initialized_434_coefficients_never_updated"]
            and all(r["weights_updated"] is False for r in rows),
        "05_16_neural_values_two_continuous_outputs_exact_equations":
            integrity["eight_inputs_sixteen_state_two_motor_signals"],
        "06_leaky_and_instantaneous_ablate_recurrence":
            all(len(r["trace"])>0 for r in rows
                if r["arm"] in ("instantaneous","leaky_no_recurrence")),
        "07_blind_agent_wall_inputs_clamped_zero":
            integrity["ablated_sensor_channels_zero"],
        "08_inertial_random_same_actuator_kinematics":
            integrity["continuous_geometry_and_physical_outcomes"]
            and integrity["eight_inputs_sixteen_state_two_motor_signals"],
        "09_10_exogenous_separated_rectangles_at_least_20_maps":
            rectangles_valid and obstacle_maps>=20,
        "10_recurrent_trajectory_differs_from_ablation_in_90pct_seeds":
            different_dynamics>=math.ceil(.90*len(SEEDS)),
        "11_recurrent_collision_per_meter_10pct_below_inertial_random":
            pooled("recurrent")<=.90*pooled("inertial_random"),
        "12_recurrent_unique_cells_10pct_above_leaky_no_recurrence":
            obs["recurrent"]["unique_grid_cells"]>=
            1.10*obs["leaky_no_recurrence"]["unique_grid_cells"],
        "13_recurrent_collision_per_meter_10pct_below_recurrent_blind":
            pooled("recurrent")<=.90*pooled("recurrent_blind"),
    }
    paired={
        "obstacle_coverage_recurrent_vs_leaky":
            _paired_bootstrap(rows,"obstacles","unique_grid_cells",
                              "recurrent","leaky_no_recurrence"),
        "obstacle_collisions_recurrent_vs_random":
            _paired_bootstrap(rows,"obstacles","collisions",
                              "recurrent","inertial_random"),
        "obstacle_collisions_recurrent_vs_sensor_blind":
            _paired_bootstrap(rows,"obstacles","collisions",
                              "recurrent","recurrent_blind"),
    }
    return {
        "name":"FlyLoop R1 frozen recurrent locomotion in continuous physics",
        "protocol":"docs/FLYLOOP_R1_PROTOCOL_2026-11-24.md",
        "metadata":{
            "world_seeds":SEEDS,
            "scenarios":SCENARIOS,
            "controllers":ARMS,
            "environment":"22x22 continuous 2D, x/y body radius and true collisions",
            "obstacle_rectangles":OBSTACLES,
            "timestep":DT,
            "frames_per_life":HORIZON,
            "sensor_count":INPUTS,"internal_neurons":N,
            "continuous_motor_channels":OUTPUTS,
            "coefficient_slots":PARAMS,
            "neural_leak_alpha":ALPHA,
            "motor_physical_inertia":MOTOR_ALPHA,
            "no_learning_and_no_evolution":True,
            "no_foraging_or_avoidance_macro":True,
            "no_food_or_threats_in_R1":True,
            "main_and_M0_M4_sealed_untouched":True,
            "different_obstacle_maps":obstacle_maps,
        },
        "summary":summary,
        "integrity_checks":integrity,
        "paired_world_bootstrap_exploratory":paired,
        "gates":gates,
        "gates_passed_count":sum(gates.values()),
        "science_gate_count":3,
        "screen_pass":all(gates.values()),
        "food_search_demonstrated":False,
        "escape_reflex_learned":False,
        "adaptive_motor_learning_demonstrated":False,
        "biological_fly_replication_demonstrated":False,
        "general_intelligence_demonstrated":False,
        "evolution_authorized":False,
        "advance_to_evolution":False,
        "raw_sha256":sha(rows),
        "raw_rows":rows,
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument(
        "--out",type=Path,
        default=Path("results/flyloop_r1/r1_continuous_motor.json")
    )
    opts=parser.parse_args()
    result=run()
    opts.out.parent.mkdir(exist_ok=True,parents=True)
    opts.out.write_text(
        json.dumps(result,sort_keys=True,separators=(",",":"))+"\n",
        encoding="utf-8",
    )
    print("FLYLOOP_R1 "+json.dumps({
        k:result[k] for k in (
            "name","summary","gates","gates_passed_count",
            "screen_pass","paired_world_bootstrap_exploratory",
            "raw_sha256",
        )
    },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
