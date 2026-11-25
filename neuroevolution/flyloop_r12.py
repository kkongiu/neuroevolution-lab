"""R12: finite external actuator interventions, not autonomous learning."""
import argparse
import copy
import itertools
import json
import math
import random
from pathlib import Path
from neuroevolution import flyloop_r1 as r1, flyloop_r2 as r2, flyloop_r9 as r9

SEEDS = tuple(range(12400,12424))
GEOMETRIES = ("obstacles","nearfield")
PLANTS = ("original","signed","matched_forward")
COMMANDS = tuple(itertools.product((-1.,0.,1.), repeat=2))
SEQUENCES = tuple(itertools.product(COMMANDS, repeat=2))

class World(r9.World):
    def __init__(self,seed,geometry):
        self.seed=seed
        self.geometry=self.scenario=geometry
        self.regime="static"
        self.initial_rectangles=(r1.procedural_rectangles(seed) if geometry=="obstacles" else r2.nearfield_rectangles(seed))
        self.rectangles=list(self.initial_rectangles)
        self.body=r1.Body(heading=random.Random(seed*11113+51).uniform(-math.pi,math.pi))
        self.frame=0
        self.visited={(int(self.body.x),int(self.body.y))}
        self.collisions=0
        self.distance=0.
        self.trace=[]
        self.changed=False

def target_speed(throttle, plant):
    if plant=="original":
        return r1.V_MAX*r1.sigmoid(1.6*throttle)
    if plant=="signed":
        return r1.V_MAX*throttle
    if plant=="matched_forward":
        return math.copysign(r1.V_MAX*r1.sigmoid(1.6*abs(throttle)),throttle) if throttle else 0.
    raise ValueError(plant)

def integrate(world, command, plant):
    if plant=="original":
        return world.micro_integrate(command)
    # Exact R9 integration except target speed mapping.
    b=world.body
    if b.energy<=0 or b.integrity<=0:
        raise AssertionError("Dead body")
    throttle,steer=command
    if not all(-1<=v<=1 for v in command):
        raise ValueError(command)
    before=(b.x,b.y)
    b.speed=(1-r9.MICRO_ALPHA)*b.speed+r9.MICRO_ALPHA*target_speed(throttle,plant)
    b.turn=(1-r9.MICRO_ALPHA)*b.turn+r9.MICRO_ALPHA*r1.W_MAX*math.tanh(1.6*steer)
    b.heading=(b.heading+b.turn*r9.MICRO_DT+math.pi)%(2*math.pi)-math.pi
    nx=b.x+b.speed*math.cos(b.heading)*r9.MICRO_DT
    ny=b.y+b.speed*math.sin(b.heading)*r9.MICRO_DT
    hit=int(world.blocked(nx,ny))
    if hit:
        world.collisions+=1
        b.integrity=max(0.,b.integrity-r1.COLLISION_INJURY/r9.SUBSTEPS)
    else:
        b.x,b.y=nx,ny
    b.energy=max(0.,b.energy-r1.ENERGY_PER_TICK/r9.SUBSTEPS)
    b.prev_collision=hit
    moved=math.hypot(b.x-before[0],b.y-before[1])
    world.distance+=moved
    world.visited.add((int(b.x),int(b.y)))
    world.frame+=1
    return {"collision":hit,"moved":round(moved,9)}

def alive(w):
    return w.body.energy>0 and w.body.integrity>0

def source(seed,geo):
    w=World(seed,geo)
    brain=r2.MotorLearner(seed,"recurrent_frozen")
    snapshots=[]
    for macro in range(160):
        if not alive(w): break
        _,_,_,_,command=brain.command(w.perceive())
        for sub in range(4):
            if not alive(w): break
            previous=w.body.prev_collision
            p=w.micro_integrate(command)
            if p["collision"] and not previous and len(snapshots)<3:
                snapshots.append(copy.deepcopy(w))
    return snapshots

def recovery_step(history):
    for i in range(3,len(history)):
        last=history[i-3:i+1]
        if not any(p["collision"] for p in last) and sum(p["moved"] for p in last)>=.035:
            return i+1
    return None

def trial(snapshot,plant,sequence):
    w=copy.deepcopy(snapshot)
    history=[]
    for command in sequence:
        for _ in range(16):
            if not alive(w): break
            history.append(integrate(w,command,plant))
    return {"sequence":sequence,"first_recovery_step":recovery_step(history),
            "distance":sum(p["moved"] for p in history),"contacts":sum(p["collision"] for p in history),
            "steps":len(history),"death":not alive(w)}

def snapshot_record(w):
    b=w.body
    return {"frame":w.frame,"rectangles":w.rectangles,
            "body":{k:getattr(b,k) for k in ("x","y","heading","speed","turn","energy","integrity","prev_collision")}}

def summarize(rows, plant):
    pairs=[]
    byseed={}
    for row in rows:
        for state in row["states"]:
            a=any(t["first_recovery_step"] is not None for t in state["trials"]["original"])
            b=any(t["first_recovery_step"] is not None for t in state["trials"][plant])
            pairs.append((a,b))
            byseed.setdefault(row["seed"],[]).append(int(b)-int(a))
    n=len(pairs)
    ds=[sum(v)/len(v) for v in byseed.values()]
    rng=random.Random(124001)
    draws=sorted(sum(rng.choice(ds) for _ in ds)/len(ds) for _ in range(5000)) if ds else []
    return {"states":n,"seeds_with_contacts":len(ds),"both":sum(a and b for a,b in pairs),
            "original_only":sum(a and not b for a,b in pairs),"alternative_only":sum(b and not a for a,b in pairs),
            "neither":sum(not a and not b for a,b in pairs),
            "original_reachable_fraction":sum(a for a,b in pairs)/n if n else None,
            "alternative_reachable_fraction":sum(b for a,b in pairs)/n if n else None,
            "paired_state_delta":sum(int(b)-int(a) for a,b in pairs)/n if n else None,
            "seed_cluster_mean_delta":sum(ds)/len(ds) if ds else None,
            "seed_cluster_bootstrap95":[draws[125],draws[4874]] if draws else None}

def run():
    rows=[]
    for seed in SEEDS:
        for geo in GEOMETRIES:
            states=[]
            for w in source(seed,geo):
                states.append({"snapshot":snapshot_record(w),"trials":{
                    plant:[trial(w,plant,seq) for seq in SEQUENCES] for plant in PLANTS}})
            rows.append({"seed":seed,"geometry":geo,"states":states})
    contrasts={plant:summarize(rows,plant) for plant in PLANTS[1:]}
    primary=contrasts["signed"]
    gate=(primary["seeds_with_contacts"]>=12 and primary["paired_state_delta"]>=.20 and primary["seed_cluster_bootstrap95"][0]>0)
    sequence_counts={}
    for plant in PLANTS:
        ts=[t for row in rows for state in row["states"] for t in state["trials"][plant]]
        sequence_counts[plant]={"trials":len(ts),"successes":sum(t["first_recovery_step"] is not None for t in ts),
                                "success_fraction":sum(t["first_recovery_step"] is not None for t in ts)/len(ts) if ts else None}
    return {"rows":rows,"contrasts":contrasts,"by_geometry":{
        geo:{p:summarize([r for r in rows if r["geometry"]==geo],p) for p in PLANTS[1:]} for geo in GEOMETRIES},
        "sequence_counts":sequence_counts,"primary_gate_pass":gate,"learned_escape_demonstrated":False}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",default="results/flyloop_r12/results.json")
    args=parser.parse_args()
    result=run()
    path=Path(args.out)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,sort_keys=True,separators=(",",":")))
    print(json.dumps({k:v for k,v in result.items() if k!="rows"},indent=2))
if __name__=="__main__":
    main()
