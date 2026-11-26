"""R13B: prospective sensory ablations, paired frozen test continuations."""
import argparse
import copy
import json
import math
import random
from pathlib import Path
from neuroevolution import flyloop_r1 as r1, flyloop_r12 as r12, flyloop_r13 as r13

SEEDS=range(14100,14124)
ARMS=("full","original","delayed","minus_touch","minus_proprio","minus_nociception","minus_rear")
REGIMES=("static","changed")
MASKS={"original":range(24),"minus_touch":range(3,19),
       "minus_proprio":range(19,22),"minus_nociception":range(22,24),
       "minus_rear":range(3)}

def observe(world, history, arm):
    extra=world.extra_senses()
    history.append(extra)
    if arm=="delayed":
        extra=history[-21] if len(history)>20 else (0.,)*24
    masked=set(MASKS.get(arm,()))
    return world.perceive()+tuple(0. if i in masked else x for i,x in enumerate(extra))

def alive(w):
    return w.body.energy>0 and w.body.integrity>0

def macro(w,brain,critic,history,arm,phase,t,training,rng,trace=None):
    observed=observe(w,history,arm)
    baseline,hidden,_=brain.forward(observed[:8])
    command=r12.GRID[rng.randrange(len(r12.GRID))] if training and t%4==0 else critic.choose(hidden,observed,baseline)
    phi=r13.features(hidden,observed,command)
    distance,injury=0.,0.
    for _ in range(r12.SUBSTEPS):
        if not alive(w):
            break
        before=w.body.integrity
        step=w.micro_integrate(command)
        damage=before-w.body.integrity
        phase.add(w,step,damage)
        distance+=step["moved"]
        injury+=damage
        if trace is not None:
            trace.append(dict(macro=t,position=step["position"],heading=step["heading"],
                speed=step["speed"],actual_speed=w.actual_speed,turn=step["turn"],
                command=command,collision=step["collision"],moved=step["moved"],injury=damage))
    if training:
        critic.learn(phi,(r1.clamp(distance/(r1.V_MAX*r12.MICRO_DT*r12.SUBSTEPS),0.,1.),
                          r1.clamp(injury/r1.COLLISION_INJURY,0.,1.)))
        phase.updates+=1

def metrics(trace, start, heading, phase):
    m=phase.report()
    prev=heading
    turning=0.
    visited=set()
    revisits=0
    reverse=0
    for f in trace:
        turning+=abs((f["heading"]-prev+math.pi)%(2*math.pi)-math.pi)
        prev=f["heading"]
        cell=tuple(int(x) for x in f["position"])
        revisits+=cell in visited
        visited.add(cell)
        reverse+=f["actual_speed"] < -1e-9
    net=math.dist(start,trace[-1]["position"]) if trace else 0.
    m.update(turning=turning,turning_per_distance=turning/max(m["distance"],1e-12),
             net_displacement=net,net_path_ratio=net/max(m["distance"],1e-12),
             revisit_fraction=revisits/max(len(trace),1),reverse_fraction=reverse/max(len(trace),1))
    return m

def simulate(seed,geometry,arm,train=r13.TRAIN,test=r13.TEST):
    if arm not in ARMS:
        raise ValueError(arm)
    w=r13.World(seed,geometry)
    brain=r1.Circuit(seed,"recurrent")
    critic=r13.Critic()
    history=[]
    rng=random.Random(seed*17239+13)
    phase=r13.Phase()
    for t in range(train):
        if not alive(w):
            break
        macro(w,brain,critic,history,arm,phase,t,True,rng)
    rows=[]
    for regime in REGIMES:
        body,core,model,hist=copy.deepcopy((w,brain,critic,history))
        if regime=="changed":
            body.rectangles=list(body.initial_rectangles[::2])
        start=(body.body.x,body.body.y)
        heading=body.body.heading
        trace=[]
        evaluation=r13.Phase()
        for t in range(test):
            if not alive(body):
                break
            macro(body,core,model,hist,arm,evaluation,train+t,False,rng,trace)
        assert model.weights==critic.weights
        rows.append(dict(seed=seed,geometry=geometry,arm=arm,regime=regime,
            training=phase.report(),test=metrics(trace,start,heading,evaluation),
            survived=alive(body) and len(trace)==test*r12.SUBSTEPS,
            split_pose=(*start,heading),split_rectangles=w.rectangles,
            test_rectangles=body.rectangles,weights=critic.weights,trace=trace))
    return rows

def aggregate(rows):
    result=r13.aggregate(rows)
    for k in ("turning_per_distance","net_path_ratio","revisit_fraction","reverse_fraction"):
        result[k]=sum(r["test"][k] for r in rows)/max(len(rows),1)
    return result

def intervals(rows,arm):
    by={(r["seed"],r["arm"]):r for r in rows}
    pairs=[(by[s,arm],by[s,"full"]) for s in SEEDS]
    rng=random.Random(141414)
    values=[[],[]]
    for _ in range(2000):
        sampled=[pairs[rng.randrange(len(pairs))] for _ in pairs]
        a,b=(aggregate([p[i] for p in sampled]) for i in (0,1))
        values[0].append(a["collision_density"]-b["collision_density"])
        values[1].append((a["cells"]-b["cells"])/len(pairs))
    return {k:[sorted(v)[50],sorted(v)[1949]] for k,v in zip(("density_difference","mean_cells_difference"),values)}

def run():
    rows=[row for g in r13.GEOMETRIES for s in SEEDS for a in ARMS for row in simulate(s,g,a)]
    summary={}
    contrasts={}
    for g in r13.GEOMETRIES:
        summary[g]={}
        contrasts[g]={}
        for regime in REGIMES:
            subset=[r for r in rows if r["geometry"]==g and r["regime"]==regime]
            summary[g][regime]={a:aggregate([r for r in subset if r["arm"]==a]) for a in ARMS}
            contrasts[g][regime]={a:intervals(subset,a) for a in ARMS if a!="full"}
    p=summary["nearfield"]["changed"]
    f,o,d=(p[a] for a in ("full","original","delayed"))
    gates=dict(collision_vs_original=o["collision_density"]>0 and f["collision_density"]<=.8*o["collision_density"],
               distance=f["distance"]>=.9*o["distance"],cells=f["cells"]>=.9*o["cells"],
               survival=f["survived"]>=o["survived"],
               collision_vs_delayed=d["collision_density"]>0 and f["collision_density"]<=.85*d["collision_density"])
    return dict(summary=summary,contrasts=contrasts,gates=gates,passed=all(gates.values()),rows=rows)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",default="results/r13b.json")
    args=p.parse_args()
    result=run()
    path=Path(args.out)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,sort_keys=True,separators=(",",":")))
    print(json.dumps({k:v for k,v in result.items() if k!="rows"},indent=2))
