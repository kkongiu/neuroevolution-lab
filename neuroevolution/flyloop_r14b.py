"""R14B: local food acquisition curriculum; no scripted motor response."""
import argparse
import json
import math
import random
from pathlib import Path
from neuroevolution import flyloop_r1 as r1, flyloop_r12 as r12, flyloop_r13 as r13, flyloop_r14 as r14

SEEDS=range(16100,16124)
ARMS=("immediate","delayed","masked")
TRAIN_EPISODES=40
TEST_EPISODES=20
GAMMA=.9
HORIZON=8

def make_world(seed,episode,far=False):
    s=seed*1000+episode
    w=r14.World(s,"nearfield",False)
    w.rectangles=[]
    rng=random.Random(s*61331+7)
    for _ in range(10000):
        angle=rng.uniform(-math.pi,math.pi)
        radius=rng.uniform(2.,3.) if far else rng.uniform(1.,2.)
        p=(w.body.x+radius*math.cos(angle),w.body.y+radius*math.sin(angle))
        if not w.blocked(*p):
            w.initial_food=(p,)
            w.food=[p]
            return w
    raise RuntimeError("Cannot place food")

def returns(gains):
    return [min(1.,sum(GAMMA**j*g/12. for j,g in enumerate(gains[i:i+HORIZON]))) for i in range(len(gains))]

def update_head(critic,index,phi,target):
    pred=critic.predict(phi)[index]
    rate=.08*(target-pred)/max(1.,sum(x*x for x in phi))
    for j,x in enumerate(phi):
        critic.weights[index][j]+=rate*x

def episode(seed,index,arm,critic,training=True,far=False,horizon=40):
    w=make_world(seed,index,far)
    brain=r1.Circuit(seed,"recurrent")
    rng=random.Random(seed*17239+index*7919+13)
    exploration=[r12.GRID[rng.randrange(len(r12.GRID))] for _ in range(horizon)]
    records=[]
    trace=[]
    for t in range(horizon):
        if not w.food or w.body.energy<=0 or w.body.integrity<=0:
            break
        observed=w.perceive_extended()+(w.odor() if arm!="masked" else (0.,)*3)
        base,h,_=brain.forward(observed[:8])
        cmd=exploration[t] if training and t%4==0 else critic.choose_metabolic(h,observed,base,1-w.body.energy/100.,4.)
        phi=r13.features(h,observed,cmd)
        distance=damage=gain=0.
        for _ in range(r12.SUBSTEPS):
            if not w.food or w.body.energy<=0 or w.body.integrity<=0:
                break
            before=w.body.integrity
            step=w.micro_integrate(cmd)
            damage+=before-w.body.integrity
            distance+=step["moved"]
            gain+=step["food_gain"]
            trace.append(dict(step,macro=t,command=cmd))
        if training:
            update_head(critic,0,phi,min(1.,distance/(r1.V_MAX*r12.MICRO_DT*r12.SUBSTEPS)))
            update_head(critic,1,phi,min(1.,damage/r1.COLLISION_INJURY))
            if arm=="immediate":
                update_head(critic,2,phi,min(1.,gain/12.))
        records.append((phi,gain))
    if training and arm!="immediate":
        for (phi,_),target in zip(records,returns([g for _,g in records])):
            update_head(critic,2,phi,target)
    cost=sum(f["metabolic_cost"] for f in trace)
    gain=sum(f["food_gain"] for f in trace)
    assert abs(60.-cost+gain-w.body.energy)<1e-7
    return dict(index=index,far=far,encounter=not w.food,macros=len(records),food=w.initial_food,
                distance=sum(f["moved"] for f in trace),damage=100.-w.body.integrity,
                energy=w.body.energy,cost=cost,gain=gain,trace=trace)

def simulate(seed,arm):
    c=r14.Critic()
    training=[episode(seed,i,arm,c) for i in range(TRAIN_EPISODES)]
    frozen=[w[:] for w in c.weights]
    near=[episode(seed,100+i,arm,c,False,False,80) for i in range(TEST_EPISODES)]
    far=[episode(seed,200+i,arm,c,False,True,80) for i in range(TEST_EPISODES)]
    assert frozen==c.weights
    return dict(seed=seed,arm=arm,training=training,near=near,far=far,weights=frozen)

def fraction(rows,phase):
    episodes=[e for row in rows for e in row[phase]]
    return sum(e["encounter"] for e in episodes)/len(episodes)

def run():
    rows=[simulate(s,a) for s in SEEDS for a in ARMS]
    summary={a:{p:fraction([r for r in rows if r["arm"]==a],p) for p in ("training","near","far")} for a in ARMS}
    d=summary["delayed"]
    gates=dict(far_vs_immediate=d["far"]-summary["immediate"]["far"]>=.10,
               far_vs_masked=d["far"]-summary["masked"]["far"]>=.10,near=d["near"]>=.30)
    intervals={}
    by={(r["seed"],r["arm"]):r for r in rows}
    for arm in ("immediate","masked"):
        differences=[fraction([by[s,"delayed"]],"far")-fraction([by[s,arm]],"far") for s in SEEDS]
        rng=random.Random(161616)
        values=sorted(sum(rng.choice(differences) for _ in differences)/len(differences) for _ in range(2000))
        intervals[arm]=[values[50],values[1949]]
    return dict(summary=summary,gates=gates,passed=all(gates.values()),intervals=intervals,rows=rows)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",default="results/r14b.json")
    args=p.parse_args()
    result=run()
    path=Path(args.out)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,sort_keys=True,separators=(",",":")))
    print(json.dumps({k:v for k,v in result.items() if k!="rows"},indent=2))
