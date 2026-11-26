"""R15: engineered local taxis prior, factorial learning diagnostic."""
import argparse
import json
import math
import random
from pathlib import Path
from neuroevolution import flyloop_r1 as r1, flyloop_r12 as r12, flyloop_r13 as r13, flyloop_r14 as r14, flyloop_r14b as b

SEEDS=range(17100,17124)
ARMS=("neither","innate","learning","both")

def innate_command(base,odor,hunger):
    f,l,r=odor
    blend=r1.clamp(2*hunger*max(odor),0.,1.)
    target=(math.tanh(4*f),math.tanh(4*(l-r)))
    return tuple((1-blend)*x+blend*y for x,y in zip(base,target)),blend

def choose(critic,h,observed,base,hunger,innate,learning):
    biased,blend=innate_command(base,observed[-3:],hunger) if innate else (base,0.)
    if not learning:
        return biased
    def score(cmd):
        p,i,g=critic.predict(r13.features(h,observed,cmd))
        prior=.6*blend*(1-sum((x-y)**2 for x,y in zip(cmd,biased))/8.)
        return p-2*i+4*hunger*g+prior
    return max((tuple(biased),)+r12.GRID,key=score)

def episode(seed,index,arm,critic,training=True,far=False,masked=False,horizon=40):
    w=b.make_world(seed,index,far)
    brain=r1.Circuit(seed,"recurrent")
    rng=random.Random(seed*17239+index*7919+13)
    exploration=[r12.GRID[rng.randrange(len(r12.GRID))] for _ in range(horizon)]
    innate=arm in ("innate","both")
    learning=arm in ("learning","both")
    records=[]
    trace=[]
    for t in range(horizon):
        if not w.food or w.body.energy<=0 or w.body.integrity<=0:
            break
        odor=(0.,)*3 if masked else w.odor()
        observed=w.perceive_extended()+odor
        base,h,_=brain.forward(observed[:8])
        hunger=1-w.body.energy/100.
        cmd=exploration[t] if training and t%4==0 else choose(critic,h,observed,base,hunger,innate,learning)
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
            trace.append(dict(step,macro=t,command=cmd,odor=odor))
        if training and learning:
            b.update_head(critic,0,phi,min(1.,distance/(r1.V_MAX*r12.MICRO_DT*r12.SUBSTEPS)))
            b.update_head(critic,1,phi,min(1.,damage/r1.COLLISION_INJURY))
        records.append((phi,gain))
    if training and learning:
        for (phi,_),target in zip(records,b.returns([g for _,g in records])):
            b.update_head(critic,2,phi,target)
    cost=sum(f["metabolic_cost"] for f in trace)
    gain=sum(f["food_gain"] for f in trace)
    assert abs(60.-cost+gain-w.body.energy)<1e-7
    return dict(index=index,far=far,masked=masked,encounter=not w.food,macros=len(records),
                food=w.initial_food,distance=sum(f["moved"] for f in trace),
                damage=100.-w.body.integrity,energy=w.body.energy,cost=cost,gain=gain,trace=trace)

def simulate(seed,arm,train_n=40,test_n=20):
    c=r14.Critic()
    training=[episode(seed,i,arm,c) for i in range(train_n)]
    frozen=[w[:] for w in c.weights]
    near=[episode(seed,100+i,arm,c,False,False,False,80) for i in range(test_n)]
    far=[episode(seed,200+i,arm,c,False,True,False,80) for i in range(test_n)]
    masked=[episode(seed,200+i,arm,c,False,True,True,80) for i in range(test_n)]
    assert frozen==c.weights
    return dict(seed=seed,arm=arm,training=training,near=near,far=far,masked=masked,weights=frozen)

def run():
    rows=[simulate(s,a) for s in SEEDS for a in ARMS]
    summary={a:{p:b.fraction([r for r in rows if r["arm"]==a],p) for p in ("training","near","far","masked")} for a in ARMS}
    comparisons=(("innate_vs_neither","innate","far","neither","far",.10),
                 ("both_vs_learning","both","far","learning","far",.10),
                 ("both_vs_innate","both","far","innate","far",.05),
                 ("both_vs_masked","both","far","both","masked",.10))
    gates={n:summary[a][p]-summary[c][q]>=threshold for n,a,p,c,q,threshold in comparisons}
    gates["both_near"]=summary["both"]["near"]>=.50
    by={(r["seed"],r["arm"]):r for r in rows}
    intervals={}
    for name,a,p,c,q,_ in comparisons:
        delta=[b.fraction([by[s,a]],p)-b.fraction([by[s,c]],q) for s in SEEDS]
        rng=random.Random(171717)
        values=sorted(sum(rng.choice(delta) for _ in delta)/len(delta) for _ in range(2000))
        intervals[name]=[values[50],values[1949]]
    return dict(summary=summary,gates=gates,passed=all(gates.values()),intervals=intervals,rows=rows)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",default="results/r15.json")
    args=p.parse_args()
    result=run()
    path=Path(args.out)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,sort_keys=True,separators=(",",":")))
    print(json.dumps({k:v for k,v in result.items() if k!="rows"},indent=2))
