"""R18B: learned visual intervention, no scripted detours."""
import argparse
import json
import math
import random
from pathlib import Path
from neuroevolution import flyloop_r1 as r1, flyloop_r12 as r12, flyloop_r13 as r13, flyloop_r14 as r14, flyloop_r14b as b, flyloop_r16 as r16, flyloop_r18a as eye18

SEEDS=range(22100,22124)
ARMS=("full","pooled","always")
SCENARIOS=eye18.SCENARIOS
HORIZON=200

class Critic(r14.Critic):
    def __init__(self):
        self.weights=[[0.]*len(r13.features([0.]*16,[0.]*50,(0.,0.))) for _ in range(3)]

def pool(values):
    return tuple([max(values[:4])]*4+[values[4]]+[max(values[5:])]*4)

def state_bin(obs):
    proximity,light=obs[-18:-9],obs[-9:]
    peak=max(light)
    i=max(range(9),key=lambda j:light[j])
    direction=3 if peak<.05 else 0 if i<4 else 1 if i==4 else 2
    diff=max(proximity[5:])-max(proximity[:4])
    return (direction,int(peak>=.3),int(proximity[4]>=.3),
            -1 if diff<-.15 else 1 if diff>.15 else 0,int(obs[29]>=.5))

def choose(c,e,h,obs,baseline,hunger,mode):
    if mode=="sham":
        return baseline,False
    def score(cmd):
        p,i,g=c.predict(r13.features(h,obs,cmd))
        return p-2*i+4*hunger*g
    if mode=="always":
        cmd=max((baseline,)+r12.GRID,key=score)
        return cmd,cmd!=baseline
    state=state_bin(obs)
    ref=e.bounds(state,baseline)
    if ref is None:
        return baseline,False
    value=score(baseline)
    eligible=[]
    for cmd in r12.GRID:
        if r16.motor_bin(cmd)==r16.motor_bin(baseline):
            continue
        bounds=e.bounds(state,cmd)
        if bounds is not None and bounds[0]>ref[1]+.10 and score(cmd)>value+.10:
            eligible.append(cmd)
    return (max(eligible,key=score),True) if eligible else (baseline,False)

def episode(seed,index,scenario,arm,c,e,training=False,sham=False,horizon=HORIZON):
    w=eye18.WindowWorld(seed,index,scenario)
    eye=eye18.CompoundEye()
    brain=r1.Circuit(seed,"recurrent")
    rng=random.Random(seed*17239+index*7919+13)
    exploration=[r12.GRID[rng.randrange(len(r12.GRID))] for _ in range(horizon)]
    records=[]
    trace=[]
    crossed=False
    interventions=opportunities=conflicts=conflict_interventions=0
    start=(w.body.x,w.body.y,w.body.heading)
    for t in range(horizon):
        if not w.food or w.body.energy<=0 or w.body.integrity<=0:
            break
        prox,light=eye.observe(w)
        original=w.perceive()
        base,h,_=brain.forward((original[0],*eye.pooled(prox),*original[4:]))
        hunger=1-w.body.energy/100.
        baseline=eye18.visual_motor_adapter(base,prox,eye.light_channels(light),hunger,eye,True)
        delivered=(pool(prox)+pool(light)) if arm=="pooled" else prox+light
        obs=w.perceive_extended()+delivered
        assert len(obs)==50
        explore=training and t%4==0
        mode="sham" if sham else "always" if arm=="always" else "gated"
        cmd,override=(exploration[t],False) if explore else choose(c,e,h,obs,baseline,hunger,mode)
        conflict=prox[4]>=.3 and max(light)>=.3
        opportunities+=not explore
        interventions+=override
        conflicts+=conflict and not explore
        conflict_interventions+=conflict and override
        phi=r13.features(h,obs,cmd)
        distance=damage=gain=0.
        for _ in range(r12.SUBSTEPS):
            if not w.food or w.body.energy<=0 or w.body.integrity<=0:
                break
            before=w.body.integrity
            step=w.micro_integrate(cmd)
            distance+=step["moved"]
            damage+=before-w.body.integrity
            gain+=step["food_gain"]
            crossed|=w.body.x>w.wall_x+.08+r1.RADIUS
            trace.append(dict(step,macro=t,command=cmd,baseline=baseline,
                              override=override,explore=explore,conflict=conflict,
                              proximity=prox,light=light,delivered_vision=delivered))
        p=min(1.,distance/(r1.V_MAX*r12.MICRO_DT*r12.SUBSTEPS))
        injury=min(1.,damage/r1.COLLISION_INJURY)
        if training:
            b.update_head(c,0,phi,p)
            b.update_head(c,1,phi,injury)
            records.append((phi,gain,p,injury,hunger,state_bin(obs),cmd))
    if training:
        for (phi,g,p,i,hunger,state,cmd),target in zip(records,b.returns([v[1] for v in records])):
            b.update_head(c,2,phi,target)
            e.add(state,cmd,p-2*i+4*hunger*target)
    cost=sum(f["metabolic_cost"] for f in trace)
    gain=sum(f["food_gain"] for f in trace)
    assert abs(60.-cost+gain-w.body.energy)<1e-7
    return dict(seed=seed,index=index,scenario=scenario,arm=arm,sham=sham,start=start,
                lamp=w.lamp,rectangles=w.rectangles,opening=w.opening,
                encounter=not w.food,crossed=crossed,contacts=sum(f["collision"] for f in trace),
                damage=100.-w.body.integrity,distance=sum(f["moved"] for f in trace),
                interventions=interventions,opportunities=opportunities,conflicts=conflicts,
                conflict_interventions=conflict_interventions,trace=trace)

def simulate(seed,arm,train_n=60,test_n=20,horizon=HORIZON):
    c=Critic()
    e=r16.Evidence()
    train=[episode(seed,i,SCENARIOS[i%3],arm,c,e,True,False,horizon) for i in range(train_n)]
    weights=[w[:] for w in c.weights]
    evidence=dict(e.data)
    tests={}
    for mode in (("always",) if arm=="always" else ("gated","sham")):
        tests[mode]={p:[episode(seed,100+i,p,arm,c,e,False,mode=="sham",horizon)
                         for i in range(test_n)] for p in SCENARIOS}
    assert weights==c.weights and evidence==e.data
    return dict(seed=seed,arm=arm,training=train,tests=tests,weights=weights,
                evidence=sorted(evidence.items()))

def aggregate(rows):
    n=len(rows)
    return dict(episodes=n,encounter=sum(r["encounter"] for r in rows)/n,
                crossed=sum(r["crossed"] for r in rows)/n,
                mean_contacts=sum(r["contacts"] for r in rows)/n,
                mean_damage=sum(r["damage"] for r in rows)/n,
                mean_distance=sum(r["distance"] for r in rows)/n,
                intervention_fraction=sum(r["interventions"] for r in rows)/max(1,sum(r["opportunities"] for r in rows)),
                conflict_intervention_fraction=sum(r["conflict_interventions"] for r in rows)/max(1,sum(r["conflicts"] for r in rows)),
                conflicts=sum(r["conflicts"] for r in rows))

def run():
    rows=[simulate(s,a) for s in SEEDS for a in ARMS]
    return summarize(rows)

def summarize(rows):
    def scope(a,m,p,seed=None):
        return [ep for r in rows if r["arm"]==a and (seed is None or r["seed"]==seed)
                for ep in r["tests"][m][p]]
    combos=(("full","gated"),("full","sham"),("pooled","gated"),("pooled","sham"),("always","always"))
    summary={a+"_"+m:{p:aggregate(scope(a,m,p)) for p in SCENARIOS} for a,m in combos}
    comparisons=(("full_vs_sham","glass","full","gated","full","sham",.05),
                 ("full_vs_pooled","glass","full","gated","pooled","gated",.05),
                 ("retain_free","free","full","gated","full","sham",-.05))
    gates={}
    intervals={}
    for name,p,a,m,d,k,threshold in comparisons:
        delta=[aggregate(scope(a,m,p,s))["encounter"]-aggregate(scope(d,k,p,s))["encounter"] for s in SEEDS]
        gates[name]=sum(delta)/len(delta)>=threshold
        rng=random.Random(222222)
        values=sorted(sum(rng.choice(delta) for _ in delta)/len(delta) for _ in range(2000))
        intervals[name]=[values[50],values[1949]]
    full=summary["full_gated"]["glass"]
    gates["contacts"]=full["mean_contacts"]<=summary["full_sham"]["glass"]["mean_contacts"]
    gates["conflict_interventions"]=full["conflict_intervention_fraction"]>=.01
    gates["sealed_zero"]=all(not r["encounter"] and not r["crossed"] for a,m in combos for r in scope(a,m,"sealed"))
    return dict(summary=summary,training={a:aggregate([ep for r in rows if r["arm"]==a for ep in r["training"]]) for a in ARMS},
                gates=gates,passed=all(gates.values()),intervals=intervals,rows=rows)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",default="results/r18b.json")
    args=p.parse_args()
    result=run()
    path=Path(args.out)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,sort_keys=True,separators=(",",":")))
    print(json.dumps({k:v for k,v in result.items() if k!="rows"},indent=2))
