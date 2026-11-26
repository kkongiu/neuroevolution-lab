"""R16 conservative learned veto; only local observations reach the controller."""
import argparse
import json
import math
import random
from pathlib import Path
from neuroevolution import flyloop_r1 as r1, flyloop_r12 as r12, flyloop_r13 as r13, flyloop_r14 as r14, flyloop_r14b as b, flyloop_r15 as r15

SEEDS=range(18100,18124)
ARMS=("innate","always","gated","sham")
PHASES=("training","near","far","masked","obstacle")

def state_bin(obs):
    odor=obs[-3:]
    peak=max(odor)
    speed=obs[4]
    return (max(range(3),key=lambda i:odor[i]) if peak>=.05 else 3,
            int(peak>=.4),int(obs[8+21]>=.5),
            -1 if speed<-.15 else 1 if speed>.15 else 0)

def motor_bin(cmd):
    return tuple(min(4,max(0,int(math.floor((v+1)*2+.5)))) for v in cmd)

class Evidence:
    def __init__(self):
        self.data={}
    def add(self,state,cmd,value):
        key=(state,motor_bin(cmd))
        n,s,q=self.data.get(key,(0,0.,0.))
        self.data[key]=(n+1,s+value,q+value*value)
    def bounds(self,state,cmd):
        n,s,q=self.data.get((state,motor_bin(cmd)),(0,0.,0.))
        if n<8:
            return None
        mean=s/n
        se=math.sqrt(max(0.,(q-s*s/n)/(n-1))/n)
        return mean-2*se,mean+2*se

def choose(critic,evidence,hidden,observed,base,hunger,arm):
    instinct,blend=r15.innate_command(base,observed[-3:],hunger)
    if arm=="innate":
        return instinct,False
    if arm=="always":
        cmd=r15.choose(critic,hidden,observed,base,hunger,True,True)
        return cmd,cmd!=instinct
    state=state_bin(observed)
    reference=evidence.bounds(state,instinct)
    if reference is None:
        return instinct,False
    def score(cmd):
        p,i,g=critic.predict(r13.features(hidden,observed,cmd))
        return p-2*i+4*hunger*g
    baseline=score(instinct)
    eligible=[]
    for cmd in r12.GRID:
        if motor_bin(cmd)==motor_bin(instinct):
            continue
        bounds=evidence.bounds(state,cmd)
        if bounds is not None and bounds[0]>reference[1]+.10 and score(cmd)>baseline+.10:
            eligible.append(cmd)
    if not eligible:
        return instinct,False
    return max(eligible,key=score),True

def make_world(seed,index,far,obstacle):
    if not obstacle:
        return b.make_world(seed,index,far)
    for attempt in range(100):
        w=b.make_world(seed,index+1000*attempt,True)
        x,y=w.body.x,w.body.y
        fx,fy=w.food[0]
        mx,my=(x+fx)/2,(y+fy)/2
        # Fixed square at midpoint, with room for both body and food.
        rect=(mx-.24,my-.24,mx+.24,my+.24)
        w.rectangles=[rect]
        if not w.blocked(x,y) and not w.blocked(fx,fy) and all(r1.RADIUS<v<r1.SIDE-r1.RADIUS for v in rect):
            return w
    raise RuntimeError("No valid obstacle layout")

def episode(seed,index,arm,critic,evidence,training=True,far=False,masked=False,obstacle=False,horizon=40):
    w=make_world(seed,index,far,obstacle)
    brain=r1.Circuit(seed,"recurrent")
    rng=random.Random(seed*17239+index*7919+13)
    exploration=[r12.GRID[rng.randrange(len(r12.GRID))] for _ in range(horizon)]
    mode="gated" if arm=="sham" and training else "innate" if arm=="sham" else arm
    learning=training and arm!="innate"
    records=[]
    trace=[]
    interventions=opportunities=0
    for t in range(horizon):
        if not w.food or w.body.energy<=0 or w.body.integrity<=0:
            break
        odor=(0.,)*3 if masked else w.odor()
        obs=w.perceive_extended()+odor
        base,h,_=brain.forward(obs[:8])
        hunger=1-w.body.energy/100.
        explore=training and t%4==0
        instinct,_=r15.innate_command(base,odor,hunger)
        cmd,override=(exploration[t],False) if explore else choose(critic,evidence,h,obs,base,hunger,mode)
        opportunities+=not explore
        interventions+=override
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
            trace.append(dict(step,macro=t,command=cmd,instinct=instinct,override=override,explore=explore,odor=odor))
        p=min(1.,distance/(r1.V_MAX*r12.MICRO_DT*r12.SUBSTEPS))
        i=min(1.,damage/r1.COLLISION_INJURY)
        if learning:
            b.update_head(critic,0,phi,p)
            b.update_head(critic,1,phi,i)
        records.append((phi,gain,p,i,hunger,state_bin(obs),cmd))
    if learning:
        for (phi,g,p,i,hunger,state,cmd),target in zip(records,b.returns([v[1] for v in records])):
            b.update_head(critic,2,phi,target)
            evidence.add(state,cmd,p-2*i+4*hunger*target)
    cost=sum(f["metabolic_cost"] for f in trace)
    gain=sum(f["food_gain"] for f in trace)
    assert abs(60.-cost+gain-w.body.energy)<1e-7
    return dict(index=index,food=w.initial_food,rectangles=w.rectangles,encounter=not w.food,
                distance=sum(f["moved"] for f in trace),damage=100.-w.body.integrity,
                energy=w.body.energy,cost=cost,gain=gain,interventions=interventions,
                opportunities=opportunities,trace=trace)

def simulate(seed,arm,train_n=40,test_n=20):
    c=r14.Critic()
    evidence=Evidence()
    training=[episode(seed,i,arm,c,evidence) for i in range(train_n)]
    weights=[w[:] for w in c.weights]
    counts=dict(evidence.data)
    result=dict(seed=seed,arm=arm,training=training,weights=weights,
                evidence=[(k,v) for k,v in sorted(counts.items())])
    for phase,offset,far,masked,obstacle in (
        ("near",100,False,False,False),("far",200,True,False,False),
        ("masked",200,True,True,False),("obstacle",300,True,False,True)):
        result[phase]=[episode(seed,offset+i,arm,c,evidence,False,far,masked,obstacle,80) for i in range(test_n)]
    assert weights==c.weights and counts==evidence.data
    return result

def aggregate(rows,phase):
    episodes=[e for row in rows for e in row[phase]]
    n=len(episodes)
    opp=sum(e["opportunities"] for e in episodes)
    return dict(encounter=sum(e["encounter"] for e in episodes)/n,episodes=n,
                intervention_fraction=sum(e["interventions"] for e in episodes)/max(1,opp),
                mean_distance=sum(e["distance"] for e in episodes)/n,
                mean_damage=sum(e["damage"] for e in episodes)/n)

def run():
    rows=[simulate(s,a) for s in SEEDS for a in ARMS]
    summary={a:{p:aggregate([r for r in rows if r["arm"]==a],p) for p in PHASES} for a in ARMS}
    comparisons=(("retain_far","gated","far","innate","far",-.05),
                 ("improve_over_always","gated","far","always","far",.10),
                 ("obstacle_vs_sham","gated","obstacle","sham","obstacle",.05),
                 ("retain_near","gated","near","innate","near",-.05))
    gates={n:summary[a][p]["encounter"]-summary[c][q]["encounter"]>=threshold for n,a,p,c,q,threshold in comparisons}
    gates["nontrivial_interventions"]=summary["gated"]["obstacle"]["intervention_fraction"]>=.01
    by={(r["seed"],r["arm"]):r for r in rows}
    intervals={}
    for name,a,p,c,q,_ in comparisons:
        delta=[aggregate([by[s,a]],p)["encounter"]-aggregate([by[s,c]],q)["encounter"] for s in SEEDS]
        rng=random.Random(181818)
        values=sorted(sum(rng.choice(delta) for _ in delta)/len(delta) for _ in range(2000))
        intervals[name]=[values[50],values[1949]]
    return dict(summary=summary,gates=gates,passed=all(gates.values()),intervals=intervals,rows=rows)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",default="results/r16.json")
    args=p.parse_args()
    result=run()
    path=Path(args.out)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,sort_keys=True,separators=(",",":")))
    print(json.dumps({k:v for k,v in result.items() if k!="rows"},indent=2))
