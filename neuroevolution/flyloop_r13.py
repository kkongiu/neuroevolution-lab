"""Exploratory slow-only learning; no scripted escape or fast motor."""
import argparse
import json
import math
import random
from pathlib import Path
from neuroevolution import flyloop_r1 as r1, flyloop_r2 as r2, flyloop_r12 as r12, flyloop_r12c as senses

SEEDS=range(13100,13124)
GEOMETRIES=("obstacles","nearfield")
ARMS=("full","original","delayed","fixed")
TRAIN,TEST=600,200

class World(senses.World):
    def __init__(self,seed,geometry):
        # Same original physical initialization, with prospectively new seeds.
        self.seed,self.geometry,self.regime,self.scenario=seed,geometry,"static",geometry
        self.initial_rectangles=(r1.procedural_rectangles(seed) if geometry=="obstacles" else r2.nearfield_rectangles(seed))
        self.rectangles=list(self.initial_rectangles)
        self.body=r1.Body(heading=random.Random(seed*11113+51).uniform(-math.pi,math.pi))
        self.frame,self.collisions,self.distance=0,0,0.
        self.visited={(int(self.body.x),int(self.body.y))}
        self.trace=[]
        self.changed=False
        self.plant="reverse"
        self.touch=(0.,)*8
        self.duration=(0,)*8
        self.actual_speed=self.motor_target=self.blocked_fraction=self.injury=0.

def features(hidden,observed,action):
    # No absolute position or map.
    a,b=action
    state=tuple(hidden)+tuple(observed)
    return (1.,a,b,a*a,b*b,a*b,*state,
            *(x*a for x in state),*(x*b for x in state))

class Critic:
    def __init__(self):
        n=len(features([0.]*16,[0.]*senses.SLOW_INPUTS,(0.,0.)))
        self.weights=[[0.]*n for _ in range(2)]
    def predict(self,phi):
        return tuple(r1.clamp(offset+r1.dot(w,phi),0.,1.) for offset,w in zip((.5,.25),self.weights))
    def learn(self,phi,targets):
        prediction=self.predict(phi)
        norm=max(1.,sum(x*x for x in phi))
        for w,p,t in zip(self.weights,prediction,targets):
            rate=.08*(t-p)/norm
            for j,x in enumerate(phi):
                w[j]+=rate*x
    def choose(self,hidden,observed,baseline):
        menu=(tuple(baseline),)+r12.GRID
        def score(cmd):
            p,i=self.predict(features(hidden,observed,cmd))
            return p-2*i
        return max(menu,key=score)

class Phase:
    def __init__(self):
        self.distance=0.
        self.hits=0
        self.injury=0.
        self.cells=set()
        self.frames=[]
        self.updates=0
    def add(self,w,step,injury):
        self.distance+=step["moved"]
        self.hits+=step["collision"]
        self.injury+=injury
        self.cells.add((int(w.body.x),int(w.body.y)))
        self.frames.append((step["collision"],step["moved"]))
    def report(self):
        opportunities,recoveries=0,0
        f=self.frames
        # Non-overlapping contact episodes, only full 32-frame horizons.
        i=0
        while i<len(f)-32:
            if f[i][0] and (i==0 or not f[i-1][0]):
                opportunities+=1
                found=None
                for j in range(i+1,i+30):
                    window=f[j:j+4]
                    if not any(h for h,d in window) and sum(d for h,d in window)>=.035:
                        found=j+4
                        break
                if found is not None:
                    recoveries+=1
                i=found if found is not None else i+33
            else:
                i+=1
        return dict(distance=self.distance,hits=self.hits,injury=self.injury,cells=len(self.cells),
                    microsteps=len(f),opportunities=opportunities,recoveries=recoveries,updates=self.updates)

def simulate(seed,geometry,arm,train=TRAIN,test=TEST):
    if arm not in ARMS:
        raise ValueError(arm)
    w=World(seed,geometry)
    brain=r1.Circuit(seed,"recurrent")
    critic=Critic()
    rng=random.Random(seed*17239+13)
    phases=[Phase(),Phase()]
    extra_history=[]
    trace=[]
    for macro in range(train+test):
        if w.body.energy<=0 or w.body.integrity<=0:
            break
        training=macro<train
        phase=phases[not training]
        original=w.perceive()
        extra=w.extra_senses()
        extra_history.append(extra)
        observed=original+(extra if arm=="full" else
            extra_history[-21] if arm=="delayed" and len(extra_history)>20 else
            (0.,)*len(extra))
        baseline,hidden,_=brain.forward(original)
        if training and macro%4==0:
            command=r12.GRID[rng.randrange(len(r12.GRID))]
        elif arm=="fixed":
            command=baseline
        else:
            command=critic.choose(hidden,observed,baseline)
        phi=features(hidden,observed,command)
        distance,injury=0.,0.
        events=[]
        for _ in range(r12.SUBSTEPS):
            if w.body.energy<=0 or w.body.integrity<=0:
                break
            before=w.body.integrity
            step=w.micro_integrate(command)
            damage=before-w.body.integrity
            phase.add(w,step,damage)
            distance+=step["moved"]
            injury+=damage
            events.append((step["collision"],step["moved"],damage))
        targets=(r1.clamp(distance/(r1.V_MAX*r12.MICRO_DT*r12.SUBSTEPS),0.,1.),
                 r1.clamp(injury/r1.COLLISION_INJURY,0.,1.))
        if training and arm!="fixed":
            critic.learn(phi,targets)
            phase.updates+=1
        trace.append(dict(macro=macro,observed=observed,hidden=hidden,command=command,events=events,targets=targets))
    return dict(seed=seed,geometry=geometry,arm=arm,training=phases[0].report(),test=phases[1].report(),
                survived=len(trace)==train+test and w.body.integrity>0 and w.body.energy>0,
                weights=critic.weights,trace=trace)

def aggregate(rows):
    sums={k:sum(r["test"][k] for r in rows) for k in ("distance","hits","injury","cells","opportunities","recoveries")}
    sums["survived"]=sum(r["survived"] for r in rows)
    sums["collision_density"]=sums["hits"]/max(sums["distance"],1e-12)
    return sums

def bootstrap(rows,ref):
    pairs=[(next(r for r in rows if r["seed"]==s and r["arm"]=="full"),
            next(r for r in rows if r["seed"]==s and r["arm"]==ref)) for s in SEEDS]
    rng=random.Random(131313)
    values=[]
    for _ in range(2000):
        sample=[pairs[rng.randrange(len(pairs))] for _ in pairs]
        a,b=(aggregate([p[i] for p in sample]) for i in (0,1))
        values.append(a["collision_density"]-b["collision_density"])
    values.sort()
    return [values[50],values[1949]]

def run():
    rows=[simulate(s,g,a) for g in GEOMETRIES for s in SEEDS for a in ARMS]
    summary={g:{a:aggregate([r for r in rows if r["geometry"]==g and r["arm"]==a]) for a in ARMS} for g in GEOMETRIES}
    primary=summary["nearfield"]
    f,o,d=(primary[a] for a in ("full","original","delayed"))
    gates=dict(collision_vs_original=o["collision_density"]>0 and f["collision_density"]<=.8*o["collision_density"],
               distance=f["distance"]>=.9*o["distance"],cells=f["cells"]>=.9*o["cells"],
               survival=f["survived"]>=o["survived"],
               collision_vs_delayed=d["collision_density"]>0 and f["collision_density"]<=.85*d["collision_density"])
    near=[r for r in rows if r["geometry"]=="nearfield"]
    return dict(summary=summary,gates=gates,passed=all(gates.values()),
                bootstrap_density_difference={a:bootstrap(near,a) for a in ("original","delayed")},rows=rows)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",default="results/r13.json")
    args=p.parse_args()
    result=run()
    path=Path(args.out)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,sort_keys=True,separators=(",",":")))
    print(json.dumps({k:v for k,v in result.items() if k!="rows"},indent=2))
