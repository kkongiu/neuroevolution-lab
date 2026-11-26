"""Exploratory R14 homeostatic food-energy learning, no scripted pursuit."""
import argparse
import json
import math
import random
from pathlib import Path
from neuroevolution import flyloop_r1 as r1, flyloop_r12 as r12, flyloop_r13 as r13, flyloop_r13b as r13b

SEEDS=range(15100,15124)
ARMS=("metabolic","no_preference","no_odor","no_food")
TRAIN,TEST=600,200

class World(r13.World):
    def __init__(self,seed,geometry,food=True):
        super().__init__(seed,geometry)
        self.body.energy=60.
        rng=random.Random(seed*61331+14014)
        points=[]
        for _ in range(10000):
            p=(rng.uniform(r1.RADIUS,r1.SIDE-r1.RADIUS),rng.uniform(r1.RADIUS,r1.SIDE-r1.RADIUS))
            if not self.blocked(*p) and math.dist(p,(self.body.x,self.body.y))>=1.:
                points.append(p)
                if len(points)==32:
                    break
        if len(points)!=32:
            raise RuntimeError("Insufficient food placement")
        self.initial_food=tuple(points) if food else ()
        self.food=list(self.initial_food)

    def odor(self):
        b=self.body
        result=[0.]*3
        for x,y in self.food:
            dx,dy=x-b.x,y-b.y
            d=math.hypot(dx,dy)
            if d>4.:
                continue
            angle=math.atan2(dy,dx) if d>1e-12 else b.heading
            if self.ray_distance(angle)+1e-9<min(d,r1.RAY_LENGTH):
                continue
            # Original ray clips at 3.5; explicitly test visibility beyond it.
            if d>r1.RAY_LENGTH and any(
                self._ray_box(b.x,b.y,math.cos(angle),math.sin(angle),
                    x1-r1.RADIUS,y1-r1.RADIUS,x2+r1.RADIUS,y2+r1.RADIUS)<d
                for x1,y1,x2,y2 in self.rectangles):
                continue
            for i,offset in enumerate((0.,math.pi/2,-math.pi/2)):
                diff=abs((angle-b.heading-offset+math.pi)%(2*math.pi)-math.pi)
                if diff<=math.pi/2:
                    result[i]=max(result[i],1-d/4.)
        return tuple(result)

    def micro_integrate(self,command):
        energy_before=self.body.energy
        step=super().micro_integrate(command)
        b=self.body
        original_cost=energy_before-b.energy
        extra_cost=min(b.energy,.025+.02*abs(b.speed)/r1.V_MAX)
        b.energy-=extra_cost
        gained=0.
        consumed=None
        if b.energy>0 and b.integrity>0 and b.energy<95.:
            for i,p in enumerate(self.food):
                if math.dist(p,(b.x,b.y))<=.55:
                    consumed=self.food.pop(i)
                    gained=min(12.,100.-b.energy)
                    b.energy+=gained
                    break
        step.update(energy_before=energy_before,energy=b.energy,
                    metabolic_cost=original_cost+extra_cost,food_gain=gained,
                    consumed=consumed,discarded=12.-gained if consumed is not None else 0.)
        return step

class Critic(r13.Critic):
    def __init__(self):
        n=len(r13.features([0.]*16,[0.]*35,(0.,0.)))
        self.weights=[[0.]*n for _ in range(3)]

    def predict(self,phi):
        return tuple(r1.clamp(offset+r1.dot(w,phi),0.,1.)
                     for offset,w in zip((.5,.25,0.),self.weights))

    def choose_metabolic(self,hidden,observed,baseline,hunger,preference):
        def score(cmd):
            p,i,g=self.predict(r13.features(hidden,observed,cmd))
            return p-2*i+preference*hunger*g
        return max((tuple(baseline),)+r12.GRID,key=score)

def simulate(seed,geometry,arm,train=TRAIN,test=TEST):
    if arm not in ARMS:
        raise ValueError(arm)
    w=World(seed,geometry,arm!="no_food")
    brain=r1.Circuit(seed,"recurrent")
    critic=Critic()
    rng=random.Random(seed*17239+13)
    phases=[]
    for training,length,offset in ((True,train,0),(False,test,train)):
        phase=r13.Phase()
        trace=[]
        start=(w.body.x,w.body.y)
        heading=w.body.heading
        energy=w.body.energy
        food_start=len(w.food)
        for t in range(length):
            if not r13b.alive(w):
                break
            observed=w.perceive_extended()+(w.odor() if arm!="no_odor" else (0.,)*3)
            base,hidden,_=brain.forward(observed[:8])
            hunger=1-w.body.energy/100.
            cmd=r12.GRID[rng.randrange(len(r12.GRID))] if training and t%4==0 else critic.choose_metabolic(hidden,observed,base,hunger,0. if arm=="no_preference" else 4.)
            phi=r13.features(hidden,observed,cmd)
            distance=injury=gain=0.
            for _ in range(r12.SUBSTEPS):
                if not r13b.alive(w):
                    break
                before=w.body.integrity
                step=w.micro_integrate(cmd)
                damage=before-w.body.integrity
                phase.add(w,step,damage)
                distance+=step["moved"]
                injury+=damage
                gain+=step["food_gain"]
                trace.append(dict(step,macro=t+offset,command=cmd,actual_speed=w.actual_speed,injury=damage))
            if training:
                critic.learn(phi,(r1.clamp(distance/(r1.V_MAX*r12.MICRO_DT*r12.SUBSTEPS),0.,1.),
                                  r1.clamp(injury/r1.COLLISION_INJURY,0.,1.),min(1.,gain/12.)))
                phase.updates+=1
        m=r13b.metrics(trace,start,heading,phase)
        m.update(food_gain=sum(f["food_gain"] for f in trace),
                 food_count=sum(f["consumed"] is not None for f in trace),
                 metabolic_cost=sum(f["metabolic_cost"] for f in trace),
                 discarded=sum(f["discarded"] for f in trace),
                 initial_energy=energy,final_energy=w.body.energy,
                 food_start=food_start,food_remaining=len(w.food),alive=r13b.alive(w))
        phases.append(dict(metrics=m,trace=trace))
    return dict(seed=seed,geometry=geometry,arm=arm,rectangles=w.rectangles,
                initial_food=w.initial_food,training=phases[0],test=phases[1],
                survived=r13b.alive(w) and len(phases[1]["trace"])==test*r12.SUBSTEPS,
                weights=critic.weights)

def aggregate(rows):
    sums={}
    for phase in ("training","test"):
        sums[phase]={k:sum(r[phase]["metrics"][k] for r in rows)
                     for k in ("food_gain","food_count","distance","cells","hits","injury","metabolic_cost","discarded")}
        sums[phase]["collision_density"]=sums[phase]["hits"]/max(sums[phase]["distance"],1e-12)
        for k in ("net_path_ratio","revisit_fraction","reverse_fraction","food_remaining","final_energy"):
            sums[phase][k]=sum(r[phase]["metrics"][k] for r in rows)/max(len(rows),1)
        sums[phase]["alive"]=sum(r[phase]["metrics"]["alive"] for r in rows)
    sums["survived"]=sum(r["survived"] for r in rows)
    for k in ("food_gain","distance","cells"):
        sums[k]=sum(sums[p][k] for p in ("training","test"))
    return sums

def intervals(rows,arm):
    pairs=[(next(r for r in rows if r["seed"]==s and r["arm"]=="metabolic"),
            next(r for r in rows if r["seed"]==s and r["arm"]==arm)) for s in SEEDS]
    delta=[sum(p[0][phase]["metrics"]["food_gain"]-p[1][phase]["metrics"]["food_gain"] for phase in ("training","test")) for p in pairs]
    rng=random.Random(151515)
    values=sorted(sum(rng.choice(delta) for _ in delta)/len(delta) for _ in range(2000))
    return [values[50],values[1949]]

def run():
    rows=[simulate(s,g,a) for g in r13.GEOMETRIES for s in SEEDS for a in ARMS]
    summary={g:{a:aggregate([r for r in rows if r["geometry"]==g and r["arm"]==a]) for a in ARMS} for g in r13.GEOMETRIES}
    p=summary["nearfield"]
    f,n,o=(p[a] for a in ("metabolic","no_preference","no_odor"))
    gates=dict(ingestion_vs_no_preference=n["food_gain"]>0 and f["food_gain"]>=1.2*n["food_gain"],
               distance=f["distance"]>=.9*n["distance"],cells=f["cells"]>=.9*n["cells"],
               survival=f["survived"]>=n["survived"],
               ingestion_vs_no_odor=o["food_gain"]>0 and f["food_gain"]>=1.15*o["food_gain"])
    near=[r for r in rows if r["geometry"]=="nearfield"]
    return dict(summary=summary,gates=gates,passed=all(gates.values()),
                ingestion_difference_intervals={a:intervals(near,a) for a in ("no_preference","no_odor")},rows=rows)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",default="results/r14.json")
    args=p.parse_args()
    result=run()
    path=Path(args.out)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,sort_keys=True,separators=(",",":")))
    print(json.dumps({k:v for k,v in result.items() if k!="rows"},indent=2))
