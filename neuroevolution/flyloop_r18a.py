"""R18A diagnostic: separate physical proximity from optical transmission."""
import argparse
import json
import math
import random
from pathlib import Path
from neuroevolution import flyloop_r1 as r1, flyloop_r12 as r12, flyloop_r14 as r14, flyloop_r15 as r15, flyloop_r17 as r17

SEEDS=range(21100,21124)
ARMS=("sparse","dense","clearance","memory","masked")
SCENARIOS=("free","glass","opaque","sealed")
HORIZON=200

def wrap(a):
    return (a+math.pi)%(2*math.pi)-math.pi

class WindowWorld(r14.World):
    def __init__(self,seed,index,scenario):
        super().__init__(seed*1000+index,"nearfield",False)
        rng=random.Random(seed*17311+index*7919)
        wall_x=11.
        start_dist=rng.uniform(1.8,2.5)
        gap=rng.uniform(1.5,2.5)
        self.gap_side=1 if rng.random()<.5 else -1
        # Local barrier connected to one world edge, opening at other end of local panel.
        # Full sealed control connects both world edges.
        edge=11.+self.gap_side*rng.uniform(1.1,1.8)
        self.body.x=wall_x-start_dist
        self.body.y=11.+rng.uniform(-.3,.3)
        self.body.heading=rng.uniform(-.45,.45)
        self.visited={(int(self.body.x),int(self.body.y))}
        self.lamp=(wall_x+1.7,11.+rng.uniform(-.3,.3))
        self.initial_food=(self.lamp,)
        self.food=list(self.initial_food)
        self.wall_x=wall_x
        self.gap=gap
        # Two panels with one opening of known width near the central region.
        lo=edge-gap/2
        hi=edge+gap/2
        self.opening=(lo,hi)
        if scenario=="free":
            self.rectangles=[]
        elif scenario=="sealed":
            self.rectangles=[(wall_x-.08,0.,wall_x+.08,r1.SIDE)]
        else:
            self.rectangles=[(wall_x-.08,0.,wall_x+.08,lo),
                             (wall_x-.08,hi,wall_x+.08,r1.SIDE)]
        self.initial_rectangles=list(self.rectangles)
        self.transmission=0. if scenario=="opaque" else .9
        assert not self.blocked(self.body.x,self.body.y)
        assert not self.blocked(*self.lamp)
        assert hi-lo>2*r1.RADIUS
    def light_at(self):
        b=self.body
        dx,dy=self.lamp[0]-b.x,self.lamp[1]-b.y
        d=math.hypot(dx,dy)
        angle=math.atan2(dy,dx) if d>1e-12 else b.heading
        value=max(0.,1-d/8.)
        for rect in self.rectangles:
            if self._ray_box(b.x,b.y,math.cos(angle),math.sin(angle),*rect)<d:
                value*=self.transmission
        return wrap(angle-b.heading),value

class CompoundEye:
    def __init__(self,sectors=9):
        if sectors<3 or sectors%2!=1:
            raise ValueError("odd sector count >=3")
        self.sectors=sectors
        self.width=math.pi/sectors
        self.centers=tuple(-math.pi/2+(i+.5)*self.width for i in range(sectors))
    def observe(self,w,masked=False):
        proximity=tuple(max(1-w.ray_distance(w.body.heading+a+offset)/r1.RAY_LENGTH
                            for offset in (-self.width/3,0.,self.width/3))
                        for a in self.centers)
        bearing,brightness=w.light_at()
        light=[0.]*self.sectors
        if not masked and -math.pi/2<=bearing<=math.pi/2:
            i=min(self.sectors-1,max(0,int((bearing+math.pi/2)/self.width)))
            light[i]=brightness
        return proximity,tuple(light)
    def light_channels(self,light):
        # Shared continuous local phototaxis channels, no global heading.
        f=sum(v*max(0.,math.cos(a)) for a,v in zip(self.centers,light))
        l=sum(v*max(0.,math.sin(a)) for a,v in zip(self.centers,light))
        r=sum(v*max(0.,-math.sin(a)) for a,v in zip(self.centers,light))
        return f,l,r
    def pooled(self,proximity):
        # Match original center/positive/negative input order.
        n=self.sectors//2
        return (proximity[n],max(proximity[n+1:]),max(proximity[:n]))

def visual_motor_adapter(base,proximity,light,hunger,eye,clearance=False):
    command,_=r15.innate_command(base,light,hunger)
    if not clearance:
        return command
    center,left,right=eye.pooled(proximity)
    # Symmetric local clearance bias only, no gap or route information.
    return (r1.clamp(command[0]*(1-.8*center),-1.,1.),
            r1.clamp(command[1]+.8*(right-left),-1.,1.))

def episode(seed,index,scenario,arm,horizon=HORIZON):
    w=WindowWorld(seed,index,scenario)
    eye=CompoundEye()
    brain=r1.Circuit(seed,"recurrent")
    memory=r17.Memory()
    trace=[]
    crossed=False
    detection_before_contact=False
    first_contact=None
    visible=0
    start=(w.body.x,w.body.y,w.body.heading)
    for t in range(horizon):
        if not w.food or w.body.energy<=0 or w.body.integrity<=0:
            break
        proximity,light=eye.observe(w,arm=="masked")
        original=w.perceive()
        sensed=original if arm=="sparse" else (original[0],*eye.pooled(proximity),*original[4:])
        base,_,_=brain.forward(sensed)
        cue=eye.light_channels(light)
        visible+=max(light)>0.
        if arm=="memory":
            cue,_=memory.observe(cue)
        command=visual_motor_adapter(base,proximity,cue,1-w.body.energy/100.,eye,arm in ("clearance","memory","masked"))
        detected=max(sensed[1:4])>0.
        if first_contact is None and detected:
            detection_before_contact=True
        for _ in range(r12.SUBSTEPS):
            if not w.food or w.body.energy<=0 or w.body.integrity<=0:
                break
            step=w.micro_integrate(command)
            if arm=="memory":
                memory.rotate(w.body.turn*r12.MICRO_DT)
            if step["collision"] and first_contact is None:
                first_contact=len(trace)
            if w.body.x>w.wall_x+.08+r1.RADIUS:
                crossed=True
            trace.append(dict(step,macro=t,command=command,proximity=proximity,light=light,
                              used_proximity=sensed[1:4],effective_light=cue))
    gain=sum(f["food_gain"] for f in trace)
    cost=sum(f["metabolic_cost"] for f in trace)
    assert abs(60.-cost+gain-w.body.energy)<1e-7
    return dict(seed=seed,index=index,scenario=scenario,arm=arm,start=start,lamp=w.lamp,
                rectangles=w.rectangles,opening=w.opening,encounter=not w.food,
                crossed=crossed,precontact=detection_before_contact,first_contact=first_contact,
                contacts=sum(f["collision"] for f in trace),damage=100.-w.body.integrity,
                distance=sum(f["moved"] for f in trace),steps=len(trace),visible=visible,
                energy=w.body.energy,trace=trace)

def aggregate(rows):
    n=len(rows)
    successes=[r for r in rows if r["encounter"]]
    return dict(episodes=n,encounter=len(successes)/n,crossed=sum(r["crossed"] for r in rows)/n,
                precontact=sum(r["precontact"] for r in rows)/n,
                mean_contacts=sum(r["contacts"] for r in rows)/n,
                mean_damage=sum(r["damage"] for r in rows)/n,
                mean_distance=sum(r["distance"] for r in rows)/n,
                mean_success_seconds=sum(r["steps"] for r in successes)*r12.MICRO_DT/len(successes) if successes else None,
                visible_macros=sum(r["visible"] for r in rows))

def run():
    rows=[episode(s,i,p,a) for s in SEEDS for i in range(20) for p in SCENARIOS for a in ARMS]
    summary={p:{a:aggregate([r for r in rows if r["scenario"]==p and r["arm"]==a]) for a in ARMS} for p in SCENARIOS}
    comparisons=(("dense_glass","glass","dense","sparse",.05),
                 ("clearance_glass","glass","clearance","dense",.05),
                 ("retain_free","free","clearance","sparse",-.05),
                 ("memory_increment","glass","memory","clearance",None),
                 ("light_dependence","glass","clearance","masked",None))
    gates={}
    intervals={}
    for name,p,a,b,threshold in comparisons:
        if threshold is not None:
            gates[name]=summary[p][a]["encounter"]-summary[p][b]["encounter"]>=threshold
        delta=[aggregate([r for r in rows if r["seed"]==s and r["scenario"]==p and r["arm"]==a])["encounter"]-
               aggregate([r for r in rows if r["seed"]==s and r["scenario"]==p and r["arm"]==b])["encounter"] for s in SEEDS]
        rng=random.Random(212121)
        values=sorted(sum(rng.choice(delta) for _ in delta)/len(delta) for _ in range(2000))
        intervals[name]=[values[50],values[1949]]
    gates["sealed_zero"]=all(not r["encounter"] and not r["crossed"] for r in rows if r["scenario"]=="sealed")
    gates["masked_zero"]=all(not any(f["light"]) and not any(f["effective_light"]) for r in rows if r["arm"]=="masked" for f in r["trace"])
    return dict(summary=summary,gates=gates,passed=all(gates.values()),intervals=intervals,rows=rows)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",default="results/r18a.json")
    args=p.parse_args()
    result=run()
    path=Path(args.out)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,sort_keys=True,separators=(",",":")))
    print(json.dumps({k:v for k,v in result.items() if k!="rows"},indent=2))
