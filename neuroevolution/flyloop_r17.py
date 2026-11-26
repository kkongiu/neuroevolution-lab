"""R17 sensory factorial diagnostic; no learned veto or scripted detour."""
import argparse
import json
import math
import random
from pathlib import Path
from neuroevolution import flyloop_r1 as r1, flyloop_r12 as r12, flyloop_r14b as b, flyloop_r15 as r15, flyloop_r16 as r16

SEEDS=range(20100,20124)
ARMS=("original","memory","attenuated","both")
PHASES=("free","midpoint","offset","masked")

def attenuated_odor(w):
    body=w.body
    out=[0.]*3
    for x,y in w.food:
        d=math.hypot(x-body.x,y-body.y)
        if d>4.:
            continue
        angle=math.atan2(y-body.y,x-body.x) if d>1e-12 else body.heading
        occluded=any(w._ray_box(body.x,body.y,math.cos(angle),math.sin(angle),
                      x1-r1.RADIUS,y1-r1.RADIUS,x2+r1.RADIUS,y2+r1.RADIUS)<d
                      for x1,y1,x2,y2 in w.rectangles)
        strength=(1-d/4.)*(.25 if occluded else 1.)
        for i,offset in enumerate((0.,math.pi/2,-math.pi/2)):
            if abs((angle-body.heading-offset+math.pi)%(2*math.pi)-math.pi)<=math.pi/2:
                out[i]=max(out[i],strength)
    return tuple(out)

class Memory:
    def __init__(self):
        self.strength=0.
        self.bearing=0.
    def rotate(self,turn):
        self.bearing=(self.bearing-turn+math.pi)%(2*math.pi)-math.pi
    def observe(self,odor):
        peak=max(odor)
        if peak>0.:
            f,l,r=odor
            self.bearing=math.atan2(l-r,f)
            self.strength=peak
            return odor,False
        self.strength*=2**(-1/10)
        if self.strength<.01:
            self.strength=0.
            return odor,False
        return tuple(self.strength if abs((self.bearing-a+math.pi)%(2*math.pi)-math.pi)<=math.pi/2 else 0.
                     for a in (0.,math.pi/2,-math.pi/2)),True

def make_world(seed,index,phase):
    if phase=="free":
        return b.make_world(seed,index,True)
    if phase=="midpoint":
        return r16.make_world(seed,index,True,True)
    for attempt in range(100):
        w=b.make_world(seed,index+1000*attempt,True)
        x,y=w.body.x,w.body.y
        fx,fy=w.food[0]
        d=math.hypot(fx-x,fy-y)
        sign=1 if (seed+index)%2 else -1
        mx=(x+fx)/2-sign*.75*(fy-y)/d
        my=(y+fy)/2+sign*.75*(fx-x)/d
        rect=(mx-.24,my-.24,mx+.24,my+.24)
        w.rectangles=[rect]
        if not w.blocked(x,y) and not w.blocked(fx,fy) and all(r1.RADIUS<v<r1.SIDE-r1.RADIUS for v in rect) and max(w.odor())>0.:
            return w
    raise RuntimeError("No valid offset layout")

def episode(seed,index,phase,arm,horizon=80):
    w=make_world(seed,index,phase)
    brain=r1.Circuit(seed,"recurrent")
    memory=Memory()
    use_memory=arm in ("memory","both")
    use_attenuation=arm in ("attenuated","both")
    trace=[]
    active=losses=reacquisitions=0
    prev_visible=None
    for t in range(horizon):
        if not w.food or w.body.energy<=0 or w.body.integrity<=0:
            break
        odor=(0.,)*3 if phase=="masked" else attenuated_odor(w) if use_attenuation else w.odor()
        visible=max(odor)>0.
        losses+=prev_visible is True and not visible
        reacquisitions+=prev_visible is False and visible
        prev_visible=visible
        sensed,used=memory.observe(odor) if use_memory else (odor,False)
        active+=used
        base,_,_=brain.forward(w.perceive())
        command,_=r15.innate_command(base,sensed,1-w.body.energy/100.)
        for _ in range(r12.SUBSTEPS):
            if not w.food or w.body.energy<=0 or w.body.integrity<=0:
                break
            step=w.micro_integrate(command)
            if use_memory:
                memory.rotate(w.body.turn*r12.MICRO_DT)
            trace.append(dict(step,macro=t,command=command,odor=odor,effective_odor=sensed,memory_active=used))
    gain=sum(f["food_gain"] for f in trace)
    cost=sum(f["metabolic_cost"] for f in trace)
    assert abs(60.-cost+gain-w.body.energy)<1e-7
    return dict(seed=seed,index=index,phase=phase,arm=arm,food=w.initial_food,rectangles=w.rectangles,
                encounter=not w.food,distance=sum(f["moved"] for f in trace),
                damage=100.-w.body.integrity,losses=losses,reacquisitions=reacquisitions,
                active=active,macros=len({f["macro"] for f in trace}),trace=trace)

def aggregate(rows):
    n=len(rows)
    return dict(episodes=n,encounter=sum(r["encounter"] for r in rows)/n,
                mean_damage=sum(r["damage"] for r in rows)/n,
                mean_distance=sum(r["distance"] for r in rows)/n,
                losses=sum(r["losses"] for r in rows),reacquisitions=sum(r["reacquisitions"] for r in rows),
                memory_fraction=sum(r["active"] for r in rows)/max(1,sum(r["macros"] for r in rows)))

def run():
    rows=[episode(s,offset+i,p,a) for s in SEEDS for p,offset in zip(PHASES,(100,200,300,300)) for i in range(20) for a in ARMS]
    summary={a:{p:aggregate([r for r in rows if r["arm"]==a and r["phase"]==p]) for p in PHASES} for a in ARMS}
    comparisons=(("memory_offset","memory","offset","original","offset",.05),
                 ("attenuation_midpoint","attenuated","midpoint","original","midpoint",.10),
                 ("retain_free","memory","free","original","free",-.05))
    gates={}
    intervals={}
    for name,a,p,c,q,threshold in comparisons:
        gates[name]=summary[a][p]["encounter"]-summary[c][q]["encounter"]>=threshold
        delta=[aggregate([r for r in rows if r["seed"]==s and r["arm"]==a and r["phase"]==p])["encounter"]-
               aggregate([r for r in rows if r["seed"]==s and r["arm"]==c and r["phase"]==q])["encounter"] for s in SEEDS]
        rng=random.Random(202020)
        values=sorted(sum(rng.choice(delta) for _ in delta)/len(delta) for _ in range(2000))
        intervals[name]=[values[50],values[1949]]
    masked={(r["seed"],r["index"],r["arm"]):r for r in rows if r["phase"]=="masked"}
    gates["masked_exact"]=all(masked[s,i,a]["trace"]==masked[s,i,c]["trace"] for s in SEEDS for i in range(300,320) for a,c in (("memory","original"),("both","attenuated")))
    return dict(summary=summary,gates=gates,passed=all(gates.values()),intervals=intervals,rows=rows)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",default="results/r17.json")
    args=p.parse_args()
    result=run()
    path=Path(args.out)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,sort_keys=True,separators=(",",":")))
    print(json.dumps({k:v for k,v in result.items() if k!="rows"},indent=2))
