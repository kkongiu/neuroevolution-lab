"""R19D: innate local stimulus competition; no learning."""
import argparse
import gzip
import json
import math
import random
from pathlib import Path
from neuroevolution import flyloop_r18a as r, flyloop_r1 as r1, flyloop_r12 as r12

SEEDS=range(25100,25124)
ARMS=("legacy","vector","affordance","masked","no_proximity")

def command(base,prox,light,hunger,eye,arm):
    if arm=="legacy":
        return r.visual_motor_adapter(base,prox,eye.light_channels(light),hunger,eye,True)
    if arm=="masked":
        light=(0.,)*len(light)
    if arm=="no_proximity":
        prox=(0.,)*len(prox)
    center=prox[len(prox)//2]
    peak=max(light)
    strength=r1.clamp(2*hunger*peak,0.,1.)
    if peak<=0.:
        return r.visual_motor_adapter(base,prox,eye.light_channels(light),hunger,eye,True)
    bearing=sum(v*a for a,v in zip(eye.centers,light))/sum(light)
    if arm=="vector":
        turn=math.tanh(4*strength*math.sin(bearing)-2*sum(p*math.sin(a) for p,a in zip(prox,eye.centers))/len(prox))
    else:
        scores=[strength*math.cos(a-bearing)-2*p for a,p in zip(eye.centers,prox)]
        weights=[math.exp(8*(s-max(scores))) for s in scores]
        turn=math.tanh(4*sum(a*w for a,w in zip(eye.centers,weights))/sum(weights))
    throttle=math.tanh(4*peak)*(1-.8*center)
    return ((1-strength)*base[0]+strength*throttle,
            (1-strength)*base[1]+strength*turn)

def episode(seed,index,scenario,arm,horizon=r.HORIZON):
    w=r.WindowWorld(seed,index,scenario)
    eye=r.CompoundEye()
    brain=r1.Circuit(seed,"recurrent")
    start=(w.body.x,w.body.y,w.body.heading)
    dx,dy=w.lamp[0]-start[0],w.lamp[1]-start[1]
    d=math.hypot(dx,dy)
    blocked=any(w._ray_box(start[0],start[1],dx/d,dy/d,x0-r1.RADIUS,y0-r1.RADIUS,x1+r1.RADIUS,y1+r1.RADIUS)<d
                for x0,y0,x1,y1 in w.rectangles)
    trace=[]; crossed=False
    for t in range(horizon):
        if not w.food or w.body.energy<=0 or w.body.integrity<=0:
            break
        prox,light=eye.observe(w,arm=="masked")
        original=w.perceive()
        sensed=(original[0],*eye.pooled(prox if arm!="no_proximity" else (0.,)*9),*original[4:])
        base,_,_=brain.forward(sensed)
        cmd=command(base,prox,light,1-w.body.energy/100.,eye,arm)
        for _ in range(r12.SUBSTEPS):
            if not w.food or w.body.energy<=0 or w.body.integrity<=0:
                break
            step=w.micro_integrate(cmd)
            crossed|=w.body.x>w.wall_x+.08+r1.RADIUS
            trace.append(dict(step,macro=t,command=cmd,proximity=prox,light=light))
    assert abs(60.-sum(f["metabolic_cost"] for f in trace)+sum(f["food_gain"] for f in trace)-w.body.energy)<1e-7
    return dict(seed=seed,index=index,scenario=scenario,arm=arm,start=start,lamp=w.lamp,
                rectangles=w.rectangles,opening=w.opening,initially_blocked=blocked,
                encounter=not w.food,crossed=crossed,contacts=sum(f["collision"] for f in trace),
                damage=100.-w.body.integrity,distance=sum(f["moved"] for f in trace),trace=trace)

def aggregate(rows):
    n=len(rows)
    return dict(episodes=n,encounter=sum(x["encounter"] for x in rows)/n if n else None,
                crossed=sum(x["crossed"] for x in rows)/n if n else None,
                mean_contacts=sum(x["contacts"] for x in rows)/n if n else None,
                mean_damage=sum(x["damage"] for x in rows)/n if n else None,
                mean_distance=sum(x["distance"] for x in rows)/n if n else None)

def shard(seed,out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    summary=[]
    for arm in ARMS:
        with (out/f"{seed}-{arm}.jsonl.gz").open("wb") as f:
            with gzip.GzipFile(fileobj=f,mode="wb",mtime=0,filename="") as z:
                for p in r.SCENARIOS:
                    for i in range(20):
                        ep=episode(seed,i,p,arm)
                        z.write((json.dumps(ep,sort_keys=True,separators=(",",":"))+"\n").encode())
                        summary.append({k:v for k,v in ep.items() if k!="trace"})
        print(seed,arm,flush=True)
    (out/f"{seed}-summary.json").write_text(json.dumps(summary,sort_keys=True,separators=(",",":")))

def combine(directory):
    rows=[]
    for s in SEEDS:
        paths=list(Path(directory).rglob(f"{s}-summary.json"))
        assert len(paths)==1
        part=json.loads(paths[0].read_text())
        assert len(part)==400 and {(x["seed"],x["arm"],x["scenario"],x["index"]) for x in part}=={(s,a,p,i) for a in ARMS for p in r.SCENARIOS for i in range(20)}
        rows.extend(part)
    def scope(a,p,s=None,blocked=False):
        return [x for x in rows if x["arm"]==a and x["scenario"]==p and (s is None or x["seed"]==s) and (not blocked or x["initially_blocked"])]
    summary={a:{p:aggregate(scope(a,p)) for p in r.SCENARIOS} for a in ARMS}
    blocked={a:aggregate(scope(a,"glass",blocked=True)) for a in ARMS}
    gates={}; intervals={}
    for name,p,a,b,only,threshold in (
        ("glass","glass","affordance","legacy",False,.05),
        ("blocked_glass","glass","affordance","legacy",True,.05),
        ("retain_free","free","affordance","legacy",False,-.05),
        ("light_dependence","glass","affordance","masked",False,0.)):
        aa=aggregate(scope(a,p,blocked=only))["encounter"]
        bb=aggregate(scope(b,p,blocked=only))["encounter"]
        gates[name]=aa is not None and bb is not None and aa-bb>=threshold
        delta=[]
        for s in SEEDS:
            x=aggregate(scope(a,p,s,only))["encounter"]
            y=aggregate(scope(b,p,s,only))["encounter"]
            if x is not None and y is not None:
                delta.append(x-y)
        rng=random.Random(251251)
        values=sorted(sum(rng.choice(delta) for _ in delta)/len(delta) for _ in range(2000)) if delta else []
        intervals[name]=[values[50],values[1949]] if values else None
    gates["contacts"]=summary["affordance"]["glass"]["mean_contacts"]<=summary["legacy"]["glass"]["mean_contacts"]
    gates["sealed"]=all(not x["encounter"] and not x["crossed"] for x in rows if x["scenario"]=="sealed")
    return dict(summary=summary,initially_blocked_glass=blocked,gates=gates,passed=all(gates.values()),intervals=intervals)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int); p.add_argument("--out"); p.add_argument("--combine")
    args=p.parse_args()
    if args.combine:
        print(json.dumps(combine(args.combine),indent=2))
    else:
        assert args.seed in SEEDS
        shard(args.seed,args.out)
