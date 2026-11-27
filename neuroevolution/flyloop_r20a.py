"""R20A: static odor field and local multisensory taxis, no learning."""
import argparse
import gzip
import json
import math
import random
from pathlib import Path
from neuroevolution import flyloop_r19d as d, flyloop_r18a as r, flyloop_r1 as r1, flyloop_r12 as r12

SEEDS=range(26100,26124)
ARMS=("light_only","odor_only","combined_occluded","combined_routed","no_proximity")

def clear(w,a,b):
    dx,dy=b[0]-a[0],b[1]-a[1]
    length=math.hypot(dx,dy)
    if length<1e-12:
        return True
    return not any(w._ray_box(a[0],a[1],dx/length,dy/length,*rect)<length-1e-9 for rect in w.rectangles)

def odor_at(w,x,y,routed):
    if any(x0<=x<=x1 and y0<=y<=y1 for x0,y0,x1,y1 in w.rectangles):
        return 0.
    a=(x,y); b=w.lamp
    length=math.dist(a,b)
    if not clear(w,a,b):
        if not routed or len(w.rectangles)!=2:
            return 0.
        # The two wall panels form one opening. Environment-only path length.
        lo,hi=w.opening
        left=w.wall_x-.08-1e-6; right=w.wall_x+.08+1e-6
        # Visibility graph: direct and corner-to-corner segments, shortest valid polyline.
        nodes=[a,b]+[(u,v) for u in (left,right) for v in (lo+1e-6,hi-1e-6)]
        dist=[math.inf]*len(nodes); dist[0]=0.
        remaining=set(range(len(nodes)))
        while remaining:
            i=min(remaining,key=lambda j:(dist[j],j))
            remaining.remove(i)
            if i==1 or not math.isfinite(dist[i]):
                break
            for j in remaining:
                if clear(w,nodes[i],nodes[j]):
                    dist[j]=min(dist[j],dist[i]+math.dist(nodes[i],nodes[j]))
        length=dist[1]
    return max(0.,1-length/8.)

def olfaction(w,eye,routed):
    b=w.body
    samples=tuple(odor_at(w,b.x+.25*math.cos(b.heading+a),b.y+.25*math.sin(b.heading+a),routed)
                  for a in eye.centers)
    lo,hi=min(samples),max(samples)
    odor=tuple((v-lo)*hi/(hi-lo) for v in samples) if hi-lo>1e-9 else (0.,)*len(samples)
    return samples,odor

def combine_cues(light,odor,arm):
    if arm=="light_only":
        return light
    if arm=="odor_only":
        return odor
    return tuple(max(l,o) for l,o in zip(light,odor))

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
        prox,light=eye.observe(w)
        samples,odor=olfaction(w,eye,arm!="combined_occluded")
        cue=combine_cues(light,odor,arm)
        original=w.perceive()
        sensed=(original[0],*eye.pooled(prox if arm!="no_proximity" else (0.,)*9),*original[4:])
        base,_,_=brain.forward(sensed)
        cmd=d.command(base,prox,cue,1-w.body.energy/100.,eye,"no_proximity" if arm=="no_proximity" else "affordance")
        for _ in range(r12.SUBSTEPS):
            if not w.food or w.body.energy<=0 or w.body.integrity<=0:
                break
            step=w.micro_integrate(cmd)
            crossed|=w.body.x>w.wall_x+.08+r1.RADIUS
            trace.append(dict(step,macro=t,command=cmd,proximity=prox,light=light,odor_samples=samples,odor=odor,cue=cue))
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
    def scope(a,p,s=None):
        return [x for x in rows if x["arm"]==a and x["scenario"]==p and (s is None or x["seed"]==s)]
    summary={a:{p:aggregate(scope(a,p)) for p in r.SCENARIOS} for a in ARMS}
    gates={}; intervals={}
    for name,p,b,threshold in (
        ("glass","glass","light_only",.05),
        ("retain_free","free","light_only",-.05),
        ("multisensory","glass","odor_only",.05),
        ("opaque","opaque","light_only",.05),
        ("routed_vs_occluded","glass","combined_occluded",None)):
        delta=[aggregate(scope("combined_routed",p,s))["encounter"]-aggregate(scope(b,p,s))["encounter"] for s in SEEDS]
        if threshold is not None:
            gates[name]=sum(delta)/len(delta)>=threshold
        rng=random.Random(261261)
        values=sorted(sum(rng.choice(delta) for _ in delta)/len(delta) for _ in range(2000))
        intervals[name]=[values[50],values[1949]]
    gates["contacts"]=summary["combined_routed"]["glass"]["mean_contacts"]<=summary["light_only"]["glass"]["mean_contacts"]
    gates["sealed"]=all(not x["encounter"] and not x["crossed"] for x in rows if x["scenario"]=="sealed")
    return dict(summary=summary,gates=gates,passed=all(gates.values()),intervals=intervals)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int); p.add_argument("--out"); p.add_argument("--combine")
    args=p.parse_args()
    if args.combine:
        print(json.dumps(combine(args.combine),indent=2))
    else:
        assert args.seed in SEEDS
        shard(args.seed,args.out)
