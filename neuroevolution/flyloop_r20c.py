"""R20C — innate thermal protection, no learning."""
import argparse
import gzip
import json
import math
import random
from pathlib import Path
from neuroevolution import flyloop_r18a as r, flyloop_r19d as d, flyloop_r1 as r1, flyloop_r12 as r12

SEEDS=range(28100,28124)
ARMS=("baseline","additive","priority")
SCENARIOS=("cold","hot")
HORIZON=200
ANGLES=tuple(i*math.pi/4 for i in range(8))

def world(seed,index):
    w=r.WindowWorld(seed,index,"free")
    rng=random.Random(seed*17311+index*7919+2021)
    w.body.x=w.body.y=11.
    w.body.heading=rng.uniform(-math.pi,math.pi)
    theta=w.body.heading+rng.uniform(-math.radians(40),math.radians(40))
    w.lamp=(11+3*math.cos(theta),11+3*math.sin(theta))
    w.body.energy=90.
    w.food=[]; w.initial_food=()
    w.visited={(11,11)}
    return w

def heat(w,x,y,amplitude):
    return amplitude*max(0.,1-math.dist((x,y),w.lamp)/2.5)

def thermal_sense(w,amplitude):
    b=w.body
    return heat(w,b.x,b.y,amplitude),tuple(heat(w,b.x+.25*math.cos(b.heading+a),b.y+.25*math.sin(b.heading+a),amplitude) for a in ANGLES)

def protect(baseline,temperature,samples,arm):
    urgency=r1.clamp((temperature-.35)/.25,0.,1.)
    if arm=="baseline" or urgency==0.:
        return baseline,0.
    x=sum(v*math.cos(a) for a,v in zip(ANGLES,samples))
    y=sum(v*math.sin(a) for a,v in zip(ANGLES,samples))
    mag=math.hypot(x,y)
    # Cooler direction in body coordinates; reverse if source is ahead.
    escape=(-x/mag, -y/mag) if mag>1e-12 else (-1.,0.)
    if arm=="priority":
        cmd=tuple((1-urgency)*b+urgency*e for b,e in zip(baseline,escape))
    else:
        cmd=tuple(r1.clamp(b+urgency*e,-1.,1.) for b,e in zip(baseline,escape))
    return cmd,urgency

def episode(seed,index,scenario,arm,horizon=HORIZON):
    w=world(seed,index); eye=r.CompoundEye()
    brain=r1.Circuit(seed,"recurrent")
    amplitude=float(scenario=="hot")
    start=(w.body.x,w.body.y,w.body.heading)
    trace=[]; thermal_damage=0.; exposure=warm=slow=0
    min_distance=math.dist(start[:2],w.lamp)
    radial_anchor=min_distance; radial_direction=0; reversals=0
    for t in range(horizon):
        if w.body.energy<=0 or w.body.integrity<=0: break
        prox,light=eye.observe(w)
        original=w.perceive()
        base,_,_=brain.forward((original[0],*eye.pooled(prox),*original[4:]))
        baseline=d.command(base,prox,light,.6,eye,"affordance")
        # Safety pathway samples at physical microstep rate, baseline remains macro-rate.
        for _ in range(r12.SUBSTEPS):
            if w.body.energy<=0 or w.body.integrity<=0: break
            temp,samples=thermal_sense(w,amplitude)
            cmd,urgency=protect(baseline,temp,samples,arm)
            step=w.micro_integrate(cmd)
            pos=(w.body.x,w.body.y)
            actual=heat(w,*pos,amplitude)
            loss=min(w.body.integrity,8*max(0.,actual-.65)*r12.MICRO_DT)
            w.body.integrity-=loss
            thermal_damage+=loss
            exposure+=actual>.65
            warm+=.15<=actual<=.65
            slow+=abs(w.body.speed)<.05
            distance=math.dist(pos,w.lamp)
            min_distance=min(min_distance,distance)
            if abs(distance-radial_anchor)>=.1:
                direction=1 if distance>radial_anchor else -1
                reversals+=radial_direction!=0 and direction!=radial_direction
                radial_direction=direction; radial_anchor=distance
            trace.append(dict(step,macro=t,command=cmd,baseline=baseline,
                heat=actual,heat_before=temp,thermal_samples=samples,urgency=urgency,
                thermal_damage=loss,integrity=w.body.integrity,radial_distance=distance))
    assert abs(90.-sum(f["metabolic_cost"] for f in trace)-w.body.energy)<1e-7
    assert abs(100.-w.body.integrity-thermal_damage-sum(f["collision"] for f in trace)*r1.COLLISION_INJURY/r12.SUBSTEPS)<1e-7 or w.body.integrity==0.
    n=len(trace)
    return dict(seed=seed,index=index,scenario=scenario,arm=arm,start=start,lamp=w.lamp,
        thermal_damage=thermal_damage,exposure_seconds=exposure*r12.MICRO_DT,
        warm_fraction=warm/max(n,1),slow_fraction=slow/max(n,1),
        min_distance=min_distance,final_distance=math.dist((w.body.x,w.body.y),w.lamp),
        visited_light=min_distance<=.55,radial_reversals=reversals,
        distance=sum(f["moved"] for f in trace),contacts=sum(f["collision"] for f in trace),
        survived=w.body.energy>0 and w.body.integrity>0,steps=n,
        final_energy=w.body.energy,final_integrity=w.body.integrity,trace=trace)

def aggregate(rows):
    n=len(rows)
    return dict(episodes=n,**{k:sum(x[k] for x in rows)/n for k in (
        "thermal_damage","exposure_seconds","warm_fraction","slow_fraction","min_distance",
        "final_distance","visited_light","radial_reversals","distance","contacts","survived","final_integrity")})

def shard(seed,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    summary=[]
    for arm in ARMS:
        with (out/f"{seed}-{arm}.jsonl.gz").open("wb") as f:
            with gzip.GzipFile(fileobj=f,mode="wb",mtime=0,filename="") as z:
                for scenario in SCENARIOS:
                    for i in range(20):
                        ep=episode(seed,i,scenario,arm)
                        trace=ep.pop("trace")
                        # Keep real macro endpoints, all damaging samples and terminal point.
                        ep["trace"]=[dict(step=j+1,position=v["position"],heat=v["heat"],
                            integrity=v["integrity"],command=v["command"],urgency=v["urgency"],
                            thermal_damage=v["thermal_damage"]) for j,v in enumerate(trace)
                            if j%4==3 or v["thermal_damage"]>0 or j==len(trace)-1]
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
        assert len(part)==120 and {(x["seed"],x["index"],x["scenario"],x["arm"]) for x in part}=={(s,i,p,a) for i in range(20) for p in SCENARIOS for a in ARMS}
        rows.extend(part)
    def scope(a,p,s=None):
        return [x for x in rows if x["arm"]==a and x["scenario"]==p and (s is None or x["seed"]==s)]
    summary={a:{p:aggregate(scope(a,p)) for p in SCENARIOS} for a in ARMS}
    p=summary["priority"]["hot"]; b=summary["baseline"]["hot"]; a=summary["additive"]["hot"]
    def clean(x):return {k:v for k,v in x.items() if k!="arm"}
    gates=dict(damage_reduction=b["thermal_damage"]>0 and p["thermal_damage"]<=.5*b["thermal_damage"],
        priority_vs_additive=p["thermal_damage"]<=a["thermal_damage"],
        cold_invariance=all(list(map(clean,scope(x,"cold")))==list(map(clean,scope("baseline","cold"))) for x in ARMS),
        retain_movement=p["distance"]>=.5*b["distance"],
        warm_occupancy=p["warm_fraction"]>=.25,
        survival=p["survived"]>=b["survived"])
    intervals={}
    for metric in ("thermal_damage","warm_fraction","slow_fraction","distance"):
        for control in ("baseline","additive"):
            delta=[aggregate(scope("priority","hot",s))[metric]-aggregate(scope(control,"hot",s))[metric] for s in SEEDS]
            rng=random.Random(281281)
            v=sorted(sum(rng.choice(delta) for _ in delta)/len(delta) for _ in range(2000))
            intervals[metric+"_vs_"+control]=[v[50],v[1949]]
    return dict(summary=summary,gates=gates,passed=all(gates.values()),intervals=intervals)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int);p.add_argument("--out");p.add_argument("--combine")
    args=p.parse_args()
    if args.combine: print(json.dumps(combine(args.combine),indent=2))
    else:
        assert args.seed in SEEDS
        shard(args.seed,args.out)
