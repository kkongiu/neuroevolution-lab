"""R20D: unchanged innate mechanisms in a new environment."""
import argparse
import gzip
import json
import math
import random
from pathlib import Path
from neuroevolution import flyloop_r20b as b, flyloop_r20c as c, flyloop_r19d as d, flyloop_r18a as r, flyloop_r1 as r1, flyloop_r12 as r12

SEEDS=range(29100,29124)
ENERGIES=(60,90)
ARMS=("adaptive_hot","adaptive_cold","no_food","fixed_hot","clamped_hot")
HORIZON=400

def world(seed,index,energy,arm):
    w=r.WindowWorld(seed,index,"free")
    rng=random.Random(seed*17311+index*7919+2022)
    angle=rng.uniform(-math.pi,math.pi)
    w.lamp=(11.,11.)
    w.body.x=11.+1.37*math.cos(angle)
    w.body.y=11.+1.37*math.sin(angle)
    w.body.heading=r.wrap(angle+math.pi+rng.uniform(-.25,.25))
    theta=angle+math.radians(120)*(1 if rng.random()<.5 else -1)
    source=(11.+3.5*math.cos(theta),11.+3.5*math.sin(theta))
    w.initial_food=(source,)
    w.food=[] if arm=="no_food" else [source]
    w.body.energy=float(energy)
    w.visited={(int(w.body.x),int(w.body.y))}
    return w

def episode(seed,index,energy,arm,horizon=HORIZON):
    w=world(seed,index,energy,arm); eye=r.CompoundEye()
    brain=r1.Circuit(seed,"recurrent")
    amplitude=float(arm!="adaptive_cold")
    mode="fixed" if arm=="fixed_hot" else "clamped" if arm=="clamped_hot" else "adaptive"
    start=(w.body.x,w.body.y,w.body.heading)
    trace=[]; meal=None; departure=None; returned=None
    outside=inside=warm=0; damage=0.
    for t in range(horizon):
        if w.body.energy<=0 or w.body.integrity<=0: break
        prox,light=eye.observe(w)
        samples,odor=b.odor(w,eye,False)
        hunger=1-w.body.energy/100.
        wl,wo=b.weights(mode,hunger,1-energy/100.)
        cue=tuple(max(wl*l,wo*o) for l,o in zip(light,odor))
        original=w.perceive()
        base,_,_=brain.forward((original[0],*eye.pooled(prox),*original[4:]))
        baseline=d.command(base,prox,cue,.6,eye,"affordance")
        for _ in range(r12.SUBSTEPS):
            if w.body.energy<=0 or w.body.integrity<=0: break
            temp,sensors=c.thermal_sense(w,amplitude)
            cmd,urgency=c.protect(baseline,temp,sensors,"priority")
            step=w.micro_integrate(cmd)
            pos=(w.body.x,w.body.y)
            actual=c.heat(w,*pos,amplitude)
            loss=min(w.body.integrity,8*max(0.,actual-.65)*r12.MICRO_DT)
            w.body.integrity-=loss; damage+=loss
            n=len(trace)+1
            radius=math.dist(pos,w.lamp)
            in_warm=.875<=radius<=2.125
            warm+=in_warm
            outside=outside+1 if radius>2.125 else 0
            if departure is None and outside>=20:
                departure=dict(step=n,energy=w.body.energy)
            if step["food_gain"]>0:
                assert meal is None
                meal=dict(step=n,energy=w.body.energy)
            if meal is not None and n>meal["step"]:
                inside=inside+1 if in_warm else 0
                if returned is None and inside>=20:
                    returned=dict(step=n,energy=w.body.energy)
            trace.append(dict(step,step_index=n,heat=actual,integrity=w.body.integrity,
                thermal_damage=loss,urgency=urgency,weights=(wl,wo),command=cmd,radius=radius))
    assert abs(energy-sum(x["metabolic_cost"] for x in trace)+sum(x["food_gain"] for x in trace)-w.body.energy)<1e-7
    return dict(seed=seed,index=index,energy=energy,arm=arm,start=start,lamp=w.lamp,
        food=w.initial_food[0] if arm!="no_food" else None,departure=departure,meal=meal,
        returned=returned,steps=len(trace),thermal_damage=damage,
        warm_fraction=warm/max(1,len(trace)),survived=w.body.energy>0 and w.body.integrity>0,
        final_energy=w.body.energy,contacts=sum(x["collision"] for x in trace),
        trace=trace)

def aggregate(rows):
    n=len(rows)
    meals=[x for x in rows if x["meal"] is not None]
    eligible=[x for x in meals if x["steps"]-x["meal"]["step"]>=20 or x["returned"] is not None]
    departures=[x for x in rows if x["departure"] is not None]
    return dict(episodes=n,departure=len(departures)/n,ingestion=len(meals)/n,
        return_eligible=len(eligible),returned=sum(x["returned"] is not None for x in eligible)/len(eligible) if eligible else None,
        mean_departure_energy=sum(x["departure"]["energy"] for x in departures)/len(departures) if departures else None,
        mean_thermal_damage=sum(x["thermal_damage"] for x in rows)/n,
        warm_fraction=sum(x["warm_fraction"] for x in rows)/n,
        survival=sum(x["survived"] for x in rows)/n)

def shard(seed,out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True); summary=[]
    for arm in ARMS:
        with (out/f"{seed}-{arm}.jsonl.gz").open("wb") as f:
            with gzip.GzipFile(fileobj=f,mode="wb",mtime=0,filename="") as z:
                for energy in ENERGIES:
                    for i in range(20):
                        ep=episode(seed,i,energy,arm)
                        trace=ep.pop("trace")
                        events={ep[k]["step"] for k in ("departure","meal","returned") if ep[k] is not None}
                        ep["trace"]=[{k:v[k] for k in ("step_index","position","energy","heat","integrity","urgency","weights","food_gain")}
                            for j,v in enumerate(trace) if j%4==3 or j==len(trace)-1 or v["step_index"] in events or v["thermal_damage"]>0]
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
        assert len(part)==200 and {(x["seed"],x["arm"],x["energy"],x["index"]) for x in part}=={(s,a,e,i) for a in ARMS for e in ENERGIES for i in range(20)}
        rows.extend(part)
    def scope(a,e,s=None):
        return [x for x in rows if x["arm"]==a and x["energy"]==e and (s is None or x["seed"]==s)]
    summary={a:{str(e):aggregate(scope(a,e)) for e in ENERGIES} for a in ARMS}
    p=summary["adaptive_hot"]["90"]
    gates=dict(ingestion=p["ingestion"]>=.2,
        adaptive_gain=p["ingestion"]-summary["clamped_hot"]["90"]["ingestion"]>=.1,
        return_after_meal=p["returned"] is not None and p["returned"]>=.2,
        zero_damage=all(x["thermal_damage"]==0 for x in rows if x["arm"]=="adaptive_hot"))
    intervals={}
    for name,metric,control in (("departure_food_effect","departure","no_food"),
        ("ingestion_clamped","ingestion","clamped_hot"),("ingestion_fixed","ingestion","fixed_hot")):
        for e in ENERGIES:
            delta=[aggregate(scope("adaptive_hot",e,s))[metric]-aggregate(scope(control,e,s))[metric] for s in SEEDS]
            rng=random.Random(291291)
            values=sorted(sum(rng.choice(delta) for _ in delta)/len(delta) for _ in range(2000))
            intervals[f"{name}_{e}"]=[values[50],values[1949]]
    return dict(summary=summary,gates=gates,passed=all(gates.values()),intervals=intervals)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int); p.add_argument("--out"); p.add_argument("--combine")
    args=p.parse_args()
    if args.combine: print(json.dumps(combine(args.combine),indent=2))
    else:
        assert args.seed in SEEDS
        shard(args.seed,args.out)
