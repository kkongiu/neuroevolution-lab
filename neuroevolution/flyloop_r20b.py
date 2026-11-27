"""R20B: competing innate motivations; no training."""
import argparse
import gzip
import json
import math
import random
from pathlib import Path
from neuroevolution import flyloop_r20a as a, flyloop_r19d as d, flyloop_r18a as r, flyloop_r1 as r1, flyloop_r12 as r12

SEEDS=range(27100,27124)
ENERGIES=(30,60,90)
ARMS=("original","fixed","adaptive","clamped","persistent_adaptive","persistent_clamped")
HORIZON=200

def world(seed,index,energy):
    w=r.WindowWorld(seed,index,"free")
    rng=random.Random(seed*17311+index*7919+2020)
    w.body.x=w.body.y=11.
    w.body.heading=rng.uniform(-math.pi,math.pi)
    w.body.energy=float(energy)
    w.visited={(11,11)}
    angle=math.radians(40)*(1 if rng.random()<.5 else -1)
    def point(offset):
        theta=w.body.heading+offset
        return (11+3*math.cos(theta),11+3*math.sin(theta))
    w.lamp=point(angle)
    w.initial_food=(point(-angle),)
    w.food=list(w.initial_food)
    return w

def odor(w,eye,persistent):
    if not w.food and not persistent:
        return (0.,)*eye.sectors,(0.,)*eye.sectors
    # Free arena, same static distance field and antenna contrast as R20A.
    b=w.body
    source=w.initial_food[0]
    samples=tuple(max(0.,1-math.dist(source,(b.x+.25*math.cos(b.heading+x),
                  b.y+.25*math.sin(b.heading+x)))/8.) for x in eye.centers)
    lo,hi=min(samples),max(samples)
    return samples,tuple((v-lo)*hi/(hi-lo) for v in samples) if hi-lo>1e-9 else (0.,)*eye.sectors

def weights(arm,hunger,initial_hunger):
    if arm in ("original","fixed"):
        return 1.,1.
    h=initial_hunger if arm in ("clamped","persistent_clamped") else hunger
    scale=max(h,1-h)
    return (1-h)/scale,h/scale

def episode(seed,index,energy,arm,horizon=HORIZON):
    w=world(seed,index,energy)
    eye=r.CompoundEye()
    brain=r1.Circuit(seed,"recurrent")
    start=(w.body.x,w.body.y,w.body.heading)
    source=w.initial_food[0]
    trace=[]; first=None; meal=None; lamp_time=None; food_time=None
    persistent=arm.startswith("persistent")
    for t in range(horizon):
        if w.body.energy<=0 or w.body.integrity<=0:
            break
        prox,light=eye.observe(w)
        samples,smell=odor(w,eye,persistent)
        hunger=1-w.body.energy/100.
        wl,wo=weights(arm,hunger,1-energy/100.)
        cue=tuple(max(wl*l,wo*o) for l,o in zip(light,smell))
        original=w.perceive()
        base,_,_=brain.forward((original[0],*eye.pooled(prox),*original[4:]))
        cmd=d.command(base,prox,cue,hunger if arm=="original" else .6,eye,"affordance")
        for _ in range(r12.SUBSTEPS):
            if w.body.energy<=0 or w.body.integrity<=0:
                break
            step=w.micro_integrate(cmd)
            time=len(trace)+1
            pos=(w.body.x,w.body.y)
            if food_time is None and math.dist(pos,source)<=.55:
                food_time=time
                if first is None:
                    first="food"
            if lamp_time is None and math.dist(pos,w.lamp)<=.55:
                lamp_time=time
                if first is None:
                    first="lamp"
            if step["food_gain"]>0:
                assert meal is None
                meal=time
            trace.append(dict(step,macro=t,command=cmd,light=light,odor_samples=samples,
                              odor=smell,weights=(wl,wo),hunger=hunger))
    assert abs(energy-sum(f["metabolic_cost"] for f in trace)+sum(f["food_gain"] for f in trace)-w.body.energy)<1e-7
    return dict(seed=seed,index=index,energy=energy,arm=arm,start=start,lamp=w.lamp,food=source,
                first=first,food_time=food_time,lamp_time=lamp_time,meal=meal,
                post_meal_lamp=meal is not None and lamp_time is not None and lamp_time>meal,
                final_energy=w.body.energy,contacts=sum(f["collision"] for f in trace),trace=trace)

def aggregate(rows):
    n=len(rows)
    meals=[x for x in rows if x["meal"] is not None]
    eligible=[x for x in meals if x["lamp_time"] is None or x["lamp_time"]>x["meal"]]
    return dict(episodes=n,food_first=sum(x["first"]=="food" for x in rows)/n,
                lamp_first=sum(x["first"]=="lamp" for x in rows)/n,
                no_visit=sum(x["first"] is None for x in rows)/n,
                meal_fraction=len(meals)/n,post_meal_eligible=len(eligible),
                post_meal_lamp=sum(x["post_meal_lamp"] for x in eligible)/len(eligible) if eligible else None,
                mean_contacts=sum(x["contacts"] for x in rows)/n,
                mean_final_energy=sum(x["final_energy"] for x in rows)/n)

def shard(seed,out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    summary=[]
    for arm in ARMS:
        with (out/f"{seed}-{arm}.jsonl.gz").open("wb") as f:
            with gzip.GzipFile(fileobj=f,mode="wb",mtime=0,filename="") as z:
                for energy in ENERGIES:
                    for i in range(20):
                        ep=episode(seed,i,energy,arm)
                        # Compact actual trajectory: all macro endpoints and meal/arrival microsteps.
                        trace=ep.pop("trace")
                        ep["trace"]=[dict(step=j+1,position=v["position"],energy=v["energy"],
                                         collision=v["collision"],food_gain=v["food_gain"],
                                         weights=v["weights"],light=v["light"],odor=v["odor"])
                                     for j,v in enumerate(trace) if j%4==3 or j==len(trace)-1 or
                                     j+1 in (ep["meal"],ep["food_time"],ep["lamp_time"])]
                        z.write((json.dumps(ep,sort_keys=True,separators=(",",":"))+"\n").encode())
                        summary.append({k:v for k,v in ep.items() if k!="trace"})
        print(seed,arm,flush=True)
    (out/f"{seed}-summary.json").write_text(json.dumps(summary,sort_keys=True,separators=(",",":")))

def combine(directory):
    rows=[]
    for seed in SEEDS:
        paths=list(Path(directory).rglob(f"{seed}-summary.json"))
        assert len(paths)==1
        part=json.loads(paths[0].read_text())
        assert len(part)==360 and {(x["seed"],x["arm"],x["energy"],x["index"]) for x in part}=={(seed,a,e,i) for a in ARMS for e in ENERGIES for i in range(20)}
        rows.extend(part)
    def scope(arm,energy,seed=None):
        return [x for x in rows if x["arm"]==arm and x["energy"]==energy and (seed is None or x["seed"]==seed)]
    summary={arm:{str(e):aggregate(scope(arm,e)) for e in ENERGIES} for arm in ARMS}
    def bootstrap(delta):
        if not delta: return None
        rng=random.Random(271271)
        vals=sorted(sum(rng.choice(delta) for _ in delta)/len(delta) for _ in range(2000))
        return [vals[50],vals[1949]]
    gates={}; intervals={}
    for name,key,aa,ea,ab,eb,threshold in (
        ("hungry_food_first","food_first","adaptive",30,"adaptive",90,.20),
        ("sated_lamp_first","lamp_first","adaptive",90,"adaptive",30,.20),
        ("retain_food_first","food_first","adaptive",30,"fixed",30,-.05)):
        delta=[aggregate(scope(aa,ea,s))[key]-aggregate(scope(ab,eb,s))[key] for s in SEEDS]
        gates[name]=sum(delta)/len(delta)>=threshold
        intervals[name]=bootstrap(delta)
    by={(x["seed"],x["index"],x["energy"],x["arm"]):x for x in rows}
    pairs=[]
    for s in SEEDS:
        for i in range(20):
            for e in ENERGIES:
                x=by[s,i,e,"persistent_adaptive"]; y=by[s,i,e,"persistent_clamped"]
                if all(v["meal"] is not None and (v["lamp_time"] is None or v["lamp_time"]>v["meal"]) for v in (x,y)):
                    pairs.append((x,y))
    n=len(pairs)
    effect=sum(x["post_meal_lamp"]-y["post_meal_lamp"] for x,y in pairs)/n if n else None
    gates["post_meal_sample"]=n>=30
    gates["post_meal_switch"]=effect is not None and effect>=.05
    delta=[]
    for s in SEEDS:
        pp=[(x,y) for x,y in pairs if x["seed"]==s]
        if pp: delta.append(sum(x["post_meal_lamp"]-y["post_meal_lamp"] for x,y in pp)/len(pp))
    intervals["post_meal_switch_seed_equal"]=bootstrap(delta)
    return dict(summary=summary,gates=gates,passed=all(gates.values()),intervals=intervals,
                paired_persistent=dict(n=n,effect=effect,adaptive_switches=sum(x["post_meal_lamp"] for x,y in pairs),
                clamped_switches=sum(y["post_meal_lamp"] for x,y in pairs)))

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int); p.add_argument("--out"); p.add_argument("--combine")
    args=p.parse_args()
    if args.combine:
        print(json.dumps(combine(args.combine),indent=2))
    else:
        assert args.seed in SEEDS
        shard(args.seed,args.out)
