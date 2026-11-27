"""R21B: factorial thermal physiology; no new cognitive circuitry."""
import argparse, gzip, json, math, random
from pathlib import Path
from neuroevolution.flyloop_r21a import world, sense, predator_command, move_predator, flight, visible, DT
from neuroevolution import flyloop_r1 as r1, flyloop_r12 as r12, flyloop_r18a as r, flyloop_r19d as d, flyloop_r20b as b, flyloop_r20c as c
SEEDS=range(31100,31124)
MODELS=('immediate','damage_only','reflex_only','inertia','fast_cooling')
ARMS=tuple(m+':'+p for m in MODELS for p in ('absent','escape'))
def thermal_update(body,ambient,model):
    tau=1. if model=='fast_cooling' and ambient<body else 3.
    return ambient+(body-ambient)*math.exp(-DT/tau)

def episode(seed,index,scenario,arm,horizon=200):
    model,presence=arm.rsplit(':',1)
    arm=presence
    w,p=world(seed,index,scenario); eye=r.CompoundEye()
    brain=r1.Circuit(seed,"recurrent"); hunter=r1.Circuit(seed+100000,"recurrent")
    start=(w.body.x,w.body.y,w.body.heading); predator_start=(p.x,p.y,p.heading)
    body_heat=c.heat(w,w.body.x,w.body.y,float(scenario=='lantern'))
    peak_body=body_heat; exposure=0
    trace=[]; captured=False; contact=0; thermal=0.; escape_steps=0; seen=0; food_gain=0.; min_distance=2.5; predator_contacts=0
    for t in range(horizon):
        if captured or w.body.energy<=0 or w.body.integrity<=0:break
        prox,light=eye.observe(w,scenario!="lantern")
        _,odor=b.odor(w,eye,False)
        wl,wo=b.weights("adaptive",1-w.body.energy/100.,.1)
        cue=tuple(max(wl*l,wo*o) for l,o in zip(light,odor))
        obs=w.perceive()
        base,_,_=brain.forward((obs[0],*eye.pooled(prox),*obs[4:]))
        ordinary=d.command(base,prox,cue,.6,eye,"affordance")
        signal=sense(p,w.body,w.rectangles,4.5) if arm!="absent" and p.energy>0 else (0.,)*12
        pc=predator_command(p,hunter,signal,w.rectangles) if arm!="absent" and p.energy>0 else (0.,0.)
        for _ in range(r12.SUBSTEPS):
            if captured or w.body.energy<=0 or w.body.integrity<=0:break
            active=arm!="absent" and p.energy>0
            threat=sense(w.body,p,w.rectangles,3.5,arm!="unoccluded") if active else (0.,)*12
            seen+=bool(max(threat))
            cmd,u=flight(ordinary,threat if arm in ("escape","unoccluded") else (0.,)*12,prox,eye)
            escape_steps+=u>0
            amplitude=float(scenario=="lantern")
            temp,sensors=c.thermal_sense(w,amplitude)
            before=cmd
            reflex_temp=body_heat if model in ('reflex_only','inertia','fast_cooling') else temp
            cmd,urgency=c.protect(cmd,reflex_temp,sensors,'priority')
            step=w.micro_integrate(cmd)
            if active:
                move_predator(p,pc,w.rectangles); predator_contacts+=p.prev_collision
            actual=c.heat(w,w.body.x,w.body.y,amplitude)
            body_heat=thermal_update(body_heat,actual,model)
            peak_body=max(peak_body,body_heat)
            exposure+=actual>.65
            damage_temp=body_heat if model in ('damage_only','inertia','fast_cooling') else actual
            loss=min(w.body.integrity,8*max(0.,damage_temp-.65)*DT)
            w.body.integrity-=loss;thermal+=loss;food_gain+=step["food_gain"]
            q=(w.body.x,w.body.y); pp=(p.x,p.y)
            distance=math.dist(q,pp)
            if active:
                min_distance=min(min_distance,distance)
                contact=contact+1 if distance<=.75 and visible(q,pp,w.rectangles) else 0
                if contact>=4:
                    captured=True;p.energy=min(100.,p.energy+12)
            else:contact=0
            trace.append(dict(step,predator=list(pp) if arm!="absent" else None,threat=threat,escape=u,heat=actual,integrity=w.body.integrity,captured=captured,body_heat=body_heat,urgency=urgency,before_protection=before,command=cmd,thermal_loss=loss))
    assert abs(90-sum(x["metabolic_cost"] for x in trace)+food_gain-w.body.energy)<1e-7
    return dict(seed=seed,index=index,scenario=scenario,arm=model+':'+arm,start=start,predator_start=predator_start,lamp=w.lamp if scenario=="lantern" else None,food=w.initial_food,rectangles=w.rectangles,captured=captured,survived=not captured and w.body.energy>0 and w.body.integrity>0,steps=len(trace),capture_seconds=len(trace)*DT if captured else None,food_gain=food_gain,ate=food_gain>0,thermal_damage=thermal,peak_body=peak_body,final_body=body_heat,exposure_seconds=exposure*DT,escape_fraction=escape_steps/max(1,len(trace)),seen_fraction=seen/max(1,len(trace)),min_distance=min_distance if arm!="absent" else None,contacts=sum(x["collision"] for x in trace),predator_contacts=predator_contacts,trace=trace)

def aggregate(rows):
    return dict(episodes=len(rows),**{k:sum(x[k] for x in rows)/len(rows) for k in ('captured','survived','ate','thermal_damage','peak_body','final_body','exposure_seconds','contacts','escape_fraction')})

def shard(seed,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);summary=[]
    for arm in ARMS:
        with (out/f"{seed}-{arm.replace(':','-')}.jsonl.gz").open("wb") as f:
            with gzip.GzipFile(fileobj=f,mode="wb",mtime=0,filename="") as z:
                for i in range(20):
                    ep=episode(seed,i,"lantern",arm);trace=ep.pop("trace")
                    ep["trace"]=[dict(step=j+1,**{k:v[k] for k in ("position","predator","energy","integrity","escape","captured","food_gain","heat","body_heat","urgency","before_protection","command","thermal_loss")}) for j,v in enumerate(trace) if j%4==3 or j==len(trace)-1 or v["food_gain"]>0 or v["thermal_loss"]>0]
                    z.write((json.dumps(ep,sort_keys=True,separators=(",",":"))+"\n").encode())
                    summary.append({k:v for k,v in ep.items() if k!="trace"})
        print(seed,arm,flush=True)
    (out/f"{seed}-summary.json").write_text(json.dumps(summary,sort_keys=True,separators=(",",":")))

def combine(directory):
    rows=[]
    for s in SEEDS:
        paths=list(Path(directory).rglob(f"{s}-summary.json"));assert len(paths)==1
        part=json.loads(paths[0].read_text())
        assert len(part)==200 and {(x["seed"],x["index"],x["arm"]) for x in part}=={(s,i,a) for i in range(20) for a in ARMS}
        rows.extend(part)
    def scope(a,s=None):return [x for x in rows if x["arm"]==a and (s is None or x["seed"]==s)]
    summary={a:aggregate(scope(a)) for a in ARMS}
    p=summary["inertia:escape"];q=summary["immediate:escape"];n=summary["inertia:absent"];base=summary["immediate:absent"]
    gates=dict(survival=p["survived"]-q["survived"]>=.2,injury=p["thermal_damage"]<=5,food=base["ate"]>0 and n["ate"]>=.8*base["ate"],absent_injury=n["thermal_damage"]<=5,absent_zero=all(not x["captured"] for x in rows if x["arm"].endswith(":absent")))
    intervals={}
    for a in ARMS:
        for metric in ("survived","thermal_damage"):
            control="immediate:"+a.split(":")[1]
            delta=[aggregate(scope(a,s))[metric]-aggregate(scope(control,s))[metric] for s in SEEDS]
            rng=random.Random(311311);v=sorted(sum(rng.choice(delta) for _ in delta)/len(delta) for _ in range(2000))
            intervals[a+":"+metric]=[v[50],v[1949]]
    return dict(summary=summary,gates=gates,passed=all(gates.values()),intervals=intervals)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--seed",type=int);p.add_argument("--out");p.add_argument("--combine");args=p.parse_args()
    if args.combine:print(json.dumps(combine(args.combine),indent=2))
    else:
        assert args.seed in SEEDS
        shard(args.seed,args.out)
