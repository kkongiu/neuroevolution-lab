"""Innate motivation preserved; learned steering restricted to sensory conflict."""
import argparse
import gzip
import json
import random
from pathlib import Path
from neuroevolution import flyloop_r18b as r, flyloop_r16 as r16
from neuroevolution import flyloop_r1 as r1, flyloop_r12 as r12, flyloop_r13 as r13, flyloop_r14b as b, flyloop_r18a as eye18

SEEDS=range(24100,24124)
MODES=("original","restricted","sham")

def eligible(obs):
    return obs[-14]>=.3 and max(obs[-9:])>=.05

def choose(c,e,h,obs,baseline,hunger,sham=False):
    if sham or not eligible(obs):
        return baseline,False
    state=r.state_bin(obs)
    ref=e.bounds(state,baseline)
    if ref is None:
        return baseline,False
    def score(cmd):
        _,i,g=c.predict(r13.features(h,obs,cmd))
        return -2*i+4*hunger*g
    value=score(baseline)
    options=[]
    for turn in (-1.,-.5,0.,.5,1.):
        cmd=(baseline[0],turn)
        if r16.motor_bin(cmd)==r16.motor_bin(baseline):
            continue
        bounds=e.bounds(state,cmd)
        if bounds is not None and bounds[0]>ref[1]+.1 and score(cmd)>value+.1:
            options.append(cmd)
    return (max(options,key=score),True) if options else (baseline,False)

def episode(seed,index,scenario,arm,c,e,training=False,sham=False,horizon=r.HORIZON):
    w=eye18.WindowWorld(seed,index,scenario)
    eye=eye18.CompoundEye()
    brain=r1.Circuit(seed,"recurrent")
    rng=random.Random(seed*17239+index*7919+13)
    exploration=[r12.GRID[rng.randrange(len(r12.GRID))] for _ in range(horizon)]
    records=[]
    trace=[]
    crossed=False
    interventions=opportunities=conflicts=conflict_interventions=0
    start=(w.body.x,w.body.y,w.body.heading)
    for t in range(horizon):
        if not w.food or w.body.energy<=0 or w.body.integrity<=0:
            break
        prox,light=eye.observe(w)
        original=w.perceive()
        base,h,_=brain.forward((original[0],*eye.pooled(prox),*original[4:]))
        hunger=1-w.body.energy/100.
        baseline=eye18.visual_motor_adapter(base,prox,eye.light_channels(light),hunger,eye,True)
        delivered=prox+light
        obs=w.perceive_extended()+delivered
        assert len(obs)==50
        conflict=eligible(obs)
        explore=training and t%4==0 and conflict
        cmd,override=( (baseline[0],exploration[t][1]),False) if explore else choose(c,e,h,obs,baseline,hunger,sham)
        assert cmd[0]==baseline[0] and (conflict or cmd==baseline)
        opportunities+=not explore
        interventions+=override
        conflicts+=conflict and not explore
        conflict_interventions+=conflict and override
        phi=r13.features(h,obs,cmd)
        distance=damage=gain=0.
        for _ in range(r12.SUBSTEPS):
            if not w.food or w.body.energy<=0 or w.body.integrity<=0:
                break
            before=w.body.integrity
            step=w.micro_integrate(cmd)
            distance+=step["moved"]
            damage+=before-w.body.integrity
            gain+=step["food_gain"]
            crossed|=w.body.x>w.wall_x+.08+r1.RADIUS
            trace.append(dict(step,macro=t,command=cmd,baseline=baseline,
                              override=override,explore=explore,conflict=conflict,
                              proximity=prox,light=light,delivered_vision=delivered))
        p=min(1.,distance/(r1.V_MAX*r12.MICRO_DT*r12.SUBSTEPS))
        injury=min(1.,damage/r1.COLLISION_INJURY)
        if training:
            b.update_head(c,0,phi,p)
            b.update_head(c,1,phi,injury)
            records.append((phi,gain,p,injury,hunger,r.state_bin(obs),cmd))
    if training:
        for (phi,g,p,i,hunger,state,cmd),target in zip(records,b.returns([v[1] for v in records])):
            b.update_head(c,2,phi,target)
            e.add(state,cmd,-2*i+4*hunger*target)
    cost=sum(f["metabolic_cost"] for f in trace)
    gain=sum(f["food_gain"] for f in trace)
    assert abs(60.-cost+gain-w.body.energy)<1e-7
    return dict(seed=seed,index=index,scenario=scenario,arm=arm,sham=sham,start=start,
                lamp=w.lamp,rectangles=w.rectangles,opening=w.opening,
                encounter=not w.food,crossed=crossed,contacts=sum(f["collision"] for f in trace),
                damage=100.-w.body.integrity,distance=sum(f["moved"] for f in trace),
                interventions=interventions,opportunities=opportunities,conflicts=conflicts,
                conflict_interventions=conflict_interventions,trace=trace)


def simulate(seed,model,train_n=60,test_n=20,horizon=r.HORIZON):
    if model=="original":
        row=r.simulate(seed,"full",train_n,test_n,horizon)
        return dict(seed=seed,model=model,training=row["training"],tests={"original":row["tests"]["gated"]})
    c=r.Critic(); e=r16.Evidence()
    train=[episode(seed,i,r.SCENARIOS[i%3],model,c,e,True,horizon=horizon) for i in range(train_n)]
    weights=[w[:] for w in c.weights]; evidence=dict(e.data)
    tests={m:{p:[episode(seed,100+i,p,model,c,e,False,m=="sham",horizon)
                  for i in range(test_n)] for p in r.SCENARIOS} for m in MODES[1:]}
    assert weights==c.weights and evidence==e.data
    return dict(seed=seed,model=model,training=train,tests=tests,weights=weights,evidence=sorted(evidence.items()))

def strip(row):
    def clean(ep):
        return {k:v for k,v in ep.items() if k!="trace"}
    return dict(seed=row["seed"],model=row["model"],training=[clean(ep) for ep in row["training"]],
                tests={m:{p:[clean(ep) for ep in eps] for p,eps in phases.items()} for m,phases in row["tests"].items()})

def shard(seed,out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    summaries=[]
    for model in ("original","restricted"):
        row=simulate(seed,model)
        with (out/f"{seed}-{model}.json.gz").open("wb") as f:
            with gzip.GzipFile(fileobj=f,mode="wb",mtime=0,filename="") as z:
                for chunk in json.JSONEncoder(sort_keys=True,separators=(",",":")).iterencode(row):
                    z.write(chunk.encode())
        summaries.append(strip(row)); del row
        print(seed,model,flush=True)
    (out/f"{seed}-summary.json").write_text(json.dumps(summaries,sort_keys=True,separators=(",",":")))

def combine(directory):
    rows=[]
    for seed in SEEDS:
        paths=list(Path(directory).rglob(f"{seed}-summary.json"))
        assert len(paths)==1
        pair=json.loads(paths[0].read_text())
        assert [v["model"] for v in pair]==["original","restricted"]
        for row in pair:
            assert row["seed"]==seed and len(row["training"])==60
            for phases in row["tests"].values():
                assert set(phases)==set(r.SCENARIOS) and all(len(v)==20 for v in phases.values())
        rows.extend(pair)
    def scope(mode,p,seed=None):
        return [ep for row in rows if mode in row["tests"] and (seed is None or row["seed"]==seed) for ep in row["tests"][mode][p]]
    summary={m:{p:r.aggregate(scope(m,p)) for p in r.SCENARIOS} for m in MODES}
    gates={}; intervals={}
    for name,p,m,threshold in (("glass_vs_original","glass","original",.05),("glass_vs_sham","glass","sham",.05),("retain_free","free","sham",0.)):
        delta=[r.aggregate(scope("restricted",p,s))["encounter"]-r.aggregate(scope(m,p,s))["encounter"] for s in SEEDS]
        gates[name]=sum(delta)/len(delta)>=threshold
        rng=random.Random(241241)
        values=sorted(sum(rng.choice(delta) for _ in delta)/len(delta) for _ in range(2000))
        intervals[name]=[values[50],values[1949]]
    gates["retain_free"]=all({k:v for k,v in a.items() if k not in ("sham","arm")}=={k:v for k,v in b.items() if k not in ("sham","arm")} for a,b in zip(scope("restricted","free"),scope("sham","free")))
    gates["contacts"]=summary["restricted"]["glass"]["mean_contacts"]<=summary["sham"]["glass"]["mean_contacts"]
    gates["conflict_interventions"]=summary["restricted"]["glass"]["conflict_intervention_fraction"]>=.01
    gates["sealed_zero"]=all(not ep["crossed"] and not ep["encounter"] for m in MODES for ep in scope(m,"sealed"))
    gates["restriction_invariants"]=True # asserted for every restricted macro before artifact creation
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
