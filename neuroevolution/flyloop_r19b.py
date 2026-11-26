"""Temporal action options on unchanged R18B body and sensory system."""
import argparse
import gzip
import json
import math
import random
from pathlib import Path
from neuroevolution import flyloop_r18b as r, flyloop_r16 as r16
from neuroevolution import flyloop_r1 as r1, flyloop_r12 as r12, flyloop_r13 as r13, flyloop_r14b as b, flyloop_r18a as eye18

SEEDS=range(23100,23124)
DURATIONS=(1,4,8)
OPTIONS=tuple((cmd,d) for d in DURATIONS for cmd in (None,)+r12.GRID)
MODES=("single","temporal","no_interrupt","sham")

class Options:
    def __init__(self):
        self.data={}
    def add(self,state,option,value):
        key=(state,option)
        n,s,q=self.data.get(key,(0,0.,0.))
        self.data[key]=(n+1,s+value,q+value*value)
    def bounds(self,state,option):
        n,s,q=self.data.get((state,option),(0,0.,0.))
        if n<8:
            return None
        se=math.sqrt(max(0.,(q-s*s/n)/(n-1))/n)
        return s/n-2*se,s/n+2*se
    def choose(self,state):
        eligible=[]
        for cmd,d in OPTIONS:
            if cmd is None:
                continue
            ref=self.bounds(state,(None,d))
            alt=self.bounds(state,(cmd,d))
            if ref is not None and alt is not None and alt[0]>ref[1]+.1:
                eligible.append((alt[0]-ref[1],cmd,d))
        if not eligible:
            return None
        _,cmd,d=max(eligible)
        return cmd,d
    def abort(self,state,cmd,remaining):
        d=max(v for v in DURATIONS if v<=remaining)
        ref=self.bounds(state,(None,d))
        alt=self.bounds(state,(cmd,d))
        return ref is not None and alt is not None and ref[0]>alt[1]+.1

def option_return(rewards,planned):
    return sum(.9**i*v for i,v in enumerate(rewards))/sum(.9**i for i in range(planned))

def episode(seed,index,scenario,c,e,opt,training=False,mode="temporal",horizon=r.HORIZON):
    w=eye18.WindowWorld(seed,index,scenario)
    eye=eye18.CompoundEye()
    brain=r1.Circuit(seed,"recurrent")
    rng=random.Random(seed*17239+index*7919+13)
    records=[]; trace=[]; options=[]
    active=None
    decisions=held=aborts=starts=interventions=opportunities=conflicts=conflict_interventions=0
    crossed=False
    previous_injury=0.
    start=(w.body.x,w.body.y,w.body.heading)
    def finish(reason):
        nonlocal active
        if active is not None:
            state,option,rewards,started,explore=active
            value=option_return(rewards,option[1])
            if training and rewards:
                opt.add(state,option,value)
            options.append(dict(start=started,option=option,elapsed=len(rewards),reason=reason,
                                explore=explore,value=value))
            active=None
    for t in range(horizon):
        if not w.food or w.body.energy<=0 or w.body.integrity<=0:
            break
        prox,light=eye.observe(w)
        original=w.perceive()
        base,h,_=brain.forward((original[0],*eye.pooled(prox),*original[4:]))
        hunger=1-w.body.energy/100.
        baseline=eye18.visual_motor_adapter(base,prox,eye.light_channels(light),hunger,eye,True)
        obs=w.perceive_extended()+prox+light
        state=r.state_bin(obs)
        if active is not None:
            _,(command,duration),rewards,_,_=active
            remaining=duration-len(rewards)
            if remaining==0:
                finish("complete")
            elif mode!="no_interrupt" and command is not None and not active[4]:
                if previous_injury>0 or opt.abort(state,command,remaining):
                    aborts+=1
                    finish("injury" if previous_injury>0 else "learned")
        if active is None:
            explore=training and decisions%4==0
            decisions+=1
            selected=OPTIONS[rng.randrange(len(OPTIONS))] if explore else opt.choose(state) if mode!="sham" else None
            if selected is not None:
                active=(state,selected,[],t,explore)
                starts+=1
        else:
            explore=active[4]
        if active is None:
            cmd,override=r.choose(c,e,h,obs,baseline,hunger,"sham" if mode=="sham" else "gated")
        else:
            _,(command,duration),rewards,_,explore=active
            if rewards and command is not None:
                held+=1
            cmd=baseline if command is None else command
            override=command is not None and cmd!=baseline and not explore
        conflict=prox[4]>=.3 and max(light)>=.3
        opportunities+=not explore; interventions+=override
        conflicts+=conflict and not explore; conflict_interventions+=conflict and override
        phi=r13.features(h,obs,cmd)
        distance=damage=gain=0.
        for _ in range(r12.SUBSTEPS):
            if not w.food or w.body.energy<=0 or w.body.integrity<=0:
                break
            before=w.body.integrity
            step=w.micro_integrate(cmd)
            distance+=step["moved"]; damage+=before-w.body.integrity; gain+=step["food_gain"]
            crossed|=w.body.x>w.wall_x+.08+r1.RADIUS
            trace.append(dict(step,macro=t,command=cmd,baseline=baseline,override=override,
                              explore=explore,conflict=conflict,proximity=prox,light=light,
                              option=active[1] if active else None,option_start=active[3] if active else None))
        p=min(1.,distance/(r1.V_MAX*r12.MICRO_DT*r12.SUBSTEPS))
        injury=min(1.,damage/r1.COLLISION_INJURY)
        previous_injury=damage
        if active is not None:
            active[2].append(p-2*injury+4*hunger*gain/12.)
        if training:
            b.update_head(c,0,phi,p); b.update_head(c,1,phi,injury)
            records.append((phi,gain,p,injury,hunger,state,cmd))
    finish("terminal")
    if training:
        for (phi,g,p,i,hunger,state,cmd),target in zip(records,b.returns([v[1] for v in records])):
            b.update_head(c,2,phi,target)
            e.add(state,cmd,p-2*i+4*hunger*target)
    assert abs(60.-sum(f["metabolic_cost"] for f in trace)+sum(f["food_gain"] for f in trace)-w.body.energy)<1e-7
    return dict(seed=seed,index=index,scenario=scenario,mode=mode,start=start,lamp=w.lamp,
                rectangles=w.rectangles,opening=w.opening,encounter=not w.food,crossed=crossed,
                contacts=sum(f["collision"] for f in trace),damage=100.-w.body.integrity,
                distance=sum(f["moved"] for f in trace),interventions=interventions,opportunities=opportunities,
                conflicts=conflicts,conflict_interventions=conflict_interventions,held=held,starts=starts,
                aborts=aborts,macros=len(set(f["macro"] for f in trace)),options=options,trace=trace)

def simulate(seed,model,train_n=60,test_n=20,horizon=r.HORIZON):
    if model=="single":
        row=r.simulate(seed,"full",train_n,test_n,horizon)
        return dict(seed=seed,model=model,training=row["training"],tests={"single":row["tests"]["gated"]})
    c=r.Critic(); e=r16.Evidence(); opt=Options()
    train=[episode(seed,i,r.SCENARIOS[i%3],c,e,opt,True,horizon=horizon) for i in range(train_n)]
    frozen=([w[:] for w in c.weights],dict(e.data),dict(opt.data))
    tests={m:{p:[episode(seed,100+i,p,c,e,opt,False,m,horizon) for i in range(test_n)]
              for p in r.SCENARIOS} for m in MODES[1:]}
    assert frozen==(c.weights,e.data,opt.data)
    return dict(seed=seed,model=model,training=train,tests=tests,weights=c.weights,
                evidence=sorted(e.data.items()),option_evidence=sorted(opt.data.items(),key=lambda v:repr(v[0])))

def strip(row):
    def clean(ep):
        return {k:v for k,v in ep.items() if k not in ("trace","options")}
    return dict(seed=row["seed"],model=row["model"],training=[clean(ep) for ep in row["training"]],
                tests={m:{p:[clean(ep) for ep in eps] for p,eps in phases.items()} for m,phases in row["tests"].items()})

def dump(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"))

def shard(seed,out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    summaries=[]
    for model in ("single","temporal"):
        row=simulate(seed,model)
        with (out/f"{seed}-{model}.json.gz").open("wb") as f:
            with gzip.GzipFile(fileobj=f,mode="wb",mtime=0,filename="") as z:
                for chunk in json.JSONEncoder(sort_keys=True,separators=(",",":")).iterencode(row):
                    z.write(chunk.encode())
        summaries.append(strip(row))
        del row
        print(seed,model,flush=True)
    (out/f"{seed}-summary.json").write_text(dump(summaries))

def combine(directory):
    rows=[]
    for seed in SEEDS:
        paths=list(Path(directory).rglob(f"{seed}-summary.json"))
        assert len(paths)==1
        pair=json.loads(paths[0].read_text())
        assert [v["model"] for v in pair]==["single","temporal"]
        for row in pair:
            assert row["seed"]==seed and len(row["training"])==60
            for phases in row["tests"].values():
                assert set(phases)==set(r.SCENARIOS) and all(len(v)==20 for v in phases.values())
        rows.extend(pair)
    def scope(mode,p,seed=None):
        return [ep for row in rows if mode in row["tests"] and (seed is None or row["seed"]==seed) for ep in row["tests"][mode][p]]
    def agg(eps):
        return dict(r.aggregate(eps),held_fraction=sum(ep.get("held",0) for ep in eps)/max(1,sum(ep.get("macros",0) for ep in eps)),
                    aborts=sum(ep.get("aborts",0) for ep in eps))
    summary={m:{p:agg(scope(m,p)) for p in r.SCENARIOS} for m in MODES}
    gates={}; intervals={}
    for name,p,m,threshold in (("glass_vs_single","glass","single",.05),("glass_vs_sham","glass","sham",.05),("retain_free","free","single",-.05)):
        delta=[agg(scope("temporal",p,s))["encounter"]-agg(scope(m,p,s))["encounter"] for s in SEEDS]
        gates[name]=sum(delta)/len(delta)>=threshold
        rng=random.Random(231231)
        values=sorted(sum(rng.choice(delta) for _ in delta)/len(delta) for _ in range(2000))
        intervals[name]=[values[50],values[1949]]
    g=summary["temporal"]["glass"]
    gates["contacts"]=g["mean_contacts"]<=summary["single"]["glass"]["mean_contacts"]
    gates["persistence"]=g["held_fraction"]>=.01
    gates["sealed"]=all(not ep["crossed"] and not ep["encounter"] for m in MODES for ep in scope(m,"sealed"))
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
