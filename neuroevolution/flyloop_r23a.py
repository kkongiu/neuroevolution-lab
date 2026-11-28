"""R23A: single-agent, deterministic supply, independent genome and layout."""
import argparse,gzip,hashlib,json,math
from pathlib import Path
from neuroevolution import flyloop_r20d as parent
from neuroevolution import flyloop_r20b as b, flyloop_r20c as c, flyloop_r19d as d, flyloop_r18a as r, flyloop_r1 as r1, flyloop_r12 as r12

SEEDS=range(35100,35112)
ENV_SEEDS=range(45100,45112)
ARMS=('finite','renew10','renew5')
HORIZON=1000
DT=r12.MICRO_DT

def schedule(source,arm,horizon):
    interval={'finite':horizon*r12.SUBSTEPS+1,'renew10':200,'renew5':100}[arm]
    return tuple((tick,source,12.) for tick in range(0,horizon*r12.SUBSTEPS,interval))

def digest(plan):
    return hashlib.sha256(json.dumps(plan,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def episode(genome,environment,index,arm,horizon=HORIZON):
    w=parent.world(environment,index,90,'adaptive_hot')
    eye=r.CompoundEye();brain=r1.Circuit(genome,'recurrent')
    source=w.initial_food[0];plan=schedule(source,arm,horizon)
    w.food=[];cursor=0;tick=0;births=0;meals=[];returns=[]
    cost=damage=gain=0.;outside=inside=0;departures=[];departed=False
    start=(w.body.x,w.body.y,w.body.heading)
    trace=[];min_cost=float('inf');max_cost=0.
    def supply():
        nonlocal cursor,births
        while cursor<len(plan) and plan[cursor][0]==tick:
            w.food.append(source);cursor+=1;births+=1
    for t in range(horizon):
        if w.body.energy<=0 or w.body.integrity<=0:break
        supply()
        prox,light=eye.observe(w)
        _,odor=b.odor(w,eye,False)
        wl,wo=b.weights('adaptive',1-w.body.energy/100.,.1)
        cue=tuple(max(wl*l,wo*o) for l,o in zip(light,odor))
        obs=w.perceive()
        base,_,_=brain.forward((obs[0],*eye.pooled(prox),*obs[4:]))
        baseline=d.command(base,prox,cue,.6,eye,'affordance')
        for _ in range(r12.SUBSTEPS):
            if w.body.energy<=0 or w.body.integrity<=0:break
            supply()
            temp,sensors=c.thermal_sense(w,1.)
            cmd,urgency=c.protect(baseline,temp,sensors,'priority')
            st=w.micro_integrate(cmd);tick+=1
            pos=(w.body.x,w.body.y)
            actual=c.heat(w,*pos,1.)
            loss=min(w.body.integrity,8*max(0.,actual-.65)*DT)
            w.body.integrity-=loss;damage+=loss
            cost+=st['metabolic_cost'];gain+=st['food_gain']
            min_cost=min(min_cost,st['metabolic_cost']);max_cost=max(max_cost,st['metabolic_cost'])
            radius=math.dist(pos,w.lamp)
            outside=outside+1 if radius>2.125 else 0
            inside=inside+1 if .875<=radius<=2.125 else 0
            event=False
            if outside>=20 and not departed:
                departures.append(tick);departed=True;event=True
            if st['consumed'] is not None:
                meals.append(dict(tick=tick,energy=w.body.energy,gain=st['food_gain']))
                event=True
            if inside>=20 and departed:
                returns.append(tick);departed=False;event=True
            if tick%20==0 or event or loss>0 or w.body.energy<=0 or w.body.integrity<=0:
                trace.append(dict(tick=tick,position=pos,energy=w.body.energy,integrity=w.body.integrity,heat=actual,food=len(w.food),gain=st['food_gain'],command=cmd))
    assert abs(90-cost+gain-w.body.energy)<1e-7
    assert births==len(w.food)+len(meals)
    assert max_cost<=.05+1e-9
    return dict(genome=genome,environment=environment,index=index,arm=arm,start=start,lamp=w.lamp,source=source,plan=plan,plan_hash=digest(plan),births=births,remaining=len(w.food),meals=meals,departures=departures,returns=returns,steps=tick,final_energy=w.body.energy,thermal_damage=damage,metabolic_cost=cost,min_step_cost=min_cost if tick else None,max_step_cost=max_cost,survived=w.body.energy>0 and w.body.integrity>0,cause='injury' if w.body.integrity<=0 else ('starvation' if w.body.energy<=0 else 'censored'),trace=trace)

def shard(seed,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);rows=[]
    with (out/f'{seed}.jsonl.gz').open('wb') as f:
        with gzip.GzipFile(fileobj=f,mode='wb',mtime=0,filename='') as z:
            for env in ENV_SEEDS:
                for i in range(2):
                    for arm in ARMS:
                        ep=episode(seed,env,i,arm)
                        z.write((json.dumps(ep,sort_keys=True,separators=(',',':'))+'\n').encode())
                        rows.append({k:v for k,v in ep.items() if k not in ('trace','plan')})
                print(seed,env,flush=True)
    (out/f'{seed}-summary.json').write_text(json.dumps(rows,sort_keys=True,separators=(',',':')))

def aggregate(rows):
    n=len(rows)
    return dict(episodes=n,survival=sum(x['survived'] for x in rows)/n,repeated_feeding=sum(len(x['meals'])>=5 for x in rows)/n,mean_meals=sum(len(x['meals']) for x in rows)/n,mean_returns=sum(len(x['returns']) for x in rows)/n,mean_energy=sum(x['final_energy'] for x in rows)/n,mean_thermal_damage=sum(x['thermal_damage'] for x in rows)/n,mean_births=sum(x['births'] for x in rows)/n,mean_remaining=sum(x['remaining'] for x in rows)/n,mean_metabolic_cost=sum(x['metabolic_cost'] for x in rows)/n)

def combine(directory):
    rows=[]
    for s in SEEDS:
        paths=list(Path(directory).rglob(f'{s}-summary.json'));assert len(paths)==1
        part=json.loads(paths[0].read_text())
        assert {(x['genome'],x['environment'],x['index'],x['arm']) for x in part}=={(s,e,i,a) for e in ENV_SEEDS for i in range(2) for a in ARMS}
        assert len(part)==72;rows.extend(part)
    for e in ENV_SEEDS:
        for i in range(2):
            for a in ARMS:
                assert len({x['plan_hash'] for x in rows if x['environment']==e and x['index']==i and x['arm']==a})==1
    summary={a:aggregate([x for x in rows if x['arm']==a]) for a in ARMS}
    gates={a:dict(survival=summary[a]['survival']>=.8,repeated=summary[a]['repeated_feeding']>=.7,thermal=summary[a]['mean_thermal_damage']<=1,gain=summary[a]['survival']-summary['finite']['survival']>=.3) for a in ARMS[1:]}
    margins={key:{str(s):{a:aggregate([x for x in rows if x[key]==s and x['arm']==a])['survival'] for a in ARMS} for s in seeds} for key,seeds in (('genome',SEEDS),('environment',ENV_SEEDS))}
    return dict(summary=summary,gates=gates,passed=any(all(v.values()) for v in gates.values()),survival_seed_margins=margins)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int);p.add_argument('--out');p.add_argument('--combine');args=p.parse_args()
    if args.combine:print(json.dumps(combine(args.combine),indent=2))
    else:
        assert args.seed in SEEDS
        shard(args.seed,args.out)
