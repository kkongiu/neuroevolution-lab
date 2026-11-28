"""R23B factorial motor diagnostic; no new motivation or thermal law."""
import argparse,gzip,json,math
from pathlib import Path
from neuroevolution import flyloop_r23a as prev
from neuroevolution.flyloop_r23a import parent,b,c,d,r,r1,r12,schedule,digest,DT,HORIZON,aggregate
SEEDS=prev.SEEDS
ENV_SEEDS=prev.ENV_SEEDS
ARMS=('baseline','inhibition','coordination','both')
SCENES=('hot','cold')
def motor(base,prox,cue,eye,arm):
    ordinary=d.command(base,prox,cue,.6,eye,'affordance')
    peak=max(cue)
    if peak<=0:return ordinary,False
    strength=r1.clamp(1.2*peak,0.,1.)
    bearing=sum(v*a for a,v in zip(eye.centers,cue))/sum(cue)
    scores=[strength*math.cos(a-bearing)-2*p for a,p in zip(eye.centers,prox)]
    weights=[math.exp(8*(s-max(scores))) for s in scores]
    direction=sum(a*w for a,w in zip(eye.centers,weights))/sum(weights)
    turn=math.tanh(4*direction)
    conflict=base[1]*turn<0 and abs(base[1])>.1 and abs(turn)>.1
    if arm not in ('inhibition','both'):return ordinary,conflict
    # Directional competition: suppress opposite-sign sectors relative to the
    # dominant signed aggregate. Preserve symmetric front case.
    if abs(direction)>1e-9:
        weights=[w if a*direction>=0 else 0. for a,w in zip(eye.centers,weights)]
        turn=math.tanh(4*sum(a*w for a,w in zip(eye.centers,weights))/sum(weights))
    # Suppress opposing spontaneous steering in proportion to stimulus strength.
    spontaneous=base[1]*(1-strength) if base[1]*turn<0 else base[1]
    return (ordinary[0],(1-strength)*spontaneous+strength*turn),conflict

def coordinate(cmd):
    # Smooth motor coupling, not a stop/turn/restart state machine.
    return (cmd[0]*(1-.75*abs(cmd[1])),cmd[1])

def episode(genome,environment,index,arm,horizon=HORIZON,scene='hot'):
    w=parent.world(environment,index,90,'adaptive_hot')
    eye=r.CompoundEye();brain=r1.Circuit(genome,'recurrent')
    source=w.initial_food[0];plan=schedule(source,'renew10',horizon)
    w.food=[];cursor=0;tick=0;births=0;meals=[];returns=[]
    cost=damage=gain=0.;outside=inside=0;departures=[];departed=False
    start=(w.body.x,w.body.y,w.body.heading)
    amp=float(scene=='hot'); near=0; turning=0; conflicts=0; min_distance=math.dist((w.body.x,w.body.y),source); trace=[];min_cost=float('inf');max_cost=0.
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
        baseline,conflict=motor(base,prox,cue,eye,arm)
        for _ in range(r12.SUBSTEPS):
            if w.body.energy<=0 or w.body.integrity<=0:break
            supply()
            temp,sensors=c.thermal_sense(w,amp)
            cmd,urgency=c.protect(baseline,temp,sensors,'priority')
            before=cmd
            if arm in ('coordination','both'):cmd=coordinate(cmd)
            st=w.micro_integrate(cmd);tick+=1
            dist=math.dist((w.body.x,w.body.y),source);min_distance=min(min_distance,dist)
            near+=dist<1.;turning+=abs(cmd[1])>.5 and abs(cmd[0])>.2;conflicts+=conflict
            pos=(w.body.x,w.body.y)
            actual=c.heat(w,*pos,amp)
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
            if tick%4==0 or dist<1. or event or loss>0 or w.body.energy<=0 or w.body.integrity<=0:
                trace.append(dict(tick=tick,position=pos,energy=w.body.energy,integrity=w.body.integrity,heat=actual,food=len(w.food),gain=st['food_gain'],command=cmd,ordinary=baseline,pre_coordination=before,base=base,heading=w.body.heading,speed=w.body.speed,turn=w.body.turn,urgency=urgency,food_distance=dist,conflict=conflict))
    assert abs(90-cost+gain-w.body.energy)<1e-7
    assert births==len(w.food)+len(meals)
    assert max_cost<=.05+1e-9
    return dict(scene=scene,min_distance=min_distance,near_seconds=near*DT,turning_seconds=turning*DT,conflict_seconds=conflicts*DT,genome=genome,environment=environment,index=index,arm=arm,start=start,lamp=w.lamp,source=source,plan=plan,plan_hash=digest(plan),births=births,remaining=len(w.food),meals=meals,departures=departures,returns=returns,steps=tick,final_energy=w.body.energy,thermal_damage=damage,metabolic_cost=cost,min_step_cost=min_cost if tick else None,max_step_cost=max_cost,survived=w.body.energy>0 and w.body.integrity>0,cause='injury' if w.body.integrity<=0 else ('starvation' if w.body.energy<=0 else 'censored'),trace=trace)

def shard(seed,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);rows=[]
    with (out/f'{seed}.jsonl.gz').open('wb') as f:
        with gzip.GzipFile(fileobj=f,mode='wb',mtime=0,filename='') as z:
            for env in ENV_SEEDS:
                for i in range(2):
                    for scene in SCENES:
                        for arm in ARMS:
                            ep=episode(seed,env,i,arm,scene=scene)
                            z.write((json.dumps(ep,sort_keys=True,separators=(',',':'))+'\n').encode())
                            rows.append({k:v for k,v in ep.items() if k not in ('trace','plan')})
                print(seed,env,flush=True)
    (out/f'{seed}-summary.json').write_text(json.dumps(rows,sort_keys=True,separators=(',',':')))

def combine(directory):
    rows=[]
    for s in SEEDS:
        p=list(Path(directory).rglob(f'{s}-summary.json'));assert len(p)==1
        part=json.loads(p[0].read_text());assert len(part)==192
        assert {(x['genome'],x['environment'],x['index'],x['scene'],x['arm']) for x in part}=={(s,e,i,c,a) for e in ENV_SEEDS for i in range(2) for c in SCENES for a in ARMS}
        rows+=part
    summary={}
    for scene in SCENES:
        summary[scene]={}
        for a in ARMS:
            z=[x for x in rows if x['scene']==scene and x['arm']==a]
            result=aggregate(z)
            result.update(first_meal=sum(bool(x['meals']) for x in z)/len(z),**{k:sum(x[k] for x in z)/len(z) for k in ('min_distance','near_seconds','turning_seconds','conflict_seconds')})
            summary[scene][a]=result
    hot=summary['hot'];assert abs(hot['baseline']['survival']-274/288)<1e-9
    gates={a:dict(survival=hot[a]['survival']>=hot['baseline']['survival'],repeated=hot[a]['repeated_feeding']>=hot['baseline']['repeated_feeding'],thermal=hot[a]['mean_thermal_damage']<=1) for a in ARMS[1:]}
    return dict(summary=summary,gates=gates,diagnostic_only=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int);p.add_argument('--out');p.add_argument('--combine');args=p.parse_args()
    if args.combine:print(json.dumps(combine(args.combine),indent=2))
    else:shard(args.seed,args.out)
