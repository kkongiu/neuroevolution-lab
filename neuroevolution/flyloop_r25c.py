"""R25C: transfer food sensory priority into renewable hot aquarium."""
from neuroevolution import flyloop_r24b as old, flyloop_r25b as v
import random
from neuroevolution import flyloop_r24a as physics, flyloop_r12c as tactile
from types import MethodType
import argparse,gzip,hashlib,json,math
from pathlib import Path
from neuroevolution import flyloop_r20d as parent
from neuroevolution import flyloop_r20b as b, flyloop_r20c as c, flyloop_r19d as d, flyloop_r18a as r, flyloop_r1 as r1, flyloop_r12 as r12

SEEDS=tuple(range(35100,35112))+tuple(range(39100,39112))
ENV_SEEDS=range(45100,45112)
ARMS=('odor','combined','hard','soft')
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
    source=w.initial_food[0];plan=schedule(source,'renew10',horizon)
    w.micro_integrate=MethodType(old.ground_integrate,w)
    distance=rotation=reverse=0.; flips=last_sign=0; peak_lat=0.
    w.food=[];cursor=0;tick=0;births=0;meals=[];returns=[]
    cost=damage=gain=0.;outside=inside=0;departures=[];departed=False
    start=(w.body.x,w.body.y,w.body.heading)
    trace=[];min_cost=float('inf');max_cost=0.
    warm=0;post_meal_returns=0;meal_since_return=False;visual_weight_sum=0.;visible=0
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
        vision=v.retina(w,eye,source)
        fused,alpha=v.fuse(odor,vision,arm)
        cue=tuple(max(wl*l,wo*o) for l,o in zip(light,fused))
        obs=w.perceive()
        base,_,_=brain.forward((obs[0],*eye.pooled(prox),*obs[4:]))
        baseline=d.command(base,prox,cue,.6,eye,'affordance')
        for _ in range(r12.SUBSTEPS):
            if w.body.energy<=0 or w.body.integrity<=0:break
            supply()
            temp,sensors=c.thermal_sense(w,1.)
            cmd,urgency=c.protect(baseline,temp,sensors,'priority')
            old=(w.body.x,w.body.y,w.body.heading)
            st=w.micro_integrate(cmd);tick+=1
            visual_weight_sum+=alpha if alpha is not None else 0.
            visible+=int(sum(vision)>1e-9)
            q=w.body; dx=q.x-old[0];dy=q.y-old[1]
            moved=math.hypot(dx,dy);distance+=moved
            rotation+=abs((q.heading-old[2]+math.pi)%(2*math.pi)-math.pi)
            signed=(dx*math.cos(q.heading)+dy*math.sin(q.heading))/DT
            reverse+=DT if signed<-.02 else 0.
            sign=1 if q.turn>.1 else (-1 if q.turn<-.1 else 0)
            if sign:
                flips+=int(last_sign!=0 and sign!=last_sign);last_sign=sign
            peak_lat=max(peak_lat,abs(q.speed*q.turn))
            pos=(w.body.x,w.body.y)
            actual=c.heat(w,*pos,1.)
            loss=min(w.body.integrity,8*max(0.,actual-.65)*DT)
            w.body.integrity-=loss;damage+=loss
            cost+=st['metabolic_cost'];gain+=st['food_gain']
            min_cost=min(min_cost,st['metabolic_cost']);max_cost=max(max_cost,st['metabolic_cost'])
            radius=math.dist(pos,w.lamp)
            outside=outside+1 if radius>2.125 else 0
            inside=inside+1 if .875<=radius<=2.125 else 0
            warm+=int(.875<=radius<=2.125)
            event=False
            if outside>=20 and not departed:
                departures.append(tick);departed=True;event=True
            if st['consumed'] is not None:
                meals.append(dict(tick=tick,energy=w.body.energy,gain=st['food_gain'],speed=q.speed,bearing=v.bearing(w,source),position=pos,heading=q.heading))
                meal_since_return=True
                event=True
            if inside>=20 and departed:
                returns.append(tick);departed=False;event=True
                post_meal_returns+=int(meal_since_return);meal_since_return=False
            if tick%4==0 or event or loss>0 or w.body.energy<=0 or w.body.integrity<=0:
                trace.append(dict(tick=tick,position=pos,energy=w.body.energy,integrity=w.body.integrity,heat=actual,food=len(w.food),gain=st['food_gain'],command=cmd,heading=w.body.heading,speed=w.body.speed,turn=w.body.turn,signed=signed,vision=vision,odor=odor,visual_weight=alpha,light=light,urgency=urgency))
    assert abs(90-cost+gain-w.body.energy)<1e-7
    assert births==len(w.food)+len(meals)
    assert max_cost<=.05+1e-9
    return dict(genome=genome,environment=environment,index=index,arm=arm,start=start,lamp=w.lamp,source=source,plan=plan,plan_hash=digest(plan),births=births,remaining=len(w.food),meals=meals,departures=departures,returns=returns,steps=tick,final_energy=w.body.energy,thermal_damage=damage,metabolic_cost=cost,min_step_cost=min_cost if tick else None,max_step_cost=max_cost,survived=w.body.energy>0 and w.body.integrity>0,cause='injury' if w.body.integrity<=0 else ('starvation' if w.body.energy<=0 else 'censored'),trace=trace,distance=distance,rotation=rotation,reverse_seconds=reverse,flips=flips,peak_lateral=peak_lat,warm_fraction=warm/max(1,tick),post_meal_returns=post_meal_returns,visual_weight=visual_weight_sum/max(1,tick) if arm!='combined' else None,visible_fraction=visible/max(1,tick))

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
    return dict(episodes=n,survival=sum(x['survived'] for x in rows)/n,repeated_feeding=sum(len(x['meals'])>=5 for x in rows)/n,mean_meals=sum(len(x['meals']) for x in rows)/n,mean_returns=sum(len(x['returns']) for x in rows)/n,mean_energy=sum(x['final_energy'] for x in rows)/n,mean_thermal_damage=sum(x['thermal_damage'] for x in rows)/n,mean_births=sum(x['births'] for x in rows)/n,mean_remaining=sum(x['remaining'] for x in rows)/n,mean_metabolic_cost=sum(x['metabolic_cost'] for x in rows)/n,mean_distance=sum(x['distance'] for x in rows)/n,mean_rotation=sum(x['rotation'] for x in rows)/n,reverse_fraction=sum(x['reverse_seconds'] for x in rows)/sum(x['steps']*DT for x in rows),flips_per_minute=60*sum(x['flips'] for x in rows)/sum(x['steps']*DT for x in rows),rotation_per_distance=sum(x['rotation'] for x in rows)/max(1e-12,sum(x['distance'] for x in rows)),warm_fraction=sum(x['warm_fraction'] for x in rows)/n,mean_post_meal_returns=sum(x['post_meal_returns'] for x in rows)/n,mean_meal_speed=mean(m['speed'] for x in rows for m in x['meals']),mean_abs_meal_bearing=mean(abs(m['bearing']) for x in rows for m in x['meals']),visual_weight=mean(x['visual_weight'] for x in rows if x['visual_weight'] is not None),visible_fraction=mean(x['visible_fraction'] for x in rows))

def mean(xs):
    xs=list(xs)
    return sum(xs)/len(xs) if xs else None

def combine(directory):
    rows=[]
    for s in SEEDS:
        paths=list(Path(directory).rglob(f'{s}-summary.json'));assert len(paths)==1
        part=json.loads(paths[0].read_text())
        assert {(x['genome'],x['environment'],x['index'],x['arm']) for x in part}=={(s,e,i,a) for e in ENV_SEEDS for i in range(2) for a in ARMS}
        assert len(part)==96;rows.extend(part)
    for e in ENV_SEEDS:
        for i in range(2):
            for a in ARMS:
                assert len({x['plan_hash'] for x in rows if x['environment']==e and x['index']==i and x['arm']==a})==1
    old_rows=[x for x in rows if x['genome'] in SEEDS[:12] and x['arm']=='odor']
    assert sum(x['survived'] for x in old_rows)==281
    assert sum(len(x['meals'])>=5 for x in old_rows)==278
    assert all(x['peak_lateral']<=.8+1e-9 for x in rows)
    summary={};contrasts={}
    for cohort,seeds in (('all',SEEDS),('old',SEEDS[:12]),('new',SEEDS[12:])):
        scope=[x for x in rows if x['genome'] in seeds]
        summary[cohort]={a:aggregate([x for x in scope if x['arm']==a]) for a in ARMS}
        contrasts[cohort]={}
        for arm in ARMS[1:]:
            contrasts[cohort][arm]={}
            for metric in ('survived','repeated'):
                ds=[]
                for seed in seeds:
                    def score(a):
                        return mean(float(x['survived'] if metric=='survived' else len(x['meals'])>=5) for x in scope if x['genome']==seed and x['arm']==a)
                    ds.append(score(arm)-score('odor'))
                rng=random.Random(2503)
                boot=sorted(mean(rng.choices(ds,k=len(ds))) for _ in range(2000))
                contrasts[cohort][arm][metric]=dict(delta=mean(ds),interval=[boot[50],boot[1949]])
    return dict(summary=summary,contrasts=contrasts,baseline_reproduced=True,diagnostic_only=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int);p.add_argument('--out');p.add_argument('--combine');args=p.parse_args()
    if args.combine:print(json.dumps(combine(args.combine),indent=2))
    else:
        assert args.seed in SEEDS
        shard(args.seed,args.out)
