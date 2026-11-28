"""R24C: unchanged sensorimotor controller, target angle manipulation."""
import argparse,gzip,json,math,random
from pathlib import Path
from types import MethodType
from neuroevolution import flyloop_r24b as g
from neuroevolution import flyloop_r1 as r1, flyloop_r18a as r, flyloop_r19d as d, flyloop_r20b as b, flyloop_r12 as r12

SEEDS=tuple(range(35100,35112))+tuple(range(36100,36112))
ANGLES=(0,45,-45,90,-90,135,-135,180)
PLANTS=('legacy','ground')
SPEEDS=(0.,.6)
CUES=('odor','masked')
DT=r12.MICRO_DT
HORIZON=150

def bearing(w,source):
    q=w.body
    return r.wrap(math.atan2(source[1]-q.y,source[0]-q.x)-q.heading)

def episode(seed,angle,plant,speed,cue_mode,horizon=HORIZON):
    w=r.WindowWorld(45100,0,'free')
    q=w.body;q.x=q.y=11.;q.heading=0.;q.speed=speed;q.turn=0.;q.energy=50.
    w.visited={(11,11)}
    source=(11+3*math.cos(math.radians(angle)),11+3*math.sin(math.radians(angle)))
    w.initial_food=(source,);w.food=[source]
    if plant=='ground':w.micro_integrate=MethodType(g.ground_integrate,w)
    eye=r.CompoundEye();brain=r1.Circuit(seed,'recurrent')
    trace=[];distance=rotation=reverse=0.;last_sign=flips=misses=0
    alignment=0. if abs(angle)<=20 else None
    near=None;inside=False;meal=None;min_dist=3.;cost=gain=0.;contacts=0
    first_speed=[];first_throttle=[];first_turn=[];first_closing=[]
    initial_cue=None
    for t in range(horizon):
        if not w.food or q.energy<=0 or q.integrity<=0:break
        prox,_=eye.observe(w,masked=True)
        _,odor=b.odor(w,eye,False)
        _,wo=b.weights('adaptive',1-q.energy/100.,.5)
        cue=tuple(wo*x if cue_mode=='odor' else 0. for x in odor)
        if initial_cue is None:
            initial_cue=dict(peak=max(cue),bearing=sum(v*a for v,a in zip(cue,eye.centers))/sum(cue) if sum(cue)>0 else None)
        obs=w.perceive()
        base,_,_=brain.forward((obs[0],*eye.pooled(prox),*obs[4:]))
        cmd=d.command(base,prox,cue,.6,eye,'affordance')
        for _ in range(r12.SUBSTEPS):
            if not w.food or q.energy<=0 or q.integrity<=0:break
            x,y,h=q.x,q.y,q.heading;old_dist=math.dist((x,y),source)
            st=w.micro_integrate(cmd)
            tick=len(trace)+1;time=tick*DT
            dist=math.dist((q.x,q.y),source);err=bearing(w,source)
            moved=math.dist((x,y),(q.x,q.y));distance+=moved
            rotation+=abs(r.wrap(q.heading-h))
            signed=((q.x-x)*math.cos(q.heading)+(q.y-y)*math.sin(q.heading))/DT
            closing=(old_dist-dist)/DT
            reverse+=DT if signed<-.02 else 0.
            sign=1 if q.turn>.1 else -1 if q.turn<-.1 else 0
            if sign:
                flips+=int(last_sign!=0 and sign!=last_sign);last_sign=sign
            if alignment is None and abs(err)<=math.radians(20):alignment=time
            if dist<=1.:
                if near is None:near=time
                inside=True
            elif inside:
                misses+=1;inside=False
            min_dist=min(min_dist,dist)
            cost+=st['metabolic_cost'];gain+=st['food_gain'];contacts+=st['collision']
            if st['consumed'] is not None:meal=time
            if tick<=20:
                first_speed.append(max(0.,signed));first_throttle.append(cmd[0])
                first_turn.append(abs(q.turn));first_closing.append(closing)
            trace.append(dict(tick=tick,position=(q.x,q.y),heading=q.heading,
                speed=q.speed,signed=signed,turn=q.turn,bearing=err,distance=dist,
                closing=closing,command=cmd,cue=cue,energy=q.energy,collision=st['collision']))
    assert abs(50-cost+gain-q.energy)<1e-7
    assert (meal is not None)==(not w.food)
    mean=lambda xs:sum(xs)/len(xs) if xs else None
    return dict(seed=seed,angle=angle,plant=plant,speed=speed,cue=cue_mode,
        source=source,start=(11.,11.,0.),initial_cue=initial_cue,steps=len(trace),
        meal=meal,alignment=alignment,near=near,misses=misses,min_dist=min_dist,
        distance=distance,rotation=rotation,reverse_seconds=reverse,flips=flips,contacts=contacts,
        first_speed=mean(first_speed),first_throttle=mean(first_throttle),first_turn=mean(first_turn),
        first_closing=mean(first_closing),first_second_complete=len(first_speed)==20,trace=trace)

def shard(seed,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);rows=[]
    with (out/f'{seed}.jsonl.gz').open('wb') as f:
        with gzip.GzipFile(fileobj=f,mode='wb',mtime=0,filename='') as z:
            for angle in ANGLES:
                for p in PLANTS:
                    for speed in SPEEDS:
                        for c in CUES:
                            ep=episode(seed,angle,p,speed,c)
                            z.write((json.dumps(ep,sort_keys=True,separators=(',',':'))+'\n').encode())
                            rows.append({k:v for k,v in ep.items() if k!='trace'})
    (out/f'{seed}-summary.json').write_text(json.dumps(rows,sort_keys=True,separators=(',',':')))

def mean(xs):
    xs=list(xs)
    return sum(xs)/len(xs) if xs else None

def aggregate(rows):
    return dict(n=len(rows),meal=mean(x['meal'] is not None for x in rows),
        meal_time=mean(x['meal'] for x in rows if x['meal'] is not None),
        aligned_2s=mean(x['alignment'] is not None and x['alignment']<=2 for x in rows),
        near=mean(x['near'] is not None for x in rows),
        near_miss=mean(x['misses']>0 for x in rows),
        mean_misses=mean(x['misses'] for x in rows),
        min_distance=mean(x['min_dist'] for x in rows),
        first_speed=mean(x['first_speed'] for x in rows),
        first_throttle=mean(x['first_throttle'] for x in rows),
        first_turn=mean(x['first_turn'] for x in rows),
        first_closing=mean(x['first_closing'] for x in rows),
        initial_peak=mean(x['initial_cue']['peak'] for x in rows),
        mean_distance=mean(x['distance'] for x in rows),
        mean_rotation=mean(x['rotation'] for x in rows),
        contacts=mean(x['contacts'] for x in rows),
        reverse_fraction=sum(x['reverse_seconds'] for x in rows)/sum(x['steps']*DT for x in rows),
        flips_per_minute=60*sum(x['flips'] for x in rows)/sum(x['steps']*DT for x in rows))

def contrast(rows,seeds,metric):
    ds=[]
    for seed in seeds:
        a=[x for x in rows if x['seed']==seed]
        ds.append(mean(x[metric] for x in a if abs(x['angle']) in (90,135))-mean(x[metric] for x in a if x['angle']==0))
    rng=random.Random(2403)
    boot=sorted(mean(rng.choices(ds,k=len(ds))) for _ in range(2000))
    return dict(delta=mean(ds),interval=[boot[50],boot[1949]])

def combine(directory):
    rows=[]
    for seed in SEEDS:
        paths=list(Path(directory).rglob(f'{seed}-summary.json'));assert len(paths)==1
        part=json.loads(paths[0].read_text())
        assert len(part)==64
        assert {(x['seed'],x['angle'],x['plant'],x['speed'],x['cue']) for x in part}=={(seed,a,p,s,c) for a in ANGLES for p in PLANTS for s in SPEEDS for c in CUES}
        rows.extend(part)
    assert all(x['first_second_complete'] for x in rows)
    summary={};contrasts={};signed={}
    for cohort,seeds in (('all',SEEDS),('old',SEEDS[:12]),('new',SEEDS[12:])):
        summary[cohort]={};contrasts[cohort]={};signed[cohort]={}
        for p in PLANTS:
            for speed in SPEEDS:
                for c in CUES:
                    key=f'{p}_{speed}_{c}'
                    scope=[x for x in rows if x['seed'] in seeds and x['plant']==p and x['speed']==speed and x['cue']==c]
                    summary[cohort][key]={str(a):aggregate([x for x in scope if abs(x['angle'])==a]) for a in (0,45,90,135,180)}
                    signed[cohort][key]={str(a):aggregate([x for x in scope if x['angle']==a]) for a in ANGLES}
                    contrasts[cohort][key]={m:contrast(scope,seeds,m) for m in ('first_speed','first_throttle')}
    return dict(summary=summary,signed_angles=signed,contrasts=contrasts,diagnostic_only=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int);p.add_argument('--out');p.add_argument('--combine');a=p.parse_args()
    if a.combine:print(json.dumps(combine(a.combine),indent=2))
    else:
        assert a.seed in SEEDS
        shard(a.seed,a.out)
