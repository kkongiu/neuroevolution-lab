"""R21D: shared energy-dependent capacity, extended observation."""
import argparse,gzip,json,math,random
from pathlib import Path
from neuroevolution import flyloop_r21c as prev, flyloop_r1 as r1, flyloop_r12 as r12, flyloop_r18a as r, flyloop_r19d as d, flyloop_r20b as b, flyloop_r20c as c
from neuroevolution.flyloop_r21c import alive,clone_world,clone_command,world,sense,flight,visible,thermal_update,DT
SEEDS=range(33100,33124)
ARMS=('normal50','weak50','normal90','weak90')
SCENARIOS=prev.SCENARIOS
def capacity(energy,weak=True):
    return r1.clamp(energy/30.,0.,1.) if weak else 1.
def limit(command,factor):
    if factor==1.:return command
    return (command[0]*factor,math.atanh(factor*math.tanh(1.6*command[1]))/1.6)

def episode(seed,index,scenario,arm,horizon=1000):
    model='inertia'
    variant=arm
    weak=variant.startswith('weak')
    arm='escape'
    w,p=world(seed,index,scenario); eye=r.CompoundEye()
    brain=r1.Circuit(seed,"recurrent"); hunter=r1.Circuit(seed+100000,"recurrent")
    hunter=r1.Circuit(seed,'recurrent')
    pw=clone_world(w,p,seed,index,'clone_equal_energy' if variant.endswith('90') else 'clone')
    p=pw.body
    predator_heat=c.heat(pw,p.x,p.y,float(scenario=='lantern'))
    prey_low=pred_low=0; predator_death_step=None
    predator_damage=0.; predator_gain=0.; predator_cost=0.; predator_distance=0.
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
        pc=clone_command(pw,w.body,hunter,eye,scenario) if alive(p) else (0.,0.)
        for _ in range(r12.SUBSTEPS):
            if captured or w.body.energy<=0 or w.body.integrity<=0:break
            active=alive(p)
            threat=sense(w.body,p,w.rectangles,3.5,arm!="unoccluded") if active else (0.,)*12
            seen+=bool(max(threat))
            cmd,u=flight(ordinary,threat if arm in ("escape","unoccluded") else (0.,)*12,prox,eye)
            escape_steps+=u>0
            amplitude=float(scenario=="lantern")
            temp,sensors=c.thermal_sense(w,amplitude)
            before=cmd
            reflex_temp=body_heat if model in ('reflex_only','inertia','fast_cooling') else temp
            cmd,urgency=c.protect(cmd,reflex_temp,sensors,'priority')
            if active:
                _,ps=c.thermal_sense(pw,amplitude)
                pcommand,pu=c.protect(pc,predator_heat if variant!='clone_no_heat' else 0.,ps,'priority')
            prey_low+=w.body.energy<30.
            pred_low+=active and p.energy<30.
            prey_capacity=capacity(w.body.energy,weak)
            pred_capacity=capacity(p.energy,weak)
            cmd=limit(cmd,prey_capacity)
            if active:pcommand=limit(pcommand,pred_capacity)
            step=w.micro_integrate(cmd)
            if active:
                pst=pw.micro_integrate(pcommand)
                predator_contacts+=pst['collision']; predator_cost+=pst['metabolic_cost']; predator_distance+=pst['moved']
                predator_heat=thermal_update(predator_heat,c.heat(pw,p.x,p.y,amplitude),'inertia')
                ploss=min(p.integrity,8*max(0.,predator_heat-.65)*DT) if variant!='clone_no_heat' else 0.
                p.integrity-=ploss; predator_damage+=ploss
            if active and not alive(p):predator_death_step=len(trace)+1
            actual=c.heat(w,w.body.x,w.body.y,amplitude)
            body_heat=thermal_update(body_heat,actual,model)
            peak_body=max(peak_body,body_heat)
            exposure+=actual>.65
            damage_temp=body_heat if model in ('damage_only','inertia','fast_cooling') else actual
            loss=min(w.body.integrity,8*max(0.,damage_temp-.65)*DT)
            w.body.integrity-=loss;thermal+=loss;food_gain+=step["food_gain"]
            q=(w.body.x,w.body.y); pp=(p.x,p.y)
            distance=math.dist(q,pp)
            if active and alive(p) and alive(w.body):
                min_distance=min(min_distance,distance)
                contact=contact+1 if distance<=.75 and visible(q,pp,w.rectangles) else 0
                if contact>=4 and p.energy<95.:
                    captured=True; predator_gain=min(12.,100.-p.energy);p.energy+=predator_gain
            else:contact=0
            trace.append(dict(step,predator=list(pp) if arm!="absent" else None,threat=threat,escape=u,heat=actual,integrity=w.body.integrity,captured=captured,body_heat=body_heat,urgency=urgency,before_protection=before,command=cmd,thermal_loss=loss,predator_energy=p.energy,predator_integrity=p.integrity,predator_heat=predator_heat,predator_command=pcommand if active else (0.,0.),prey_capacity=prey_capacity,predator_capacity=pred_capacity))
    assert abs(90-sum(x["metabolic_cost"] for x in trace)+food_gain-w.body.energy)<1e-7
    assert abs((90. if variant.endswith('90') else 50.)-predator_cost+predator_gain-p.energy)<1e-7
    cause='capture' if captured else ('injury' if w.body.integrity<=0 else ('starvation' if w.body.energy<=0 else 'censored'))
    pred_cause='injury' if p.integrity<=0 else ('starvation' if p.energy<=0 else 'censored')
    return dict(prey_cause=cause,predator_cause=pred_cause,predator_death_step=predator_death_step,prey_low_seconds=prey_low*DT,predator_low_seconds=pred_low*DT,prey_energy=w.body.energy,prey_distance=sum(v['moved'] for v in trace),predator_damage=predator_damage,predator_energy=p.energy,predator_alive=alive(p),predator_distance=predator_distance,seed=seed,index=index,scenario=scenario,arm=variant,start=start,predator_start=predator_start,lamp=w.lamp if scenario=="lantern" else None,food=w.initial_food,rectangles=w.rectangles,captured=captured,survived=not captured and w.body.energy>0 and w.body.integrity>0,steps=len(trace),capture_seconds=len(trace)*DT if captured else None,food_gain=food_gain,ate=food_gain>0,thermal_damage=thermal,peak_body=peak_body,final_body=body_heat,exposure_seconds=exposure*DT,escape_fraction=escape_steps/max(1,len(trace)),seen_fraction=seen/max(1,len(trace)),min_distance=min_distance if arm!="absent" else None,contacts=sum(x["collision"] for x in trace),predator_contacts=predator_contacts,trace=trace)

def aggregate(rows):
    n=len(rows)
    out=dict(episodes=n)
    for k in ('captured','survived','ate','thermal_damage','predator_damage','predator_energy','predator_alive','prey_energy','prey_distance','predator_distance','prey_low_seconds','predator_low_seconds'):
        out[k]=sum(x[k] for x in rows)/n
    for who in ('prey','predator'):
        out[who+'_causes']={cause:sum(x[who+'_cause']==cause for x in rows) for cause in ('capture','starvation','injury','censored')}
    return out

def shard(seed,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);summary=[]
    for arm in ARMS:
        with (out/f'{seed}-{arm}.jsonl.gz').open('wb') as f:
            with gzip.GzipFile(fileobj=f,mode='wb',mtime=0,filename='') as z:
                for scenario in SCENARIOS:
                    for i in range(10):
                        ep=episode(seed,i,scenario,arm);trace=ep.pop('trace')
                        ep['trace']=[dict(step=j+1,**{k:v[k] for k in ('position','predator','energy','integrity','predator_energy','predator_integrity','body_heat','predator_heat','prey_capacity','predator_capacity','captured','food_gain')}) for j,v in enumerate(trace) if j%20==19 or j==0 or j==len(trace)-1 or v['food_gain']>0 or j+1==ep['predator_death_step']]
                        z.write((json.dumps(ep,sort_keys=True,separators=(',',':'))+'\n').encode())
                        summary.append({k:v for k,v in ep.items() if k!='trace'})
        print(seed,arm,flush=True)
    (out/f'{seed}-summary.json').write_text(json.dumps(summary,sort_keys=True,separators=(',',':')))

def combine(directory):
    rows=[]
    for s in SEEDS:
        paths=list(Path(directory).rglob(f'{s}-summary.json'));assert len(paths)==1
        part=json.loads(paths[0].read_text())
        assert len(part)==120 and {(x['seed'],x['index'],x['scenario'],x['arm']) for x in part}=={(s,i,p,a) for i in range(10) for p in SCENARIOS for a in ARMS}
        rows.extend(part)
    def scope(a,p,s=None):return [x for x in rows if x['arm']==a and x['scenario']==p and (s is None or x['seed']==s)]
    summary={a:{p:aggregate(scope(a,p)) for p in SCENARIOS} for a in ARMS}
    intervals={};equivalent=0
    for e in ('50','90'):
        for scene in SCENARIOS:
            x=scope('normal'+e,scene);y=scope('weak'+e,scene)
            for u,v in zip(x,y):
                if u['prey_low_seconds']==u['predator_low_seconds']==0:
                    assert {k:w for k,w in u.items() if k!='arm'}=={k:w for k,w in v.items() if k!='arm'}
                    equivalent+=1
            for metric in ('survived','captured'):
                delta=[aggregate(scope('weak'+e,scene,s))[metric]-aggregate(scope('normal'+e,scene,s))[metric] for s in SEEDS]
                rng=random.Random(331331);v=sorted(sum(rng.choice(delta) for _ in delta)/len(delta) for _ in range(2000))
                intervals[e+':'+scene+':'+metric]=[v[50],v[1949]]
    return dict(summary=summary,intervals=intervals,above_threshold_equivalent_pairs=equivalent)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int);p.add_argument('--out');p.add_argument('--combine');args=p.parse_args()
    if args.combine:print(json.dumps(combine(args.combine),indent=2))
    else:
        assert args.seed in SEEDS
        shard(args.seed,args.out)
