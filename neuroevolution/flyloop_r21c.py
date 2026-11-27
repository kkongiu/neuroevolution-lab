"""R21C: shared physiology, slower predator clone."""
import argparse,gzip,json,math,random
from pathlib import Path
from neuroevolution import flyloop_r21b as parent, flyloop_r21a as a, flyloop_r1 as r1, flyloop_r12 as r12, flyloop_r18a as r, flyloop_r19d as d, flyloop_r20b as b, flyloop_r20c as c
from neuroevolution.flyloop_r21b import thermal_update, world, sense, flight, visible, DT
SEEDS=range(32100,32124)
ARMS=('legacy','clone','clone_equal_energy','clone_no_heat')
SCENARIOS=('free','obstacle','lantern')
def alive(body):
    return body.energy>0 and body.integrity>0

class CloneWorld(r.WindowWorld):
    def micro_integrate(self,command):
        # Scale target speed, not steering or timestep. Parent reverse plant.
        return super().micro_integrate((command[0]*.95/r1.V_MAX,command[1]))

def clone_world(w,p,seed,index,variant):
    pw=CloneWorld(seed,index,'free')
    pw.body=p
    p.energy=90. if variant=='clone_equal_energy' else 50.
    pw.rectangles=list(w.rectangles);pw.lamp=w.lamp
    pw.food=[];pw.initial_food=()
    pw.visited={(int(p.x),int(p.y))}
    return pw

def clone_command(pw,target,brain,eye,scenario):
    prox,light=eye.observe(pw,scenario!='lantern')
    source=(target.x,target.y)
    pw.initial_food=(source,)
    # Reuse the same odor sampler; virtual food exists only during sensing.
    pw.food=[source] if alive(target) and visible((pw.body.x,pw.body.y),source,pw.rectangles) else []
    _,odor=b.odor(pw,eye,False)
    pw.food=[]
    wl,wo=b.weights('adaptive',1-pw.body.energy/100.,.5)
    cue=tuple(max(wl*l,wo*o) for l,o in zip(light,odor))
    obs=pw.perceive()
    base,_,_=brain.forward((obs[0],*eye.pooled(prox),*obs[4:]))
    return d.command(base,prox,cue,.6,eye,'affordance')

def legacy_metrics(ep,seed,index,scenario):
    # Legacy archives did not record predator energy or integrity; don't invent them.
    start=ep['predator_start'][:2]
    distance=0.
    for v in ep['trace']:
        pos=v['predator'];distance+=math.dist(start,pos);start=pos
        v.update(predator_energy=None,predator_integrity=None,predator_heat=None,predator_command=None)
    ep.update(predator_damage=0.,predator_energy=None,predator_alive=None,predator_distance=distance)
    return ep

def episode(seed,index,scenario,arm,horizon=200):
    model='inertia'
    if arm=='legacy':
        old=parent.episode(seed,index,scenario,'inertia:escape',horizon)
        old['arm']=arm
        return legacy_metrics(old,seed,index,scenario)
    variant=arm
    arm='escape'
    w,p=world(seed,index,scenario); eye=r.CompoundEye()
    brain=r1.Circuit(seed,"recurrent"); hunter=r1.Circuit(seed+100000,"recurrent")
    hunter=r1.Circuit(seed,'recurrent')
    pw=clone_world(w,p,seed,index,variant)
    p=pw.body
    predator_heat=c.heat(pw,p.x,p.y,float(scenario=='lantern'))
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
            step=w.micro_integrate(cmd)
            if active:
                pst=pw.micro_integrate(pcommand)
                predator_contacts+=pst['collision']; predator_cost+=pst['metabolic_cost']; predator_distance+=pst['moved']
                predator_heat=thermal_update(predator_heat,c.heat(pw,p.x,p.y,amplitude),'inertia')
                ploss=min(p.integrity,8*max(0.,predator_heat-.65)*DT) if variant!='clone_no_heat' else 0.
                p.integrity-=ploss; predator_damage+=ploss
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
            trace.append(dict(step,predator=list(pp) if arm!="absent" else None,threat=threat,escape=u,heat=actual,integrity=w.body.integrity,captured=captured,body_heat=body_heat,urgency=urgency,before_protection=before,command=cmd,thermal_loss=loss,predator_energy=p.energy,predator_integrity=p.integrity,predator_heat=predator_heat,predator_command=pcommand if active else (0.,0.)))
    assert abs(90-sum(x["metabolic_cost"] for x in trace)+food_gain-w.body.energy)<1e-7
    assert abs((90. if variant=='clone_equal_energy' else 50.)-predator_cost+predator_gain-p.energy)<1e-7
    return dict(predator_damage=predator_damage,predator_energy=p.energy,predator_alive=alive(p),predator_distance=predator_distance,seed=seed,index=index,scenario=scenario,arm=variant,start=start,predator_start=predator_start,lamp=w.lamp if scenario=="lantern" else None,food=w.initial_food,rectangles=w.rectangles,captured=captured,survived=not captured and w.body.energy>0 and w.body.integrity>0,steps=len(trace),capture_seconds=len(trace)*DT if captured else None,food_gain=food_gain,ate=food_gain>0,thermal_damage=thermal,peak_body=peak_body,final_body=body_heat,exposure_seconds=exposure*DT,escape_fraction=escape_steps/max(1,len(trace)),seen_fraction=seen/max(1,len(trace)),min_distance=min_distance if arm!="absent" else None,contacts=sum(x["collision"] for x in trace),predator_contacts=predator_contacts,trace=trace)

def aggregate(rows):
    n=len(rows)
    result=dict(episodes=n)
    for k in ('captured','survived','ate','thermal_damage','predator_damage','predator_energy','predator_alive','predator_distance','contacts','predator_contacts'):
        vs=[x[k] for x in rows if x[k] is not None]
        result[k]=sum(vs)/len(vs) if vs else None
    return result

def shard(seed,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);summary=[]
    for arm in ARMS:
        with (out/f"{seed}-{arm}.jsonl.gz").open('wb') as f:
            with gzip.GzipFile(fileobj=f,mode='wb',mtime=0,filename='') as z:
                for scenario in SCENARIOS:
                    for i in range(20):
                        ep=episode(seed,i,scenario,arm);trace=ep.pop('trace')
                        ep['trace']=[dict(step=j+1,**{k:v[k] for k in ('position','predator','energy','integrity','body_heat','predator_heat','predator_energy','predator_integrity','command','predator_command','captured','food_gain')}) for j,v in enumerate(trace) if j%4==3 or j==len(trace)-1 or v['food_gain']>0]
                        z.write((json.dumps(ep,sort_keys=True,separators=(',',':'))+'\n').encode())
                        summary.append({k:v for k,v in ep.items() if k!='trace'})
        print(seed,arm,flush=True)
    (out/f'{seed}-summary.json').write_text(json.dumps(summary,sort_keys=True,separators=(',',':')))

def combine(directory):
    rows=[]
    for s in SEEDS:
        paths=list(Path(directory).rglob(f'{s}-summary.json'));assert len(paths)==1
        part=json.loads(paths[0].read_text())
        assert len(part)==240 and {(x['seed'],x['index'],x['scenario'],x['arm']) for x in part}=={(s,i,p,a) for i in range(20) for p in SCENARIOS for a in ARMS}
        rows.extend(part)
    def scope(a,p,s=None):return [x for x in rows if x['arm']==a and x['scenario']==p and (s is None or x['seed']==s)]
    summary={a:{p:aggregate(scope(a,p)) for p in SCENARIOS} for a in ARMS}
    p=summary['clone'];q=summary['legacy']
    gates=dict(lantern_survival=p['lantern']['survived']-q['lantern']['survived']>=.2,nontrivial_predator=p['free']['captured']>0,thermal=p['lantern']['thermal_damage']<=5,cold_invariance=all([{k:v for k,v in x.items() if k!='arm'} for x in scope('clone',s)]==[{k:v for k,v in x.items() if k!='arm'} for x in scope('clone_no_heat',s)] for s in ('free','obstacle')))
    intervals={}
    for control in ('legacy','clone_no_heat','clone_equal_energy'):
        for scene in SCENARIOS:
            delta=[aggregate(scope('clone',scene,s))['survived']-aggregate(scope(control,scene,s))['survived'] for s in SEEDS]
            rng=random.Random(321321);v=sorted(sum(rng.choice(delta) for _ in delta)/len(delta) for _ in range(2000))
            intervals[control+':'+scene]=[v[50],v[1949]]
    return dict(summary=summary,gates=gates,passed=all(gates.values()),survival_intervals=intervals)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int);p.add_argument('--out');p.add_argument('--combine');args=p.parse_args()
    if args.combine:print(json.dumps(combine(args.combine),indent=2))
    else:
        assert args.seed in SEEDS
        shard(args.seed,args.out)
