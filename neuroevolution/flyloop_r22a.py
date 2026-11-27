"""R22A: two independent prey, shared food, persistent predator."""
import argparse,gzip,json,math,random
from pathlib import Path
from neuroevolution import flyloop_r21d as prev, flyloop_r1 as r1, flyloop_r12 as r12, flyloop_r18a as r, flyloop_r19d as d, flyloop_r20b as b, flyloop_r20c as c
from neuroevolution.flyloop_r21d import alive,capacity,limit,world,flight,sense,visible,thermal_update,DT
SEEDS=range(34100,34124)
ARMS=('one','two','two_no_predator')
SCENARIOS=prev.SCENARIOS
def pos(w):return (w.body.x,w.body.y)
def extra_world(w,seed,index):
    x=r.WindowWorld(seed,index,'free')
    x.rectangles=list(w.rectangles);x.lamp=w.lamp
    h=w.body.heading
    for angle in (h+math.pi/2,h-math.pi/2,h+math.pi):
        q=(w.body.x+2*math.cos(angle),w.body.y+2*math.sin(angle))
        if not x.blocked(*q):break
    else:raise AssertionError('no second spawn')
    x.body.x,x.body.y=q;x.body.heading=h;x.body.energy=90.
    x.food=[];x.initial_food=();x.visited={(int(q[0]),int(q[1]))}
    return x
def odor_sources(w,eye,sources):
    # Local componentwise maximum of the same individual odor fields.
    old_food,old_initial=w.food,w.initial_food
    fields=[]
    for q in sources:
        w.food=[q];w.initial_food=(q,)
        fields.append(b.odor(w,eye,False)[1])
    w.food,w.initial_food=old_food,old_initial
    return tuple(max((f[i] for f in fields),default=0.) for i in range(eye.sectors))
def command(w,brain,eye,sources,scene):
    prox,light=eye.observe(w,scene!='lantern')
    odor=odor_sources(w,eye,sources)
    wl,wo=b.weights('adaptive',1-w.body.energy/100.,.1)
    cue=tuple(max(wl*l,wo*o) for l,o in zip(light,odor))
    obs=w.perceive()
    base,_,_=brain.forward((obs[0],*eye.pooled(prox),*obs[4:]))
    return d.command(base,prox,cue,.6,eye,'affordance'),prox
def episode(seed,index,scene,arm,horizon=1000):
    w,p=world(seed,index,scene)
    food=list(w.food);initial_food=list(food)
    prey=[w]
    if arm!='one':prey.append(extra_world(w,seed,index))
    hunter=prev.clone_world(w,p,seed,index,'clone')
    agents=prey+[hunter]
    enabled=[True]*len(prey)+[arm!='two_no_predator']
    brains=[r1.Circuit(seed if i in (0,len(prey)) else seed+200000,'recurrent') for i in range(len(agents))]
    for x in agents:x.food=[];x.initial_food=()
    eye=r.CompoundEye();amp=float(scene=='lantern')
    heat=[c.heat(x,*pos(x),amp) for x in agents]
    causes=[None if e else 'absent' for e in enabled]
    costs=[0.]*len(agents);gains=[0.]*len(agents);damage=[0.]*len(agents)
    initial_energy=[x.body.energy for x in agents]
    contact=[0]*len(prey);captures=[];trace=[];step=0
    start=[pos(x) for x in agents]
    def active(i):return enabled[i] and causes[i] is None and alive(agents[i].body)
    def snapshot():
        return dict(step=step,positions=[pos(x) for x in agents],energy=[x.body.energy for x in agents],integrity=[x.body.integrity for x in agents],heat=list(heat),causes=list(causes),food=list(food))
    trace.append(snapshot())
    for t in range(horizon):
        if not any(active(i) for i in range(len(agents))):break
        ordinary=[]
        for i,x in enumerate(agents):
            sources=food if i<len(prey) else [pos(y) for j,y in enumerate(prey) if active(j) and visible(pos(x),pos(y),x.rectangles)]
            ordinary.append(command(x,brains[i],eye,sources,scene) if active(i) else ((0.,0.),(0.,)*eye.sectors))
        for _ in range(r12.SUBSTEPS):
            if not any(active(i) for i in range(len(agents))):break
            step+=1;commands=[];events=False
            for i,x in enumerate(agents):
                cmd,prox=ordinary[i]
                if active(i):
                    if i<len(prey) and active(len(prey)):
                        cmd,_=flight(cmd,sense(x.body,hunter.body,x.rectangles,3.5),prox,eye)
                    _,sensors=c.thermal_sense(x,amp)
                    cmd,_=c.protect(cmd,heat[i],sensors,'priority')
                    cmd=limit(cmd,capacity(x.body.energy))
                commands.append(cmd)
            # All commands computed from the same pre-step state.
            for i,x in enumerate(agents):
                if not active(i):continue
                st=x.micro_integrate(commands[i]);costs[i]+=st['metabolic_cost']
                heat[i]=thermal_update(heat[i],c.heat(x,*pos(x),amp),'inertia')
                loss=min(x.body.integrity,8*max(0.,heat[i]-.65)*DT)
                x.body.integrity-=loss;damage[i]+=loss
                if not alive(x.body):
                    causes[i]='injury' if x.body.integrity<=0 else 'starvation';events=True
            # Shared food allocated once, nearest eligible prey; index breaks exact ties.
            for q in list(food):
                eligible=[i for i,x in enumerate(prey) if active(i) and x.body.energy<95 and math.dist(pos(x),q)<=.55]
                if eligible:
                    i=min(eligible,key=lambda j:(math.dist(pos(prey[j]),q),j))
                    gain=min(12.,100.-prey[i].body.energy);prey[i].body.energy+=gain;gains[i]+=gain
                    food.remove(q);events=True
            j=len(prey)
            eligible=[]
            for i,x in enumerate(prey):
                touching=active(i) and active(j) and math.dist(pos(x),pos(hunter))<=.75 and visible(pos(x),pos(hunter),x.rectangles)
                contact[i]=contact[i]+1 if touching else 0
                if contact[i]>=4:eligible.append(i)
            if eligible and hunter.body.energy<95:
                i=min(eligible,key=lambda k:(math.dist(pos(prey[k]),pos(hunter)),k))
                causes[i]='capture';gain=min(12.,100.-hunter.body.energy)
                hunter.body.energy+=gain;gains[j]+=gain;captures.append([step,i]);contact[i]=0;events=True
            if step%20==0 or events:trace.append(snapshot())
    if trace[-1]['step']!=step:trace.append(snapshot())
    for i,x in enumerate(agents):
        assert abs(initial_energy[i]-costs[i]+gains[i]-x.body.energy)<1e-7
    assert len(captures)==len({v[1] for v in captures})
    assert sum(gains[:-1])<=12*len(initial_food)+1e-9
    return dict(seed=seed,index=index,scene=scene,arm=arm,steps=step,start=start,lamp=w.lamp if amp else None,rectangles=w.rectangles,initial_food=initial_food,causes=[x or 'censored' for x in causes],captures=captures,gains=gains,energy=[x.body.energy for x in agents],damage=damage,prey_survivors=sum(x is None for x in causes[:-1]),predator_survived=causes[-1] is None,trace=trace)
def shard(seed,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);summary=[]
    for arm in ARMS:
        with (out/f'{seed}-{arm}.jsonl.gz').open('wb') as f:
            with gzip.GzipFile(fileobj=f,mode='wb',mtime=0,filename='') as z:
                for scene in SCENARIOS:
                    for i in range(10):
                        ep=episode(seed,i,scene,arm)
                        z.write((json.dumps(ep,sort_keys=True,separators=(',',':'))+'\n').encode())
                        summary.append({k:v for k,v in ep.items() if k!='trace'})
        print(seed,arm,flush=True)
    (out/f'{seed}-summary.json').write_text(json.dumps(summary,sort_keys=True,separators=(',',':')))
def combine(directory):
    rows=[]
    for s in SEEDS:
        paths=list(Path(directory).rglob(f'{s}-summary.json'));assert len(paths)==1
        part=json.loads(paths[0].read_text())
        assert len(part)==90 and {(x['seed'],x['index'],x['scene'],x['arm']) for x in part}=={(s,i,p,a) for i in range(10) for p in SCENARIOS for a in ARMS}
        rows.extend(part)
    summary={}
    for a in ARMS:
        summary[a]={}
        for p in SCENARIOS:
            z=[x for x in rows if x['arm']==a and x['scene']==p];n=len(z);pop=1 if a=='one' else 2
            summary[a][p]=dict(episodes=n,prey_survival=sum(x['prey_survivors'] for x in z)/(n*pop),any_prey_survives=sum(x['prey_survivors']>0 for x in z)/n,first_prey_survival=sum(x['causes'][0]=='censored' for x in z)/n,predator_survival=sum(x['predator_survived'] for x in z)/n if a!='two_no_predator' else None,captures=sum(len(x['captures']) for x in z),double_captures=sum(len(x['captures'])==2 for x in z),prey_causes={c:sum(x['causes'][:-1].count(c) for x in z) for c in ('capture','starvation','injury','censored')},predator_causes={c:sum(x['causes'][-1]==c for x in z) for c in ('starvation','injury','censored','absent')},food_gain=sum(sum(x['gains'][:-1]) for x in z))
    return dict(summary=summary)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int);p.add_argument('--out');p.add_argument('--combine');args=p.parse_args()
    if args.combine:print(json.dumps(combine(args.combine),indent=2))
    else:
        assert args.seed in SEEDS
        shard(args.seed,args.out)
