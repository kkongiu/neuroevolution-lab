"""R21A: paired predator-prey experiment, frozen circuits."""
import argparse, gzip, json, math, random
from pathlib import Path
from neuroevolution import flyloop_r1 as r1, flyloop_r12 as r12, flyloop_r18a as r, flyloop_r19d as d, flyloop_r20b as b, flyloop_r20c as c

SEEDS=range(30100,30124)
ARMS=("absent","blind","escape","unoccluded")
SCENARIOS=("free","obstacle","lantern")
ANGLES=tuple(i*math.tau/12 for i in range(12))
DT=r12.MICRO_DT

def blocked(x,y,rects,radius):
    return not(radius<=x<=r1.SIDE-radius and radius<=y<=r1.SIDE-radius) or any(x0-radius<=x<=x1+radius and y0-radius<=y<=y1+radius for x0,y0,x1,y1 in rects)

def visible(p,q,rects):
    dx,dy=q[0]-p[0],q[1]-p[1]
    dist=math.hypot(dx,dy)
    return dist<1e-12 or not any(r1.World._ray_box(*p,dx/dist,dy/dist,*rect)<dist for rect in rects)

def sense(body,target,rects,reach,occluded=True):
    p=(body.x,body.y); q=(target.x,target.y)
    distance=math.dist(p,q)
    values=[0.]*12
    if distance<reach and (not occluded or visible(p,q,rects)):
        angle=r.wrap(math.atan2(q[1]-p[1],q[0]-p[0])-body.heading)
        i=int(math.floor(angle/(math.tau/12)+.5))%12
        values[i]=1-distance/reach
    return tuple(values)

def bearing(values):
    return math.atan2(sum(v*math.sin(a) for a,v in zip(ANGLES,values)),sum(v*math.cos(a) for a,v in zip(ANGLES,values)))

def flight(base,threat,prox,eye):
    u=min(1.,2*max(threat))
    if not u:return base,0.
    danger=bearing(threat)
    # Select locally accessible forward headings away from danger.
    scores=[-math.cos(a-danger)-2*p for a,p in zip(eye.centers,prox)]
    weights=[math.exp(8*(s-max(scores))) for s in scores]
    turn=math.tanh(4*sum(a*w for a,w in zip(eye.centers,weights))/sum(weights))
    # Exact front symmetry: use the existing motor tendency to break tie.
    if abs(turn)<.05 and math.cos(danger)>.5:
        turn=1. if base[1]>=0 else -1.
    return ((1-u)*base[0]+u,(1-u)*base[1]+u*turn),u

def predator_proximity(body,rects):
    out=[]
    for offset in (0.,r1.SENSOR_ANGLE,-r1.SENSOR_ANGLE):
        a=body.heading+offset; ux,uy=math.cos(a),math.sin(a)
        ds=[3.5]
        if ux>1e-12:ds.append((r1.SIDE-.45-body.x)/ux)
        if ux<-1e-12:ds.append((.45-body.x)/ux)
        if uy>1e-12:ds.append((r1.SIDE-.45-body.y)/uy)
        if uy<-1e-12:ds.append((.45-body.y)/uy)
        ds.extend(r1.World._ray_box(body.x,body.y,ux,uy,x0-.45,y0-.45,x1+.45,y1+.45) for x0,y0,x1,y1 in rects)
        out.append(r1.clamp(1-min(ds)/3.5,0.,1.))
    return tuple(out)

def predator_command(p,brain,prey_signal,rects):
    prox=predator_proximity(p,rects)
    base,_,_=brain.forward((1.,*prox,p.speed/.95,p.turn/r1.W_MAX,float(p.prev_collision),1-p.energy/100))
    strength=min(1.,2*(1-p.energy/100)*max(prey_signal))
    angle=bearing(prey_signal)
    throttle=(1-strength)*base[0]+strength
    turn=(1-strength)*base[1]+strength*math.tanh(4*angle)
    return (r1.clamp(throttle*(1-.8*prox[0]),-1.,1.),r1.clamp(turn+.8*(prox[2]-prox[1]),-1.,1.))

def move_predator(p,cmd,rects):
    alpha=r12.MICRO_ALPHA
    p.speed=(1-alpha)*p.speed+alpha*.95*cmd[0]
    p.turn=(1-alpha)*p.turn+alpha*r1.W_MAX*math.tanh(1.6*cmd[1])
    p.heading=r.wrap(p.heading+p.turn*DT)
    x=p.x+p.speed*math.cos(p.heading)*DT;y=p.y+p.speed*math.sin(p.heading)*DT
    p.prev_collision=int(blocked(x,y,rects,.45))
    if not p.prev_collision:p.x,p.y=x,y
    p.energy=max(0.,p.energy-.05)

def world(seed,index,scenario):
    w=r.WindowWorld(seed,index,"free")
    rng=random.Random(seed*17311+index*7919+2121)
    heading=rng.uniform(-math.pi,math.pi)
    w.rectangles=[]
    if scenario=="obstacle":
        w.rectangles=[(10.,9.,12.,13.)]
        heading=rng.uniform(-.65,.65)
        w.body.x,w.body.y=7.8,rng.uniform(8.4,13.6)
    else:
        w.body.x,w.body.y=11.,11.
    w.body.heading=heading
    w.body.energy=90.
    p=r1.Body(x=w.body.x-2.5*math.cos(heading),y=w.body.y-2.5*math.sin(heading),heading=heading,energy=50.)
    w.lamp=(w.body.x+1.37*math.cos(heading),w.body.y+1.37*math.sin(heading))
    source=(w.body.x+3.5*math.cos(heading+1.4),w.body.y+3.5*math.sin(heading+1.4))
    w.initial_food=(source,) if scenario=="lantern" else ()
    w.food=list(w.initial_food)
    w.visited={(int(w.body.x),int(w.body.y))}
    assert not w.blocked(w.body.x,w.body.y) and not blocked(p.x,p.y,w.rectangles,.45)
    return w,p

def episode(seed,index,scenario,arm,horizon=200):
    w,p=world(seed,index,scenario); eye=r.CompoundEye()
    brain=r1.Circuit(seed,"recurrent"); hunter=r1.Circuit(seed+100000,"recurrent")
    start=(w.body.x,w.body.y,w.body.heading); predator_start=(p.x,p.y,p.heading)
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
            cmd,urgency=c.protect(cmd,temp,sensors,"priority")
            step=w.micro_integrate(cmd)
            if active:
                move_predator(p,pc,w.rectangles); predator_contacts+=p.prev_collision
            actual=c.heat(w,w.body.x,w.body.y,amplitude)
            loss=min(w.body.integrity,8*max(0.,actual-.65)*DT)
            w.body.integrity-=loss;thermal+=loss;food_gain+=step["food_gain"]
            q=(w.body.x,w.body.y); pp=(p.x,p.y)
            distance=math.dist(q,pp)
            if active:
                min_distance=min(min_distance,distance)
                contact=contact+1 if distance<=.75 and visible(q,pp,w.rectangles) else 0
                if contact>=4:
                    captured=True;p.energy=min(100.,p.energy+12)
            else:contact=0
            trace.append(dict(step,predator=list(pp) if arm!="absent" else None,threat=threat,escape=u,heat=actual,integrity=w.body.integrity,captured=captured))
    assert abs(90-sum(x["metabolic_cost"] for x in trace)+food_gain-w.body.energy)<1e-7
    return dict(seed=seed,index=index,scenario=scenario,arm=arm,start=start,predator_start=predator_start,lamp=w.lamp if scenario=="lantern" else None,food=w.initial_food,rectangles=w.rectangles,captured=captured,survived=not captured and w.body.energy>0 and w.body.integrity>0,steps=len(trace),capture_seconds=len(trace)*DT if captured else None,food_gain=food_gain,ate=food_gain>0,thermal_damage=thermal,escape_fraction=escape_steps/max(1,len(trace)),seen_fraction=seen/max(1,len(trace)),min_distance=min_distance if arm!="absent" else None,contacts=sum(x["collision"] for x in trace),predator_contacts=predator_contacts,trace=trace)

def aggregate(rows):
    n=len(rows)
    return dict(episodes=n,**{k:sum(x[k] for x in rows)/n for k in ("captured","survived","ate","food_gain","thermal_damage","escape_fraction","seen_fraction","contacts","predator_contacts")})

def shard(seed,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);summary=[]
    for arm in ARMS:
        with (out/f"{seed}-{arm}.jsonl.gz").open("wb") as f:
            with gzip.GzipFile(fileobj=f,mode="wb",mtime=0,filename="") as z:
                for scenario in SCENARIOS:
                    for i in range(20):
                        ep=episode(seed,i,scenario,arm);trace=ep.pop("trace")
                        ep["trace"]=[dict(step=j+1,**{k:v[k] for k in ("position","predator","energy","integrity","escape","captured","food_gain","heat")}) for j,v in enumerate(trace) if j%4==3 or j==len(trace)-1 or v["food_gain"]>0]
                        z.write((json.dumps(ep,sort_keys=True,separators=(",",":"))+"\n").encode())
                        summary.append({k:v for k,v in ep.items() if k!="trace"})
        print(seed,arm,flush=True)
    (out/f"{seed}-summary.json").write_text(json.dumps(summary,sort_keys=True,separators=(",",":")))

def combine(directory):
    rows=[]
    for s in SEEDS:
        paths=list(Path(directory).rglob(f"{s}-summary.json"));assert len(paths)==1
        part=json.loads(paths[0].read_text())
        assert len(part)==240 and {(x["seed"],x["index"],x["scenario"],x["arm"]) for x in part}=={(s,i,p,a) for i in range(20) for p in SCENARIOS for a in ARMS}
        rows.extend(part)
    def scope(a,p,s=None):return [x for x in rows if x["arm"]==a and x["scenario"]==p and (s is None or x["seed"]==s)]
    summary={a:{p:aggregate(scope(a,p)) for p in SCENARIOS} for a in ARMS}
    e=summary["escape"];b=summary["blind"];a=summary["absent"]
    gates=dict(free=e["free"]["survived"]-b["free"]["survived"]>=.1,obstacle=e["obstacle"]["survived"]-b["obstacle"]["survived"]>=.1,lantern=e["lantern"]["captured"]<b["lantern"]["captured"],food=a["lantern"]["ate"]>0 and e["lantern"]["ate"]>=.5*a["lantern"]["ate"],thermal=e["lantern"]["thermal_damage"]<=a["lantern"]["thermal_damage"]+1,absent_zero=all(not x["captured"] for x in rows if x["arm"]=="absent"))
    intervals={}
    for p in SCENARIOS:
        delta=[aggregate(scope("escape",p,s))["survived"]-aggregate(scope("blind",p,s))["survived"] for s in SEEDS]
        rng=random.Random(301301);v=sorted(sum(rng.choice(delta) for _ in delta)/len(delta) for _ in range(2000))
        intervals[p]=[v[50],v[1949]]
    return dict(summary=summary,gates=gates,passed=all(gates.values()),survival_intervals=intervals)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--seed",type=int);p.add_argument("--out");p.add_argument("--combine");args=p.parse_args()
    if args.combine:print(json.dumps(combine(args.combine),indent=2))
    else:
        assert args.seed in SEEDS
        shard(args.seed,args.out)
