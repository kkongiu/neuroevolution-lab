"""R24A: deterministic planar motor plant bench, no neural controller."""
import argparse,json,math
from dataclasses import dataclass,asdict
from pathlib import Path
from neuroevolution import flyloop_r1 as r1, flyloop_r12 as r12

@dataclass
class State:
    x:float=0.
    y:float=0.
    heading:float=0.
    vx:float=0.
    vy:float=0.
    omega:float=0.
    v:float=0.
    work:float=0.

@dataclass(frozen=True)
class Params:
    mass:float=1.
    inertia:float=.045
    drag:float=1.5
    yaw_drag:float=.15
    forward:float=1.8
    reverse:float=.45
    torque:float=.30
    lateral:float=.8

GROUND=Params()
AIR=Params(drag=.75,yaw_drag=.10,forward=.9,reverse=.9,torque=.20)
MODELS=('legacy','ground','air')

def step(s,cmd,model,dt):
    throttle,steer=cmd
    if not all(math.isfinite(x) and -1<=x<=1 for x in cmd):raise ValueError(cmd)
    if model=='legacy':
        assert abs(dt-r12.MICRO_DT)<1e-10
        alpha=r12.MICRO_ALPHA
        s.v+=(r1.V_MAX*throttle-s.v)*alpha
        s.omega+=(r1.W_MAX*math.tanh(1.6*steer)-s.omega)*alpha
        s.heading+=s.omega*dt
        s.vx=s.v*math.cos(s.heading);s.vy=s.v*math.sin(s.heading)
        s.x+=s.vx*dt;s.y+=s.vy*dt
        return dict(force=None,torque=None,lateral=abs(s.v*s.omega),signed=s.v)
    p=GROUND if model=='ground' else AIR if model=='air' else None
    if p is None:raise ValueError(model)
    torque=p.torque*steer
    oldvx,oldvy=s.vx,s.vy
    oldspeed=math.hypot(oldvx,oldvy)
    s.omega+=(torque-p.yaw_drag*s.omega)/p.inertia*dt
    if model=='ground':
        force=throttle*(p.forward if throttle>=0 else p.reverse)
        s.v+=(force-p.drag*s.v)/p.mass*dt
        bound=p.lateral/max(abs(s.v),1e-12)
        s.omega=max(-bound,min(bound,s.omega))
        s.heading+=s.omega*dt
        s.vx=s.v*math.cos(s.heading);s.vy=s.v*math.sin(s.heading)
        signed=s.v; lateral=abs(s.v*s.omega)
        s.work+=(abs(force*s.v)+abs(torque*s.omega))*dt
        magnitude=abs(force)
    else:
        s.heading+=s.omega*dt
        hx,hy=math.cos(s.heading),math.sin(s.heading)
        if throttle>=0:
            fx,fy=p.forward*throttle*hx,p.forward*throttle*hy
            # Lateral correction is a bounded force normal to actual velocity.
            if oldspeed>1e-12:
                nx,ny=-oldvy/oldspeed,oldvx/oldspeed
                side=p.lateral*(hx*nx+hy*ny)
                fx+=side*nx;fy+=side*ny
        elif oldspeed>1e-12:
            brake=min(-throttle*p.reverse,max(0.,p.mass*oldspeed/dt-p.drag*oldspeed))
            fx,fy=-brake*oldvx/oldspeed,-brake*oldvy/oldspeed
        else:fx=fy=0.
        # Bound net transverse force, including forward thrust.
        if oldspeed>1e-12:
            nx,ny=-oldvy/oldspeed,oldvx/oldspeed
            transverse=fx*nx+fy*ny
            correction=max(-p.lateral,min(p.lateral,transverse))-transverse
            fx+=correction*nx;fy+=correction*ny
            lateral=abs(fx*nx+fy*ny)/p.mass
        else:lateral=0.
        s.vx+=(fx-p.drag*oldvx)/p.mass*dt
        s.vy+=(fy-p.drag*oldvy)/p.mass*dt
        signed=s.vx*hx+s.vy*hy;s.v=signed
        magnitude=math.hypot(fx,fy)
        s.work+=(abs(fx*s.vx+fy*s.vy)+abs(torque*s.omega))*dt
    s.x+=s.vx*dt;s.y+=s.vy*dt
    return dict(force=magnitude,torque=torque,lateral=lateral,signed=signed)

def commands(name,t):
    if name=='rest':return (0.,0.)
    if name=='straight':return (1.,0.)
    if name=='stop':return (1. if t<3 else 0.,0.)
    if name=='reverse':return (1. if t<3 else -1.,0.)
    if name=='pivot':return (0.,1.)
    if name=='alternating':return (1.,1. if int(t*5)//5%2==0 else -1.)
    if name.startswith('turn'):
        return (float(name.split('_')[1]),float(name.split('_')[2]))
    raise ValueError(name)
CASES=('rest','straight','stop','reverse','pivot','alternating')+tuple(f'turn_{v}_{a}' for v in (.25,.5,1.) for a in (-1,1))

def run_case(model,name,dt=.005,duration=8):
    if model=='legacy':dt=r12.MICRO_DT
    steps=round(duration/dt);sample=round(.05/dt);hold=round(.2/dt)
    s=State();trace=[dict(t=0.,**asdict(s),force=0.,torque=0.,lateral=0.,signed=0.)]
    peak_lat=peak_force=peak_torque=0.;distance=reverse=rotation=0.
    for i in range(steps):
        cmd=commands(name,(i//hold)*.2)
        oldx,oldy,h=s.x,s.y,s.heading
        m=step(s,cmd,model,dt)
        peak_lat=max(peak_lat,m['lateral'])
        peak_force=max(peak_force,m['force'] or 0.)
        peak_torque=max(peak_torque,abs(m['torque'] or 0.))
        distance+=math.hypot(s.x-oldx,s.y-oldy)
        rotation+=abs(s.heading-h)
        reverse+=dt if m['signed']<-.02 else 0.
        if (i+1)%sample==0:trace.append(dict(t=round((i+1)*dt,9),**asdict(s),**m,command=cmd))
    return dict(model=model,case=name,dt=dt,final=asdict(s),distance=distance,rotation=rotation,reverse_seconds=reverse,peak_lateral=peak_lat,peak_force=peak_force if model!='legacy' else None,peak_torque=peak_torque if model!='legacy' else None,trace=trace)

def run():
    rows=[run_case(m,c) for m in MODELS for c in CASES]
    fine=[run_case(m,c,.0025) for m in MODELS[1:] for c in CASES]
    convergence=[]
    for x in fine:
        y=next(z for z in rows if (z['model'],z['case'])==(x['model'],x['case']))
        convergence.append(dict(model=x['model'],case=x['case'],position_error=math.hypot(x['final']['x']-y['final']['x'],x['final']['y']-y['final']['y']),heading_error=abs(x['final']['heading']-y['final']['heading'])))
    return dict(rows=rows,convergence=convergence)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    result=run();path=Path(a.out);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,sort_keys=True,separators=(',',':')))
    print(json.dumps(dict(summary=[{k:v for k,v in x.items() if k!='trace'} for x in result['rows']],convergence=result['convergence']),indent=2))
