"""Sensory extension preserving R12 mechanics; no automatic motor policy."""
import math
from neuroevolution import flyloop_r1 as r1, flyloop_r9 as r9, flyloop_r12 as r12

EPS = 1e-9
EXTRA_NAMES = (
    "rear", "rear_positive", "rear_negative",
    *("touch_%d" % i for i in range(8)),
    *("contact_duration_%d" % i for i in range(8)),
    "actual_signed_speed", "target_speed", "blocked_fraction",
    "injury_increment", "remaining_integrity",
)
SLOW_INPUTS = 8 + len(EXTRA_NAMES)
FAST_INPUTS = 7 + len(EXTRA_NAMES)

def contact_sectors(start, end, heading, rectangles):
    """First entering faces of the original inflated collision geometry."""
    x,y = start
    dx,dy = end[0]-x,end[1]-y
    events = []
    lo,hi = r1.RADIUS,r1.SIDE-r1.RADIUS
    for origin, delta, normal_lo, normal_hi in (
        (x,dx,(-1,0),(1,0)), (y,dy,(0,-1),(0,1))
    ):
        if delta < 0 and origin+delta < lo:
            events.append(((lo-origin)/delta,normal_lo))
        if delta > 0 and origin+delta > hi:
            events.append(((hi-origin)/delta,normal_hi))
    for x1,y1,x2,y2 in rectangles:
        enter,leave = -math.inf,math.inf
        faces = []
        valid = True
        for origin,delta,a,b,nlo,nhi in (
            (x,dx,x1-r1.RADIUS,x2+r1.RADIUS,(1,0),(-1,0)),
            (y,dy,y1-r1.RADIUS,y2+r1.RADIUS,(0,1),(0,-1))
        ):
            if abs(delta)<1e-15:
                if not a<=origin<=b:
                    valid=False
                    break
                continue
            t0,t1=(a-origin)/delta,(b-origin)/delta
            normal=nlo if delta>0 else nhi
            if t0>t1:
                t0,t1=t1,t0
            if t0>enter+EPS:
                enter,faces=t0,[normal]
            elif abs(t0-enter)<=EPS:
                faces.append(normal)
            leave=min(leave,t1)
        if valid and enter<=leave+EPS and -EPS<=enter<=1+EPS and leave>=0:
            events.extend((max(0.,enter),n) for n in faces)
    sectors=[0.]*8
    if events:
        first=min(t for t,_ in events)
        for t,(nx,ny) in events:
            if abs(t-first)<=EPS:
                angle=(math.atan2(ny,nx)-heading)%(2*math.pi)
                sector=int(math.floor(angle/(math.pi/4)+.5))%8
                sectors[sector]=1.
    return tuple(sectors)

class World(r12.World):
    def __init__(self,seed,geometry,regime="static",plant="reverse"):
        super().__init__(seed,geometry,regime,plant)
        self.touch=(0.,)*8
        self.duration=(0,)*8
        self.actual_speed=0.
        self.motor_target=0.
        self.blocked_fraction=0.
        self.injury=0.

    def micro_integrate(self,command):
        b=self.body
        x,y,integrity=b.x,b.y,b.integrity
        result=super().micro_integrate(command)
        dx=b.speed*math.cos(b.heading)*r12.MICRO_DT
        dy=b.speed*math.sin(b.heading)*r12.MICRO_DT
        self.touch=contact_sectors((x,y),(x+dx,y+dy),b.heading,self.rectangles) if result["collision"] else (0.,)*8
        self.duration=tuple(min(32,d+1) if t else 0 for d,t in zip(self.duration,self.touch))
        self.actual_speed=((b.x-x)*math.cos(b.heading)+(b.y-y)*math.sin(b.heading))/(r12.MICRO_DT*r1.V_MAX)
        self.motor_target=r12.target_speed(self.plant,command[0])/r1.V_MAX
        intended=math.hypot(dx,dy)
        actual=math.hypot(b.x-x,b.y-y)
        self.blocked_fraction=r1.clamp(1-actual/intended,0.,1.) if intended>EPS else 0.
        self.injury=r1.clamp((integrity-b.integrity)/(r1.COLLISION_INJURY/r12.SUBSTEPS),0.,1.)
        return result

    def extra_senses(self):
        rear=tuple(r1.clamp(1-self.ray_distance(self.body.heading+math.pi+a)/r1.RAY_LENGTH,0.,1.)
                   for a in (0.,r1.SENSOR_ANGLE,-r1.SENSOR_ANGLE))
        return (*rear,*self.touch,*(d/32 for d in self.duration),
                r1.clamp(self.actual_speed,-1.,1.),self.motor_target,
                self.blocked_fraction,self.injury,self.body.integrity/100.)

    def perceive_extended(self):
        return (*super().perceive(),*self.extra_senses())

    def perceive_fast_extended(self):
        return (*r9.local_fast_sensors(super().perceive()),*self.extra_senses())

class SlowCircuit(r1.Circuit):
    """Same 16 units; zero-start extra afferents, no policy change at init."""
    def __init__(self,seed,arm="recurrent"):
        if arm!="recurrent":
            raise ValueError("R12C adapter supports recurrent arm only")
        super().__init__(seed,arm)
        self.extra_weights=[[0.]*len(EXTRA_NAMES) for _ in range(r1.N)]

    def forward(self,observed):
        if len(observed)!=SLOW_INPUTS:
            raise ValueError("Wrong slow sensory dimension")
        base,extra=observed[:8],observed[8:]
        old=self.h
        self.h=[(1-r1.ALPHA)*old[j]+r1.ALPHA*math.tanh(
            r1.dot(self.weights["w_input"][j],base)+self.weights["bias"][j]
            +r1.dot(self.weights["w_recurrent"][j],old)+r1.dot(self.extra_weights[j],extra))
            for j in range(r1.N)]
        motor=tuple(r1.clamp(r1.dot(self.weights["w_motor"][i],self.h)+self.weights["motor_bias"][i],-1.,1.) for i in range(r1.OUTPUTS))
        return motor,tuple(self.h),tuple(observed)

class FastCircuit(r9.FastCircuit):
    """Same four units and motor outputs; zero-start extra afferents."""
    def __init__(self,seed):
        super().__init__(seed)
        self.extra_weights=[[0.]*len(EXTRA_NAMES) for _ in range(r9.FAST_NEURONS)]

    def observe(self,observed):
        if len(observed)!=FAST_INPUTS:
            raise ValueError("Wrong fast sensory dimension")
        base,extra=observed[:7],observed[7:]
        old=self.h
        self.h=[(1-r9.FAST_STATE_ALPHA)*old[j]+r9.FAST_STATE_ALPHA*math.tanh(
            r1.dot(self.input[j],base)+r1.dot(self.recurrent[j],old)+self.bias[j]
            +r1.dot(self.extra_weights[j],extra)) for j in range(r9.FAST_NEURONS)]
        return tuple(self.h)
