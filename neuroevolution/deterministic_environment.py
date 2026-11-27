"""Pure, precomputed external events; no dependency on organisms."""
from dataclasses import dataclass, asdict
import hashlib
import json
import random
import math

@dataclass(frozen=True)
class FoodEvent:
    tick: int
    resource_id: int
    position: tuple[float,float]
    energy: float

@dataclass(frozen=True)
class EnvironmentPlan:
    environment_seed: int
    horizon: int
    interval: int
    side: float
    clearance: float
    rectangles: tuple
    events: tuple[FoodEvent,...]

    def canonical(self):
        return json.dumps(asdict(self),sort_keys=True,separators=(',',':'),allow_nan=False)
    def digest(self):
        return hashlib.sha256(self.canonical().encode('utf-8')).hexdigest()

def make_plan(environment_seed, horizon, interval=200, side=22., clearance=.55, rectangles=(), energy=12.):
    if not all(type(v) is int for v in (environment_seed,horizon,interval)) or horizon<0 or interval<=0:
        raise ValueError('integer seed and valid tick limits required')
    if not all(math.isfinite(v) for v in (side,clearance,energy)) or not 0<=clearance<side/2 or energy<=0:
        raise ValueError('invalid geometry or energy')
    rects=tuple(sorted(tuple(float(v) for v in q) for q in rectangles))
    if any(len(q)!=4 or not all(math.isfinite(v) for v in q) or q[0]>q[2] or q[1]>q[3] for q in rects):
        raise ValueError('invalid rectangle')
    rng=random.Random(environment_seed)
    events=[]
    for tick in range(0,horizon,interval):
        for _ in range(10000):
            x,y=(round(rng.uniform(clearance,side-clearance),9) for _ in range(2))
            if not any(a-clearance<=x<=c+clearance and b-clearance<=y<=d+clearance for a,b,c,d in rects):
                break
        else:
            raise ValueError('no valid resource location')
        events.append(FoodEvent(tick,len(events),(x,y),float(energy)))
    return EnvironmentPlan(environment_seed,horizon,interval,float(side),float(clearance),rects,tuple(events))

class Inventory:
    """Apply schedule once per consecutive integer tick; consumption is separate."""
    def __init__(self,plan):
        self.plan=plan
        self.tick=-1
        self.cursor=0
        self.available={}
    def advance(self,tick):
        if tick!=self.tick+1 or tick>=self.plan.horizon:
            raise ValueError('expected consecutive in-range tick')
        self.tick=tick
        births=[]
        while self.cursor<len(self.plan.events) and self.plan.events[self.cursor].tick==tick:
            event=self.plan.events[self.cursor]
            self.available[event.resource_id]=event
            births.append(event)
            self.cursor+=1
        return tuple(births)
    def consume(self,resource_id):
        return self.available.pop(resource_id)
