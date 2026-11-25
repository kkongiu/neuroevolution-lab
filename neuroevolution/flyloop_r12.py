"""Motor-only diagnostic; no learned or scripted escape policy."""
import argparse
import copy
import json
import math
import random
from itertools import product
from pathlib import Path
from neuroevolution import flyloop_r1 as r1, flyloop_r9 as r9, flyloop_r11 as r11

PLANTS = ("forward", "brake", "reverse")
GRID = tuple(product((-1., -.5, 0., .5, 1.), repeat=2))
SUBSTEPS, MICRO_DT, MICRO_ALPHA = r9.SUBSTEPS, r9.MICRO_DT, r9.MICRO_ALPHA

def target_speed(plant, throttle):
    if not math.isfinite(throttle) or not -1 <= throttle <= 1:
        raise ValueError("Invalid throttle")
    if plant == "forward":
        return r1.V_MAX * r1.sigmoid(1.6 * throttle)
    if plant == "brake":
        return r1.V_MAX * max(0., throttle)
    if plant == "reverse":
        return r1.V_MAX * throttle
    raise ValueError("Unknown plant")

class World(r11.World):
    def __init__(self, seed, geometry, regime="static", plant="forward"):
        super().__init__(seed, geometry, regime)
        if plant not in PLANTS:
            raise ValueError("Unknown plant")
        self.plant = plant

    def micro_integrate(self,command):
        """One ACTUAL .05s physical integration, same for all five arms.

        New micro-discretization is NOT identical to original .20s R1
        integration; collision damage/energy scaled by 1/4 for duration.
        """
        b=self.body
        if b.energy<=0 or b.integrity<=0:
            raise AssertionError("Dead body cannot be integrated")
        throttle,steer=command
        if not all(math.isfinite(v) and -1.<=v<=1. for v in command):
            raise ValueError("Out-of-range motor command")
        before=(b.x,b.y,b.heading,b.speed,b.turn,b.energy,b.integrity)
        wanted_speed=target_speed(self.plant, throttle)
        wanted_turn=r1.W_MAX*math.tanh(1.6*steer)
        b.speed=(1-MICRO_ALPHA)*b.speed+MICRO_ALPHA*wanted_speed
        b.turn=(1-MICRO_ALPHA)*b.turn+MICRO_ALPHA*wanted_turn
        b.heading=(b.heading+b.turn*MICRO_DT+math.pi)%(2*math.pi)-math.pi
        nx=b.x+b.speed*math.cos(b.heading)*MICRO_DT
        ny=b.y+b.speed*math.sin(b.heading)*MICRO_DT
        hit=int(self.blocked(nx,ny))
        if hit:
            self.collisions+=1
            b.integrity=max(0.,b.integrity-r1.COLLISION_INJURY/SUBSTEPS)
        else:
            b.x,b.y=nx,ny
        b.energy=max(0.,b.energy-r1.ENERGY_PER_TICK/SUBSTEPS)
        b.prev_collision=hit
        moved=math.hypot(b.x-before[0],b.y-before[1])
        self.distance+=moved
        self.visited.add((int(b.x),int(b.y)))
        self.frame+=1
        return {
            "collision":hit,
            "moved":round(moved,9),
            "position":[round(b.x,9),round(b.y,9)],
            "heading":round(b.heading,9),
            "speed":round(b.speed,9),
            "turn":round(b.turn,9),
            "energy":round(b.energy,9),
            "integrity":round(b.integrity,9),
            "before_position":[round(before[0],9),round(before[1],9)],
            "before_heading":round(before[2],9),
        }


def snapshots(seed, geometry):
    world = World(seed, geometry)
    rng = random.Random(seed * 48131 + 12012)
    previous = False
    count = 0
    for tick in range(640):
        if world.body.energy <= 0 or world.body.integrity <= 0:
            break
        if tick % 4 == 0:
            cmd = tuple(rng.uniform(-1, 1) for _ in range(2))
        hit = bool(world.micro_integrate(cmd)["collision"])
        if hit and not previous:
            yield copy.deepcopy(world)
            count += 1
            if count == 3:
                break
        previous = hit

def probe(snapshot, plant, command):
    world = copy.deepcopy(snapshot)
    world.plant = plant
    energy, integrity = world.body.energy, world.body.integrity
    start_distance = world.distance
    steps, run, earliest = [], [], None
    for i in range(32):
        if world.body.energy <= 0 or world.body.integrity <= 0:
            break
        step = world.micro_integrate(command)
        steps.append(step)
        run = [] if step["collision"] else (run + [step["moved"]])[-4:]
        if earliest is None and len(run) == 4 and sum(run) >= .035:
            earliest = i + 1
    return dict(plant=plant, command=command, success=earliest is not None,
                earliest=earliest, collisions=sum(s["collision"] for s in steps),
                distance=world.distance-start_distance, energy_cost=energy-world.body.energy,
                integrity_cost=integrity-world.body.integrity, steps=steps)

def run():
    rows, counts = [], []
    for geometry in r11.GEOMETRIES:
        for seed in r11.SEEDS:
            n = 0
            for snap in snapshots(seed, geometry):
                n += 1
                b = snap.body
                rows.append(dict(seed=seed, geometry=geometry, onset=snap.frame,
                    state={k:getattr(b,k) for k in ("x","y","heading","speed","turn","energy","integrity","prev_collision")},
                    rectangles=snap.rectangles,
                    probes=[probe(snap,p,c) for p in PLANTS for c in GRID]))
            counts.append(dict(seed=seed, geometry=geometry, snapshots=n))
    summary = {}
    for plant in PLANTS:
        trials = [p for r in rows for p in r["probes"] if p["plant"] == plant]
        recoverable = sum(any(p["success"] for p in r["probes"] if p["plant"]==plant) for r in rows)
        summary[plant] = dict(trials=len(trials), successes=sum(p["success"] for p in trials),
            success_fraction=sum(p["success"] for p in trials)/len(trials) if trials else None,
            recoverable_snapshots=recoverable, fraction=recoverable/len(rows) if rows else None)
    gate = len(rows)>=12 and len({r["seed"] for r in rows})>=6 and (
        summary["reverse"]["fraction"]-summary["forward"]["fraction"] >= .20)
    return dict(snapshot_counts=counts, snapshots=rows, summary=summary, advancement_gate=gate)

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out", default="results/r12.json")
    args=parser.parse_args()
    result=run()
    path=Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, sort_keys=True, separators=(",",":")))
    print(json.dumps({"summary":result["summary"],"advancement_gate":result["advancement_gate"]},indent=2))
