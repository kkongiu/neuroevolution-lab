"""M5.8: reproducible progressive worlds and structured exogenous shocks.

STANDALONE environment-generation and audit layer. This module does not
control agents, teach correct actions or alter M5.7 physics yet. Shocks are
generated solely from world seed, tick and zone, never from agent behavior.

Rules remain stable across worlds; layouts, food placement, order of shocks
and temporary routes differ. A guaranteed north-south and east-west spine
prevents blocking every route; all initially accessible cells are connected.
No final/source/replication world identifiers are ever used by this module.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path

WIDTH = 11
INTERIOR = range(1, WIDTH - 1)
TICKS = 60
DEV_SEEDS = tuple(range(2000, 2064))
TRANSFER_SEEDS = tuple(range(2064, 2096))  # only a future, separate development cohort
ALLOWED_SEEDS = DEV_SEEDS + TRANSFER_SEEDS
ZONES = ("shelter", "frontier", "wild")
EVENT_TYPES = ("drought", "detour", "predator_shift", "compound")
ZONE_FOOD = (4, 3, 2)
ZONE_EVENT_PROB = (.03, .10, .18)
# Reserved for later experiment, NOT delivered as free observations to an agent.


@dataclass(frozen=True)
class Shock:
    tick: int
    kind: str
    zone: int
    duration: int
    location: tuple[int, int] | None
    intensity: float

    def as_dict(self) -> dict:
        return {"tick": self.tick, "kind": self.kind, "zone": self.zone,
                "duration": self.duration,
                "location": list(self.location) if self.location is not None else None,
                "intensity": self.intensity}


def region(cell: tuple[int, int]) -> int:
    x, y = cell
    if x not in INTERIOR or y not in INTERIOR:
        raise ValueError("Cell is outside interior")
    return min(2, (x - 1) // 3)


def reachable(cells: set[tuple[int, int]],
              start: tuple[int, int] = (5, 5)) -> set[tuple[int, int]]:
    if start not in cells:
        return set()
    discovered = {start}
    queue = deque([start])
    while queue:
        x, y = queue.popleft()
        for p in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
            if p in cells and p not in discovered:
                discovered.add(p)
                queue.append(p)
    return discovered


def make_world(seed: int, *, ticks: int = TICKS) -> dict:
    """Generate all random events *upfront*, independent of future actions."""
    if type(seed) is not int or seed not in ALLOWED_SEEDS:
        raise ValueError("Only M5.8 development or reserved novel-world check seeds")
    if type(ticks) is not int or not 1 <= ticks <= TICKS:
        raise ValueError("Invalid development horizon")
    rng = random.Random(1_300_031*seed + 580_017)
    interior = {(x, y) for x in INTERIOR for y in INTERIOR}
    # The backbone is accessible for every seed, with links across all zones.
    backbone = {(x, 5) for x in INTERIOR} | {(5, y) for y in INTERIOR}
    # Zone-dependent obstacle rate: progressively more cluttered territory.
    # Rejection sampling only blocks cells when remaining space is connected.
    obstacles = set()
    for x, y in sorted(interior - backbone):
        if rng.random() < (.04, .14, .25)[region((x, y))]:
            candidate = obstacles | {(x, y)}
            open_cells = interior - candidate
            if reachable(open_cells) == open_cells:
                obstacles = candidate
    open_cells = interior - obstacles
    if reachable(open_cells) != open_cells:
        raise AssertionError("Generated inaccessible cell")
    food = []
    for zone, count in enumerate(ZONE_FOOD):
        candidates = sorted(p for p in open_cells
                            if region(p) == zone and p != (5, 5))
        if len(candidates) < count:
            raise AssertionError("Insufficient cells for zone food")
        food.extend(rng.sample(candidates, count))
    food = sorted(food)
    shocks = []
    for tick in range(1, ticks + 1):
        for z in range(3):
            if rng.random() >= ZONE_EVENT_PROB[z]:
                continue
            kind = rng.choice(EVENT_TYPES)
            duration = rng.randint(2, 5)
            candidates = sorted(c for c in open_cells
                                if region(c) == z and c not in backbone
                                and c not in food)
            target = rng.choice(candidates) if candidates else None
            # detour must never block a permanent link or currently placed
            # resource; placement is an event *proposal* for the later adapter.
            if kind in ("detour", "compound") and target is None:
                kind = "drought"
            shocks.append(Shock(tick, kind, z, duration, target,
                                round(rng.uniform(.2, .7), 4)).as_dict())
    result = {
        "protocol": "m5.8-development-generator-v1",
        "seed": seed, "ticks": ticks,
        "size": WIDTH, "zones": list(ZONES),
        "walls": [list(p) for p in sorted(obstacles)],
        "food": [list(p) for p in food],
        "backbone": [list(p) for p in sorted(backbone)],
        "events": shocks,
        "parameters": {
            "zone_food": list(ZONE_FOOD),
            "zone_event_probability": list(ZONE_EVENT_PROB),
            "zone_obstacle_probability": [.04, .14, .25],
            "events_exogenous": True,
            "events_precomputed": True,
            "event_proposals_not_applied_to_m5_7": True,
            "spatial_progression_not_level_unlocks": True
        },
    }
    data = json.dumps(result, sort_keys=True, separators=(",", ":"))
    result["world_sha256"] = hashlib.sha256(data.encode()).hexdigest()
    return result


def audit(seeds: tuple[int, ...] = DEV_SEEDS) -> dict:
    if not seeds or len(seeds) != len(set(seeds)) or any(
            type(s) is not int or s not in DEV_SEEDS for s in seeds):
        raise ValueError("Audit must use unique development seeds only")
    worlds = [make_world(s) for s in seeds]
    counts = [0, 0, 0]
    events = [0, 0, 0]
    known = set()
    for world in worlds:
        if world["world_sha256"] in known:
            raise AssertionError("Duplicate world")
        known.add(world["world_sha256"])
        cells = {(x, y) for x in INTERIOR for y in INTERIOR}
        walls = {tuple(x) for x in world["walls"]}
        opened = cells - walls
        backbone = {tuple(x) for x in world["backbone"]}
        food = [tuple(x) for x in world["food"]]
        if not backbone <= opened or reachable(opened) != opened:
            raise AssertionError("World not navigable")
        if len(food) != 9 or len(set(food)) != 9 or set(food) & walls:
            raise AssertionError("Food constraints broken")
        for z in range(3):
            if sum(region(p) == z for p in food) != ZONE_FOOD[z]:
                raise AssertionError("Zone food allocation violated")
            counts[z] += sum(region(p) == z for p in walls)
            events[z] += sum(e["zone"] == z for e in world["events"])
        for e in world["events"]:
            if not 1 <= e["tick"] <= world["ticks"]:
                raise AssertionError("Event outside simulation")
            if e["kind"] not in EVENT_TYPES:
                raise AssertionError("Unknown exogenous event")
            if e["location"] is not None and (
                    tuple(e["location"]) in backbone or
                    tuple(e["location"]) in food):
                raise AssertionError("A shock targets protected resource/path")
    if not events[0] < events[1] < events[2]:
        raise AssertionError("Generated shock frequency did not progress by zone")
    summary = {
        "status": "world_generator_only_not_connected_to_ecology",
        "world_count": len(worlds),
        "world_seeds": list(seeds),
        "independent_world_shas": len(known),
        "mean_obstacles_by_zone": [n/len(seeds) for n in counts],
        "events_by_zone": events,
        "event_proposals_total": sum(events),
        "worlds_digest": hashlib.sha256(
            json.dumps(worlds, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    return {"summary": summary, "worlds": worlds}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path,
                        default=Path("results/m5-8/world-generator-development.json"))
    args = parser.parse_args()
    report = audit()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n",
                        encoding="utf-8")
    print("M5_8_WORLD_AUDIT "+json.dumps(report["summary"], sort_keys=True),
          flush=True)


if __name__ == "__main__":
    main()
