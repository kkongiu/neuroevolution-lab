"""M5.8 matched-budget counterfactual spatial curricula.

Variants are generated from the same canonical world seed:
- progressive: native shelter/frontier/wild increasing obstacles/shocks.
- uniform: SAME number of obstacles, food resources and scheduled events,
  redistributed approximately uniformly across three spatial zones.
- reversed: x-reflection of progressive physical geometry + exogenous shocks;
  easy/hard zones inverted while conserving all totals.

Shocks are all precomputed; neither variant sees an organism or reward.
Seed 2000–2063 and generator's original protocol remain unchanged.
"""
from __future__ import annotations

import hashlib
import json
import random

from .m5_8_worlds import (
    DEV_SEEDS, EVENT_TYPES, INTERIOR, WIDTH, make_world, reachable, region,
)

VARIANTS = ("progressive", "uniform", "reversed")
SPINE = {(x,5) for x in INTERIOR} | {(5,y) for y in INTERIOR}
INTERIOR_CELLS = {(x,y) for x in INTERIOR for y in INTERIOR}


def variant_world(seed: int, variant: str) -> dict:
    if variant not in VARIANTS:
        raise ValueError("Unregistered experiment variant")
    source = make_world(seed)
    if variant == "progressive":
        return source

    world = dict(source)
    rng = random.Random(3_330_091*seed + 5808)
    old_walls = [tuple(p) for p in source["walls"]]
    old_food = [tuple(p) for p in source["food"]]
    events = [dict(e) for e in source["events"]]

    if variant == "reversed":
        def flip(p):
            return (WIDTH-1-p[0],p[1])
        walls = {flip(p) for p in old_walls}
        food = {flip(p) for p in old_food}
        for e in events:
            e["zone"] = 2-e["zone"]
            if e["location"] is not None:
                e["location"] = list(flip(e["location"]))
    else:
        # Matched total static burden, non-preselected route locations.
        target_count = len(old_walls)
        counts = [target_count//3+int(z < target_count%3) for z in range(3)]
        walls = set()
        for z, n in enumerate(counts):
            candidates = sorted(p for p in INTERIOR_CELLS-SPINE
                                if region(p) == z)
            rng.shuffle(candidates)
            accepted = 0
            for p in candidates:
                candidate = walls|{p}
                remaining = INTERIOR_CELLS-candidate
                if reachable(remaining) == remaining:
                    walls = candidate
                    accepted += 1
                    if accepted == n:
                        break
            if accepted < n:
                raise AssertionError("Cannot balance obstacle budget")
        available = INTERIOR_CELLS-walls
        food = set()
        for z in range(3):
            candidates = sorted(p for p in available
                                if region(p) == z and p != (5,5))
            food.update(rng.sample(candidates,3))
        # Equalize event-zone exposure while retaining the complete original
        # temporal event list, event types, durations, and intensities.
        # This is a *control redistribution*, NOT new independent randomness.
        event_slots = [i%3 for i in range(len(events))]
        rng.shuffle(event_slots)
        for e, z in zip(events, event_slots):
            e["zone"] = z
            candidates = sorted(p for p in available-SPINE-food
                                if region(p) == z)
            e["location"] = list(rng.choice(candidates)) if candidates else None
            if e["kind"] in ("detour","compound") and not candidates:
                raise AssertionError("No safe target for matched event type")

    if len(walls) != len(old_walls) or len(food) != len(old_food):
        raise AssertionError("Static environmental budgets differ")
    if len(events) != len(source["events"]):
        raise AssertionError("Changed the exogenous event count")
    if reachable(INTERIOR_CELLS-walls) != INTERIOR_CELLS-walls:
        raise AssertionError("Counterfactual world is not traversable")
    world["walls"] = [list(p) for p in sorted(walls)]
    world["food"] = [list(p) for p in sorted(food)]
    world["events"] = events
    world["variant"] = variant
    world["parent_world_sha256"] = source["world_sha256"]
    world["parameters"] = dict(source["parameters"])
    world["parameters"]["counterfactual_variant"] = variant
    canon=json.dumps({k:v for k,v in world.items() if k!="world_sha256"},
                     sort_keys=True,separators=(",",":"))
    world["world_sha256"] = hashlib.sha256(canon.encode()).hexdigest()
    return world


def audit_variants(seeds: tuple[int,...] = DEV_SEEDS) -> dict:
    if not seeds or len(seeds) != len(set(seeds)) or any(
            s not in DEV_SEEDS for s in seeds):
        raise ValueError("Development-only unique world seeds")
    out = {"variants":{}, "world_count":len(seeds)}
    for variant in VARIANTS:
        rows = [variant_world(seed,variant) for seed in seeds]
        walls = [0,0,0]
        food = [0,0,0]
        events = [0,0,0]
        for w in rows:
            original = make_world(w["seed"])
            if len(w["walls"]) != len(original["walls"]) or (
                    len(w["food"]) != len(original["food"]) or
                    len(w["events"]) != len(original["events"])):
                raise AssertionError("Unmatched environmental budget")
            for z in range(3):
                walls[z] += sum(region(tuple(p))==z for p in w["walls"])
                food[z] += sum(region(tuple(p))==z for p in w["food"])
                events[z] += sum(e["zone"]==z for e in w["events"])
        out["variants"][variant] = {
            "wall_counts_by_zone":walls, "food_counts_by_zone":food,
            "events_by_zone":events,
            "worlds_sha256":hashlib.sha256(json.dumps(
                rows,sort_keys=True,separators=(",",":")).encode()).hexdigest()
        }
    return out
