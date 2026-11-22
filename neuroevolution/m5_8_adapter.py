"""M5.8 physical adapter for exogenous schedule, no extra agent observations.

- Attach a generator-defined, entirely navigable world to original M5.0
  ecology with the same energy ledger and original action actuator.
- Apply drought by suppressing *potential* future regrowth attempts, never
  deleting existing food (which would require additional accounting).
- Apply temporary detours as real walls, only if no body, predator, or food
  occupies the cell and connectivity of the entire accessible territory holds.
- Apply predator shifts as an extra physical M5.5B predator movement/capture
  step while the event is active; this is programmed weather-like pressure,
  not an adapted predator mind.
- All events are seeded/precomputed, independent of organism choices; only
  safety-related *application* may be skipped due to occupancy/connectivity.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field, replace

from . import m5_0 as base, m5_5b
from .m5_8_worlds import ALLOWED_SEEDS, WIDTH, reachable, region
from .m5_8_variants import VARIANTS, variant_world

DIRS = base.DIRECTIONS


def attach(eco: base.Ecology, world: dict) -> None:
    """Replace only geometry and starting population coordinates, not physics."""
    variant = world.get('variant', 'progressive')
    if (eco.seed not in ALLOWED_SEEDS or variant not in VARIANTS
            or world != variant_world(eco.seed, variant)):
        raise ValueError("Only canonical development world is accepted")
    if eco.settings.size != WIDTH or eco.settings.initial_population != 8:
        raise ValueError("Expected calibrated physical world")
    interior = {(x, y) for x in range(1, WIDTH-1) for y in range(1, WIDTH-1)}
    border = {(x, y) for x in range(WIDTH) for y in range(WIDTH)
              if x in (0, WIDTH-1) or y in (0, WIDTH-1)}
    walls = border | {tuple(p) for p in world["walls"]}
    open_cells = interior - walls
    food = {tuple(p) for p in world["food"]}
    if reachable(open_cells) != open_cells or len(food) != 9:
        raise AssertionError("World failed physical validation")
    placements = sorted(open_cells - food)
    starts = random.Random(eco.seed * 4243 + 193).sample(
        placements, eco.settings.initial_population)
    eco.world = replace(eco.world, walls=frozenset(walls),
                        initial_food=tuple(sorted(food)))
    eco.food = food
    eco.accessible = tuple(sorted(open_cells))
    for agent, position in zip(eco.agents.values(), starts):
        agent.position = position
    eco.food_spawned = len(food)
    if abs(base.energy_balance(eco)) > 1e-6:
        raise AssertionError("Energy changed by physical world installation")


@dataclass
class ActiveShocks:
    """Tracks time-limited *precomputed* shocks with body-safe walls."""
    world: dict
    static_walls: frozenset[tuple[int, int]]
    applied: int = 0
    skipped_detours: int = 0
    drought_suppressed_attempts: int = 0
    predator_extra_steps: int = 0
    live_detours: set[tuple[int, int]] = field(default_factory=set)

    @classmethod
    def create(cls, eco: base.Ecology, world: dict) -> "ActiveShocks":
        return cls(world, eco.world.walls)

    def at_tick(self, eco: base.Ecology, predators: list[base.Point],
                tick: int) -> tuple[set[int], bool]:
        """Apply active event proposals; return drought zones and aggression."""
        active = [e for e in self.world["events"]
                  if e["tick"] <= tick < e["tick"] + e["duration"]]
        drought = set()
        predator_aggression = False
        obstacle_candidates = set()
        for e in active:
            kind = e["kind"]
            if kind in ("drought", "compound"):
                drought.add(e["zone"])
            if kind in ("predator_shift", "compound"):
                predator_aggression = True
            if kind in ("detour", "compound") and e["location"] is not None:
                obstacle_candidates.add(tuple(e["location"]))
        bodies = {a.position for a in eco.agents.values()}
        occupied = bodies | set(predators) | set(eco.food)
        interior = {(x, y) for x in range(1, WIDTH-1)
                    for y in range(1, WIDTH-1)}
        accepted: set[tuple[int, int]] = set()
        # Include original static walls, keep permanent route spine connected.
        # When a temporary wall cannot safely remain, skip it.
        for p in sorted(obstacle_candidates):
            if p in occupied or p in self.static_walls:
                self.skipped_detours += 1
                continue
            candidate = accepted | {p}
            free = interior - self.static_walls - candidate
            if reachable(free) != free:
                self.skipped_detours += 1
                continue
            accepted = candidate
        self.live_detours = accepted
        eco.world = replace(eco.world,
                            walls=frozenset(self.static_walls | accepted))
        if not bodies.isdisjoint(eco.world.walls):
            raise AssertionError("Temporary wall enclosed a living organism")
        self.applied += len([e for e in self.world["events"] if e["tick"] == tick])
        return drought, predator_aggression

    def modify_regrowth(self, attempts: list[tuple[bool, base.Point]],
                        drought_zones: set[int]) -> list[tuple[bool, base.Point]]:
        out = []
        for want, p in attempts:
            suppress = want and region(p) in drought_zones
            self.drought_suppressed_attempts += int(suppress)
            out.append((want and not suppress, p))
        return out

    def run_extra_predator_step(self, eco: base.Ecology,
                                predators: list[base.Point],
                                tick: int, enabled: bool) -> int:
        if not enabled:
            return 0
        self.predator_extra_steps += 1
        # Different tick key than the canonical first step.
        return m5_5b.step_predators(eco, predators, tick + 100_000)
