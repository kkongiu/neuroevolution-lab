# FlyLoop M3: prospectively frozen protocol (24 November 2026)
Status: PRE-RESULT, immutable scientific thresholds and samples.

## Question
M2 showed no reliably superior blind foraging. Does keeping one-step action and *physically observed* consequence history give an original 14-weight actor-critic a real and causally attributable advantage, without adding parameters, a map, scent, food/hazard lookahead, or prewritten action plans?

## Locked organisms and environments
- Same unmodified M0 8-real-weight shared linear actor and 6-real-weight critic, temperature, TD(0) gradient learning rates, real energy/health reward and five primitive actions N/E/S/W/WAIT; initial body 70 energy,100 health, 16 regrowing food sites, 9 damage sites, 13x13 grid, 480-tick max per life.
- All arms retain M2's BLIND local sensory mask: no precontact directional food signal, adjacent food bit, hazard lookahead or hazard-linked injury bit. Wall lookahead plus body hunger/injury still visible. Actual energy/health changes after an action are observable and can influence the next step. No absolute world position, global map, hidden food labels, target route or schedule supplied to the network.
- 32 NEW world seeds 7900..7931, disjoint from all previous M0/M1/M2 seeds. Three scenarios: baseline fixed M0 gate switch at tick 240; procedural M2 connected random maze with 10 independent wall posts, random two disjoint passage pairs switching at 240; relocation fixed M0 wall with NO gate switch, all 16 food sites exogenously moved at tick 240 and originals restored at tick 360, via M1's corrected map generator independent of the agent's position. All seven controllers see the same map, resources, event schedule and 14 INITIAL neural weights per scenario/world.
- No learning across worlds, no evolution, no parameter tuning after evaluation.

## Precommitted temporal input encoding (explicit researcher prior)
Reuse five otherwise ZERO indices in the exact M2 blind actor input: index 3 = alignment of candidate direction with the previous *attempted* movement, index 4 = alignment times previous wall collision, index 1 = alignment times most recently observed positive physical reward scaled by the known food energy increment (17), index 2 = alignment times recent observed damage (health loss divided by 4), index 6 = alignment times exponential average of previous *physical* rewards scaled/clipped to [-1,1], with decay coefficient 0.75. WAIT alignment zero. No memory at first step. Critic previously-unused blind indices 3 and 4 optionally contain previous observed reward/damage, again never any future event. This is a hand-designed sensory representation, not spontaneous memory architecture, and has NO increase in trainable weights.

## Seven experimental arms
1. frozen: original frozen stochastic M2 blind policy, no plastic update, no memory.
2. random_motion: uniformly sample N/E/S/W without waiting, no wall correction, no plasticity.
3. plastic_plain: original M2 blind TD online policy, no memory.
4. motor_memory: same TD learner but only remembered last movement direction and last collision; feedback/reward history inputs zero.
5. temporal_feedback: same TD learner and previous motor history plus TRUE last observed physical reward/damage and EMA.
6. temporal_sham: same motor memory, but feedback reward/damage/EMA drawn from an OLDER observed event from the same living organism at least two ticks earlier; never the last event and never a future event. All shuffling uses a separate RNG that never changes the actor RNG. If no older event available, zero feedback.
7. feedback_frozen: same TRUE action+outcome history inputs as temporal_feedback, but actor and critic weights fixed; tests whether merely embedding history is enough without learning.

All seven keep 14 original adjustable weight positions (even unused weight positions in nonlearning arms); only feature meaning in previously zero blind channels changes. No intrinsic novelty bonus in M3.

Preselected count = 3 scenarios x 7 arms x 32 world seeds = exactly 672 continuous lifetimes, maximum 322560 physical steps. Fully deterministic second rerun must reproduce the JSON byte-for-byte, complete logs available.

## Outcomes
Record all pre-action raw blind sensory matrices, effective action/critic input, selected history source time and physical prior-feedback values, position/action/energy/health/physical reward/food/hazard/collision per real step, actual map/event schedules, neural starting and terminal 14 weights, survival, pooled food per 100 actually lived ticks, total food, damage, collisions, barrier crossings, unique food-site contacts, and food in relocation phases [0,239], [240,359], [360,479] (zero for deaths). World SEED is independent unit for exploratory paired 1000-draw bootstrap intervals. Do not compare simulated timesteps as independent replicates.

## Eighteen predeclared gates (ALL must pass for narrow M3 DEV science success)

Integrity:
1. Exactly 672 balanced episodes, complete repeat byte identical (Actions).
2. Physical motion, energy, integrity and no-wall constraints verified across every trace; no resurrection.
3. Identical maps, hazard/food, schedules and initialized 14 weights within all paired controls.
4. Raw BLIND 8-action and 6-critic sensory channels always exclude odor, food and hazard lookahead; real consequences only after movement.
5. Every nonzero feedback memory derives exclusively from own earlier physical outcomes (source tick < action tick); first step has no memory; no maps/coordinates/future in input.
6. Sham source is <= current tick minus 2; separate RNG and no fabricated data for early steps.
7. Same 8+6 weight topology; frozen/random/frozen-feedback never update, TD arms update their original weights.
8. Unaugmented frozen, random and plastic_plain baseline trajectories exactly reproduce original M2 blind baseline code with same seed, world and RNG.
9. Procedural layouts differ over >=25 of 32 new world seeds; 10 added wall posts, two disjoint passage pairs, fully connected traversable cells and valid resource locations in both phases.
10. Relocation map is exogenous and identical across paired controllers; tick 240 move and 360 restore (if alive); wall does not simultaneously change.
11. Every original full trajectory and negative result archived, original M0/M1/M2 and M5.6 unchanged, all stop flags false.

Scientific:
12. Baseline/blind true-feedback food per 100 living steps >= plain plastic x1.15.
13. Baseline/blind true-feedback food rate >= uniform random_motion x1.10.
14. Procedural/blind true-feedback food rate >= uniform random_motion x1.10.
15. Procedural/blind true-feedback food rate >= plain plastic x1.15.
16. Procedural/blind true-feedback food rate >= motor_memory x1.10.
17. Procedural/blind true-feedback food rate >= temporal_sham x1.10.
18. During actual food relocation ticks 240..359, mean true-feedback food per world >= plain plastic x1.15, including zero for dead organisms.

Additional non-gate control: true-feedback vs feedback_frozen plasticity check, and full hazard/collision outcomes to detect harmful exploration. A green software CI does NOT make a failed science gate pass. Do not retune gates or seed sets; publish negative results exactly. The temporal encoding itself is engineered and cannot prove spontaneous goal invention, positional memory or new movement primitives.

All stop flags permanently false: general_intelligence_demonstrated, spatial_memory_verified, autonomous_operator_invention_verified, evolution_authorized, advance_to_evolution.
