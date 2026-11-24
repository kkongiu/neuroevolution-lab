# FlyLoop R1 — result: sustained recurrent locomotion, but not purposeful exploration
**24 November 2026 · PROSPECTIVE science screen FAIL 11/13, software CI SUCCESS.**

This is a NEW branch based directly on \`main\`, independent from the draft M0–M4 stack. Study preregistered in [FLYLOOP_R1_PROTOCOL_2026-11-24.md](FLYLOOP_R1_PROTOCOL_2026-11-24.md) before source tests and scientific results. Do not modify R1 efficacy thresholds because of results.

## Complete official evidence

- Completed green GitHub Actions [run #45374556037](https://github.com/kkongiu/neuroevolution-lab/actions/runs/45374556037).
- **24 preregistered seeds 9100–9123**, 2 worlds ×5 controls = **240 lives** with up to 600 continuous 0.2s motor integration steps.
- Artifact \`flyloop-r1-continuous-motor-results\`, ID **14910203098**, retained through **24 December 2026**.
- Two full JSON files **86,458,938 bytes EACH**, verified byte-identical by GitHub Actions AND by separate archive SHA-256 recomputation. Full JSON SHA-256: \`e498686b709558863b746f69febbc77699f31053140e5b697876bfa9829facd2\`. Canonical raw-row SHA-256 independently recomputed: \`6789383e40517d6b0893b447fcc2db9f21e4482a7e94234e337f92eaa9c0b226\`.
- **142,335 real body motion steps** audited again from raw GitHub artifact, **0 independent physical/accounting errors**, **48** scenario-seed-matched worlds and initial poses, **24** identical initial neural weight matrices reused across arms, **24** distinct exogenously generated obstacle maps.
- Every movement, collision, exact eight pre-action sensor numbers, all 16 neural activations, two continuous motor commands, damped speed/steer, x/y/heading, physical integrity/energy and initialized 434 network coefficients preserved. No weight updates at any timestep.

## Physical organism

A small simulated body moves in a continuous 22×22 plane with body radius 0.3, speed and angular-velocity actuators, inertia, walls and collisions, three short-range proximity rays, previous collision bit, speed/turn and energy state. The circuit has **16 persistent tanh units** and two analog motor outputs, with recurrence and sensory input. All initial random coefficients are FROZEN throughout this study. No food, attacks, flee reflexes, pathfinding, fitness/reward shaping, motor learning, offspring or evolution. **We test generation of locomotor dynamics, not goal-directed intelligence.**

Controls share the same initialized network matrices, physical motors and world seeds: recurrent, leaky state but NO recurrent connection contribution, instantaneous feedforward state with motor inertia still present, recurrent but wall/proximity senses masked, and smooth temporally correlated random motor commands. The ablations have equal coefficient slots, but NOT equal *active* parameter counts, because matrices are masked; no misleading parameter matching claim is made.

## Main predeclared outcomes: obstacle world (mean over 24 seeds)

| Controller | Unique 1×1 tiles | Physical collisions/life | Collisions/meter, pooled | Distance/life | Mean absolute turn (rad/s) |
|---|---:|---:|---:|---:|---:|
| Inertial random motor | **35.83** | 216.38 | 4.813 | 44.96 | 0.374 |
| Instantaneous neural response | 25.42 | 119.75 | 2.254 | 53.12 | 0.592 |
| Leaky state, NO recurrence | 26.25 | 103.25 | 1.837 | 56.19 | 0.670 |
| **Fully recurrent** | **19.71** | 65.13 | 1.032 | 63.12 | 0.962 |
| Recurrent but wall sensors masked | **12.00** | **16.08** | **0.209** | **76.86** | **1.011** |

**Critical observation**: fewer hits is NOT sufficient evidence of obstacle avoidance. The best collision score was obtained by the agent that was BLIND to wall sensors, but its average coverage was only 12 cells. Recurrent networks generated curved, compact recurrent trajectories rather than effective coverage. Adding wall sensing led to MORE collision events per physical meter and more covered cells. This is the opposite of a clean avoidance-reflex demonstration.

### Additional exploratory orbit diagnostic from independent raw trajectories

For each of 24 obstacle lives per arm we independently integrated signed angular velocity across time to estimate **absolute net completed-turn equivalents** (sum of angular velocity×dt divided by 2π), and measured RMS radius about the trajectory's own centroid:

| Controller | Mean absolute net turn equivalents/life | Mean trajectory RMS spatial radius |
|---|---:|---:|
| Inertial random | 1.38 | 3.70 |
| Instantaneous | 10.52 | 3.50 |
| Leaky, no recurrent links | 12.16 | 3.23 |
| **Recurrent** | **17.59** | **1.98** |
| Recurrent, wall blind | **19.16** | **0.99** |

These are exploratory, not preregistered endpoints. They give quantitative evidence of repeated turning and compact circling. They do NOT identify the cause of individual turns or imply an intentional orbiting goal.

The recurrent motor dynamics ARE measurably distinct from the leaky and instantaneous controls, and yield much more physical movement than a passive agent. But activity/smoothness is not equivalent to useful navigation. Faster turning and remaining in a small patch can mechanically minimize wall strikes.

### Prospective efficacy tests (UNMODIFIED)

1. Recurrent obstacle collision events per meter <= 90% of inertial-random value: **PASS** (1.032 vs 4.813).
2. Recurrent obstacle unique tiles >= 110% of nonrecurrent leaky value: **FAIL** (19.71 vs 26.25); paired seed difference **−6.54** tiles (exploratory 95% bootstrap **[−18.83,+6.25]**).
3. Recurrent obstacle collision events per meter <= 90% of identical recurrent network WITHOUT wall perception: **FAIL** (1.032 vs 0.209). Masking wall perception produces LESS collision, but the resulting motion is mostly confined circling.

**Integrity criteria 10/10 pass, scientific efficacy 1/3 pass, total 11/13 — R1 DEVELOPMENT SCIENCE SCREEN FAIL.** No change to prespecified goals and no post-result hyperparameter adjustment.

## Independent raw-data verification

The downloaded GitHub artifact was inspected outside the experimental module. Audit independently recomputed the SHA-256 of both full JSON files and the canonical raw rows and validated all 142,335 physical transitions:
- x/y continuity and command → actual movement or blockage;
- no body-radius-inflated rectangle or world-boundary penetration;
- real motor low-pass integration and heading updates;
- physical energy and integrity decrements;
- observed net distance, collision totals and grid coverage;
- equal original weights/body pose across controls on same seed;
- no prohibited wall sensors in the recurrent-blind control;
- no neural activity in the random motor control and **no weight changes** in any arm.

**Observed independent audit discrepancies = ZERO.**

## Honest interpretation / next step

R1 successfully demonstrates a small **continually evolving neural circuit can generate analog movement**. It does NOT show emergent foraging, danger escape, useful navigation, obstacle avoidance, learned motor behavior, or biological fly reproduction. Its characteristic weakness is compact self-turning: the network can be continuously active without producing purposeful coverage.

**The next falsifiable step is R2, NOT a retrospective R1 fix:** determine whether plastic motor connections or a minimal homeostatic error signal can favor *physical progress and sensory responsiveness* without adding prewritten escape, forage or wall-follow routines. Precommit adversarial tests against simple random-persistent and nonrecurrent controllers, and separate artificial shaping from spontaneously selected behavior. The alternative is to let circuits compete across generations; evolution is intentionally NOT authorized by R1.

All STOP FLAGS remain false: food_search_demonstrated, escape_reflex_learned, adaptive_motor_learning_demonstrated, biological_fly_replication_demonstrated, general_intelligence_demonstrated, evolution_authorized, advance_to_evolution. Draft PR, NO merge into main; original sealed M5.6 and unmerged M0–M4 are untouched.
