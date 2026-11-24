# FlyLoop R1 — continuous recurrent motor dynamics: frozen prospective protocol
**24 November 2026 — written before the first full R1 scientific run.**

## Why this is a separate research branch

FlyLoop M0–M4 use a discrete five-choice softmax actor. Their results show adaptation with unusually informative sensor representations but no robust genuinely blind resource-search strategy. The new hypothesis is about the substrate, not another symbolic decision module: can a continuously updated recurrent motor circuit exhibit useful sensory-conditioned locomotion that a nonrecurrent or merely inertial controller cannot? M0–M4 (unmerged draft PRs), sealed M5.6 and \`main\` are never modified by R1. The R1 branch starts from **main** and is self-contained: not a stack on M4.

**R1 deliberately tests movement and contacts ONLY. No food, foraging, learned escape, evolutionary updates or online synaptic plasticity yet.** Biological fly analogy is conceptual, not a model of Drosophila circuitry. Success means motion control, not intelligence, survival adaptation or learned policy. All comparisons are made BEFORE making claims of emergence.

## Mechanism fixed before results

- A bounded **22 × 22 continuous 2D world**, point-mass body with physical radius 0.30, heading in radians, translational velocity and angular velocity, discrete numerical integration time step **dt=0.20**, 600 frames/life. Same actuator limits and inertia in all control arms.
- Eight sensory numbers: constant input, local wall proximity within distance 3.5 along forward, left and right rays; actual current forward speed; actual angular velocity; previous collision signal; fraction of depleted energy. Wall sensor detects **walls only**, not food, tasks or future actions. No map coordinate, obstacle route, future state, threat, reward or preprogrammed avoidance command in neural inputs.
- Body energy starts at 100 and drains 0.02/tick; physical contact costs 0.25 integrity (starts at 100), collisions are completely blocking, no teleporting through obstacles; death at zero energy/integrity or horizon. The *physical sensor* can detect walls before collision, as in biological sensory guidance. Energy never increases in R1. No model receives a hand-coded "turn away" action.
- Exactly 16 tanh internal neurons, two continuous motor outputs for forward speed and angular speed. Initially fixed randomly seeded matrices: input 16×8, recurrence 16×16, motor readout 2×16, biases 16+2 = **434 coefficient slots**. No learned weights, genotype, episodic task feedback or evolution. Every arm starts with the SAME seeded matrices. Same initial body position/orientation and world geometry within (scenario,seed).
- The recurrent controller evolves
  \`h[t+1] = (1-alpha)*h[t] + alpha*tanh(W_in*x[t] + W_rec*h[t] + b)\`,
  \`alpha=0.30\`. Motor readout is a function of h, with the same physical low-pass actuator inertia in all arms. No discrete action choosing, search/escape macros or pathfinding.
- The random matrix and actuator output scaling, ray distance, integration and coefficient initialization are fixed in source and reported verbatim. There is no training on the study's maps or retuning after running it. Continuous motion already includes hard-coded mechanics (motor damping, wall collision and speed limits); do **not** call these emergent.

## Five matched experimental controllers

1. \`recurrent\`: full persistent h and **W_rec * h**. 434 coefficient slots initialized (including recurrence). NO learning.
2. \`leaky_no_recurrence\`: h persists through same alpha low-pass, but the W_rec*h contribution is ablated to exactly zero. Shares identical initialized matrices, though recurrent coefficient slots become inactive; therefore this is NOT parameter-matched in *active* degrees of freedom.
3. \`instantaneous\`: tanh(W_in*x+b) recalculated at each frame without neural temporal state, but EXACT same actuator physics/inertia (important control).
4. \`recurrent_blind\`: exact recurrent controller and state dynamics, but wall proximity and contact history inputs clamped to zero (preserving body speed/turn/energy inputs); tests whether sensor access is causally useful. No hidden wall input elsewhere.
5. \`inertial_random\`: 2 independent seeded low-frequency autoregressive motor command streams, using same motor low-pass inertia, movement limits and obstacles. No wall-correction, no neural inference, no learning. Matrices still initialized identically but unused. This controls for mere smooth persistent random locomotion.

## Two world scenarios, both with the same seeds

1. \`empty\`: external boundary only; the body starts inside world, no internal obstacles.
2. \`obstacles\`: 10 procedural axis-aligned solid rectangles generated independently of organism from \`world_seed\`. Avoid overlap with spawn and other rectangles by a deterministic rejection sampler; all collisions use geometry inflated by body radius. No special physics/passing rules or collision avoidance algorithm.

Exactly **24 new preselected independent world seeds 9100–9123**, disjoint from any prior FlyLoop study. Same initial neural coefficients and body initial conditions within each pair. Total = **2 ×5 ×24=240 complete lifetimes** of up to 600 frames (max 144,000 movement integrations). No training episodes, online plasticity, reseeding until success, or controller selection. Run twice, require byte-identical archives.

### Archived per-frame measurements

For every physical frame: pre-move x/y/heading, eight real sensory inputs, all 16 neural states (except inertial random, which records zero state), raw motor command, actuator-filtered speed/turn, real move, collision indicator, energy, integrity; all world rectangles, full initialization and neural weight hash. Aggregate distance traveled, net displacement, net/path length, 1×1 cell coverage, mean speed, mean absolute turn, collision events, collision per physical meter (with explicit zero handling), fraction of moving frames and lifespan. Those are observational metrics; smoothness itself is NOT objective fitness.

### Thirteen fixed QA / scientific gates — all required

Integrity (10):
1. Exactly 240 balanced seed×scenario×arm full trajectories and byte-identical rerun.
2. All physical x/y/heading/speed/turn/energy/integrity states legal, no wall penetration or teleportation.
3. Same starting body and EXACT initial 434 coefficient slots for all five arms per world seed; same rectangular map within scenario+seed.
4. No food, resource collection, food smell, learned reward, synaptic updates, reproduction, evolution, or engineered follow-wall / escape command.
5. 16 neural states and two continuous motor commands, with fixed coefficients and correct leaky recurrence equation on every audited frame.
6. \`leaky_no_recurrence\` truly eliminates W_rec term while preserving 16-state leak and SAME actuator inertia; \`instantaneous\` has no neural memory.
7. \`recurrent_blind\` has EXACTLY zero wall-range/contact sensor inputs while keeping the rest, and does not consult the map elsewhere.
8. \`inertial_random\` uses only autoregressive independent motor commands and SAME physical motor filter / collision handling; never receives map input.
9. Procedural rectangles do not overlap spawn, are fixed independent of animal position, and obstacles actually differ across >=20 of 24 maps.
10. Recurrence changes realized motion trajectory relative to leaky and instantaneous controls in >=90% of world seeds; this is evidence of different dynamics, not necessarily an advantage.

Science (3, NOT assured to pass):
11. In \`obstacles\`, **recurrent collisions per meter** <= 0.90 × inertial random collisions per meter (aggregate across whole 24-seed population).
12. In \`obstacles\`, **recurrent unique 1×1 cells visited** >= 1.10 × leaky-no-recurrence mean coverage.
13. In \`obstacles\`, **recurrent collisions per meter** <= 0.90 × recurrent-blind collisions per meter, showing that real obstacle sensing matters under recurrent dynamics.

All thirteen must pass for narrow R1 DEV screen success. Software CI success is **not** proof of science success. Preserve all failures, per-seed uncertainty, and full code/trajectories without retuning thresholds after seeing data. A positive score would not mean the network learned: in R1 *all neural weights are frozen*. Any apparent behavior is generated by random initial circuits plus engineered sensor and actuator physics; causal claims need further ablations. If all science criteria fail, design R2 causal motor-plasticity study without retroactively rewriting R1.

**Stop flags all false:** \`food_search_demonstrated=false\`, \`escape_reflex_learned=false\`, \`adaptive_motor_learning_demonstrated=false\`, \`biological_fly_replication_demonstrated=false\`, \`general_intelligence_demonstrated=false\`, \`evolution_authorized=false\`, \`advance_to_evolution=false\`.
