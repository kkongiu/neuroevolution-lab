# FlyLoop R5 — frozen prospective protocol for physical action interventions
**24 November 2026. Written and committed BEFORE first complete R5 evaluation.**

## Purpose
FlyLoop R4 found a small action-conditional prediction advantage on **observational** trajectories, but not a preregistered improvement in navigation. Merely replacing the action features in a predictor with older values did NOT actually make the organism physically execute the old actions, and therefore cannot on its own prove an intervention.

R5 performs the missing experiment: **from one identical bodily, neural and environmental pre-action state, physically execute five DIFFERENT motor commands in FIVE independent cloned worlds**, and compare both observed one-step sensory consequences and actual six-step body positions. No hypothetical observation is treated as real. All intervention branches use exact R1 motor physics, not a learned simulator or a wall-following policy.

Separate claims strictly:
(A) experimentally changing a motor command can produce a physical sensor/body consequence;
(B) a 3-sensor predictor trained on normal observations actually predicts those action-specific physical consequences on an off-policy intervention;
(C) improved navigation/foraging or agency is NOT tested here and must never be claimed.

## Frozen design

- R1 22×22 continuous physical world, radius 0.3, speed/rotation actuators and inertia, three real short-range wall rays within 3.5 units, eight local sensor channels, 16 tanh recurrent neurons, initial 434 network coefficients, no food/predators/global map/route/future input. SAME exact R1/R2/R4 source code and learning algorithms, unchanged.
- 24 NEW fixed seeds **10300–10323**, never used in M0–M4/R1–R4. Two geometry families: \`obstacles\` = R1's exogenous procedural 10 rectangles, \`nearfield\` = R2's independently exogenous near-spawn 10 rectangles. Per seed×scene there is ONE 600-step **frozen recurrent-motor physical life**, using exactly R2's fixed seeded Gaussian motor noise (σ=0.12) but NO motor synaptic plasticity. Exactly **48 real training/evaluation lives** (24 ×2), one for each world.
- On each life **three independently trained R4-style 84-coefficient next-wall-sensor heads**, identical zero initialization, head LMS rate .014, same actual real post-action wall-ray targets: \`true\` sees executed 2D motor command and six ray×command features; \`blind\` motor features all zero; \`yoked\` uses a real motor command of same organism at least six past frames old (separate RNG), before six samples use zero. No head learns an unrealized counterfactual label.
- **Training ONLY on steps t=0..299** of the natural recurrent-policy trajectory (300 physical action-observation pairs/life); then freeze all 3×84 head weights for t=300..599. No network/motor/head adaptation during probe clones, no cross-world or cross-lifetime transfer of weights. Freeze both training weights and original 434 recurrent coefficients. This explicitly tests off-policy command interventions without post-hoc fitting.
- Exactly **30 prospective probe snapshots per life**, taken at t = 300,310,320,...,590 inclusive (no performance-dependent selection), yielding **1440 independent starting-state snapshots**, nested within 48 worlds. For each, clone the full same pre-action world state (position, orientation, energy, health, speed, angular speed, collision memory, obstacles, world tick), and retain the same 16-valued neural pre-action state and identical frozen predictive weights.
- At each snapshot, use EXACTLY five forced commands, all constrained [-1,1]:
  1. \`natural\`: the actual executed command produced by the frozen recurrent circuit and common seeded noise on the original real trajectory.
  2. \`steer_left\`: (natural throttle, **−0.85** steer)
  3. \`steer_right\`: (natural throttle, **+0.85** steer)
  4. \`throttle_low\`: (**0.05** throttle, natural steer)
  5. \`throttle_high\`: (**0.95** throttle, natural steer).
  These commands are **experimental scientist interventions**, not actions selected by the fly; any suggestion the fly decided these would be incorrect.
- In every cloned world, actually execute its assigned forced command for **ONE real motor integration** then record new x/y, heading, speed/turn, energy/integrity, collision and **actual** post-action three wall readings. Continue the SAME commanded motor in that same clone for FIVE further REAL integration steps (six total), recording physical terminal x/y. Never update actor/hidden neurons or head weights during cloned trials; the held constant motor signal explicitly isolates physical actuator causality. This is not a simulation inside the organism's head.
- Each snapshot yields **five physically realized intervention trials**, **7200 real one-step physically branched observations**, **7200 six-step realized terminal poses**, with 4 non-natural contrasts per snapshot = **5760 realized paired contrasts**. The original natural world advances ONCE using its actual natural command independently of cloned probes, and the one-step natural-branch outcome MUST agree with that actual original action.
- On each forced trial evaluate BEFORE physically executing the candidate:
  - \`true_head\`: original R4 true trained head, with current 16 hidden, current rays, and THIS forced action.
  - \`blind_head\`: original R4 blind head, motor channels zero, same hidden/rays.
  - \`yoked_head\`: R4 yoked head, same hidden/rays and a previously actually executed motor at least six steps old, the SAME historical yoke across the five candidate trials of a snapshot.
  - \`persistence\`: exactly current wall rays (no action).
  The actual next-sensor readings come only from the corresponding executed physical clone. Model coefficients unchanged across all intervention trials, so comparison is paired.
- For the four non-natural trials per snapshot, measure **physically actual sensor-difference vector** relative to the natural trial, and **predicted sensor-difference vector** from the SAME true head with the corresponding forced vs natural motor command. The no-action constant-difference baseline is zero vector. A contrast is predeclared "sensor-informative" if max of the three absolute *physically measured* one-step ray differences is at least **0.005**; selection is outcome-based but **does not depend on any predictor error**, and both overall unselected and informative-only comparisons must be reported. Independent unit of inference is WORLD SEED; snapshots/frames are nested technical replicates, not independent organisms.
- Secondary count: an intervention snapshot is "physically divergent" if the final six-step locations of the experimentally forced \`steer_left\` and \`steer_right\` clones differ by at least **0.05 units**. Do not mistake physical divergence for learned motor competence.
- Full data must include both original 600-frame natural sensor/action training/evaluation trajectory, exact pre-action cloned body snapshot, trained frozen head weights and hashes, yoke source tick and exact historical command, five actual applied actions, all 1-step and 6-step REAL physical outcomes, all 3 model predictions per forced action, raw sensor differences, metrics, original neural initialization. Run the entire experiment twice and require **byte-for-byte identical JSON results**, including all records. No cherry-picked seed, state, trial or actuator after results.

## Metrics
- Average one-step MSE of \`true\`, \`blind\`, \`yoked\` and \`persistence\` across ALL held-out intervention trials, not just sensory-active transitions, overall and by scene.
- Mean per-intervention sensor contrast MSE vs zero-contrast baseline on ALL 5760 pairs and on preregistered informative pairs only; number and fraction informative reported.
- Mean physical 6-step left-vs-right terminal distance and fraction >=0.05, overall and scene-stratified.
- Paired world-seed bootstrap95 exploratory intervals for per-world differences of predictor MSE and contrast losses, NEVER treat 7200 branches as independent seeds.
- Training MSE and frozen-head change, number of actual training updates; compare train-phase vs probe-phase with caution due altered action distribution.

## Sixteen hard gates — ALL must pass for narrow DEV-science success

**Integrity (10):**
1. Exactly 48 original 600-step natural lives, 1440 fixed snapshots, 7200 REAL forced action branches, 5760 candidate-vs-natural contrasts, and complete repeat byte-identical.
2. All physically executed branches start from an **exact cloned full preaction body/world state**, and each uses actual original R1 actuator/rigid-obstacle physics with no penetration, wall tunneling, pre-death reanimation or post-branch information leakage.
3. Natural experimental clone's real first-step x/y, heading/speed/turn, energy/integrity, collision and post-wall rays exactly agree with the independently advancing original natural world action at same tick.
4. Same pretrained 16-hidden 8-local-sensor R1 brain, original 434 network coefficients never update; no absolute map/food/future sensor in network or predictor input.
5. All three 84-weight heads initially exactly zero, trained on identical 300-step natural actual sensory targets with exact R4 LMS and then 100% frozen during 300..599 testing and all branch probes.
6. Exactly five declared (bounded) motor commands applied per cloned snapshot, with no motor correction or privileged physics guidance and identical neural pre-state and model weights across trials.
7. Yoked head receives only same organism's actual previously executed action with source tick ≤ probe tick−6; separate PRNG, no future or fabricated action; blind head command/interactions always zero.
8. All predictions computed from pre-movement samples, and every comparison target is a sensor reading produced AFTER an actually executed forced action, never an imaginary future or model forecast.
9. 24 distinct valid exogenous ten-rectangle maps per environment family, no overlap with spawn, same environment across 5 trials of a snapshot; no extra learnable parameters or rewiring.
10. Complete raw frame/branch logs, full repeat and original source unchanged, no false claims of acquired goal/navigation/intelligence/evolution.

**Scientific (6):**
11. At least **60%** of all 1440 snapshots have physically different six-step positions for forcibly executed \`steer_left\` vs \`steer_right\` by >=0.05; proves actuators can cause physical changes under identical initial conditions, NOT learned competence.
12. At least **100** preregistered sensor-informative candidate-vs-natural contrasts over all worlds, and >=**20 of 24 world seeds** show at least one informative contrast in either scene; if insufficient signal this test fails, not rescued by ex-post lowering threshold.
13. On all 7200 actually executed heldout one-step interventions, frozen **TRUE head average MSE ≤0.95 × blind-head MSE**.
14. On all 7200 actually executed heldout one-step interventions, frozen **TRUE head average MSE ≤0.95 × yoked-head MSE**.
15. On informative physically executed contrasts, learned TRUE-head paired sensor-difference MSE ≤**0.90 × zero-contrast baseline MSE**. Zero denominator or too few informative contrasts => FAIL.
16. Both scene families separately: on informative pairs in EACH scene, learned TRUE contrast MSE ≤**0.95 × zero-contrast baseline MSE**, with >=20 informative pairs in each scene; no single easy environment can establish generalization.

All 16 required. Green CI and satisfied *physics* gates cannot rescue failed predictor efficacy gates. Scientific outcomes must be reported honestly as OBSERVED physical causal differences under forced external action, or learned model ability to predict them, separately. Nothing here tests an organism choosing better actions autonomously.

**STOP FLAGS always false:** \`autonomous_control_demonstrated\`, \`food_search_demonstrated\`, \`escape_reflex_learned\`, \`autonomous_goal_invention_demonstrated\`, \`general_intelligence_demonstrated\`, \`biological_fly_replication_demonstrated\`, \`evolution_authorized\`, \`advance_to_evolution\`.
