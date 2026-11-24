# FlyLoop R3 — VERIFIED RESULT: learns next sensory readings, NOT more effective motor exploration

**24 November 2026.** Prospective screen **FAIL: 12/16 gates**. All **10/10 integrity** gates PASS, both **2/2 sensory prediction** efficacy gates PASS, all **4/4 motor-control** efficacy gates FAIL. [Frozen prospective protocol](FLYLOOP_R3_PROTOCOL_2026-11-24.md) committed before the first entire evaluation. No outcome, seed range, sensory/prediction learning rate, motor readout formula or predeclared efficacy threshold was modified after seeing results.

## What the result supports and does NOT support

An original 16-neuron recurrent controller with the same 434 randomly initialized neural coefficients can **learn a small one-step predictive relationship** between its own continuous physical motion and the three subsequently perceived wall-proximity readings. The added 66-coefficient head is researcher-designed and trained by real physical consequences, not supervised by maps. Prediction error on an EXACTLY matched trajectory is substantially lower when taught by the *correct* next local observation than by an older, causally decoupled observation.

But **motor plasticity minimizing the prediction error does not make the organism explore more or strike obstacles less** than the original frozen recurrent motor controller. The prediction is better; the *use* of the prediction for behavior is not.

This DOES NOT demonstrate general intelligence, goal invention, food foraging, learned escape or fly-like abilities. The error-minimization goal, wall-ray representation, and policy eligibility learning rule are **explicitly researcher engineered**, and minimizing surprise can favor self-predictable routes and circling.

## Frozen design, complete reproducibility

- **384 continuous independent lives**: 24 NEW world seeds **9700–9723** × two independent 10-obstacle procedural environments (\`obstacles\`, \`nearfield\`) × 8 paired controllers, each ≤600 simulation frames.
- Every seed×environment has exactly the same initial 434 neural coefficients, body pose, world map and motor-noise RNG stream for applicable arms.
- Unchanged R1 physics, 16 recurrent tanh units and two continuously controlled speed/steer actuators. No new action primitives, map input, hidden food, neural-planning procedure, genetic mutation, or evolutionary selection.
- **Additional 66-weight 3-sensor one-step head**: real current three rays +16 hidden states +actual executed two-motor command +bias; online clipped LMS with fixed rate .014. All head weights initially zero, exactly equivalent to predicting no sensor change. Head trained only **AFTER** the action from REAL next sensor value; sham labels are from the same organism at least SIX steps earlier, with an independent RNG.
- An original 34-coefficient motor readout may be updated with researcher-coded stochastic eligibility on \`negative sensory prediction MSE\`, after actual physical feedback. A separate arm uses R2's researcher-coded progress/contact reward for context.
- **Eight arms**: recurrent frozen, R2 physical-progress plastic, true predictor/passive frozen motor, true predictor+motor plastic, OLD-sensor sham predictor+motor plastic, persistence-only frozen head+motor plastic, old-sensor sham head on SAME PASSIVE physical path, inertial random movement.
- **All complete evaluation data repeated byte-for-byte**. Software regression tests for R1, R2 and R3 PASS; software success is not equivalent to scientific pass.

### Primary nearfield prediction result, late phase (frames 300–599)

| Predictor mode | Mean real-target prediction MSE | Same physical trajectory? |
|---|---:|---|
| Persistence (next reading = current) | **0.020050792** | Yes, identical passive path |
| OLD/shuffled actual older readings | **0.053387073** | Yes, identical passive path |
| **Learned next actual wall readings** | **0.016864999** | Yes, identical passive path |

The correct self-supervised head reduces late-phase error by **15.89%** relative to merely preserving the current sensor reading, and **68.41%** relative to sham training on older sensory readings. Both exceed preregistered 15% relative-error gates **11 and 12 (PASS)**. The two passive arms have **identical body movements, collisions, poses and physical inputs**; only the training labels/predictor weights differ. This isolates a useful learned one-step predictive association in this artificial sensor world. It is not equivalent to learned future navigation.

### Motor-control outcome, same nearfield worlds (mean over 24 independent seeds)

| Controller | Unique 1×1 cells | Contacts per actual meter | RMS trajectory radius |
|---|---:|---:|---:|
| Recurrent, frozen motor | **24.79** | **0.6708** | 2.642 |
| **Learned predictor + prediction-error motor plasticity** | **24.63** | **0.6971** | 2.620 |
| OLD-sensor sham predictor + same motor plasticity | 22.58 | 0.7354 | 2.427 |
| Frozen persistence head + same motor plasticity | 24.63 | 0.7051 | 2.568 |
| R2 physical-progress/contact motor plasticity | **26.63** | **0.3750** | 2.853 |
| Smooth inertial random movement | 33.92 | 5.1149 | 3.123 |

The prediction-error motor controller **does not exceed matched frozen** coverage (24.63 vs24.79), and it collides **more** per meter (0.6971 vs0.6708). The predictor motor's observed advantage over the sham-predictor motor is +2.04 grid cells per world (24.63 vs22.58, +9.0%) with exploratory **paired world-seed bootstrap 95% [+0.25,+4.21]**. That is an exploratory hint worth further study, but it **does NOT reach the prospectively fixed +10% threshold**, nor does it outperform the simpler frozen-head motor (both cover 24.63 cells). There is therefore no robust preregistered motor benefit.

The second independently generated \`obstacles\` geometry confirms the limitation: prediction-error motor explores **25.29** cells vs frozen **24.83**, sham **23.92**, R2-progress **29.33** and inertial random **39.96**; contacts per meter are prediction-motor **0.6778**, frozen **0.6310**. No claim of learned obstacle avoidance follows.

## Original 16 frozen gates: 12/16 overall FAIL

Integrity and provenance, gates **1–10: ALL PASS**
1. All 384 balanced raw organism lives and byte-identical complete replay.
2. Continuous physical body constraints and all motor commands legal, no teleporting through walls.
3. Identical exogenous maps, initial body state and 434 neural coefficients per paired seed, with matched exogenous motor noise.
4. Exactly 16 original hidden neurons and eight local sensory inputs, without global map/food or future observations.
5. Only up to 66 head and 34 existing motor-output coefficients may learn; upstream neural weights fixed.
6. Prediction before physical action, actual next-sensor labels AFTER real action.
7. Sham labels from own old observed sensory records, at least six ticks old; passive heads on identical trajectories.
8. Frozen head predicts current sensors unchanged.
9. Ten independent valid rectangles in each of 24 distinct worlds per layout family.
10. Complete raw trajectories, signed findings, unchanged prior studies, forbidden intelligence/evolution claims false.

Scientific efficacy:
11. Correct passive predictor late error ≤85% of same-trajectory persistence: **PASS** (0.016865 vs0.020051).
12. Correct passive predictor late error ≤85% of SAME-TRAJECTORY older-sham predictor: **PASS** (0.016865 vs0.053387).
13. Predictive motor cell coverage ≥ frozen ×1.10: **FAIL**.
14. Predictive motor contacts/meter ≤ frozen ×0.90: **FAIL**.
15. Predictive motor coverage ≥ shuffled-predictive motor ×1.10: **FAIL** (improvement ~9%, below 10%).
16. Predictive motor coverage ≥ persistence-head motor ×1.10: **FAIL**.

**Verdict: 10 integrity +2 predictive efficacy =12/16 PASS, 4 motor-control criteria FAIL, overall FAIL.** Results cannot be reclassified after the test.

## Technical QA fix before successful final run

The first full CI science run finished all 384 simulations but crashed in the cohort summary because of a coding mistake: a boolean trajectory equality comparison was accidentally nested inside \`all(bool)\`, which expects an iterable. We removed only that erroneous inner \`all(...)\` wrapper. **No model dynamics, prediction values, motor update, world generation, study seeds, sample selection or scientific thresholds changed.** All prior R1/R2/R3 unit tests passed before and after this audit-only fix, and the final run repeated all 384 lives byte-identically.

## Independent raw-data audit and provenance

Official successful [GitHub Actions run #45437895508](https://github.com/kkongiu/neuroevolution-lab/actions/runs/45437895508); downloadable artifact \`flyloop-r3-predictive-sensorimotor-results\`, artifact ID **14942356903**, retained for 30 days through **24 December 2026**.

The downloaded archive was audited independently, **without importing experiment modules**, and contains TWO full JSON files of **252,192,327 bytes EACH**, exactly identical by SHA-256:

\`138d8988bf0ae9b9d27f3cadff9d4cc99282913ed25ed95d10be140568426ad2\`

The complete raw-row canonical SHA-256 independently recomputed and matched the original source result:

\`d5ee19d2b4ca4cbd12039721a1f30311cffc2d9262a8baddf4dea19b7b554906\`

Independent audit: **384 real lives, 48 paired worlds, 229,721 physical motor frames, 23,834 wall-collision events, 114,198 sensory-predictor training updates, 56,800 historic-sham observations**; **ZERO errors**. The audit independently reconstructed every head's per-step prediction and all 66 final weights from the logged chronological experience, verified actual next sensor readings and strictly older sham targets, legal motion and obstacle clearance, energy/integrity decrement, original frozen upstream coefficients and equality of the two passive trajectories. Reproduction script and detailed JSON audit are available as separate downloadable files alongside the full GitHub artifact.

## Interpretation / next scientific decision

**R3 demonstrates causal, physical self-supervised *one-step sensory prediction* in a small artificial sensorimotor circuit. It does not demonstrate the corresponding improvement in physical navigation.** Better prediction should not be misidentified as learned goal pursuit: minimizing prediction surprise may prefer repeating a predictable orbit.

A possible R4 science question (NOT yet implemented): determine whether an agent can use the sensory predictor to evaluate **controllability** of its own effects rather than simply minimizing sensory error. The controls must separate prediction learning, policy information usage, movement persistence, static-sensor bias and multiple genuinely novel geometries, without supplying a hand-coded escape command. Preserve R3's full original failure.

Stop flags remain FALSE:
\`goal_invention_demonstrated=false\`, \`self_learned_planning_demonstrated=false\`, \`successful_obstacle_avoidance_demonstrated=false\`, \`food_search_demonstrated=false\`, \`biological_fly_replication_demonstrated=false\`, \`evolution_authorized=false\`, \`advance_to_evolution=false\`.

DRAFT PR #93 stacked on R2 PR #92, \`main\`, R1/R2, M0–M4 and sealed M5.6 untouched.
