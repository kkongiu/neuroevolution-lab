# FlyLoop R6 — Intervention-trained sensory prediction transfers to NEW worlds and motor commands; action-effect accuracy still falls short

**24 November 2026 · COMPLETE, independently verified.**

**Prospectively fixed scientific screen: FAIL, 15/17 gates passed (10/10 integrity, 5/7 scientific efficacy).** Software regression and double complete science run: PASS. The result is substantially stronger than R5 on **held-out action-conditioned prediction**, but it is not a complete verified causal effect model and is NOT autonomous control, navigation, foraging, escape or intelligence.

[Preregistered original R6 protocol](FLYLOOP_R6_PROTOCOL_2026-11-24.md) committed BEFORE R6 source code, scientific results or any seed evaluations. All original seeds, coefficients, intervention magnitudes and efficacy thresholds were retained without post-hoc tuning.

## Exact scientific question

R5 demonstrated real physical motor-action effects under cloned-state interventions but found a natural-observation-trained sensory predictor had only a weak advantage over a no-action model and NO reliable advantage over a temporally mismatched command control. R6 tests whether training a sensory predictor on **actual diverse motor interventions**, rather than ordinary self-chosen locomotion, improves model generalization.

R6's 3 sensor-prediction heads are fitted by the **external experimenter across multiple independent synthetic lives**; the trained weights are then frozen and transplanted into fully unseen worlds. This is **NOT within-organism autonomous learning**. Motor policies and the 16 recurrent core neurons remain frozen; no organism decides which experimental intervention to perform. Thus a positive head prediction is evidence of researcher-designed self-supervised model fitting under real intervention, not fly-like cognitive/goal emergence.

## Locked design and complete evidence

- **24 preregistered TRAIN seeds 10500–10523**, in two exogenous 10-obstacle families (\`obstacles\` and \`nearfield\`) =48 genuine training worlds.
- **24 disjoint HELDOUT seeds 10600–10623**, same two independently generated scene families =48 never-trained test worlds. Neither seed set appears in M0–M4/R1–R5.
- **96 total full 240-step natural continuous-body lives**, zero original deaths, all predetermined 10 snapshot times t=40,60,...,220 present. All natural 16-neuron recurrent states, 8 local sensors, 434 initial neural coefficients, same seeded R2 motor noise and real physical contacts/energy preserved. Original upstream network and motor readout never updated.
- At each of **960 physically attained states** the experimenter clones the exact same entire real world/body state FIVE times and physically executes a forced continuous 2-motor command for **3 ACTUAL R1 physical frames**, recording real terminal body/collision/wall rays. This totals **4,800 actual physically executed intervention trials and 3,840 paired natural-vs-forced real sensory contrasts** (2,400 and 1,920 respectively for train and heldout). All 4,800×3 = **14,400 actual forced physical integrations**, and 96×240 = **23,040 natural physical movements**. No experimental clone died; no resampling.
- TRAIN alternatives: forced steering **−0.95/+0.95**, throttle **0.10/0.90**, and natural command. HELDOUT alternatives: novel steering **−0.55/+0.55**, throttle **0.30/0.70**, plus the original natural command as the common baseline. No heldout world or forced-action magnitude used in training.
- Three **84-coefficient** prediction heads, initially all ZERO, use exact R4 28 input features: existing 16 recurrent activations, local three wall rays, actual 2-motor command, six motor×ray products and bias. All are trained with the SAME REAL after-three-physical-steps sensory target and the ORIGINAL R4 fixed online LMS update LR=.014, once per actual training trial (2,400 updates per head, 7,200 total). Output is current ray plus learned clipped offset, 3 channels.
  - **TRUE:** receives the actual physically executed command matched to actual outcome.
  - **BLIND:** motor command/interaction channels ZERO, so fewer active parameters although same 84 allocated slots.
  - **SHUFFLED:** the SAME full 84-feature capacity and SAME within-snapshot action multiset, but associated each sample's actual sensory result with the NEXT physically executed command in a cyclic five-way rotation from the same exact original pre-state. No invented target, no additional training data.
- Training occurs in fixed single pass scene/seed/tick/action order, not by selecting best seeds or repeating epochs. Full model weights **frozen before heldout evaluation**, all 3×84 head coefficients unchanged throughout 2,400 test forced trials. No test-label fitting, no agent motor planning, no neural evolution, no global map or food.

## Main heldout result, 48 NEVER-SEEN worlds and forced motor settings

| Predictor for terminal actual wall rays after 3 real motor steps | Mean physical outcome MSE (lower better) |
|---|---:|
| No learned model: repeat current sensors (persistence) | 0.061439654 |
| Learned MOTOR-BLIND head | 0.056731486 |
| Fully capacitated SHUFFLED-ACTION head | 0.059441375 |
| **TRUE trained head paired with actually executed motor actions** | **0.048915327** |

Actual physically matched motor experience reduces heldout prediction error by:
- **13.78% vs MOTOR BLIND**; preregistered ≥10% required: **PASS**.
- **17.71% vs capacity-matched SHUFFLED-ACTION training**; ≥10% required: **PASS**.
- **20.38% vs raw sensor PERSISTENCE**; ≥15% required: **PASS**.

Exploratory 95% paired bootstrap over **24 independent heldout WORLD SEEDS**, with both worlds nested per seed (never falsely treating intervention frames as independent trials):
- TRUE-minus-MOTOR-BLIND average MSE **−0.00781616**, exploratory 95% CI **[−0.01038620,−0.00567200]**.
- TRUE-minus-SHUFFLED average MSE **−0.01052605**, exploratory 95% CI **[−0.01385919,−0.00766752]**.

This is a meaningful **out-of-sample action-conditioned sensory prediction advantage** under real training and heldout physical interventions, not just a motor-input weight significance result from R4 or a label mismatch from R5.

### Physics / actual intervention sufficiency

In heldout worlds, **416/480 =86.67%** of otherwise identical starting-state snapshots separate at least 0.04 distance units after externally forcing −0.55 vs+0.55 steering for 3 physical steps, vs the preregistered minimum 80%. **900/1,920** heldout forced-vs-natural physical sensory contrasts were informative (max real three-wall-ray change ≥0.02), appearing in all 24 world seeds and both scene families (478 nearfield, 422 ordinary). There are enough genuinely physical action effects to challenge the model, even with never-trained actuation magnitudes.

### Crucial remaining failure: accuracy of the measured *differences caused by actions*

For each informative heldout forced-vs-natural contrast, both sides are ACTUALLY physically executed clones of the SAME starting body/world state. Predict each of their terminal three-wall rays with the SAME frozen TRUE head and subtract its predicted values. Score that delta against the REAL measured interventional sensor delta; compare with the no-action-effect baseline which predicts delta zero.

| Heldout physically informative action-effect contrasts | True-head predicted delta MSE | Predict no action effect MSE | Relative improvement |
|---|---:|---:|---:|
| **Both world families** (900 pairs) | **0.033227615** | 0.038572136 | **13.85%**, below locked 20% → FAIL |
| Nearfield (478 pairs) | 0.039049315 | 0.043002796 | **9.19%**, below locked 10% → FAIL |
| Ordinary obstacles (422 pairs) | 0.026633367 | 0.033553521 | **20.62%**, clears 10% scene requirement only |

All 1,920 unselected heldout action-effect contrasts also archived: TRUE predicted delta MSE 0.01596142 vs zero effect 0.01809819, ~11.8% advantage. This is a real but **incomplete prediction of causal sensory differences**, and it is not legitimate to declare R6 successful by weakening the 20% predeclared threshold or reporting only the favorable world generator.

## Full original 17 preregistered gates

**10/10 integrity gates PASS:**
1. All 96 full natural lives / 960 fixed snapshots / 4,800 actually executed 3-step physical interventions / 3,840 real contrasts; entire exact deterministic replay.
2. Legal actual R1 physical body, wall collision, energy/integrity, absorbing death and identical cloned pre-states.
3. Natural first-step experimental physical clone agrees EXACTLY with independently advanced natural organism.
4. Original frozen 16 recurrent neurons, 8 actual local sensors, 434 original coefficient slots.
5. Three exactly 84-weight zero-initialized heads, original fixed LMS .014, updated only in actual physical training worlds, original motor weights unchanged.
6. Correct zero-action blind input and exact same actually executed rotated action multiset in 84-weight shuffled control, same sensory target and 2,400 updates each.
7. Distinct test-world seeds and unseen forced action magnitudes; no heldout predictor update or head reparameterization.
8. Five declared bounded experimentally forced motors per actual original state and predictions before actual observations; no hallucinated physical effect.
9. Both unchanged 10-obstacle procedural world families with 24 different valid maps each in BOTH train/test cohorts.
10. Full original/forced physical trajectory/weight raw data, exactly matched replay and false food/intelligence/evolution claims.

**Scientific gates 5/7 PASS, 2 FAIL:**
11. Real physical three-step left/right steering effect ≥80%: **PASS**, 86.67%.
12. Sufficient physical sensor informative pairs, seed coverage and two-scene coverage: **PASS**, 900/24 seeds/478+422.
13. TRUE heldout absolute MSE ≤90% BLIND: **PASS**.
14. TRUE heldout absolute MSE ≤90% equal-capacity SHUFFLED: **PASS**.
15. TRUE heldout absolute MSE ≤85% PERSISTENCE: **PASS**.
16. TRUE model informative heldout real action-delta MSE ≤80% zero-delta baseline: **FAIL** (only 13.85% lower).
17. TRUE model informative action-delta MSE ≤90% zero in BOTH unseen scene families: **FAIL** (nearfield only 9.19% lower).

**OVERALL 15/17, FROZEN DEVELOPMENT SCIENCE SCREEN FAIL.** Science result is a qualified positive for **out-of-distribution absolute action-conditioned prediction** and a clear negative for the preregistered level of **specific differential causal effect accuracy**. Software/test success must never be conflated with science screen success.

## Full independent physical + learning audit

Original [successful GitHub Actions run #45476065724](https://github.com/kkongiu/neuroevolution-lab/actions/runs/45476065724): original unchanged R1–R5 regression unit suites + R6 physics/frozen-head tests all PASS, first completed entire study and a **second entire train+test replay byte-identical**.

Artifact: \`flyloop-r6-intervention-trained-heldout-transfer\`, GitHub artifact ID **14959092929**, retained through **24 December 2026**. Contains two complete full raw JSON files **36,270,098 bytes each**, both SHA256:
\`cea9832270268fcac38920e3ce6e08f6a08e76bc891c52d2dcffd94b6b980b28\`.
The canonical SHA256 of all 96 raw life+clone+sensory training records:
\`4afb66e57600786a664e43aeca7ea68955c19774b3de930c6d49f6b7a0772e7e\`.

A separate standalone Python 3 standard-library auditor, with **zero imports of any R1–R6 neuroevolution source code**, independently reimplemented and validated:
- Both random 10-rectangle map generators and 96 original full worlds;
- Deterministic original 434 coefficient generator, all 16 recurrent tanh unit states, seeded Gaussian motor perturbations and original executed 2-motor commands for **23,040 actual natural body frames**;
- R1 physical continuous actuator inertia, bounds/walls, collision, energy/integrity and 3 wall rays;
- Actual 5-way cloned 3-step physical alternatives for **4,800 real forced interventions**, **14,400 physical forked integration frames**, **3,840 real action-effect contrasts**;
- All **7,200 exact chronological head LMS updates** from 2,400 actually physical training targets, original shuffled-motor association, pre-action predictions and original vs cloned natural-physics identity;
- Frozen 3×84 trained weights, never modified by any heldout test outcome; reconstructs SHA-256 of final heads;
- All actual 2,400 heldout model prediction MSEs, 900 physical informative effect deltas, independent primary result summaries and full 96-row SHA-256.

**ZERO independently observed discrepancies**, exact original/replay full dataset checksums. Standalone audit source and downloadable audit JSON accompany raw GitHub artifact. This is computational validation of a synthetic physical simulator, NOT empirical biological evidence.

## Where this leaves the larger FlyLoop hypothesis

R1: continuous self-running neurons can generate motion, but not good exploration.
R2: a researcher-designed physical progress/contact objective changes motor readouts and lowers collisions yet tends to confine movement.
R3: one-step actual local sensor consequences become predictable, but reducing surprise does not improve locomotion.
R4: a command observed only as a predictor feature adds limited association, without real physical interventions.
R5: true physical forced motor interventions do causally change sensor/body effects, but an ordinary-policy-trained predictor does not robustly generalize.
**R6: training on real varied physical interventions produces a substantively BETTER action-conditioned predictor in completely NEW worlds and forced-action magnitudes, but not enough precision in sensor EFFECT DELTAS, and no agent-level motor self-control is tested.**

Recommended R7 hypothesis (NOT performed or authorized to evolve): a learnable short-horizon *change-specific* predictive representation that trains directly on the differential effects of two ACTUALLY executed actions from the SAME state, with matched shuffled-/blind-motor controls. Hold out entire world seed sets and new actuation strengths AGAIN; measure both one-model action-effect error and downstream autonomous navigation separately. Avoid adding engineered "escape" or "forage" routines and do not retrofit R6.

**All flags remain false:** autonomous_control_demonstrated, food_search_demonstrated, escape_reflex_learned, autonomous_goal_invention_demonstrated, general_intelligence_demonstrated, biological_fly_replication_demonstrated, evolution_authorized, advance_to_evolution.

**PR #96 remains DRAFT** stacked on R5 #95. Main, sealed M5.6 and all earlier M0–M4/R1–R5 untouched.
