# FlyLoop R5 — Real physical interventions confirm motor consequences, NOT learned causal prediction

**24 November 2026. Frozen DEV science screen: FAIL, 10/16 fixed gates.** Software CI **PASS**; twice-executed whole scientific output byte-identical; independently reconstructed actual physical branch outcomes and model weights **PASS**. This report retains *all* 48 original world lives, including two early organism deaths. The experiment does NOT show that a fly-like organism knows how to control itself.

## The causal distinction that R4 left open

R4 switched the motor-command INPUT to a predictor while the organism physically executed its original motor command. That establishes only a predictive *association*. R5 instead takes an EXACT cloned pre-action body/environment state, with the same frozen 16-neuron pre-action state and 3 already-trained prediction heads, then forces and PHYSICALLY EXECUTES one of five different motor commands in FIVE independent real R1-physics branches. This yields genuine experimentally induced sensory and positional effects.

R5 asks separately:
1. **Can different imposed actions actually cause different real sensor/body outcomes?** YES, strongly.
2. **Does a frozen action-conditioned predictor understand those forced consequences better than a motor-blind or older-motor comparison?** Only a modest advantage over motor blind, **not** over older-motor control or at the predeclared performance threshold.
3. **Does the agent choose better actions autonomously?** NOT TESTED. The forced actions belong to the experimental investigator, NOT to the organism.

This result corrects an important interpretation risk: positive causal *physical actuator sensitivity* is NOT evidence that the learned neural model represents the causal mapping well, nor evidence of autonomous agency, foraging or fly behavior.

## Original fixed protocol and disclosed pre-completion erratum

[FLYLOOP_R5_PROTOCOL_2026-11-24.md](FLYLOOP_R5_PROTOCOL_2026-11-24.md) was committed **before** the full experiment. Exactly 24 NEW world seeds **10300–10323**, two independently generated 10-obstacle physical arenas, 48 original continuous lives. Three R4 action-conditioned sensory heads (each 84 weights, all initialized zero) train on the FIRST **300 real** action-observation steps of the original recurrent-policy trajectory, then ALL THREE and the initial 434 neural coefficients stay **FROZEN** for testing. No motor learning, no food, no path-planning, no evolutionary selection.

At t=300,310,...590, five identical-world physical branches were planned:
- \`natural\`: actual normally executed recurrent motor command;
- \`steer_left\`: same throttle, experimentally forced steering = −0.85;
- \`steer_right\`: same throttle, experimentally forced steering = +0.85;
- \`throttle_low\`: experimentally forced throttle = 0.05, original steering;
- \`throttle_high\`: experimentally forced throttle = 0.95, original steering.

Each trial physically executes a REAL first motor step, produces a REAL next three-wall-ray observation and proceeds up to six real motor steps. All five branches start from the same full pre-action body/world state; original neural state and three trained head weights are fixed; the original natural world independently continues once. Its actual one-step outcome must be IDENTICAL to the natural experimental clone. The true head receives the forced motor; blind head has all motor inputs zero; yoked head has its own genuinely previously executed motor at least six ticks old. All three are evaluated BEFORE the forced physical outcome and trained only on previous actual natural steps, never on experiment labels.

**Transparent QA correction before completed data:** The first attempted full CI evaluation hit a physical experimental clone that died before the sixth forced motor integration, and the code initially raised an error. BEFORE any completed science result, we committed a documented physical-death-handling erratum: a dead body becomes an **absorbing state**, no resurrection, never execute actions on dead bodies, record the number of physically executed frames, and retain its last real position as the six-step-horizon terminal position. The *same* locked seeds/snapshots/commands/16 evaluation gates remain. Original lives that die early are **never substituted**: their missing scheduled snapshots cause the **exact-count gate to FAIL**. The original Git protocol and the erratum are both preserved. This fix neither rescued any prospective criterion nor changed model dynamics, neuron parameters or live-episode physical outcomes.

## Completed raw experiment and quality screen

The originally planned 48 ×30 fixed snapshots ×5 branches = 7,200 realized single-step trials was **not fully reached** because **two original organisms died before step 600**. All original worlds are preserved, no replacing or cherry-picking.

| Metric | Actually observed |
|---|---:|
| Original independent physical lives | **48** (2 ended early) |
| Actual original real motor steps | **28,447** |
| Exact-state experimental snapshots actually reached | **1,406** / 1,440 planned |
| One-step physically executed forced intervention branches | **7,030** / 7,200 planned |
| Actually realized forced-vs-natural sensory contrasts | **5,624** / 5,760 planned |
| Contrasts with real sensory effect max |Δray| ≥0.005 | **2,121** (37.71% of available) |
| True physical six-step steer-left vs steer-right position divergence ≥0.05 | **1,372 / 1,406 = 97.58%** |
| Mean physical terminal position difference between forced steering extremes | **0.590** world units |
| Experimental branches dying before completing six physical integrations | **5**, all correctly censored/absorbed |
| R4-style original neural coefficients changed | **ZERO** |
| Real head LMS updates reconstructed by independent audit | **43,200 = 48 ×300 ×3** |

**The externally imposed actions genuinely affect movement**, strongly. An identical initial body/world/neural state produces different subsequent real positions under the different *actually executed* commands. This is a **physical causal effect of the motor actuator**, not evidence of an independently selected behavioral goal.

### Predictive quality on ALL 7,030 real off-policy one-step motor interventions

| Frozen one-step sensor predictor | Mean actual-real-future 3-ray MSE (lower better) |
|---|---:|
| Current sensor unchanged (persistence) | 0.021936471 |
| Motor-blind trained head (zero action inputs) | 0.017907916 |
| **TRUE trained head receiving the actual forced motor command** | **0.017188108** |
| YOKED trained head receiving a past motor command ≥6 steps old | **0.017163002** |

The true head beats the motor-blind head by **4.02%**, but misses the preregistered **5%** required threshold. It is in fact **0.15% worse** than the independent older-motor yoked head. Exploratory *paired WORLD-SEED* bootstrap95 intervals for true-minus-blind MSE are **[−0.0012103, −0.0002481]** (small positive advantage over motor-blind in this synthetic cohort), while true-minus-yoked are **[−0.0002750,+0.0002567]** (no reliable separation). This is NOT a successful out-of-policy action-causal predictive model.

### Predictive quality of ACTUAL measured forced-vs-natural sensory contrasts

For each of 5,624 actually physically forced contrasts, use the truly measured next-sensor difference between forced action and natural action from the exact same starting state. The only predicted contrast is the SAME frozen true head evaluated on forced vs natural motor. Compare against the zero-difference null predictor. No counterfactual label was invented.

| Contrast population | Frozen TRUE model delta MSE | Predict no sensor effect (zero) MSE | Relative model improvement |
|---|---:|---:|---:|
| ALL 5,624 contrasts | 0.00286930 | 0.00295494 | 2.90% |
| 2,121 PREDECLARED physically sensor-informative contrasts | **0.00735286** | 0.00783244 | **6.12%** |
| Nearfield informative contrasts (n=1,169) | 0.00717559 | 0.00789388 | 9.10% |
| Ordinary-obstacles informative contrasts (n=952) | 0.00757054 | 0.00775700 | 2.40% |

The **6.12%** improvement on real informative contrasts misses the preregistered **10%** threshold. The nearfield result alone passes a 5% per-scene threshold, but the ordinary-obstacles environment does NOT, so the original **both-scenes** gate also FAILS. No post-hoc narrowing to the favorable environment or the selected 2,121 useful effects is permitted to declare science success.

## All 16 prospective gates, fixed and truthfully scored

**Integrity: 8 / 10 PASS**
1. Exactly 48 complete 600-frame natural lives / 1,440 planned snapshots / 7,200 physical branches / 5,760 contrasts: **FAIL** (48 lives but 2 die early; 1,406/7,030/5,624 only). No substitution.
2. Identical pre-action cloned body/world and legal physical motor integration: PASS.
3. Natural experimental clone exactly matches independently advanced original physical life: PASS.
4. Original 16-neuron/8-local-sensor/434-neural-coefficient body and brain frozen: PASS.
5. Three 84-coefficient heads trained on 300 real earlier steps, then frozen: PASS.
6. Exactly five bounded externally forced commands at each ACTUAL snapshot: PASS.
7. Yoked action from same actual organism at least six old ticks, no future or fabricated history: PASS.
8. All predictions pre-action, all targets from actually physically executed branches: PASS.
9. Two different exogenous valid 24-map obstacle sets: PASS.
10. Every planned 30-snapshot raw population complete: **FAIL** because two original lives died before some predetermined snapshots. All *actually available* raw actions and outcomes ARE archived in full; none discarded.

**Scientific efficacy: 2 / 6 PASS**
11. ≥60% actual six-step forced steer-left vs steer-right trajectories diverge at least 0.05 units: **PASS (97.58%)**. This verifies actuator physics and is NOT learned navigation.
12. ≥100 physically sensor-informative contrasts present in ≥20 seed worlds: **PASS (2,121; all 24 seeds)**. Real sensory effects exist.
13. True model held-out real-sensor MSE ≤95% motor-blind: **FAIL** (4.02% better, <5%).
14. True model held-out real-sensor MSE ≤95% yoked: **FAIL** (true head slightly worse).
15. True model informative physical contrast-MSE ≤90% zero-contrast: **FAIL** (6.12% better, <10%).
16. True model physically informative contrast-MSE ≤95% zero in BOTH worlds: **FAIL** (nearfield 9.10% benefit; ordinary geometry 2.40%, below 5%).

**OVERALL 10 /16, preregistered DEV science screen FAIL.** A successful GitHub Actions workflow does not make the predeclared model efficacy criteria pass. The two passed science gates show physics/effect sufficiency, not successful learned counterfactual model or agency.

## Two complete reproducible runs and independent external reconstruction

- Official green [GitHub Actions #45444759811](https://github.com/kkongiu/neuroevolution-lab/actions/runs/45444759811). Unchanged R1–R4 unit/regression suites PASS, R5 cloned-world unit/physical-death regression tests PASS, the entire finished R5 study executes TWICE byte-identically.
- Workflow artifact **flyloop-r5-physical-cloned-state-interventions** ID **14943764313**, 30-day retention through **2026-12-24**.
- Two FULL raw JSON outputs **48,026,368 bytes each**, SHA256 both:
  \`5d5c4aa6ea8f6125a8d4d2c1d26147009fb3c559cca5eac43b5f933650bca103\`.
- Independently recomputed compact canonical SHA256 of all 48 full original-life raw rows (including all forced branches):
  \`1fc3f301d86ed008e949c35660cf462356f4dbf1c407521608fbb34a231c3556\`
  — exactly agrees with R5's recorded source value.

A separate **independent Python standard-library audit script WITHOUT importing any R1/R2/R3/R4/R5 model modules** reimplemented:
1. Exogenous two 10-obstacle geometry generators from original seeded equations;
2. Original initialized 434 neural coefficients, **all 16 recurrent units**, input sensors and independent policy motor-noise stream;
3. Legal actual physical motion, actuator inertia, boundary and rectangle impacts, death and physiological decrement;
4. Every one of the three original 84-weight head LMS updates over first 300 REAL action/sensation pairs, with exact original training observations;
5. Old-action yoke provenance ≥6 original steps, pre-action prediction and 5-way cloned state physical action interventions;
6. All actual next-ray measurements and all model MSEs, actual forced-vs-natural 3-ray differences, informative selection and six-step real position divergences;
7. Five cases of physical early-death absorbing experimental clones; two physically dead natural lives not replaced;
8. Full per-head frozen predictions, original vs natural cloned action identity and complete numerical summaries.

Independently verified **28,447 original motion frames, 7,030 forced first-step physical trials, 5,624 exact-state action contrasts, 43,200 model training updates, 2,121 informative contrast effects, 1,372 divergent steering probes**, with **ZERO independent audit discrepancies**. An external detailed audit JSON and reproducible standalone Python source accompany the full downloaded artifact.

This is computational consistency validation of a **synthetic** world, not a validation against real insect physiology or actual animal experiments.

## What to test next

R5 closes the biggest ambiguity from R4: **physical motor commands do cause different real body and sensor outcomes** under controlled intervention. However, the learned action-conditioned predictor only modestly anticipates the differences and is no better than the model trained with old motor commands. The bottleneck is **how the nervous circuit represents and uses action-dependent consequences**, not lack of physical motor effects.

Future R6 (NOT performed and no authority to evolve): train the model on an explicitly precommitted mixture of physically executed motor interventions in a TRAINING subset of worlds, then test on completely HELD-OUT new world seeds and new motor action settings. Compare with identically complex motor-blind and shuffled-command controls. This could discriminate **lack of action-intervention exposure** from an inadequate neural/sensory representation. Do not silently use the current test interventions to fit R5 retrospectively.

**All STOP FLAGS false:** autonomous_control_demonstrated, food_search_demonstrated, escape_reflex_learned, autonomous_goal_invention_demonstrated, general_intelligence_demonstrated, biological_fly_replication_demonstrated, evolution_authorized, advance_to_evolution. Main/M5.6/M0–M4/R1–R4 unchanged; R5 is a draft stacked on R4 PR #94.
