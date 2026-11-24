# FlyLoop R7 — online predictive motor closure improves exploration, but NOT collisions or causal prediction enough

**24 November 2026 · Completed, reproducible, independently audited. Original preregistered DEV science screen FAIL: 15/17 gates pass (10/10 integrity, 5/7 efficacy).** This is the first R-series experiment in which a within-life learned local sensor model **actually changes the physical motor command chosen at the next step**. The finding is a *qualified behavioral improvement*, not yet safe navigation, independent goal invention, food seeking, escape or a biological fly.

[Immutable prospective protocol](FLYLOOP_R7_PROTOCOL_2026-11-24.md) was committed **before the implementation or complete R7 simulation**, and was not retuned after the results. All original R1–R6 source, M0–M4, sealed M5.6, and main remain unchanged.

## What actually exists and what is deliberately engineered

A continuously active R1 circuit has **16 recurrent tanh neurons** receiving only **8 elementary local sensor/body signals** and generating two continuous analog outputs for forward speed and turning. The original 434 neural coefficients remain fixed in all R7 arms except the explicitly identified R2 positive-control arm, which follows R2's existing output-only physical-progress/contact-learning rule.

For the first time R7 adds **THREE zero-initialized, 84-coefficient sensory prediction heads that learn during the SAME organism's lifetime** from physically executed movements and actually observed next wall rays. No head weights are imported from the externally trained R6 study. From frame 60 onward, the *correct-action* head can influence movement by evaluating **five small local variations around the existing analog motor command** (original base, ±0.35 turn and ±0.20 throttle). The real command chosen maximizes the model's predicted change of the three local wall sensors **relative to what the model predicts for the base command**. An action is physically executed ONCE and then the model learns from the actual sensory result. No physical lookahead, clone or global map is available inside the organism.

**This five-candidate menu and 'maximize predicted sensor variation' objective are researcher-coded.** They must not be misrepresented as an emergent strategy, a self-discovered wish to explore, or a learned wall-avoidance law. The actual upstream recurrent motor coefficients DO NOT learn in the R7 true-guided arm; its behavior adapts because its online learned prediction head changes the action that is physically executed.

**Six paired controls**: original recurrent frozen; same frozen motor with a *passively* trained head; true model-guided loop; historically mispaired actual motor-input model-guided loop (same organism's actual command ≥6 previous ticks old, independently sampled); uniformly random local 5-command menu; and original R2 physical progress/contact readout-plastic motor benchmark. All get the same initial random neural matrices, body, map and exogenous R2 seeded motor noise. Passive model learning alone MUST and DOES produce exactly the same physical trajectory as the frozen motor.

## Exactly preregistered study, no cherry-picking

- **16 new seeds 10900–10915** ×2 physical obstacle generators (\`obstacles\` R1 and \`nearfield\` R2) ×2 matched regimes (\`static\` and \`removal\`) ×6 arms = **384 persistent continuous-body lives**.
- Every life completes **480 real R1 physical motor integrations**, 184,320 real physical steps overall. No original organism dies early; no missing seeds or replacement episodes.
- \`removal\` starts with EXACTLY the same ten real obstacles as \`static\`, then removes exogenously selected original obstacle indices 1,3,5,7,9 immediately **BEFORE** the 240th-step sensor reading. Its first 240 physical/neural/action frames are EXACTLY identical to the corresponding static world for each seed/geometry/arm. The switch signal is not given to the brain: changes are accessible only through real proximity readings.
- Full initial/final neural matrices, all three final predictor heads, real sensor and 16-neuron states per time step, five action candidates, their complete pre-action predicted three-sensor readings, chosen command, real post-action sensors, actual body pose/contacts/health/energy, heads' historic motor input source, update records and every relevant trajectory metric archived twice. All original and repeat complete JSONs are BYTE-IDENTICAL.

## PREDECLARED PRIMARY outcome — nearfield obstacle removal, post-switch frames 240–479

**16 independent world seeds** for each control, equal exposure of **240 actual physical steps** each.

| Controller | Mean unique 1×1 cells post-switch | Pooled real collisions per meter | Post-switch RMS route radius | Model-driven action overrides |
|---|---:|---:|---:|---:|
| **Frozen recurrent control** | **5.125** | **0.733296** | 0.5582 | 0% |
| Passive online predictor, NO action influence | 5.125 | 0.733296 | 0.5582 | 0% |
| Uniform RANDOM choice from same action menu | 6.750 | 0.745084 | 0.6479 | 79.1% |
| Historic-yoked motor-model guides actions | 7.188 | 0.843963 | 0.7355 | 95.2% |
| **TRUE online learned model guides actual actions** | **8.250** | **0.744028** | **0.8763** | **96.0%** |
| R2 physical progress/contact reward controls original neural motor weights | 7.500 | **0.057423** | 0.7951 | 0% |

**Broadens real exploration:** TRUE model explores **60.98%** more grid cells than frozen, **22.22%** more than random-menu, and **14.78%** more than historic-yoked model selection. It also physically enters **3.3125 genuinely new post-switch cells not visited before switch**, compared to 1.0625 for frozen; this latter endpoint is informative but secondary. More spatial spread is also supported by larger RMS route radius (0.8763 versus 0.5582); this is not simply the low-collision tight-orbit failure from R1 and R2. Its learned model changes the real base motor on 96.02% of post-switch physical actions; head learning is therefore behaviorally connected.

**BUT collision density does not improve:** TRUE model reports **0.7440 collisions/m** vs frozen **0.7333** (+1.46%, NOT the preregistered −10% target), while R2's simpler hand-shaped progress/contact motor learner gets dramatically lower **0.0574 collisions/m** in the same changed geometry. This is NOT a successful *safe* obstacle-avoidance system. On \`nearfield/static\` no-removal worlds, TRUE predictive novelty motor collision density is **1.2388 collisions/m versus frozen 0.6791** (~82% MORE collisions), despite TRUE coverage 8.00 vs 5.25. A model selecting predicted sensory variation can be drawn into sensor-rich obstacles and thus hit walls more, making safety distinctly worse in some scenes.

### Causal/predictive model control

On the EXACT SAME passive-world physical trajectory, true and ≥6-step-historic-yoked models score the real sensory outcome:
- TRUE model postchange next-ray MSE **0.013329187**;
- OLD-motor-mispaired YOKED model next-ray MSE **0.013534825**.

TRUE is only **1.52%** lower-error, below the prospectively required **5%**. Thus gate 11 FAILS: the stronger exploration does **not** establish a sufficiently robust correct action-conditioned prediction advantage. It may partly reflect model output amplitude and the researcher-designed selection rule rather than accurate learned causal motor consequences.

### Uncertainty: do not inflate 16 seeds into 3,840 independent steps

An exploratory **paired seed-level 95% bootstrap**, with 16 independent seeds rather than 240 temporally correlated frames each:
- TRUE minus frozen postchange cell coverage: mean **+3.125 cells**, exploratory interval **[+1.125,+4.875]**.
- TRUE minus random menu: mean **+1.500**, interval **[0.000,+2.8125]**.
- TRUE minus older-yoked head: mean **+1.0625**, interval **[−0.1875,+2.1875]**.
- TRUE minus R2 progress benchmark: mean **+0.750**, interval **[−1.625,+2.6875]**.

Prereg coverage thresholds are exceeded against frozen, random and yoked controls; however, the interval against the **historically mispaired model includes zero**, and the interval against random touches zero. It would be too strong to declare a robust advantage specifically caused by *correctly learned* action outcomes. The signal is clear against frozen movement in this small, synthetic cohort.

In the second \`obstacles/removal\` geography, TRUE covers 8.250 cells vs frozen 6.625 (+24.5%) but incurs 0.6391 vs0.5286 physical collisions/m, again exposing the exploration/safety trade-off. No favorable-only geometry was substituted into the preregistered primary test.

## Full original 17 fixed gates

**Integrity 10/10 PASS:** full 384 balanced and exact twice-repeated result; original R1 no-penetration/energy/injury physics; same initialized 434 neural weights, body/map and seeded motor noise across arms; no future/map/food sensor; heads all zero at birth and trained ONLY after real physically observed sensations; authentic five continuous candidates based solely on preaction model states; historic yoked source is at least six real executed steps old; passive trained head physically and exactly matches frozen baseline; static and switched worlds identical until frame240 and remove exactly five walls; all raw physical/neural action and explicit stop flags archived.

**Scientific efficacy 5/7 PASS, 2 FAIL:**
- **11: FAIL** — passive head correct-motor predictive MSE not ≥5% better than past-yoked head (only 1.52%).
- **12: PASS** — true action-guided model's post-switch nearfield coverage ≥110% frozen.
- **13: FAIL** — true action-guided post-switch nearfield contact density not ≤90% frozen.
- **14: PASS** — coverage ≥110% historic-yoked-guided.
- **15: PASS** — coverage ≥110% uniformly random same-menu control.
- **16: PASS** — route RMS radius ≥90% frozen; no trivially tiny circle.
- **17: PASS** — model physically overrides base continuous motor on ≥10% of actual postchange steps (observed 96.0%).

**TOTAL 15/17 PASS but the prospectively required all-criteria R7 DEVELOPMENT-SCIENCE SCREEN FAIL.** A working learning-action loop, plus broader spatial exploration, is a genuine engineering accomplishment. It is NOT validated effective adaptive navigation, safety, independent internal goals, fly-like intelligence or learned escape.

## CI, negative control integrity and external independent audit

Official successful [GitHub Actions run #45479176516](https://github.com/kkongiu/neuroevolution-lab/actions/runs/45479176516), all unchanged R1–R6 regression suites PASS, all R7 unit tests PASS after a narrow **test-only** fix to an assertion comparing mathematically equal binary floating-point values \`−0.45\` and \`−0.449999…\`. No model, seed, environment, scientific gate or coefficient was changed following outcome.

GitHub artifact \`flyloop-r7-within-life-online-action-results\`, ID **14961447843**, retained until **24 December 2026**. Two full complete JSON outputs of **369,408,944 bytes each**, EXACT identical SHA256:
\`b9db6c8e3926cee54c5d4dac1d04409700f5f5f8b8c2f64341026a74aa5e02e7\`.
Canonical complete 384 raw-life JSON SHA256:
\`163009042737b31fbb08c909e6093d74548a2fb516aa4474e551bb6b432d5294\`.
Downloaded ZIP SHA256:
\`3e7fda2a224169cfee1d9ff7cf9791a349550c29fc36cd48ee09ef4c403bf87f\`.

**Independent external audit without importing R1–R7 simulation modules:** all **384 lives**, **184,320 real physical frame records**, historic-yoked motor origins, environmental transition, physically legal obstacle clearance, body continuity, every true/old sensor prediction loss, head training-event count and paired frozen/passive plus pre-change static/removal trajectories verified against downloaded raw data and exact repeat/hash. **96 lives selected in advance for deep external replay**, namely seeds 10900,10905,10910,10915 ×both geometries×both regimes×all six controls (25% of cohort): individually reconstruct R1 physical actuator dynamics, wall-ray geometry, deterministic 16 recurrent neural transitions, R2 seeded motor perturbations, three 84-weight online prediction heads, preaction candidate predictions and selection scores, authentic old-action sham provenance and R2 positive-control motor adaptation. These 96 deep lives cover **46,080 real body steps**, **115,200 independently recomputed head learning updates** and **460,800 independently scored model/candidate combinations**. No discrepancies outside declared small cross-Python floating tolerances. (Audit used Python 3.13 while GitHub used 3.11, so very close model candidate-score ties may choose a different numerical maximizer; it recorded **2,345 near ties in the 96 deep lives** and checked that the ORIGINAL logged chosen motor was within tolerance of the source preaction model score; the raw model score itself and each logged real motor are retained exactly. Do not represent the deep sampled audit as complete reexecution of all 384 lives.) No animal experiment, biological or real-world validation is implied.

The independent JSON report and standalone audit Python script are available as separate downloadable files in the conversation; the complete experiment and software audit run at the same full original cohort scope on GitHub Actions.

## Scientific interpretation and next bounded test

R7 establishes a **real within-organism online learning→continuous motor selection link**: the fixed recurrent circuit can adapt its executed physical motor commands in response to the head's ongoing actual sensory training and can traverse more of the same synthetic physical space than a frozen motor, passive model and the predeclared comparators. But the model's *correctness* advantage over the sham head is weak and the model actively increases wall strikes in some environments. We have therefore **NOT demonstrated robust sensorimotor competence or autonomously self-discovered navigation**.

A defensible R8 (NOT run here) would keep the original continuous neural body and same-world/seed controls, introduce no global path/map or programmed escape macro, and independently test whether a predictive objective that trades sensory novelty against **physically learned contact consequences** can lower collision density without losing spatial coverage. Crucially predeclare the full collision–coverage Pareto requirements, noise/action-menu ablations, changing-world generalization and separate the objective provided by the researcher from any genuine autonomously acquired objective.

**Stop flags remain FALSE:** food_search_demonstrated, escape_reflex_learned, autonomous_goal_invention_demonstrated, general_intelligence_demonstrated, biological_fly_replication_demonstrated, evolution_authorized, advance_to_evolution. \`closed_loop_action_selection_implemented=true\`, but \`effective_adaptive_navigation_demonstrated=false\`.

**PR #97 remains DRAFT stacked on R6 #96**, main/M5.6/M0–M4/R1–R6 source untouched.
