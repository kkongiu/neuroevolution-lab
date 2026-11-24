# FlyLoop R2 — verified: online motor learning reduces contacts but NARROWS exploration

**24 November 2026 · preregistered science DEV screen FAIL (10/13, with all 9/9 integrity gates passing and only 1/4 efficacy gates passing).** The finished GitHub Actions software suite and full deterministic replay **PASS**.

[Prospectively fixed protocol](FLYLOOP_R2_PROTOCOL_2026-11-24.md) committed before the first full 288-life science comparison. No scientific threshold, world seed, learning rate, neural parameter, exploration noise or actual outcome was retuned after observing results. Original FlyLoop R1, M0–M4, sealed M5.6 and main are untouched.

## What was actually tested

This is the FIRST experiment in the R branch with **motor plasticity during an organism's life**, not food foraging, avoidance reflexes or natural evolution. The existing 16 recurrent tanh neurons continuously drive two real-valued physical actuators (speed/turn). All 434 initial neural coefficient slots are shared across controls, but **only 34 motor-output readout coefficients can adapt** in plastic arms. The recurrent synapses, sensor representation and 16 hidden-unit biases remain fixed.

A researcher-imposed body consequence is computed AFTER each actual movement: net 12-tick physical displacement normalized by maximum physical speed, minus 1.2 for actual wall contact. A fixed stochastic eligibility/EMA baseline rule (learning rate 0.0018) changes motor weights in response to motor noise and these measured physical consequences. The objective, feature engineering, Gaussian exploration and learning rule are all researcher specified. Thus **this is not self-invented purpose**, emergent resource search, "flee", or a fly nervous system.

Exactly **24 new world seeds 9400–9423 ×2 exogenous worlds ×6 matched controllers =288 lives**, each at most 600 continuous physical integration steps:
- \`recurrent_frozen\`: frozen recurrent readout with SAME additive random motor perturbations;
- \`recurrent_plastic\`: online readout plasticity using correct current physical progress+contact;
- \`recurrent_sham\`: same learner but reward taken from same organism's REAL experience ≥5 steps earlier;
- \`collision_only\`: plasticity given ONLY true physical contact penalty, not net progress;
- \`leaky_plastic\`: same learning but recurrent connection contribution removed;
- \`inertial_random\`: original R1 temporally correlated random motor driver, distinct policy baseline.

No food, simulated predators, reproduction, evolutionary algorithm or task-specific motor command. World layouts and initial pose/weights are the same within all six paired arms. The first scenario uses R1's 10 procedural obstacles on fresh seeds; the nearfield scenario uses a different exogenous 10-obstacle generator that places walls nearer spawn.

## Results (means over 24 world seeds)

### Difficult *nearfield* geometry — primary preregistered comparison

| Controller | Unique 1×1 zones | Contacts/meter pooled | Total distance per life | Mean trajectory RMS radius |
|---|---:|---:|---:|---:|
| Original frozen recurrent, with same motor noise | **29.46** | 0.847 | 64.27 | **3.42** |
| Correct physical-feedback motor plasticity | **23.21** | **0.644** | 66.57 | 2.32 |
| Feedback from ≥5 frames earlier (sham) | 28.29 | 0.887 | 65.17 | 2.86 |
| Physical collision cost ONLY | 26.67 | 0.762 | 65.72 | 3.03 |
| Leaky NO recurrence with same plasticity | **30.88** | 0.774 | 64.77 | 3.80 |
| Temporally correlated random movement | 28.79 | 6.156 | 41.01 | 2.84 |

Correct current physical feedback gives ~**24% less collisions per traveled meter** vs frozen (0.644 vs 0.847). But it also gives ~**21% LESS area visited** (23.21 vs 29.46), and a more tightly confined route (2.32 vs 3.42 RMS radius). The agent can appear physically "safer" by turning more and staying local. This is **not a successful exploration/avoidance demonstration**.

Correct-feedback plastic mean coverage minus frozen mean coverage: **−6.25 zones per world** (exploratory paired-seed bootstrap 95% **[−16.08,+2.46]**). Compared with older-feedback sham: **−5.08 zones**, bootstrap **[−12.96,+0.92]**. The estimated difference is not unequivocal under these small paired independent worlds, but **there is no evidence for the required positive gain**.

### Original procedural obstacle geometry — secondary, held out from gates

| Controller | Unique zones | Contacts/meter |
|---|---:|---:|
| Frozen matched recurrent | **33.38** | 0.732 |
| Correct-feedback plastic recurrent | 24.00 | **0.519** |
| Older-feedback sham | 27.75 | 0.646 |
| Collision-only plastic | 27.00 | 0.636 |
| Leaky nonrecurrent plastic | 26.17 | 1.168 |
| Inertial random | **39.63** | 3.644 |

Again the learned motor readout lowers contacts per meter but narrows traveled area. The negative observation is not an artifact of only the new nearfield geometry.

## Original prospective screening: 10 PASS, 3 FAIL

Integrity **9/9 PASS**:
1. Exactly 288 balanced lives, verified byte-identical complete repeat;
2. Legal continuous body physics, no wall penetration or resurrection;
3. Same world, initial pose and 434 initial weights for six paired arms;
4. 8 local neural sensory channels with no map/route/food hint;
5. Only 34 original motor-output coefficients can adapt (input/recurrent/hidden unchanged);
6. Actual net displacement over ≤12 steps and contact feedback only AFTER action;
7. Same motor perturbation sequence on overlapping live prefixes, sham only from actual past ≥5 ticks old;
8. 10 exogenous valid rectangles and 24 distinct worlds per scene;
9. Complete raw history and no food/reproduction/selection/evolution claims.

Science **1/4 PASS**:
- 10. Nearfield unique zones: correct feedback >= frozen ×1.10 — **FAIL** (23.21 vs29.46).
- 11. Nearfield collisions per traveled meter <= frozen ×0.90 — **PASS** (0.644 vs0.847).
- 12. Nearfield unique zones: correct feedback >= older-sham ×1.10 — **FAIL** (23.21 vs28.29).
- 13. Nearfield unique zones: correct feedback >= collision-only ×1.10 — **FAIL** (23.21 vs26.67).

**Overall preregistered scientific screen FAIL: 10/13.** No redefinition of success as "fewest collisions" after seeing confined/circular trajectories.

## QA bug correction before final validation

The first completed R2 code version incorrectly demanded that *every episode of every learning arm* finish with numerically changed weights, although the preregistered gate only demanded that exclusively motor readout weights be *eligible* for updating. In 16 legitimately collision-free, zero-gradient episodes the collision-only arm applied its rule but actual readout values remained unchanged. This is physically and mathematically valid.

A narrowly scoped **software QA gate fix** removed this false overconstraint, required genuine motor-weight changes somewhere within each learner's full cohort, and added a unit test explicitly checking legal zero-gradient updates. **The experiment protocol, seed worlds, neurons, actuator, learning rule, science gates and real metrics were NOT changed.** This corrected 1 false QA failure and did not rescue any efficacy test.

## Independent download and audit

- Completed green [GitHub Actions run #45428800720](https://github.com/kkongiu/neuroevolution-lab/actions/runs/45428800720).
- GitHub Actions artifact \`flyloop-r2-physical-motor-plasticity\`, ID **14938971501**, expires **24 December 2026**.
- Downloaded ZIP holds two full outputs, each **~277.2 MiB uncompressed**, SHA-256 independently recomputed and identical:
  \`6e2236ab016f09443a071015511d42f1e863aafc6805fdf4505b741c5e4011dc\`.
- Canonical SHA-256 independently reconstructed from full 288 raw life records:
  \`5ea66875c25f709221affc9a4dc0218f84f61f858662d03b11508c5c54b7c9bb\`.
- Independently verified **288 complete lives, 172,581 physical movement frames, 48 world/seed pairings and 28,560 older actual feedback references**, with **ZERO accounting/physics/causality errors**. Reconstructed position continuity, no solid wall/boundary penetration, no movement after collision, real energy/integrity cost, actual feedback from previous-window physical displacements and independently seeded shuffled training source time. Initial neural worlds/pairing consistent; no original input/recurrent/hidden coefficient modification.
- 16 valid zero-net-change episodes noted in QA audit; neither a bug nor a sign that arbitrary hidden coefficients were adjusted.

The independent audit is additionally supplied in a JSON file with the downloadable full raw artifact, separate from the R2 model's self-report.

## Interpretation and next falsifiable step

The internal recurrent loop *can* be continuously active and its output weights *can* change based on real bodily outcomes. Our explicit displacement/contact reward improves collisions-per-distance, but fails the more demanding need to explore. It may exploit curvature or limited-range cycling, as in R1.

A more effective R3 research test would avoid another hand-coded "move farther" instruction and instead isolate **sensory consequences of self-motion**, asking whether learning to predict the next wall sensor reading creates locomotor competence; do not inject "turn away if wall" or silently learn from absolute map coordinates. It needs matched no-learning, swapped-feedback and nonrecurrent controls and genuinely new geometries. Until supported, it is a research proposal, not an achieved autonomous organism.

All stop flags are FALSE: \`food_search_demonstrated\`, \`avoidance_reflex_learned\`, \`unprogrammed_objective_emerged\`, \`general_intelligence_demonstrated\`, \`biological_fly_replication_demonstrated\`, \`evolution_authorized\`, \`advance_to_evolution\`.

PR #92 remains DRAFT and stacked on R1 PR #91; **main and all earlier experiments are intact**.
