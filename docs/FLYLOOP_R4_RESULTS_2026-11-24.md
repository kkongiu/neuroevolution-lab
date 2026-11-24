# FlyLoop R4 — REAL motor command adds small predictive information, but does not improve motor behavior

**24 November 2026. Prospective R4 scientific result: FAIL, 11/18 fixed gates.** All **11/11 integrity** gates PASS, **0/7 scientific** gates PASS. GitHub software CI and full deterministic replay PASS. No prospective threshold, neural structure, action/sensor representation, world seed, model learning coefficient or motor learning rule changed after observing outcomes.

[Original frozen R4 protocol](FLYLOOP_R4_PROTOCOL_2026-11-24.md), committed before any full R4 science experiment.

## Why this matters

R3 had demonstrated that the organism's synthetic recurrent circuit can learn the next wall-sensor readings from physical motion, but minimizing sensory surprise does not reliably improve locomotion. R4 asks a harder question: is a prediction actually sensitive to the **action the organism executed**, or can the wall sensor be predicted equally well from its existing neural state and sensory history? And, if the predictor learns motor influence, does a motor policy that tries to increase this influence explore more than honest controls?

**Result:** The true current motor command improves local one-step predictions by **a small but consistent amount** on the matched sensory trajectory. But its improvement over **a command from several steps earlier** is just ~**0.96%** in prediction MSE, below the preregistered 15% gate. R4 motor learning from this engineered action-conditioning gain does NOT increase exploration or reduce physical wall strikes enough to pass its gates.

This result is not a biological fly imitation, discovered goal, acquired escape reflex or demonstrated ability to search for food. The preference for action-dependent predictive gain is a **researcher-designed intrinsic reward**.

## Locked setup, no retuning

- Exactly **336 complete continuous-body lives**: 24 entirely NEW seeds **9950–9973**, ×2 unchanged exogenous ten-rectangle world generators (\`obstacles\` and \`nearfield\`), ×7 paired controllers. All complete the maximum 600 frames = **201,600 physical steps**. Time step 0.2, original R1 22×22 physical body/obstacles.
- Original recurrent nervous system: **16 persistent tanh neurons and 434 original neural coefficients**, original 8 local sensory inputs and 2 continuous speed/turn motors, initial parameters/body/motor perturbation streams paired across every controller in same world. Upstream neural coefficients never updated. Only original 34 motor-output coefficients can adapt in plastic arms.
- Three independently learned **84-coefficient** one-step sensory heads with exactly the same actual next three wall-ray target, training AFTER physical movement:
  - **True:** receives the actually executed two-channel motor command, six local-ray×command interaction features, current hidden state and current rays.
  - **Blind:** motor and its interactions removed, keeping the same current hidden state and rays; the original 84 coefficient slots remain, although only a subset are active.
  - **Yoked:** substituted motor command from this same individual's past, **at least 6 executed steps old**, using an independent PRNG. No future/current motor command information.
- In \`passive_probes\` ALL THREE heads train on the **exact same physical trajectory** from an unchanging neural motor policy, never on independently chosen easier worlds. The true head is also reevaluated *before its update* with a historically substituted motor command, using the unchanged true-head coefficients and same actual future sensor target. No hypothetical counterfactual physical outcome is invented.
- The \`agency_gain\` motor learner uses \`clip((MSE_blind−MSE_true)/(MSE_blind+0.01),−1,+1)\` after the real action, with R2's exact researcher-specified eligibility update and motor noise. The \`sham_gain\` arm uses YOKED instead of TRUE, and \`prediction_error\` uses negative actual sensory MSE as in R3. Other controls: original noisy \`recurrent_frozen\`, R2's physical-progress/contact-reward motor learner, and inertial-correlated random movement. No food, global target, planning, prewritten forage/flee or evolutionary selection.

## Primary nearfield results (24 paired new world seeds)

### Late sensor prediction: passive heads, identical real trajectories, frames 300–599

| Sensor head/evaluation | Real next-sensor MSE (lower better) |
|---|---:|
| **TRUE current executed motor command** | **0.018284652** |
| Motor **BLIND** (zero command/interactions) | 0.019526824 |
| YOKED actual command sampled from 6+ frames earlier | 0.018461220 |
| Same trained TRUE head, evaluated with older YOKED command | 0.018552053 |

Correct motor action makes prediction **6.36% better than action-blind**, **0.96% better than an older yoked motor command**, and **1.44% better than the SAME true head given a mismatched historic command**. All are below the prospectively fixed **15%** relative-error improvement needed for gates 12–14.

The independent external audit also calculated exploratory paired 95% bootstrap intervals using **24 world seeds, not the 7,200 consecutive late frames**:
- Blind minus true MSE: **+0.0012422** per seed, bootstrap95 **[+0.0007375,+0.0018047]**, positive in **23/24 seeds**.
- Yoked minus true MSE: **+0.0001766**, bootstrap95 **[+0.0001008,+0.0002644]**, positive in **21/24**.
- Same trained true head, older command minus actual command: **+0.0002674**, bootstrap95 **[+0.0001228,+0.0004780]**, positive in **22/24**.

These observational intervals indicate the small positive association is reasonably consistent **within this locked synthetic set**, but there is no robust large efficacy improvement or demonstrated independent real-world command intervention. The test is not a substitute for counterfactual physical replay where only command differs while starting from exactly the same state.

Diagnostics: only ~**5.85%** of late passive frames have all three wall readings *exactly unchanged*; the mean absolute change averaged over the three ray channels is **0.07557** per frame. Actual motor vs yoked command has mean absolute per-channel difference **0.17387**. Thus the weak action-specific effect is not simply caused by every sensor being frozen, or every mismatched action being identical. The recurrent hidden state and current rays may already encode much of the predictable near-term variation, but this potential explanation has NOT been directly isolated experimentally.

### Motor performance in the same nearfield worlds

| Motor policy | Mean 1×1 cells explored | Pooled contacts / real meter | RMS route radius |
|---|---:|---:|---:|
| Original frozen recurrent | **23.125** | 0.3900 | 2.632 |
| **True motor-conditioned agency gain** | **23.875** | **0.3749** | 2.712 |
| Old-command SHAM agency gain | **24.000** | **0.3347** | 2.693 |
| Negative sensory prediction error (R3 method) | 23.208 | 0.3924 | 2.591 |
| R2 physical-progress/contact update | 20.083 | 0.3603 | 2.308 |
| Inertial correlated random movement | 38.125 | 4.8201 | 3.934 |

The TRUE agency-gain policy explores only **3.24%** more cells than frozen (23.875 vs23.125), not the **10%** required; and cuts physical contact density by **3.87%**, not the **10%** required. It is not better than the yoked-action motor learner on cell exploration or collisions. The apparent advantage relative to prediction-error learning is too small to pass the +10% criterion.

For independent \`obstacles\` geometry, **agency gain 18.500 cells vs frozen 18.542 cells** and **0.4491 vs0.4532 contacts/meter**: again not a robust advantage. The original R2 progress-reward motor controller explores fewer cells but collides less; it is an informative example of how physical safety can arise from remaining confined.

## Full preregistered gate verdict

**11/11 integrity gates PASS:** full balanced population and complete deterministic replay; original R1 physics; paired world/body/seed/noise; exact 16 neuron/8 local sensory substrate; only 84×3 prediction coefficients plus possible 34 motor readout weights; true and sham labels only real post-action three wall readings; true/zero/older motor inputs; sham older than 6 frames; passive heads have exactly the same actual physical trajectory, before-update command-swapped evaluation; 24 independent valid obstacle maps for each generator; full positive/negative raw logs and false safety/claim flags.

**0/7 efficacy gates PASS:**
12. TRUE head late MSE ≤85% motor BLIND: **FAIL** (0.018285 vs0.019527).
13. TRUE head late MSE ≤85% old-command YOKED: **FAIL** (0.018285 vs0.018461).
14. Same TRUE head using executed command late MSE ≤85% using older command: **FAIL** (0.018285 vs0.018552).
15. Agency learner explored cells ≥110% frozen: **FAIL**.
16. Agency learner contacts/meter ≤90% frozen: **FAIL**.
17. Agency learner explored cells ≥110% SHAM learner: **FAIL**.
18. Agency learner explored cells ≥110% negative-prediction-error learner: **FAIL**.

Overall R4 development-science screen **FAIL: 11/18**. CI success does NOT upgrade any of the seven predeclared scientific failures.

## Complete and independently verified raw evidence

- Completed green [GitHub Actions run #45440238377](https://github.com/kkongiu/neuroevolution-lab/actions/runs/45440238377), original run and second entire replay byte-identical.
- Raw workflow artifact \`flyloop-r4-action-controllability-results\`, artifact ID **14942559176**, retention until **24 December 2026**.
- Two complete full JSON files **267,997,344 bytes EACH**, byte-identical SHA256:
  \`124c56c9034628441a0f50ce6d5f8f6007c48fbb6a3e660f68b4660d02395583\`.
- Independently recomputed canonical raw-row SHA256:
  \`9075f5d850d1e3364c874a615674f428ce3719dcfde5b96dd5aa2ffc4aa1df22\`, matches source metadata.
- Downloaded GitHub ZIP independently audited with separate Python standard-library script that imports NO R1/R2/R3/R4 modules. Independently checked **336 complete lives**, **48 distinct seed×world paired cohorts**, **201,600 physical frames**, **18,645 real wall contacts**, **345,600 chronological head-update events**, and **199,584 sourced historic motor-command samples**, **ZERO discrepancies**.
- Audit directly replays the 3×84 head coefficient LMS updates from logged current neural/sensor/motor inputs and actual next observations, reconstructs each pre-action prediction and same-head older-command evaluation, validates real post-contact energy/integrity, body clearance and continuity, original/final neural frozen-coefficient constraints, and reproduces summary metrics. The independent audit JSON and script are available with the downloadable original archive.

## Interpretation and next falsifiable issue

R3 showed a significant amount of **predictable sensor change**; R4 shows that explicitly supplying the *executed action* adds only **modest incremental predictive information** beyond hidden neural state/current local sensation. The system has **not** learned to take more effective actions from these predictions, and definitely has not independently invented intrinsic goals.

Before any major additional architecture or fitness optimization, the next clean question would be an **actual controlled action-intervention experiment**: replay the same body/neural/world state with different physically executed motor commands, separately measure effects on later perception and travel, and test whether the learned model predicts those differences out-of-sample. This would avoid calling a shuffled-action model a proven counterfactual effect. Do not rewrite R4's locked results to make it pass. R4 remains an intentional scientifically negative result.

**Stop flags remain FALSE:** \`food_search_demonstrated\`, \`escape_reflex_learned\`, \`autonomous_goal_invention_demonstrated\`, \`general_intelligence_demonstrated\`, \`biological_fly_replication_demonstrated\`, \`evolution_authorized\`, \`advance_to_evolution\`.

PR #94 remains DRAFT, stacked on R3 #93. Main, sealed M5.6, M0–M4 and R1–R3 left unchanged.
