# FlyLoop R8 — Online learning of REAL physical injury and failed movement does not yet prevent repeated collisions

**24 November 2026 · Complete, prospectively fixed R8 scientific screen: FAIL (14/19 gates PASS).** All **10/10 integrity** and **4/9 scientific efficacy** gates passed. Software regression, complete twice-run byte-identical replay and independently reconstructed raw-body/physical-predictor audit: ALL PASS.

[Original fixed prospective R8 protocol](FLYLOOP_R8_PROTOCOL_2026-11-24.md) was committed **BEFORE the R8 implementation and results**. No seeds, objective, architectures, numerical coefficients, trial counts, model training algorithm or preregistered gate thresholds were changed after inspecting the outcome. Main, original R1–R7, M0–M4 and sealed M5.6 left untouched.

## Experiment: did the fly learn anything from its own collisions?

R7 already showed that a self-running recurrent circuit can **use its learned local sensory predictions to select actual physical motor commands**, but its contact recurrence remained very high (~86.8% in a previous R7 comparison). The question for R8 is narrower: if the same body learns the **real consequences of the last executed movement**—wall contact, actual displacement, and integrity injury—can it turn away from harmful repeats WITHOUT a scripted escape procedure?

**YES to physical-outcome prediction, NO to the preregistered useful behavior improvement.** The real physical predictor fitted during a life learns much better than the same organism's temporally mismatched older-real-outcome sham on identical frozen trajectories; however the fixed researcher-supplied decision score fails to significantly reduce repeated collisions and generalizes poorly to an alternate obstacle family.

**This is NOT spontaneous discovery of survival, fear, internal motivation or intelligence.** The preference for predictable movement and low physical damage, the five local motor options, their relative weights and the training rule were explicitly invented by the researcher. Further, R1's standardized collision always causes precisely 0.25 integrity loss, so the "contact" and "injury" targets are REDUNDANT—not two independent sensory signals.

### Immutable original design
- **16 fresh seeds 11200–11215** × original two different ten-obstacle world generators (\`nearfield\` and \`obstacles\`) × static vs five-wall removal at t=160 × six fixed controllers = **384 REAL continuous organism lives**, each 320 actual motor integration frames = **122,880 physical steps**. No deaths, rejected seeds or cherry-picked lives; environments paired with same original map, body and original 434 initialized neural coefficients; identical R2 σ=.12 motor noise stream on shared alive prefixes.
- Unchanged original R1 16 recurrent tanh neurons, 8 local sensory inputs (three real wall rays, physical velocity/turn, previous contact and body energy), original 434 coefficients, analog two-motor physical speed/inertia and actual collision injury (.25) and energy (.02 per frame). All original input/recurrent/hidden weights stay fixed; only the original R2 physical-progress positive-control arm adapts its existing 34 motor readout coefficients.
- Original R7 three **84-coefficient online next-wall-sensor heads** remain; at every step, R8 adds **two 84-coefficient 3-output physical consequence heads** initialized all ZERO, trained only AFTER actual movement using the same current 28 features (16 hidden units, three local wall rays, executed 2-motor command, six ray×command interactions and bias).
- TRUE head learns the current real 0/1 collision, real movement divided by max possible body travel, and real bodily integrity decrement divided by .25. Older-sham head uses **the SAME organism's own genuinely physically executed outcome at least SIX frames in the past**; independent seeded RNG and no imaginary labels. Both are scored against ACTUAL current physical effects.
- At t≥50, R8 physically executes the one candidate of five small R7 analog motor perturbations with the highest **researcher-coded pre-action score**:
  \`0.25*sqrt(predicted sensory-ray-change MSE) + 0.60*predicted normalized movement − 0.50*predicted collision − 0.50*predicted injury\`.
  No wall-map access, simulated physical future of nonexecuted candidates, path planner or \`if collision: steer_left\` macro. The selected motor is integrated ONCE; the observed real bodily consequences are used for subsequent learning. The five choices are a fixed human-designed mechanism, not emergent policy architecture.
- Six paired controllers: \`frozen\` (both heads learn but motor frozen, for exact same-trajectory prediction comparison), \`r7_novelty\` (original sensory-change-only motor selection), \`r8_true_physics\`, \`r8_older_sham\` (the older-outcome head guides the identical scorer), \`random_menu\` (same candidate choices, independent random selection), \`r2_progress\` (original hand-coded R2 physical net-displacement minus contact reward). No task script or autonomous goal.

## Main preregistered behavioral result — nearfield / walls removed at tick 160

All metrics are measured ONLY over physical frames t=160–319, across 16 independent paired world seeds. No frames treated as independent organisms.

| Arm | Distinct 1×1 cells | Actual collisions per meter | Collision followed by another collision | RMS spatial route radius | Meters per life |
|---|---:|---:|---:|---:|---:|
| Original noisy frozen recurrent | 8.875 | 0.9382 | 91.89% | 1.056 | 17.453 |
| Original **R7 sensory-change-only** | **9.6875** | 0.9592 | **86.84%** | **1.264** | 17.593 |
| **R8 true learned bodily consequence selector** | **8.8125** | **0.9083** | **86.57%** | **1.037** | **18.441** |
| R8 older-real-outcome sham selector | 8.375 | 1.7030 | 91.95% | 1.097 | 16.515 |
| Identical motor menu, random choice | 9.8125 | 1.2187 | 91.05% | 1.229 | 16.668 |
| Original R2 physical-progress/contact readout learner | 7.4375 | **0.6601** | 95.45% | 0.904 | 16.759 |

The correctly trained R8 bodily model lowers collision density **only 5.31%** vs R7 (required **25%**), with almost identical repeated contact probability (**86.57%** vs **86.84%**, instead of required **20%** relative reduction). It keeps **90.97%** of R7's explored 1×1 grid cells, meeting the original minimum 90% retention gate, and walks **4.82% more meters**, but its RMS spatial spread is only **82.06%** of R7's, falling below the 85% gate. It falls narrowly below the random control's 90% cell-coverage threshold (8.8125 vs required 8.83125), which must still count as a FAIL.

The model actually changes its motor choice on **97.97%** of postchange frames. This verifies it participates in a causal motor-control feedback loop, but **a high intervention rate does not mean the decisions are useful**.

Among all R8 true postchange episodes, repeated collision count is **232** with **268** actually observed collision-followup opportunities; R7 records **231/266**. The very small percentage improvement masks **one more actual repeated collision** in this cohort. Both quantities, counts and denominators, are disclosed, and no success claim is made.

### Learned physical outcome prediction works, at least in matched life

On the **exact same frozen recurrent physical trajectories** in nearfield+removal, late real-target 3-output MSE:
- TRUE online physical head: **0.013211868**.
- SAME organism's real-experience older-outcome sham: **0.057142685**.
- Error reduction ~**76.9%**, above preregistered required ≥15%, so physical prediction learning gate **PASS**.

This is only a local single-step synthetic association about bodily consequences, not a demonstration that selected motor actions generalize to effective navigation.

### CRITICAL generalization failure — other R1 procedural obstacle family

On \`obstacles\` + \`removal\`, R8 TRUE controller performs **MUCH WORSE** than the original R7 sensory-only controller:

| Metric | R7 sensory-change | R8 true physical consequences |
|---|---:|---:|
| Unique grid cells | **10.00** | 6.00 |
| Contacts per actual meter | **1.3779** | 2.2726 |
| Actual repeated collision fraction | **92.82%** | 95.97% |
| RMS spatial trajectory radius | **1.282** | 0.896 |
| Actual mean travel distance | **17.19** | 15.81 |

R8 is **40% worse in cell coverage**, **65% higher collisions/meter**, and more spatially confined. This severely limits any claim that physical-consequence learning by itself solves locomotion. Static-world and all other arms are also retained in the original raw dataset.

## All 19 precommitted gates

**Integrity 10/10 PASS:**
1. Exactly 384 balanced lives and exact full independent duplicate JSON at byte level.
2. Legal R1 physical body, actuator inertia, injury/energy and no resurrection/penetration.
3. Exact paired world/pose/434 initial coefficients and matched Gaussian motor noise across arms.
4. Original 16 recurrent units and 8 local sensors, input/recurrent/hidden coefficients frozen; no map/food/future input.
5. Two original 84-weight zero-initialized bodily models + original 3×84 online local wall-sensor models, trained only after action; fixed offsets and coefficient bounds.
6. True bodily targets ONLY from real just-executed motor and actual post-action physical effects.
7. Older-sham label from SAME animal's real history ≥6 frames old, independent RNG, no fabricated observations.
8. Exact five candidates and pre-action fixed score, no physical lookahead.
9. Exact static/removal pre-t160 match and only five exogenous walls removed at t160.
10. Full raw trajectories, honest outcomes, unchanged prior studies, no food/goal/evolution claims.

**Science 4/9 PASS:**
11. True physical model MSE ≤85% old-outcome model MSE on SAME frozen trajectory: **PASS**.
12. TRUE repeated-contact fraction ≤80% R7: **FAIL** (86.57% vs 86.84%).
13. TRUE collision density per real meter ≤75% R7: **FAIL** (0.9083 vs 0.9592).
14. TRUE distinct visited cells ≥90% R7: **PASS** (8.8125 vs 9.6875).
15. TRUE distinct visited cells ≥90% random menu: **FAIL** by 0.01875 cells.
16. TRUE RMS route radius ≥85% R7: **FAIL** (1.037 vs 1.264).
17. TRUE repeated contacts ≤85% older-outcome sham: **FAIL** (86.57% vs 91.95%).
18. TRUE actually overrides analog base motor ≥10% of postchange actions: **PASS** (97.97%).
19. TRUE actual traveled distance ≥90% R7: **PASS** (18.44 vs 17.59).

**Overall screen FAIL 14/19; no post-hoc threshold changes or reclassification.** Exploratory paired 16-world-seed bootstrap of TRUE minus R7 visited cells: mean **−0.875** cells; 95% exploratory interval **[−4.50, +3.625]**, inconclusive gain, cannot support learned navigation advantage.

## Reproducibility and truly separate raw-data audit

- GREEN [GitHub Actions completed run #45481374195](https://github.com/kkongiu/neuroevolution-lab/actions/runs/45481374195). All unchanged original R1–R7 unit/regression tests and new R8 physical-label/strict-old-sham/matched-world/physics tests PASS.
- Two ENTIRE physically executed R8 study JSONs are **byte-identical**, each **315,201,053 bytes** with SHA-256:
  \`63b64681bc0220ffa0155c348bf8e10a6d24b05407114a15b3de6c20e83cef19\`.
- Archive GitHub Actions artifact \`flyloop-r8-online-physical-consequence-results\`, ID **14962390603**, expires **2026-12-24**.
- Canonical 384-raw-lives SHA-256 **independently recomputed and matching**:
  \`4890628aefc234291e42900089e566fc0750e89a60cba44c3cd3fdb082666901\`.
- Separate standalone Python standard-library [independent audit JSON](FLYLOOP_R8_INDEPENDENT_AUDIT.json), **imports NO R1–R8 neuroevolution modules**. It reconstructed **all 384 lives, 122,880 real physical body frames**, actual collision and all .25 injury events, motor/speed/turn inertia, wall/boundary clearance, physical wall-proximity ray-casting, preaction candidate scores and the selected command, and independently replayed BOTH 84-coefficient physical consequence heads' entire chronological LMS updates. It verified all **120,576** older real sham target source references and summary physical counts/coverage plus both full raw files. **ZERO independent audit anomalies**.
- Scope caveat: the independent script does **not independently recompute the separate original R4 sensory-model coefficient update history**, which remains covered by original R1–R7 regression tests and current internally complete physical replay. This is a software/scientific consistency audit of a SYNTHETIC artificial world, not independent empirical animal data.

## Next scientific interpretation

We have now demonstrated a critical distinction:
- A continuously moving agent CAN **learn what physical effects its actions produced** in its own lifetime.
- A human-specified motor score CAN use those predictions to select different, actually executed, continuous actions.
- **Even with these two working capabilities, the action selector DOES NOT learn a useful general obstacle-avoidance policy**, nor reliably break up repeated wall impacts. Optimizing an approximate one-step bodily cost model does not automatically produce multi-step behavior competence.

A future R9, if authorized, should test whether physical contact persistence depends on *motor inertia and spatial/temporal credit assignment*, not reflexively add a scripted turn: compare longer physically observed temporal consequences and recurrent motor control, action-conditioned frozen/sham controls, and new heldout geometries. No prewritten escape behavior. Keep the R8 negative evidence immutable.

STOP FLAGS all FALSE: \`food_search_demonstrated\`, \`escape_reflex_learned\`, \`autonomous_goal_invention_demonstrated\`, \`general_intelligence_demonstrated\`, \`biological_fly_replication_demonstrated\`, \`evolution_authorized\`, \`advance_to_evolution\`. Code fact \`closed_loop_physical_prediction_guides_action_implemented=true\`; success claim \`effective_collision_learning_demonstrated=false\`.

PR #98 remains DRAFT stacked on R7 #97. All earlier branches/main are unmodified.
