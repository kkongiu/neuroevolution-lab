# FlyLoop M4 — preregistered local chemical sensing and temporal comparison
**24 November 2026. Protocol frozen before the first full M4 science evaluation.**

## Scientific purpose

The M3 experiment failed its efficacy gates: a blind organism with an engineered last-action / reward trace did not learn better-than-random food search in a procedurally generated maze. M4 tests a narrower biologically inspired sensorimotor hypothesis: can a small online learner exploit **ONLY local, non-directional chemical concentration** measured at successive locations to improve foraging, without directly knowing the location of resources?

This is NOT a reconstructed insect sense organ and NOT an autonomous invention of chemotaxis. The neural input **alignment×concentration-change** and spatial field are explicitly engineered; the learned weights must determine whether this trace is useful. Positive controls and time-shuffled controls must prevent us from confusing movement persistence with learned chemical tracking.

## Locked world and signal

- 13×13 physical world, five primitive actions (N,E,S,W,WAIT), 16 regrowing food patches, nine damage sites, persistent 70 initial energy and 100 integrity, maximum 480 steps per life, food regrowth every 17 ticks. Same original M0 physical energy/health reward, 8 real-valued shared action-conditional actor weights +6 real-valued critic weights, same sampling temperature, TD(0) and learning rates, no new trainable parameters. No genetic evolution or world maps supplied to the neural controller.
- Exactly **32 fresh world seeds 8300–8331**, distinct from all previous M0/M1/M2/M3 evaluation seeds. All seven arms share exactly the same initial 14 neural weights and initial world within each seed+scenario. Changes in food availability after eating are legitimate consequences of each agent's actions.
- Scenarios: \`baseline\` (M0 gate pair switches at t=240); \`procedural\` (independently generated M2 maze with 10 posts, two randomly selected disjoint gate pairs, all traversable cells connected before and after t=240); \`relocation\` (fixed M0 wall, no gate change, ALL 16 food sites moved exogenously at tick 240 and original locations restored at 360, exactly M1's corrected position-independent replacement method).
- **No hidden food/hazard preview**: all action features start from M2's \`blind\` raw sensor mask. It exposes locally detectable impassable walls, current energy and injury, but NO food at neighboring cell, global food position, directional precomputed path scent, or neighboring hazard.
- An independently implemented **local scalar odor field** exists in the environment. Each currently available food source emits a unit source. The field at a cell is the sum of \`0.65 ** d\` over sources whose **shortest wall-respecting walk path length d <=7**, divided by 4 and clipped to [0,1]. This is a simplified stationary graph-diffusion proxy, **not a physics-accurate airborne diffusion model**. It never travels through an impassable wall without a path; its magnitude changes when food is eaten/regrows or relocated and when the gates switch. Add deterministic spatial/temporal pseudonoise with absolute amplitude <=0.015, clipped to [0,1]. The organism can read **ONLY the one scalar at its current position at the current tick**, never samples neighboring cells or the future map. The sensor engine knows food locations; the neural controller never receives them.
- At t=0 no temporal comparison exists. At tick t>=1, the sensory controller can compare its own scalar reading at tick t with its scalar reading at tick t−1 (actual observed samples only). The comparison is a signed difference, clipped after multiplication by 6 to [-1,1]. It may also use exponentially weighted average with coefficient 0.75. The latest executed movement defines alignment of each candidate move with the last attempted movement (dot product of N,E,S,W,WAIT vectors), and last wall collision is observable. WAIT has alignment 0. Input features use ONLY causal already-observed values.

## Seven frozen controller arms

All have exact M0 14-weight actor+critic and same action repertoire:
1. \`frozen\`: M2 original blind stochastic frozen controller; **no smell input**, no weight updates.
2. \`random_motion\`: uniform N,E,S,W without WAIT or collision correction; no smell, no weights update.
3. \`plastic_plain\`: M2 original blind actor–critic online TD; **no smell input**.
4. \`motor_only\`: same TD learning and engineered prior-movement/collision alignment as M3, **no smell input**. Exact M3 motor-only trajectory expected on M3 historical seeds in the baseline.
5. \`chemo_true\`: same TD learning, same motor history, PLUS its actual local odor scalar and the *correct* difference between the two most recent odor samples. Previously zero blind feature slots are reused: actor input #3 movement alignment; #4 alignment×previous collision; #1 alignment×current odor difference; #2 alignment×odor-difference EMA; critic #3 current odor concentration, #4 current odor difference. **No direction to food is directly programmed**; hand-supplied movement alignment is an inductive bias.
6. \`chemo_sham\`: same as \`chemo_true\`, including current actual odor scalar to the critic, but actor's signed odor difference and critic's difference are taken from a **different, strictly older** self-observed pair of samples (source tick <= current tick−2), selected by independent pseudo-RNG, never future or the just-observed delta. This preserves a plausible scent-history marginal while breaking current action-consequence timing. At t<3, sham odor delta is set to zero.
7. \`chemo_frozen\`: correct real local scalar+delta and movement history as \`chemo_true\`, with original weights FROZEN; isolates mere structure/sensing from learned use.

No adaptive tuning of the dilution/scaling/noise, hand-coded 'follow gradient' plan, wall-following macro, reinforced novelty bonus, additional neurons or trained external oracle. Environment produces a field; actor inputs do NOT contain any food site coordinate, path, velocity vector to source or destination odor sample.

Total **3 scenarios × 7 arms ×32 seeds = 672 continuous lives** (up to 322560 steps). Execute FULL experiment TWICE and assert byte-for-byte equal outputs. All 672 trajectories, initial/final neural weights, raw blind inputs, single-cell scent, true and sham temporal samples, memory-source timestamps and physical movement outcomes must be archived.

## Metrics and inference

Predeclared primary endpoint per scenario/arm: **pooled food collected per 100 actually lived ticks** (total food / total active ticks), avoiding reward just for living longer. Secondary: mean food/life, lifespan, 480-step survival, food after gate/relocation, food in [240,359] relocation phase (dead lives counted as ZERO), distinct food sites, wall collisions, health damage and crossings. Paired bootstrap 95% intervals resample the complete **world seed**, not individual steps.

All comparisons to \`chemo_sham\`, \`motor_only\` and \`random_motion\` use same scenarios, seeds, physiology and initial parameters. If true odor feedback beats these controls only in the baseline geometry but not procedural, that is not robust generalization. No cross-world neural learning is done: geometry shift here means transfer of a **learning mechanism**, not trained weight transfer.

## 18 prospective integrity and scientific gates: ALL required

Integrity:
1. Exactly 672 complete, balanced scenario×seed×arm raw trajectories and a byte-identical complete second run.
2. Actual primitive movement, no wall tunneling/resurrection, energy/integrity and physical reward correctly audited.
3. Paired initial maps, resource/hazard positions, environmental switch schedule and all 14 original neural weights identical across every seven-arm scenario×seed group.
4. All *raw blind* action/critic features have zero pre-contact food smell/destination-food/hazard-lookahead bits, and still show permitted wall and physiology.
5. The field is a **single scalar from agent's current position**, bounded [0,1], and no physical odor passes impassable walls without a legal <=7-step path; noise <=0.015. No neighbor field vector, future field, map, food list, route or global coordinates in input.
6. All true scent differences come from consecutive **actually observed local odor samples** (with t-1), and t=0 delta is zero; recorded actor features are reconstructable from these actual samples and past action. No lookahead.
7. All sham scent differences/EMA use actual older samples from the same life with recorded source tick <=current−2 (never newest real sample); independent shuffle RNG does not advance actor RNG. No phantom older history at t<3.
8. Both M2 blind plain+frozen+random and M3 blind motor-only baselines reproduce their original code on historical seeds in baseline; no silent optimizer or physics change. Same 14 trainable weight positions with frozen and random weights unchanged and TD weights updated.
9. Procedural maps are 10-post, reachable, >25 distinct out of 32 worlds; food/hazards valid. Relocation is exogenous and same replacement food set across seven arms on a world seed, with exactly t=240 relocation and t=360 restoration (if organism alive), no simultaneous gate shift.
10. All 672 complete raw trajectories, exact seeds and weights, real odor history, full negative results and all STOP flags archived; old M0/M1/M2/M3 + sealed M5.6 unchanged.

Science:
11. In **baseline**, true chemo food per 100 live ticks >= \`plastic_plain\` ×1.15.
12. In **baseline**, true chemo food per 100 live ticks >= \`random_motion\` ×1.10.
13. In **baseline**, true chemo food per 100 live ticks >= \`motor_only\` ×1.10.
14. In **procedural**, true chemo food per 100 live ticks >= \`random_motion\` ×1.10.
15. In **procedural**, true chemo food per 100 live ticks >= \`motor_only\` ×1.10.
16. In **procedural**, true chemo food per 100 live ticks >= \`chemo_sham\` ×1.10.
17. In **relocation phase ticks 240–359**, true chemo cohort mean food (zero for dead) >= \`motor_only\` ×1.15.
18. In **procedural**, true chemo food per 100 live ticks >= \`chemo_frozen\` ×1.15 (evidence that learned weights contribute beyond the handcrafted representation).

**All 18 must pass** to claim narrow M4 DEV-science success. Software CI success is distinct from scientific efficacy. Report every failing gate and any hazardous costs. Do not alter fixed thresholds or selected seed sets after seeing the results.

**STOP FLAGS always false:** \`general_intelligence_demonstrated=false\`, \`spatial_memory_verified=false\`, \`autonomous_operator_invention_verified=false\`, \`evolution_authorized=false\`, \`advance_to_evolution=false\`. No claim of a biological fly, novel sensory invention, autonomous abstract reasoning or actually learned chemotactic strategy beyond supported controls.
