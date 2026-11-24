# FlyLoop M2 — prospectively frozen contact-only sensory ablation and independent procedural worlds
**24 November 2026 — protocol before first complete M2 science run.**

## Question and interpretation
M0 and M1 establish that an online 14-real-weight actor–critic collects more food in small synthetic environments than an initialized but frozen actor, including when directional smell is suppressed. M1's ostensibly scent-free model still knows **before moving** whether the destination contains food. M2 must remove that privilege without changing the learner and without hiding potentially negative findings.

**Primary question:** does online plasticity show practical improvement over both frozen and an explicit **uniform-moving random baseline** when it cannot directly sense either nearby food or nearby hazards? Does behavior change in a separate PROCEDURALLY generated wall layout? A positive result is **not** invention of sensory modalities, abstract goals, new movement actions or a positional memory.

### Immutable experimental configuration
- All animals are the exact M0 **8-parameter shared linear actor + 6-parameter linear TD critic**, identical softmax temperature, physical energy/integrity reward and actor/critic learning rates. No new neurons, loops, symbolic hypotheses, maps, planner or reproduction.
- 13×13 grid, 16 independent renewing food sites (8 on each side), 9 damaging hazards, bounded 0–100 energy and integrity, 70 energy initially, food regrowth after 17 steps, five N/E/S/W/WAIT primitive actions. Every lifetime ends at death or timestep 480; barrier openings change at step 240. Agent never receives coordinates or event schedule.
- **24 new world seeds 7600–7623**, disjoint from M0 (7100–7131) and M1 (7300–7331). All worlds **paired** on map, hazard/food, initialized actor/critic and physical starting state across 4 sensing modes × 4 controller arms per geometry.
- Two **geometry families**:
  1. \`baseline\`: M1's barrier at x=6, openings (2,10) switching to (5,8) at t=240. Food/hazards from original seeded M0 procedure, but on NEW M2 seeds.
  2. \`procedural\`: a separate reproducible generator draws two first gate openings and two distinct later ones, and places up to ten interior wall posts chosen only when both gate phases preserve whole-map reachability. Food/hazards seeded from separate random stream, in unblocked cells (8 food left, 8 right). Its generator does NOT condition on actions, rewards or locations reached by any organism. Both phases preserve paths between all traversable cells; walls may nonetheless invalidate an occupied closing-gate cell and kill the agent. For a given seed, all sensory and actor arms see an identical initial world and gate-switch plan.

### Four fixed **sensory masks**, by increasing deprivation
All masks act on **both** the action feature vector and 6-dimensional critic feature vector, plus the novelty signature derived from the masked sensory state:
1. \`full\`: M1's engineered SHORT-RANGE PASSABLE-PATH food cue (≤4 movement steps) with deterministic bounded ±0.08 pseudo-noise, food in adjacent candidate cell, visible adjacent wall and hazard, hunger/injury, WAIT; no X/Y.
2. \`no_scent\`: identically zero directional food cue and hunger×cue entries; still sees the explicitly coded destination food and hazard bits. Food smell MUST be exactly zero.
3. \`contact_food\`: also masks the destination-food bit. Only *AFTER* stepping into a food cell does the body register changed energy and the plastic policy receive true physical TD reward. Hazard and wall lookahead still exist.
4. \`blind\`: masks both the food and hazard lookahead, and directional food cue. Retains wall lookahead and the local internal physiological state; physical damage and food are observed via their actual effects on energy/integrity after contact. With these inputs, the 14-weight policy cannot store specific map positions and can adapt only generic action preferences such as WAIT vs move and legal-move selection.

The researcher-specified binary wall channel, energy, damage and policy-sharing prior are **still privileged structure**. No claim that blind means 'no information whatsoever': the body still receives proprioceptive physiological consequences.

### Four FIXED controllers, no learning across episodes
A. \`frozen\`: seeded M0 stochastic softmax 14 weights, never updates.
B. \`random_motion\`: uniform random N/E/S/W, never WAIT, no collision correction or route finder, no weights update. Exposes possible gain from moving more often and not waiting; it is deliberately NOT a learned controller.
C. \`plastic\`: M0 exact online TD(0) actor–critic updated from real energy/health changes, no engineer-designed bonus.
D. \`plastic_novelty\`: same plus M0's explicit **hand-coded** sensory-pattern-frequency novelty reward; NOT learned curiosity.

All initial 14 weights are seeded identically per world, even for the uniform-motion control (though unused). Random policy draw streams are independently deterministic; all four controllers have different trajectories due solely to their chosen actions and subsequent physical experience. The environment's food schedule and map do NOT depend on the controller beyond its legitimate effects of consuming food and setting that site's regrowth cooldown.

Total = **2 geometries × 4 sensory masks × 4 controllers × 24 seeds = 768 persistent lifetimes**, each ≤480 real transitions. Evaluation samples and hyperparameters frozen before running all 768. Full rerun from same code must produce byte-identical scientific JSON.

### Outcome measures and QA
Per life store full action-by-action trace with physical energy, health, reward, food, collision, damage, and the complete pre-action masked action/critic feature matrix; initial map and barrier shift; original and final weights; all action types and number of parameter updates. Compute: survival steps, food consumed, food per **100 actually lived ticks** (ratio of pooled total food to pooled steps for each arm, avoiding spurious opportunity inflation), hazard contacts, wall hits, unique food sites, revisits, unique cells, and post-switch food and completed crossings. Dead agents contribute zero food after death and not invented survival time.

The PRIMARY paired statistics use world seed as the independent unit. Exploratory 1,000-draw paired-world-seed bootstrap intervals may contextualize effect magnitudes but do not replace threshold gates. The "procedural" family is independent relative to previous fixed M0/M1 geometry, **not transfer of a trained policy**: weights are initialized anew for each world. No module-selection from those seeds and no revisiting thresholds after results.

### 17 prospective DEV science/integrity gates, ALL required
1. **768 exact complete scenario×mask×arm×seed rows**, with complete deterministic second science run byte-for-byte identical (the duplicate check runs in Actions).
2. All agents use five legal primitives, no wall penetration or teleportation, energy and health remain within bounds; no step after death.
3. Every (geometry,seed) has exactly the same initial map, external gate-switch schedule and starting 14 weights across all 16 sensor/controller combinations.
4. \`no_scent\`, \`contact_food\` and \`blind\` have EXACT zero directional cue entries in every action and critic state (but \`full\` does not).
5. \`contact_food\` and \`blind\` have EXACT zero adjacent-food input in every candidate action and critic state; any food effect occurs only AFTER environmental action.
6. \`blind\` has EXACT zero adjacent-hazard lookahead in all action features. True hazard health/energy effects remain intact.
7. \`procedural\` maps differ from original fixed layout and are independently generated, have at least 20 distinct initial map signatures across 24 seeds, have valid food/hazard positions and 2 disjoint early/late gate openings; accessible cells remain connected in both phases.
8. Researcher-independent gate switching really occurs only at t=240, and no future event or absolute position appears in neural input.
9. Frozen and uniform-moving controllers keep ALL weights unchanged; plain plastic and novelty agents perform nonzero online weight updates; \`random_motion\` NEVER waits (and NEVER bypasses walls).
10. Exactly 768 full real trajectories, weights, params and metadata archived; old sealed M5.6 and original M0/M1 unaffected; evolution and intelligence proof flags FALSE.
11. In \`baseline/full\`, plain plastic **food per 100 alive ticks** ≥ frozen ×1.15.
12. In \`baseline/no_scent\`, plain plastic food per 100 alive ticks ≥ frozen ×1.15.
13. In \`baseline/contact_food\`, plain plastic food per 100 alive ticks ≥ frozen ×1.15.
14. In \`baseline/blind\`, plain plastic food per 100 alive ticks ≥ frozen ×1.15.
15. In \`baseline/blind\`, plain plastic food per 100 alive ticks ≥ the **random_motion** baseline ×1.10 (stronger evidence than simply learning to move instead of WAIT).
16. In \`procedural/blind\`, plain plastic food per 100 alive ticks ≥ random_motion ×1.10.
17. In \`procedural/blind\`, plain plastic food per 100 alive ticks ≥ **60% of its own procedural/full** rate, quantifying representational cost.

**If a science gate fails, label overall DEV screen FAIL.** CI software/test correctness is NOT a scientific outcome; preserve negative results. Do not relax gates, select favorable worlds, or claim generality from synthetic ecology. No autonomous invention of new actions has been tested.

Full results, both runs, checksums, tests and per-world traces must appear on this PR before any further scientific decision.

STOP FLAGS always false:
\`general_intelligence_demonstrated=false\`,
\`autonomous_goal_invention_demonstrated=false\`,
\`spatial_memory_verified=false\`,
\`independent_biological_replication_complete=false\`,
\`evolution_authorized=false\`,
\`advance_to_evolution=false\`.
