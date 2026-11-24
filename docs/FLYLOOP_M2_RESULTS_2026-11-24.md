# FlyLoop M2 — Full preregistered result and independently audited sensory-ablation limits

**24 November 2026 | [Successful GitHub Actions science run #45351267233](https://github.com/kkongiu/neuroevolution-lab/actions/runs/45351267233) | scientific development screen: 12/17 PASS, 5/17 FAIL, OVERALL FAIL.**

## Headline and interpretation

M0/M1's extremely effective foraging relied **strongly on researcher-engineered early food information**, not merely on an emergent general ability to find sustenance. In M2, making food entirely invisible *until physical contact* largely destroys the plastic controller's efficiency advantage over a frozen or uniformly moving organism. If danger is also invisible until contact, it also loses its substantial survival advantage. The simple uniform-motion organism then eats more *per unit time alive* than the plastic controller on both tested map families, although those pairwise rate-difference confidence intervals include zero.

**This is scientifically useful negative evidence: we have learned where the current minimal architecture stops working. No new mechanism was inserted to turn FAIL into PASS.**

Do not call the sensor-masked policy biologically "blind" in an absolute sense: it still senses walls, physiology and physical reward, and is still a 14-weight artificial actor–critic. Nor does "full" smell simulate real airflow: M1 calculates a researcher-defined reachable nearby-food cue.

The [frozen preregistered M2 protocol](FLYLOOP_M2_PROTOCOL_2026-11-24.md) was committed at SHA \`6dfac09794c9ef6a68d790180106b39b9432a8dd\` **before** any full scientific execution. The 17 quantitative/pass-fail gates have not been modified after observing outcomes.

## Design and experimental invariants

- **768 separate lives:** 2 geometry families ×4 identical-architecture sensory conditions ×4 control policies ×24 new world seeds 7600–7623; each lifetime persisted for up to **480** sensor–action–reward ticks.
- Exactly **14 real-valued network parameters** from M0 (8 shared candidate-action actor weights +6 critic weights), SAME TD(0) plasticity, physical reward and softmax temperature, no new neurons or global map. Five elementary actions N/E/S/W/WAIT.
- **Two world generators:** M0 baseline barriers and an independent 24-map connected *procedural* geometry with ten additional immutable impassable posts, two disjoint early/late gap pairs, 16 regrowing food sites and 9 damaging sites. For each world seed, identical world contents, future gate schedule, starting position and initial weights across all **16** sensory/controller combinations. The procedurally generated worlds are fresh testing environments, **NOT cross-world retention of a policy**, as all actor/critic weights start afresh.
- **Four actual sensory conditions:**
  - \`full\`: engineered short-range accessible-path food cue, immediately adjacent food bit and immediately adjacent hazard bit;
  - \`no_scent\`: no directional food cue but adjacent food and hazard visible;
  - \`contact_food\`: no scent and no adjacent food bit; food still changes actual energy when stepped onto; adjacent hazard remains visible;
  - \`blind\`: no pre-contact scent, food or hazard lookahead; wall sensing, hunger/injury in internal critic, and physical consequences after action remain.
- **Four controls:** \`frozen\` (stochastic nonlearning), \`random_motion\` (uniform N/E/S/W, no WAIT or path correction, no learning), \`plastic\` (online TD from physical energy+integrity), and \`plastic_novelty\` (same TD plus *researcher-programmed* sensory-pattern novelty bonus).

The natural hypothesis "an emerging ability to seek food" would have predicted some preserved competence with food invisible; the stronger prespecified test also required exceeding the physically moving random baseline, rather than simply learning to press movement instead of WAIT.

## Results: food per 100 ACTUALLY lived time steps

This *rate* normalizes feeding opportunities by actual survival time; comparisons of total lifetime food alone can be misleading because plastic creatures sometimes live much longer.

| Geometry / sensory condition | Frozen | Random moving | Online plastic | Plastic + explicitly coded novelty |
|---|---:|---:|---:|---:|
| Baseline / all food and danger inputs | 5.483 | 6.156 | **13.359** | 13.803 |
| Baseline / zero directional smell | 5.510 | 6.156 | **10.104** | 9.566 |
| Baseline / food visible only at contact | 5.719 | **6.156** | 5.955 | 5.773 |
| Baseline / both food and danger contact-only | 5.663 | **6.156** | 5.777 | 5.796 |
| Procedural / all cues available | 4.921 | 6.229 | **12.821** | 12.917 |
| Procedural / no directional smell | 5.537 | 6.229 | **9.106** | 9.705 |
| Procedural / food contact only | 5.376 | **6.229** | 6.198 | 5.674 |
| Procedural / food and danger contact only | 5.765 | **6.229** | 5.748 | 5.507 |

### Causal interpretation: food cue versus hazard cue

**Removing *only directional food smell*** does NOT collapse plastic foraging:
baseline plastic food rate 10.104 vs frozen 5.510 and random-motion 6.156. Baseline plastic survival **480/480** steps on all 24 world seeds. In the procedural generator, no-scent plastic rate is 9.106 vs 5.537 frozen.

**Removing the *adjacent food indicator*** while retaining adjacent danger sensing DOES collapse the efficient foraging rate: baseline **5.955 plastic vs5.719 frozen and6.156 random_motion**. The preregistered +15% food-rate threshold versus frozen FAILS. Nonetheless, plain plastic survives 480/480 steps on **all 24 seeds** in both geometries and contacts roughly 9.8–10.0 hazards versus >20 for the uniform-motion policy. Thus this controller **still learns useful danger-related behavior** when the hazard sensor remains; it is not "incapable of learning" in general.

**Removing *also the adjacent hazard indicator*** eliminates that protection: baseline blind plastic **326 mean lifetime steps** vs **401.4** for nonlearning uniformly moving and **357.6** for frozen. On procedural maps it lives **311** vs **329.1** uniform motion. It feeds at **5.777 per100 steps** (baseline) and **5.748 per100** (procedural), both below random motion rates **6.156** and **6.229**, respectively.

This does not prove the plastic agent is *statistically worse* than the random-motion policy: the independently calculated **paired seed bootstrap 95% intervals for food-rate differences include zero**, as detailed below. The prospective criteria require positive advantages, not just parity.

### Secondary metrics and uncertainty

| World / comparison (plastic minus reference) | Observed feeding rate difference per 100 steps | Exploratory 95% paired-world bootstrap interval |
|---|---:|---|
| Baseline no-scent minus frozen | **+4.594** | [+3.615,+5.546] |
| Baseline no-scent minus uniform_motion | **+3.948** | [+2.956,+4.849] |
| Baseline contact-food minus frozen | +0.236 | [−0.366,+0.811] |
| Baseline contact-food minus uniform_motion | −0.201 | [−0.900,+0.426] |
| Baseline blind minus frozen | +0.114 | [−0.275,+0.536] |
| Baseline blind minus uniform_motion | −0.379 | [−0.914,+0.170] |
| Procedural blind minus frozen | −0.018 | [−0.579,+0.520] |
| Procedural blind minus uniform_motion | −0.481 | [−1.428,+0.457] |

These are **exploratory**, independently recalculated paired-world *pooled rate* bootstrap intervals, 2,000 seed resamples; random seed 76051, resampling **24 whole world seeds** and never individual timesteps. The gate checks use the original frozen 24 worlds and exact specified percentage thresholds.

**Food-rate retention** in the most deprived procedural blind arm is **5.748/12.821 = 44.8%** of its own procedural full-food-sensor performance, failing the prospectively required 60% level.

A new research-coded novelty bonus does not fix this limitation: baseline blind novel rate 5.796 versus plain plastic 5.777, procedural blind novel rate **5.507 versus5.748**. This comparison is exploratory; M2 did not preregister a novelty-improvement success threshold.

## 17 frozen development gates (12 PASS / 5 FAIL)

The first **10 integrity gates ALL PASS**:

1. Exactly 768 balanced lifetimes and a second byte-identical complete run — **PASS**.
2. Physically legal actions, bounded body physiology, no wall penetration/resurrection — **PASS**.
3. Identical world seeds, map/shift conditions and initial neural weights across each 16-policy paired block — **PASS**.
4. No scent whatsoever in all three restricted modes — **PASS**.
5. No adjacent food input in \`contact_food\` or \`blind\` — **PASS**.
6. No adjacent hazard lookahead in \`blind\` — **PASS**.
7. 24 distinct procedural map layouts, accessible in both gate phases and with free food/hazard positions — **PASS**.
8. Physically correct tick-240 gate changes and no leaked event schedule/coordinates into neural input — **PASS**.
9. Frozen and uniform-motion weights frozen, uniform motion never WAIT, both plastic actors update synapses — **PASS**.
10. Complete raw trajectories, parameters and fail-safe no-evolution flags — **PASS**.

The 7 **scientific gates**:

11. Baseline/full plastic feeding rate ≥1.15×frozen — **PASS** (13.359 vs5.483).
12. Baseline/no-scent plastic feeding rate ≥1.15×frozen — **PASS** (10.104 vs5.510).
13. Baseline/contact-food plastic feeding rate ≥1.15×frozen — **FAIL** (5.955 vs5.719; only +4.1%).
14. Baseline/blind plastic feeding rate ≥1.15×frozen — **FAIL** (5.777 vs5.663; only +2.0%).
15. Baseline/blind plastic rate ≥1.10×uniform_motion — **FAIL** (5.777 vs6.156).
16. Procedural/blind plastic rate ≥1.10×uniform_motion — **FAIL** (5.748 vs6.229).
17. Procedural/blind rate ≥60% of own procedural/full rate — **FAIL** (44.8%).

**Overall frozen scientific screen = FAIL**, not 12/17 "scientific successes". Software CI being green confirms program execution, not intelligence.

## Independent raw-archive audit

[GitHub Actions green run #45351267233](https://github.com/kkongiu/neuroevolution-lab/actions/runs/45351267233), with original M0 and M1 regression tests PASS plus all new M2 tests, 768-life science and full second science run all PASS.

- GitHub Actions artifact \`flyloop-m2-blind-sensors-procedural-results\`, ID **14901015621**, 30-day retention through **24 December 2026**.
- Independently extracted archive contains exactly two complete UTF-8 JSON datasets of **140,639,199 bytes each**; both **byte-identical** with full-file SHA-256 **\`ac7849a2a5b9d31c9e658e3544c5416e1db87095feda0b67bc005f1abf709cd2\`**.
- All **768 organism lifetimes**, **301,268 actual elementary moves**, **650 logged environmental gate changes** and original/final parameter arrays are present.
- An independent second script inspected every recorded action's 5×8 neural actor feature inputs and 6-dimensional critic inputs: **ZERO disallowed scent, food or hazard channels** across all restricted modes; **all starting maps and actor/critic weights precisely paired** within geometry+seed.
- Another independent action-by-action physics recheck found **ZERO discontinuities in position, invalid motor transitions or mismatches between actual physical rewards and the recorded energy/integrity changes**.
- The compact raw-row canonical SHA-256 was **recomputed from the archived data and matched** M2's logged value: \`4d9d0f0dd893f59cc738cc80e17f43e1c78a4597464126c7da2f1e7dc1af5f7a\`.

## Research decision

The current organism has useful online plasticity, but its impressive synthetic foraging effectiveness is heavily dependent on researcher-supplied **pre-contact resource detection**. Even when plasticity is real, the controller's action representation is too impoverished to learn an explicit location, heading, route, or long-horizon food-search policy when food is no longer visible in adjacent cells. A uniform-moving random creature can match or exceed its feeding rate. This is a representational bottleneck, not something solved by giving the existing critic more confidence or adding a hard-coded "search for food" instruction.

**Next hypothesis, not yet evaluated:** retain the same body and actuator primitives but test whether a truly local delayed cue/memory trace (e.g. heading + short-lived recent sensory transitions, with an equivalent parameter-budget and memoryless control) allows *learned* action persistence and navigation without directly providing the destination food bit. That would make a substantive M3/M2.1 experiment. Pre-fix it and require defeating a random-motion baseline in *previously unseen procedural worlds*, not simply beating frozen weights with an engineered detector. Never claim new action invention or a fly brain.

**No merges.** PR stacked on M1 #87, M0 #86 and older hypotheses; main and M5.6 sealed unchanged. All stop flags FALSE:
\`general_intelligence_demonstrated=false\`,
\`autonomous_goal_invention_demonstrated=false\`,
\`spatial_memory_verified=false\`,
\`independent_biological_replication_complete=false\`,
\`evolution_authorized=false\`,
\`advance_to_evolution=false\`.
