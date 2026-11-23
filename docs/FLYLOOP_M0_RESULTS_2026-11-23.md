# FlyLoop M0 — Verified results, continuous 2D organism

**23 November 2026 | Scientific CI run [#45269254051](https://github.com/kkongiu/neuroevolution-lab/actions/runs/45269254051) SUCCESS | prospective M0 DEV screen: 9/10 (FAIL overall).**

This is the FIRST genuinely continuous sensor-action-learning experiment in the new FlyLoop branch. The agents have energy, damage, local perception, movement and persistent physiology, and live through one independently changed environment. M5.6 remains sealed. Unlike a fly, these agents have a deliberately hand-designed ecology and a small artificial policy, and the novelty bonus is explicit. A positive behavioral score is NOT evidence of spontaneous structure invention.

[Protocol fixed before the first complete Actions science evaluation](FLYLOOP_M0_PROTOCOL_2026-11-23.md). **No after-the-fact changes to thresholds or training conditions.**

## Experimental controls and actual implementation

A 13×13 grid, 16 replenishing food patches, nine damaging hazards, five primitive actions (N,E,S,W,WAIT), energy and integrity that persist every time step. At step **240 of at most 480**, the middle vertical wall's old passages at y=2,10 close and new ones at y=5,8 open. The policy never receives its absolute location, unseen maps or the passage-reconfiguration schedule.

Exactly **32 paired world seeds × three controls =96 separate continuous lifetimes**, seeded 7100–7131. All three arms start with the same eight-action-conditional-weight stochastic softmax actor and six-weight critic (14 initialized real parameters total), same map and physiology. The research-coded sensors include hunger, injury, adjacent food/obstacle/hazard and nearby food scent. An online actor–critic adapts to **actual physical changes in energy and integrity**, not hard-coded correct moves.

- **frozen:** stochastic original action policy, no learned weight updates.
- **plastic:** same policy with online TD-based actor-critic modification.
- **plastic_novelty:** same learning rule with *researcher-programmed* additional sensory novelty bonus proportional to inverse square-root sensor-pattern encounter count. It did not spontaneously invent curiosity.

All controls use the same five primitive moves. No hard-coded action labelled forage, flee, climb or gate-solve exists. A zero-novelty coefficient reproduces EXACTLY the plastic agent's entire trajectory and final parameters.

## Observed results across all 32 worlds (means)

| Metric | Frozen original policy | Online plasticity | Plasticity + engineer-defined novelty |
|---|---:|---:|---:|
| Food items consumed during lifetime | **21.09** | **67.19** | **75.22** |
| Steps lived, max 480 | **376.84** | **465.00** | **480.00** |
| Proportion surviving all 480 steps | **34.38%** | **93.75%** | **100.00%** |
| Hazard contacts per lifetime | **21.25** | **9.69** | **10.50** |
| Collisions with wall | 45.88 | 38.22 | 36.50 |
| Revisited previously harvested food | **13.47** | **54.41** | **61.97** |
| Unique visited cells | 55.09 | 72.09 | 74.72 |
| Food consumed *after* gate switch | **7.59** | **37.19** | **43.06** |
| Completed gate crossings *after* switch | **0.22** | **1.28** | **1.47** |
| Mean integrity (while active) | 57.19 | 74.59 | 72.62 |

**Real improvement:** Compared with the fixed actor, plasticity yields **+46.09 food items per lifetime** (exploratory paired world-seed bootstrap 95% **[+40.78,+51.16]**), **+88.16 steps survived** (bootstrap **[+47.25,+128.66]**), and **+29.59 food items after changing barrier** (bootstrap **[+24.78,+33.91]**). The controller adapts within a continuous lifetime rather than being reset after each sensory query.

The fixed original controller also eats; the comparison is NOT between living and completely immobile agents. It is between stochastically acting fixed and plastic neural weights with equal action/sensor primitives.

**Added novelty** vs identical plain plasticity increases food **+8.03** per lifetime (exploratory paired-bootstrap **[+4.47,+11.69]**), post-switch food **+5.88** (**[+2.63,+9.56]**), but additional completed post-switch gate crossings average only **+0.19**, bootstrap **[−0.28,+0.66]**, missing the preregistered *+0.50 crossings* minimum. On this map, novelty improves resource collection but **does not demonstrate reliably better new-passage discovery**. Hazard contacts are also slightly higher with novelty than plain plasticity, underscoring that exploration has costs. No claim of truly learned novelty motivation.

**Important experimental limit:** all three controllers see **engineered food scent** and have physical physiology whose food-energy relationships are explicitly specified. Some apparent competent foraging may be encouraged by this strong researcher-designed representation. The agent does not invent the sensor, movement primitive or purpose of being alive. This is an applied minimal artificial ecology, not an independently evolving organism.

## Frozen gates: 9 PASS, 1 FAIL

1. 32×3=96 complete paired rows, deterministic duplicate full scientific run — **PASS**.
2. Legal primitive actions, physically constrained steps, valid energy/health, no passage through walls — **PASS**.
3. Exact same seed map and same initialized 14 weights across all three arms — **PASS**.
4. Frozen weights invariant; plastic and novelty both updated online weights — **PASS**.
5. Zero novelty beta gives byte-identical trajectory/weights to the plain plastic controller — **PASS**.
6. Independent reconfiguration from passages (2,10) to (5,8), no privileged gate schedule input — **PASS**.
7. Mean plastic food ≥110% of frozen — **PASS** (67.19 vs 21.09).
8. Mean plastic lifetime ≥105% of frozen — **PASS** (465.00 vs 376.84).
9. Mean novelty post-change food ≥110% of plastic — **PASS** (43.06 vs 37.19).
10. Mean novelty completed post-change crossings ≥ plastic +0.50 — **FAIL** (1.47 vs 1.28, **+0.19**).

**Science screen FAIL overall despite 9/10 gates and software CI success.** No threshold changed. This pilot should NOT be called an evolved fly brain, high-level planning or novel abstract reasoning.

## Reproducibility and audit

- Official successful GitHub Actions workflow [#45269254051](https://github.com/kkongiu/neuroevolution-lab/actions/runs/45269254051); Python test suite and two complete science commands succeeded.
- Artifact \`flyloop-m0-ecology-results\`, ID **14864006980**, retention through **2026-12-23**. The downloaded ZIP was independently inspected: each of \`m0_persistent_ecology.json\` and \`m0_repeat.json\` is **21,104,863 bytes**; contents identical **byte-for-byte**.
- Exactly **96 raw organism lifetimes**, **32 world seeds**, and **42,302 logged environment trace entries**, with real actions, position, energy, integrity, food/hazard/collision and the original/final controller weights.
- Canonical compact raw-row SHA-256 independently recomputed from the archive: **\`406c7648657c8695e29d9622087d0cc9d187053090599d30e4d23c6d39de449e\`**; matches the reported result.
- 14 actor+critic real-valued parameters per organism; Python float storage makes this *not equivalent to a 56-byte float32 implementation*. No performance/memory comparison against M5.6 or M5.34 is justified.

**Causal interpretation:** Online plasticity improves feeding/survival on these paired synthetic world maps and engineering constraints. Novelty bonus helps consumption, but its explicit crossing advantage does not clear a preregistered gate. Behavior remains partly shaped by research-chosen sensors, reward and world structure. The next high-value step is to **remove or degrade the engineered food-scent channel** and evaluate transfer to layout/patch schedules that were not part of the fixed M0 test. Reuse the same action set and controlled seeds without selective retuning, and compare sensory ablations rather than adding cognition modules.

No general intelligence, spontaneous abstract alternative action, fly brain replication or evolutionary fitness claim. Flags remain:
\`general_intelligence_demonstrated=false\`;
\`autonomous_operator_invention_verified=false\`;
\`independent_ecology_replication_complete=false\`;
\`evolution_authorized=false\`;
\`advance_to_evolution=false\`.

FlyLoop M0 remains a **DRAFT PR stacked on M5.35**, no merge to main and original sealed M5.6 unchanged.
