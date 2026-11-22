# M5.31 — VERIFIED result: learned sensory synapse gates, but hard pruning hurts triple interactions

**22 November 2026. GitHub Actions [#45070896363](https://github.com/kkongiu/neuroevolution-lab/actions/runs/45070896363) SUCCESS.**
M5.31 scientific tests and original M5.6 regressions passed, followed by the full
**156-row, 468-context** controlled evaluation. The **10/10 narrow
prespecified screening gates PASS**, but the strongest conclusion is limited:
reward-conditioned consolidation of connections to one HUMAN-PROGRAMMED
conjunction neuron improves three-sensor generalization relative to the
old seven-operator menu; it does **not** teach the brain to invent new
computation primitives, and hard pruning is demonstrably worse than
keeping the soft neuron on triple-AND problems.

The [prospectively declared protocol](M5_31_SYNAPTIC_GROWTH_PROTOCOL_2026-11-22.md)
was published before the completed CI science run. Pilot task seeds
8000/9000/10000 families were used for implementation exploration and
explicitly excluded from the 11100–11407 results; this is **fresh DEV
synthetic seed replication only**, not independent task-family replication.
No original sealed M5.6 ecology cohort was accessed and no evolution
or main merge was performed.

## What genuinely learned

The candidate sees **eight disjoint support sensory cues**, each with a
randomly chosen binary action (selected by the research harness, NOT an
autonomous curiosity/exploration policy) and that action's actual
success/failure. It never receives correct unchosen outcomes, the rule's
parity coefficients, the target interaction mask or the eight unseen
exam cue outcomes.

For each externally visible A/B/C context, 16 randomly initialized
differentiable models optimize five base module floats, four input-synapse
gates and the soft neuron's correction strength using 300 full-batch
support-only SGD iterations. The soft architecture has a **researcher-
supplied multiplicative conjunction** and a human-engineered XOR-like
residual correction. The eight best soft candidate restarts propose
hard incoming-edge masks using fixed thresholds (0.5 gate and alpha);
a hard mask is admissible only with two or more input connections.
A new product-module fit selects a final hard mask by **support-only**
MSE plus a 0.012 penalty for each consolidated synapse. Zero-conjunction
topology is always allowed. This produces **task-specific connectivity
adaptation and pruning**, not de novo invention of a new type of neuron.

The M5.30 seven-complete-operator menu is used only as a directly
comparable **reference arm**, not as the candidate's set of choices.

## Actual frozen unseen sensory-cue accuracy

Three independent optimizer seeds 881/883/887, with 12 plain parity,
16 pairwise-AND, 16 triple-AND and 8 XOR-of-two-disjoint-pairs synthetic
DEV seeds, each with three externally identified contexts and eight
unseen query cues per context; total **156 row trials, 468 task contexts,
3,744 unseen cue grades per arm**.

| Method | Parity | Pair AND | Triple AND | Two disjoint ANDs, outside one-neuron class |
|---|---:|---:|---:|---:|
| **M5.31 grown/pruned synapses** | **99.65%** | **79.86%** | **77.60%** | **42.01%** |
| M5.29 fixed product, real reward | 99.31% | 56.08% | 66.32% | 37.85% |
| M5.31 grown/pruned, scrambled support rewards | 49.65% | 50.87% | 50.17% | 49.65% |
| Random grown topology, real reward | 63.43% | 62.76% | 67.80% | 42.36% |
| **M5.30 fixed seven-operator menu** | 99.65% | **83.68%** | 71.88% | 41.49% |
| Privileged TRUE mask; human oracle on represented tasks only | 99.31% | **96.88%** | **100.00%** | 37.85%* |
| **M5.31 continuous soft neuron without pruning** | **100.00%** | 75.35% | **82.64%** | 42.88% |

*In the two-pair family, no single conjunction can represent the target.
The privileged topology arm explicitly **falls back to no correction**,
so its 37.85% is NOT a literal upper-bound oracle for that family.

### Primary comparisons

- **Triple input, M5.31 vs M5.30:** **+5.73 percentage points**,
  exploratory two-level paired bootstrap **[+3.99, +7.55] pp**.
  The narrow +5pp triple transfer gate passes.
- **Triple input, hard/pruned vs still-continuous soft M5.31:**
  **−5.03 pp**, bootstrap **[−7.81, −2.34] pp**.
  This is an important **negative result about our current pruning rule**:
  the best new architecture is currently the *unpruned* soft neuron for
  triple interactions, not the hard/discrete 492-bit one.
- **Pair input, hard growth vs M5.30:** **−3.82 pp**
  **[−7.12, −0.52] pp**. The older, hand-supplied complete operator
  menu remains better on the interactions it was designed for.
- **Pair input, hard growth vs fixed product:** **+23.78 pp**
  **[+19.70, +28.04] pp**. Versus feedback corruption the difference
  is **+28.99 pp** **[+24.74, +33.68] pp**, consistent with meaningful
  use of experienced reward.
- **Triple input, hard growth vs fixed product:** **+11.28 pp**
  **[+6.86, +15.80] pp**.
- **Two independent conjunctions:** 42.01% for M5.31 hard growth,
  versus M5.30 41.49%, fixed product 37.85%, and scrambled reward 49.65%.
  Our **one-neuron representational capacity is not sufficient** for
  this synthetic family. Performance is even below chance-like 50%
  and the result is retained unmodified; no broad claim about other
  neural architectures.

The paired intervals are **exploratory** and based on only three
optimizer initializations, small shared synthetic task cohorts and
correlated A/B/C contexts. They do NOT establish real-world agent
viability, general cognitive abstraction or broad independent replication.

### Direct structural-plasticity audit

- Pairwise interactions: **83.33%** of contexts consolidate at least
  two connections; mean 1.78 final connections, with **52.78%** exact
  correct incoming masks.
- Triple interactions: **79.17%** consolidate at least two connections;
  mean 1.91 final connections, with only **25.00%** exact correct masks.
- Plain parity: **99.07%** of contexts avoid any nonzero neuron
  entirely, and 99.07% choose the exact zero-connection topology.
- Outside-class two independent ANDs: 79.17% consolidate something
  despite the family not fitting a single conjunction. This is an
  **overfitting warning**, not evidence that the agent discovered
  the two true operators.

The final code stores five **real float32** product weights (160 bits)
plus a **four-bit actual active-synapse mask** per task context, i.e.
**164 bits per context / 492 bits for visible A/B/C**. These are only
*posttraining persistent task-specific code bits*. Sixteen simultaneous
soft initializations, up to eight fresh product refits per context,
historical supports, gradients, inference implementation, fixed
conjunction primitive, and compute/memory overhead are **excluded**.
No matched-compute, faster-training, energy-efficiency or tiny-brain claim
is justified. The single-conjunction model remains strongly structured
by an engineered neural primitive.

## Prospective gates and scientific disposition

All ten prespecified screening checks are true:
1. Pair new-cue accuracy >=70%: **79.86%, PASS**.
2. Triple new-cue accuracy >=70%: **77.60%, PASS**.
3. Pair beats fixed product >=8pp: **+23.78pp, PASS**.
4. Triple beats M5.30 fixed operator menu >=5pp: **+5.73pp, PASS**.
5. Pair beats scrambled-feedback growth >=15pp: **+28.99pp, PASS**.
6. At least 20% pair contexts grow >=2 synapses: **83.33%, PASS**.
7. Exact train/exam cue disjointness: **PASS**.
8. Candidate signature receives no hidden true interaction: **PASS**.
9. Three final task codes exactly 492 bits: **PASS**.
10. Out-of-one-neuron-family negative stress explicitly preserved: **PASS**.

**10/10 is a prespecified narrow DEV screen, NOT 10 independent scientific
successes.** The candidate does not learn shared metaplasticity, invent
a conjunction neuron, select its own support actions intelligently,
operate across hidden contexts, or survive real evolutionary ecology.

The raw JSON artifact \`m5-31-synaptic-growth-development\`
([run #45070896363](https://github.com/kkongiu/neuroevolution-lab/actions/runs/45070896363))
preserves **every one of the 156 trial rows**, synaptic masks, soft
strengths/gates, per-context support-fit costs, accuracy of all seven
arms, and complete cohort metadata. Actions retention expires
**22 December 2026**. Source and fixed seed protocol remain in Git.
**Canonical raw-row SHA-256**
\`34861c069d2f1237ad92e5dfc8240e85c76ded01b1f25cb6cc39c01f02c01afc\`.
An independent download and recomputation from the artifact produced
the exact same SHA-256 from sorted-key compact JSON serialization,
confirming data/summary identity.

## Stop/go and next hypothesis

- \`local_structural_plasticity_screen_pass=true\`
- \`autonomously_invented_neural_operator=false\`
- \`shared_meta_plastic_controller_learned=false\`
- \`independent_task_family_replication_complete=false\`
- \`evolution_authorized=false\`
- \`advance_to_evolution=false\`

**STOP for evolution and claims of general cognition.** The next
well-motivated M5.32 hypothesis is **learning whether/when to preserve
continuous synapses, and enabling a second separately trainable nonlinear
unit**, with strict matched-parameter/compute controls and independent
task-family stress. Prefer learning connectivity and pruning policy
over simply enlarging handcoded rule menus. This proposal has NOT been
implemented or independently validated in M5.31.
