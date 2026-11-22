# M5.10 — Growing modular neural circuits and return-to-context retention

**2026-11-22. Status: experiment implemented, run and audited; qualified positive evidence for contextual interference reduction, NO evidence yet that growth itself improves reasoning over equivalent preallocation.** Abstract diagnostic not physically embodied M5.8 ecology, no reproduction.

## Research question

Can a biological-style controller grow new neural circuits rather than overwrite fixed shared synapses, retain earlier learned associations, and save allocation capacity by allocating neurons only when contexts become encountered?

Native M5.6 recurrent controller: each independent module is **8 hidden neurons, 260 neural parameters**. Per-action individual policy gradient and scalar observed reward; no correct-action supervision, future information, or hidden rule provided.

A development world presents **A → B → A → C → B**, 160 actions per phase (800 actions total), balanced binary cues in every 40-trial window. The hidden correct action for context `c` is `cue XOR private_rule[c]`; rule bits generated per seed, not supplied in observations. Context **identity is directly observable** via a cue feature. That is an important engineered advantage: this is a test of contextual modular retention, NOT spontaneous change-point detection in a feature-identical environment.

### Three matched architectures

1. `small_shared`: 1 common native 8-unit neural RNN, 260 parameters; sequential contexts interfere because they share synapses.
2. `large_preallocated`: 3 separate 8-unit RNNs (24 hidden units, 780 parameters), instantiated at organism birth. Engineered context-to-expert lookup chooses the correct circuit.
3. `growing_modules`: starts with one 8-unit RNN (260 parameters); allocates a new 8-unit circuit upon first *sensed new context*, up to **24 units total**; stores, freezes while idle, and recovers earlier circuits when the context returns. **No access to hidden rule**, and no learned growth trigger: this experiment checks feasibility and consequences of structural growth with a known sensory distinction.

The initial genotype of each per-context circuit, learning rate, stochastic action tickets, feedback opportunities and frozen inactive behavior are **identical** across preallocated and growing architectures. This preallocation control is deliberate: they must produce identical answer and routing traces, proving that any differences concern delayed allocation and not improved policy learning. The baseline small network uses the same context 0 initialized circuit, but no specialist networks.

**Allocation cost proxy**: sum of currently allocated hidden units over all 800 action ticks. It is NOT an actual joules-per-neuron energy accounting, which would also have to consider idle circuitry, training, physical birth, and simulation time. The modular reference does not constitute a monolithic densely connected 24-unit network; the current experiment cannot resolve whether dense brain expansion helps beyond fixed modularity.

## Real executed evidence

[GitHub Actions run #45029687086](https://github.com/kkongiu/neuroevolution-lab/actions/runs/45029687086) completed, unit tests passed and raw 24 × 3 = **72 simulation records** archived as `m5-10-growing-modular-brain-contextual-recall` (30-day retention). Results digest: `93cd59d7c9b59a7f6ca9e2b994d0f32bdbd8eac91243857d5da3d79e1ae972d3`.

| Metric | 8-unit shared | 24-unit preallocated | Growing 8→24 |
| --- | ---: | ---: | ---: |
| Mean accuracy over 800 trials | 71.13% | 75.45% | **75.45%** |
| Late first A phase | 78.96% | 78.96% | 78.96% |
| Initial window when A returns | 76.56% | 85.42% | **85.42%** |
| Late first B phase | 69.69% | 77.60% | 77.60% |
| Initial window when B returns | 63.23% | 86.56% | **86.56%** |
| Final window first C phase | 75.63% | 76.15% | 76.15% |
| Modules initially | 1 | 3 | **1** |
| Modules at end | 1 | 3 | 3 |
| Allocated neuron-ticks | **6,400** | 19,200 | **14,080** |

For A recall growing minus small = **+8.85 percentage points**; for B recall = **+23.33 pp**. Improvement measured within the same cue generation / founder seeds, but no independent sealed validation on a different task family, no inferential confidence intervals yet. A returning-context accuracy even exceeds first-visit late-window accuracy for modular variants, partly because each expert retains its personal weights and hidden states; that is **context-specific retention**, not proof of task-general intelligence.

**Preallocation vs growth**: Both `answer_trace_sha256` and `active_expert_trace_sha256` match **on every one of the 24 seeds**, by test and all-run audit. Their achieved decision competence is equal **by construction**. Growth saves 5,120 / 19,200 = **26.67% allocated neuron-ticks** compared with preallocation, but we cannot infer actual energy savings without a cost model.

### Promotion gate

Four explicit development screen criteria:

- Growing first return A >= 65%: **PASS** (85.42%)
- Growing first return B >= 65%: **PASS** (86.56%)
- Growing beats small return A by at least 10 pp: **FAIL** (+8.85 pp)
- Growing/preallocation same behavior: **PASS**

**3 of 4 conditions passed. No promotion to evolution or ecological claims.**

## What this does and does not validate

Supported:
- Implemented **structural neural expansion** while alive without overwriting earlier individual circuits.
- Showed modular conditional selection materially improves retention versus a single shared 8-unit brain, especially on returning context B.
- Verified equivalent preallocation learns and acts identically, so growth is a **resource allocation policy** and not independently superior as a learning algorithm in this setup.
- Demonstrated incremental allocation cost savings for this specific A/B/A/C/B schedule.

Not supported:
- No learned or autonomous decision about when to grow; allocation triggered by a programmer-readable new sensory-context feature.
- No genuinely unknown task structure, unmarked rule reversal, no causal world model or reasoning generality.
- No biological reproduction, inherited grow-vs-retain predispositions, physical energy costs or predator ecology.
- The fixed single shared model has fewer total synapses, so its inferiority cannot distinguish catastrophic overwriting from insufficient parameter count. Comparing against monolithic parameter-matched 24-unit recurrent controllers and parameter-matched no-growth specialists is an open experimental need.

## Next preregisterable work

1. Separate **capacity from growth**: parameter-matched dense 24-unit recurrent baseline (same observations, trial budget), plus equivalent modular preallocation as already validated.
2. Build a learner that decides when to allocate via persistent *prediction error* and uncertainty, WITHOUT receiving an explicit context identifier and without having access to change-point ground truth. Measure false growth rate, accuracy, remembered skill and allocated computational cost.
3. Give new modules an explicit physical resource cost, pruning budget and possible reuse; compare with conserved initial models.
4. Only after independent evidence of online skill development and rule-change generalization, include heritable initial growth/plasticity parameters in a later population-level evolutionary study. Leave source/final/replication cohort seeds untouched.

All changes remain on a stacked **draft** branch; `main` unchanged.
