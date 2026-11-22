# M5.27 — Compact concept induction with frozen unseen-cue tests

**2026-11-22. Real verified development experiment. 12/12 prespecified developmental tests pass for the narrowly defined affine/parity concept-compression task; 0 evidence that the agent has learned how to invent its own inductive bias.** Evolution, main and protected original M5.6 cohorts remain untouched.

## Question and change relative to M5.26

M5.26 preserved three known-context task competencies using only 12 literal records (one per sensory cue per inferred context), beating same-capacity FIFO/reservoir and keeping A/B/C at 100%. But tests repeatedly used the **same four known cue patterns**; they did not distinguish memory from abstract rule understanding.

M5.27 uses **four input bits**, 16 possible input patterns per context. Only **eight** are presented during training, selected randomly per context and seed so that the augmented affine feature matrix has full rank five. The other eight are **never shown in training** and are evaluated in frozen exams, before any old-task refresher. It isolates **functional extrapolation within a predefined Boolean family**, not general natural-world intelligence.

**Major deliberate difference:** A/B/C context identity is **explicitly observable in M5.27** to isolate concept induction. This is NOT a replication of M5.26's hidden-context recognition. The environment's *true rule coefficients* are never provided to a non-oracle learner.

## Researcher-provided hypothesis class — the decisive caveat

An affine Boolean function on four input bits has the form:

\`answer = bias XOR (w0 AND x0) XOR (w1 AND x1) XOR (w2 AND x2) XOR (w3 AND x3)\`.

Thus its complete conceptual description is **five bits** (one bias plus four Boolean coefficients). A model that is **programmed in advance to search only this family** can infer its rule from five linearly independent observations, and can answer all 16 possible inputs once identified. It is not necessary to store eight, sixteen or hundreds of repeated episodes permanently.

### What is learned from the organism's own experience

The proposed \`AffineCompiler\` uses incremental Gaussian elimination over GF(2). From a chosen binary action \`a\` and the reward actually observed \`r ∈ {0,1}\`, it computes the logically implied desired action as \`a XOR (1−r)\`. This relies on the explicitly known **two-action, deterministic, complementary binary reward** structure; it does **not** call the environment for the counterfactual outcome or receive a teacher-provided correct action.

It maintains only independent equations for the chosen rule. During acquisition it may require **up to five independent six-bit equations = 30 bits of transient solver state per context**. Once the rule is uniquely solved, it discards the equations and retains only a five-bit coefficient code. After all three tasks A/B/C are learned, the **persistent compiled rule codes occupy 3 × 5 = 15 bits**. This is not a claim that the entire organism, its context input, transient solver, computation or simulated Python object requires only 15 bits.

The algorithm and correct family are **hand-designed**. Therefore the successful result is *inference of rule parameters within a known hypothesis class*, **not the neural invention of rules or a general-purpose learned executive**.

## Tasks, exposures, controls and cohorts

- New main development cohort: **24 seeds 5100–5123**, no overlap with earlier experimental seeds or original sealed M5.6 validation.
- Original conflicting task notion retained: **A and B are exact negations of one another**; C uses a different parity mask. All masks require at least two contributing features, preventing the result from being a trivial one-bit association. Parameters vary with seed.
- 3 tasks A/B/C, sequential **A (320 actual trials) → B (320) → C (320) → A (160) → B (160)**, giving **1,280 lived chosen-action outcomes** per subject. Each phase repeatedly presents only its task's prespecified eight **training-only** input patterns, balanced and shuffled in eight-cue mini-blocks.
- Every frozen test evaluates **the eight never-trained patterns for each context** (ten repeats each), as well as eight known patterns for a separately reported recall check. Exams update no memory, model parameters or random state.
- All main-arm candidates except the privileged oracle receive exactly **one model-update call per actually experienced reward**, but arithmetic and capacity differ dramatically by architecture. The oracle receives the exact rule coefficients and is an **unfair diagnostic upper bound** with no learning updates.
- An additional **12 independent development seeds 5200–5211**, four arms each, are tested on a **deliberately nonaffine** Boolean function containing an AND interaction. This is a boundary/negative-control stress test, not a held-out validation set for an unrestricted class of tasks.

### Main comparison arms

1. **\`concept_compressor\`**: five-bit-per-context affine/parity solver trained only with actual action/outcome feedback, the primary candidate.
2. **\`episodic8\`**: at most eight complete, personally experienced context/cue/action/reward records globally; unknown cue defaults to a fixed action.
3. **\`episodic24\`**: all eight distinct trained cue records per context can be preserved (24 global). Even perfect familiar-cue recall provides **no rule to solve unseen patterns**.
4. **\`online_neural\`**: a 24-tanh-hidden-unit **265-parameter** chosen-action reward predictor updated once per lived trial. This is an intentionally compact but real neural comparator, not an optimized search across all possible neural architectures.
5. **\`wrong_prior\`**: a model restricted to constants and one-feature rules, an incorrect hypothesis class for the actual parity tasks.
6. **\`scrambled_reward\`**: same GF(2) concept-solver but trained with independent randomized reward bits instead of its actual feedback, to test whether truthful experience causes the correct rule.
7. **\`oracle_rule\`**: privileged true coefficients, never supplied to other learners.

**216 total trajectories**: 24 main development seeds × seven arms = **168**, plus 12 nonaffine stress seeds × four arms = **48**.

## Verified CI and artifacts

[**GitHub Actions #45060894586 — SUCCESS**](https://github.com/kkongiu/neuroevolution-lab/actions/runs/45060894586). All **six new scientific/unit tests** and **five original M5.6 learner regression tests** passed, followed by the full cohort experiment and archival upload.

Raw main rows SHA-256: **c718aeaf72b67358a9b0141f8473dc8e3004a95548f024fb02d8386c0e91411f**. The full machine-readable report including all 216 per-trajectory rows, A/B/C frozen seen versus heldout accuracy, code bits and regression controls is in the **30-day** GitHub Actions artifact \`m5-27-rule-compression-development\`.

### Truly unseen input accuracies (24-seed main cohort)

The pre-refresher compound is the mean of **A heldout after C, B heldout first acquisition, B heldout after C, C heldout first acquisition**. These tests exclude every input pattern trained under the corresponding context.

| Treatment | A unseen after C | B unseen after C | C unseen after C | Four-metric unseen compound | Mean seen-cue accuracy after C |
|---|---:|---:|---:|---:|---:|
| **Affine concept compressor (5 bits × three rules)** | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **100.00%** |
| Episodic memory 24 records | 48.44% | 51.04% | 47.40% | 49.48% | **100.00%** |
| Episodic memory 8 global records | 48.44% | 51.04% | 47.40% | 49.48% | 66.84% |
| Online 24-hidden-unit neural model | 43.75% | 43.23% | 52.08% | 45.83% | 69.10% |
| Incorrect single-feature-only hypothesis class | 29.17% | 29.69% | 29.69% | 29.56% | 70.49% |
| Compressor with randomized reward | 45.31% | 46.88% | 46.35% | 46.35% | 46.88% |
| Privileged true-rule oracle | 100% | 100% | 100% | 100% | 100% |

The 100% generalization of the affine candidate is achieved on all 24 main development seeds. Every one of the three compiled codes is exactly five bits, with **zero retained GF(2) basis equations** after learning; no contradiction occurs under the affine task family. The 24-record episodic reader perfectly answers known training cues but does not extrapolate to the held-out half, showing that retrieval of literal observations is distinct from functional rule inference.

The compact online neural comparator does not learn parity generalization well under the tested architecture, optimization budget and 320 samples per context. It **does not show neural networks are incapable** of learning parity or compressing rules with other architectures/training.

### Paired bootstrap comparisons (24 main development seeds)

All differences are \`concept_compressor\` minus the named comparator on the four-metric unseen compound; 1200 fixed-seed paired bootstrap draws. Intervals are exploratory, **not independent confirmatory replication**.

| Comparator | Mean advantage | 95% paired bootstrap interval |
|---|---:|---|
| Episodic24 | **+50.52 percentage points** | [+47.01,+53.78] |
| Episodic8 | **+50.52 pp** | [+47.01,+53.78] |
| Online neural predictor | **+54.17 pp** | [+50.39,+57.81] |
| Incorrect hypothesis family | **+70.44 pp** | [+68.62,+72.14] |
| Randomized-reward concept inference | **+53.65 pp** | [+48.44,+58.72] |
| Oracle true coefficients | 0 pp | [0,0] |

These strong differences reflect a **correct researcher-provided hypothesis family**, not discovery of the family by the learner.

## Nonaffine counterexample — explicit limit of generalization

We separately introduce \`answer = affine_parity(x) XOR (x0 AND x1)\`, which lies outside the programmed affine class. The learned compiler still receives only its own real chosen-action reward but cannot represent the true function with five affine bits. On 12 new **development stress seeds**, its mean unseen compound accuracy falls to:

| Stress treatment | Unseen compound |
|---|---:|
| **Affine compressor (wrong family)** | **52.34%** |
| Literal episodic24 | 48.70% |
| Wrong single-feature prior | 40.36% |
| Privileged correct-rule oracle | **100.00%** |

The compressor also registers **about 205 prediction/equation inconsistencies per life** under nonaffine stress. This is direct evidence that **a compact stored formula is not sufficient unless the hypothesis class can express reality**. The stress is a predeclared controlled counterexample, not an exploratory repair of the failed cases.

## Preregistered development screening: 12/12 pass

All twelve conditions pass:
- A retained on unseen cues after C ≥90%.
- B retained on unseen cues after C ≥90%.
- B first acquired on unseen cues ≥90%.
- C first acquired on unseen cues ≥90%.
- Combined four-metric unseen accuracy ≥90%.
- ≥+25pp compared with 24-record literal memory.
- ≥+25pp compared with eight-record literal memory.
- ≥+18pp compared with wrong single-feature prior.
- ≥+18pp compared with randomized-reward training.
- All 24 organisms compile three five-bit rules exactly.
- No within-family observed-rule contradictions.
- The nonaffine stress reveals loss of generalization (≤85% unseen).

**\`affine_concept_generalization_screen_pass=true\`**. Equally important: **\`learned_neural_rule_compression_demonstrated=false\`**, \`independent_task_family_replication_complete=false\`, \`evolution_authorized=false\`, \`advance_to_evolution=false\`.

## Interpretation and what we have NOT accomplished

**Supported:** In a known binary deterministic task family, an organism-like learning process can use only feedback from actions actually taken to infer a compact functional representation, preserve it across sequential conflicting tasks, and correctly predict eight completely *untrained* sensory combinations per context. This is **rule-parameter learning**, not just remembering examples. The positive result is causally supported by wrong-prior, reward-scramble, literal-memory and explicit out-of-family controls.

**Not supported:** the neural brain learning how to INVENT a rule class; a trainable executive learning memory-write decisions; biological synaptic stability without external symbolic memory; inference when the environment does not reveal context identity; open-world concepts or new task families; stochastic or multiclass rewards; memory/compute equivalence to the baselines; independent confirmation.

The current solution is a **hybrid engineered symbolic-memory learner**, not a learned general-purpose neural abstraction process. To claim a self-directing neural brain, future work must allow it to **learn the functional class itself** or neural representations from lived data, with independent held-out task-family evaluation and honest memory/compute caps. This is an architectural research direction, **not authorization to run evolution or unseal earlier validation sets**.

**Draft stacked on M5.26. \`main\` unchanged; no original M5.6 sealed access or ecological evolution.**
