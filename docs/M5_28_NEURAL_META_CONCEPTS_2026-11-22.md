# M5.28 — Learning neural concept representations: same-rule transfer improves, wholly unseen Boolean rules collapse

**Date: 2026-11-22. Status: verified negative development result for transferable NEURAL abstraction. Seven of eleven preregistered scientific/invariant checks pass. Evolution stays blocked.**

## Why M5.28

M5.27 attained 100% on genuinely unseen sensory combinations and compressed each A/B/C task to five bits, but the learner was explicitly given a researcher-programmed affine/parity hypothesis class and streaming GF(2) elimination. It inferred the *coefficients of a known model*, not a transferable neuronal concept or an autonomously invented representation.

The next research question is different: can a small, truly trainable nonlinear neural network **learn its own task representation from experienced action/reward examples** through many prior training lives, and use that representation on both novel examples and **entirely new functions**? Crucially, distinguish these two generalization axes rather than reporting only accuracy on unseen cue combinations.

## The actual neural learner

A DeepSets-style network implements an example-to-concept neural encoder and a concept-to-action neural decoder, using only ordinary tanh activations and backpropagated reward-prediction errors:

- Each task support experience includes **four binary sensory bits, the chosen binary action and its realized success/failure**. Given the environment's known *two deterministic complementary actions*, the observation logically implies which of the two actions is correct for that exact experience. This is a stated structural assumption; no world model parameters, hidden function mask or privileged correct-action interface is supplied to the neural candidate.
- A trainable **5×32 tanh example encoder** produces an embedding for each of eight personally experienced action outcomes. These are averaged into a **32-dimensional float32 task code**. The code is the only task-specific state passed forward to the frozen readout.
- A **(4+32)×128 tanh decoder plus sigmoid action head** predicts the proper action for a novel cue. The 5,057 shared neural parameters are updated via standard **Adam gradients** during meta-training. No XOR equation, parity basis, GF(2) solver, enumerated concept dictionary or true rule coefficient is embedded in the neural encoder or decoder.
- During frozen evaluation, the shared neural parameters remain unchanged. Each novel organism independently receives eight support experiences per context and creates its own three **32-float concept codes**. A/B/C context **labels are explicitly supplied** so the correct slot can be addressed: this is **not** a replication of M5.26's hidden-context identification.
- **No claim of memory compression by bits**: three 32-float32 codes occupy **3,072 bits (384 bytes)**, excluding neural weights/decoder, while three 5-bit GF(2) rule codes need **15 bits** in M5.27 and eight compact four-bit cue + one-bit inferred-label examples per context need only **120 bits** of literal task-example content. This is a learned neural embedding, but not an efficient code under the measured storage units.

Training is genuinely gradient-based; however, the network architecture, its pooling structure, reward interpretation, metadata/context slots and synthetic task distribution remain **researcher-chosen**. A failure is not evidence that every neural architecture or meta-learning method fails to generalize.

## Causal split designed before full CI evaluation

The four-bit input world has 16 sensory configurations. An affine Boolean function has 16 possible parity masks and two bias bits, giving 32 possible functions.

**Only 24 functions** (12 masks × both biases) appear in neural meta-training. All **eight functions associated with masks 3, 6, 10 and 12** (both biases each) are **excluded entirely from meta-training**. The split is at the full-function level, not merely at a task-seed level.

For each task instance, exactly **eight input patterns** form the support experiences and the other **eight never-trained input patterns** are used only for frozen assessment; support features have rank five for the separately tagged symbolic GF(2) positive control.

**Three independently initialized neural models** are trained for **60,000 meta-task episodes each** using real personally experienced action-reward outcomes. A matching **three corrupted-reward neural models** get the **same generated rules, eight support cues, eight query cues and actual chosen actions** at each episode but independently randomized reward bits for training. Thus the real and corruption controls have matched meta-training dose; only reward information differs. Total **six × 60,000 = 360,000 meta-training episodes**. Meta-training query reward is legitimate *observed feedback* on those earlier meta-training tasks; no evaluation query reward is ever shown.

The frozen models are evaluated using all of:
1. **New task instances of previously meta-trained functions**: 24 new task seeds **5800–5823**. The full functions occurred during prior meta-training, but these support/query input partitions and exam instances are new.
2. **Completely withheld functions**: 24 new task seeds **5900–5923** generated solely from masks 3,6,10,12, with no prior meta-training exposure to those complete Boolean functions.
3. **Out-of-family nonaffine interaction**: 12 new stress seeds **6000–6011**, with an added AND interaction; this is outside the affine/parity model family itself.

A/B/C rule contexts are sequentially presented to the evaluation organism, with B negating A and C a different rule; its three independent compact codes remain preserved. Since its shared weights are frozen during evaluation and context identity is given, **successful old-task recall is supplied by keeping the codes in external task-specific slots**. It does not establish continual gradient plasticity in the shared neural weights.

## Explicit comparison arms

For **each frozen neural initialization, seed and evaluation cohort**, all five arms receive the same eight chosen-action support records and zero query-label feedback:

1. **Trained neural task encoder/decoder** — real reward backpropagation across 60,000 earlier meta-training tasks.
2. **Corrupted-reward-trained neural model** — same dose and meta-training task/cue/actions, reward randomized.
3. **Untrained neural model** — identical initialization and architecture, no meta-training.
4. **Literal episodic memory** — preserves exactly the eight seen cue/reward examples per context, guesses action zero for entirely absent cues.
5. **Human-programmed GF(2) symbolic reference** — separately labeled M5.27-style affine-rule solver, using the same actual chosen-action support feedback; it is a **privileged structural-inductive-bias reference**, not part of the neural architecture or a neural learning demonstration.

All evaluation query cues are withheld from the support examples; no model can alter shared neural weights or episodic memory during frozen examination. The symbolic reference is expected to score 100% on affine functions because the experimenter encoded the exact hypothesis family. Storage and processing costs between symbolic/neural/literal mechanisms are not comparable.

## Verified GitHub experiment and artifacts

[**GitHub Actions #45064520277 — SUCCESS**](https://github.com/kkongiu/neuroevolution-lab/actions/runs/45064520277). All six new M5.28 scientific/unit tests and all five original M5.6 neural learner regressions passed. All six 60,000-step meta-trainings and the full frozen evaluation ran, producing **900 per-model/seed/arm evaluation rows** (three model initializations × [24 familiar + 24 withheld + 12 out-of-family seeds] × five arms).

Canonical JSON raw-row SHA256: **07fbb705ef766b5e35838e4481a9a9cd647b1baa75bc730d96e32cdfb7edc010**. All meta-model initialization hashes, original reward-only learning protocol, task code sizes, per-arm A/B/C seen/unseen scores and clustered uncertainty are archived in Actions artifact **m5-28-neural-meta-concept-transfer-development** (30-day retention).

### Primary frozen unseen-cue results — three independently initialized neural learners

| Treatment | New cue, familiar FUNCTION | New cue, wholly UNSEEN FUNCTION | Nonaffine stress |
| --- | ---: | ---: | ---: |
| **Actual reward-trained neural encoder** | **75.58%** | **27.26%** | 53.13% |
| Same network, random meta-rewards | 48.90% | 51.27% | 49.42% |
| Same initial network without training | 50.93% | 48.38% | 50.23% |
| Literal eight-record-per-context archive | 47.40% | 52.43% | 51.74% |
| **Researcher-programmed GF(2) reference** | **100.00%** | **100.00%** | 50.00% |

These figures average held-out eight-cue accuracy for A/B/C at the end of the three-context curriculum, across all 24/24/12 task seeds and **three independent neural training initializations**. All 8 evaluation cue patterns are absent from the per-task support examples; in the withheld-function cohort the **functions themselves** have also never appeared in meta-training.

### Meaning of this result

**Neural learning on a known-function distribution is real**: on fresh tasks of already-seen full Boolean functions, the meta-trained network reaches **75.58%**, compared with **50.93%** for the initial same network and **48.90%** when the meta-training reward is scrambled. This is an actual benefit from reward-guided neural parameter learning across earlier tasks.

**Transfer to entirely new rules fails**: for masks never encountered in training, the same model falls to **27.26%**, worse than 48.38% untrained, 51.27% reward-scrambled and 52.43% literal baseline. This is **negative transfer / over-specialization** on this specific architecture and finite function distribution, not proof that neural methods cannot learn abstractions.

The full-code inference is even actively mismatched on support cues for withheld functions: neural seen-cue accuracy falls to approximately **5.03%**, consistent with severe category-level inversion or anti-generalization. We cannot attribute a unique mechanism without additional representational diagnostics; the observation alone only shows systematic failure outside known function templates.

Across the three independent gradient-training initializations, unseen-rule frozen accuracies were **25.69%, 28.99%, 27.08%**; the weakness is not confined to one random initialization.

### Hierarchical bootstrap (3 independent training seeds, 24 task seeds per main cohort)

Differences are meta-trained neural minus named comparator on unseen-cue A/B/C mean. Bootstrap resamples the trained-model seed and then evaluation task seed (1200 fixed-seed draws), rather than pretending all task rows under one shared encoder are statistically independent. There are only **three** independent model-training initializations, so intervals are exploratory and should not be treated as definitive population-wide uncertainty.

| Main test and comparator | Delta (percentage points) | Hierarchical 95% bootstrap (pp) |
| --- | ---: | --- |
| Familiar functions vs untrained model | **+24.65** | [+20.66,+27.95] |
| Familiar functions vs corrupted-reward training | **+26.68** | [+23.44,+29.80] |
| Wholly unseen functions vs untrained model | **−21.12** | [−25.41,−16.90] |
| Wholly unseen functions vs corrupted reward | **−24.02** | [−27.89,−20.14] |
| Wholly unseen functions vs literal episodic | **−25.17** | [−28.36,−21.99] |
| Wholly unseen functions vs symbolic reference | **−72.74** | [−75.41,−69.91] |

The sign change between conditions is the main finding. It is a compelling **warning against treating learned familiarity with specific task identities as general abstraction**.

## Preregistered screening: seven of eleven checks pass

**PASS (7):**
- Fresh sensory cues from *already meta-trained* rules ≥65% (**75.58%**).
- Improvement over untrained neural on those familiar functions ≥15pp (**+24.65pp**).
- Improvement over corrupted-reward-trained neural ≥15pp (**+26.68pp**).
- Separately provided symbolic GF(2) positive reference ≥95% on withheld affine rules (**100%**).
- Exact function-mask train/test sets disjoint.
- Evaluation query outcomes never provided to learner or fed into learning.
- Frozen evaluation never updates neural model parameters.

**FAIL (4):**
- Entirely unseen rule masks ≥75% (**27.26%**).
- Beats literal episodic records by ≥10pp on new rules (**−25.17pp**).
- Beats untrained same network by ≥10pp on new rules (**−21.12pp**).
- Beats reward-corrupted meta-trained neural by ≥10pp on new rules (**−24.02pp**).

Machine-readable **\`transferable_neural_concept_screen_pass=false\`**, \`independent_task_family_replication_complete=false\`, \`evolution_authorized=false\`, \`advance_to_evolution=false\`.

## What must NOT be concluded and next test

The experiment **does not show that neural abstraction is impossible**. It shows that this particular mean-pooled, 32-float task encoder and 128-tanh decoder, trained on 24 of a 32-function Boolean generator without an explicit compositional inductive bias, fails to recombine knowledge for **the eight held-out full functions**, despite learning similar task templates well.

Reasons a later test might behave differently (hypotheses, NOT results):
- A compositionally structured neural representation may learn reusable operators and reconfigure them for new combinations, rather than implicitly memorizing entire familiar functions.
- Predicting whole functions from pooled support events may require pairwise/relational attention to preserve interactions between examples, not only averaged per-example embeddings.
- Wider diversity of prior training functions, a more suitable learning objective and an independent architecture can reduce memorized-template bias, but simply unblinding the eight held-out masks to tune hyperparameters **would invalidate the holdout**.
- A learned rule-code representation must also be evaluated against its **actual storage cost**; here the 3,072-bit learned code is far less compact than the 15-bit researcher-engineered symbolic code, so there is no demonstrated neural compression.

A defensible next research step is to predeclare an **explicitly compositional learned architecture** (e.g. shared neural operator modules) or an attentional relational learner, and evaluate it on a **new disjoint function-mask holdout and task seeds**, with properly measured memory and compute and no claim of generalization beyond that controlled grammar. This is not an automatic authorization to proceed with evolution or ecological expansion.

**Draft PR stacked on M5.27; \`main\` untouched, original M5.6 sealed cohorts untouched, no evolution.**
