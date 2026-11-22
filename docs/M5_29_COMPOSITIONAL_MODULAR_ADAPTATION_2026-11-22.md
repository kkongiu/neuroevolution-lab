# M5.29 — Modular differentiable adaptation: successful composition ONLY with a researcher-provided product operator

**22 November 2026. Experiment COMPLETED on a new, explicitly declared development split.** The narrowly scoped scaffolded-composition development screen passes **11/11** checks, but **the composition operator was not learned** and out-of-family generalization remains poor. No autonomous general concept discovery, evolutionary advance or main-branch merge.

## Motivation and design choice

M5.28 finally used a genuinely trainable neural support-to-query encoder, rather than the M5.27 hand-programmed GF(2) solver. It learned familiar complete functions (75.6% across fresh sensory splits), but **anti-generalized** (27.3%) on wholly unseen Boolean functions. Instead of merely extending training, M5.29 asks whether **a suitable modular neural architecture** permits transferable task-specific fitting and memory.

**Main intervention:** a five-module differentiable network has one floating scalar per input feature and one task bias. Given a binary sensory vector, the outputs of active feature modules and bias module are multiplied. These five scalars are **genuinely optimized by gradient descent on each task's eight actually experienced chosen-action rewards**. There are **eight random candidate starts**, each trained by **180 full-batch SGD update steps**; only the candidate with the smallest **support-only loss** is retained. None of the eight exam cue outcomes contributes to training or candidate selection.

During fitting, multiple candidate weights and all optimization gradients use float64; the final winning five task parameters are **explicitly cast to float32 before ANY query scoring**. A regression test asserts the stored weights consume exactly **20 bytes per context**. This ensures the 480-bit three-context code figure is not based on an unstated 64-bit runtime representation.

The model's multiplication operation is a **hand-programmed compositional inductive bias** that exactly fits an XOR/parity-style task family. This is therefore **not an autonomously learned combination operator, learned library of latent neural primitives, proof of open-ended plasticity, or discovery of the parity hypothesis class from data**. The generalization result is meaningful mainly as an **architectural causal experiment**: a representational prior matched to the underlying task structure prevents the failures of the generic averaged-embedding network.

The particular per-task module **values** are learned by gradient descent; the common module structure and multiplication are **fixed by the researcher**, not learned across tasks. The module weights are not transferred from earlier tasks: each new task gets its own eight random initializations. The learned task-specific solution is stored as five floats, and A/B/C codes are placed in three **externally labeled** context slots. This experiment does **not** retest hidden-context recognition or synaptic interference across subsequent training of a shared brain.

## Task family, cohorts and information firewall

Four sensory bits yield 16 possible cues. The school generates affine/parity rules; A and B are opposite functions, C another independently chosen mask. Eight sensory cues are observed for each task, **eight disjoint cues are never encountered by that organism before frozen assessment**. Every observed action receives only its actual 0/1 success; the model logically derives a target sign from the chosen action and received reward **because the environment has only two deterministic complementary actions**. There is no oracle call for an unchosen action and the environment's rule coefficient is not passed to a learning arm.

A whole-function split separates:
- **Meta-training familiar rule functions:** 12 masks × 2 biases = 24; mask values are all except (5, 9, 11, 14).
- **Entirely withheld functions:** **masks 5, 9, 11, 14**, both biases. These eight full functions are **not used to meta-train the pooled comparison neural network**.
- **Development evaluation:** 24 seeds **6100–6123** for familiar complete functions, 24 seeds **6200–6223** for withheld complete functions, and 12 seeds **6300–6311** for a deliberately nonaffine added AND interaction. All differ from preceding project development seeds and original sealed M5.6 cohorts.

This is a different whole-mask split than M5.28's masks (3,6,10,12). It is **not independent task-family replication**, because both experiments use the same known Boolean parity generator and the architecture was intentionally chosen to match that generator. The whole-function holdout remains useful for separating support-task inference from memorized specific full-function templates.

**Seven arms** are evaluated for each of three model initializations × (24 familiar + 24 withheld + 12 nonaffine) seed tasks = **1,260 individual rows**:
1. **Modular product + SGD** — task-adapted multiplicative modules with eight support-only random restarts (primary experimental candidate).
2. **Same modular product + scrambled support reward** — identical task cues, action choice and optimizer, but reward bits independently randomized.
3. **Same modular product with no task adaptation** — same initial module weights, no task-specific gradient steps.
4. **Additive module gradient model** — same five trainable scalars, number of restarts and gradient steps, but summation instead of multiplication. This isolates the *combination operator*.
5. **Literal eight-example episodic lookup** — remembers every personally experienced support cue/action/reward, guesses a fixed action on new cues.
6. **Pooled neural DeepSets baseline from M5.28** — independently trained three times using the NEW permissible 24 complete functions, **60,000 historical rewarded meta-training episodes per run**, or 180,000 previous meta-training episodes total. This is a new comparison using the same current function split, **not recycled M5.28 headline results**.
7. **GF(2) symbolic reference** — the experimenter-programmed exact affine solver, separately marked as a hypothesis-class reference; it is **not** part of the modular neural candidate.

Every evaluation arm receives the **identical eight actually chosen action/outcome support observations per task**. No query outcome is supplied to the candidate. The pooled model's shared weights are frozen at evaluation, but the modular model does **adapt five task-specific parameters from the eight support experiences** before its frozen query exam. This is a legitimate few-shot-adaptation distinction, not an equally matched inference compute comparison.

## Quantitative results: truly novel whole functions

[**GitHub Actions #45066965905 — SUCCESS, final post-quantization validation**](https://github.com/kkongiu/neuroevolution-lab/actions/runs/45066965905): all five new M5.29 scientific tests and five original M5.6 regressions passed. Full three-run pooled baseline meta-training, 1,260 arm/seed evaluations, frozen checks and artifact upload completed.

Canonical SHA-256 of raw evaluation rows **9285fac0a12c198aef09879ce04a675533407c74d58d68674a1b96b61b3ece59**. Raw per-task A/B/C accuracy, selected module values, support-only losses, selected restart indices, adaptation update counts, resource measurements and exploratory hierarchical bootstrap results are archived in the 30-day Actions artifact **m5-29-compositional-modules-development**.

### Frozen accuracy on cues NOT encountered in each task's eight support experiences

| Architecture | Familiar full functions / new cues | **Entirely withheld full functions / new cues** | Nonaffine AND stress |
| --- | ---: | ---: | ---: |
| **Differentiable multiplicative module + task SGD** | **99.19%** | **97.22%** | **59.49%** |
| Re-meta-trained generic pooled neural encoder | 70.66% | 38.48% | 47.92% |
| Additive differentiable modules, same optimizer | 63.43% | 35.94% | 62.50% |
| Multiplicative modules, random reward | 50.46% | 52.03% | 48.38% |
| Multiplicative modules, no adaptation | 48.26% | 49.31% | 50.69% |
| Literal eight-example memory, unknown cue defaults to zero | 48.26% | 49.31% | 50.69% |
| **Human-programmed symbolic GF(2) reference** | **100%** | **100%** | 48.61% |

All figures are means over three independent random model initializations and 24 task seeds per affine cohort (12 for nonaffine stress), scored on the eight held-out sensory patterns for each of three externally labeled A/B/C task slots. The design deliberately uses a known deterministic Boolean rule family; the positive result is **within-family composition**, not novel-world cognition.

Against the new-split pooled neural control, the modular product model's benefit on held-out complete functions is **+58.74 percentage points**. The same modules combined additively score **35.94%**, showing that mere extra task-specific parameter updates are insufficient; the **researcher-chosen multiplication** is decisive. Randomizing support feedback reduces performance to **52.03%**, confirming that true experienced consequences contain necessary learning information. A product module with no adaptation is only **49.31%**, confirming that adjusting module weights per new task matters.

The human-coded GF(2) reference remains 100% as expected when the hypothesis family is exactly correct. The task-adapted product misses that ceiling by **2.78 points**, due to optimization variability, not inability of its hypothesis class to represent correct affine rules.

### Adversarial boundary: AND interactions outside the product parity representation

When the simulator adds a nonaffine AND interaction to each rule, the same modular product model declines to **59.49% unseen** (while support examples average substantially higher). This is direct evidence that the engineered product prior cannot universally express novel interactions. It is not evidence that all modular networks would fail on these functions.

The generic pooled neural baseline is **47.92%** under this shift; the additive modular control **62.50%**. This confirms that a representation matched to parity does **not** uniformly dominate under a changed rule generator.

## Memory and compute accounting

The primary candidate stores five adapted **float32 weights per context**: 3 × 5 × 32 = **480 bits** of task-specific code, versus the generic pooled neural's 3 × 32 float32 values = **3,072 bits** and the researcher-symbolic reference's **15 bits**. This does not include the chosen modular function implementation, historical meta-learner weights of the comparator, or the actual support records needed during fitting. Most notably, **eight candidate starts × five float64 parameters, their optimization gradients and all eight support experiences occupy temporary working memory**. The code is not evidence of 480-bit total organism capacity.

Each successful modular task fit performs eight restarts × 180 full-batch SGD steps = **1,440 parameter update iterations over eight real support examples**, plus restart selection using *training-support loss* only. In contrast, the pooled meta model gets **60,000 earlier task episodes** per independent initialization and then only a single frozen query inference. **Full training/optimization compute is not matched**. Do not claim the modular method is faster or more energy-efficient.

The modular network does not learn a library of cross-task synaptic operators: **module values are re-optimized afresh** for each task, and the shared operator is hardcoded. This means the result is a powerful **structural inductive-bias control**, rather than the autonomous executive abstraction mechanism originally sought.

## Exact preregistered 11/11 development gates pass

1. Completely withheld full rules score ≥90%: **PASS, 97.22%**.
2. Improvement over new-split pooled meta neural ≥15pp: **PASS, +58.74pp**.
3. Improvement over additive SGD modules ≥15pp: **PASS, +61.28pp**.
4. Improvement over literal eight-record memory ≥25pp: **PASS, +47.92pp**.
5. Improvement over reward-scrambled product SGD ≥25pp: **PASS, +45.20pp**.
6. Improvement over no-adaptation product ≥25pp: **PASS, +47.92pp**.
7. Explicit performance loss on nonaffine AND stress (≤85%): **PASS, 59.49%**.
8. Train and withheld whole rule masks disjoint: **PASS**.
9. Evaluation query rewards unavailable to learner: **PASS**.
10. No shared neural weights change during evaluation: **PASS**. Note that the candidate does train new **task-specific** weights from support, by design.
11. Three task-specific module codes occupy 480 float32 bits: **PASS**.

**Machine status:**
- \`architecturally_scaffolded_composition_screen_pass=true\`
- \`autonomously_learned_composition_operator=false\`
- \`independent_task_family_replication_complete=false\`
- \`evolution_authorized=false\`
- \`advance_to_evolution=false\`

## Next research question, strictly plasticity-focused

The immediate mechanistic hypothesis is that **compositional structure** addresses the negative transfer found in M5.28, at the expense of supplying the appropriate grammar. For progress toward a self-directed brain, the next experiment must make the brain **learn the operator or choose among operators from its own action-outcome experiences**, rather than giving it product for parity or additive composition for other rule families.

A suitable controlled next trial would offer a small set of **trainable candidate composition mechanisms** trained across multiple task families, then evaluate on genuinely unseen *families*, not merely new masks of parity. It should include stochastic feedback, tightly bounded per-task adaptation, explicitly tracked compute/storage, independent development/replication cohorts, and direct comparison with a fixed oracle operator selector. Do not automatically graduate to ecological evolution based on successful known-family composition.

**Draft PR stacked on M5.28; \`main\` and sealed original M5.6 cohorts untouched.**
