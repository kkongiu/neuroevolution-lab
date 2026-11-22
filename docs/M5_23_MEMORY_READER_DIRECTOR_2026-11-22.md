# M5.23 — Specializing the neural brain into memory, reader and director

**Date 2026-11-22. Status: completed positive *memory-access* development experiment.** The full design achieves perfect recall and acquisition in this tiny school, but **there is NO evidence that a learned director adds value over simple memory-only lookup**. Ten of thirteen preregistered exploratory criteria pass; overall specialized-controller STOP/GO fails. This is an important architectural diagnostic, **not proof that biological plasticity or open-ended continual learning has been solved**. Evolution, ecology expansion and use of original sealed world cohorts remain blocked.

## Motivation: why freezing a neural expert was insufficient

M5.22 attempted (1) two neural populations with a self-frozen archive and a learned soft mixture director, and (2) alternating which group can learn. Both failed: after learning B and C, A/B recall was near chance despite freezing one expert. The missing distinction was **storing a representation** versus **retrieving it reliably when behavior requires it**.

M5.23 explicitly isolates three roles:

1. **Writer and stable autobiographical memory**: each actual chosen action and its observed reward is appended after the experience is completed to a capped archive. Training on B or C does not rewrite old A records. Context buckets have **43 A, 43 B and 42 C** places, total **128**; the assignment is made by the researchers using the **explicitly available task identity**. This is an engineered episodic key-value store, NOT a new class of biological neurons or autonomous discovery of memory categories.
2. **Reader**: a similarity-weighted estimate of success from the archive, comparing the observed context, x, y and the candidate action with stored personal events. Four similarity weights are initialized to a fixed heuristic and can be updated from actually experienced reward prediction errors in the learned-reader arms. The reader never calls the world oracle for a candidate action.
3. **Director and fast network**: a 16-hidden-unit action-conditioned neural success predictor, with **145 trainable parameters**, and a **10-parameter logistic routing gate** combining the fast prediction and the archived prediction. The gate is updated only with feedback from the chosen action. The mixture and address are intentionally small; this is not an executive general intelligence.

All eight arms instantiate the same **159 total possible model parameters**, founder initialization and bounded archive. Ablations activate different gradient subsets; equal *update calls* cannot be mistaken for equal *active parameter update counts*. No replay weight updates except the explicitly separate fivefold-cost positive reference.

## Training environment and controlled ablations

- Fresh **development-only seeds 4700–4723** (24), disjoint from M5.6 sealed and earlier experiment cohorts. No independent task-family replication has been performed.
- Identical researcher-provided A→B→C→A→B task sequence: **400 A XOR decisions, 400 B inverse-XOR decisions, 400 C copy-x decisions, 160 A and 160 B refresher decisions**, for **1,520 actual lived action outcomes per individual**.
- Each 40-trial block contains exactly ten of each of four binary (x,y) patterns. Two actions, immediate binary reward. The **task identity A/B/C is an explicitly observable one-hot input** to the fast network and archive addressing. The policy never sees the environment's privileged correct action, only actual success/failure of what it chose.
- All frozen exams, including A and B immediately after C and **before** any refresher, reshuffle the **same four familiar input patterns per context**. They do not modify fast weights, reader, routing gate, memory or action RNG. Additional exams force **archive-only** and **fast-only** readouts using the same individual, so preserved content can be distinguished from effective access.
- Historical memory read cost and available records are deliberately counted for all non-replay arms, including ablated arms. Their memory capacity is exactly 128 and archive-reading computations are **422,895 old-record comparisons per seed during learning, plus 656,160 per seed during frozen exams**, the same count across all seven non-replay arms (not efficient deployment figures). Duplicate episodes are aggregated by multiplicity for speed, **preserving their total weighting** in the similarity-based reader.

### Eight arm protocol

| Treatment | Memory read available to policy | Train reader similarity | Train director | Train fast weights | Weight-update calls/life |
| --- | --- | --- | --- | --- | ---: |
| **Fast only** | No; still computed for matched accounting | No | No | Yes | 1,520 |
| **Archive only** | Yes, 100% trust in archive | No | No | **No** | 1,520 bookkeeping calls, **0 fast weight updates** |
| **Fixed fusion** | Yes, fixed 50/50 weight | No | No | Yes | 1,520 |
| **Fixed reader + learned director** | Yes | No | Yes | Yes | 1,520 |
| **Learned reader + learned director** | Yes | Yes | Yes | Yes | 1,520 |
| **Context-blind learned reader/director** | Yes; addressing cannot distinguish A from B | Yes except context feature | Yes | Yes | 1,520 |
| **Corrupted-memory director** | Yes; archived reward replaced with independent random bit | Yes | Yes | Yes; fast brain still trains on **real** reward | 1,520 |
| **Replay 5× positive reference** | Yes | Yes | Yes | Yes | **7,600** (NOT compute-matched) |

The mixed predictor is trained by the log-likelihood gradient of the **personally received scalar outcome**. The decision gate is allowed to give different weights to fast and memory predictions; the reader's four similarity coefficients also get differentiable updates unless ablated. The memory store itself is not updated by a gradient and cannot rewrite historical outcomes when training on a new context.

## Verified empirical results

[Successful GitHub Actions run #45056107218](https://github.com/kkongiu/neuroevolution-lab/actions/runs/45056107218) ran **6 new scientific-invariant tests** plus **5 original M5.6 neural regressions**, all passing. Completed **24 development organisms × 8 treatments = 192 full trajectories**, with full raw per-individual frozen exam metrics, learned gate diagnostics, read-compute ledgers and 24-seed paired-bootstrap comparisons archived as **m5-23-specialized-memory-reader-director-development** (30-day retention).

Canonical JSON raw rows SHA-256: **17846811710a155ec27b556b0a7d4e1e8335525fa9e56e6d8d272842b419076d**.

### Frozen exam success rates — 24-seed means (%)

| Treatment | A initial | **A after C** | B first learned | **B after C** | C first learned | Mean four-metric pre-refresher | A archive readout after C | B archive readout after C |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fast-only | 100.00 | 48.96 | 85.42 | 56.25 | 95.83 | 71.61 | **100.00** | 97.92 |
| **Memory-only** | **100.00** | **100.00** | **100.00** | **100.00** | **100.00** | **100.00** | **100.00** | **100.00** |
| Fixed fusion 50/50 | 100.00 | 98.96 | 100.00 | 100.00 | 100.00 | 99.74 | 100.00 | 100.00 |
| **Learned director, fixed reader** | **100.00** | **100.00** | **100.00** | **100.00** | **100.00** | **100.00** | **100.00** | **100.00** |
| **Learned director and learned reader** | **100.00** | **100.00** | **100.00** | **100.00** | **100.00** | **100.00** | **100.00** | **100.00** |
| Context-blind reader and learned director | 100.00 | 50.00 | 85.42 | 51.04 | 100.00 | 71.61 | 50.00 | 50.00 |
| Randomized stored outcomes | 82.29 | 45.83 | 78.13 | 55.21 | 95.83 | 68.75 | 50.00 | 52.08 |
| Replay with 5× weight updates | **100.00** | **100.00** | **100.00** | **100.00** | **100.00** | **100.00** | **100.00** | **100.00** |

All old-skill retention tests are frozen **before** the return to old-task teaching. This is **development performance** only; exam patterns are the familiar four binary cues in each externally labeled context.

### Causal result A — a retrievable archive prevents forgetting on this exact benchmark

Fast-only after C is near chance on A (**48.96%**) and B (**56.25%**); its **own archived observations**, read independently, would nonetheless return **100% on A** and **97.92% on B**. It is therefore clear within this model that the *representation of correct consequences is stored*, but ordinary fast synaptic weights and actions have lost effective access.

The full learned-reader/director scores **100% on A, B and C** without old-task refresher, using the exact same 1,520 lived episodes as fast-only. Across 24 paired seeds, its pre-refresher joint score beats fast-only by **+28.39 percentage points**, exploratory paired bootstrap 95% interval **[+26.04, +30.99] pp**. This supports **memory access** as the decisive mechanism in this deterministic classroom.

### Causal result B — context-indexed addressing is essential here

When the associative reader is deliberately stripped of the *context* feature, its A and B direct archived readout accuracy becomes **50% / 50%**, despite identical experience record capacity and a director still observing context in its fast features. The full specialist beats this ablation by **+28.39 pp [26.82, 30.21]** in pre-refresher score.

When the archive content is replaced with independent random reward bits, performance drops to **68.75% pre-refresher**, versus **100%** with genuine records. The full model beats the corrupted-memory control by **+31.25 pp [27.86, 34.64]**. Its director gate generally learns to down-weight unreliable memory (mean gate around **0.35**, versus **0.948** with true history); thus the director adjusts its trust in memory, but this alone does not prove improved decision quality over simple retrieval when memory is trustworthy.

### Causal result C — **the learned director is not needed in this small school**

This is the most important limitation:

- **Memory-only** with *fixed, hand-initialized addressing* scores **100%**, exactly equal to the learned reader/director.
- **Fixed addressing + learned gate** scores **100%**, exactly equal to the learned reader/director. Thus *learning similarity weights* yields **0 pp** demonstrated advantage.
- **Fixed 50/50 fusion** scores **99.74%** versus the full model **100%**, a negligible **+0.26 pp** difference; exploratory paired bootstrap **[0, +0.78] pp**.
- The learned director's mean archive-fusion gate is **0.948**: it mostly trusts the correctly indexed archive, so this simple environment does not require complex arbitration.

The user-inspired separation of memory and learning works in the narrow measured sense that the old experiences remain accessible; **we have not demonstrated that self-learned executive control adds value beyond engineered associative lookup**.

### All preregistered development criteria: 10/13 pass

Criteria passed (10):
- Full model learns A, B and C to ≥90% (3 pass).
- Retains A and B immediately after C at ≥90% (2 pass).
- Stored A and B are directly retrievable after C at ≥90% (2 pass).
- Beats fast-only on joint frozen performance by ≥10pp (1 pass, +28.39pp).
- Beats context-blind reader by ≥5pp (1 pass, +28.39pp).
- Beats corrupted-content reader by ≥10pp (1 pass, +31.25pp).

Criteria failed (3):
- Learned director versus **fixed 50/50 fusion +5pp**: **FAIL**, +0.26pp.
- Learned director versus **memory-only +5pp**: **FAIL**, 0pp.
- Learned addressing versus **fixed addressing +5pp**: **FAIL**, 0pp.

The machine-readable result accurately states **memory_retrieval_supported=true**, **executive_added_value_screen=false**, **development_screen_pass=false**, **independent_task_replication_complete=false**, **evolution_authorized=false**, **advance_to_evolution=false**.

## Scientific boundaries and next focused diagnosis

This result means an **explicit context-indexed episodic store is sufficient for four deterministic binary patterns**, not that an organism has developed brain-like plasticity.

- World identity A/B/C is available to the learner and to the researcher-defined 43/43/42 archive partitions; this makes choosing the right memory category trivial compared to inferring changing latent contexts from incomplete feedback.
- No new sensory combinations are tested; 100% memory lookup in the same four observations is an expected capacity ceiling, not compositional generalization.
- Archive-only is a **non-neural associative memory**, and its high accuracy does not mean a neural network learned to consolidate synapses without rehearsal. It makes the distinction between stored data and adaptive synapses more concrete.
- Extra memory lookups are expensive relative to fast-only deployment, even though work was artificially matched in these comparisons; costs in forward operations, storage and physical time must be counted in a real organism.
- The learned director is overly redundant here; it mostly trusts memory that already maps familiar cues/action to feedback.
- Memory capacity partitions are fixed by the laboratory; the system has not autonomously invented categories or decided what to discard.
- All results on one 24-seed development cohort; no independent novel-task confirmatory replication.

A scientifically focused next phase, *not launched automatically*, is to **hide the world identity from the reader** and require the director to **infer the current regime from temporal feedback**, with the same A/B/C worlds and a fixed 128-record budget, plus context-label ablations and a known-context upper-bound control. Unlike merely removing context from a static single-decision mapping, the director would need **temporal working memory of recent outcomes** to make hidden-context changes identifiable. A second diagnostic could test whether a new, previously unseen (x,y) cue combination requires the fast learner (outside the scope of this run). Only such tests can show whether the regista contributes something beyond keyed memory.

**Stop decision: MEMORY retrieval solved in this very small known-context school; SPECIALIZED LEARNED EXECUTIVE advantage NOT demonstrated. No ecological evolution, synaptic growth, sealed validation cohort use, or merge to main.**
