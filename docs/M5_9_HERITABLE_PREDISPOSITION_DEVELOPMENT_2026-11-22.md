# M5.9 — Heritable neural predisposition without inheriting lifetime memories

**Date:** 2026-11-22. **Status:** fully executed **minimal abstract proof of concept**, no compelling evidence yet for evolved learning speed or general problem-solving. This is **not physically integrated into M5.8 ecology**, no energy conservation or predator simulation in the puzzle arena.

## What is genuinely implemented

- Genotype is the **260-dimensional neural weight initialization** (native M5.6 Learner). Each individual starts with `Learner.fresh(genotype)`; during-life REINFORCE updates only the individual's private weights.
- Offspring inherits a Gaussian-mutated copy of the **original genetic initialization**, not its parent's during-life learned numerical weights, recurring hidden state, visit history or learned model predictor. Tests artificially modify the parent's learned state and verify complete fresh reset for the child.
- Binary feedback-dependent tasks: the organism sees a cue and phase bit, chooses one of two actions, and obtains reward only after its own *actual* solution; the hidden contextual mapping is never included in sensory inputs. Identical observations can correspond to opposite target choices in different episodes, so remembering past rewards is needed in principle.
- A problem episode counts as solved after **at least seven correct choices in ten attempts**, fixed prior to running; this criterion is noisy relative to the 50% guessing baseline (pure chance P[>=7/10] = 17.1875%). Selection is based on solved episodes, with **one baseline weight + each solved episode**, not an automatic biological birth or explicit fitness oracle.
- Exactly ten offspring slots per generation in each of three matched arms:
  1. `solution_plastic`: offspring-parent probability increases with actual solved episodes; synapses learn within life;
  2. `solution_frozen`: same solution-driven parent selection, but all in-life synapses are frozen;
  3. `random_parent_plastic`: identical parent-slot budget, but parents chosen with equal probability independent of episode success; in-life synapses remain plastic.
- Twelve evolutionary lineage seeds **3000–3011**, all initially identical across arms, ten generations each, ten organisms/population, four changing-rule training problems per evaluated generation, ten attempts each.
- Twelve separately generated novel task-world seeds **3050–3061**, five fresh episodes, evaluated after evolution. There is no reuse of previously precomputed M5.6 holdout/replication worlds. However, these novel tasks come from the same simple cue/context rule family, NOT a fundamentally unseen problem family.

## Real executed results

[Successful GitHub Actions run #45022863938](https://github.com/kkongiu/neuroevolution-lab/actions/runs/45022863938) ran five new tests, five old policy regression tests, 36 independent evolving lineages (12 × 3), and archived full ten-generation ancestor/descendant statistics as `m5-9-genetic-learning-predisposition-development` (30-day retention). SHA256 of canonical runs: `e31f48668e1bbe974a8cf2b38b164277b2593e7e6d43efec5f2134c23d6849e4`.

| Arm | Mean novel-task correct responses | Mean novel episodes solved / organism (out of 5) | Accuracy on novel tasks with lifelong learning disabled | Mean late - early accuracy during new tests |
|---|---:|---:|---:|---:|
| Random parent, plastic | 50.33% | 1.017 | 49.73% | **−5.13 pp** |
| Solvers reproduce, frozen | 50.17% | 0.925 | 50.17% | **−2.27 pp** |
| Solvers reproduce, plastic | **51.92%** | **1.075** | **51.52%** | **−2.17 pp** |

These small effects are near random-choice accuracy of 50%. The apparent advantage of the problem-solver plastic arm over random parent plastic is **+1.59 pp**, potentially noise. The increase from switching lifelong plasticity off in final tests is **only +0.40 pp** for the solution-plastic selected offspring, far too little to claim that the evolved genotype improved learning capability. Late accuracy is *worse* than early accuracy in all arms: **no evidence of accelerated within-episode learning** on these tasks. Evaluating the same genetic descendants with plasticity disabled changes their within-test weights by design and must not be conflated with the hereditary frozen-selection arm.

## Critical limitations and honest interpretation

- The inherited neural weights can encode **prior biases and useful fixed strategies**, not necessarily general `how-to-learn` ability. There is no independently evolved learning-rate/metaplasticity parameter yet.
- No inherited in-life memories is guaranteed in code, but *genetic* predispositions can contain evolutionarily accumulated associations with the training task family. True skill generalization must use altered task structures and counterbalanced rule distributions.
- This experiment is **abstract cognitive tasks**, not a world physically populated by evolving problem-solving organisms. The ten births per generation are externally imposed **matched selection slots** and are not yet tied to parental body energy costs.
- Chance success at the per-episode threshold is not rare. Merely breeding on the number of seven-of-ten episode successes can unintentionally breed random favorable policies.
- **No positive scientific claim is supported** beyond functional memory/genotype separation and a reproducible mechanism for outcome-sensitive reproductive selection. A 1.6-point gap across these 12 replicate lines should not be interpreted as proven cognitive evolution.

## Proposed M5.9 next experiment

1. Introduce multi-step physical problem tasks with delayed consequences and stronger chance-adjusted completion thresholds, or explicitly meter per-individual improvement within a task before allowing extra reproduction.
2. Add **metaplastic genotype parameters** governing learning sensitivity, bounded learning rate, or model-use thresholds without copying personal experience across births.
3. Include a **reward-yoked reproduction control** with the same number and timing of children but parent identities shuffled, and assess independently whether learning accelerates on held-aside task families.
4. When competence signal is observed in simple tasks, integrate physical energy-constrained reproduction into M5.8 with preserved ecology ledgers.
5. Maintain M5.6 sealed source/final/replication worlds unused. `main` unchanged; changes live in stacked M5.9 draft PR.
