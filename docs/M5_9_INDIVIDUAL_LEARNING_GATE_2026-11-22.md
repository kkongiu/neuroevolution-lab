# M5.9c — Individual learner competence gate: immediate feedback and reversal

**Date:** 2026-11-22. **Status:** completed developmental diagnostic, **first evidence of within-lifetime stationary-rule acquisition**, but **adaptation/reversal criteria failed**. No evolutionary parent selection, no reproduction and no physically embodied simulation in this experiment.

## Registered experiment

A native 260-weight M5.6 recurrent learner observes a balanced binary cue, chooses between two actions, receives outcome *only after acting*, and receives positive scalar reward on success and negative on failure. The correct action is the cue XOR a hidden binary rule. For ticks 1–160 the rule is stable; at tick 161 it flips **without signaling the change**. The network is not reset. Each of the 24 dedicated development seeds 3200–3223 is evaluated in three paired arms with exactly identical cue schedules and initial genotypes: `plastic` (synaptic policy learning), `frozen` (no synaptic updates), `shuffled_feedback` (synaptic updates but a random, noncontingent reward signal). Real previous action correctness remains part of the observation in all three arms; only the reward channel is changed in the shuffled control. All cues are balanced within each 40-trial scoring window. **No answer labels** and no future-oracle signals are provided.

Predeclared screening criteria:
- Plastic initial learning gain at least +10 pp greater than frozen gain.
- Following the rule reversal, last-window minus first-window accuracy at least +10 pp.
- Plastic final reversal accuracy at least 65%.
- Plastic final reversal at least +10 pp over frozen on matched seeds.
- Plastic final reversal at least +10 pp over shuffled reward on matched seeds.

These are pragmatic development thresholds, NOT a registered inferential significance test, and passing them would require independent confirmation.

## Verified results

[GitHub Actions #45027993429](https://github.com/kkongiu/neuroevolution-lab/actions/runs/45027993429): four new test cases and four prior delayed-reward regression cases passed; all 24 × 3 = **72 individual trajectories** completed; full JSON report archived under artifact `m5-9c-individual-learning-prerequisite`. SHA256 of all per-seed results: `05ffa01f7eaf0f9ebcc0493c6450bd72609a8cd6645dc12417e49b1d2ad3daa0`.

| Condition | Stable rule, first 40 | Stable rule, last 40 | Reversed rule, first 40 | Reversed rule, last 40 | Mean synaptic L2 drift |
| --- | ---: | ---: | ---: | ---: | ---: |
| Plastic learner | 49.38% | **61.88%** | 37.19% | **46.35%** | 0.327 |
| Frozen learner | 48.33% | 47.50% | 51.88% | **49.38%** | 0 |
| Randomized reward | 48.23% | 46.98% | 51.15% | **49.38%** | 0.334 |

Initial stable-rule learning gain **+12.5 pp** in plastic versus **−0.83 pp** frozen (and −1.25 pp shuffled). This is the first clearly measurable exploratory signal that contingent feedback can induce useful changes in a single individual on a stationary simple rule.

On reversal the plastic learner drops to 37.19% and recovers to only 46.35%; mean within-reversal recovery +9.17 pp but still below the original chance 50% and well below frozen/shuffled final accuracy 49.38%. Paired plastic minus frozen final reversal accuracy **−3.02 pp**, same as plastic minus randomized-reward; the network has not demonstrated stable rule adaptation. The model received 320 actual synaptic updates per test organism on average; mere weight changes are not sufficient proof of competent adaptation.

**Gate: 1 of 5 criteria passed. DO NOT advance to multistep delayed-reward curricula.**

## Limitations

- Cue XOR rule with balanced binary cues is an engineered simple conditioning task, not general reasoning or an embodied physical problem.
- The trained recurrent controller receives a previous correctness signal, which may serve as explicit short-term feedback without revealing the correct current action.
- The 24 seeds, comparison and thresholds are development diagnostics. Confidence intervals, independent new family, and an explicitly frozen test protocol remain absent; avoid confirmatory claims.
- There is no inheritance, reproduction or selection in this run; M5.9c purposely blocks evolutionary selection until competence is demonstrated.
- Current reward and one-step gradient may induce inertia, making extinction of an old habit after reversal slow. Improving this requires a separately declared development experiment and controls, not retuning this same 24-seed result and labeling it validation.

## Next step

Develop a **separate reversal-learning intervention** on disjoint development seeds: change-point response/reversible-learning adaptation (e.g. bounded reward-driven plasticity, memory reset on evidence of drift) versus unchanged baseline and placebo resets; same observation and reward budget. Pre-register new-world criteria prior to running. Continue to keep all original M5.6 source/final/replication worlds sealed and `main` unchanged.
