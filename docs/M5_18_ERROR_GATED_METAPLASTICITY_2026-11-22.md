# M5.18 — Prediction-error-driven metaplasticity: negative development screen

**Date:** 2026-11-22. **Status:** completed, reproducible development-only experiment; prediction-error-based protection adjustment does NOT solve the stability–plasticity dilemma in the benchmark. Do not proceed to evolutionary selection, more complex tasks or sealed validation.

## Scientific motivation and stop criterion

M5.17 established a robust tradeoff: with the same one-update-per-lived-episode budget and no historical replay, EWC-like fixed synaptic anchor protection retained original rule A but substantially impaired acquisition of rules B and C. Before expanding the ALife organism, M5.18 tests **whether a protection coefficient that relaxes during large or persistent unexpected consequences** can preserve A while still learning B and C. The user explicitly made satisfactory plasticity a prerequisite for progressing. This test is a direct stop/go diagnostic, not a claim that biological plasticity is solved.

## Experimental protocol

New independent **development-only seeds 4100–4123** (24). Same deterministic matched per-seed founder 16-unit tanh action-conditioned reward predictor, with **145 possible learned parameters** and two actions; no structural growth, pruning, new neuron formation, reproduction or evolution. Every condition receives the same experiences/cue order in the explicitly labeled A/B/C classroom, and the learner observes **only whether its chosen action succeeded**, never the privileged correct-action label.

Curriculum A (400 experiences, XOR) → B (400, inverse XOR) → C (400, x) → return A (160) → return B (160), 1,520 lived experiences total. All four possible binary (x,y) pairs are balanced in each 40-step learning block. After each task phase, all previously encountered rules are examined with synapses frozen **before any learning on the next phase**. Exams reshuffle **previously seen cue types**, NOT out-of-distribution examples. Task identities and phase-end consolidation schedules remain researcher-supplied; no spontaneous context discovery or autonomous selection of when to consolidate is claimed.

All variants use the same base learning rate, predictor, phase-end parameter sensitivity estimates, parameter anchors and parameter update rule, except for the chosen metaplastic protection treatment. No-replay conditions use exactly **1,520 parameter-update passes**; replay conditions use exactly **7,600 passes**, one immediate real outcome and four historically sampled experiences per live episode, with the original M5.17 **128-record** explicit task-partitioned memory. Never compare replay vs no replay as a same-compute contrast.

### Intervention: feedback-derived online protection gate

After the chosen action's **actual** scalar reward arrives, before any weight update, compute squared prediction error `(observed_outcome - predicted_success_probability)^2`. Maintain an 0.92-decay fast and 0.995-decay slow exponential moving error estimate. Define positive unexpected elevation `max(0,fast-slow)` and high residual error `max(0,fast-0.08)`; then:

```
pressure = max(0, fast_error - slow_error) + 0.7 * max(0, fast_error - 0.08)
protection_fraction = clip(1 / (1 + 7 * pressure), lower=0.10, upper=1.00)
```

The coefficient multiplies both prior-weight-consolidation gradient stiffness and elastic anchor-to-old-weight penalty. Thus surprising errors temporarily relax the constraints, while lower errors reinforce stability. **No correct target, future event or task-change signal enters the protection gate.** However, consolidation anchors are still deliberately created at known curriculum phase boundaries. This is an experimenter-designed global scalar metaplasticity heuristic, NOT self-discovered localized synaptic control.

### Seven arms (all matched within appropriate update-budget comparison)

1. `online`: ordinary online outcome predictor, no consolidation/replay, one update.
2. `replay`: context-partitioned replay but no consolidation, five updates.
3. `plasticity_only`: fixed full-strength consolidation, no replay, one update.
4. `loose_plasticity_only`: **constant 0.25 strength** consolidation, no replay, one update; tests whether lower protection alone suffices.
5. `adaptive_plasticity_only`: error-derived adaptive protection strength, no replay, one update.
6. `replay_plasticity`: fixed protection plus replay, five updates.
7. `adaptive_plasticity_replay`: adaptive protection plus replay, five updates.

## Verified real execution

[Successful GitHub Actions run #45040634080](https://github.com/kkongiu/neuroevolution-lab/actions/runs/45040634080) ran all five new scientific controls, five original M5.6 regression tests, and **24 × 7 = 168 complete trajectories** of 1,520 lived decisions each. The job archived raw per-seed frozen exam scores, gate metrics, and learning statistics as `m5-18-metaplasticity-development` for 30 days.

Raw-result row digest **SHA-256 `df19580009ea172f641804923cf148046a51a75558e8efc4d2408dc546810857`**.

### Frozen exam mean (%) — 24 separate development seeds

| Condition | A retained after learning B | B first learned | A retained after C | C first learned | Final mean A/B/C |
|---|---:|---:|---:|---:|---:|
| No protection, online only | 38.54 | **87.50** | 52.08 | **97.92** | 83.33 |
| **Fixed protection, no replay** | **97.92** | 71.88 | **98.96** | 72.92 | 77.08 |
| **Constant weak protection (0.25), no replay** | 90.63 | **78.13** | 92.71 | **88.54** | **81.60** |
| **Error-gated protection, no replay** | 96.88 | 71.88 | **98.96** | 71.88 | 76.74 |
| Replay only (five updates) | **100.00** | **100.00** | **100.00** | **100.00** | **100.00** |
| Replay + fixed protection | **100.00** | 90.63 | 100.00 | 94.79 | 95.14 |
| Replay + error-gated protection | **100.00** | 89.58 | 100.00 | 94.79 | 94.79 |

The adaptive no-replay gate reduces protection in practice: its **mean protection factor 0.7719**, its mean lowest factor across seeds **0.2897**, and about **52.10%** of lived experiences had factors below 0.8. This proves the gate is not inert, yet it fails to improve new-skill acquisition over fixed protection: B **71.88% vs 71.88%**, C **71.88% vs 72.92%**. It retains A comparably (**96.88% vs fixed 97.92%**). Its final mean **76.74% vs 77.08% fixed**, a small negative difference.

The deliberately **constant weaker-protection** control performs better at learning C (**88.54%**), retaining A after B (**90.63%**), and the final mean (**81.60%**). This is not a complete solution (B still below 85%), but shows there is a basic regularization-strength tradeoff which the present error-gated heuristic does not optimize.

With replay, replay-only remains strictly stronger in this tiny, explicit-context curriculum: it scores 100% in these exams, whereas combining error-gated consolidation with replay impairs B (89.58%) and C (94.79%). Replay samples and all repeated-variant gradient counts are equal, but the chosen protocol cannot attribute physiological meaning to old synapse protection.

### Predeclared stop/go gate (no post-hoc threshold adjustment)

1. Adaptive no-replay A retained after B ≥85%: **PASS**, 96.88%.
2. Adaptive no-replay A retained after C ≥80%: **PASS**, 98.96%.
3. Adaptive no-replay B learned ≥85%: **FAIL**, 71.88%.
4. Adaptive no-replay C learned ≥85%: **FAIL**, 71.88%.
5. Adaptive no-replay final mean A/B/C ≥85%: **FAIL**, 76.74%.
6. Adaptive B ≥ fixed-protection B +8 percentage points: **FAIL**, 0 pp.
7. Adaptive C ≥ fixed-protection C +8 pp: **FAIL**, −1.04 pp.
8. Adaptive A-after-B no more than 10 pp below fixed: **PASS**, −1.04 pp.
9. Adaptive+replay B no more than 5 pp below replay-only: **FAIL**, −10.42 pp.
10. Adaptive+replay C no more than 5 pp below replay-only: **FAIL**, −5.21 pp.

**3/10 criteria pass.** This is a negative development result, not a near-success. The implemented function explicitly reports `development_screen_pass=false`, `independent_replication_complete=false`, `evolution_authorized=false`, `advance_to_evolution=false`.

## Mechanistic reading and limits

- The single **global** metaplasticity coefficient may relax many weights at once without identifying precisely which old connections can be reused safely. A useful next hypothesis is **synapse-/module-local error and conflict-specific gating**, not simply more aggressive global unfreezing.
- Conversely, it may preserve too much on average (mean ~0.77 vs the 0.25 constant control), but one cannot claim this single fact is the causal explanation without a new dose-matched control. Do not post-hoc tune parameters on these same 4100–4123 seeds and call that independent validation.
- Consolidation uses explicit experimenter-known phase boundaries and task labels. This is a partially programmed school, not an autonomous self-modifying organism.
- Only four input combinations and binary action choice. In such an easy fully observable benchmark, equal-budget replay can already hit ceiling; no reason to infer it is the optimal method in realistic worlds.
- The experiment contains no per-seed confidence intervals nor independent preregistered replication. No new neuron or active-edge infrastructure and no measured energy costs. Progression to ecology/evolution would be scientifically premature.

## Decision: STOP further complexity until plasticity is demonstrated

**Stop adding problem contexts, synapse-growth mechanisms, simulated predators, and evolutionary selection.** Keep the positive but limited M5.16–M5.17 findings, and use M5.18 as a falsification of this specific adaptive scheme.

A new purely *plasticity-centered* protocol could test, on newly allocated development cohorts and before any evolutionary work:
1. Local per-weight or neuron-group plasticity gates based on expected gradient conflict with prior protected competence, rather than a global error threshold.
2. Fixed and truly **dose-matched average-protection controls**, plus counterfactual old-task recall without episodic replay; preserve the equal gradient budget.
3. Consolidation based on internally detected stability/novelty rather than lab-provided stage endpoints, with explicit tests against task-boundary oracle knowledge.
4. An operational stop/go criterion requiring **both** high retention and high initial acquisition of all three skills, across independent fresh tasks, before any study expansion.

No changes to `main` or protected M5.6 cohorts. This PR remains draft and stacked on M5.17.
