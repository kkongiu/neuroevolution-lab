# M5.19 — Gradient conflict constraints and coordinate-local plasticity: negative stop/go

**Date:** 2026-11-22. **Status:** completed development-only diagnostic. **1/8 preregistered screening gates pass.** The proposed local-gating method fails to retain old skills and learn new ones simultaneously. Evolution, ecology extension and new curricula must remain blocked.

## Scientific motivation

M5.17 and M5.18 demonstrated a stability–plasticity dilemma: freezing salient weights helps remember A but blocks learning B/C, whereas unrestricted learning acquires B/C while forgetting A. The global prediction-error-based metaplasticity gate tested in M5.18 failed (3/10 developmental criteria). M5.19 tests two next hypotheses: (1) old-task reference gradients can restrict *only harmful directions* of a new gradient, and (2) local coordinate gates can selectively suppress interfering synapses without freezing the whole network.

The work is explicitly plasticity-centered: **no new contexts, no synapse formation/pruning, no evolution and no genetic reproduction**.

## Exact design and independent cohort

- **Fresh development-only seeds 4200–4223, 24 independent founders**, disjoint from all M5.9–M5.18 development and the protected original M5.6 SOURCE/HOLDOUT/REPLICATION cohorts. No sealed access.
- Identical matched per-seed **16-tanh-hidden-unit, 145-parameter action-conditioned reward predictors** in every arm, with same initialization and learning rate 0.067.
- All five treatments receive exactly **1,520 lived environment interactions** and exactly **1,520 parameter-weight updates**, using the same externally specified cue schedule and task order: A=400 (XOR), B=400 (inverse XOR), C=400 (copy x), return A=160, return B=160. Each 40-trial block balances all four (x,y) pairs. Cues and task-context identity remain researcher-specified. Policies make different choices as they learn, so their *actual reward trajectories* are not identical.
- All treatments allocate the **same total 128-example memory capacity** partitioned explicitly as **43 A / 43 B / 42 C**. Only actually experienced (x,y,context,chosen_action,success) records can be stored; no teacher-correct actions, future facts or unchosen-action rewards enter the policy.
- **All five arms** compute historical reference gradients, even if the chosen control ignores them, yielding **7,360 individual historical gradient evaluations and 1,840 context-mean reference vectors per arm per seed**. This is an intentionally *equal-work* diagnostic, not an efficient deployment. Each past context supplies up to one most-recent actually experienced record per each of four familiar cue patterns; thus reference samples are bounded but laboratory-context-indexed.
- Crucially, **no arm performs additional direct weight training on its historic examples**. Old experiences affect weight learning only as reference constraints in projection arms. The experiment is therefore **memory-assisted learning, NOT memory-free plasticity**.
- Frozen old-task examinations follow every phase, before any return-to-old-task training. They use new shuffles of the same four cue patterns and cannot modify parameters, old experiences or action RNG; thus no truly unfamiliar contexts or open-world generalization is measured.
- All hyperparameters and evaluation criteria were fixed before evaluating cohort 4200–4223, and no development results are subsequently relabeled as independent replication.

## Five treatments

1. **Online:** one gradient step from the reward of the chosen action, no consolidation or projection.
2. **Fixed protection:** M5.17/M5.18-style salience/anchor protection at full strength, adding phase-boundary parameter importance; same single gradient update per lived experience.
3. **Weak constant protection:** identical model and consolidation but stiffness/anchor contribution scaled by constant 0.25 (comparison to rule out mere relaxation).
4. **Gradient projection:** reference gradient for each previously observed context is averaged over up to four actual past experienced action/outcome records. If an incoming new-task *ascent* update is antiparallel (negative dot product) to a reference gradient, remove that component by orthogonal projection. Three passes handle multiple old-context constraints. This is a **GEM-style first-order heuristic**. Nonlinearity/finite-step updates mean avoiding negative gradient dot products does **not guarantee no measured forgetting**.
5. **Projection + local gating:** in addition to the same projection, multiply only coordinates with a negative per-parameter alignment to any previous-context reference gradient by 0.25. Other parameters retain full update scale. The local gate is a **hard-coded conflict heuristic**, not learned self-organized synaptic plasticity.

Only fixed/weak consolidation has externally triggered anchor formation at known curriculum ends. All arms exploit known sensory task identities and researcher-defined memory partitions. No extrapolation to hidden contexts is justified.

## Verified actual GitHub Actions execution

[Workflow run #45046051182](https://github.com/kkongiu/neuroevolution-lab/actions/runs/45046051182) **SUCCESS**:
- Five M5.19 scientific-invariant unit tests pass, including no-label leakage, reproducible frozen probes, same founder weights, gradient sign/projection check, 128-record limit, true per-weight conflict suppression and original sealed cohort firewall.
- Five original M5.6 neural-controller regression tests pass.
- **24 seeds × 5 treatments = 120 complete experimental trajectories**, 1,520 lived decisions per trajectory, full raw per-seed results archived as `m5-19-gradient-conflict-plasticity-development` (30 days).
- Canonical raw-row SHA-256: `fea3687519eaa7c6890327c1a7ce74684462f0bb96fc1c6dc6ca0b819ff195dd`.

### Main frozen exam results (means across 24 seeds)

| Treatment | Initial A mastered | A retained immediately after B | **A retained immediately after C** | **B mastered immediately after B** | B retained immediately after C | **C mastered immediately after C** | Mean final A,B,C after revisits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Online only | 100.00% | 48.96% | 56.25% | 90.63% | 57.29% | 97.92% | 79.17% |
| Full fixed protection | 100.00% | **100.00%** | **100.00%** | 72.92% | 66.67% | 80.21% | 80.90% |
| Constant weak protection (0.25) | 100.00% | 87.50% | 96.88% | 82.29% | 64.58% | 94.79% | 85.76% |
| **Historical gradient projection** | 100.00% | 78.13% | 76.04% | **92.71%** | 66.67% | **97.92%** | **95.49%** |
| Projection + coordinate-local gating | 100.00% | 71.88% | 73.96% | 79.17% | 70.83% | 95.83% | 89.24% |

Numbers are development cohort estimates, not confirmed independent replication. New/sealed test worlds were not consulted.

### Interpretation

- **Projection helps acquisition but does not preserve enough of the old mapping.** Compared with fixed full protection it raises first-time B from **72.92% to 92.71%** and first-time C from **80.21% to 97.92%**, but drops frozen A after C from **100% to 76.04%**. After returning to A/B it finishes with a high 95.49% average, which is **not equivalent to avoiding forgetting before refresher experiences**.
- **Local suppression hurts the proposed solution.** Local+projection scores only **79.17%** on first-time B and **73.96%** on frozen A after C. The mean number of locally negatively aligned coordinate opportunities suppressed per full run was **52,179.5**, but these per-parameter sign conflicts may reflect noise rather than harmful changes. Blindly suppressing them likely removes legitimate new-learning degrees of freedom. This is a hypothesis, not a proven causal explanation.
- **Weak constant protection wins the proposed simple tradeoff score** (average A frozen after C, B first learned, C first learned): **91.32%** vs **88.89%** projection and **82.99%** local+projection, though its B acquisition remains below the 90% criterion and B recall after C remains weak.
- **Context B is significantly forgotten by all non-replay treatments after C**, ranging from approximately **57%–71%**. The proposed method has not solved multi-task continual learning, even if old A returns to high performance after re-training.

### Strict predeclared development stop/go

The experimentally proposed local method must meet the following conditions:
1. A after B ≥90%: **FAIL** 71.88%.
2. A after C ≥90%: **FAIL** 73.96%.
3. B first learned ≥90%: **FAIL** 79.17%.
4. C first learned ≥90%: **PASS** 95.83%.
5. B retained after C ≥90%: **FAIL** 70.83%.
6. Final mean A/B/C ≥90%: **FAIL** 89.24%.
7. Local-projection tradeoff vs weak fixed protection ≥+5 percentage points: **FAIL**, approx **−8.33 pp**.
8. Standard projection tradeoff vs weak fixed protection ≥+5 percentage points: **FAIL**, approx **−2.43 pp**.

**1/8 passed.** Machine states: `development_screen_pass=false`, `independent_replication_complete=false`, `evolution_authorized=false`, `advance_to_evolution=false`.

## Methodological limitations

- The constraint reference gradients are computed from old experienced outcomes at *current weights* and averaged per context. They are approximate local directions; near a learned optimum a reference gradient can shrink or cancel, losing information about the consequences of larger parameter changes. Thus a first-order negative dot-product criterion can fail to protect old knowledge.
- Limited four-cue deterministic environments, explicit visible task labels, researcher-defined 43/43/42 episodic partitions, and external phase-end consolidation for fixed/weak controls cannot establish open-ended self-directed plasticity.
- For strict fairness, all arms compute the same quantity of reference gradients, including controls, but this overstates compute required by simpler deployments. One stored observation per x/y pattern gives a weak summary of many different prior actions, and may be biased by whichever actions the agent recently preferred.
- The local arm's 0.25 multiplier is not learned from experience. Suppressing every coordinate whose gradient conflicts with a historical mean does **not necessarily identify harmful synapses** and can reduce B learning.
- Loss-gradient projection protects a first-order surrogate of old-task negative log-likelihood, not actual frozen test accuracy or true biological memory.
- There are no paired confidence intervals, error bars, multiple new task families, or independent holdout results yet. A positive result would still require replication before an evolutionary advance.

## Research-only proposal for the next step (not yet implemented)

Do **not** tune M5.19 hyperparameters on the same seed cohort and claim success. Consider only a focused non-expanding M5.20 with a fresh cohort and advance-blocked status:

1. **Empirical old-loss trust region:** construct a proposed update from the new real outcome, then verify its effect on old **actually experienced** examples *before committing* the update. Backtrack/modify only if held old losses increase beyond an explicit per-example or aggregate tolerance. Avoid using teacher-correct labels or the frozen examination observations to guide training.
2. **Separate old-loss preservation and learning progress:** control both the new-task immediate outcome and old-task loss variation; don't let a stationary old-gradient close to zero misleadingly indicate memory is safe. Count extra forward passes and candidate-update proposals for all arms; memory access incurs real compute costs.
3. **Distribution shifts and hidden context only after a robust within-task local mechanism is established**. No automatic new curricula/evolution.
4. **Only a fresh independent replication** with strong retention **and** B/C mastery can unblock the stop criterion.

**main, genetic evolution and original M5.6 sealed worlds remain unchanged.**
