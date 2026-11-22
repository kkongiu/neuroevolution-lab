# M5.20 — Tabular meta-reinforcement learning for selective plasticity

**Date:** 2026-11-22. **Status:** completed DEV-only trial, **negative**: one of nine prespecified stop/go criteria passes. **Do NOT expand task complexity, trigger evolutionary selection, merge to main, or access original sealed M5.6 worlds.**

## Research question

Can a tiny, explicitly feedback-trained reinforcement controller select the strength of synaptic consolidation at different neuron groups so that a lower-level predictive learner retains older competence **and** masters new contexts? Earlier M5.17–M5.19 demonstrated that fixed elastic weight consolidation protects old A but prevents learning B/C, whereas free updates learn new material but forget old material. This experiment tests **meta-RL regulating the learning process itself**, not RL directly selecting environment actions.

## Model, experiences and explicit limitations

- New development cohort **4300–4323** (24 independent seeds); disjoint from earlier DEV cohorts 3300–4223 and the original sealed M5.6 source, holdout and replication sets.
- Same per-seed initial lower-level **16-hidden-unit nonlinear action-conditioned reward predictor**, with 145 weight parameters, fixed graph and no synaptic birth/pruning. The predictor sees two binary cues plus an explicitly supplied context identity (A/B/C) and evaluates its own two possible actions. Environment-only desired action is never supplied to it as a label.
- Same A (400 actual interactions, XOR) → B (400, inverse XOR) → C (400, copy x) → return A (160) → return B (160) curriculum. **1520 lived actions, exactly one neural weight update per action**, for every arm. All four binary cue pairs are presented equally within each 40-trial block, and exam observations are reshuffles of the same four patterns, **not unseen-input generalization**.
- Frozen 80-trial exams after each curriculum phase on each previously encountered rule, with **no update to weights, memory, controller, or selection RNG**. Critically, tests of A/B immediately after new tasks occur **before refresher training**.
- All arms store up to exactly **128 total personal experienced action-feedback records**, externally partitioned into 43 A, 43 B and 42 C entries. All arms evaluate exactly the same number of read-only historical reward-prediction reference observations, even controls that do not use the resulting score for their protection policy. The reference forward-pass count is **3940 per seed/arm**. There is no historical replay-based *synaptic weight* training. This is **memory-assisted metaplasticity**, not an autonomous memory-free controller.
- Consolidation is triggered at experimenter-known curriculum boundaries for **all conditions**; that is not autonomous initiation of consolidation. The four groups are fixed groups of four hidden units each, not emergent neural modules.

## Two-level learning implementation

The **lower-level brain** makes action choices and updates weights from actual success/failure of its chosen action. At the end of each **40 lived decisions**, a separate **tabular Q-learning meta-controller** chooses how strongly each of its four neuron groups is allowed to change during the following block. Its action set of elastic consolidation factors is:

```
STRENGTHS = (0.10, 0.30, 0.60, 1.00)
```

A controller state contains **only** two bits computed from the agent's lived experience: recent successful action fraction below 80% and old-experience prediction Brier loss above 0.15. It does not receive the current task identity or future outcomes. Each of the four groups independently chooses an action using epsilon-greedy exploration and its own Q-table row, with **discounted bootstrap**:

```
Q[group, state, action] += 0.34 * (
   meta_reward + 0.70 * max_a Q[group, next_state, a]
   - Q[group, state, action]
)
```

The **meta-reward is engineered by the researcher**, not a biological homeostatic signal. Each 40-step block provides:

```
R_meta = 0.8 * (current_success_rate - 0.5)
       + 0.5 * (current_success_rate - previous_success_rate)
       - 1.2 * max(0, old_Brier_after - old_Brier_before)
```

Brier error is computed ONLY on the organism's own historically observed action/result records from other contexts, using the lower-level model's predicted success probabilities. It never uses exam labels, teacher-correct actions or the unchosen action's reward. Historical observations remain unchanged by these read-only reference checks.

**Important control:** Q values are **not updated during the first A phase**, because no old-task consolidation anchors exist yet, so an action of the plasticity regulator has no physical consequence. Only the subsequent 28 forty-action blocks permit Q updates. The learned controller therefore receives **112 actual Q-value updates per life** (28 effective blocks × four groups). This is a small sample for a nonstationary RL control problem; performance should not be interpreted as an upper bound on the usefulness of meta-RL generally.

## Seven arms

1. `online`: zero old-weight protection, no meta-policy learning.
2. `fixed_strong`: old-weight protection factor 1.0 in all four groups, never learned.
3. `fixed_weak`: old-weight protection factor 0.30 in all four groups, never learned.
4. `random_controller`: one uniformly random one-of-four protection strength per group each block, no Q learning.
5. `heuristic_controller`: manually selected factor based on the same observable recent-success and old-error summary, no Q learning.
6. `q_learning_controller`: tabular Q-learning receiving only the meta-reward formula above.
7. `scrambled_reward_q`: same Q architecture, learning updates, decision count and exploration schedule, but Q receives an independent uniform(-1,1) teaching signal instead of true meta-reward. This **independently random signal is not an identical-marginal shuffle**; call it a corrupted-meta-reward control, not strict shuffled reward.

All variants run the same lower-level training update count and old-experience forward evaluations. Q-based arms perform an **additional 112 small Q-table parameter updates per lifetime**; actual extra computation is not claimed to be exactly the same as nonlearning arms. Each group receives the same global meta-reward, which leaves the per-group credit-assignment problem unresolved.

## Real verified experiment

**[Successful GitHub Actions run #45047289806](https://github.com/kkongiu/neuroevolution-lab/actions/runs/45047289806).** All five new scientific consistency tests and five original M5.6 regression tests passed. **24 fresh development seeds × seven treatments = 168 full training trajectories**. Action-level data not archived, but all per-seed frozen exam metrics, controller update counts, memory/compute ledger and Q/checksum metrics are stored in the JSON Actions artifact `m5-20-rl-metaplasticity-development` (30-day retention).

Canonical rows SHA256: `e53527d1ee22aa2db047f117e7a97d14fb3a4cc0fd8d466769ac8c8ad76524ec`.

### Frozen examination accuracies (%) — 24 DEV seeds

| Arm | A after initial A | A after B | **A after C** | **B first learned** | **B after C** | **C first learned** | Final A/B/C mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| Online only (zero protection) | 95.83 | 40.63 | 50.00 | **93.75** | 59.38 | **100.00** | 81.94 |
| Constant strong protection 1.00 | 95.83 | **96.88** | **96.88** | 76.04 | 59.38 | 80.21 | 76.74 |
| Constant weak protection 0.30 | 95.83 | 88.54 | 92.71 | 81.25 | 71.88 | 90.63 | 83.68 |
| Random regulator (four choices) | 95.83 | **90.63** | **96.88** | 78.13 | 68.75 | 91.67 | **84.03** |
| Handcrafted heuristic regulator | 95.83 | 77.08 | 93.75 | 83.33 | 70.83 | 92.71 | 79.51 |
| **Q-learning meta-controller** | 95.83 | 82.29 | 88.54 | 84.38 | 68.75 | 93.75 | 82.99 |
| Q-learning with corrupted meta-reward | 95.83 | 88.54 | 93.75 | 82.29 | 68.75 | 91.67 | 83.33 |

All scores are per-trajectory frozen exam means over 80 balanced questions, averaged across the 24 seeds. Independent cohorts were not used to replicate the protocol.

### Central result: the meta-controller does not learn a superior regulation policy

The Q-learning regulator makes real decisions and performs real bootstrapped Q updates, but *it does not beat* weak protection, a random controller, or corrupted-meta-reward Q learning on the combined *pre-refresher* stability/acquisition score:

| Arm | Mean of A-after-C, B-first, B-after-C, C-first |
|---|---:|
| Fixed weak 0.30 | 84.11% |
| Random regulator | 83.85% |
| **Q-learning regulator** | **83.85%** |
| Corrupted-meta-reward Q | 84.11% |

Its final A/B/C mean after the two relearning periods is **82.99%**. Most importantly, it retains B at only **68.75% after C**, failing to solve continual learning.

There is **no evidence** that the learned Q policy extracts useful reward-dependent meta-learning above random choice in this small model. It might be undertrained, the engineered reward might misalign with long-term retention, or the four independent groups might have poor group-specific credit assignment; these explanations are **hypotheses, not demonstrated causes**.

## Why this experiment is not a conclusive evaluation of meta-RL in general

- The meta-agent receives just **28 causally meaningful control feedback opportunities**, then only **112 shared-reward group Q updates**. Multiple group choices are combined but rewarded by the same aggregate outcome, making which group caused benefit ambiguous.
- The meta-reward penalizes the **increase in old loss within the most recent 40-action block**, not **persistently elevated loss on already-forgotten tasks**. A large one-time forgetting event could be penalized just once. This could incentivize policies that do not recover previously forgotten skills.
- The task identity is observable and consolidation phase boundaries are externally injected. A potentially powerful agent-learning-context mechanism is not tested.
- In the first 400 A experiences, the protection action has no effect, so active Q learning rightly stays disabled; there is no meta-training transferred across lives.
- The current lower-level world is a tiny, four-input-pattern binary-action deterministic school. This does not demonstrate transferable general intelligence.
- The Q policy's state is only two threshold bits; more nuanced group-specific conflict or uncertainty signals might be necessary for causal policy selection.
- The negative-control meta reward distribution is different from the real meta reward; a strict marginal-matched permutation would further strengthen causal attribution.
- No independent replication, no genetic optimization and no statistical confidence intervals. Even a positive run would be development screening only.

## Predeclared plasticity stop/go — 1/9 passed

1. Q A retained after B ≥90%: **FAIL**, 82.29%.
2. Q A retained after C ≥90%: **FAIL**, 88.54%.
3. Q B first acquired ≥90%: **FAIL**, 84.38%.
4. Q B retained after C ≥90%: **FAIL**, 68.75%.
5. Q C first acquired ≥90%: **PASS**, 93.75%.
6. Q final mean ≥90%: **FAIL**, 82.99%.
7. Q composite tradeoff ≥ constant weak +5pp: **FAIL**, −0.26 pp.
8. Q composite tradeoff ≥ random regulation +5pp: **FAIL**, 0.00 pp.
9. Q composite tradeoff ≥ scrambled-meta-reward Q +5pp: **FAIL**, −0.26 pp.

**1/9 development gates pass.** Code returns `development_screen_pass=false`, `independent_replication_complete=false`, `evolution_authorized=false`, `advance_to_evolution=false`. *The inability to meet the plasticity stop gate means no ecological or evolutionary expansion.*

## Targeted research direction, without launching a new experiment yet

If further work is authorized, a better meta-RL investigation should **separate the question of learnability of the meta-policy from the need for stable lower-level learning**:

1. **Meta-train across many independent simulated lifetimes** on only the same small curriculum types, learning the regulation policy across resets while never carrying over a lower-level organism's own experiences. Count all meta-training samples, hold out new seed families, and compare to a frozen/naively copied policy. Passing only the within-lifetime benchmark isn't enough.
2. Use an auxiliary old-task **absolute loss** and old-task memory-retention cost, not only the recent old-loss increase. Never expose sealed frozen exam scores or correct-action targets to the learner.
3. Credit assignment should be **group-specific or counterfactual** (one group changed at a time in diagnostics), because one global reward cannot cleanly identify which synapse group caused improvement.
4. Compare against both **constant weak strength and marginal-matched shuffled meta-reward**, and budget old-data references and Q updates.
5. Continue with truly new task domains only **after** both retention and new-task learning pass independent screening. Continue to block evolution and sealed original cohorts.

No changes to `main` or original sealed validation seeds.
