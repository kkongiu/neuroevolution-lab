# M5.15 — The Predictive Classroom: outcome learning before structural growth

**2026-11-22. Status: completed development experiment.** This establishes that a small nonlinear, action-conditioned outcome predictor can reliably solve a *within-distribution*, fully observable two-bit XOR and a three-action classroom bandit given sufficient actual action feedback. It does **not** establish unprompted insight, general compositional transfer, ecological survival, or a unique benefit of episodic replay. No evolution promoted.

## Motivation

In M5.14, 160 reward-only trials with the original native REINFORCE policy failed to teach the standalone XOR operator (roughly 48% in the final window). Therefore a compositional C task was not a meaningful test of whether synaptic growth enables generalized knowledge. M5.15 separates a simpler prerequisite: can an individual learn to predict the consequences of its own actions in a controlled school?

## School and examination protocol

- **New development seeds 3800–3823** (24, disjoint from all previous M5.9–M5.14 development and original sealed M5.6 cohorts).
- **Two different tasks, each trained as a separate brain**; there is no transfer between them:
  - **XOR:** two binary sensory cues `x,y`, two possible actions, realized outcome = 1 iff chosen action equals `x XOR y`; chance ~50%.
  - **TERNARY:** same two cues, **three actions**; realized outcome = 1 iff chosen action equals `x+y` (0,1,2). A failure does not say which of the remaining two actions would succeed. Chance ~33.3%.
- Per organism and task: **1,200 actual environment interactions**, then **120 exam decisions** with network parameters frozen and no learning updates. Every 40 trials, cue pairs (00,01,10,11) each appear ten times; the exam reshuffles those same pairs. **No unseen state pairs, unseen task structures or adversarial shift** are tested.
- Action feedback is **only the success/failure of the action the organism chose**, not a desired-action label. Environment-only `target()` computes correctness; the learner receives only the observed scalar consequence.
- Predictor uses a **16-unit tanh hidden-layer** action-conditioned reward predictor `P(reward=1|x,y,a)`, trained by gradient updates on *experienced action outcomes*. Learner selects mostly the predicted best action with limited independent exploration that decreases over training. The sole memory available for replay is a bounded FIFO of **128 actually experienced (state, action, outcome) records**; no preloaded or oracle examples.
- The native baseline is the original **8-hidden-neuron/260-weight M5.6 RNN REINFORCE learner**, trained with 1,200 outcomes and greedily evaluated from the frozen policy's softmax preference. Because predictor and native RNN have **different architectures and optimizer/update rules**, any performance gap against native cannot be attributed to the learning objective alone.

## Five development arms

1. `native_reinforce`: original RNN policy-gradient learning, historical benchmark, 1,200 parameter updates.
2. `predict_online`: predictor updates from **only the newest real experience**, 1,200 parameter updates.
3. `predict_repeat`: predictor receives five repeated updates of the most recent real experience, **6,000** updates; same compute-update count as replay, no historical experience sampling.
4. `predict_replay`: one update from current experience plus four sampled past experiences, buffer capped at 128, **6,000** updates; *exact same initial predictor weights as repeat* for every seed and task.
5. `predict_uncoupled`: same replay mechanism and updates as replay but receives **an independent Bernoulli(0.5)** outcome instead of the true outcome. The independent-noise negative control does not perfectly match the true reward marginal distribution and should not be called a strict shuffled-reward control.

New actions and test targets are never passed into `PredictiveLearner` except the chosen action and its realized 0/1 feedback. The exams do not call `experience()` and exact parameter equality is enforced before/after exam.

## Actual executed results

**[Successful GitHub Actions #45035821172](https://github.com/kkongiu/neuroevolution-lab/actions/runs/45035821172).** Four new unit-test cases and five original controller regression tests pass. **24 development seeds × 2 school tasks × 5 arms = 240 trajectories**, each 1,200 training + 120 frozen exam decisions. Raw per-organism curves plus frozen exam results archived in artifact `m5-15-predictive-classroom-development`, 30-day retention.

Canonical JSON raw rows SHA256 `65a02a0cf67c272739437207ac1d9e799b668932bdd792e6e9c648db71a9f2e3`.

| Arm | XOR frozen exam | XOR final 120 training | Ternary frozen exam | Ternary final 120 training | Updates |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native REINFORCE | 75.56% | 72.05% | 67.74% | 66.46% | 1,200 |
| Online prediction, 1 update | **100%** | 96.49% | 93.75% | 88.65% | 1,200 |
| Repeat current experience 5× | **100%** | 96.49% | **100%** | 94.97% | 6,000 |
| Replay past experiences 5 updates | **100%** | 96.25% | **100%** | 95.42% | 6,000 |
| Uncoupled false feedback | 47.92% | 49.72% | 32.29% | 32.40% | 6,000 |

All averages over 24 seeds, frozen 120-decision examinations. Not statistically corrected confirmatory tests. XOR chance is 50% and ternary chance is 33.3%; the uncoupled group is near chance.

### Interpretation

- Both predictive learning and repeated/replayed updates are able to learn XOR on the **four experienced input combinations**; a separate three-action task also reaches 100% with both repeat and replay. The native policy improves with longer exposure compared with M5.14, from its previous approximately 48% after only 160 XOR steps to 75.56% frozen exam after 1,200; do NOT attribute the entire previous failure to network architecture or solely to REINFORCE.
- Predictor online reaches 100% XOR and 93.75% ternary with the **same number of update passes** as native, but the model architecture, optimization and exploration differ, so a mechanistic attribution to the reward-prediction objective is **not isolated**.
- **Replay does not beat the equal-update repeat control**: both achieve exactly 100% on each frozen exam. This is a strong practical warning against claiming memory is necessary for these stationary elementary tasks.
- The control receiving independent reward noise performs near chance, supporting that learning is due to real action-outcome correlations rather than direct target-label leakage. It is not equivalent to a perfectly marginal-matched permutation.
- This is closer to learning a four-entry rewarded state-action mapping than to genuinely extrapolating an abstract XOR relation to unseen symbolic domains. No compositional generalization claim is warranted.

## Development gate (set before run)

| Criterion | Result |
| --- | --- |
| Replay XOR exam >= 85% | **PASS**: 100% |
| Replay ternary exam >= 75% | **PASS**: 100% |
| Replay XOR >= native +10 pp | **PASS**: +24.44 pp |
| Replay ternary >= native +10 pp | **PASS**: +32.26 pp |
| Replay XOR >= equal-update repeated current +4 pp | **FAIL**: 0 pp |
| Replay ternary >= equal-update repeated current +4 pp | **FAIL**: 0 pp |
| Replay beats uncoupled outcome in both tasks by >10 pp | **PASS** |

**5/7 passed; `advance_to_evolution=false`.** The failure is specifically lack of demonstrated added value from *long-term experience replay* in stationary tasks, despite very strong elementary reward-prediction learning.

## Next step: genuine school final exam

Development-only follow-ups (fresh unseen seed ranges, new preregistration, no use of original sealed sets):

1. **Nonstationary curricula:** A then B then A, with reward mapping changing; compare predictive learner with/without rehearsal **at matched update budgets**. Replay may now improve retention, but can also prevent necessary adaptation. Freeze pre-retraining return probes.
2. **C compositional transfer:** after reliable learning of elementary primitives, present a new combination of roles and cues *before any C-specific outcome feedback*. Do not use handcrafted C-specialized oracle features unless controls isolate them.
3. **Partial feedback and noisy/delayed consequences:** three or more actions, uncertain rewards, and genuine novel sensory contexts. Explicitly measure error calibration and exploration trade-off.
4. Only after demonstrating adaptation AND retention can synaptic formation/pruning be reintroduced as an independent intervention, with a fixed resource ledger.

**No changes to `main`, no reproduction, no original sealed M5.6 source/holdout/replication seeds.** This PR is a stacked development draft above M5.14.
