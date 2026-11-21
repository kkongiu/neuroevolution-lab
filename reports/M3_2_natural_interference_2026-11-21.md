# M3.2 — Sequential learning without injected damage: full experiment

**Date:** 21 November 2026  
**Scientific outcome: PRIMARY HYPOTHESIS NOT CONFIRMED.**  
**Operational outcome: successful GitHub Actions, all tests passed.**

- [Protocol preregistered before implementation/full run](../docs/M3_2_PROTOCOL.md)
- [Original frozen run #44899315401](https://github.com/kkongiu/neuroevolution-lab/actions/runs/44899315401)
- [Implementation](../neuroevolution/m3_2.py) and [PR #11](https://github.com/kkongiu/neuroevolution-lab/pull/11) merged to main
- [Per-seed archived CSV](M3_2_seeds_2026-11-21.csv)
- Original artifact: **m3-2-natural-interference**, ID **14705675380**, expiry **21 December 2026** (the artifact also includes full per-task and counterfactual data).

## 1. Why M3.2 differs fundamentally from M3.1

M3.1 succeeded in a **deliberately engineered damage-and-repair** environment. That proved that a diagnostic can choose an action with a positive local effect, but its corruption was artificially aligned with the renewal mechanism.

M3.2 **does not alter learned weights artificially**. A *single persistent* 16-unit ReLU network experiences 12 successive supervised classification tasks with stochastic novelty and exact recurrence of latent task rules, changing its weights through normal SGD alone. New learning can interfere with previous representations. At each boundary we permit either no intervention or renewal of the **three least-active hidden units** on the previous task's training samples.

There is a fixed budget of **four renewal actions over eleven transitions**. All competitive arms receive **exactly four**; controls without renewal and with renewal every boundary are explicitly *nonmatched diagnostic arms*.

## 2. Frozen design and data splits

- Meta-training **6 seed**: 16000–16005; meta-validation **3 seed**: 17000–17002.
- Independent main final test **20 seed**: 18000–18019, using line/XOR/circle/stripe.
- Independent shifted-family final test **12 seed**: 19000–19011, replacing XOR/circle with checker/wedge.
- 12 tasks/sequence; 48 SGD steps × batch 2 per task; width 16, learning rate 0.04.
- Validation of learning uses 64 independently generated heldout examples per task, with AUC measurements at steps 0, 12, 24, 36, 48.
- Three separately evolved numeric decision policies on meta-training/validation data (8 candidates × 4 generations each): full state, current accuracy/time only, and time only.
- Full state includes current and prior training accuracy, gradient cosine similarity and low-activity fraction, plus time. **All observations are from training data.** Neither final-test labels nor latent task identities/return status are input features.
- Selection fitness was frozen as learning-curve AUC minus 0.08 × average old-task forgetting. The final test was not used to optimize or change any parameters.

**AUC here is area under the *within-task learning accuracy curve*, not ROC AUC.** The statistical unit is the entire seed sequence, not individual data points or tasks.

## 3. Main 20-seed results

| Arm | Learning AUC | Forgetting of old tasks | Renewals / lifetime |
| --- | ---: | ---: | ---: |
| **Full state (adaptive)** | **74.46%** | **21.51 pp** | 4 |
| Independently learned **time-only** | **74.77%** | **22.10 pp** | 4 |
| Independently learned **accuracy-only** | **74.84%** | **21.52 pp** | 4 |
| Uniform four interventions | 74.29% | 20.65 pp | 4 |
| Shuffled, same renewal count | 74.09% | 20.64 pp | 4 |
| Random units, **same intervention timing** | 72.96% | 19.55 pp | 4 |
| Never renew (diagnostic, different budget) | 73.43% | 19.50 pp | 0 |
| Renew at every boundary (diagnostic, different budget) | 75.24% | 22.80 pp | 11 |

### Preregistered primary hypothesis: FAILED

- **Adaptive minus independently evolved time-only AUC:** **−0.316 percentage points**.
- Paired seed-level descriptive bootstrap 95%: **−0.837 to +0.100 pp**.
- Of 20 seeds, adaptive wins **9**, loses **10**, and ties **1**. This is not evidence for a useful full-state controller.

**Required secondary gate also fails:**
- Adaptive minus independently evolved accuracy-only: **−0.382 pp**, bootstrap 95% **−0.983 to +0.130 pp** (8 positive / 12 negative).

Consequently, the claim that representation/gradient signals improve renewal scheduling over simpler training feedback is **not confirmed**.

### Exploratory contrasts: a narrower positive effect

- Adaptive vs **dose-matched shuffled**: **+0.368 pp**, interval **−0.235 to +0.963 pp**; not robust.
- Adaptive vs **uniform**: **+0.167 pp**, interval **−0.184 to +0.546 pp**; not robust.
- Adaptive vs **no renewal**: **+1.025 pp**, interval **+0.613 to +1.433 pp**; this comparison has *different renewal budgets*, so it cannot show that adaptive scheduling is superior.
- Adaptive vs **random-unit selection with the same schedule**: **+1.494 pp**, interval **+0.621 to +2.324 pp**; 16/20 positive, 4/20 negative. This is consistent with the prior evidence that selecting *which* low-activity units to renew matters. The effect is secondary/exploratory, and forgetting is **higher** in the full-state arm (21.51 vs 19.55 pp).

A strategy that renews at *every* boundary has slightly higher main AUC than any 4-action strategy (**75.24%**) but uses nearly three times as many interventions and has worse forgetting (**22.80 pp**). It is not a fair same-budget alternative.

## 4. Shifted-family results on 12 new seed

| Arm | Learning AUC | Forgetting |
| --- | ---: | ---: |
| **Adaptive** | **64.44%** | 11.11 pp |
| Time-only | **64.56%** | 11.53 pp |
| Accuracy-only | **64.28%** | 11.27 pp |
| Shuffled (4 actions) | **64.22%** | 11.39 pp |
| Uniform (4 actions) | **64.60%** | 11.45 pp |
| Random units (same timing) | **64.01%** | 11.44 pp |
| No renewal | **64.71%** | 10.65 pp |
| Renewal at every transition | **64.07%** | 12.35 pp |

- Adaptive minus time-only: **−0.126 pp**, paired bootstrap 95% **−0.997 to +0.928 pp** (4 positive, 8 negative).
- Adaptive minus accuracy-only: **+0.161 pp**, interval **−0.790 to +1.167 pp** (5 positive, 7 negative).
- Adaptive minus shuffled: **+0.220 pp**, interval **−0.768 to +1.372 pp** (7 positive, 5 negative).
- Adaptive minus random units: **+0.425 pp**, interval **−0.975 to +1.848 pp** (6 positive, 6 negative).

The low-activity selection advantage **does not replicate decisively** on the shifted families. In this new task mix, the simple *no-renewal* diagnostic has the highest AUC of the eight arms, although differences are small.

## 5. Local causal intervention measurements

Unlike a naive retrospective before/after comparison, for the adaptive controller we cloned its **actual persistent pre-boundary network** and evaluated both alternatives (preserve or renew) on identical training data with the same updates, then committed *only the policy-chosen branch* for the next task.

These estimates answer: **would renewing right now improve the next task on this exact state?** They do not answer whether a different sequence of actions would lead to a globally superior *entire lifelong trajectory*.

| Local effect of renew − preserve | Main final test | Shifted final test |
| --- | ---: | ---: |
| Independent sequences | 20 | 12 |
| Paired pre-boundary states | **220** | **132** |
| Mean **one-task learning AUC** effect | **+0.439 pp** | **−0.095 pp** |
| Mean **old-task forgetting** increase | **+0.710 pp** | **+0.517 pp** |
| Local oracle-action agreement | **46.8%** | **53.8%** |

In the main experiment, 101 of 220 local renewals would increase the diagnostic *combined* AUC–forgetting value, 88 would decrease it, and 31 would tie. In the shifted test, 59 of 132 improve it, 71 worsen it and 2 tie. **The adaptive controller is not reliably choosing the locally better intervention**, despite its actions responding to its observed network state.

It is important that renewal can have a small positive *learning* effect and a larger *memory-loss* effect simultaneously: optimizing only new-task AUC conceals this cost.

## 6. Decisions and recurring tasks

The adaptive policy shows real behavioral sensitivity, but **sensitivity is not utility**:
- Main test: **106/220** requested transitions where state features changed the action compared with time-only masking, excluding forced-budget decisions. **39** actual decisions were forced by quota feasibility.
- Shifted test: **71/132** state-sensitive unforced requests; **13** forced decisions.

The adaptive policy's AUC on regular final tasks is **71.68% for new latent tasks** and **78.62% for returning tasks**. On the shifted final tasks, the corresponding figures are **63.22%** and **65.99%**. The difference should **not** be interpreted as a controlled causal effect of prior experience: task difficulty and sample distribution also vary between novel and returning events.

## 7. Overall scientific decision

**M3.2 is a useful negative result for the primary hypothesis.** Without pre-engineered corruption, on this small recurrent supervised task family, watching gradient conflict and hidden-unit activity did **not** produce better intervention timing than watching only the task index or accuracy.

However, there is a *narrower provisional signal* that resetting the least-active units is beneficial relative to choosing random units at the same intervention times, at least on the regular family. That improvement comes with worse forgetting and did not generalize clearly to shifted task families.

### Limitations

1. All latent tasks share the same 2D inputs and no task identity/context cue. Some tasks can assign contradictory labels to the same points. **Perfect retention may be impossible by construction**: a critical obstacle for studying genuine continual lifelong memory.
2. A very small, externally evolved policy and tiny 16-unit network may lack sufficient representational or optimization capacity. The null does not exclude better policies in other settings.
3. One-step counterfactuals identify conditional immediate effects only. Evaluating entirely alternative lifelong trajectories would require a separately specified experiment.
4. Identical SGD steps and intervention counts are **not identical real CPU costs**; the extra diagnostic gradient calculations have overhead. The full run recorded ~46 seconds, including meta-search and diagnostic counterfactuals, not a per-policy normalized benchmark.
5. The task families of the shifted test are still synthetic. Two families overlap with meta-training.
6. Paired bootstrap intervals are **descriptive**, and multiple secondary comparisons were performed.

### Proposed next experiment (M3.3)

**Before making the controller more complicated, change the world to make lifelong memory *possible*.** Introduce an explicit contextual cue or distinct identifiable task regime, so that different tasks no longer require contradictory labels for the same input. Then preregister a representation/forgetting benchmark with a larger number of tasks and seeds, and evaluate whether adaptive restructuring improves a joint learning–retention metric over time-only and accuracy-only policies under the same wall-time budget.

Freeze new seed ranges and task family settings before evaluating. The original M3.2 results and its final-test seeds **must remain unchanged**.

## Provenance and files

Original GitHub run: [#44899315401](https://github.com/kkongiu/neuroevolution-lab/actions/runs/44899315401). Artifact name **m3-2-natural-interference**, ID **14705675380**, expiry **2026-12-21**.

Contains:
- summary.json: seed-level paired bootstrap and all aggregate measurements.
- seed_metrics.csv: **256** seed-by-policy rows (32 seeds × 8 arms).
- task_metrics.csv: **3,072** task observations (32 seeds × 12 tasks × 8 arms).
- local_counterfactuals.csv: **352** paired local preserve/renew comparisons (20×11 + 12×11).

The main per-seed AUC and forgetting values are also archived in [M3_2_seeds_2026-11-21.csv](M3_2_seeds_2026-11-21.csv) permanently alongside this report. No retuning was performed after viewing final test data.
