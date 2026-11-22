# M5.13 — Synaptic pruning and long-delayed recall (development)

**Date:** 2026-11-22. **Status:** completed; positive evidence relative to matched random edge deletion, but **no compelling superiority over no pruning or short-term usage pruning**. Evolutionary gate blocked.

## Question and experimental design

Can a slow historical trace of synapse usage preserve an earlier competence after increasingly lengthy activity in other contexts? We tested actual online learning of A for 160 binary cue/reward decisions, then alternating B/C for a predeclared **0, 160, 480 or 960** decisions, then returned to **A for 40 probe decisions with NO synaptic weight changes or structural pruning**. Previous-success sensory feedback still updates on each probe, so the probe is not entirely passive, but network parameters are frozen. Context labels are directly observable to the agent, though hidden rules/target responses never are.

New M5.13 **development-only seeds 3600..3619**, separate from earlier development cohorts and the original M5.6 sealed sets. Four conditions from an identical per-seed 8-neuron M5.6 recurrent learner, 260 possible edges, 200 active from the same initial mask/genotype:

- `static`: no synapse deletion;
- `random`: uniformly random pruning;
- `recent_usage`: probability of pruning inversely weighted by fast-decay local pre/post coactivation;
- `historical_usage`: same with half the score from slow-decay lifetime coactivation.

All pruning arms remove **4 edges every 40 training decisions**, down to a floor of 120. Each arm uses the **same exact deletion count and tick schedule**, and matched cue/order/action-ticket random streams. No context oracle determines pruning; the rule bits are used by the simulator for feedback alone. No births or neuron additions. No reproduction/evolution.

Longer delay entails more pruning before the probe (expected for this life-course design), so delay length and accumulated structural deletion are **not separately identified**. Additional fixed-budget/prune-then-wait control is needed to isolate pure time effects.

## Actual executed results

[Successful GitHub Actions run #45032129885](https://github.com/kkongiu/neuroevolution-lab/actions/runs/45032129885) passed three new M5.13 tests, prior M5.6 regressions and executed **20 seeds × 4 delays × 4 arms = 320 complete trajectories**. Raw JSON artifact `m5-13-delayed-memory-pruning-development` retained 30 days; SHA-256 rows: `e23ce21a5ee3b08335ee75eee95a927525062145e88287d4ad1b488fc256a3fe`.

**Frozen-probe accuracy when original context A returns:**

| Intervening trials | Fixed 200 | Random prune | Recent-use prune | Historical-use prune | Active edges at probe in pruning arms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 80.00% | 78.88% | 79.75% | 79.75% | 184 |
| 160 | 69.75% | 68.88% | 67.63% | 68.88% | 168 |
| 480 | 62.00% | 60.00% | 61.00% | 61.63% | 136 |
| 960 | **65.50%** | 57.13% | 63.75% | **64.50%** | **120** |

Acquisition phase last-40 accuracy ~77% for all arms (differences in first-phase pruning). At 960 distraction steps, historical-pruning vs matched random-pruning **+7.375 pp**; historical vs fast recent usage **+0.75 pp**; historical vs fixed **−1.0 pp** (64.5% versus 65.5%). Historic pruning uses an average of **150 active synapses during training**, versus 200 for the fixed control, and ends with **120 vs 200**.

The accuracy change is **nonmonotonic**: 960-step gap gives higher retrieval accuracy than 480 for several arms. The intervening problems repeat A/B/C-specific patterns and the learned networks continue adapting, so this cannot be characterized as a simple exponential memory decay over elapsed time or a purely temporal benefit. Larger replicated studies or alternate distraction schedules are warranted. There are no paired confidence intervals and this is not a sealed independent test.

## Predeclared developmental promotion gate

1. Long-delay historical vs recent-use >=5 percentage points: **FAIL** (+0.75 pp)
2. Long-delay historical vs uniform random >=5 pp: **PASS** (+7.375 pp)
3. Long-delay historical absolute accuracy >=65%: **FAIL** (64.5%)
4. Long-delay historical >=95% of fixed long-delay accuracy: **PASS** (64.5% / 65.5% > 95%)
5. Historical utilizes fewer synapses than fixed: **PASS** (150 vs 200 mean during training)

**3/5 pass; `advance_to_evolution=false`.**

## Interpretive limits and next steps

- History-sensitive pruning beats random rewiring at long interference in this specific development setting; it does not clearly beat recent-use pruning or immutable wiring.
- No context-free problem discovery, no hippocampal-style episodic replay, no biologically validated synaptic mechanism, no spontaneous neuron growth and no physical energy cost. Reduced active-edge count remains an allocation proxy: current dense implementation still computes all potential synapses.
- Long delay and more pruning are confounded. Next clean test should keep all arms at the **same edge count** across gap lengths, compare context-interference vs quiet/no-task delay and test multiple nonrepeating task families.
- Never unlock the sealed M5.6 cohorts until fully independent preregistered protocol and explicit approval. Stacked draft only; `main` untouched.
