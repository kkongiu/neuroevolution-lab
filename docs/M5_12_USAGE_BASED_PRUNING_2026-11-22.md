# M5.12 — Probabilistic pruning preferentially removing underused synapses

**Date:** 2026-11-22. **Status:** executed developmental experiment, **negative/inconclusive for improved cognitive recall**; measured reduced synapse occupancy and some advantage over random pruning. No evolution or neuron genesis.

## Experiment

Recurrent M5.6 learner with eight hidden units and 260 possible synaptic weight slots. Independent development seeds **3500–3523**, all distinct from earlier experiments and original sealed source/holdout/replication sets. Task sequence visible-context A→B→A→C→B, 160 choices per context (800 total), same hidden cue XOR rule per recurring context, no answer label given. Reward arrives only after actual decisions; the context ID is sensed by the organism.

All pruning variants start from **the same set of 200 active synapses** within each seed and share the same weights, cues, stochastic choice tickets, reward schedule and observed hidden rules. For each pruning variant exactly **four synapses are removed every 40 decisions**, for 20 events and 80 deletions in total, reaching **120 active synapses**. No structural addition occurs, no synapses added; experiment isolates pruning rather than growth.

Six arms:
- `static_200`: same initial 200 active synapses, never prune.
- `random_prune`: remove four uniformly random active edges per scheduled event.
- `magnitude_prune`: preferentially stochastically remove edges with small absolute weights.
- `usage_prune`: preferentially stochastically remove edges with low fast-decaying pre/post coactivation usage.
- `usage_consolidated`: preferentially remove low score with an equal mixture of fast activity and **slow-decaying lifetime usage**. Historical coactivity is an aggregated trace, NOT episodic memory replay, causal counterfactual importance, or a formal synaptic consolidation process.
- `full_260`: full dense 260 edges permanently available, capacity ceiling.

Probabilistic selection without replacement is implemented by assigning pruning likelihood proportional to a bounded exponentially decreasing function of usage/importance. All arms use same deletion timetable and count; no task ground-truth labels enter synapse selection.

**Scientific caveat:** the native M5.6 RNN still performs full-size dense array operations, so lower number of active edge weights is an **allocation proxy** and not measured savings in runtime, FLOPs, electrical energy or computational cost.

## Completed measured results

[GitHub Actions run #45031504704](https://github.com/kkongiu/neuroevolution-lab/actions/runs/45031504704) passed new structural tests plus original controller regression and executed **24 seeds × 6 arms = 144 individual simulations**; full results archived as `m5-12-probabilistic-pruning-development`. Canonical rows SHA-256 `d4cd29ca75976be356b47b54fb40aa8597a39cde6dc0708bdc53d0d70a519a30`.

| Arm | Overall action accuracy | Recall A, first 40 when A returns | Recall B, first 40 when B returns | Mean active synapses | Deletions |
| --- | ---: | ---: | ---: | ---: | ---: |
| Static 200 | 69.27% | 76.15% | 64.06% | 200 | 0 |
| Uniform random pruning | 67.16% | 74.17% | 60.31% | 162 | 80 |
| Small-magnitude pruning | 68.69% | 75.42% | 63.96% | 162 | 80 |
| Recent usage pruning | 67.76% | 75.42% | 61.56% | 162 | 80 |
| Usage + slow historical trace | **68.51%** | 74.90% | **64.79%** | **162** | 80 |
| Dense 260 | 71.16% | 78.85% | 64.90% | 260 | 0 |

### Main comparisons

- Slow historical usage versus recent usage: **64.79% vs 61.56% on returning B (+3.23 pp)**. Supports further study of a slower consolidation/retention trace, not a validated causal mechanism of memory.
- Historical usage versus fully matched uniform random pruning: **+4.48 pp on return B** (64.79% vs 60.31%). The predeclared threshold was **+5 pp**, so **FAIL** (do not round it into success).
- Historical usage versus static-200: **+0.73 pp on return B**, far below the fixed **+5 pp** competence gain criterion. Neither these data nor the smaller total synapse budget prove that pruning outperforms stable wiring.
- Static-200 versus historical-usage overall: 69.27% versus 68.51%, so **98.9% relative accuracy**, passing the 95% preservation screen while using **19% fewer mean active synapse slots** (162 vs 200). Retention for B from original learned-B late-window to return-B first window remains negative in all arms; the problem of forgetting is NOT solved.

### Four fixed development criteria

1. Historical usage beats matched random pruning return-B by >=5 pp: **FAIL** (+4.48 pp).
2. Historical usage beats static-200 return-B by >=5 pp: **FAIL** (+0.73 pp).
3. Historical usage retains >=95% static-200 overall accuracy: **PASS**.
4. Historical usage uses fewer mean active synapses than static-200: **PASS**.

**Only 2/4 passed; `advance_to_evolution=false`.**

## Methodological limitations and next tests

- A slow utilization trace can shield synapses useful in earlier periods, but does not explicitly validate preservation of distinct tasks' memories. More realistic delayed return and disjoint new task families are necessary.
- No synaptic birth, no neuron death, no dynamic network graph beyond fixed 260 slots, no physical conserved resource allocation or neural reproduction in this experiment.
- The stochastic-pruning temperature and 50% historic-usage weighting were development choices: avoid retroactively tuning them and calling a rerun on the same seeds confirmatory.
- There are no paired-seed confidence intervals or independent final test. Correct action rates around 60–70% in an engineered two-choice problem are not evidence for general reasoning.
- A future rigorous test should impose long non-use periods, distinguish low-activity-but-essential connections, and compare forgetting across multiple recurrent contexts, using new development seeds. Consider actual sparse computation only after a proven benefit.

This stacked draft is based on M5.11; `main` remains unchanged and sealed original M5.6 test cohorts are unused.
