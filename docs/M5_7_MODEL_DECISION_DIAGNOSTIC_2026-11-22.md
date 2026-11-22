# M5.7 — One-step learned-model decision diagnostic (2026-11-22)

**Status:** development-only diagnostic, NOT prospective confirmatory evidence.

## New capability
M5.7 introduces `neuroevolution/m5_7_decision.py`. Each agent can query its own learned sensory transition predictor for **all physically legal cardinal moves**, without accessing future physical world state. Predicted 19-channel next sensory values are evaluated with a transparent **engineered** utility combining the next-tick food-consumption signal (weight +2), bodily energy reserve (+0.5), and nearby predator channels (−1.5 each). These preference weights are programmer supplied and **NOT** an emergent learned valuation or full planning. Model prediction is one step, and the sensor-based prediction of predators is incomplete as predator movement occurs after the prey's move. Any apparent survival advantage may be architecture- or reward-specific.

## Three-arm strict intervention
All arms have identical 30-tick training histories with prediction-progress curiosity and frozen-then-reset transient hidden state. After tick 30, intrinsic reward and numerical learning are disabled. Only model access at decision time differs:
- **disabled:** original recurrent policy chooses movement, predictor is not consulted
- **model:** legal actions are evaluated using their own learned forecasts
- **permuted:** the same forecasts are scrambled between action labels before selection, as an action-consequence causal control

No extra food/energy, map oracle, future sensor readings or real survival-target labels are injected. Falls back to original recurrent motor controller when predictor lacks enough own-history training. Pairing and energy ledger tested.

## 20-world development diagnostic
The 20 seeds **1080–1099 ARE REUSED** from M5.6 development prediction-progress screening, and thus are **not** a new independent cohort. One predator, 9 initial food, regeneration probability 0.55, 60 ticks split 30/30. Source/final/replication world IDs all remain unopened.

| Decision intervention | At cutoff (tick 30) | Final (tick 60) | Mean model-based actions | Mean food after cutoff |
|---|---:|---:|---:|---:|
| disabled | 19/20 | 9/20 | 0 | 3.40 |
| model | 19/20 | 11/20 | 58.70 | 3.80 |
| permuted | 19/20 | 10/20 | 49.95 | 3.40 |

**Model minus disabled: +2/20 = +10 pp net.** Paired changes: model improves 6 worlds and worsens 4, 10 ties. **Model minus permuted: +1/20 = +5 pp net.** Both too small and too dependent on reused development worlds for inference. An improvement over model-disabled alone is insufficient: causality requires reliable advantage versus action-label-permuted and other matched model controls.

## Evidence and integrity
CI: [M5.7 Actions run 45017027988](https://github.com/kkongiu/neuroevolution-lab/actions/runs/45017027988) passed all 5 new tests, all M5.6 predictor/RNN regression tests, and all 60 development trials. Raw simulation artifact archived in this run (retention 30 days). SHA256 of canonical rows: `bb5fe1296c50e9339bf46d814186fdb792cfdc61cf15ef412394f68c90cdc8cc`.

A test initially failed due to an incorrectly specified comparison in the *test itself*, unrelated to physical behavior. It was corrected to compare equal food forecast before versus after adding predator risk, then the workflow passed. No final/replication worlds consumed, `main` untouched.

## Required next stage
1. Validate counterfactual predictive accuracy, calibration and action-dependent signal separately from survival outcomes, preferably on a new **development cohort** not previously selected based on results.
2. Add a genuinely learned model of value/risk (or explicitly precommit engineered proxy), confidence bounds and long-horizon causal forecasts; calibrate utility on development worlds only.
3. Freeze model, reward budgets, paired baselines and all primary outcomes before opening any sealed final cohort. **Do not claim emergent survival understanding on these diagnostic results.**
