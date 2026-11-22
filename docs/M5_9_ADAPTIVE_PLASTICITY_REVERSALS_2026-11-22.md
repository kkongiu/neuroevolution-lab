# M5.9d — Bounded adaptive plasticity, three unannounced reversals

**Date:** 2026-11-22. **Status:** completed, developmental **negative** result. No evolutionary promotion. Abstract two-action problem, not embodied ecology.

## Protocol and engineering

New draft branch stacked on M5.9c; native 260-weight `Learner.learn` now accepts bounded per-update `learning_rate` in [0,.08] while preserving original default 0.02 and existing M5.6 behavior. Three spontaneous hidden-rule reversals after every 160 attempts, four 160-trial phases, each with 40-trial scoring windows and perfectly balanced cue counts. 24 independent M5.9d development seeds 3224–3247. Neither the motor network nor error monitor receives the hidden rule, schedule, correct answer or a reversal notice; only actual success or failure after its action.

Primary controls:
- `constant`: original learning rate .020.
- `decay`: gradual predetermined .020 → .005.
- `adaptive`: surprise-sensitive rate determined by fast (alpha .2) minus slow (alpha .025) exponentially smoothed error frequency, bounded .005–.080. No supervised label or external change-point oracle.
- `random_rate`: independent pseudorandom schedule in .005–.080; importantly **mean .04253, not matched to adaptive's mean .01564**. Its apparent superiority is confounded by larger learning rate exposure and must not be used to claim a timing advantage.
- `yoked_shuffled_rate`: **new fairer control**, generated from the matched seed's *adaptive* counterfactual learning-rate sequence, deterministically shuffled before the placebo test. By construction has exactly the same complete multiset and cumulative dose of learning rates per seed as the adaptive arm, but without its temporal alignment to the placebo organism's realized errors. This is an exploratory counterfactual and its rate sequence is derived from the adaptive arm, not an autonomous online policy. No correct answers given to either arm.

Each condition receives the same number of opportunities/observations (640), the same exogenous stimulus sequence and hidden-rule schedule, and the same genome seed, but action trajectories diverge with learning. 24 seeds × 5 arms = **120 trajectories**. Test suite checks rate bounds, reproducibility, exact matched dose, no inherited state, no cohort misuse, and historical M5.6/M5.9c regressions.

## Verified real results

[Successful Actions run 45028732821](https://github.com/kkongiu/neuroevolution-lab/actions/runs/45028732821). Full rows archived under `m5-9d-adaptive-plasticity-multi-reversal` (30 days), SHA256 `2777fb15a17f6348d0ab21d0416a49d2b8d1620e9fc00bf03df62ac4b9a740da`.

| Control | Mean learning rate | Phase 1 first → last | Mean recovery over phase 2,3,4 | Phase 4 final window |
| --- | ---: | ---: | ---: | ---: |
| Constant | .02000 | 48.23% → 62.50% | **+8.06 pp** | 48.54% |
| Decay | .01250 | 48.13% → 60.94% | +3.30 pp | 44.38% |
| Adaptive surprise | .01564 | 46.98% → 54.17% | **+2.15 pp** | 49.58% |
| Random wider-rate schedule (dose-confounded) | .04253 | 48.65% → 78.02% | +15.83 pp | 48.44% |
| Yoked exact-dose time-shuffled | **.01564** | 48.65% → 56.98% | **+5.35 pp** | 48.85% |

Adaptive minus constant mean reversal recovery **−5.90 pp**. Adaptive minus exact-dose yoked shuffled placebo **−3.19 pp**. None of the four prospectively fixed development-screen criteria passed: initial acquisition >=8pp, final 65% reversal accuracy, 10pp reversal advantage over constant, or 10pp advantage over dose-matched placebo. **Evolutionary promotion is blocked.**

## Interpretation

The current detector incorrectly treats the difference between fast and slow error averages as an indicator of an exogenous change point. It will also react to reward fluctuations caused by exploration, and sometimes learns too slowly when the error trend is stable. Its tiny average learning rate may constrain retention and reversals; the exact-dose placebo still performs better, meaning the simple timing intervention itself is not beneficial in this diagnostic.

The lack of evidence of efficient reversal learning is not proof that dynamic plasticity is biologically impossible. This particular engineering implementation and selection criterion failed on the specified development problem. Adding generations, feeding the detector actual change timestamps or tuning against these same seeds and reporting generalization would be misleading.

**Next scientific step:** first validate a causal change detector (e.g., compare prequential log loss or cumulative evidence under competing stationary vs reversal models) against false-positive / negative controls, and test its effect on adaptation using separate development seeds with the timing of environmental changes hidden. Evaluate rule-change detection metrics separately from learning-rate changes. Do not perform reproductive selection or unlock original sealed M5.6 source/holdout/replication cohorts yet.

Original `main` untouched. No original M5.6 sealed source/final/replication worlds used.
