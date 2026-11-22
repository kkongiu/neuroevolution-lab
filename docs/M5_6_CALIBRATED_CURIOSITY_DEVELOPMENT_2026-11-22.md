# M5.6 calibrated curiosity versus survival — DEVELOPMENT-ONLY result

**Status:** exploratory developmental screening. No confirmatory held-out or replication worlds opened. **Date:** 2026-11-22.

**Provenance:** [GitHub Actions SUCCESS 45012356795](https://github.com/kkongiu/neuroevolution-lab/actions/runs/45012356795), full raw 168 paired world rows archived in the workflow artifact, row digest `0bfe031b6a066b61cb4bdf7c9a63df62ee213b32e9430f3863975666a71b8d06`.

## Design

24 new development worlds, IDs 1032–1055, distinct from the 12 calibration worlds 1020–1031, original pilot worlds 1000–1011, and all reserved source, holdout and replication worlds. The 7 conditions share starting worlds. Same original predator motor/capture rules, energy accounting and scripted reproduction. The environment was selected *solely* from **no-curiosity** baseline calibration: 1 predator, 60 ticks, 9 initial food, 0.55 food-regeneration probability. Each policy updates weights via one-step truncated recurrent policy gradient with its own reward, except the explicitly frozen control.

| Condition | Final population persistence | Difference vs zero curiosity |
| --- | ---: | ---: |
| No curiosity | 11/24 (45.8%) | 0 pp |
| Weak novelty, 0.25× | 10/24 (41.7%) | −4.2 pp |
| Moderate novelty, 1× | 10/24 (41.7%) | −4.2 pp |
| Strong novelty, 4× | 15/24 (62.5%) | +16.7 pp |
| Engineered energy-gated novelty | 10/24 (41.7%) | −4.2 pp |
| Randomized novelty feedback placebo | 11/24 (45.8%) | 0 pp |
| Frozen synaptic updates | 13/24 (54.2%) | +8.3 pp |

With strong novelty, 6 paired seeds improved and 2 worsened vs no curiosity; the other 16 had the same binary outcome. The 4/24 net advantage is preliminary and uncertain. The energy gate is an **engineered** reference, not evidence of emergent regulation.

### Secondary descriptive observations

| Condition | Mean food collected | Mean movements | Mean first-visited cells | Mean final population |
| --- | ---: | ---: | ---: | ---: |
| No curiosity | 19.63 | 233.13 | 95.83 | 1.04 |
| Strong novelty | 20.67 | 240.04 | 100.63 | 1.50 |
| Frozen learning | 19.29 | 236.50 | 100.92 | 1.33 |

These differences may reflect longer life and larger surviving populations; do **not** interpret aggregate movement/first-visit differences as isolated causal evidence of intrinsic exploratory behavior. Reward dose and number of weight-changing updates differ dramatically by arm; they are NOT matched learning-budget controls. The chosen strong-reward coefficient (4×) was already present in the first exploratory pilot, but this cohort remains developmental, not an independent confirmatory test.

## Interpretation and strict gate

There is a **signal**, not evidence sufficient to establish a survival benefit. The frozen-weight comparator survives 13/24 worlds; reward and gradient dose differ between policies; neither retention after removing reward nor genuine learned hunger-sensitive self-regulation has been demonstrated. Curiosity-as-visit-novelty is NOT the yet-unimplemented predictive learning-progress reward.

**No final M5.6 test is authorized by this result.** Next requirements: choose and freeze a complete executable design, match reward/learning budgets, implement the preregistered six-arm memory-reset and scripted controls, train and freeze the source-weight bank, verify transfer without intrinsic reward and only then consume sealed world sets. Never tune hyperparameters from held-out performance.

The code lives in draft PR #52, and main is unchanged.
