# M5.8 — Progressive stochastic world generator: development-only audit

**Date:** 2026-11-22. **Status:** generator and exogenous event proposals implemented and tested; events are **NOT yet enacted in physical M5.7 ecology** and cannot support survival or intelligence claims.

## Goal

Evaluate whether embodied organisms can learn and transfer strategies across spatially progressive opportunities and structured, surprising but *conditionally learnable* environmental challenges, without reward shaping, experience points, game levels, a correct-action oracle or preferential treatment by the simulator.

## Frozen generator v1

- 11 × 11 grid with 9 × 9 interior and all accessible cells mutually connected, including a permanent horizontal/vertical route spine.
- Three adjacent x-axis zones `shelter`, `frontier`, `wild`, with increasing random obstacle probabilities: **0.04, 0.14, 0.25**.
- Nine initial resources allocated **4, 3, 2** by zone (easy/frontier/hard), never placed on an obstacle.
- An exogenous 60-tick schedule generated from the world seed before any agent acts. Per-tick event proposal probability per zone **0.03, 0.10, 0.18**. Events: `drought`, `detour`, `predator_shift`, `compound`; durations 2–5 ticks, intensity 0.2–0.7. Detour proposals only target unoccupied-by-initial-food, non-spine cells; dynamic collision with live agents must be safely handled in the future adapter.
- Generator never receives agent behavior, surviving-population counts, prediction errors or chosen actions as inputs. No curriculum promotion rule.
- 64 dedicated **development** world IDs 2000–2063. Prohibits other IDs, including all known source/final/replication M5.6 reserves.

### Actual audit

[GitHub Actions successful execution 45017932212](https://github.com/kkongiu/neuroevolution-lab/actions/runs/45017932212): 4 targeted unit tests passed and 64 worlds generated with 64 distinct hashes, fully traversable open cells, protected spine and valid resources. JSON for all worlds and all planned events archived as workflow artifact.

| Quantity | Shelter | Frontier | Wild |
| --- | ---: | ---: | ---: |
| Mean static obstacles per world | 1.09375 | 2.296875 | 5.40625 |
| Precomputed event proposals over 64 worlds × 60 ticks | 114 | 380 | 695 |

Total event proposals: **1189**. SHA256 all generated world records: `629a7c5a61cf824cc560056a44fc469d89ce344d1a5477df5e6f9a4385e6f185`.

**Interpretation restrictions:** Event probabilities and quantities are programmer-specified difficulty scaffolding, not proof of effective or smoothly increasing *agent experienced* difficulty. Events may occur concurrently, and a later dynamic world adapter must resolve collisions, prevent obstacle closure of escape routes, preserve physical calorie accounting and ensure predation rules stay fixed. Drought cannot silently delete food/energy without tracking conservation. Altering predator behavior is a *programmed environmental perturbation*, not a naturally learned predator strategy.

## Pre-experiment causal design

After physical integration and independent engineering validation, freeze world rules and compare **matched experience budgets**:
1. spatial progressive vs uniform environmental difficulty;
2. same level distributions in permuted zone/order assignments;
3. no model access vs true learned forecast access vs action-label-permuted forecast access;
4. persistence, predicted-effect calibration, food/energy and predator avoidance;
5. transfer to geographically/geometrically novel but **unopened** independent worlds.

No post-hoc tuning on the reserved sealed cohorts. Report both successful exploration and impossibility rates. Evaluate physical outcomes at the WORLD level, not treating descendants as independent replications.

## Repository boundaries

Created on `experiment/m5-8-progressive-stochastic-worlds` extending M5.7 draft; neither M5.7 nor M5.8 is merged into `main`. The standalone generator intentionally makes no changes to earlier ecology code.
