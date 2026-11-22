# M5.8 — Enacted progressive stochastic worlds: 384-trial exploratory result

**Date:** 2026-11-22. **Status:** completed development engineering and exploratory survival comparison; not a validated demonstration of progressive-curriculum learning or predictive generalization.

**Evidence:** [Successful GitHub Actions run #45018386258](https://github.com/kkongiu/neuroevolution-lab/actions/runs/45018386258), including full 64-world exhaustive matched-budget tests, physics / energy audits, all M5.7 regression tests, 384 physical simulations and archived world-level CSV-compatible JSON. Row SHA256 `4eff1619ba0f9f0cd2d0bc49a9d6fecb890f9a7a9017ed916584acc871e3c00a`. All raw results available in the Actions artifact `m5-8-enacted-physical-experiment` (30-day retention).

## What was physically integrated?

In original M5.0/M5.5B ecology with M5.7 RNN predictor, attach 11×11 seeded world geometry and resources from M5.8. All shock proposals are precomputed solely from seed/tick/zone, never from agent behavior:
- Drought temporarily suppresses **future** food regrowth candidates in affected zones, without destroying already extant physical calories.
- Detour creates temporary **real obstacles** only when no agent, predator or food occupies the target and the entire accessible interior remains connected. Unsafe events are skipped, and skips counted. A permanently open route spine is preserved.
- Predator shifts cause one extra physical moving/capturing step by the original M5.5B predator for each tick with a live shift event; actual caloric death ledger remains balanced. This is explicitly programmer-controlled environmental pressure, not autonomous predator adaptation.
- Compound event applies combinations. No future outcomes, maps or shock calendar exposed to the agent; only existing local sensory consequences.

**Every trial:** 60 ticks, 8 initial organisms, one physical predator, 9 initial food resources, neural prediction-progress learning for 30 ticks, after which policy numerical weights and predictor learning/reward are stopped. Compare physically identical initial worlds/training histories *within each spatial variant* with actual model-based action selection vs original RNN access disabled.

## Matched-budget spatial variants

Three spatial conditions use **the same per-seed count** of static obstacles, food, exogenous event proposals (same timestamps, type, length and strength), but different spatial allocation. Geography changes across variants, so within-seed **variant comparisons are not same-history counterfactuals**; model-access comparisons *within one variant* are same-history counterfactuals.

| Across 64 seeds | Shelter / x 1–3 | Frontier / x 4–6 | Wild / x 7–9 |
|---|---:|---:|---:|
| Progressive: walls | 70 | 147 | 346 |
| Uniform: walls | 210 | 186 | 167 |
| Reversed: walls | 346 | 147 | 70 |
| Progressive: initial food | 256 | 192 | 128 |
| Uniform: initial food | 192 | 192 | 192 |
| Reversed: initial food | 128 | 192 | 256 |
| Progressive: precomputed events | 114 | 380 | 695 |
| Uniform: precomputed events | 415 | 398 | 376 |
| Reversed: precomputed events | 695 | 380 | 114 |

Total resources and environmental burdens match exactly by seed. Some events may overlap or be skipped physically for safety; count of proposals alone does not establish equal actual experienced difficulty. Static accessibility checked in all 64 worlds and all three variants.

## Final survival — actual world-level outcomes

| Spatial distribution | Without model access | Learned forecasts used | Difference |
|---|---:|---:|---:|
| Progressive | **10/64** | **13/64** | +3 worlds (+4.69 pp) |
| Uniform | **8/64** | **15/64** | +7 worlds (+10.94 pp) |
| Reversed | **10/64** | **15/64** | +5 worlds (+7.81 pp) |

The model access benefit **does not establish significance or robust generalization**: same limited worlds, same learned utility, no freshly sealed/tested environments, no confidence intervals used for a claim, and 3 exploratory contrasts. It also does not separate benefits from fixed engineered food/energy/predator preference scores from actually learned meaningful risk forecasting. **Progressive environments did not outperform uniform or reversed alternatives.** There is no empirical support here for the claim that progression is the better training curriculum.

At the training/evaluation boundary, world persistence rates were: progressive **49/64**, uniform **52/64**, reversed **51/64** (same values for model and disabled within each variant). By end of tick 60, overall persistence is low. Stochastic pressure is high, including roughly **25.48 extra predator steps per world** on average. This may be so harsh that it obscures learning signal; calibrate on development baseline only if proposing a next protocol.

## Key limitations and follow-up

1. This tests **spatial allocation of challenges within a 60-tick world**, *not* sequential lesson promotion, adaptive difficulty, generation-long cross-world learning or transfer to unseen worlds. Agents' synaptic changes are nonheritable. Avoid calling this game-style curriculum mastery.
2. Training duration is matched by tick, **not by number of experiences or gradient updates**, since birth and premature deaths vary across environments.
3. Shock proposals are exogenous, but whether detours are physically applied depends on agent occupancy and safe connectivity; count actual skips.
4. To evaluate predictive causal reasoning, compare true action forecasts not just to model-off but also to **action-label-shuffled forecasts on the same M5.8 worlds**; that arm has not yet been run.
5. Develop a prospective transfer protocol with worlds held aside from all exploratory analysis, plus prospective survival and predictive calibration tests; do not consume existing M5.6 sealed cohorts without complete prerequisite freeze.

## Repository integrity
M5.8 remains a draft branch and draft PR #54 targeting M5.7 draft PR #53. `main` unchanged. Original M5.6, M5.7 development histories remain untouched when M5.8 world mode is disabled. Original neural motor/predator energy logic remains active when enabled.
