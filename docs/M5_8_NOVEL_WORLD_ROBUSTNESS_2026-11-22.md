# M5.8 — Baseline pressure calibration and novel-development-world robustness

**Date:** 2026-11-22. **Status:** implemented and executed, outcome **negative/inconclusive** for robust predictive survival; pressure calibration target not met. No use of original M5.6 source/holdout/replication cohorts.

## Reproducible evidence

[Successful GitHub Actions run #45021188000](https://github.com/kkongiu/neuroevolution-lab/actions/runs/45021188000): 4 strict robustness tests, 4 historical generator tests, baseline calibration, **288 ecological trials** on **32 previously unseen M5.8 development world seeds 2064–2095** (three spatial challenge arrangements × three motor intervention modes). Complete machine-readable rows plus 48 calibration runs archived as the artifact `m5-8-baseline-pressure-and-novel-development-worlds`, 30-day retention. Full new-world rows SHA256: `f5c9313101d0354625b2c44159085b52fc714f93512d49aee83ea2282aed3295`.

## Pressure calibration — failed to achieve balanced difficulty

Predeclared baseline-only criterion: nearest 40% world-level final survival across 24 **previously used** development seeds 2000–2023, no model-based motor decisions, progressive geography, 60 ticks with learning cutoff at 30, same ecological rules. Alternative `moderate` executes a second predator physical step only on every other tick during shock periods; `full` performs it during every shocked tick.

| Candidate | No-model survivors / 24 | Average extra predator steps | Distance from 40% |
|---|---:|---:|---:|
| Full | 5/24 = 20.83% | 27.29 | 19.17 pp |
| Moderate | 4/24 = 16.67% | 13.83 | 23.33 pp |

**Automatically selected full**, before inspecting 2064–2095. The candidate set and target were implemented in code prior to the new-world run. However, both variants miss the intended 40% baseline considerably: environment remains too hostile. A less aggressive predator schedule did **not** monotonically raise survival in this small sample, underscoring nonlinear trajectories, spatial shocks and noise. This is a failed *difficulty-target calibration*, not an unqualified success.

## Three-arm predictive causal check on 32 independent new world seeds

Every group trains its own network from scratch for 30 ticks in its own world, then stops all policy/predictor weight updates and intrinsic bonus for ticks 31–60. Within a world arrangement and seed, all three experimental motor interventions share exactly the same training trajectory, including physical state and numerical predictor updates at cutoff. The intervention affects only post-cutoff action decisions:

- `disabled`: recurrent policy, ignores predictor
- `model`: chooses from its locally trained action-conditioned next-sensory forecasts
- `permuted`: action labels intentionally scrambled, same set of forecasts

Same physical predator, food, world geometry and event proposal schedule within each matched triplet. The measured world persistence is:

| Geography | Disabled | Model | Permuted | Model − disabled | Model − permuted |
|---|---:|---:|---:|---:|---:|
| Progressive | 6/32 | 5/32 | 4/32 | −1 world | +1 world |
| Uniform | 1/32 | 3/32 | 1/32 | +2 worlds | +2 worlds |
| Reversed | 5/32 | 4/32 | 2/32 | −1 world | +2 worlds |
| Total of 96 correlated geometry/seed cells | **12/96** | **12/96** | **7/96** | **0** | **+5** |

**Do NOT treat 96 rows as 96 independent biological worlds**: the same 32 seeds generate correlated spatial variants. There is no demonstrated advantage over disabling model access. The +5/96 against scrambled forecasts is an exploratory lead, not a validated generalized causal ability. The earlier 384-trial developmental positive effect did not replicate in these newly generated worlds.

## Scope and critical limitations

1. This run tests **out-of-sample generalization of the learning *procedure*** to 32 new worlds, **not the transfer of any learned synaptic weights or individual knowledge from one world to another**. Each founder's policy and forward predictor are initialized afresh in each world. True cross-world learning transfer needs an explicit source training stage with frozen numerical weight bank, new-world tests where those weights are loaded, and a source-reset matched control.
2. No independent M5.6 preregistered final or replication cohorts opened.
3. Reward scales/gradient counts are not matched between unrelated geographically different arms; the three matched motor interventions **within** each seed/world share exactly the same prefix.
4. One-step sensory forecasting + a programmed utility for expected food/energy and nearby predators is not autonomous learning of value nor long-horizon planning.
5. The full-vs-moderate calibration says the current target difficulty remains unresolved; further tuning must use *new development-only calibration seeds*, never retune from this new-world contrast and then claim independent confirmation on it.

## Decision

**M5.8 does not yet support the claim that learned predictive reasoning robustly improves survival, or that spatial progression beats uniform/reversed curricula.** Primary followup should be a calibrated, learnable task with measurable forecast accuracy, not merely lower predator counts or further unblinded hyperparameter search. Real synaptic transfer and explicit prediction accuracy tests remain outstanding.

Code remains in draft PR #54, stacked on draft M5.7 PR #53. Main unchanged.
