# FlyLoop M4 — Verified local chemical sensing result, 24 November 2026

**Status: 12/18 preregistered science/integrity gates PASS; overall DEVELOPMENT SCIENCE SCREEN FAIL.** All 10 integrity gates PASS, only 2 of 8 prespecified scientific thresholds PASS. The full GitHub Actions job and byte-for-byte repeat PASS.

- [Original preregistered protocol](FLYLOOP_M4_PROTOCOL_2026-11-24.md), written BEFORE the first full M4 run.
- **[Complete successful GitHub Actions run #45359509501](https://github.com/kkongiu/neuroevolution-lab/actions/runs/45359509501)**, including M0–M3 regression tests, new M4 tests and entire 672-life byte-identical repeat. Previous identical-code science run #45359462600 also passed.
- Raw artifact: \`flyloop-m4-local-odor-causal-results\`, ID **14903956669**, retained through **24 December 2026**. Two 236,741,757-byte full raw JSONs, independently SHA256-verified byte identical:
  \`b5b3905c218e95d3c728f20197bec1f7e1e9f43dead9f379943fb8060cbbd647\`.
- Exact canonical compact raw-rows SHA256 independently recomputed and matched:
  \`d9dd264af4a6995b0acc33cb7ae22bafebac33ca54d0bdf433e4355837e61e7e\`.

## What was tested, without retroactive tuning

The original M0 14-real-weight actor–critic (8 shared actor, 6 critic), original physical energy/integrity TD(0) reward and learning rates, and same five primitive moves, no extra trainable weights, and **M2's BLIND pre-contact sensor mask**. No nearby food, hidden hazard, source coordinates, or direction of food is given to the actor.

A scalar local smell is calculated by the ENVIRONMENT only as the sum of \`0.65**distance\` from active food sources reachable by a legal path of <=7 cells, divided by 4, clipped to [0,1], plus deterministic bounded ±0.015 measurement noise. At each tick the organism reads **only concentration at its current position**. The policy compares its two consecutive actually observed scalar samples, and an **explicitly engineer-designed movement-alignment × signed smell-change feature** gives an action-conditional signal. This is a *stationary graph diffusion proxy*, NOT real airflow/olfaction or autonomous discovery of a sensory representation.

Three continuous-world scenarios × **32 new predetermined world seeds** 8300..8331 × seven same-initialized controls = **672 genuine lifetimes**:
- Baseline gate switch at t=240.
- Procedural 10-post connected mazes and random disjoint gate pairs (the LEARNING RULE, not trained weights, is tested across worlds).
- Exogenous movement of 16 food sites at t=240 and restoration at t=360 with no simultaneous gate change.

Seven controls: frozen blind stochastic actor; uniform N/E/S/W motion with no learning; original blind plastic TD; motor-only previous-action context +TD; true consecutive local smell difference +motor+TD; the same current local smell but actor/critic **signed change from an OLDER strictly past smell sample (SHAM)** +motor+TD; and same correct smell inputs with frozen original weights. All seven use same initial map and same 14 initial weights per scenario+seed, no cross-world learning.

## Primary endpoint — food per 100 ACTUALLY lived steps

| Controller | Baseline fixed barrier | Procedural novel barriers | Food moved/restored |
|---|---:|---:|---:|
| Frozen blind no adaptation | 5.520 | 5.474 | 5.579 |
| Uniform random N/E/S/W | 6.605 | 6.308 | 6.425 |
| Plain blind TD plastic | 6.269 | 5.933 | 5.987 |
| Previous motor-action history +TD | 6.472 | 6.105 | **6.541** |
| Older mismatched smell-change history +motor+TD | 6.406 | 5.527 | 6.470 |
| **True consecutive smell change +motor+TD** | **6.839** | **6.654** | 6.507 |
| Correct smell features but frozen neural weights | 5.547 | 5.264 | 5.506 |

M4 does provide *evidence of useful temporal chemical signal alignment* **within the new procedural map distribution**. The true local-odor learner's food rate **6.654** beats its shuffled-past-consequence counterpart **5.527**, an increase of **+1.128 foods per 100 lived ticks** (about +20.4%); the complete-world paired bootstrap exploratory 95% interval is **[+0.567,+1.672]**. The exact predefined **+10% over SHAM criterion PASSES**.

The true local-odor learner also beats **correct-odor-with-frozen-weights** in the procedural environment **6.654 vs5.264** (about +26.4%, passing prespecified +15%). Therefore the benefit is not explained solely by fixed random actor weights + engineered odor features.

**However**, its performance compared with moving random control is **6.654 vs6.308** (only **+5.5%**), short of the preregistered +10%; paired world-seed 95% interval for this rate difference **[−0.557,+1.146]** includes zero. It similarly misses +10% against motor-only **6.654 vs6.105** (+9.0%). In the baseline geometry the true-odor rate **6.839** misses the required +15% against plain plastic (**6.269**), +10% against random (**6.605**), and +10% against motor-only (**6.472**). Therefore **NOT demonstrated superior to random movement or motor-only under the preregistered screen**.

When food relocates at t=240, the 120-step relocation-phase food per organism is **5.438 with true odor vs 5.563 with motor-only** (dead agents count as zero); prespecified +15% improvement criterion FAILS. It is not evidence of the sought alternative-behavior strategy on changing resources.

All controllers still show substantial hazard contacts, wall collisions, and limited final survival. In procedural worlds true-odor organisms live an average of **338.1/480 steps**, with **15.6%** surviving all 480; the uniform-motion organism survives **353.7** steps, with **28.1%** completing the life. Improved odor-feeding rate therefore does not imply improved overall survival.

## Prospective 18 gates: 12 PASS, 6 FAIL

**10/10 integrity gates PASS:**
1. Exactly 672 balanced complete real episodes; full byte-identical second run.
2. Legal physiology/movement and wall constraints for all movements.
3. Identical per-seed maps/environment schedules and initialized 14 neural weights across all seven control arms.
4. Complete zeroed raw pre-contact food/hazard sensors and no privileged signal in neural input.
5. Only local current cell scalar smell, graph-path wall occlusion and bounded noise.
6. Consecutive true current+previous actually observed odor values, causal movement alignment and zero delta on the first tick.
7. SHAM odor differences use only older-than-current/previous *already observed* samples, with a separate RNG.
8. Exact original M2/M3 frozen/plain/random/motor baseline replay, original M0 TD and exact weight count.
9. Connected, novel procedural mazes and exogenous matching food relocation.
10. All actual trajectories, negative findings, checksums, weights and disabled evolution flags archived.

**2/8 prospective efficacy gates PASS:**
11. Baseline TRUE vs plain plastic ≥15% — **FAIL**.
12. Baseline TRUE vs uniform movement ≥10% — **FAIL**.
13. Baseline TRUE vs motor-only ≥10% — **FAIL**.
14. Procedural TRUE vs uniform movement ≥10% — **FAIL**.
15. Procedural TRUE vs motor-only ≥10% — **FAIL**.
16. Procedural TRUE vs SHAM older differences ≥10% — **PASS**, +20.4%.
17. Food-relocation phase TRUE vs motor-only ≥15% — **FAIL**.
18. Procedural TRUE vs frozen-chemo ≥15% — **PASS**, +26.4%.

ALL 18 were required for a narrow developmental science result: **OVERALL FAIL**. Green software CI is never equated with scientific success.

## Independent audit of raw GitHub artifact

After the completed run, the entire Actions ZIP was downloaded and checked using an independent Python implementation **without importing any FlyLoop model source**. It validated:
- Both complete JSON files have the same size **236,741,757 bytes** and the identical full SHA256 printed above.
- Compact raw-row SHA256 independently reconstructed from all 672 records exactly matches the published evaluation digest.
- All **672 organism lives**, **96** matched scenario×seed groups with equal maps and equal original neural weights.
- All **232,034 real physical moves**, **638 environmental event records**, with independent action→position, wall checks and actual physical reward=Δenergy+1.25×Δintegrity reconstruction.
- All **65,610** checked real-current/delayed chemical sample comparisons; all **32,464** strictly earlier SHAM-source references tested to be older than the last sample; none pointed to future information.
- Every raw pre-contact actor feature was checked for prohibited food/hazard signal; sampled odor bounded and prior sample/delta/EMA reconstructed correctly.
- **ZERO detected anomalies** across these independent checks. The audit checks described do not independently re-simulate all environmental odor fields at all points; the software unit tests verify the wall-bounded field calculation.

## Scientific interpretation

The biologically inspired idea of comparing scalar chemical sensation over time **has at least a measurable task-specific signal**: recent correctly timed odor differences outperform temporally mismatched odor history in the procedural layout. That is narrower and more credible than the memory-only M3 result.

It is **not** robustly stronger than a non-learning uniform-motion organism, and it fails the new-resource phase. The model does not demonstrate spatial memory, a newly invented navigation strategy, generalized adaptive autonomy, real fly neurobiology, or evolution.

A sensible next experimental question is whether memory should retain a **short history of sensory input and outcomes with learnable compression**, rather than a handcrafted signed alignment with the previous direction; maintain real-vs-sham controls and verify better-than-uniform motion under *fresh* environments. Do not merely enlarge a trained model or add a path-planner.

Stop flags remain FALSE: \`general_intelligence_demonstrated\`, \`spatial_memory_verified\`, \`autonomous_operator_invention_verified\`, \`evolution_authorized\`, \`advance_to_evolution\`.

**Provenance:** DRAFT PR stacked on M3 PR #89; neither \`main\` nor sealed M5.6 nor M0–M3 source was changed.
