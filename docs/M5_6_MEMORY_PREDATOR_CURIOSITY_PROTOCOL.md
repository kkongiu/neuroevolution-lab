# M5.6 — Memory × predators × curiosity: prospective protocol (DESIGN ONLY)

Date: 2026-11-22. Status: preregistration draft; NO new experimental worlds, outcomes, or claims.

## Repository and dependency gate
This is the **neuroevolution-lab** project, NOT the independent **alife-lab** artificial-chemistry repository. M5.5B/C physical predators and M4.4A online RNN memory are on distinct unmerged branches (PR #49/#50 and #51). Do not imply they can already run together. First implement a reviewed integration on this branch, with exact unchanged historical implementations and tests, then freeze an executable version of this protocol in a separate commit BEFORE accessing test seeds.

## Primary question
Does intrinsic novelty reward, coupled to a learned recurrent memory, improve *physical* survival under predators compared with the same memory, training budget, observations and ecology without novelty reward?

## Factorial arms
A: RNN memory + intrinsic novelty reward + physical predators.
B: same RNN memory + zero novelty reward + identical predators (primary control).
C: memory reset at each decision + novelty + predators (memory ablation).
D: memory + novelty + no predators (ecological pressure check).
E: memory + randomized novelty feedback + predators (reward placebo).
F: fixed, programmed exploration without intrinsic reward + predators (behavioral baseline).
No supervised correct-action labels or external guiding of motor choices in primary ecological evaluation; if unavoidable, label that experiment *scaffolded*, not autonomous.

## Pairing and independence
Freeze ≥32 independent source world seeds and ≥32 never-seen held-out world seeds in a manifest with hashes before training/evaluation. For each seed, share initial food, terrain, organisms, predator locations, resources and common keyed stochastic draws across arms where physically valid. Record every failure/extinction. The independent unit is a world seed, NOT each organism, tick, or descendant. Training and final held-out seeds must not overlap. Freeze source-trained weights/genomes before any held-out world execution. No outcome-driven retuning or seed fishing.

## Operational definitions
Novelty = first visit to a location by a given individual, tracked only from its own sensory/memory history; prevent leakage of unseen map or future state. Novelty bonus is bounded and charged against an equal explicit reward/training budget; zero bonus in control B, with all other settings matched. Record reward magnitudes and energy costs separately. Predators must move, detect, chase and capture under native physical rules. Recurrent memory persists only within an individual lifetime; newborn initialization must be explicit and identical across arms. Do not conflate learned weights with inherited genotype.

## Measurements
Primary: paired difference A−B in probability of world-level population persistence at frozen horizon.
Secondary: restricted mean time to extinction, mean population AUC, physical predator captures per organism-tick, unique accessible cells visited per energy spent, food gained, distance, energy ledger residual, reward budget and diversity of surviving lineages.
Retention: evaluate frozen learned policy in new held-out worlds with novelty bonus OFF and training disabled, compared with its own pretraining policy and arm B; also causal hidden-state reset and shuffled cue controls. Novelty persistence alone is not evidence of learning.

## Statistical gates (to freeze before any run)
Primary favorable claim requires paired bootstrap 95% CI for world-level survival difference A−B entirely above zero AND improvement ≥10 percentage points; report exact numerator/denominator and interval. Correct multiple secondary comparisons or mark exploratory. A result with higher exploration but no survival advantage is NOT positive. Reproduce the same directional primary effect in an independent second held-out cohort without retuning. Any failed integrity check blocks scientific interpretation.

## Integrity and stop conditions
1. Historical predator and memory branch tests PASS separately.
2. Integration must preserve original predator physics and use native legal motor decisions, energy, food and births; no hidden scripted policy advantages.
3. Check exact mass/energy accounting and paired initial-state hashes, determinism and source/test isolation.
4. Freeze complete executable protocol, code, all hyperparameters, seeds, duration and sample size before outcomes.
5. Execute source training, commit full-precision frozen learned weights, then run held-out campaign with immutable logs and all nulls.
6. If memory and predator engines cannot be integrated without changing original laws, STOP and report incompatibility; do not claim a completed experiment.

## Current verified status
Existing PR #49: real predators, 180 worlds, physical feasibility, no mutation benefit demonstrated.
Existing PR #50: 1728 causal-transplant worlds, negative mutation advantage.
Existing PR #51: reward-trained RNN corridor experiment source stage in progress; NOT an autonomous ecological memory + predator experiment.
**No M5.6 simulations have been launched.**
