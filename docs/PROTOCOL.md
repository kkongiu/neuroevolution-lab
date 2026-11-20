# M1 experimental protocol — preregistered design

## Research question

Under an equal number of candidate evaluations and identical initial genomes, does allowing restricted neural-structure mutations improve generalization in a reversing binary-cue task compared with numeric-weight mutations only?

## Treatments

- **Fixed:** recurrent network with immutable connection masks and active-unit flags; mutable numeric weights.
- **Topology:** same initial population, same numeric mutation operators, but some mutations toggle connections or active units.
- All networks use the same maximum hidden width and the same evaluation budget. A topology mutation *replaces* a weight mutation attempt, so the two arms do **not** have an equal count of numeric weight updates. Report this caveat.

## World

Binary cue, initially hidden binary regime, deterministic mapping from cue and regime to reward. Regime reverses every specified interval. Inputs provide current cue and previous cue/action/reward. Neural state is reset at the beginning of each episode.

## Measurements

Primary: held-out fraction of correct actions from the top candidate chosen by **training fitness**. Secondary: fraction correct in the first four steps after a regime reversal, and active network connections. Fixed and topology arms share training and holdout seeds.

Holdout is only observed and logged, not used by the algorithm to choose offspring. Nevertheless, repeated human tuning after looking at holdout scores can leak information: reserve further independent seeds for final confirmation.

## Controls and limitations

- Deterministic RNG; config and RNG state included in a checkpoint.
- Independently compare multiple paired seed runs, not only seed 42.
- Keep compute and maximum network width equal. Report active-edge counts because inference work may differ.
- Compare *accuracy*, not a composite fitness rewarding sparsity, to avoid structurally favoring one condition.
- No proof of statistical significance without repetitions and uncertainty intervals.
- Reversing binary-cue task is a **proxy for adaptation**, not a test of autonomous goals.
- No arbitrary Python/code modification by agents. All mutation operators are closed and inspectable.
- CI smoke runs verify plumbing, not scientific success.

## Falsification

A null or negative outcome (topology no better or worse on independent holdout seeds) is an admissible result. A difference on training seeds alone is not evidence of generalized improvement.

## M2 gate

Only after M1 is replicated should we introduce inheritable *learning-rule* variation, with a third arm and matched budgets. Do not change M1 rules retroactively.
