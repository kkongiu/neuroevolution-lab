# M3 — Unpredictable novelty, recurrence and causal state ablations

**Preregistration:** this file was committed before the first full M3 GitHub Actions campaign. The values below must not be tuned after reading sealed test results.

## Research question

M2B established that a controller can react to neural-state observations without outperforming a controller that sees only task index. Would neural-state awareness become useful in a world with unpredictable transitions, old tasks returning and genuinely new tasks appearing?

This is a constrained numerical controller trained by **external evolution**, not a neural network that rewrites its code or discovers independent interests.

## World

A bank of 2D supervised binary classification tasks is generated independently per seed. At each new task boundary, a local pseudorandom stream chooses either:

- **Novel:** instantiate a previously unseen latent classification task, or
- **Return:** resample examples from **exactly the same latent classification rule** as a previous task. New training/test points are drawn; stored answers aren't replayed.

The first two tasks are novel; thereafter each task has independent 48% return probability. There is no fixed alternation by task family. Neither novelty/return status nor task ID is accessible to the controller; they are recorded for post-hoc stratified analysis only.

The *meta-training world* and *regular test* use line/XOR/circle/stripe task families with randomized parameters and randomized order. The independent *shifted-world test* adds two previously unseen families, checker and wedge, replacing XOR/circle. The world still shares 2D inputs and two families, so the shifted test is **a partial distribution shift**, not wholly new intelligence.

## Locked settings

- Meta-training **4 seeds**: 8000–8003.
- Meta-validation **3 seeds**: 9000–9002.
- Independent regular final test **12 seeds**: 10000–10011.
- Independent shifted-family final test **12 seeds**: 11000–11011.
- 10 tasks per seed; 48 SGD steps × batch 2 per task; MLP with 16 ReLU units; learning rate 0.04.
- Within-task held-out classification accuracy evaluated at steps 0, 12, 24, 36, 48. No validation labels used for the controller's decision.
- Meta-search: two independent populations of 8 controllers for 5 generations; controller of five numerical coefficients and 0 / half / max action grammar.
- Renew at most 20% of units at a boundary (3 units); exact fixed total quota round(0.45 × 3 × 9) = **12** renewed units per sequence.
- Fitness during meta-training and meta-validation = mean AUC − 0.05 × aggregate forgetting. All quota-matched controllers have exactly equal unit-renewal totals.
- Selection on meta-validation chooses one controller among generation leaders. All controls and test splits remain sealed until that selection is complete.

## Competitors

**Primary:** the state-aware adaptive controller versus an **independently evolved** time-only policy with the same quota, network and compute for SGD. The controller sees current-task prediction errors on TRAINING points, previous-task training error, recent unit activity and elapsed index. The time-only controller sees only elapsed index.

Additional controls: (a) same adaptive policy but ablate state inputs to zero at test time (state_blind); (b) uniform quota schedule; (c) permuted adaptive quota schedule; (d) same adaptive schedule with randomly selected units instead of low-activity units; (e) no reset; (f) max reset every boundary; (g) fully fresh network per task.

For the first six arms, renewal is EXACTLY 12 units per seed. The no-reset, max-reset and fresh arms have different resource use and are **diagnostics, not dose-matched controls**.

After selecting the policy, compare it against time-only both on **regular** and **shifted-family** test sequences. The **primary independent comparison is on the shifted-family test**; the regular test is secondary. Report paired seed-level bootstrap 95% intervals and signs, not per-example pseudo-replicates.

## Causal/behavioral controls

- Report how often removing state inputs *changes the policy's requested action*; forced actions necessary to spend the quota must be reported separately.
- Compare the full controller with itself with its state inputs blinded, AND with an independently trained time-only controller. These answer related but different questions.
- Report average learning-curve AUC **separately for novel and return events**, not only the aggregate.
- Same latent tasks and training examples for all treatments within each seed; different seeds across meta-training/validation/final evaluation.
- Compare retention/forgetting separately from learning speed.
- Report the number of selected units and CPU overhead limitation. Same unit budget and SGD steps are NOT identical total CPU cost.

## What counts as success / failure

Evidence of meaningful state-dependent adaptation requires:
1. a **positive** paired shift-test adaptive-minus-time-only AUC difference;
2. nonzero change in requested actions when state observations are ablated;
3. supporting evidence that the advantage is not explained by uniform or permuted quota schedules or is not confined to the regular world;
4. no large unacknowledged retention penalty.

If the primary comparison does not improve, **report a null/negative result** even if some other baseline loses. Merely selecting a different renewal schedule is not independent goal creation; evolution optimizes experimenter-defined accuracy.

No retroactive tuning of seeds, task families, quota, weights, or primary metric. Further research must preregister a new protocol and independent test seeds.
