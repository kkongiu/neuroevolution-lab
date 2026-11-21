# M2B — Matched-budget renewal timing (new preregistered experiment)

## Background

The [M2 first test](../reports/M2_adaptive_2026-11-21.md) selected a trivial **never-reset** policy in 112/112 opportunities. That is consistent with the objective, which penalized renewal count and forgetting. As no units were ever renewed, its timing comparison became vacuous. The negative result is preserved without changing M2 code or data.

**M2B changes the scientific question.** At *equal total renewal budget*, can a controller that sees current and prior training signals decide **when** to renew more effectively than a controller that only sees task index, uniform timing, or a shuffled schedule?

## Frozen experimental specification (before new test)

- Meta-training seeds: **4000–4003**.
- Meta-validation seeds: **5000–5002** (select the champion among generation winners).
- Independent sealed test seeds: **6000–6015**.
- Same task families and neural architecture as M2, 8 sequential tasks; batch size 2; 64 SGD steps/task; 16 hidden ReLU units, learning rate 0.04, 48 held-out examples/task.
- 5 meta-generations, 8 policy candidates/generation for each of **two separately evolved controllers**.
- Both controllers can request 0, half, or maximum number of neuron renewals per boundary. Maximum = **20%** of the network per task (3 units here).
- **Exactly half of the maximum total reset budget must be spent across 7 transitions:** round(0.5 × 3 × 7) = **10 units per complete sequence**. The controller cannot spend beyond the budget or finish with unused budget.
- Online feasibility limits each action; at a late transition the controller may be *forced* to renew units to exhaust the quota. **Requested** choices and **forced** choices are reported separately.
- Full controller features: task index, new task initial error on TRAINING examples, previous task error on TRAINING examples, previous-task hidden-unit activity from TRAINING examples.
- **Time-only** controller: train/evolve the exact same policy architecture and budget but zero out every state/learning feature. It sees only the task index.
- Selection fitness on train and validation seeds = average held-out learning-curve AUC − 0.05 × retention loss. No reset-count penalty, since budget is identical by design. The same evaluation family used for meta-training is **not** counted as final independent evidence.

## Final test arms (same seed, tasks, train and evaluation streams)

1. **adaptive:** selected full controller, budget exactly 10 units.
2. **time_only:** independently evolved time-only policy, budget exactly 10.
3. **uniform:** even distribution of exactly 10 low-activity renewals over the seven boundaries.
4. **permuted:** shuffle the adaptive controller's actual reset counts across boundaries (same total dose), renew low-activity units.
5. **random_units:** copy adaptive **actual** reset count at every boundary but select renewed hidden units randomly rather than by low activity.
6. **no_reset:** zero units (outside dose-matched group).
7. **fixed_max:** renew 3 units at every transition (21 total; outside dose-matched group).
8. **fresh:** reset the entire network at each task; diagnostic reference, not dose matched.

All arms start from identical weights on task 0, have equal SGD updates and examples per task. *The reset counts are equal, but CPU/inference energy costs are not proven equal*: evaluating state features, selecting units and evolving controllers adds overhead. Record that as a limitation.

## Prespecified outcomes

**Primary:** paired per-seed mean held-out within-task learning AUC **adaptive − time_only** (tasks 1–7), with seed-level descriptive bootstrap 95% interval. This isolates information about current training state beyond a task-index schedule.

**Secondary:** adaptive vs uniform/permuted/random_units/no_reset/fixed_max/fresh, forgetting, mean final accuracy, per-task reset histogram, requested vs actual renewal counts, total number of forced decisions, number of requested choices that would differ if state features were blinded.

**Interpretation gate:** Only a positive primary comparison *and* evidence that state observations change requested actions would support state-dependent adaptation. A gain only over no-reset does not count. If the selected controller uses a constant time schedule, the scientific conclusion is negative or unresolved, even if average AUC is higher.

## No retroactive tuning

The seed sets, quota, fitness weights, architecture, number of meta-generations and evaluation metrics are frozen **before** GitHub Actions runs. Any further alteration after seeing final test results must use a new versioned protocol and entirely new test seeds.

No arbitrary source modification, unbounded code execution, external data acquisition, network access, or repository writes are permitted from within the agents. A controller can request only how many low-activity units to replace within an auditable range.
