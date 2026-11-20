# M1-P — Continual learning and loss of plasticity: preregistered pilot

## Question

Can a neural network trained on a sequence of tasks become **slower at learning a new task** than a fresh network with identical capacity? Does renewing low-activity units help, beyond the effect of resetting the same number of random units?

This is a **separate experiment from M1**. M1 evolved neural controllers between generations and did **not** train their weights during the agent's lifetime. Here, within-lifetime SGD is the essential intervention. This pilot does **not** create a self-rewriting neural network or autonomous objectives.

## Frozen treatments before first full campaign

All arms use a one-hidden-layer ReLU MLP with identical width, logistic binary output, hand-coded SGD, identical learning rate and number of gradient steps **per task**. No libraries or downloads required.

1. **Continual:** retains all weights across every task.
2. **Fresh:** starts from new randomly initialized weights at the beginning of each task; controls for the trainability of the task itself and initialization variability.
3. **Low-activity reset:** retains weights but resets 20% of units with lowest mean ReLU activation **on the previous task's training inputs** when switching tasks.
4. **Random reset:** retains weights but resets the same number of randomly chosen units, with the same weight-initialization distribution; controls for the mere effect of replacement.

All four arms are initialized **identically at the first task**. The same task specification and ordered training examples are used within each seed/task across treatments.

The environment alternates four synthetic binary classification task families in a fixed, preregistered order: rotated linear separator, XOR, off-center circle, and sinusoidal stripes. Each task changes parameters with a seed. All share the same 2D input space. The classification labels come directly from the environment (supervised training); no agent chooses an objective.

## Controls and measurement

For each task:
- A fresh training stream is generated deterministically.
- A **disjoint RNG stream** produces the held-out examples used for accuracy measurements. No holdout data is used in gradient updates or in renewal decisions.
- Accuracy is evaluated before training and every 32 SGD updates, including the final update.
- Within-task **accuracy area under the learning curve (AUC)** is the primary metric; it reflects learning speed and starting quality, not just final score.
- End-task accuracy, learning gain, fraction of completely inactive ReLU units, and retention loss on previously learned tasks are secondary.
- For each seed, we average metric values over tasks **after the first**, since task zero has identical initial networks.

Primary comparisons use paired differences in per-seed mean AUC:
- continual minus fresh;
- low-activity reset minus fresh;
- random reset minus fresh;
- low-activity reset minus random reset.

The uncertainty interval is the descriptive **paired bootstrap percentile interval**, resampling *seed-level* paired differences. Individual task observations and evaluation examples are **not** treated as independent scientific replications.

## First full campaign settings

- Seed range: 100–123 (24 independent seed sequences).
- 8 sequential tasks per seed.
- 128 SGD updates per task, batch size 2, network width 24.
- Learning rate 0.04; 96 held-out samples per task; renew 20% units per switch.
- **No tuning after viewing these results** without a separately reserved confirmation set.

All numbers are low-cost pilot settings and are **not** proven optimal for exposing loss of plasticity.

## Interpretation and falsification

The critical result is not a single winning score. We want to learn:
1. Whether *continual minus fresh* declines across tasks, or is consistently negative on later tasks. A negative gap is evidence of reduced trainability **in this task family**, not universally.
2. Whether low-activity renewal reduces the gap more than a **matched random-reset** control.
3. Whether any apparent improvement in learning speed destroys retention on old tasks. Forgetting is **not** identical to loss of plasticity.

A near-zero or opposite effect is an admissible **negative result**, not a failure of the engineering work.

## Limitations and safety

- Only 2-input feedforward ReLU networks and synthetic supervised tasks; conclusions cannot generalize to LLMs or biological brains.
- Reset selection is externally programmed, rather than a learned self-modification policy.
- The number of gradient updates is matched, but renewal adds a small computational overhead. Actual CPU time/energy should be benchmarked separately.
- Observing the holdout curves repeatedly can contaminate later experiment design; confirm any discovery on new task families and held-out seeds.
- This code never generates or executes Python source, opens network connections or edits repository files from model outputs.
