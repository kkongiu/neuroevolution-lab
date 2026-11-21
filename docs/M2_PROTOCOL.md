# M2 — Learn *when* and *how much* to renew a network: frozen pilot protocol

## Falsifiable research hypothesis

Can an **evolved bounded controller** decide **when** and **how many** hidden units to renew in a continually trained network, gaining better unseen-task learning AUC than simple fixed schedules and *dose-matched* alternatives?

**This is NOT autonomous source-code rewriting or independent goal formation.** The controller is a short vector of five numeric weights selected by an externally specified evolutionary algorithm. It chooses one of three closed, auditable actions: **0 units**, **half of the permitted maximum**, or **the permitted maximum**. Renewal always targets units with low activity on the *previous task's training examples*. It cannot alter which program executes.

## Data separation

Three **disjoint seed ranges**:

- Meta-training sequences: **1000–1003** (used to evolve coefficients; these evaluation samples are deliberately part of meta-training).
- Meta-validation sequences: **2000–2002** (select one winning policy from the top candidate of each generation).
- Sealed final test sequences: **3000–3015** (never used to modify or select the controller).

Each seed produces **8 tasks** in fixed cycling families line, XOR, circle, stripe, with independent task parameters and examples. **The final test uses new seeds but the same task families**; it cannot establish out-of-distribution generalization.

At each task transition the controller only observes:
1. a constant bias;
2. accuracy error on up to 32 examples drawn from the **new task's training stream**;
3. accuracy error on the previous task's training stream;
4. fraction of almost-inactive units measured on previous-task training inputs;
5. progress through the task sequence.

It never sees final test performance, held-out labels, or future tasks while choosing an action. Meta-evolution *does* optimize a fitness calculated on separate evaluation examples from the **meta-training** population, which is legitimate optimization rather than final evidence.

## Neural network and treatments

Reuse the M1-P ReLU MLP, SGD code and task generator unchanged.

- Width **16**, 128 training samples per task, **64 SGD updates** per task, batch size 2.
- Learning rate 0.04; evaluation size 48; measured at steps 0/16/32/48/64.
- At most **20% of hidden units renewed at each transition** (maximum three units with width 16).
- Meta-search: population **8**, **5 generations**, numeric coefficient mutation, deterministic search seed **911**.
- Policy fitness on meta-train:
  **mean within-task held-out learning AUC over tasks 1–7**
  minus **0.025 × fraction of allowed reset budget used**
  minus **0.05 × mean retention loss**.
- Meta-validation selects among generation winners by this same fixed fitness formula.
- **All constants fixed before inspecting final test performance.**

On sealed test seeds we compare six policies:

1. **learned:** policy chooses whether to reset and number of low-activity units;
2. **continual:** no resets;
3. **fixed_low_activity:** always reset 20% of low-activity units;
4. **random_matched:** reset the **same count on the same transition** as the learned controller, but choose random units;
5. **shuffled_matched:** reset the **same total counts** as learned, but shuffle the timing across transitions, keeping low-activity selection;
6. **fresh:** new random weights for each task (trainability control, not comparable reset cost).

All arms begin with the same initial network on the first task, use the same training and evaluation data for each seed/task, and receive identical SGD budgets. The random/shuffled controls must have exactly the same total number of renewed units per seed as the learned arm. No GPU or third-party libraries needed.

## Outcomes

**Primary:** paired AUC difference of learned versus **shuffled_matched** over independent final-test seed sequences. This is the strongest control on whether *timing* matters, independent of total dose.

Secondary:
- learned versus fixed low-activity / continual / random-matched / fresh;
- mean held-out final accuracy;
- mean retained ability on old tasks (forgetting);
- reset count, frequencies of zero/half/max decisions;
- cost-penalized fitness as supplementary metric only (not primary).
- intervals: paired seed-level bootstrap percentile 95%; sign counts.

Do not count individual tasks as independent replicates. When the learned action sequence is constant, shuffled timing cannot differ: report that honestly. A **static policy** is not evidence that the controller learned when to intervene, even if its score is high.

## Failure and safety criteria

- If learned does not beat shuffled-matched on final seeds, *no demonstrated adaptive timing advantage*.
- If policy always chooses one dose, *no demonstrated state-dependent control*.
- If gain exists only relative to continual but not to fixed, the complexity is not justified.
- If AUC increases while forgetting becomes substantially worse, report the trade-off.
- If a policy optimizes the current task only by destroying knowledge of old tasks, its fitness cannot be called a net improvement.

The controller cannot edit code, invoke shell/network, read secrets, push commits, or alter the reward. The workflow only launches a fixed Python command with read-only repository permissions. Training data and final holdout are not accessible via runtime actions outside these fixed procedures.

## Next experimental gate

An observed gain would motivate **new task families, longer sequences, separate compute profiling, and fresh final test seeds**. A null result would motivate inspecting whether the tiny feature set makes the optimal action effectively static, before adding more expressive meta-learning.
