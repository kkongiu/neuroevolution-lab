# M3.2 — Natural sequential interference and bounded self-diagnostic renewal

**Preregistered before implementation or final-test evaluation.** All M1–M3.1 results stay frozen.

## Hypothesis

Without artificially corrupting weights, can a bounded controller that sees a persistent network's *training-only* diagnostics schedule neuron renewal better than an independently optimized clock-only controller using the **same intervention budget**? Can it do so without unacceptable forgetting? A null or negative result is valid.

## World and intervention

A single small ReLU network is trained sequentially on 12 supervised binary 2D classification tasks. Each seed has its own bank of latent tasks: the first two are new, then subsequent tasks repeat a past rule (45% probability, fresh examples) or instantiate a new rule. No sign flips or direct damage to learned weights. Different tasks share the same input coordinates; some have contradictory labels and perfect retention may be impossible *without a task cue*. This unavoidable limitation is reported explicitly.

Regular task families: line, XOR, circle, stripe. Final shifted-world families: line, checker, wedge, stripe. Checker/wedge are never used in meta-training. The controller does not see task identities, novelty flags, future data or held-out evaluation labels.

At each boundary choose between **preserve** and **renew the three lowest-activation units on the previous task's TRAINING inputs**. Each competitive controller must use **exactly four renewals across eleven boundaries**, with forced quota completions separately recorded. Each renewal reinitializes three hidden units using the existing network's initialization distribution. No operator is invented by the network.

## Frozen numbers and partitions

- Meta-training: seeds **16000–16005** (6 sequences); validation: **17000–17002** (3).
- Main sealed test: **18000–18019** (20); shifted sealed test: **19000–19011** (12). No overlap.
- 12 tasks; 48 SGD updates/task, batch size 2; 16 hidden ReLU units; learning rate 0.04.
- 64 held-out examples/task, evaluations at SGD steps 0/12/24/36/48. Distinct deterministic training and evaluation RNG streams.
- Six policy features: constant 1, current task TRAINING accuracy, previous task TRAINING accuracy, cosine of current/previous OUTPUT-LAYER training gradients, fraction near-inactive hidden units on previous task training, elapsed task fraction.
- Three independently optimized 6-parameter linear threshold policies: **state-aware** (all six inputs), **accuracy-only** (bias/current accuracy/time), **time-only** (bias/time). 8 candidates × 4 generations per policy, external numeric mutations seeded at 931.
- Meta-optimization objective: mean learning-curve heldout AUC on tasks 1–11 minus 0.08 × mean past-task forgetting. Meta-validation chooses the winning generation champion. No final-test score may influence these choices.

## Final arms

All see identical tasks/examples and initial weights within a seed:
1. full state-aware, 4 renewal actions;
2. independently evolved accuracy-only, 4;
3. independently evolved time-only, 4;
4. uniformly spread 4 actions;
5. same four full-state renewal actions permuted across task boundaries;
6. same four boundaries, renew random rather than least-active units;
7. no renewal (not budget matched);
8. renew every boundary (not budget matched).

Equal *SGD steps* and *renewal counts* do **not** prove identical overall CPU cost: instrumentation and decision features add overhead.

## Counterfactual measurement

On each **main/shifted final-test** task boundary for the full-state arm, clone its pre-decision network twice, then run identical task samples and SGD steps under preserve and renew. Compare one-step AUC and old-task retention. Commit only the policy-chosen branch to its persistent lifelong trajectory. The hidden counterfactual oracle is calculated ONLY for post-hoc evaluation; no policy sees it, and the two actions' potential outcomes do not influence meta-training/selection. This estimates a **local one-step causal intervention effect**, NOT the effect of an entirely different lifetime trajectory.

Diagnostic local value = current-task learning AUC minus 0.08 times forgetting on earlier task holdouts. Report its oracle regret, state sensitivity and whether renewal helped or harmed. The task and next-step outcome depend on training state, not an injected failure bit.

## Locked interpretation gate

**Primary:** on 20 independent main-test seeds, mean paired AUC(state-aware − independent time-only). Bootstrap descriptive percentile 95% across seeds; report sign counts. Strong evidence for useful internal representations additionally requires:
- positive mean adaptive − **independent accuracy-only** (otherwise state features beyond accuracy do not help);
- state inputs affect **unforced requested choices**;
- no concealed retention penalty;
- supporting comparisons with shuffled and uniform matched budgets.

Secondary: shifted test on new task families; AUC by returning/novel tasks; old-task forgetting, controlled unit-renewal costs; locally paired preserve/renew effects. Report nulls even if a secondary number looks positive. Never tune the frozen settings after seeing the sealed test.

## Safety and scope

This remains synthetic 2D supervised continual learning with **externally evolved numeric scheduling rules**. Neither autonomously modifying code nor self-generated goals nor consciousness is in scope. Repository permissions remain read-only for all GitHub Actions campaigns.
