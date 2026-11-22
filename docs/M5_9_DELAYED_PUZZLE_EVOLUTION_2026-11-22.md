# M5.9b — Delayed three-gate puzzle evolution: development result (2026-11-22)

**Status:** implemented and run successfully, **negative/inconclusive** for inherited capacity to learn or generalize; abstract task arena, NOT an embodied M5.8 reproduction experiment.

## Purpose

Improve on M5.9's overly easy one-action context task by requiring a three-action sequence. An organism sees the local stage/cue/context and must choose one of two actions per gate; the hidden rule remains opaque. No reward or error annotation is provided until the end of all three moves. Correctly completing all three is 1/8 = 12.5% for independent random choices. There are 12 attempts within an episode, with private terminal feedback on each completed attempt; the 260-synapse M5.6 recurrent controller receives terminal reward (+1 correct, −0.15 failure) for the prior 3 action caches, never correct-action labels.

An episode is counted as solved when at least 4 of 12 full sequences succeed. Under independent Bernoulli(1/8) trials this is about 5% by chance; outcomes in a stateful neural controller need not be independent. This threshold must not be interpreted as proof of rule understanding.

## True genotype separation

Only the founder's **original initial 260 neural weights**, mutated on reproduction with bounded Gaussian noise, enter the descendant. Private learned neural weights, recurrent hidden state, learned predictor weights, and episodic histories are not copied. Three matched selection arms, all exactly eight offspring slots per generation:

- `solver_plastic`: choose parents weighted by 1 + solved episodes; neural synapses can learn
- `solver_frozen`: same solution-weighted parent choice; within-life neural updates disabled
- `yoked_random_plastic`: parent choice uniform, learning enabled; identical population / reproduction slot budgets

12 matched initial evolutionary seeds 3100–3111, 8 generations, 8 organisms, 3 rule-changing training episodes per organism/generation × 12 sequential attempts. New tasks use disjoint development seeds 3150–3161, 5 novel episodes each. Note same underlying task family: altered random rules/cues, not fundamentally novel cognitive problems.

## Actual outcomes

[Successful run #45023827867](https://github.com/kkongiu/neuroevolution-lab/actions/runs/45023827867) passed all 4 new invariants tests plus all 5 historical M5.9 memory-inheritance tests, then ran all **36 evolving lineages** and archived complete histories in the `m5-9-delayed-three-gate-evolution` workflow artifact.

| Arm | Mean entire three-gate solution rate (fresh tasks) | Episodes solved per descendant (out of 5) | Test late − early | Fresh-task success with plasticity disabled |
|---|---:|---:|---:|---:|
| Solver selection + plastic | 12.2743% | 0.2604 | +0.8681 pp | 11.9271% |
| Solver selection + frozen | 12.4826% | 0.3125 | −0.2431 pp | 12.4826% |
| Random parent + plastic | 12.7604% | 0.28125 | +0.2431 pp | 12.4826% |
| Independent random actions | 12.5% | n/a | 0 expectation | 12.5% |

All are near random baseline. **The solver-selection / plastic group is weaker than randomly selected plastic offspring, by 0.4861 percentage points**, on the novel tasks. Small positive within-test learning (+0.87 pp) is not enough to demonstrate transfer or evolved learning-to-learn; may be stochastic noise or rule/cue distribution effects. There is **no confirmed emergence of cognitive adaptation**, despite the verified genetics mechanics.

Full raw-row digest: `fd015df928b7bb8085fdb09f7d2a378ecf373dc6b2f5a374b97dbd718ffbe286`.

## Why the negative result is informative

The three-step delayed terminal reward raises the temporal credit-assignment problem, but it also creates a very sparse success signal. The original M5.6 controller uses one-step truncated policy gradients. Updating cached actions after a terminal feedback is a basic *delayed REINFORCE* approximation, not a robust learnable world model, nor an explicit uncertainty-aware planner. The hidden rule may change between episodes, so global genetic improvement might be masked by task unpredictability. Fixed short budgets and selection pressure of 1+solves are not guaranteed to be enough for evolution. More generations alone risks fitting this simple puzzle family and should not be represented as a scientific cure.

## Next recommended controlled experiment

1. **Prove individual learning can solve the task before selecting for it**. Establish a positive trainable baseline with sufficient repeated feedback and comparable no-learning, shuffled-feedback and oracle *diagnostic* controls (oracle strictly outside organism) on development problems. If no learning, selecting for learned competence is uninformative.
2. Use an explicit small recurrent model with honest counterfactual evaluation, or a temporally precise policy gradient/eligibility trace. Record exact prediction calibration, time-to-first-solve, and per-episode improvement, not only terminal survival.
3. Only when individual task learning is reproducible: perform multi-generation selection, randomized parent-match/yoked reward controls, and *true genotype transfer* to disjoint task families; keep full source/holdout/replication gates sealed.
4. Physical integration with energy-costed reproduction in M5.8 only after cognitive positive controls pass. No genetic memory inheritance.

The experiment is on a draft stacked branch, not `main`. No original M5.6 sealed cohort consumed.
