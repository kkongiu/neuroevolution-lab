# M5.16 — Continual learning school: explicit context-stratified memory across A → B → A

**2026-11-22. Status: completed exploratory development experiment, strong positive result for *engineered context-stratified autobiographical replay*.** The result does **not** demonstrate autonomous task identification, unbounded life-long learning, emergent memory allocation, or compositional transfer. All development screening gates passed, but independent replication has **not** occurred and **evolution is explicitly blocked**.

## Motivation

M5.15 demonstrated reliable reward prediction but historical replay gave no improvement over repeating the current sample on stationary XOR and three-action bandits. M5.16 asks whether replay becomes specifically necessary when one learner must learn *conflicting rules sequentially* instead of optimizing a single stationary mapping.

## Task design / fairness

Exactly 24 independent development seeds **3900–3923**, disjoint from previous M5.9–M5.15 cohorts and protected M5.6 SOURCE/HOLDOUT/REPLICATION cohorts. Same initial 16-hidden-unit fully connected tanh reward predictor for all five arms, same learning-rate 0.075, 2 candidate actions, same per-phase cue sequence and data budget; only learning-update replay sampling and (in one negative control) received feedback differ.

Every block of 40 decision opportunities contains each of four binary pairs (x,y) exactly ten times.

- **A**: 640 lived episodes; rewarded if action = x XOR y. Then 80-decision frozen **A-after-A** exam.
- **B before learning**: 80-decision frozen B zero-shot exam.
- **B**: 800 lived episodes; rewarded if action = 1 XOR x XOR y, the opposite response for each cue pair. Then 80-decision frozen **A-after-B** exam **BEFORE any return-to-A learning** and 80-decision frozen **B-after-B** exam.
- **A return**: 240 lived episodes of initial A rule, then independent 80-decision frozen **A-after-return** and **B-after-return** examinations.

Total lived learning episodes = **1,680 per arm/seed**, all exams occur WITHOUT changing learner weights, episodic memory, update count or action-choice exploration state. Feedback in training is only the actual success/failure of the chosen action; exact correct action is known only to the environment. Every one of the five arms receives sensory **context identity 0/1**; this is an explicitly engineered task cue, not autonomous context inference. Exams reshuffle the same four cue patterns, hence this is NOT out-of-distribution novelty or general learning of arbitrarily many tasks.

Five arms:

1. `online`: 1 weight update from each actually experienced decision, 1,680 parameter updates total.
2. `recent_repeat`: same newest event repeated 5 updates per actual decision, **8,400 updates**.
3. `fifo_replay`: 1 newest update plus 4 samples from FIFO of latest **128** actually experienced (x,y,context,action,outcome) records, 8,400 updates. After 800 B decisions **zero** A records remain.
4. `context_replay`: 1 newest update plus 4 samples from an engineered 128-record memory divided into 64 A and 64 B slots (sample a non-empty context uniformly then a record within it), 8,400 updates. After B, 64 A records remain. **No extra memory capacity or parameter update budget** versus FIFO; sampling distribution and deliberate context-based allocation are the treatment.
5. `uncoupled_context_replay`: same stratified replay and update count, but feedback is replaced by an independent random 0/1 bit. Negative learning control, not a reward-marginal matched permutation.

Random streams for decisions, replays and false feedback are independent per seed. Both replay arms store up to **128 TOTAL** event records: there is no duplicated hidden FIFO in the context arm. The policy is the M5.15 action-conditioned 16-tanh-unit outcome predictor, not the original M5.6 recurrent learner; the extension is a diagnostic classroom, **not a deployed ecological organism**.

## Verified exploratory run

GitHub Actions [run #45037512367](https://github.com/kkongiu/neuroevolution-lab/actions/runs/45037512367) passed 5 new scientific-invariant tests, 5 original M5.6 regressions, and completed **24 organism seeds × 5 arms = 120 independent training trajectories**, archiving raw per-arm observations as `m5-16-continual-school-development` (30 days). Raw-row SHA-256 is **`81042f07363cf906ed5231ef419da6bebe8b1c0cbe6fd25154cd08543bc8b5fa`**. The experiment code was subsequently hardened to ensure that passing an exploratory screening gate does not unlock independent validation or evolution; the numerical raw rows are unchanged.

### Frozen exam results

| Arm | A after initial A | B zero-shot (before B teaching) | **A immediately after B** | B after B | A after A retraining | B after A retraining |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Online only | 100.00% | 2.08% | 33.33% | 94.79% | 97.92% | 84.38% |
| Repeat latest 5× | 100.00% | 0.00% | 27.08% | 100.00% | 100.00% | 100.00% |
| FIFO replay 128 | 100.00% | 0.00% | 84.38% | 100.00% | 100.00% | 100.00% |
| **Context-stratified replay 64+64** | **100.00%** | 0.00% | **100.00%** | **100.00%** | **100.00%** | **100.00%** |
| Context replay, randomized feedback | 48.96% | 50.00% | 50.00% | 53.13% | 45.83% | 52.08% |

All numbers are means over 24 paired development seeds on 80 frozen trials per indicated exam.

### Causal interpretation within this benchmark

- Historical **context-aware** replay preserves A throughout 800 conflicting B experiences: A frozen after B **100%**, versus **84.38%** for equal-update, equal-capacity FIFO (+15.625 pp) and **27.08%** for equal-update most-recent repetition (+72.917 pp). This supports the effect of **which past experiences are preserved** (and sampled), over merely increasing training steps.
- Context-stratified replay also retains B after the A re-learning phase at 100%, but **FIFO and repeat also do 100%**; we cannot claim uniqueness on every recall criterion.
- The B zero-shot examination is at or close to **0%** for learners of A, because B has the *exactly opposite answer map*. The later 100% B indicates learning the new context/rule, not trivial preexisting success on B.
- The independent-feedback control is around chance, supporting dependence on real action-outcome feedback.
- The contrast is **engineered explicit two-task allocation**, not spontaneous identification of which old memories matter. Protecting one 64-sample memory partition per context is an externally supplied heuristic; this design cannot scale unmodified to unbounded novel contexts.
- Despite all 7 development screens passing, this is a **development experiment on only four binary cues per context**, not a sealed independent test or a general demonstration of open-ended neural intelligence.

## Predeclared exploratory development gate

| Condition | Result |
| --- | --- |
| Stratified A learned before B >= 85% | **PASS** 100% |
| Stratified B learned before returning A >= 85% | **PASS** 100% |
| Stratified A immediately after B >= 75% | **PASS** 100% |
| Stratified A immediately after B >= FIFO +10 pp | **PASS** +15.625 pp |
| Stratified A immediately after B >= repeat +10 pp | **PASS** +72.917 pp |
| Stratified B still >=75% after A relearning | **PASS** 100% |
| Stratified A-after-B > random feedback group +10 pp | **PASS** +50 pp |

**7/7 exploration gates pass**. Machine outputs explicitly say `development_screen_pass=true`, `independent_replication_complete=false`, `evolution_authorized=false`, `advance_to_evolution=false`.

## Necessary next tests (before evolution)

1. Re-run a strictly predeclared, **independent new seed/task-family replication** without reusing 3900..3923 for model selection.
2. Change the number of task contexts from 2 to **3, 5, 10**, with an honest fixed 128-record memory budget. Remove the externally fixed 64+64 partition, compare uniform FIFO to task-blind reservoir / surprise/utility retention. Document forgetting versus memory contention.
3. Test **hidden context changes** (no explicit A/B label): learner must infer the regime from experienced outcomes before replay can be directed.
4. Test noisy feedback, at least 3 possible actions, continuous cues and truly unseen combinations, not just reshuffles of four patterns.
5. Integrate prediction learning / replay into native M5.6 ecology **only if** new independent tasks pass, paying real memory/update energy costs and preserving the genotype versus individual learned state distinction.

The laboratory school can be a useful development environment without replacing ecological validation. Original sealed world cohorts and `main` remain untouched. This PR is stacked on M5.15 and is a **draft**, not a merge request to run evolution.
