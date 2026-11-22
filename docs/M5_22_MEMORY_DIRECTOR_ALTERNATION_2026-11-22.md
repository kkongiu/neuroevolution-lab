# M5.22 — Two neural populations: an archive/director versus alternating plasticity

**2026-11-22. Status: completed negative development experiment. Both designs FAIL to solve the stability–plasticity dilemma (2/10 preregistered screens). Evolution stays blocked.**

## Question

The user proposed separating memorization from executive control between two neural populations, or alternating which population learns and which stabilizes. After M5.16–M5.21 failed to show learned continual plasticity without strong replay, we tested these distinct architectural ideas instead of yet another change to a global stiffness parameter.

This test cannot falsify modular learning or complementary learning systems generally. It tests two small specific architectures and their controls.

## Architecture and experimental firewall

New development-only seeds **4600–4623** (24; all previous development and protected M5.6 sealed cohorts excluded). Every arm shares exactly the same 154-parameter action-conditioned reward predictor: two 8-neuron tanh hidden expert populations, independent output heads, and an 8-parameter *trainable sigmoid gate* that combines the logits from the two populations. The gate depends on state and action; it learns **only from binary success/failure of the action actually chosen by the organism**, with no correct-action teacher labels or privileged feedback.

Every organism faces the same known-context grammar: **400 A (XOR), 400 B (inverse XOR), 400 C (copy x), 160 A, 160 B**, giving 1,520 real interactions. Every block of 40 observations is balanced over four (x,y) binary sensory patterns. **Context identity is explicitly supplied**, and frozen exams use new permutations of these same four patterns, *not novel task families*. All arms have exactly the same 128 autobiographical record slots (43 A + 43 B + 42 C) and initial founder weights within each seed. Frozen exam tests cannot change weights, buffers, policies or randomness.

Non-replay arms all execute **exactly 1,520 weight-update CALLS**, but the number of actually updated expert weights can differ because some are frozen. This is not identical parameter-level gradient exposure or computational cost. The replay-positive reference uses **7,600 weight-update calls** (five times as many, and is not compute-matched).

## Seven preregistered arms

1. **Joint:** both expert populations and learned gate update every step.
2. **Memory + director, self-triggered freeze:** both learn initially. Freeze group 0 when it has achieved >=90% real success over two consecutive 40-step blocks, after at least 200 experienced actions; group 1 and the learned gate keep updating. This stabilization trigger does NOT read task labels or training-phase boundaries.
3. **Externally scheduled freeze:** freeze group 0 precisely after the first 400 A training actions. This is an artificial timing/oracle control, not autonomous consolidation.
4. **Swapped scheduled freeze:** same timing, freeze group 1 instead.
5. **Alternation:** train both groups initially, then every 40 lived decisions switch which one is plastic; the gate is always plastic.
6. **Random-yoked alternation:** each later phase gives both groups exactly the same number of active update opportunities as deterministic alternation, but randomizes which group gets each 40-action window. Each group receives precisely 960 active calls in both alternation arms.
7. **Replay-positive reference:** joint training plus four actual past-experience sampled gradient updates per interaction, same 128-record memory capacity but 5× update calls.

The director is a **soft routing gate**, not a full executive recurrent controller. An eight-neuron frozen archive is not unlimited memory storage. The experiment does not test consolidation via distillation, external content-addressable memory or sleep-like replay.

## Actual verified results

[GitHub Actions #45054156263](https://github.com/kkongiu/neuroevolution-lab/actions/runs/45054156263): **success**, with 5 new scientific-invariant tests and 5 original M5.6 regressions all passing. **24 independent seeds × seven arms = 168 trajectories** of 1,520 lived actions each, complete raw per-organism frozen exam, gate, capacity and activity data archived as the Actions artifact **m5-22-dual-population-development** (30-day retention).

SHA256 of canonical raw-result rows: **2ebb04536c798a810c215045b72f130ef8de2ca61a7805d8200d23389685570a**.

### Frozen test accuracy (percent; 24-seed means)

| Arm | A immediately after initial A | A after B | A after C | B newly acquired | B after C | C newly acquired | Mean of A-after-C, B-new, B-after-C, C-new |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Both groups plastic | 93.75 | 33.33 | 47.92 | 85.42 | 55.21 | 100.00 | 72.14 |
| **Self-freezing memory + director** | 92.71 | 40.63 | 52.08 | 83.33 | 61.46 | 98.96 | 73.96 |
| Freeze first group after A (lab-scheduled) | 93.75 | 45.83 | 54.17 | 76.04 | 52.08 | 95.83 | 69.53 |
| Freeze other group after A (lab-scheduled) | 93.75 | 44.79 | 53.13 | 77.08 | 57.29 | 95.83 | 70.83 |
| **Alternating experts every 40 decisions** | 93.75 | 41.67 | 48.96 | 69.79 | 53.13 | 96.88 | 67.19 |
| Random expert, identical exposure | 93.75 | 51.04 | 56.25 | 75.00 | 54.17 | 95.83 | 70.31 |
| **Replay only, 5× training updates** | 100.00 | 97.92 | 96.88 | 92.71 | 96.88 | 97.92 | 96.09 |

All tests of old knowledge are performed **before any refresher teaching on the relevant old task**. Raw stats include gate mixture means and the number of actual gradient updates per expert.

### Important diagnostic 1: stabilizing an archive is not the same as retrieving it

The self-freezing archive eventually froze in **23/24 organisms (95.83%)**, but only **7/24 (29.17%) froze during the first A lesson**. Most archived something **after** learning had moved to B or C. Consequently the archive cannot be assumed to contain correct, isolated A knowledge.

However, this is not the only cause: the *laboratory-oracle* arm, which frozen group 0 precisely after A, retained A after C at only **54.17%**. The other expert and **the routing gate** continue to change, so old competence can still be unavailable even if one bank's weights are immutable. The hypothesis that gating/selection fails to maintain reliable recall is plausible, not independently causally proven by this experiment.

### Important diagnostic 2: simple alternation does not beat equal-exposure random scheduling

Strict 40-step alternation scores **67.19%** versus **70.31%** for the randomized-yoked schedule on the pre-refresher compound; **minus 3.13 percentage points**. Both methods have exactly **960 expert-active update calls per population**, counting both the initial fully plastic A phase and the alternated follow-up. Alternation of learning windows does **not** itself produce specialization or stable old knowledge.

### Positive reference does not establish a fair architectural comparison

Historical episodic replay of actually experienced action/outcome records obtains >90% on all four primary exams: A-after-C 96.88%, B-new 92.71%, B-after-C 96.88%, C-new 97.92%. But it uses **five times as many neural update calls**. Therefore the result shows this limited model *can* handle the fixed curriculum with more training; it is not proof of superior efficiency or unlimited lifelong learning.

### Matched-seed exploratory paired bootstrap (24 seeds, 1200 resamples)

Differences on compound of A-after-C, B newly acquired, B-after-C, C newly acquired:

| Pair | Mean difference | 95% paired bootstrap interval |
| --- | ---: | --- |
| Memory/director minus joint | +1.82 percentage points | [−0.26, +4.17] |
| Phase-bound archive minus joint | −2.60 pp | [−4.95, 0.00] |
| Strict alternation minus random-yoked | −3.13 pp | [−6.77, 0.00] |

The intervals are exploratory, not independent scientific replication. Memory/director does not demonstrate a positive advantage clearly separated from zero.

## Prespecified screening: 2/10 passed

**Memory/director:** A after C ≥90% FAIL (52.08%), B acquisition ≥90% FAIL (83.33%), B retention after C ≥90% FAIL (61.46%), C acquisition ≥90% PASS (98.96%), and +5pp versus joint on the compound FAIL (+1.82pp).

**Alternation:** A after C ≥90% FAIL (48.96%), B acquisition ≥90% FAIL (69.79%), B retention after C ≥90% FAIL (53.13%), C acquisition ≥90% PASS (96.88%), and +5pp versus exposure-matched random schedule FAIL (−3.13pp).

**Development screen: FAIL, 2/10.** Both memory and alternation separate gates are false. Run metadata explicitly reports independent_task_replication_complete=false, evolution_authorized=false and advance_to_evolution=false.

## What the result does and does not mean

This experiment rules out **these simple two-expert configurations as adequate solutions** to the project’s current 90% acquisition-and-retention target. It does **not** disprove the value of modular memory or alternating fast/slow computation in a different implementation.

Possible limitations:
- A frozen neural subnetwork stores parameters but has no guaranteed **retrieval**. Routing can drift while the bank remains unchanged.
- A single learned sigmoid gate with reward-only training may not separate the three conflicting tasks, and two experts may not provide enough capacity to assign one per problem.
- The success-based freeze rule may consolidate the wrong subject after a context switch. This was observed directly in the 7/24 initial-A freeze rate.
- Alternating plasticity on a schedule without actual transfer/consolidation of stable content is not equivalent to a true alternating teacher/student process.
- The laboratory supplies explicit context identity and known curriculum order; it does not establish open-world state inference or spontaneous memory creation.
- Different populations see fewer actual weight updates than the joint arm, despite equal overall update calls.
- Positive replay uses more gradient computation; cannot justify equal-compute superiority.

## Focused next idea (NOT implemented here)

A more faithful user-inspired memory/director system would separate three functions: **write** a stable memory, **query/read** it reliably, and **decide whether incoming experience merits a new memory**. Validate each independently with retrieval-ablated and context-routing controls, or test a slow/fast learner which copies/consolidates information across alternating roles rather than simply withholding updates for 40 steps.

Before any further work, predeclare an independent evaluation of old **A and B** before refresher, both ≥90% while still learning C ≥90%. Avoid launching new ecology or genetic evolution until those gates pass.

**This draft remains unmerged; main, original M5.6 sealed cohorts, ecological expansion and evolution remain untouched.**
