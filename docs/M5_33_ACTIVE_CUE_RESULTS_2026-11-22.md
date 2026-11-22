# M5.33 — VERIFIED: active model identification succeeds; dual-neuron generalization does NOT improve

**22 November 2026 | [CI run #45074404512](https://github.com/kkongiu/neuroevolution-lab/actions/runs/45074404512): SUCCESS**
All M5.33 experiment-integrity tests, M5.32 regression tests and
original sealed M5.6 online-learning regression tests passed, followed
by all 264 optimization/context trial rows. However the **predeclared
M5.33 science screen FAILS: only 7/11 gates pass**. Three direct
neural-learning improvement thresholds and the separate heldout
improvement threshold fail. Negative results are preserved below.

The methodology, task generator, actual 8-reward budget, candidate
grammar, seeds and 11 stop/go criteria were frozen **before the CI
science evaluation** in
[M5.33 preregistration](M5_33_ACTIVE_CUE_PROTOCOL_2026-11-22.md).
This run uses **44 NEW synthetic task seeds** under the **old familiar
artificial task families**, NOT a previously unseen task family or
independent replication.

## Experimental unit and how to interpret accuracy

- 8 parity, 12 pair-AND, 12 triple-AND, 12 XOR-of-TWO-disjoint-AND
  task seeds, each with A/B/C externally OBSERVABLE contexts.
- Two dual-neural fitting initializations **1031 and 1033**.
- **132 unique environment task-contexts × 2 neural seeds = 264 paired
  context-optimizer rows per selection policy**; four policies
  (**1056 dual-neural fits**).
- All policies begin with **the same two sensory cue IDs**, use a
  **shared precommitted random actual action for each cue**, then
  acquire SIX additional distinct sensory cues sequentially via the
  policy. Every observation has a real binary environmental reward,
  never the unchosen action outcome.
- On the resulting eight genuine experiences, train the exact same
  M5.32 **two-unit, 15-parameter continuous** model **without the
  diversity penalty**, same optimizer seed and 8 restarts ×300 SGD
  updates. This ensures matched neural training budget. The exact
  version-space planner itself has costly, preprogrammed oracle
  PRIOR STRUCTURE and its computational work is **not matched** with
  random/coverage planning.
- **Primary all-16-cue accuracy uses the SAME sensory domain across
  policies, but includes 8 observed training cues.** It is NOT
  unseen generalization accuracy. Unseen-only accuracy is also
  measured on each policy's other eight *never observed* cues; these
  sets DIFFER by policy, so simple paired differences need caution.
  The two numbers must always be reported together. The downstream
  neural learner fits essentially 100% of the observed supports but
  often performs close to chance on the unobserved remainder.

### Four cue-selection policies

**Random:** seeded uniform order, no feedback dependence.
**Coverage:** reward-independent Hamming-distance novelty.
**Active model-space entropy:** a HUMAN-ENGINEERED catalog of **112
complete truth tables** from known parity + conjunction hypotheses;
after each actual chosen-action reward, discard inconsistent possible
truth tables, and choose the cue that best splits the remaining model
possibilities (tie-break coverage and fixed random order).
The policy is NOT GIVEN THE TRUE TARGET rule, but the correct set of
potential rule families is preprogrammed by researchers. This is
**engineered epistemic planning**, not learned neural curiosity.
**Corrupted active:** same policy, but its *planning beliefs only*
receive precommitted randomized false rewards. The downstream dual
neuron still trains on genuine actual environmental reward for the
cues chosen by this damaged planner. Belief contradictions trigger
catalog reset; resets are counted and not hidden.

## Numerical scores from full raw data

All values are means of task-context/optimizer evaluations on the
prespecified cohorts; percent correct on exact target bit.

### PRIMARY: all 16 cues, EIGHT are in the training support

| Cohort | Random | Reward-blind coverage | Family-aware active | Corrupted-belief active |
|---|---:|---:|---:|---:|
| Plain parity | **99.22%** | 99.09% | 98.44% | 98.70% |
| One pair AND | **87.15%** | 82.29% | 84.64% | 86.63% |
| One triple AND | 91.49% | 89.58% | **92.71%** | 88.98% |
| TWO disjoint pair ANDs | 74.31% | **76.04%** | 72.31% | 73.52% |

For the MAIN two-disjoint-interactions cohort:
- Active against random: **−2.00 percentage points**
  on all 16 cues, exploratory hierarchical paired bootstrap 95%
  **[−4.77, +0.78]pp** — prospective +5pp improvement FAILS.
- Active against coverage: **−3.73pp**, exploratory interval
  **[−6.42, −1.48]pp** — notably WORSE than simple reward-blind
  novelty sampling on this particular predeclared dataset.
- Active against corrupted-belief planner: **−1.22pp**,
  exploratory interval **[−3.99, +1.56]pp** — no established
  benefit of truthful information-directed planning to the
  dual-neuron learner on this family.
- 72.31% all-16 active performance exceeds the >=60% absolute
  *mixed in-sample-plus-unseen* screening gate but cannot be
  interpreted as 72% genuine new-cue generalization.

### SECONDARY: eight genuinely unseen cues (DIFFERENT cue sets by arm)

| Cohort | Random | Reward-blind coverage | Family-aware active | Corrupted-belief active |
|---|---:|---:|---:|---:|
| Plain parity | **98.44%** | 98.18% | 96.88% | 97.40% |
| One pair AND | **74.31%** | 64.58% | 69.44% | 73.26% |
| One triple AND | 82.99% | 79.17% | **85.42%** | 77.95% |
| TWO disjoint pair ANDs | 48.61% | **52.26%** | **44.62%** | 47.05% |

Crucial negative on TWO-disjoint-AND tasks: active's heldout-only
accuracy is **44.62%**, versus random 48.61% and reward-blind
coverage 52.26%, so the prospective +5pp unseen gate FAILS.
The active-vs-coverage heldout difference is **−7.64pp**
(exploratory interval **[−13.02, −3.13]pp**).
This **heldout contrast cannot control identical exam difficulty**
because each sampler observes different 8 cues and therefore leaves
different 8 cue IDs as exams. Its direction corroborates, but
does not independently prove, the all-16 pattern.

The original M5.32 dual-neural-learning failure was NOT repaired
by this form of information-seeking.

## DID the engineered active planner acquire sufficient information?

**YES, for the programmer-supplied finite grammar.** Its human-built
112-function hypothesis set contains the actual full 16-cue truth
table of every synthetic task, by construction. Its adaptive
version-space cue selection leaves **exactly ONE consistent full
truth table after eight observations in 100% of all 132
environmental contexts (264 with optimizer repeats)**. It can thus
recover the target function with **100.00% full-domain accuracy**
if it DIRECTLY USES ITS HUMAN-DEFINED SYMBOLIC POSTERIOR.

The downstream dual neural network trained on those EXACT same
eight experiences **does not inherit that symbolic hypothesis class**.
Its all-16 accuracy for two-pair tasks stays 72.31%, with 100%
accuracy on supports but **only 44.62%** on never-seen cues.
Therefore the single-episode failure is not simply "we have
insufficient reward observations": **eight can uniquely identify the
function WITH THE CORRECT HAND-CRAFTED PRIOR**, yet they do not
teach our current neural architecture how to generalize.
Without the programmed truth-family prior, identifiability is not
guaranteed. This is a strong **representation/inductive-bias
mismatch**, not evidence of autonomously learned abstraction.

Theoretical posterior identifiability at 8 observations:

| Cohort | Random | Coverage | Active | Corrupted-belief sampler (scored with real observations) |
|---|---:|---:|---:|---:|
| Parity | 45.83% | 54.17% | **100%** | 58.33% |
| Pair AND | 47.22% | 58.33% | **100%** | 66.67% |
| Triple AND | 16.67% | 11.11% | **100%** | 30.56% |
| Two disjoint ANDs | 69.44% | 63.89% | **100%** | 69.44% |

For two-pair tasks active cuts the average number of remaining
possible functions from **1.389 (random)** to **1.000**:
**28% fewer** (passes predeclared >=20% reduction). Exact
identifiability improves by **+30.56pp** (passes >=+10pp).
The direct grammar-informed Bayes prediction is 100% all-16
for active sampling on every cohort, but the neural network's
heldout-only result on TWO-pair tasks remains below chance.

## Frozen 11 gate audit (7 PASS / 4 FAIL)

1. Two-pair posterior candidate count down >=20% vs random:
   **PASS** (−28.0%).
2. Two-pair full-function exact identifiability >=+10pp vs random:
   **PASS** (+30.56pp).
3. TWO-pair neural all-16 >=+5pp vs random: **FAIL** (−2.00pp).
4. TWO-pair neural own-unseen-eight >=+5pp vs random:
   **FAIL** (−3.99pp, different exam cue sets).
5. TWO-pair neural all-16 >=+5pp vs scrambled-belief:
   **FAIL** (−1.22pp).
6. TWO-pair neural all-16 >=+3pp vs reward-blind coverage:
   **FAIL** (−3.73pp).
7. TWO-pair neural all-16 accuracy >=60%:
   **PASS** (72.31%, eight support cues included).
8. True function remains in all authentic hypothesis posteriors:
   **PASS**.
9. Two shared initial cues + six distinct requests, authentic action
   and reward at all eight chosen cues for all arms:
   **PASS**.
10. Complete raw 16-cue, own unseen-8, scrambled-belief
    and coverage reporting, no leakage: **PASS**.
11. Identical 15-real-float32 neural codes with 8×300 training
    updates and evolution stop flags: **PASS**.

**Prospective development science screen = FAILED; 7/11 only.**

## Reproducibility evidence and explicit stop

[CI run #45074404512](https://github.com/kkongiu/neuroevolution-lab/actions/runs/45074404512)
successfully ran the tested protocol and uploaded full
\`m5-33-active-cue-development\` raw artifact
(ID **14774102857**, 30-day retention, expires **2026-12-22**).
The downloaded ZIP was independently opened and JSON parsed:
**264 paired context-optimizer rows**, **132 unique
environmental contexts**, 7/11 screening checks. Recomputed
canonical SHA-256 of compact sorted-keys raw-trials JSON is EXACTLY
\`c94312c712c99981db0183e8ac06c9d9dc040dedd0ece86398e4c512d9920710\`,
matching the run's stored field. Every row includes the complete
per-policy 8-step sequence, real action/reward, imagined corrupted
feedback when relevant, posterior candidate sizes, actual
complement exam IDs, 16-cue neural predictions, all scores and
15-float32 final neural weights. No raw run is suppressed.

Even if symbolic posterior search is 100% here:
- \`local_active_cue_screen_pass=false\`
- \`learned_general_curiosity_policy=false\`
- \`autonomously_invented_neural_operator=false\`
- \`shared_meta_plastic_controller_learned=false\`
- \`independent_task_family_replication_complete=false\`
- \`continual_memory_retention_verified=false\`
- \`evolution_authorized=false\`
- \`advance_to_evolution=false\`
- \`sealed_M5_6_ecology_accessed=false\`.

M5.33 is a DRAFT PR stacked on M5.32. Original sealed M5.6
ecology untouched; no merge to \`main\`.

### Next scientific hypothesis (not implemented)

M5.34 should test **the gap between information acquisition
and usable representations**, comparing a neural model trained
only from eight actions/rewards with a model that receives
*explicit, audited* human-built symbolic posterior targets or
a learned cross-task neural representation. Keep these priors
separate (not an unfair "neural invention" claim), track full
out-of-family function generalization, and vary support count
while preserving clean test leakage boundaries.
Do NOT increase the neuronal count again until the
**identifiability-vs-generalization gap** is understood.
