# M5.24 — Executive with hidden context: B is misrouted and overwritten

**Date: 2026-11-22. Status: completed negative development-only experiment, 4/10 preregistered gates passed.** No evolution, no merge into main and no original M5.6 sealed validation cohorts accessed.

## Research question

M5.23 reached perfect A/B/C retention with an autobiographical memory explicitly keyed by researcher-supplied task identity. It did NOT demonstrate an advantage for a learned executive, because memory-only lookup also reached 100%. This experiment removes the A/B/C identifier from all non-oracle learners and tests whether lived outcomes alone suffice to recognize previously encountered regimes.

The environment silently switches through A → B → C → A → B. Per-seed phase durations are varied, not communicated. The fast predictor receives only x, y and the candidate action. The organism chooses an action and is told only whether that action succeeded. No privileged correct-action label or phase boundary reaches a hidden-context agent.

## Functional roles and important limitations

The 12-hidden-unit fast reward predictor updates once on each lived chosen-action outcome. The 128-record episodic store is divided into at most three internally numbered banks of capacities 43, 43 and 42. A hand-engineered detector of contradictions between recent real outcomes and stored outcomes creates, reuses and routes banks. Its history window is 14 observations. The learned five-weight director gates between fast predictions and the chosen bank's prediction; **the bank selector itself is a handcrafted evidence algorithm, NOT a learned neural executive or Q-learning mechanism**.

Crucially, the reader assumes two **deterministically complementary** actions: failure of action 0 logically indicates success of action 1. This is a strong known structural assumption about the experiment, not a world-oracle call, but it would fail unchanged in noisy/noncomplementary environments.

All arms have the same memory cap and actual lived outcome/update calls per seed; they differ in active gate parameters and memory-query work. The scorer uses the same four binary cue combinations repeatedly. There is no independent generalization to unseen sensory categories.

## Examination methodology

With an unannounced context and contradictory task rules, choosing the correct action from x,y alone is mathematically underdetermined. Therefore a **frozen retrieval examination** first supplies **16 real observed chosen-action rewards** for random diagnostic actions in the target world. Only a transient context-belief variable may respond; no weights, persistent history, learned state or archive records are updated. Next, the organism chooses 80 actions without further outcomes on balanced, familiar cue combinations. Zero-probe controls are also reported.

These exams are **not reward-free**; their 16 actual feedback probes are disclosed and essential to resolving hidden regimes. The privileged-context control gets the true task identity and serves solely as an upper bound.

## Eight comparisons

- Latent inferred bank selection + learned director.
- Same inferred bank selection + memory-only readout.
- Same inferred bank selection + fixed 50/50 fast-memory mixture.
- Too-short 3-observation detector history, below the required evidence window.
- One bank with no regime-change detection.
- Change detector with independent randomized feedback (archives still attempt to store real outcomes).
- Fast-only predictor with no memory in action readout.
- **Oracle context**: the simulator explicitly provides A/B/C; a privileged upper-bound condition only.

## Verified CI and raw artifacts

[GitHub Actions #45057298953](https://github.com/kkongiu/neuroevolution-lab/actions/runs/45057298953) succeeded, with seven new isolation/hidden-context tests plus five original M5.6 regression tests, and **24 new development seeds (4800–4823) × eight arms = 192 full trajectories**. Follow-up [diagnostics-only run #45057513897](https://github.com/kkongiu/neuroevolution-lab/actions/runs/45057513897) also succeeded. The follow-up added **only read-only observer reporting** of stored-bank rule contents and selected exam bank; it changed no learning rule, input, reward, test task or screening threshold.

All raw per-organism scores, world-switch events, selected archive IDs and offline archive diagnostics are stored in the GitHub Actions artifact named m5-24-hidden-context-executive-development (30-day retention). Canonical SHA-256 of final diagnostic rows: **682f2cb7acc9f0bc598ee5b01007fec8d70123329c9299f7acd84db2319ca9d7**.

### Frozen accuracy with 16 real feedback probes (24-seed means)

| System | A after C | B after C | B first learned | C first learned | Four-metric score |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fast-only | 48.96% | 51.04% | 75.00% | 86.46% | 65.36% |
| **Inferred latent banks + learned director** | **100.00%** | **52.08%** | **100.00%** | **100.00%** | **88.02%** |
| Inferred banks, memory only | 100.00% | 52.08% | 100.00% | 100.00% | 88.02% |
| Inferred banks, fixed 50/50 fusion | 100.00% | 52.08% | 100.00% | 100.00% | 88.02% |
| Three-observation history | 47.92% | 52.08% | 75.00% | 93.75% | 67.19% |
| One mixed bank, no detector | 50.00% | 50.00% | 100.00% | 100.00% | 75.00% |
| Scrambled-detector feedback | 80.21% | 58.33% | 100.00% | 100.00% | 84.64% |
| Oracle task label provided | **100%** | **100%** | **100%** | **100%** | **100%** |

The four-metric compound averages A retained after C, B first learned, B retained after C and C first learned. Tests on A/B are before any refresher teaching on those old tasks. A/B/C zero-probe scores for the inferred director after C are respectively **100% / 0% / 50%**, because its default bank 0 corresponds to A: without any outcomes identifying the present regime, it has no way of selecting B or C reliably.

### Key result: wrong retrieval and lost bank identity

The final diagnostics distinguish content from retrieval selection.

**After C, all 24 inferred-director organisms select latent bank 2 when the exam requests B**; none selects bank 1. When the target is A, all 24 select bank 0. For C, 23 select bank 2 and one selects bank 1.

Offline grading of each final stored bank against the three rule maps (world labels used **only in the research report, never by the learner**) shows:

| Internal bank | Fits A | Fits B | Fits C |
| --- | ---: | ---: | ---: |
| 0 | **100.0%** | 0.0% | 50.0% |
| 1 | **83.3%** | **16.7%** | 47.9% |
| 2 | 47.9% | 52.1% | **97.9%** |

The **oracle** upper-bound condition retains clean independent bank mappings: bank 0 is 100% A, bank 1 is 100% B, bank 2 is 100% C. In the hidden-condition variant, bank 1's final recorded content is largely A-like rather than B-like. This indicates a **bank assignment, isolation and retrieval failure** during changing regimes, rather than simply a failure of the fast neural network to memorize. The final-state evidence does not yet identify the exact moment of contamination, and must not be overinterpreted as a unique causal chronology.

### Paired exploratory bootstraps (24 development seeds)

| Inferred learned director minus comparator | Mean difference | 95% paired bootstrap interval |
| --- | ---: | --- |
| One mixed, undifferentiated archive | **+13.02 pp** | [+12.50, +14.06] |
| Scrambled feedback to change detector | **+3.39 pp** | [+0.52, +6.25] |
| Privileged oracle world ID | **−11.98 pp** | [−12.50, −10.94] |
| Same latent bank inference, memory-only | **0 pp** | [0, 0] |
| Same latent bank inference, fixed fusion | **0 pp** | [0, 0] |
| Too-short context evidence | **+20.83 pp** | [+17.71, +24.22] |

The history-based detector improves performance over ignoring switches, but the learned director has **no measurable added value over memory-only or fixed fusion**.

## Preregistered decision: 4 of 10 development gates pass

PASS: A remembered after C at ≥90%; B initially learned ≥85%; C initially learned ≥85%; inferred-routing four-metric advantage over one mixed archive ≥12 percentage points.

FAIL: B remembered after C ≥90%; mean of A/B/C after 16 probes ≥90%; within 7 points of the oracle; ≥12pp advantage over scrambled-detection control; ≥5pp added value of the learned director over memory-only; ≥5pp added value over fixed fusion.

Machine-readable status: hidden-context behavior supported = false; learned executive added value = false; development screen pass = false; independent task replication complete = false; evolution authorized = false; advance to evolution = false.

## Next strictly focused design (not started by this run)

Stop task/environment expansion and evolution. The next possible **plasticity-only** hypothesis is to preserve archive identity with **evidence-constrained writing, delayed assignment to a bank until regime confidence is sufficiently high, and explicit protection against overwriting a previously validated bank during ambiguous switches**. A separate temporal belief variable can govern *retrieval* while memory writes use stricter confirmation, avoiding the current accidental coupling of mistaken retrieval and destructive storage.

A better controller test would additionally include a **read-only bank ablation**, a heuristic context selector versus a *learned* selector, a task-history-agnostic control and **fresh preregistered seeds**; otherwise improvement might come from engineering rather than learned executive intelligence. Preserve exact-memory limits and disclose extra diagnostic action feedback and compute costs.

**No merging to main; original M5.6 sealed validation cohorts remain untouched and ecology/evolution disabled.**
