# M5.26 — Selection of lifelong memories under a real persistent-record cap

**2026-11-22. Completed development experiment. Positive for engineered novelty-based retention, negative for learned executive advantage.** 24 new development seeds × 10 arms = **240 full simulated lives**. The overall predeclared screen is **11/13 PASS**; the remaining two failures are tests of the *learned director* against memory-only and fixed fusion. Evolution, ecological expansion and main merge remain blocked.

## Research motivation

M5.25 demonstrated retention of A/B/C in a three-context deterministic hidden-world school using up to 128 persistent action/outcome records (40 per latent context bank in the candidate). M5.26 asks whether the organism can retain important experiences under a **strict 12-record long-term archival cap**, rather than endlessly storing redundant repetitions. We test a **researcher-engineered novelty selector** which retains one observed (chosen action, actual reward) record for each *distinct x,y cue* in each inferred latent bank. This is an adaptive response to witnessed novelty, **not an autonomously RL-learned retention policy**, and bank capacities remain fixed by the laboratory.

The same environment and learned fast-memory gate as M5.25 are retained. The true world A/B/C is not visible to any arm. The environment silently changes in the known A→B→C→A→B order with durations varying by seed. A = XOR, B = inverse XOR, C = copy x; each world has exactly four binary x,y sensory combinations. The agent receives success/failure only for its own chosen action, **not the desired action label**.

The reader uses the explicit assumption of **two deterministic, complementary-reward actions**: knowing the observed reward for one chosen action logically determines the other's outcome. This is a strong model-world assumption and would not work unchanged for general stochastic or many-action tasks. All frozen exams use the same familiar four sensory patterns, shuffled; no new-task-family/OOD generalization is demonstrated.

## Precise memory accounting

- **Persistent long-term archive:** 12 slots in 3 internally inferred banks of 4+4+4, or a stricter 8-slot stress test split 3+3+2. Reference arm has **128 available slots** and stores **40 episodes per bank = 120 actual archived records**.
- **Separate temporary memory:** up to **12 quarantine observations** for regime recognition before archival assignment. Frozen tests also supply **16 real reward probe outcomes** to an *ephemeral context-identification workspace*. Fast neural synaptic weights and director weights require additional capacity. Therefore **never state that the entire brain has only twelve memories or bits**: twelve is a persistent archive-record cap.
- **Novelty12 writer:** use the inferred latent bank; if the x/y pair is not yet represented, admit this genuinely lived outcome. If already covered, discard duplicates. When all four cues are stored, do not replace them. Existing bank identities are protected from contradictory writes until enough personally experienced feedback is collected.
- **FIFO12:** keep the latest four personally lived records in each inferred bank; replacing the oldest on confirmed writes.
- **Reservoir12:** uniform reservoir sampling among confirmed lived records assigned to each bank.
- **8-slot arms:** same three policies under 3+3+2 capacities. A literal uncompressed key-value store cannot simultaneously preserve all twelve different context/cue examples in eight slots. A different learner could generalize from examples, but this experiment does not implement abstract rule induction.
- **Fast-only, memory-only and fixed fusion:** keep the same novelty12 writer and experienced events, changing only whether the action uses memory or the trained mixture.
- **Reference128:** same neural model and inferred-bank evidence selector, with up to 40 personal outcome episodes held in each bank.

Every seed receives the exact same environmental x/y schedule and neural initial weights for all ten arms, and makes exactly **one neural update call per personally lived environmental action**; actual chosen-action rewards and resulting updates can vary among arms. Memory-read computation, active parameter gradients, number of writes and replacements are **not strictly equal**. The selection architecture is engineered, not a self-modifying brain.

## Cohort, tests and reproducibility

- Only new development seeds **5000–5023**; original M5.6 sealed source/holdout/replication seeds were never consulted.
- **Ten arms × 24 seeds = 240 completed trajectories**, each with the same phase durations per seed.
- Each frozen exam receives **16 actual chosen-action outcomes** solely for *temporary context routing*, followed by 80 scored decisions with **no changes** to weights, archive, learned director or persistent inference state. This is not a no-feedback test. Zero-probe diagnostics are separate.
- The first experimental run contained a discovered **capacity mapping bug**: an arm named reference128 was accidentally interpreted as belonging to the 8-record group due to its name ending in 8. The code was **corrected before publishing final results**; a new unit test explicitly checks the reference capacity equals (43,43,42). Ignore results from the flawed run when interpreting the 128-capacity reference.
- **Final CI:** [GitHub Actions #45059439440](https://github.com/kkongiu/neuroevolution-lab/actions/runs/45059439440) SUCCESS on commit 7f6c796d41ddf30c9d04934725e3782892ec4573. Six scientific unit tests, five original M5.6 regressions and all 240 trajectories completed; the raw machine-readable data are archived in the 30-day artifact **m5-26-memory-budget-development**.
- Canonical raw-row SHA-256: **e36af6508058ee179a79251ddf3f9fb2ca5c4830a541fa1802424e24c8a8e60f**.

## Frozen recall, 24-organism means

| Treatment | Actual persistent records (mean) | A after C | B after C | B first acquired | C first acquired | Four-metric pre-refresher mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Novelty-based 12 slots** | **12.00** | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **100.00%** |
| FIFO 12 slots | 11.67 | 73.96% | 65.63% | 84.38% | 85.42% | 77.34% |
| Reservoir 12 slots | 11.83 | 85.42% | 82.29% | 90.63% | 87.50% | 86.46% |
| Novelty-based 8 slots | **8.00** | 94.79% | 89.58% | 93.75% | 83.33% | 90.36% |
| FIFO 8 slots | 7.67 | 67.71% | 62.50% | 82.29% | 81.25% | 73.44% |
| Reservoir 8 slots | 8.00 | 81.25% | 84.38% | 85.42% | 85.42% | 84.11% |
| Novelty12, archive-only actions | 12.00 | 100% | 100% | 100% | 100% | 100% |
| Novelty12, fixed 50/50 fast/archive fusion | 12.00 | 100% | 100% | 100% | 100% | 100% |
| Novelty12, fast-only actions (archive ignored) | 12.00 | 45.83% | 54.17% | 80.21% | 89.58% | 67.45% |
| **128-slot reference (120 actually used)** | **120.00** | **100%** | **100%** | **100%** | **100%** | **100%** |

The four-metric mean is the average of **A retained after C, first B acquisition, B retained after C, first C acquisition**, all graded before refresher teaching on old tasks. A/B/C exams after C have identical top-level outcome under novelty12 and reference128. Novelty12 achieves **90% fewer persistent records (12 vs 120 actually used)** on the same tiny task family, **not 90% less total memory**.

All **24 novelty12 organisms** finish with exactly **[4,4,4] distinct x/y cue coverage** across their three inferred memory banks. On average, only **12 observations** were admitted to the archive; approximately **1,536 duplicate events** were rejected without overwriting useful representations. FIFO12 performed ~1,539 archive admissions and ~1,527 evictions per life; reservoir12 about 66 admissions and 54 replacements. These numbers measure memory-write activity, not computation in arbitrary physical units.

## Paired exploratory confidence intervals

24 paired development seeds, 1,400 fixed-seed bootstrap draws, deltas on the preregistered four-metric mean:

| Novelty12 minus comparator | Mean gain (percentage points) | Exploratory paired 95% interval (pp) |
| --- | ---: | --- |
| FIFO12 | **+22.66** | **[+18.75, +26.56]** |
| Reservoir12 | **+13.54** | **[+10.16, +17.45]** |
| Fast-only12 | **+32.55** | **[+29.95, +35.42]** |
| Novelty8 | **+9.64** | **[+7.29, +11.72]** |
| Memory-only12 | **0.00** | [0, 0] |
| Fixed 50/50 fusion12 | **0.00** | [0, 0] |
| Reference128 | **0.00** | [0, 0] |

The intervals are development-cohort exploratory evidence, **not independent new-task replication**.

## Exact preregistered STOP/GO: 11 of 13 pass

**PASS (11):** A after C ≥90%; B after C ≥90%; B first learned ≥90%; C first learned ≥90%; mean of all three after C ≥90%; novelty12 beats FIFO12 by ≥10pp; beats reservoir12 by ≥10pp; beats fast-only12 by ≥20pp; archive budget 8 shows ≥4pp loss versus 12; each novelty12 bank covers all four personally experienced x/y cues; all archive capacity caps obeyed.

**FAIL (2):** the learned director must beat novelty12 *memory-only* readout by ≥5pp (**0pp achieved**), and must beat novelty12 *fixed fusion* by ≥5pp (**0pp achieved**). The director has again not demonstrated meaningful independent executive value. Machine flags: storage_selection_screen_pass **true**, learned_executive_added_value_screen **false**, development_screen_pass **false**, independent_task_replication_complete **false**, evolution_authorized **false**, advance_to_evolution **false**.

## Scientific conclusions and next target

**Supported:** The hand-engineered retention of one useful, witnessed outcome per distinct sensory cue preserves all previously learned A/B/C rules in a hidden-context, deterministic four-pattern school with only 12 long-term archive records. It is significantly more effective than FIFO or reservoir selection at the same persistent capacity. At only 8 slots the lack of complete cue coverage produces measurable performance losses, particularly first-time C at **83.33%** and B-after-C at **89.58%**.

**Not demonstrated:** a learned RL-based memory policy; an executive superior to archival lookup; general biological synaptic stability; automatic discovery of categories/bank capacities; rule abstraction from fewer examples; out-of-distribution generalization to unseen sensory patterns, noisy rewards or new rule families; reduced total compute or memory; independent replication. These limits must remain explicit.

**Proposed next focused research only if authorized:** test whether a neural or differentiable **rule-compression representation** trained from personally experienced observations can retain A/B/C with only 8 archival examples, while protecting old skills and generalizing to held-out cue patterns. Or learn the retention-value policy under a truly dynamic global memory budget instead of handcoded per-bank novelty. Predeclare new task families and independent checks before advancing to evolution.

**Draft unmerged, main unchanged, sealed original M5.6 cohorts unaccessed, no ecological evolution.**
