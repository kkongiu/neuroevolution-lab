# M5.25 — Specialization of memory roles: guarded writing, read-only retrieval, and learned arbitration

**2026-11-22 — real completed development experiment.** Strong **positive result for reliable archival memory**, narrowly limited to the known deterministic 3-context / 4-cue benchmark. **No evidence that an independently learned executive adds value beyond memory-only retrieval**. Do not resume evolution or merge into main.

## Why this experiment is different

M5.24 withheld teacher-provided A/B/C identities. It could remember A and C but scored **only 52.08%** on B after C. The observer traced the failure: B-exam queries selected C's latent bank; the internal bank formerly associated with B had acquired A-like records during switches. The key design intervention is not a new neural regularization coefficient: it is to **separate memory reading from authorization to write**, and to quarantine observations before any irreversible bank assignment.

## Experimental architecture

All nine arms instantiate the same 12-hidden-unit reward-predicting neural base, five trainable fast-versus-memory fusion weights, and an autobiographical record capacity of **128 actual (x,y,chosen_action,observed reward)** episodes in at most three latent banks (43/43/42 slots). Every actual action outcome triggers **one neural update call**, and experiences are never inherited between organisms.

In the guarded writer:
1. Each newly inferred memory bank is written using 40 *actually experienced* action outcomes and then made **immutable**. This is an **engineered** consolidation criterion, not autonomous biologically learned permanence.
2. When a real action outcome contradicts the current bank's known mapping, subsequent experiences go into **temporary quarantine**, not the memory bank. Reading the old bank continues; writing it is not allowed.
3. Once **12** personally lived outcomes have accrued, a **hand-coded evidence selector** computes actual reward compatibility against existing banks. If a previously validated bank matches ≥78%, it can be reactivated for reading; if no bank matches above 70% and a free slot is available, a **new bank** is created and buffered real experiences are committed to it. Ambiguous observations remain uncommitted.
4. Changes in the selected archive **do not overwrite the historical content** of a validated bank. A separate, trainable neural mixture determines the relative influence of a fast predictor and the retrieved archive estimate on actions.

No A/B/C label is supplied to any hidden-context arm: the identity and timing of switches are inferred from outcomes. The true labels are used only by the environment to calculate reward, by the **privileged oracle** comparison arm, and in separate offline diagnostic reporting after the simulation. The rule selector itself remains **hand-coded, NOT learned Q-based executive control**.

**Strong built-in structural assumption:** two actions have complementary **deterministic** binary rewards; from a lived failure of one action, the reader logically infers success of the other. This will not generalize directly to stochastic, continuous or multi-action ecologies.

## Design: 24 new development seeds × nine arms = 216 trajectories

New independent **development cohort 4900–4923**, disjoint from previous M5.6–M5.24 experiments and the protected original M5.6 sealed validation cohorts. Each seed gets a randomized *hidden* duration for the exact same A → B → C → A → B order. The sensory tasks remain A: x XOR y; B: NOT XOR; C: x. There are exactly **four possible x,y cues**. All arms receive matched cue schedules and the same number of real chosen-action outcomes and neural weight-update calls for a given seed, though their actual action/reward trajectories and **memory-read work, gradient masks, and write counts differ**. Thus do not claim identical total compute.

The arms are:
1. **M5.24 unsafe baseline**: old hand-coded evidence reader and writer, allowing ambiguous experiences to corrupt existing banks.
2. **Guarded director (candidate)**: immutable 40-record banks, 12-action quarantine and evidence-based write permission, with a learned fast/archive fusion coefficient.
3. **Guarded memory-only**: same protected writer/selector but uses archive readout only; tests whether the neural director is actually needed.
4. **Guarded fixed 50/50 fusion**: same protected writer/selector, no learned mixture control.
5. **Guarded fast-only**: protected archive still built, but action readout ignores memory, showing if archived knowledge itself matters.
6. **Guarded mutable archive**: same evidence selector, but allow writes into a bank during uncertain switches; tests the importance of write protection itself.
7. **Guarded corrupted probe**: intact learned archives, but actual reward feedback to the *temporary exam context selector* is replaced by independent random 0/1, leaving stored historical outcomes untouched. This is a causal test of accurate contextual diagnostic feedback, not an exactly distribution-matched reward permutation.
8. **Single mixed bank**: no context-change detection; same personal memory limit but no specialization.
9. **Oracle context**: the simulator provides the exact hidden A/B/C label for routing; explicitly privileged upper-bound control, **not** autonomous behavior.

### Frozen examinations and unavoidable feedback

At each curriculum checkpoint the model receives **16 exploratory actions with real observed outcomes from the test context**, solely to an ephemeral context-recognition scratch state, then makes **80 scored actions without additional feedback**. Tests are frozen in that **no learning weights, archived records or persistent controller states are modified**. We also run no-probe controls to expose that A/B/C cannot be uniquely identified from x,y alone. This is **not a reward-free examination**; the 16 diagnostic rewards contribute information. No unfamiliar cue patterns or task grammars are assessed.

The read-only retrieval tests are evaluated immediately after C, **before relearning A or B**.

## Verified execution and tests

[GitHub Actions successful run #45058303608](https://github.com/kkongiu/neuroevolution-lab/actions/runs/45058303608) passed **six new M5.25 scientific/unit tests** (including a fully controlled A→B→C→A→B sequence proving that archival banks remain unchanged, a mutable-write contamination negative control, the privileged context firewall and pure frozen exam) and **five original M5.6 neural controller regression tests**. All **216 full trajectories** finished; raw per-seed learning trajectories, probe-selected bank IDs, offline bank-content fit, capacity/write-accounting and matched-seed bootstrap contrasts were archived as **m5-25-guarded-memory-development** (30-day retention).

SHA-256 of canonical JSON raw result rows: **c6cb4899da0006d4bfd4a57f5ce5147350da78bfbb308e0908c1f5dda1df08f9**.

### Frozen success rates (mean of 24 development organisms, %)

| Condition | A after C | B after C | B first learned | C first learned | Mean pre-refresher A/B retained and B/C new |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unsafe M5.24 writer | 100.00 | **52.08** | 100.00 | 100.00 | **88.02** |
| **Guarded writer + learned director** | **100.00** | **100.00** | **100.00** | **100.00** | **100.00** |
| Guarded memory-only | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| Guarded fixed 50/50 fusion | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| Guarded fast-only (archive unused) | 45.83 | 54.17 | 75.00 | 89.58 | 66.15 |
| Guarded archive with mutable writes | **59.38** | 98.96 | 100.00 | 100.00 | 89.58 |
| Guarded but corrupted 16-outcome exam probe | 70.83 | 68.75 | 45.83 | 56.25 | 60.42 |
| One undifferentiated bank | 50.00 | 50.00 | 100.00 | 100.00 | 75.00 |
| **Privileged oracle context ID** | **100.00** | **100.00** | **100.00** | **100.00** | **100.00** |

All 24 organisms in the guarded-candidate arm scored 100% on the reported familiar-cue A/B/C examinations after C. The model reaches the **oracle ceiling** on this small task family, without being given the world identity, but only **after receiving 16 genuine diagnostic outcomes in each frozen exam**.

### Most important causal result: bank isolation and retrieval are separately verified

Post-hoc grader diagnostics of the **guarded** archive at the end of the full A→B→C→A→B life:

| Inferred archive ID | Fits A | Fits B | Fits C |
| --- | ---: | ---: | ---: |
| **0** | **100%** | 0% | 50% |
| **1** | 0% | **100%** | 50% |
| **2** | 50% | 50% | **100%** |

At the frozen phase-2 exams immediately after C, the guarded learner chose **bank 0 for A, bank 1 for B, and bank 2 for C, each in all 24 organisms**.

This contrasts directly with unsafe M5.24: with B as the exam target after C, **21/24** selected C-like bank 2, while only 3/24 selected bank 1. Its final bank 1 fit B in only **18.75%** on this new seed cohort. Thus writer isolation achieves both **stable bank contents** and **correct selection** in this particular known grammar.

A *mutable-write* control (same guarded detection but allowing writes during ambiguity) scores **59.38% on A** after C and corrupts the C bank toward B by the end of the curriculum. This provides evidence that **prohibiting unconfirmed writes is causally important**, rather than the gain arising only from a different context detector.

### Exploratory paired bootstrap on matched 24 seeds

Four-metric score combines A retained after C, B initially learned, B retained after C and C initially learned.

| Guarded candidate minus comparator | Mean advantage (pp) | Paired 95% bootstrap (pp) |
| --- | ---: | --- |
| Unsafe M5.24 writer | **+11.98** | **[+10.94, +12.50]** |
| Mutable-write guard | **+10.42** | **[+8.33, +11.98]** |
| Single mixed bank | **+25.00** | **[+25.00, +25.00]** |
| Corrupted exam probe feedback | **+39.58** | **[+30.21, +48.44]** |
| Guarded memory-only | **0.00** | [0.00, 0.00] |
| Guarded fixed fusion | **0.00** | [0.00, 0.00] |
| Oracle context | **0.00** | [0.00, 0.00] |

All intervals are **exploratory on one DEV task family**, not a post-development independent replication. A perfect measured accuracy cannot justify generalization to unseen sensory combinations or noisy environments.

## Preregistered screening: 9/12 pass, not 12/12

**PASS (9):** A after C ≥90%; B after C ≥90%; first B ≥90%; first C ≥90%; mean A/B/C after C ≥90%; within 7pp of privileged oracle; advantage over one mixed bank ≥10pp; advantage over corrupted feedback probe ≥10pp; write protection beats mutable-write control by ≥5pp.

**FAIL (3):**
- Candidate advantage over unsafe baseline preregistered ≥**12.00 percentage points**: achieved **11.9792 pp**, *missing by approximately 0.0208 pp*. This is a mathematically tiny margin and qualitatively strongly positive, but the predeclared numerical criterion **MUST NOT be rounded into a pass**.
- Learned director versus **guarded memory-only** advantage ≥5pp: **FAIL, exactly 0pp**.
- Learned director versus **guarded fixed 50/50 mixture** advantage ≥5pp: **FAIL, exactly 0pp**.

The guarded functional memory specialization is strongly supported **within this tiny task family**, but the **full composite gate fails**, since the learned executive cannot beat trivial readout and the threshold is missed slightly. Machine-readable values: **memory_specialization_screen_pass=false**, **learned_director_added_value_screen=false**, **development_screen_pass=false**, **independent_task_replication_complete=false**, **evolution_authorized=false**, **advance_to_evolution=false**.

## Interpretation and limitations

1. **The positive result is largely an engineered associative memory.** The quarantined, immutable bank strategy solves A/B/C retention in this deterministic four-input-pair school. It does not demonstrate that neural weights themselves can update forever without catastrophic interference, nor a general biological mechanism for brain plasticity.
2. **The selector is handcrafted**, not itself a learned meta-controller. The learned 5-weight gate still does not outperform memory-only or an untrained fixed mixture. A learned executive advantage remains unproven.
3. The learner exploits **binary complementary action rewards** and uses externally budgeted memory with capacity for all three contexts. A true lifelong organism needs novel stimuli, resource trade-offs, conflicts, and memory eviction/compaction without laboratory context capacity preallocation.
4. The 16 diagnostic exam outcomes reveal the currently hidden regime; zero-probe B after C remains **0%** when the agent defaults to bank 0 in this setup. So do not describe the model as knowing the context without any feedback.
5. The order A→B→C→A→B stays fixed, while durations vary by seed. The model might benefit from this limited order/grammar; independent testing on reordered tasks and newly generated rule families is required before promotion.
6. Same **real experiences** and neural update calls does not mean identical total compute or available gradient degrees of freedom. Archive queries, controller updates, and writing activity differ; memory-only can achieve 100% with fewer effective fast learning updates.
7. Result summaries and 24-seed paired confidence intervals are **development-only**. Thresholds were fixed before the DEV run; no independent out-of-distribution or sealed replication was authorized.

## Next gate: test the regista, not just the archive (not implemented)

A clean future executive test would remove the convenient repeated four-cue grammar and challenge the system with uncertainty about when a previously unseen regime is truly distinct; periodically introduce contradictory/incomplete evidence. The executive must **learn when to freeze, write, retrieve and discard** while protecting previous skills under an explicit memory budget and without being given the correct current task identifier.

Compare the learned executive against a **handcoded selector, memory-only lookup, fixed fusion, and independent randomized feedback**. Require a positive advantage and confidence interval excluding 0 on genuinely new rule sequences before even contemplating ecological evolution. This is not a request to expand ecology; it is a proposed stricter internal plasticity diagnostic.

**Main remains unchanged; original M5.6 sealed world cohorts are untouched. The PR remains draft.**
