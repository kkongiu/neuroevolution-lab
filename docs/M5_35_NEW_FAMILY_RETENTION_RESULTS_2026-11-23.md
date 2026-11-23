# M5.35 — Verified report: genuinely new synthetic-family transfer and catastrophic forgetting

**23 November 2026 | GitHub Actions [#45139099508](https://github.com/kkongiu/neuroevolution-lab/actions/runs/45139099508) SUCCESS; 7/11 prospectively frozen science gates PASS.** This is a **FAILED overall narrow development screen**, despite successful code, regression and complete science runs. Retain the negative outcomes and do not silently adjust the thresholds.

[The M5.35 protocol was published before the full run.](M5_35_NEW_FAMILY_RETENTION_PROTOCOL_2026-11-23.md)
It fixed both novel generators, complete-function splits, all training budgets and the eleven go/no-go conditions. The original sealed M5.6 ecology was never accessed and main is unchanged.

## Design: which knowledge is truly novel?

The M5.34 neural representation was pretrained on **85 old TRAIN functions**, chosen among 112 different old Boolean full 16-bit truth tables. Its **27 old HELDOUT functions** were already used for M5.34, so this old-retention benchmark is *not* a new independent confirmation.

M5.35 enumerates two researcher-defined **wholly different synthetic function families**:
- **Conditional MUX:** 96 unique possible functions, selector bit chooses one of two independently inverted branch inputs.
- **Signed EXACT-K:** 48 unique functions, exactly k in {1,2,3} active input bits after a four-bit inversion mask and possible output complement.

All 96 + 48 complete 16-bit signatures were verified to have **zero intersection** with old M5.34's 112 known-family signatures and zero MUX-vs-EXACT-K intersection.

Locked deterministic split: MUX **40 new TRAIN, 16 new HELDOUT, 40 unused**; EXACT-K **20 new TRAIN, 12 new HELDOUT, 16 unused**. Hence **60 new TRAIN/28 new HELDOUT/56 never used functions**, all disjoint from old TRAIN/HELDOUT. Function identities and any withheld function's full truth labels are excluded from *all* model training, including replay.

**Zero-shot test (H1):** only 3,000 Adam updates×32 TRAIN episodes from OLD families (96,000 *fully supervised* old training episodes per initialization), then freeze and predict on never-trained **new family** test functions using 8 authentic randomly chosen action/reward supports. This is a genuine *new synthetic-family* transfer stress, not real-world transfer or any evidence of self-invented operators.

**Second stage (H2):** fork identical old-trained 3,152-float32-parameter neural weights and reset identical Adam moments. Run **1,200 more steps, batch 32**:
- **N / new-only:** 32 novel TRAIN-function supervised episodes per update, i.e. 38,400 new episodes.
- **H / half-unique novel:** 16 novel TRAIN episodes sampled per step then DUPLICATED as a 32-example batch; 19,200 new draws, zero old.
- **R / old replay:** the EXACT same first 16 novel draws as H **plus 16 old TRAIN-function supervised episodes**, 19,200 new +19,200 old examples. The old data is researcher-supplied external memory, and R does not learn its own replay schedule.

N/H/R identical trainable float32 weights (3,152), initial checkpoint, optimizer, total 1,200 updates and batch arithmetic. H/R also match the number and identity of *sampled new episode draws*; H duplication intentionally changes batch diversity, so exact new exposure and replay mechanism should be stated, not disguised.

At evaluation all network variants receive the SAME 8 genuine chosen-action/binary-reward support observations, and score on the SAME 8 completely unseen sensory cues. No heldout episode is used for neural training or optimizer selection. Two seeds 1301/1303 ×(27 old+28 novel) whole HELDOUT truth functions ×two fixed random sensory episodes = **220 paired evaluation rows**, but only 55 distinct full target functions.

A separately fitted M5.32 15-parameter dual-soft neuron is a historical within-task comparison, **NOT** compute-/parameter-/pretraining-exposure matched to the 3,152-parameter shared MLP.

## Results: accuracy ONLY on the 8 genuinely unseen cues

| Heldout function family | Frozen old model, zero-shot on new | N: learn only novel | H: fewer novel draws, no replay | R: novel + old replay | M5.32 task-fit |
|---|---:|---:|---:|---:|---:|
| **MUX (16 unseen whole functions)** | 50.00% | **60.74%** | 59.38% | 53.91% | 50.78% |
| **EXACT-K (12 unseen)** | 55.73% | 55.21% | 57.03% | **56.77%** | 48.44% |
| **All 28 novel functions** | 52.46% | **58.37%** | **58.37%** | 55.13% | 49.78% |
| Old parity (1 previously examined function) | **93.75%** | 65.63% | 59.38% | 75.00% | **100.00%** |
| Old pair interactions (12) | **70.83%** | 49.22% | 48.18% | 64.32% | 69.79% |
| Old triple interactions (8) | 78.13% | 50.78% | 49.61% | **74.22%** | **82.81%** |
| Old two disjoint ANDs (6) | **59.90%** | 50.00% | 46.35% | 54.17% | 43.75% |
| **All 27 old heldout functions** | **71.41%** | 50.46% | 48.61% | **65.39%** | 68.98% |

The mean across 28 novel functions is weighted by their actual cohort sizes (16 MUX/12 EXACT-K); the old mean weights 27 old heldout functions. Each function has four correlated evaluations (two support episodes×two initializations); report uncertainty at unique function identity, not row-level.

### H1: zero-shot entirely new-family behavior

The old-trained network, NEVER having seen examples of either new generator, achieves **52.46%** mean genuine novel-family unseen-cue accuracy, compared with **49.78%** for the old task-adapted dual-soft baseline. Both are near chance on a balanced deterministic two-action environment. **There is no convincing zero-shot transfer to wholly new synthetic mathematical families.** The positive M5.34 *within-family* transfer does not establish H1.

### H2: new learning and forgetting

- N / novel-only raises new-family unseen accuracy **52.46% → 58.37%**, **+5.92pp**, below prespecified **+10pp** gate. Exploratory paired 95% bootstrap over **28 distinct novel function identities**: **[−0.33,+12.50]pp**, including zero. In MUX alone N goes 50.00% → **60.74%**, +10.74pp, function-cluster interval **[+1.76,+20.31]pp**; EXACT-K does not improve (−0.52pp).
- R / replay reaches **55.13%** on new functions, only **+2.68pp** from 52.46% zero-shot (cluster interval **[−2.23,+7.70]pp**), FAILING both preregistered **>=60%** and **>=+8pp** new-family criteria.
- **Catastrophic old-family forgetting without rehearsal:** old unseen performance **71.41% → 50.46%** after N (**−20.95pp**, function-cluster bootstrap **[−27.43,−14.35]pp**). For the H control old score becomes **48.61%** (**−22.80pp**) despite fewer distinct new-family draws.
- **Replay materially prevents much forgetting:** after R old unseen accuracy is **65.39%** (**−6.02pp** vs baseline; function-cluster interval **[−9.61,−2.31]pp**), about **+14.93pp** versus N (**[+9.72,+20.60]pp**) and **+16.78pp** versus H (**[+12.04,+22.34]pp**). However the preregistered acceptable forgetting limit was **≤5pp loss**: R misses it by **1.02 percentage points**.
- Replay costs **−3.24pp** on aggregate new test functions relative to N (function-cluster exploratory interval **[−6.81,+0.22]pp**), within the prespecified acceptable **10pp** trade-off. MUX pays a stronger **−6.84pp** vs N, while exact-K gains +1.56pp under replay.

**Interpretation:** Explicit access to old supervised training data plus a FIXED hand-designed 50:50 rehearsal strategy protects much existing knowledge, but does not prevent measurable forgetting and also reduces new learning in MUX. The network did not autonomously decide what to preserve or when to replay. The result supports studying plasticity/stability tradeoffs, NOT autonomous continual intelligence.

## Eleven frozen stop/go checks — 7 PASS, 4 FAIL

| Prespecified check | Outcome |
|---|---|
| Entire 28 novel HELDOUT truth signatures absent from all old/novel TRAIN functions | **PASS** |
| Exhaustive unique 96 MUX/48 EXACT-K signatures, no original 112 overlap | **PASS** |
| R novel unseen score ≥60% | **FAIL** — 55.13% |
| R novel gain over zero-shot ≥+8pp | **FAIL** — +2.68pp |
| N novel gain over zero-shot ≥+10pp | **FAIL** — +5.92pp |
| R old heldout retention within 5pp of baseline | **FAIL** — −6.02pp |
| R old retention advantage over N ≥+5pp | **PASS** — +14.93pp |
| R old retention advantage over H ≥+4pp | **PASS** — +16.78pp |
| R new family score not worse than N by >10pp | **PASS** — −3.24pp |
| Same eight authentic supports/frozen exams, 1,200-step update budgets, same 3,152 actual float32 parameter sizes, H/R common new draw stream | **PASS** |
| Full raw rows, all family subgroup errors, complete snapshots and negative outcomes preserved; no evolution | **PASS** |

**Prospective science screen = FAIL (7/11), despite all test, regression and complete evaluation CI jobs passing.** Do not reinterpret the 7 passes as seven independent breakthroughs. Absolute accuracy near 50% for novel functions and remaining six-point old retention deficit are important reasons to refrain from evolution or grand claims.

### Complete reproducibility evidence

Official GitHub Actions science run
[#45139099508](https://github.com/kkongiu/neuroevolution-lab/actions/runs/45139099508),
all jobs successful. Full raw artifact:
\`m5-35-new-family-retention-dev\`, Actions artifact ID
**14800967554** (30-day retention until **23 December 2026**).

Independently downloaded ZIP and inspected its contents:
- **220 raw evaluation rows**, **55 unique heldout whole functions**,
  including **28 new-family whole functions**;
- all actual support chosen-action/reward triples, eight strictly
  disjoint unseen exam cue IDs, all predictions for four neural
  checkpoints and M5.32 comparison;
- complete 3,152-float32 learned weight arrays, per-stage optimizer
  exposure metadata, all exact function set signatures, all
  cohort results, gate verdicts and negative cases;
- exactly **7 of 11** preregistered science gates true;
- independently recomputed compact sorted raw trial SHA-256
  **\`5524b3ce2482637196c0fabd68f1649edb00448067745da958781434d8d9666d\`**,
  matching the official job result EXACTLY.

The network has 3,152×4 = **12,608 bytes** of final float32
weights per checkpoint, excluding all episodic dataset storage,
Adam states, replay memory, fixed model code, historic training and
runtime workspace. No matched-compute claim versus original M5.32
is warranted. No real-world benchmark, hidden-context transfer or
autonomous action selection has been attempted here.

### Stop flags and next hypothesis (not implemented)

- \`local_family_transfer_and_retention_screen_pass=false\`;
- \`learned_general_curiosity_policy=false\`;
- \`autonomously_invented_neural_operator=false\`;
- \`shared_meta_plastic_controller_learned=false\`;
- \`independent_realworld_replication_complete=false\`;
- \`autonomous_continual_learning_verified=false\`;
- \`evolution_authorized=false\`;
- \`advance_to_evolution=false\`;
- \`sealed_M5_6_ecology_accessed=false\`.

M5.35 remains a **DRAFT PR** stacked on M5.34. The
original M5.6 ecology and \`main\` remain untouched.

**Next worthwhile M5.36 (proposed only):** test lower-memory
regularization (e.g., EWC / Fisher sensitivity), sparse replay,
and conditional context-based sparse adapters, with strict
memory/compute accounting and the exact old/new function-holdout
boundary. Compare forgetting and genuinely novel-family
accuracy together. In particular, **do not report the replay
benefit as evidence of spontaneous memory**: old training
function truth tables are still available to the researcher.
