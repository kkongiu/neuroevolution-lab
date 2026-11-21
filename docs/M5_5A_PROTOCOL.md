# M5.5A — PRE-REGISTERED programmable instinct scaffold with truly inherited numeric traits, not a neural discovery claim

Frozen on 21 November 2026 **before** any M5.5A implementation, candidate-genotype outcomes or new M5.5A worlds. Older M5.4C [causal exploration diagnosis](../reports/M5_4C_foraging_vs_reproduction_diagnostic_2026-11-21.md) found a source-selected original 18-gene neural movement failed due to repeated visits; its *researcher-scripted* local navigation with original neural birth rescued 23/24 normal and 12/12 hard worlds. This prospective study explicitly embraces the user's hypothesis that some preconfigured organismic capacities can be innate while the intensity/timing can be heritably varied. Do not confuse an efficient programmed reflex with spontaneous evolved skill.

## New architecture, inference scope and exact equations

A **new 8-dimensional signed REAL numeric temperament genome**, not a 401-weight neural network. It controls the gains of fixed **RESEARCHER-DESIGNED innate behavior primitives**. Built-in lawful reflexes: legal movement only, local hunger state from original internal energy, priority for adjacent-food/odor signals, bounded exploration pressure and stochastic motor escapes, innate ability to choose physical birth if age/energy/cell/population permit. No global edible-cell search, no sight beyond the native M5.0 original 15 local input channels, no executable genome, no file/network access. Every within-world birth can inherit and genuinely mutate the eight numeric genes. A new brain that *learns* these behaviors is NOT implemented here; code supplies the primitive menu. The entire 401-weight original M5.0 brain and older experimental findings are left unchanged. Specifically, **this is evolutionary optimization of programmer-given instincts, not de-novo neural intelligence**.

Inherited 8 numeric `g[0..7]`, each initialized iid Normal(0,.65) independently for EACH of **20 truly naive random founders** per world, founder RNG `random.Random(seed*2777 + 5501)`. Each gene clipped to [−2.5,+2.5] only on mutation, all finite. Initial worlds share EXACT founder numeric genome bank, positions and food schedule across intervention arms.

For every living organism, original M5.0 `observe` supplies blocked/odor/adjacent food/energy in native 15 channels, original legal moves are `m5_0.blocked_and_free`. Per-agent personal visited-cell counts and last heading are provided by **programmed innate per-life state**; newborn visit history starts only at their birth cell and heading None; no memory/genetic gene file inherited. Define hunger `h = 1 - energy / max_energy` in [0,1]. For each physically legal direction d:

`gain = 1 + 0.4*tanh(g5)*h`

`food_weight = 3.2 + 1.2*tanh(g0)` (positive 2.0..4.4)

`odor_weight = 0.8 + 0.6*tanh(g1)` (positive 0.2..1.4)

`novelty_weight = 0.20 + 0.20*tanh(g2)` (positive 0..0.4)

`inertia_weight = 0.12*tanh(g3)` signed

`noise_amplitude = 0.50 / (1 + exp(-g4))` (bounded 0..0.5)

`score[d] = gain*(food_weight*sensor[8+d] + odor_weight*sensor[4+d]) + novelty_weight/(1+personal_visits[target_neighbor]) + inertia_weight*int(d == last_heading) + noise_amplitude*Uniform[-1,+1]`

All scores numeric local, physically legal, never unrestricted position search. Motor random draws from deterministic `m5_0._local_rng(seed,tick,agent.id,83)` for each of the FOUR directions in fixed original ordinal order (including illegal) to maintain independent/common noise between competing physically masked movement alternatives; only legal directions considered. If exact-score ties, choose among maximizers with `_local_rng(seed,tick,agent.id,37)` without a visit-map lookup beyond the explicit per-agent primitive. When movement is possible, select max; otherwise physical no-op.

Birth intuition: original M5.0 native **physically legal** reproduction possible and `energy ≥ 46+6*tanh(g6)` (bounded 40..52) and independent seeded `_local_rng(seed,tick,agent.id,84).random() < 0.65 + 0.25*tanh(g7)` (bounded 0.4..0.9) -> select original PHYSICAL REPRODUCE action; otherwise choose direction as above. Same original M5.0 action cost1, newborn-energy12, birth cost20, food gain18, max-age100, max-pop64, no bypass. All movement/birth/food energetics applied via unchanged, previously cross-audited `m5_1._act_scripted` original-physics actuator with `eco.arm="heritable_no_mutation"`. This arm code is used *solely for unmodified physical legality and costs*; action selection is programmer-built innate **numeric trait** evaluator, NEVER original `m5_0.network_logits`. Label it correctly in reports.

No predators, danger flags or attacks exist in original ecology: **do NOT add aggression/flee code** without an explicit physical predator/damage/escape model. Avoidance of actual obstacles is the built-in low-level movement reflex already present.

## True genetic interventions (within-world births)

Original fixed birth mechanics first create EXACT cloned numeric parent genome via original-physics actor, then on the physical newborn **before it ever acts** apply only the registered genomic intervention, change the newborn's 8 REAL numeric trait vector and update the original physical birth event `genome` hash and cosine tracker. The newborn never inherits visit/heading state. No external trial/model optimizer, no trained genotype injection, no per-world manual interventions.

Five FROZEN arms (constant across all worlds):
1. `innate_clone`: independently randomly seeded 20 inherited-gene founders, ALL offspring exact original parent gene vectors (selection among initial founder variability is possible, but no new variation).
2. `innate_sparse_mutation`: each real birth chooses exactly **2 of 8** independent gene indices (without replacement) using `_local_rng(seed,tick,parent_id,85)`, applies iid Normal(0,.22), clamps only mutated indices to [−2.5,+2.5]; all other genes copy parent.
3. `innate_dense_mutation`: same intervention on ALL 8 genes each real birth, same gaussian .22 and clip.
4. `innate_random_newborn`: real offspring replace all eight genes with independent Normal(0,.65) values, using same separate tag85 seed. Their parents are physically the same; exact inherited traits are **not** preserved. This is an independent newborn-genotype counterfactual, not natural evolution.
5. `random_legal_movement`: same independent eight-gene **random source founders** and exact clonal native inherited temperament, SAME instinct birth policy, but with the entire innate movement score disabled and choose uniformly RANDOM among original legal directions with `_local_rng(seed,tick,agent.id,37)`. This intentionally HUMAN-programmed negative motor control, not a fifth type of organismic adaptation.

Mutation events must record parent+child full precision genetic vectors, 8-gene digest, founder origin, generation, changed indices, birth tick; all nonmutation physical birth ledger fields remain original. Enforce every recorded newborn and action matches original legal physics and state; integer native birth counters and physical energy residual≤1e-6 every 50 ticks. Record exact original initial founder gene pool and checksums per world, original wall/food generation/position and deterministic future food schedule geometry digest. All source founder traits and physical worlds match across five arms; in-world genotype trajectories differ ONLY through registered interventions, so true selection can be studied, but body physics and built-in instincts deliberately shared.

## Original independent fixed ecology experiment

- **24 fresh MAIN worlds 93000–93023**, original odor radius**2** from tick0, original obstacle density.12, regeneration probability.55, original 15×15 physical map, 20 original food/20 independent RANDOM founders, 400 physical ticks, max64 living, fixed 50-tick checkpoints.
- **12 fresh HARD SHIFTED worlds 93100–93111**, original odor radius2, original obstacle density.20, regeneration probability.40, all other settings identical.
- All previous source world seeds 79000–91111 (selected published ranges and relevant historical blocks) forbidden in this study; NO reuse of old M5.4A/B0/B0R/C test worlds or issue46 planned 92000–92111.
- **Five fixed arms × (24+12) new worlds = 180 original full 400-tick physical runs**. NO external population candidate optimizer. No change in any frozen numeric trait equation, birth threshold, search/mutation rates, epochs, world seeds or scientific criteria after seeing new world outcomes.
- Each world's full nominal food RNG schedule is generated BEFORE any actor actions, same seed across arms, original physics. No hidden new environmental shocks/shift because "SHIFTED" is harder from the beginning, as in previous published studies.

## Frozen primary and auxiliary inferences — avoid survival-only claims

Primary MAIN outcomes are real 400-tick survival (world-level binary), real native food, physical births, descendants carrying nonfounder genomes, fraction of final living descendants with genuinely mutated genotypes, max generation, max final population and mean checkpoint population (AUC normalized across 0..400). All reported equally for HARD. Reproducibility: all 20 naive founder full-precision genotype vectors and all birth genealogy events archived plus every per-world+checkpoint and 20k paired-world bootstrap statistical result. Original finite numeric genotype changes tracked WITHOUT any purported neural learning.

Predeclare world-paired 20,000 percentile bootstrap (fixed seed **93501 MAIN**, **93502 HARD**, +1000 per contrast) on each of these **six** contrasts in this order for both binary survival and population checkpoint mean AUC:
- `sparse_minus_clone`, `dense_minus_clone`, `random_newborn_minus_clone`, `sparse_minus_random_newborn`, `sparse_minus_dense`, `clone_minus_random_motor`.
Report delta mean and 95% descriptive CI, world-by-world deltas; resample independent WORLD only (not agents, births, ticks).

**Innate scaffold physical viability** gate: `innate_clone ≥18/24 MAIN`, clone positive benchmark of **programmed** behaviors (not neural).
**Mutation-specific observed advantage** gate: `sparse_minus_clone` survival effect world-paired bootstrap lower95%>0 AND population-AUC effect lower95%>0, AND at least ≥12/24 sparse world populations with real genuinely mutated offspring. Report dense/random controls and hard separately, no guarantee these gates pass.
**Evidence of causal adaptive evolution** (higher bar) CANNOT pass in M5.5A alone; would require a separately prospectively frozen lineage-transfer study where original founder vs true later mutation lineage, plus genotype-reverted offspring and random controls are assayed in independent worlds, and real mutation improves performance above exact founder/clones. No external trained genotype may be seeded as an uninformed founder. Even strong innate clone survival proves ONLY effective researcher-programmed skills.

Nonviable or null/negative outcomes stay in repo, with all source parameters and full results. A mutation producing diverse genomes does NOT prove selection, adaptation, novelty, intelligence or new innate behavioral module. Mean positive changes can arise from drift, survival conditioning or age/food differences. Never equate positive organism births with adaptation.

## Output and follow-up

`neuroevolution/m5_5a.py`, independent tests `tests/test_m5_5a.py`, first green historical CI BEFORE one-time scientific campaign, then original 180 trial/1620 50-tick checkpoint permanent CSVs, exact source founder genotypes, original complete birth events in archived artifact and a documented candid preregistered report. Later independent transfer experiment only if warranted, and separate M5.4D original native 18-gene recurrence decomposition issue46 remains open; this new innate architecture is a deliberately ALTERNATIVE line, not a claim that M5.4D itself was performed.
