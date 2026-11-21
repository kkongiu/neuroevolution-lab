# M5.5A — Innate instincts make organisms ecologically viable; **genuine birth mutations occur, but improved adaptation is NOT established**

**Original prospectively frozen physiological/genetic experiment, 21 November 2026.**

## 1. Traceable scientific evidence, prospective protocol and original genetic provenance

- [Original prospectively fixed protocol](../docs/M5_5A_PROTOCOL.md) committed **`a3aa39e4f68acc38d17a4292b4ebefbd4b72b10b` BEFORE new code or original sealed-world outcomes**. The innate algorithms, all 8 signed heritable genes, candidate genetic mutation mechanisms, initial gene RNG seed, original ecology, original 24+12 fresh world seeds, six paired comparisons, 20,000 paired bootstrap resamplings and evidence gates were fixed in advance.
- [Original full GitHub Actions research run #44940687397](https://github.com/kkongiu/neuroevolution-lab/actions/runs/44940687397) **SUCCESS** with **213 legacy and new regression tests passed**. The original study executed **five arms ×36 completely NEW original M5.0 world seeds ×400 physical ticks =180** complete source physical world-policy trials and **1,620** physical 50-tick checkpoints.
- [All 180 original complete ecological/gene-mutation per-world outcomes](M5_5A_all_180_original_genetic_trait_world_outcomes_2026-11-21.csv) and [all 1,620 original physical population/energy/genetic-mean checkpoints](M5_5A_all_1620_original_population_and_trait_checkpoints_2026-11-21.csv) are permanently versioned in git; every trial was reconstructed from the ORIGINAL job's original per-world and per-checkpoint JSON output, not invented from aggregates. All 5 matched physical world+same random initial gene-bank SHA256s per independent seed pass original source checks.
- [Complete original preregistered 20,000 world-level paired effects and results](M5_5A_all_preregistered_world_paired_mutation_statistics_2026-11-21.json).
- Original 30-day-reten­tion full raw [GitHub Actions artifact #14720899006](https://github.com/kkongiu/neuroevolution-lab/actions/runs/44940687397) `m5-5a-true-hereditary-instinct-mutation-original-world-trials` contains **all 36 full-precision sets of TWENTY originally naive founder eight-gene vectors**, EVERY physically real parent/child full-precision eight-gene genealogy event from ALL 180 worlds, original complete physical source food/energy/tick trace ledgers, `summary.json`, all 180 world rows and all 1620 checkpoints. Original 30-day archival limitation is real; original generator seeds, frozen source and 180 trial summaries remain permanently published.
- Source baseline world ranges exclude old M5.4A/B0/B0R/C and M5.4D proposed world blocks. MAIN 93000–93023: original 15×15 world, 20 food/20 independent naive founder organisms, wall density0.12, regeneration probability0.55, **odor radius2 from tick0**; HARD 93100–93111: original wall density0.20, food regeneration0.40, odor2. All original M5.0 food energy18, real basal movement cost1, min age4, max age100, physical reproductive transfer20 to born-with12, population cap64, 400 full ticks, identical source walls+food+founder positions+future physical RNG food schedules across arms.

## 2. What is programmed by nature analogy, and what is genetically variable

M5.5A is a **new EXPLICITLY HUMAN-PROGRAMMED innate behavioral scaffold**. It is NOT the old 401-weight native M5.0 neural controller, not the externally trained 18-gene RNN, and not an accidentally discovered brain. Eight **different REAL numerical heritable traits** tune relative strength of local primitive responses. There are **no code-executing genes**.

Every organism starts from a separately and independently RANDOM **8-real-gene** vector `g` sampled iid `N(0,0.65)` for each of 20 native founders per world (world source RNG `random.Random(seed*2777+5501)`). No human-chosen successful founder, no external numerical evolutionary algorithm training or silent transplant of an optimized B0R witness is used.

The programmer deliberately supplies pre-existing organism-like traits:

| Heritable numerical gene | Built-in mechanism with signed gene control |
|---|---|
| `g0` | Weight of **visible adjacent food** from original short-range physical sensory input |
| `g1` | Weight of **locally perceptible directional food scent**, not a global food location map |
| `g2` | Strength of explicitly **programmed personal visited-neighbor memory** and exploration bonus |
| `g3` | Signed persistence/avoidance of repeating the last movement direction |
| `g4` | Magnitude of original seeded stochastic motor search, permitting non-repeating trajectories |
| `g5` | Homeostatic food-seeking gain coupled to the body's real current energy deficit |
| `g6` | Energy threshold for preferring a **physically allowed original energy-costed reproductive action** |
| `g7` | Probability of attempting that real physical birth above the energy threshold |

Food response motor score is frozen `(1+0.4*tanh(g5)*h)*((3.2+1.2*tanh(g0))*adjacent_food +(0.8+0.6*tanh(g1))*directional_odor)+(0.20+0.20*tanh(g2))/(1+individual_neighbor_visits)+(0.12*tanh(g3))*same_heading+(0.50/(1+exp(-g4)))*Uniform[-1,+1]`, for the four ORIGINAL *physically legal* motor actions and original local 15 sensory values. Here hunger `h=1-current_energy/max_energy`. Only the relative primitive gains/noise/birth thresholds are inheritable, **not** the existence of these innate capabilities. Per-individual visited-neighbor counts/previous heading are human-supplied experience **not inherited by newborns**; all newborns start with an empty personal memory except their own birth cell. Low-level wall/collision avoidance is already an original fixed legal-move reflex. The original world has no predator/damage/attack mechanics: **flight and aggression are NOT faked**.

Only the human-coded primitive selection decides legal movement/reproduction. Original native M5.0 energy, real-food, age, birth legality, lineage, bookkeeping and randomized food schedule are used through previously independently verified original-physics actor `m5_1._act_scripted`; any positive result is therefore survival **because of programmer-specified instincts**, unless a separately causally proven gene-mutation benefit exists.

## 3. True new genetic variation was created on original real births

All five conditions have EXACT same **20 naive random eight-gene source genotypes**, original physical starting location and food RNG schedule for each matched world. All offspring first physically receive an exact cloned parent genome; *only after a real lawful, energy-costed birth*, source gene modifications are applied to the newborn BEFORE it can act and the original event genomic hash updated, **without modifying the original physical cost**.

Five predeclared arms:

1. `innate_clone`: exact parent-to-child **eight-gene clones**, no new variation. Original selection among existing 20 source founders is possible, but adaptation from *new mutation* is impossible.
2. `innate_sparse_mutation`: each real live birth changes **exactly two randomly chosen of eight** numeric inherited genes, each by iid Gaussian `N(0,.22)` and bounded [−2.5,+2.5].
3. `innate_dense_mutation`: each real birth changes **all eight** inherited genes at the same numeric scale.
4. `innate_random_newborn`: all eight newborn genes are freshly independent `N(0,.65)` values; no direct parental gene-trait inheritance, even though every offspring has a real native genealogy and native energy cost.
5. `random_legal_movement`: exact cloned eight-gene founders/offspring, same innate probabilistic physical reproduction and original physics, but random physically legal movement instead of programmer-supplied food/odor/hunger/novelty motor scores. The ONLY point of this control is to quantify how much the human-coded locomotion helps ecological viability.

Independent original full birth-genealogy inspection on original run archive verifies, across all 36 worlds:

| Registered condition | Real physical births | Births with changed REAL inherited genotype | Numeric coordinates changed per such birth |
|---|---:|---:|---:|
| Innate clone | **3,650** | **0** | 0 |
| Innate sparse mutation | **3,610** | **3,610** | exactly 2 |
| Innate dense mutation | **3,585** | **3,585** | exactly 8 |
| Random independent newborn | **3,595** | **3,595**, but deliberately NOT inherited parent genotype | 8 values replaced independently |
| Random legal movement | **1,694** | **0** | 0 |

**In all 180 trials there were exactly 16,134 original physically costed births**. Genuine inherited parent→child numerical **mutation** occurred on **7,195** of these births (3,610 sparse +3,585 dense). The 3,595 entirely random-genotype newborns are NOT evidence of heritable mutation-driven natural adaptation; they are a control. Parentage, source genotype, precise mutations, age/tick, physical birth events and mean energy ledgers are preserved. Exact original physical energy residual checked ≤1e−6 at every 50-tick checkpoint; no extra energy given to mutation arms.

This is true biological-model *genetic variation at reproduction* — a fundamentally different mechanism from earlier externally optimized fixed B0R witness — **but variation itself is NOT yet evidence of beneficial selection or adaptation**.

## 4. All 24 new MAIN short-odor worlds: innate programmed scaffold works BEFORE novel mutations

| Fully original original 400-tick arm | Living populations at tick400 /24 | Mean final organisms | Mean food acquired/world | Mean real births/world | Mean population at 9 checkpoints |
|---|---:|---:|---:|---:|---:|
| **Inherited innate exact clones** | **24/24** | 11.96 | 397.25 | 109.21 | 13.667 |
| **Inherited innate sparse 2/8 mutation** | **24/24** | 12.04 | 397.54 | 109.79 | 13.639 |
| **Inherited innate dense 8/8 mutation** | **24/24** | 12.33 | 398.88 | 110.88 | 13.662 |
| Independent genetically random newborn each birth | **24/24** | 12.17 | 396.33 | 108.58 | 13.796 |
| **Original real birth costs but random legal locomotion** | **14/24** | 4.63 | 214.63 | 58.00 | 8.681 |

**Vital point:** the programmed scaffold by itself gives physically viable naïve organisms **24/24 independent original short-odor worlds even with NO new mutations**, and the random newborn counterfactual also survives 24/24. This is a success of the ORIGINAL RESEARCHER-SUPPLIED INNATE DESIGN, not of newly evolved numerical adaptation. It also creates **a ceiling effect in the survival endpoint**: almost all innate conditions already have 100% viability, making it impossible to demonstrate a positive survival advantage from further mutation here.

**Preregistered mutation test:** sparse mutation vs exact clonal *binary 400-tick survival* = **0.00 pp**, 20,000 paired-WORLD bootstrap descriptive 95% **[0,0]** (both 24/24). Sparse mutation vs clones *population mean over checkpoints* = **−0.0278 organism units**, paired bootstrap 95% **[−0.3287,+0.3102]**, lower bound not positive. Dense mutations vs clones checkpoint population **−0.0046**, CI **[−0.2870,+0.2731]**, also no demonstrated advantage. Random newborn vs clones checkpoint population **+0.1296**, CI **[−0.1481,+0.3796]**, likewise no demonstrated benefit of any specific inherited novel variants. Strong genuine physical birth-mutation occurred in all 24 sparse worlds, but the jointly preregistered survival+population mutation benefit gate **FAILS**.

**The innate scaffold matters causally in these physical worlds:** compared with random legal locomotion, exact clone full innate forager succeeds **24/24 vs14/24**, paired world-level bootstrap survival difference **+41.67 percentage points**, 95% **[+20.83,+62.50] pp**. Mean checkpoint population **+4.986 organisms**, CI **[+3.727,+6.245]**. This establishes **human-coded innateness vs random locomotion** ecological benefit, not that the eight inherited genes evolved that innate movement behavior.

## 5. All 12 fresh harder short-odor worlds: still no demonstrated mutation advantage

| Fixed physical and gene-control arm | Living worlds at tick400 /12 | Mean final living | Food/world | Real births/world | Mean checkpoint population |
|---|---:|---:|---:|---:|---:|
| **Inherited fixed innate clones** | **12/12** | 10.08 | 286.50 | 85.75 | 10.583 |
| **Sparse heritable 2/8 mutation** | **12/12** | 9.00 | 286.92 | 81.25 | 10.593 |
| Dense 8/8 heritable mutation | **11/12** | 8.17 | 270.00 | 77.00 | 10.019 |
| Independent random genotype newborns | **12/12** | 9.25 | 289.17 | 82.42 | 10.333 |
| **Random legal motor, same physical birth** | **2/12** | 0.17 | 88.83 | 25.17 | 4.759 |

Sparse mutation vs clonal **binary survival difference 0 pp [0,0]** and average-population checkpoint effect **+0.0093 [−0.6019,+0.5185]**, no advantage. Dense mutation **lost** one of 12 world populations compared with clones, difference **−8.33 percentage points**, descriptive CI **[−25,0] pp**; insufficient by itself to assert systematically harmful mutation, but definitely not support for beneficial dense mutation.

Clone vs random legal movement hard-world survival **+83.33 percentage points**, paired bootstrap CI **[+58.33,+100]**, highlighting programmer-supplied locomotion's importance.

## 6. Precisely which evolutionary inference is supported?

- **Original physical and genetic feasibility established:** 20 independent naive random numeric-trait individuals per ecology, actual legal original births, exact newborn inheritance, sparse and dense random birth mutations, complete original parentage and original energy accounting; true hereditary phenotype variation exists.
- **Built-in human-defined evolutionary starting architecture is viable:** 24/24 normal and12/12 hard exact clone groups survive with original short odor2, without trained source genomes or new genetic mutations.
- **True mutation-driven adaptive improvement NOT established:** equal survivor counts across main genetically meaningful arms and no robust checkpoint AUC advantage (95% intervals include zero). The one hard dense-mutant extinction further cautions against assigning benefit to novelty of genotype alone. The random-newborn population is also viable, indicating the programmer-installed primitive action structure dominates this test's success.
- **Selection vs de-novo genetic adaptation NOT isolated:** old founder gene variants can be passively filtered by population drift and environment, even in exact clone conditions. A genuine beneficial lineage claim would require a separate preregistered *same original founder vs an actually changed descendant*, a gene-reverted control and independent real-world transplant against random newborn/clone variants. That experiment was **not** performed here, and cannot be retroactively solved by reading these heldout same worlds.
- **It is wrong to call this original RNN's self-emergent cognitive behavior:** the eight numeric temperament genes are inputs to **human-scripted** innate motility/reproduction reflexes. The old original numerical RNN's neural motor decisions are NOT used by M5.5A. Even when real descendant traits mutate naturally on births, the policy vocabulary is programmer-defined.
- **What biology suggests next:** Innate behavioral capacity can provide a physically viable scaffold, but here it may have eliminated the very selective pressure needed for mutation to provide visible survival gain. A new fully prospectively registered environment with meaningful resource competition or staged ecological changes, and original founder-vs-descendant transfer on new seeds, would be necessary before claiming adapted instinct evolution. Never tune original 93000–93111 worlds to manufacture a desired mutation benefit.

## 7. Next hypothesis: selection pressure, not simply larger brains or extra genes

Suggest **M5.5B (NOT RUN YET)**, a new prospective *physiological selection challenge* building on these eight explicitly HUMAN-coded innate genes, with NEW independent world seeds not used in any M5.5A/M5.4D tests. Conditions could include an actual predeclared resource shock or spatial heterogeneity that different innate trait values could affect. Freeze worlds, physics, mutation rates and statistical lineage transfer criterion BEFORE code/outcomes. Run genuine origin-independent source naive founders, exact clones, sparse/dense newborn mutation and random controls, and assay genuinely changed descendants **against their OWN unmutated founding genes** on totally independent source-unseen transfer worlds, not merely survivor counts.

Later, if physical predator/danger dynamics are implemented (separate frozen physics comparison), an actual innate avoidance/flight/attack module may be investigated. The current ecology has no predators; nothing of the sort was silently inserted into these five controls.

The independent older [M5.4D purely NEURAL motor exploration decomposition issue #46](https://github.com/kkongiu/neuroevolution-lab/issues/46) remains a separate valid question and is NOT resolved by this alternate innate-evolution architecture.
