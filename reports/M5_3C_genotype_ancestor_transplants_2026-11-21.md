# M5.3C — Genotype transplantation: founder competence persists; mutation-driven improvement NOT established

**Complete original frozen study, 21 November 2026.**

- [Prospectively preregistered M5.3C protocol](../docs/M5_3C_PROTOCOL.md), **commit `235fa05894c573b76f097073100069c25e172e9b`**, recorded BEFORE study code and any new-world evaluation.
- [Successful 900-trial read-only GitHub Actions original campaign #44924136791](https://github.com/kkongiu/neuroevolution-lab/actions/runs/44924136791), **165 tests passed**, all five origin families, 24 NEW main and 12 NEW shifted worlds, 400 ticks, five fixed numerical genotypes per origin.
- [Permanent exact frozen source genotype bank](M5_3C_frozen_candidate_genotypes_2026-11-21.json), [all 900 held-out world trial outcomes](M5_3C_world_transplants_2026-11-21.csv); [implementation](../neuroevolution/m5_3c.py), [regression tests](../tests/test_m5_3c.py), [PR #38](https://github.com/kkongiu/neuroevolution-lab/pull/38).
- Original run artifact `m5-3c-genotype-ancestor-transplant`, ID **14715087107**, retains 30 days: `frozen_source_candidate_bank.json` (full original genealogy and exact gene vectors), `summary.json`, `world_metrics.csv` (all original population/food/death metrics for 900 trials), `population_checkpoints.csv`, complete birth/food histories for the first new world of each shifted/ordinary split/family/arm, and frozen numeric genotype hashes.

## 1. Why this study was necessary

The M5.2A experiment established poor neural foraging as the original ecological bottleneck. M5.3A proved 401-weight and even 51-weight numerical neural controllers **can** represent viable ordinary-world foraging when their genes are engineered or externally taught, but not evolved in-world. M5.3B finally allowed random ten-gene founders to reproduce, mutate and undergo real ecological competition inside a strongly researcher-biased directional neural architecture. Its main survival: **5/24 worlds sparse mutation, 5/24 dense, 3/24 exact clones and 0/24 random newborns**. Thus inheritance **as a whole** helped versus fresh random offspring, but **mutation advantage over exact clones FAILED**.

M5.3C was frozen to test whether M5.3B's particular selected final descendants acquired a transferable numerical neural phenotype genuinely better than their **own founders**, rather than simply retaining selection-favored random founder variants.

**Sample conditioning matters:** source families were deliberately chosen as ALL FIVE PREVIOUSLY KNOWN surviving sparse-M5.3B worlds. This result is conditional on success in those five histories, NOT an unbiased success frequency across random evolution experiments.

## 2. Five predetermined lineages and exact genomes

Replayed only the **unchanged** historical M5.3B source worlds 82006, 82015, 82018, 82019 and 82023. From each, choose the **highest-generation final living individual** (tie lowest ID), trace birth-by-birth back to its originating random founder and check all ten numeric genes, exact mutation indices and original expanded 401-weight genotype SHA digests.

| Original successful M5.3B world | Chosen living descendant ID | Generation | Original founder ID |
|---|---:|---:|---:|
| 82006 | 156 | 16 | 14 |
| 82015 | 154 | 15 | 13 |
| 82018 | 116 | 11 | 4 |
| 82019 | 159 | 14 | 0 |
| 82023 | 128 | 11 | 9 |

For each family, five immutable 10-gene genotypes were frozen before any new-world evaluation: (a) exact selected `descendant`, (b) its own first-generation `ancestor`, (c) `last_step_reverted` in which the final two gene mutations alone are restored to their **immediate** parental values, (d) identical descendant numeric gene multiset shuffled to new gene channels, (e) one independent fixed Gaussian random ten-gene control.

**No phenotype engineering, teaching or genetic mutation during final tests.** All 20 new-world founders receive the exact fixed genotype, with genuine births merely copying it unchanged; all movements and births are selected by the original **unaltered** M5.0 401-weight neural logit evaluator, original 15 local observation channels and original physical energy/legal-action laws. The directional architecture itself is still **human-designed weight sharing**, not evolved from scratch.

## 3. Prospectively held-out MAIN — exact independent world results

**24 entirely NEW MAIN seeds 84000–84023**; each source × five numerical genotype arms tested in every one, 400 ticks, identical food-event candidate schedule and world map per paired heldout seed.

| Original source family | Selected mutated descendant survived | Its actual random founder survived | Last mutation reverted survived | Shuffled genes | Independently random genes |
|---|---:|---:|---:|---:|---:|
| 82006 | **24/24** | 14/24 | 24/24 | 0/24 | 0/24 |
| 82015 | **24/24** | 19/24 | 24/24 | 0/24 | 0/24 |
| 82018 | 15/24 | **24/24** | 17/24 | 0/24 | 0/24 |
| 82019 | **24/24** | 23/24 | 24/24 | 0/24 | 0/24 |
| 82023 | 23/24 | **24/24** | **24/24** | 0/24 | 0/24 |
| **Total, conditional on five successful source families** | **110/120** | **104/120** | **113/120** | **0/120** | **0/120** |

Average across all five source families and 24 new worlds:
- Descendant mean **129.87 real clone births**, **380.23 foods** collected, **11.18 final living**.
- Founder mean **135.37 births**, **359.99 foods**, **10.46 final living**.
- Last-step revert **130.66 births**, **383.69 foods**, **11.08 final living**.
- Scrambled **8.43 births**, **16.25 foods**, **0 final living**.
- Independent random **12.80 births**, **30.66 foods**, **0 final living**.

This positive survival for all five descendants, versus only five rare original heterogeneous evolutionary source worlds, is a **different experimental intervention**: transplanting 20 identical genetically selected individuals to a new world, rather than beginning with 20 separate random competing founders. It does NOT imply M5.3B already solved survival in 110/120 random-founder evolutionary worlds.

## 4. Predeclared hierarchical paired survival comparisons — all 120 MAIN family×world pairs

**Correct independent units:** five historical evolution source families crossed with 24 fresh independently sampled evaluation world seeds. Descriptive intervals resample source families *and* common test-world seed IDs with replacement for **20,000 frozen crossed-cluster replicates** (seed 87001 + predefined contrast offsets). Do not interpret the 120 transplant evaluations as 120 independent origin-family evolutionary replicates.

| Exact genotype comparison | Difference in MAIN survival | 95% descriptive crossed bootstrap | Origin families: positive/negative/tied | Predeclared conclusion |
|---|---:|---|---|---|
| **Descendant − actual founder** | **+5.00 percentage points** | **[−20.00,+30.00] pp** | **3 / 2 / 0** | **PRIMARY FAIL** |
| Descendant − shuffled exact same ten gene values | **+91.67 pp** | **[+75.00,+100.00] pp** | **5 / 0 / 0** | **NUMERIC SPECIFICITY PASS** |
| Descendant − one separately random fixed 10-gene vector per family | **+91.67 pp** | **[+75.00,+100.00] pp** | **5 / 0 / 0** | Secondary positive |
| Descendant − its *immediate parent* (last gene mutation reverted) | **−2.50 pp** | **[−10.00,+1.67] pp** | **0 / 2 / 3** | **LAST-STEP MUTATION GATE FAIL** |

Each of the five descendants individually survived ≥12/24 ordinary heldout worlds (**5/5**). Yet the MAIN primary requires all of: >0 mean descendant-minus-ancestor, lower crossed-bootstrap 95% >0, ≥4/5 source families individually improved, ≥3/5 descendant families with viable phenotype. Only last condition passes.

**No evidence of a reproducible descendant advantage over founder:** two of five mutated descendants actually survive **fewer** world distributions than their specific ancestors; confidence interval includes meaningful negative changes. Moreover, reverting the final birth's two mutations does not reduce survival; if anything, the observed sample mean is slightly more positive for reverted parents (113 vs 110 survival trials). We cannot claim those last-step mutations are beneficial.

**Positive genotype-causality control:** gene channel assignment matters enormously: all original selected genotypes survive some NEW worlds, but the same ten numeric values shuffled into other channels are extinct in all 120 tests. This shows a viable *structured numerical neural genotype*, **not new neural architecture nor proof that mutation rather than favorable founder variation invented the phenotype**. The five fixed independent random control genotypes happen also to fail across 120 matched tests; this limited sample of random vectors is NOT an estimate of the full 10-dimensional random-genome viability distribution.

## 5. Completely unseen SHIFTED environments: null ecological transfer

**12 fresh shifted seeds 85000–85011** per source family, with obstacle density .20, odor range2, regeneration .40 and original unchanged physical rules. Total 5×5×12 = 300 shifted trials.

| Genotype arm | Surviving /60 source×world observations |
|---|---:|
| Selected descendant | **0/60** |
| Actual original founder | **0/60** |
| Final mutation reverted | **1/60** |
| Same genes randomly shuffled | **0/60** |
| One independent random genome per family | **0/60** |

Every descendant and ancestor family is 0/12 for its own shifted 12 worlds; descendant−ancestor mean difference exactly 0, crossed-bootstrap 95% [0,0]. The one surviving reversion is isolated in the family whose M5.3B original source world was 82015 and is not evidence of robust generalized viability. **Shifted transfer gate FAILS.**

Because the shifted ecology changes multiple factors together (walls, odor, regrowth), this experiment cannot isolate the individual cause of failure. We do know the ordinary-world selected genotype and its actual ancestor are both insufficient under this joint environmental shift.

## 6. Decision: heritability retained; de novo improvement not shown

**What is now demonstrated:**
1. Original numerical neural physiology is expressive; M5.3A supplied a viable genotype externally.
2. Real random 10-gene founders can undergo physical birth, honest inheritance and occasional survival, and inherited continuous genomes beat fresh random offspring in M5.3B's original ordinary distribution.
3. Selected M5.3B source descendants have transferable, highly viable structured numerical genotypes on new ordinary worlds (110/120).
4. Their **actual original founder genotypes are already nearly as viable** (104/120); and the final mutations are not helpful on the preregistered survival endpoint.
5. Severe environmental change eliminates all descendant and ancestor lineages in 60/60 held-out shifted observations.

**What is NOT demonstrated:** a statistically and reproductively consistent **mutation-generated improvement** over the original random founder genome; a novel sensor/actor architecture; general-purpose environmental adaptation; open-ended biological evolution, intelligence, consciousness or executable code replication.

**Preferred explanation:** the previous advantage of inherited offspring over fresh random children is largely compatible with **selection and preservation of useful random founder genes** inside a strong researcher-wired directional neural scaffold. This is a coherent scientific result rather than a failed implementation; however, it is NOT definitive exclusion of cumulative adaptation on other fitness metrics or in other evolutionary settings.

## 7. Next independent experiment recommendation

Do **NOT** retroactively retune or reuse heldout seeds 84000–84023 and 85000–85011. A useful separately preregistered next milestone would test evolution **under changing food availability, odor and obstacles within the lifetime of an evolutionary population**, not just transplanting frozen genotypes to an OOD world after training, combined with:
- matched founding-genotype pools and **mutation-vs-clone** control;
- distinct independently sampled developmental, primary, and shifted worlds; fixed budgets and exact causal genotype comparisons;
- explicit survival, lineage replacement, new beneficial mutation spread and ability to maintain food intake after a shift;
- architectural variants or representations with learned recurrence, BUT no silent scripted foraging or scripted reproduction.

An alternative, less ambitious and cleaner immediate step is an observational **within-ecology founder lineage and beneficial mutation frequency audit** on the original M5.3B birth ledger, without making new survival claims. Either must have its own protocol and untouched sealed test seeds.

**Safety/implementation:** simulated agents cannot access tools, filesystem, networks or source code; only static numeric gene vectors are transmitted at original costed simulated births.

- [FULL independently verified source genealogy and 10-gene ancestry for all five families](M5_3C_full_verified_source_genealogies_2026-11-21.json); [independent read-only source replay verification workflow #44924597561](https://github.com/kkongiu/neuroevolution-lab/actions/runs/44924597561). This permanently preserves each intermediate parent/child gene vector, exact changed gene indices, original genetic SHA digests and original world checkpoints, independently verified against the exact previously frozen finalist-genotype bank. The lineage audit performed no M5.3C heldout-world evaluation.
