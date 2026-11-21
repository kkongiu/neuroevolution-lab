# M5.4B0-R — Eight new inheritable numerical feedback genes: genuine recurrent neural memory, but robust short-odor survival gate FAILS

**Original complete preregistered study, 21 November 2026.** Main research finding: a modest, rare 18-gene numerical-neural survival improvement is observed relative to the EXACT same ten genes with recurrence disabled, but uncertainty still includes ZERO and no trained numeric network transfers to harder short-odor habitat. **NO within-world organism evolution was performed.**

## Reproducibility and preregistration

- [Prospective 18-gene protocol](../docs/M5_4B0R_PROTOCOL.md) at original commit **`7ffd8e06510a4c08a80c7610d7e0ff2617aec1a7`**, committed **BEFORE** implementation or evaluating new experimental worlds.
- [Original completed sealed read-only scientific GitHub Actions #44932841187](https://github.com/kkongiu/neuroevolution-lab/actions/runs/44932841187): **193 regression tests PASS**, all **1,152** preregistered full 400-tick neural/ecology trials completed. Original full source+world+checkpoint artifact `m5-4b0r-eighteen-gene-bounded-recurrence`, ID **14717898803**, 30-day GitHub retention.
- [All 288 final world-policy outcomes, including positive and all negative arms](M5_4B0R_all_288_original_sealed_neural_world_outcomes_2026-11-21.csv); [full selected numerical genes, both optimiser histories, 20k matched-world bootstrap stats and gates](M5_4B0R_frozen_recurrent_genes_and_preregistered_statistics_2026-11-21.json); [all 192 full-precision external 18/10-gene candidate vectors and their external mutation ancestry](M5_4B0R_all_192_external_numeric_genomes_and_ancestry_2026-11-21.json).
- [All 864 ORIGINAL development (768) and validation (96) independently reproduced source-world numeric outcomes, COMPLETE per-world CSV](M5_4B0R_all_864_reproduced_external_development_validation_world_outcomes_2026-11-21.csv). Supplementary **SOURCE-ONLY** provenance [GitHub Actions audit #44933450090](https://github.com/kkongiu/neuroevolution-lab/actions/runs/44933450090) **SUCCESS**, independently regenerated and verified all source full-precision 192 numeric genotypes, candidate ancestry, original survivor/food/birth selection and all 864 world outcomes. It enforced an executable source-seed fence, expressly rejecting ALL 24+12 final-world seeds and any shifted ecology, so this did not add final-world tests, revise original winner, or change original scientific hypotheses.

## 1. Eight actual heritable recurrent neural weights, NOT a programmed forager

The *original exact* M5.3B `m5_3b.to_weights(g0..g9)` ten-gene programmer-chosen numerical directional food/wall/odor/birth-response scaffold is untouched. The M5.0 401-weight / 12-hidden numerical processor is also unchanged, including its 15 local inputs, five action logits, physical movement legal masking, food, age/energy, real birth and long 400-tick ecological horizon.

**New genes 10–17 specify exactly eight recurrent 401-weight slots:**

For each directional hidden neuron index i∈{0,1,2,3}, the previously all-zero recurrent part becomes

`h_i(t) = tanh(original_local_sensor_drive_i(t) + 0.65*tanh(g[10+i])*h_i(t−1) + 0.35*tanh(g[14+i])*h_((i+1) mod 4)(t−1))`.

This is **genuine numerical hidden-state recurrence**, with four bounded self-feedback and four bounded cyclic lateral-feedback weights, sum absolute incoming recurrent weights never exceeding 1.0. It is **NOT** a hard-coded visited-cell table, position, map, target, human movement policy, or artificial program installed inside the genome. The remaining 397 full weights stay as before (except these exact 8 recurrent slots).

Every organism's actual four movement logits and reproduction preference come solely from original unmodified `m5_0.network_logits` and `m5_0.act`. Every native costed birth inherits the exact same 18-gene-expanded 401-weight genome (hidden activations start at zero), with **NO within-world new mutations**. Reproduction, energy ledgers, food and physical movement are the same original M5.0 original-random-map ecology.

**Critical exact unit test:** turning off recurrence for the SAME entire 18-gene vector returns `m5_3b.to_weights(first10)` **bit for bit**, across all 401 weights, with every neural action/physical outcome identical under clamped and original first-ten-gene-only source. The original 10 genes were not reinterpreted.

All 193 historical and newly added tests passed including bounded recurrent slots, actual multi-tick hidden memory, exact same-genome recurrence-clamped negative control, all clonal genotype hashes, physical energy ledgers, all complete world outcomes and strict no-sealed-world access by external candidate search.

## 2. Two independently externally searched models, perfectly equal budget, and exact heldout controls

This **only evaluates numerical REPRESENTABILITY**, not true within-population selection or adapted offspring. Two separate EXTERNALLY researcher-optimized search programs:

- Recurrent **18 numeric genes**, external optimizer RNG **89411**, **96** candidate vectors (48 iid Gaussian and 3×16 externally two-gene-mutated candidates).
- Original exact feedforward **10 numeric genes**, equal independent RNG **89412**, identical external **96** candidate evaluations.
- All **192 candidates** evaluated for 400 ticks on the SAME four new MAIN development worlds 89000–89003: **768 trials**. Develop rank lexicographic 400-tick survival, final population, food eaten, native birth events.
- Each architecture's top six candidates ranked further on exactly eight independent validation worlds 89100–89107, 400 ticks: **96 trials**. Both selected exact full-precision genotypes frozen BEFORE accessing any final worlds. Heldout worlds never used as optimizer selection targets.
- Final eight fixed arms evaluated on exactly 24 NEW MAIN seeds 89200–89223 plus 12 NEW SHIFTED seeds 89300–89311, all odor radius2 from tick0: **288 trials**. Full original M5.0 world generator with ordinary walls .12/.55 food regen or shifted walls .20/.40 food regen. Not the matched restricted food candidate pool used in earlier M5.4A.

Final eight arms:
1. Original native numerical *externally optimized* `recurrent_witness` (18 genes).
2. `recurrence_clamped`: **same 18 genes, same initial map/founders/food schedule, but exactly eight recurrent weights set to ZERO**, nothing else changed.
3. `feedforward_witness`: **separately EXTERNALLY optimized original exact 10-gene architecture**, equal search budget.
4. `recurrent_scrambled`: same selected 18 gene values shuffled across numerical gene roles with fixed RNG89470.
5. `random_recurrent_0`, `random_recurrent_1`: independent fixed 18-gene Gaussian numerical vectors, with predetermined RNG89500,89501.
6. `random_feedforward_0`: independent fixed original 10-gene Gaussian vector with RNG89502.
7. `scripted_greedy_novelty`: separately HUMAN-programmed M5.1 bounded local food scent+personal novelty foraging and reproduction policy. Its actor is NOT original neural and it is **not trained/evolved**.

The human-coded benchmark is categorically distinct. All SEVEN numerical policy arms are genuine original M5.0 network action logits and true energy-costed clonal birth.

## 3. Developmental and independent validation selection

The 18-gene externally chosen candidate is **ID83**. Its preregistered development score over four worlds (400-tick survival count, final living individuals, total eaten food, total native births) is **[0,0,434,80]**; independent eight-world validation **[0,0,1025,204]**.

The equally-budgeted original 10-gene externally selected candidate is **ID64**; development score **[0,0,438,155]** and independent validation **[0,0,630,241]**.

**Neither externally selected final candidate survived ANY of the eight validation worlds.** No developmental recurrent candidate survived any of four development worlds; at least one original original 10-gene development candidate survived one of four development worlds, but the independently validated best selected feedforward candidate itself had zero development/validation survivors. Thus entering 24 new final test worlds with either a positive viable candidate is NOT justified. No hyperparameter, weight mapping, optimizer budget or seed interval was revised in response.

Frozen selected recurrent eighteen numeric genes (indices 0..17):

```json
[0.5523295378582509,0.20934231027063516,-0.6886718689524911,0.5144374371248042,-1.180777372732278,0.2826270979187913,-0.43590019899354193,0.49182316974818424,-0.3391318045753121,1.0713078694321516,-1.0647794580610341,-0.058626430105535504,0.9876198845872525,0.19695953067024138,-1.1635114065663652,-0.08977673847357014,0.19746478736763307,0.13822497092017016]
```

Original expanded recurrent 401-weight digest: **`c5e5795332725de4`**. Clamped version differs only in the eight exact recurrent slots (full SHA/Gaussian controls permanently recorded in scientific JSON and source audit). This genotype was chosen by an external optimizer, not by natural in-world evolution.

## 4. MAIN truly NEW unseen odor-radius2 worlds: a small positive recurrence SIGNAL, not established viability

Final world survival at original tick400, fixed original physics, **24 new independent MAIN world seeds**:

| Numerical neural / separately programmed policy | Survived tick400 /24 | Real costed births per world (mean) | Food eaten/world (mean) |
|---|---:|---:|---:|
| **18-gene genuinely recurrent numeric neural witness** | **2/24** | **26.42** | **131.67** |
| **SAME selected first ten genes, eight recurrence weights CLAMPED** | **0/24** | 11.83 | 46.29 |
| Original 10-gene equally-budgeted externally optimized numeric comparator | 0/24 | 31.04 | 84.83 |
| Selected 18 genes scrambled across slots | 0/24 | 5.63 | 9.63 |
| Random 18-gene numeric `random_recurrent_0` | 0/24 | 2.13 | 4.83 |
| Random 18-gene numeric `random_recurrent_1` | 0/24 | 4.25 | 60.75 |
| Random 10-gene numeric `random_feedforward_0` | 0/24 | 4.21 | 6.83 |
| **HUMAN-programmed local greedy+novelty search** | **24/24** | **116.33** | **403.08** |

Two rare surviving worlds for selected recurrent numeric genotype: **seed89203** (5 living agents, 68 physical native births, 336 food eaten) and **seed89206** (8 living agents, 72 births, 363 food eaten), both still with exact unchanged clonal 18-gene weights.

**Preregistered survivor advantage vs OWN SAME-GENES recurrence-clamped comparator** = **+2/24 = +8.33 percentage points**, deterministic **20,000 WORLD-PAIRED bootstrap descriptive 95% [0,+20.83] pp**, **LOWER BOUND = ZERO**, therefore no support for a positive robust survival advantage by predeclared gate.

**Paired survivor advantage vs equal-budget independently externally optimized ten-gene model** = the same **+8.33 pp**, bootstrap **[0,+20.83] pp**, again no strictly positive lower bound.

All remaining five numerical reference controls also had 0/24 survival and thus the same +8.33 pp paired observed contrast with uncertainty containing zero.

**More food and births are important secondary descriptive signals:** matched recurrence enablement increases *observed mean food eaten* from **46.29→131.67 per world** (~**+85.38** foods/world) and native real births from **11.83→26.42** (~**+14.58 births/world**) relative to the SAME numerical genes with recurrence off. This is consistent with recurrent channels changing ecological trajectories, but there is no independent preregistered paired uncertainty gate for these continuous metrics, and source success remains only 2/24. Moreover, the separately optimized old ten-gene model has MORE mean births (31.04) despite zero 400-tick survivors, showing births alone do not prove viable adaptation.

**PRIMARY numerical representability+recurrence-advantage gate FAILS**: witness ≥12/24 survivor worlds **NO (2)**; native births in ≥12/24 worlds **YES (24)**; human-script ecological viability ≥18/24 **YES (24)**; paired lower95% vs its own memory-off comparator >0 **NO ([0,+20.83])**; paired lower95% vs equal-budget external 10-gene model >0 **NO**.

The 2/24 sample is a weak encouraging observation, not a dependable neural survival policy, not evidence of naturally evolved memory, and not enough to infer statistically stable recurrence benefit on unseen worlds.

## 5. Independent SHIFTED odor-radius2 worlds: all fixed numerical genotypes extinct

On all **12** completely new independent SHIFTED original M5.0 worlds (walls .20, regen .40, odor2):

| Neural / programmed policy | Survived tick400 /12 | Mean native births | Mean food consumed |
|---|---:|---:|---:|
| Externally selected recurrent 18-gene neural | **0/12** | 11.42 | 53.33 |
| Same selected 18 genes with recurrence off | **0/12** | 10.33 | 44.08 |
| Independently externally optimized original 10-gene neural | **0/12** | 16.58 | 33.25 |
| Selected 18-gene values scrambled | 0/12 | 5.33 | 8.83 |
| Both fixed independent random 18-gene numerical controls | 0/12 each | variable | variable |
| Fixed independent random 10-gene numerical control | 0/12 | 4.00 | 6.25 |
| **Researcher-scripted local greedy+novelty physical benchmark** | **12/12** | 87.08 | 294.00 |

Recurrent minus own recurrence-clamped **0 pp**, world-paired descriptive 95% **[0,0]**. Recurrent minus independently externally optimized original 10-gene **0 pp**, **[0,0]**. **OOD transfer gate FAILS** unequivocally in tested hard worlds; [0,0] only reflects universal extinction in observed numerical arms, not equivalence of unseen neural search spaces.

## 6. Scientific interpretation: capacity added successfully, competence not yet independently demonstrated

**Engineering succeeded**: exactly eight bounded, heritable recurrent numeric weights are now implemented and actually influence four hidden neurons over time via the unchanged processor. This provides potential short-term neural state; recurrent weights no longer hardwired to zero, but it does not add a visited-location memory map or guarantee effective spatial search.

**Representational capacity finding remains NOT established:** external search of 96 recurrent candidates did not obtain a viable selected genotype on developmental/validation source environments; exactly 2/24 heldout normal worlds happened to keep a clonal recurrent population alive and none of 12 hard worlds did. Bootstrap lower bound for 2/24 observed paired advantage remains 0. We cannot claim successful adaptive navigation or natural evolution based on this.

**Physical ecology is demonstrably inhabitable:** the HUMAN-scripted positive forager survives 24/24 ordinary and 12/12 shifted, with identical physical energy/food and real births but programmed personal visit memory and programmed reproduction. Its superior ecological survival cannot be automatically attributed to recurrent memory alone, because it also differs in movement tie-break, birth timing and information it stores; no causal isolation of those three components was performed.

**Next independently preregistered hypothesis:** instead of multiplying mutation counts, identify why 18-gene recurrence does not generate robust exploration: targeted *but clearly labeled researcher-defined* coverage/novelty sensor information, a broader recurrent neural state representational search, or a different fitness objective with an independent test set. Any such change must be a NEW prospectively frozen scientific study with new world seeds, real matched controls and no recycled heldout worlds. Only after numerical representability passes independently should new *actual inside-world birth mutations* be tested for adaptive advantage over exact clones, their original uninformed random founders and random newborns under a mid-life 5→2 odor decline. Never seed externally trained numerical genotype into an allegedly random founder population.

All agents remain finite numerical states with no executable code, tools, storage or network access. All original M5.0–M5.4B0 reports/physics are unchanged.
