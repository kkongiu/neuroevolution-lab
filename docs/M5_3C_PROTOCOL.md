# M5.3C — Frozen prospective genotype-transplant protocol

**Preregister BEFORE implementation or opening any M5.3C held-out ecological outcomes.** Sources: M5.3B frozen experimental report / commit `99d0bc3ef7c58f3af8119145fabf1da947a29126` and related [issue #37](https://github.com/kkongiu/neuroevolution-lab/issues/37).

## Question and conditional inference

M5.3B genuine in-world genetic trials had only five sparse-mutation surviving worlds among 24 (82006, 82015, 82018, 82019, 82023); three original-world clone controls also survive. Thus in-world **inheritance versus random newborns** was favorable, but **new mutations versus exact cloning** and overall survival gates failed. We seek a DIRECT CAUSAL *genotype* test of whether selected surviving **mutated descendants** acquired transferable fixed neural phenotypes better than their **actual ancestors**. We explicitly condition on *already known successful source worlds*: results cannot establish how frequently random evolution succeeds without that conditioning, and cannot retroactively change M5.3B preregistered null gates.

## Strictly frozen source selection, all five worlds (no cherry picking)

1. Replay the original **unaltered** `neuroevolution.m5_3b.simulate` with original settings and `arm="sparse_2"` in the **five** previously published successful 400-tick MAIN source worlds **82006, 82015, 82018, 82019, 82023**; no other origins or external trained genes. Every origin must be genuinely alive at tick400, or abort.
2. Among surviving agents at tick400, select **the highest generation**, ties resolved by **lowest numerical agent ID**. Do NOT select based on genotype values, test-world fitness or expected attractiveness.
3. Trace its original birth parentage back to founder. Recover the *exact ten-gene founder vector*, every intermediate mutated child vector and chosen final descendant vector. Assert exact `m5_3b.gene_digest` and `m5_0.genome_digest(m5_3b.to_weights(genes))` for all original births. Record source seed, finalist ID, generation, founder ID, complete lineage path and each mutated gene index.
4. Create **five fixed test genotypes per source**, all using the same original researcher-designed 10-gene → 401 numerical neural architecture:
   - `descendant`: the selected living final child's exact 10 genes; **only this arm has real historical mutations**.
   - `ancestor`: its own original founder's ten genes (paired origin, not another founder selected because it performs poorly).
   - `last_step_reverted`: descendant vector with exactly the two mutated positions in its **final** birth restored to that immediate parent's values; tests last mutation step specifically, not entire ancestral trajectory.
   - `scrambled`: original descendant's 10 numerical gene values uniformly permuted using `random.Random(95000 + source_seed)`, preserving multiset but breaking gene-channel assignment.
   - `independent_random`: an independent random 10-gene vector N(0, .65) generated once using `random.Random(94000 + source_seed)`, fixed in all held-out worlds. This is a genotype baseline, not an evolutionary arm.
5. The five origins' selected candidates and all controls must be frozen automatically by the deterministic algorithm **before any held-out world simulation** and written into a permanent full-precision JSON or CSV. Abort if final generation 0, missing genealogy, incorrect genome digest, unchanged final mutation, identical scrambled vector or source seed leakage. No candidate adjustment after evaluation.

## Held-out transplant ecology: no additional evolution

Clone each source genotype unchanged into ALL 20 initial founders (reset 12 hidden states). Clonally inherit it unchanged on true births (no mutations). **ALL movement/reproduction decisions use unchanged M5.0 `act` five numerical neural logits** and legal action mask; unchanged physical food/obstacles/energetics. A transplant test asks whether a genetic *phenotype* survives, not whether selection can further optimize it.

- **MAIN:** independent held-out **24 new worlds 84000–84023**, 400 ticks; all 5 candidate families × 5 fixed treatments = **600 world-genotype trials**.
- **SHIFTED:** independent held-out **12 new worlds 85000–85011**, 400 ticks, same 5 candidate families ×5 treatments = **300 trials**. Shifted = original .20 walls, odor radius2, regrowth .40; main .12 walls, radius5, regrowth .55. Other original settings 15×15, 20 founders, max64, food20/cap32, 2 regeneration attempts, initial energy36/max54, food+18, basal cost1, birth cost20, child energy12, age>=4 and energy>=34 to give birth, max-age100.
- Presample exact food events per held-out seed with original M5.0. All candidate arms for each seed see the **same world, food candidate schedule, founder positions** and all identical legal physics. No test-world seed is used in candidate choice, parameter tuning or learning.
- Engineering-only smoke evaluation: **MAIN 700–701, SHIFTED 900–901**, <=40 ticks on 11×11. Smoke may replay original 400-tick source worlds because provenance must match; *never* publish smoke ecological outcomes as evidence. Every held-out seed must be disjoint from original source, historic M5.3B or M5.3A training/test seeds.

## Preregistered metrics and inference

Primary outcome is **survival of >=1 organism at tick400** (binary) for each (source family, held-out world seed, fixed genotype arm). Secondary: population alive, births, food eaten, longest generation, energy ledger and genotype-specific aborts. Rows are **not 600 independent source-replicates**: only **five independently selected original survivor families** and 24 independent test environments were sampled. All aggregate inference is conditional on five successful source worlds.

For each origin family compute paired mean survival difference across the 24 unseen main seeds for `descendant - ancestor`; also `descendant - scrambled`, `descendant - independent_random`, and `descendant - last_step_reverted`. Aggregate equally over the five **families** (not agents or births), then 24 common held-out world seeds. Bootstrap **20,000 deterministic crossed-cluster resamples**: resample five source family IDs with replacement AND the 24 common evaluation seed IDs with replacement using RNG seed **87001**, preserving paired arms and common worlds; report descriptive 2.5/97.5 percentiles. Also publish per-family deltas, counts of positive/negative/ties, and the exact 24 seed-level outcomes.

**PRIMARY descendant genotype improvement gate** passes only if:
1. Mean `descendant - ancestor` paired MAIN survival >0;
2. crossed-cluster bootstrap lower descriptive 95% strictly >0;
3. **at least four of the five** independent source families have positive individual MAIN survival difference, strictly more than negative;
4. **at least three of five** descendants survive **>=12/24** of the new MAIN evaluation worlds.

**REQUIRED numerical specificity control:** `descendant - scrambled` MAIN lower crossed-bootstrap >0, ≥4/5 families individually positive. Supports numerical genotype assignment as a cause; cannot itself prove mutation improved ancestor.

**Last-step mutation ablation (secondary, NOT part of primary):** `descendant - last_step_reverted` MAIN lower crossed-bootstrap >0 and ≥4/5 families positive. Failure means the final mutation step is not uniquely supported, but earlier mutation accumulation may still matter.

**Independent random genotype contrast (secondary):** report `descendant - independent_random` with same bootstrap and family signs; no inference to the entire random-genome distribution from only one sampled random vector per source.

**OOD robustness (separate required transfer gate):** at least 3/5 descendants each survive ≥6/12 SHIFTED worlds AND `descendant - ancestor` shifted mean >0 and bootstrap lower >0 with at least 4/5 source families positive. It may fail independently of the MAIN primary.

**Do not conflate selection conditioning with causal novel mutation discovery:** any MAIN primary passing establishes that these five source-selected final descendant **numerical genotypes** have a transferable phenotypic benefit over their particular founders, conditional on successful sources. It cannot alone establish that all improvements came from adaptive evolutionary selection rather than chance mutation with survival conditioning; stronger within-world matched mutational controls may still be needed. Full natural-evolution success requires consistently viable randomly initiated populations and OOD transfer.

## Output and code integrity

- Preregistration commit SHA must precede candidate recovery implementation and test-world run.
- Preserve full-precision candidate vectors, complete parent chain, source genotypes and digest checks; permanently commit compact selected candidates + **900 per-world genotype survival/birth rows**, all family/seed comparisons, actual positive and negative results in repository; archive detailed 50-tick population checkpoints, source lineages and original real-birth ledgers via read-only GitHub Actions for 30 days.
- Test deterministic replay, no overlap, no births off M5.0 physical law, exact clone genotype digest on EVERY transplant birth, food candidate schedule and map equal per paired origin/seed, energy ledger residual zero.
- If any primary gate fails, explicitly label M5.3C a failure of prospective mutation-driven phenotype improvement and retain honest nulls. No retrospective seed/genome tuning, no modification of historical M5.0–M5.3B laws or reports. No simulated agent ever has network, filesystem, source code or executable replication abilities.
