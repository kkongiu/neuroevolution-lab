# M5.3B — Frozen protocol: de novo neural genetic search with a 10-gene shared-direction neural architecture

**Preregister before code or any final M5.3B world score.** M5.3A found 401-parameter and even 51-parameter networks can represent viable ordinary-world policies if externally engineered/trained; those provided-genotype phenotypes FAIL all shifted worlds. M5.2B failed to evolve even a 51-weight policy from random founders. M5.3B asks whether a highly restricted, direction-shared *neural architecture* reduces the random-genotype evolutionary search barrier enough for **within-world mutation/selection** to help. **No pretrained or human-engineered genotype is transplanted into any experimental organism.**

## Scientific interventions

Each organism has a **numerical ten-gene inherited genome**, selected at random for each of 20 founders, initially Gaussian N(0, 0.65). Its genes are mapped mechanically to **a 12-hidden-unit, 401-weight numerical tanh network** consumed by the exact unmodified M5.0 `network_logits` and `act`. All recurrent weights remain exactly zero; the controller belongs to the feedforward subspace of the same 5-output recurrent architecture. *Within-lifetime* weights never change. Only **real births** can transmit/mutate the 10 underlying numeric genes. Original ecological physical birth/death/food/motor laws unchanged.

**Frozen gene-to-RNN map (10 evolvable genes):**

- `g0` adjacent-food gain, `g1` directional odor gain, `g2` blocked-direction gain, `g3..g6` four independent directional biases. Each motor hidden unit `i=0..3` computes `tanh(6*g0*adjacent_food_i + 3*g1*odor_i - 2*g2*blocked_i + g(3+i))`. Original motor logit `i` is `1.4*hidden_i`.
- `g7` energy-response gain, `g8` reproductive hidden bias, `g9` reproductive-output gain. Birth hidden `4` computes `tanh(8*g7*(normalized_energy - 0.78) + g8)`; birth logit is `2*g9*hidden_4 - 0.2`.
- All other 401 numeric RNN weights are zero, including recurrent couplings; there is NO special runtime move instruction. `m5_0.act` still picks maximum **legal** neural output amongst movement and REPRODUCE. Reproduction eligibility, cost, age, population and location remain pure M5.0 physics.

**This mapping has a strong RESEARCHER-DESIGNED inductive bias**: shared food and odor gains across directions, a privileged energy channel to birth and fixed positive output coefficients. It makes navigation *architecturally easier* but does not set any gene to a food-following or birth-positive value. Random genes may make odor aversive or favorable, and birth responses positive or negative. **Do not describe discovery of input wiring or the food reward as evolved.** Only beneficial selection among these ten freely signed random genes and their mutations can be evidence of de novo numeric evolution.

## Four independent genetic transmissions in the same world

1. `sparse_2` — PRIMARY: at each legal birth, child inherits parent `g`; randomly choose exactly **2 distinct indices of 10**, add independent Gaussian N(0, **0.16**) to each.
2. `dense_10` — same 10 genes and parent but independent Gaussian N(0, **0.16**) perturbation to ALL ten coefficients at each birth.
3. `clone` — exact inherited genes; no novel mutation (founder-variation-selection control).
4. `random_child` — at each physical birth, draw all 10 genes independently from N(0, **0.65**); no parental behavioral inheritance.

Every arm starts with the same 20 founder 10-gene vectors, same physical founder positions, food placement candidates, walls and action ordering under each paired world seed. Each agent is truly independent in continuous ecology, including real energy-expensive parent/child birth events; there is **no external elite selection, no artificial fitness assignment and no externally chosen children**. Natural survival, ecological food access and **neurally chosen reproductive actions** are the only selection pressures. Child genotype is generated AFTER unchanged M5.0 physical clonal birth, then the child's full RNN weights are expanded from the new 10 genes before it can act. Event digest, parent-child cosine and genealogical records refer to the final expanded neural weights.

## Original ecology, matched budget

Exact M5.0 physical world: 15×15, 20 founders, capacity 64, initial food 20, capacity 32, per-tick two candidate food regrowth events .55 normal/.40 shifted; obstacle fraction .12 normal/.20 shifted, odor radius 5 normal/2 shifted; energy initial 36, cap54, food +18, metabolism -1, parent birth cost20, newborn energy12, legal reproduction age≥4 and energy≥34, max age100, **400 ticks** per world. No source changes to historical experiments. Food events are presampled before action, genotypes cannot modify resources or world rules.

## Frozen seed ranges and gates

- ENGINEERING ONLY smoke seeds **700–702**, 40 ticks on 11×11, never included in evidence.
- MAIN **24 NEW independent world seeds 82000–82023** × four genetic arms ×400 ticks.
- SHIFTED **12 NEW independent world seeds 83000–83011** × four arms ×400 ticks, fixed OOD altered walls/odor/regrowth. Never use previous M5.3A teacher/genotype/test seeds.
- Full per-world CSV and population checkpoint 0/50/100/150/200/250/300/350/400, genealogy/genome trace and sampled exploratory metrics. Independent statistical unit **whole world seed**, not individual organisms, ticks, births, or mutation events.

**Primary survival/evolution gate**, all required:
1. `sparse_2` population genuinely alive at tick 400 in **≥12/24 main worlds** and generation≥3 in ≥12/24;
2. `sparse_2 - random_child` MAIN *world survival (0/1)* positive mean, lower descriptive paired bootstrap 95% **strictly >0**, and positive worlds > negative worlds.

**Necessary novel-mutation gate:** `sparse_2 - clone` MAIN world survival positive mean, lower descriptive paired bootstrap 95% >0, and more positive than negative independent worlds. If sparse survival but this fails, selection may be acting on **existing founder genotypes**, not generating new capabilities. `sparse_2 - dense_10` is a prespecified **secondary** comparison; it cannot replace failed primary. Births, longevity, maximum generation, food, parental genetic cosine and lineage concentration are also secondary/exploratory, NOT a substitute for a failed survival gate.

**OOD transfer** must show sparse survives ≥6/12 shifted worlds *and* beats random_child on shifted paired survival with positive bootstrap lower bound and majority positive signs. A main-only success without shifted success cannot be called robust ecological evolution.

**Causal phenotype requirement for a genuine adaptation claim:** if primary + mutation gates pass, a *separately preregistered* M5.3C on further **unseen** test worlds must transplant selected descendant genotypes as fixed founder genomes versus ancestor, independent random and scrambled-gene controls under matched worlds/budget. This cannot be decided retroactively on current sealed worlds. M5.3B alone can demonstrate population-level numeric hereditary advantage within one specified ecology, not open-ended evolution.

## Engineering integrity

- Numeric genotypes carry only 10 finite floats, mapping deterministic and fixed; actions are computed by original 401-weight M5 neural network and motor mask, NOT by the gene map directly.
- Tests verify initially matched founders/schedules, per-child exact mutation scope, genealogical hash/Cosine alignment, deterministic replay, ledger closure, no two living bodies overlap or move through walls, no offspring invented outside legal `m5_0.act`.
- Each simulated agent has no tools, code/file/network access, self-editing source, changeable task objective or executable self-replication.
- All main/shifted outcomes preserved even when zero-survival. No retuning gene map, initial SD, mutational SD, thresholds or sealed seed ranges after evaluation. Read-only Actions campaign only after a successful full CI run.
