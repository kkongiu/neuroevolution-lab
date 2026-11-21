# M5.4B0 — Frozen short-odor numerical-neural representability experiment

**Preregistered BEFORE M5.4B0 simulator/search implementation and BEFORE any held-out outcome.** Date 21 November 2026. Relevant prior: [M5.4A](../reports/M5_4A_ecological_factorial_2026-11-21.md), [research issue #41](https://github.com/kkongiu/neuroevolution-lab/issues/41).

## Question

Can the EXISTING original M5.3B ten-gene, 401-weight, fixed directional, feedforward-subspace numerical neural architecture represent a survival-competent policy with **odor radius TWO cells**? Previous M5.4A showed a strong −81.25 percentage point survival collapse when shrinking radius5→2, and a researcher-SCRIPTED novelty-search ecological positive control could still survive the harder world. An incapacity to express short-odor exploration is a serious possible impediment to **true in-world** adaptation. We distinguish:
1. numerical architectural representational feasibility (M5.4B0, this study), optimized **EXTERNALLY**;
2. de novo genetic adaptation by real within-world mutation/selection, **not** tested by M5.4B0.

No positive B0 result can be called natural evolution, an evolved novelty strategy, or architecture learning.

## Hard-frozen policy and original physical ecology

- EXACT unchanged `m5_3b.to_weights`: freely signed **ten numeric genes**, compiled into original `m5_0` 401-weight, 12-hidden neural processor. The existing hardwired 4 directional odor/adjacent food/wall weights, four direction biases, one reproductive energy response, output scale and output bias are **programmer supplied**. No new recurrence or visit memory can be implemented by genetic search; the original scaffold sets recurrent weights to zero.
- EXACT original `m5_0.create_ecology`, world generator, `m5_0.act` (neural logits for all four movement + reproduce preferences), original energy/food/birth laws and `_food_schedule`; unchanged 15 local sensor inputs. All newborns get exact numerical **clones** of selected fixed genotype on a birth physically permitted by original energy/age constraints. No mutation or external action preference is injected during evaluation.
- **MAIN short-sensor ecology**: world size15, obstacle density .12, initial food20, food cap32, regeneration chance .55, initial 20 agents, max64, initial energy36/54 cap, food energy+18, movement cost1, newborn energy12/birth dissipates20, minimum reproduction energy34/age4, max-age100. **Odor radius2 from tick0**.
- **SHIFTED short-sensor OOD**: wall density .20, regeneration chance .40, SAME odor radius2. Same original, not M5.4A matched nested-map sampling. This original M5.0 world generator is deliberately NOT identical to M5.4A custom controlled food-support geometry. Do not conflate the absolute survival rates.
- Full **400 ticks** in search, final selection and independent sealed test worlds, even when a trial goes extinct early. Extinction and zero fitness are retained.

## External developmental search — fixed sampling and no test-world leakage

- Search RNG **88191**, **48** independently random 10-gene candidates N(0,.65).
- External optimization **three** rounds of **16** mutated candidates each, for exactly **96 unique evaluations** in candidate index order. Maintain the best six candidates by fixed development objective; each new child selects a parent from those 6 in cyclic order and changes exactly **two** randomly selected distinct gene indices by independent Gaussian N(0,.22); selected parental genes do not mutate. The externally ranked/selected candidates are a **researcher's optimization algorithm**, and the evolutionary phrase must NEVER be used for the simulated ecology. All descendants **inside** any test world are exact clones.
- Each candidate tested on EXACT same **4 DEVELOPMENT seeds 88000–88003** in MAIN physical ecology with odor2, 400 ticks. Candidate score is the lexicographically ranked tuple:
  1. number of development worlds surviving tick400 (maximize);
  2. total surviving population across four worlds (maximize);
  3. total food eaten (maximize);
  4. total native clone births (maximize);
  5. lower ordinal candidate ID (deterministic tie-break).
  No manual hand-picking, early-stop, seed changes or adjustment after viewing outcomes.
- **Validation selection**: strictly top **6** by development ranking, assessed on untouched **8 VALIDATION seeds 88100–88107**, still MAIN physical world + odor2, 400 ticks. Choose highest tuple (validation surviving worlds, validation final population, validation food eaten, validation births, development ranking, lowest candidate ID). Freeze selected exact ten genes and expanded digest **before** sealed evaluation.
- No supervised imitation, no prebuilt M5.1 novelty heuristic, no externally trained M5.3A genes, no previously successful hand-engineered founder supplied. Search's fitness feedback is allowed ONLY in development and validation, making this clearly **external, researcher-guided optimization**. The actual M5.3B natural population never receives optimization gradients or privileged world knowledge.

## Final held-out, non-reusable numerical phenotype tests

- **MAIN:** 24 entirely fresh sealed world seeds **88200–88223** with only short odor2, original ecology.
- **SHIFTED:** 12 further entirely fresh sealed world seeds **88300–88311**, original denser walls and low regrowth with odor2.
- Each world compared on 7 fully independent **fixed** policy arms: one selected externally optimized ten-gene witness; same ten genes **scrambled** via deterministic `random.Random(88470)` positional permutation; four fixed independently random numeric ten-gene controls N(0,.65), each with RNG seeds `88600 + i` (i=0..3); original original-world M5.1 researcher-written **greedy_novelty** physical positive benchmark (with scripted movement/reproduction). Distinguish scripted positive reference from ALL original neural arms. A scrambled gene arrangement that accidentally equals the original aborts.
- Same held-out world seed yields paired map, founders, food schedule and physical actor permutation across neural arms. Script controls use same original M5.0 world, food schedule and physical birth/energy, but a different motor and reproduction policy.
- Main 24×7=168 and shifted 12×7=84 full policy-world trials. Original development 96×4=384 and independent validation 6×8=48; full experiment **684 simulations** plus unit tests. Never use held-out seeds to rerank the selected genotype.

## Frozen analysis and inference

- Primary evidence of REPRESENTABILITY: witness survives **≥12/24 MAIN worlds at tick400**, has real native births in **≥12/24 MAIN worlds**, and its 24-world mean binary survival exceeds the four random-genotype arms' average by **≥25 percentage points** with lower 20,000-world-paired bootstrap 95% **>0**. Random-control sources are only 4 fixed numeric vectors; bootstrap over new worlds does NOT imply coverage of the whole Gaussian genotype distribution.
- Independently require positive **human-scripted ecological viability** (scripted positive control survives ≥18/24 MAIN); otherwise interpret failures as possible ecology limitation.
- OOD is **separate**: witness survives ≥6/12 SHIFTED worlds at tick400 AND improves on average over random numerical controls with lower 20k paired bootstrap 95%>0. Failures remain explicit.
- Scrambled reference: report paired differences, exact survival and food; if witness beats scrambled on new worlds it shows positional numerical gene specificity but **not** automatic learning or within-world evolutionary adaptation.
- World-paired bootstrap of **20,000** replicates, RNG seed `88701` MAIN; `88702` SHIFTED. Report all 24+12 paired world outcomes (births, food, surviving individuals, genetic digests, physical energy balance and extinct tick), full original candidate search seed/parentage and development objectives and all eight validation world scores. Test deterministic replay and exact clonal birth digest. Do not interpret the 24 test worlds as 24 independent genotype-training searches.
- **If primary representational gate fails**, conclude numerical expressivity for this architecture **not demonstrated under fixed budget**, do not assert mathematical impossibility. Do NOT proceed to a purported successful natural mutation adaptation under this architecture without separately justified representation research; next review would be a new bounded recurrent/numeric memory architecture with totally new protocol. If primary passes, M5.4B1 true within-world mutation/selection can be preregistered independently, with random uninformed founders, never seeding this externally optimized witness into the actual evolutionary population.
- All findings must distinguish original programmed neural wiring from evolved numeric weights, from external optimization, from researcher-scripted physical benchmarks. No simulation agent can run arbitrary code, alter source, contact tools or networks.
