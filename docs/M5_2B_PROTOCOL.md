# M5.2B — Preregistration: can a compact, genuinely neural genotype evolve ecological persistence?

**Frozen before M5.2B implementation and before inspection of all M5.2B final-world outcomes.** Follow-up to the completed [M5.2A causal motor diagnosis](../reports/M5_2A_causal_factorial_2026-11-21.md): scripted motor search made otherwise neural populations persist (GN 20/20 versus NN 0/20). Here **every motor AND reproductive decision must be determined by the agent's own inherited neural weights**. No scripted odor-following, scripted birth threshold, external fitness selection or post-hoc parameter tuning.

## Mechanistic hypothesis

The previous 12-hidden-unit RNN has 401 weights, all perturbed by Gaussian noise SD 0.12 at birth. With a population bottleneck and limited births, this high-dimensional global mutation may destroy partial skills faster than they accumulate. A **smaller recurrent controller (2 hidden tanh units, 51 free weights)** and sparse mutation of only three coefficients per birth may be more evolvable, even though it is *less expressive*. This is a falsifiable hypothesis, NOT a guarantee. Hidden=2 is a new and restricted architecture, but all 15 original local sensors and five output action logits remain available. No action mappings, special foraging heuristics, controller priors, weights or thresholds are hardcoded.

M5.0 game physics remain unchanged (15×15, 20 founders, max population 64, food and obstacle parameters as prior protocol, lifespan 100, energy accounting and reproduction costs), 400 ticks. In this experiment the **original M5.0 action implementation** selects both movement and REPRODUCE using the neural logits, with legal action masking. Weight changes occur only at **real in-world births**, never during life. No fixed-genome reward assigned by an external observer and no elite/resampling of artificial generations.

## Arms: matched 51-parameter RNN architecture, same founders

1. **sparse_inherited** (PRIMARY): child receives parent's 51-weight genome, exactly THREE distinct randomly selected weights altered by N(0, 0.12). The remaining 48 copied verbatim. Child hidden state resets to zeros as in M5.0.
2. **dense_inherited**: original inherited Gaussian N(0,0.12) mutation to **all 51** weights at birth, original M5.0 birth mechanism with only hidden=2.
3. **clone_inherited**: exact clonal 51-weight copy with no novel mutation (standing-variation selection control).
4. **random_offspring**: wholly independent fresh 51-weight Gaussian N(0,0.45) genome per birth (inheritance negative control).

All four arms use identical initial genome vectors, founders, positions, walls, **candidate** food spawn schedules, energy, births, move legality and action ordering for a given world seed. The actual food locations diverge because policies choose different paths. The **only intended treatment difference** is the child-genome transmission operator. The arm names never alter the physical reproductive mechanism.

For sparse inheritance, the adapter will first call unchanged M5.0 act using a clonal newborn, then replace **only** the newly born child's numerical genome with a deterministic three-coefficient perturbation **before its first action**, updating the event digest and birth cosine so the historical trace remains accurate. Its parent, energy, birth time/location, lineage and physical resource flow are untouched. Tests must verify exact provenance and conservation, and verify non-sparse arms match original M5.0 exactly.

## World splits, all sealed

- Engineering smoke only: world seeds 700–702, 40 ticks, 11×11, never reported as scientific success.
- **MAIN**: 24 fresh paired worlds, seeds **77000–77023**.
- **SHIFTED**: 12 fresh paired worlds, seeds **78000–78011**, obstacle density 0.20 vs .12, odor radius 2 vs 5, regeneration probability .40 vs .55.
- All four arms complete 400 ticks; same fixed settings 2 hidden units (51 weights), 20 founders; initial weight SD .45, newborn mutation SD .12, sparse mutation count exactly 3.
- Checkpoints 0/50/100/150/200/250/300/350/400 with populations, food, ledger and genealogy. Archive full CSV/JSON/SVG, independent seed-level comparisons and representative birth events.

**Do not tune architecture, mutational scope, SD, birth eligibility, energy, world parameters or seed partitions after reading the final metrics.** No comparison with M5.2A should be treated as matched-seed proof of relative performance because different seed sets and architectures are used.

## Success gates preregistered

**PRIMARY ecological/heritability gate**: On 24 MAIN worlds, sparse_inherited must:
1. Have at least one genuinely neural organism alive after 400 ticks in **≥12/24** worlds;
2. Outperform random_offspring in paired MAIN *world-survival indicator* difference by **positive mean**, **95% paired descriptive bootstrap lower limit >0** AND more positive than negative independent worlds.
3. Reach at least generation 3 in **≥12/24** MAIN worlds.

All three conditions are required. A population merely surviving because of its random founder genotypes is *not* evidence of beneficial new mutations. The **additional novel-mutation gate** requires sparse_inherited to outperform clone_inherited on the same world-level survival metric with positive lower 95% descriptive interval and majority non-tied world signs. If that fails, at most *heritable selection among founders* is supported, not improvement from mutations.

**SECONDARY**: report sparse minus dense survival, births, generations, food eaten, descendant share, extinction ticks, and sensory/motor corrections; there is NO substitution of attractive secondary indicators for a failing primary. Distinct harder-world transfer requires sparse to survive **≥6/12 shifted worlds** and to beat random offspring on paired shifted survival with positive lower confidence bound.

**GENOTYPE-CONTROL AUDIT**: mutated weights must be changed strictly at real newborn birth, unmodified thereafter, and actually affect neural decisions; test a same-state neuron whose output changes when a causal action-head coefficient is overwritten. After a successful campaign, a separate M5.2C preregistered genotype-transplant experiment on unseen worlds would be required to demonstrate transferred learned phenotype rather than favorable founder luck. M5.2B success alone never proves open-ended evolution, autonomous goals or consciousness.

**Statistical unit**: world seed, not any agent/time tick/birth. Paired deterministic worlds, bootstrap over world differences, report all null/negative results. Strong advantages need reproducible evidence; 36 worlds are a limited scientific sample and bootstrap intervals are descriptive.

## Integrity and safety

All original source paths M5.0/M5.1/M5.2A untouched; no tools, shell, file/network access or code modification by simulated organisms. Exact resource ledger, no collisions/illegal births, newborn ID/parentage and actual genome digests checked in tests. Fully read-only dedicated GitHub Actions run after passing standard validation. Never relabel positive scripted behavior as evolved neural control.
