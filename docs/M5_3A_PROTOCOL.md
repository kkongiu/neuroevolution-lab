# M5.3A — Frozen protocol: can numerical neural weights *represent* a viable forager?

**Preregister before code and before evaluating M5.3A sealed ecological worlds.** Follows completed negative [M5.2B](../reports/M5_2B_compact_neural_2026-11-21.md) and the causal M5.2A motor/foraging diagnostic. **This is a controlled neural representational POSITIVE CONTROL, never a claim of evolution.** Any engineered or externally supervised weights are supplied by the investigator. The organisms in M5.3A do NOT evolve a new skill through mutation or birth/death selection.

## The distinction tested

M5.2A showed an external greedy-odor motor policy (with neural reproduction) sustains 20/20 main worlds, but native random neural genotypes do not. M5.2B reduced recurrent neural parameters 401→51 and mutated three weights at births; all worlds still died. We must distinguish:
- **Representational capacity**: does ANY fixed numerical neural genome with 15 local inputs and the 5 original action logits sustain an ecology?
- **Evolutionary accessibility**: could unguided mutation and in-world natural selection *discover* such a genome? **M5.3A does not answer this second question.**

## Frozen physics and physical birth control

Use M5.0 15×15 and original local 15 sensor channels, motor/action mask and **unaltered `m5_0.act`**. Same 20 founders, max 64, initial 20 foods, cap 32, two renewal candidates/tick probability .55 main/.40 shifted, wall densities .12/.20, odor radii 5/2, age max 100, initial energy 36, max 54, food +18, basal -1, reproduction energy -20 parent/+12 child, birth allowed physically age≥4 and energy≥34 when there is free adjacent space, 400 ticks.

For this **fixed-genome representational test**, transplant *one complete numerical genotype* to all 20 founders before the first move; copy that exact genome to each child through unchanged M5.0 **clonal** inheritance. This eliminates evolutionary adaptation by design and tests only the phenotype's ability to forage, survive and reproduce. The network alone chooses *all four motor actions and REPRODUCE* from its logits. No scripted motor or scripted birth decision is substituted during the ecological evaluation. Hidden state resets at birth and evolves within life normally. Recurrent matrices are permitted but trained/engineered candidates use their zero-recurrence feedforward subspace. Clones are deliberate here, not proof of evolution.

## Frozen genotype treatments

Six fixed treatments (NOT adaptive evolutionary arms):

1. **engineered_12 (PRIMARY positive witness):** 12-hidden, 401-weight numeric RNN with all recurrent coefficients zero; a *deliberately constructed*, locally reactive subnet implements four independent monotonic directional responses to adjacent food and odor, plus an energy-sensitive birth logit. All behavior occurs through the unmodified numerical neural evaluation in `m5_0.act`; no runtime scripted action choice. Predeclare weights: for direction i in 0..3, hidden_i = tanh(2 × (4×adjacent_food_i + odor_i − .5×blocked_i)); output_motor_i = hidden_i. Hidden_4 = tanh(9 × (energy_norm − 48/54)); birth logit = 2.5×hidden_4 + 0.85. Other hidden units/weights and recurrent weights are exactly zero. The physical legal-action mask from original M5.0 still applies. This is essentially human-written foraging **encoded as weights**, and MUST NEVER be presented as emerged/learned.
2. **imitation_12:** external supervised imitation of M5.1 local `greedy_odor` teacher with 12-hidden / 401-weight numerical network.
3. **imitation_2:** identical external supervised protocol, 2-hidden / 51-weight network.
4. **random_12:** one fixed independent Gaussian random genotype N(0,.45) for the original 12-hidden architecture, generated from frozen independent seed 90112, shared across all test worlds.
5. **random_2:** same with hidden=2, independent seed 90102.
6. **scrambled_12:** fixed deterministic permutation (RNG seed 90212) of all 401 values of `engineered_12` with exact same numerical multiset; breaks semantic input/output mapping while preserving weight distribution.

For `imitation_12` and `imitation_2` the entire neural policy remains numerical and independently executed. Labels come from a **scripted teacher during supervised development**, never as runtime commands in ecological evaluation. Neither external supervised imitation nor engineered witness counts as an evolutionary result.

## Frozen external supervised learning budget and splits

Generate scripted M5.1 `greedy_odor` teacher experiences by running its unchanged physics/controller in six **development** worlds seeds 79000–79005, 200 ticks, 15×15 main physics. Before each teacher action save 15 local observations, legal original motor/reproduction mask, and chosen teacher action. Sample at most 4500 examples uniformly with fixed RNG 90512. Independent **validation** worlds seeds 79100–79101, 200 ticks, sample at most 1500 with seed 90513. No world IDs, positions, hidden future actions, global food locations, age or teacher map given to neural student; physically legal action mask is same as deployment and can make age eligibility available as a **law** rather than a sensor.

Initial weights for training: independent Gaussian N(0,.075) on input→hidden and hidden→output weights and biases, with recurrent weights initialized and FIXED to zero; PRNG seeds 92012 (12 hidden) and 92002 (2 hidden). Train by online cross-entropy SGD on **only legal logits**, batch size 1, **12 full epochs**, learning rate **0.03**, shuffled deterministic example order with RNG seed 93000+hidden. No distillation of an engineered genotype or use of final worlds. Evaluate all 12 epochs on validation teacher action accuracy, select the best accuracy, earliest epoch on ties. No ecological survival selection during development or validation. Report training/validation label agreement honestly, including teacher's age dependence which sensors cannot encode directly.

## Sealed ecological comparison worlds

**Only engineering smoke:** seeds 700–701 (never use for scientific results).

- **MAIN** 24 fresh worlds seeds **80000–80023**, 400 ticks, all six fixed arms.
- **SHIFTED** 12 new harder worlds seeds **81000–81011**, 400 ticks, all six fixed arms, original M5.0 shifted obstacle/odor/regen parameters.
- Every fixed genotype is identical across all worlds of its arm; world seeds control only map, food event schedule, founder physical placement and random motor/legal birth tie-breaks.
- Record per-world survival to tick 400, final population, births, generations, food intake, motor corrections, resource stock, checkpoints every 50 ticks, initial/final genotype hashes and full ledgers. Independent unit is **world seed**; bootstrap paired over worlds (not agents/actions).

### Preregistered outcome gates

1. **Representational feasibility of the 401-weight architecture (PRIMARY):** engineered_12 has ≥1 living organism at tick 400 on ≥12/24 main worlds AND attains generation≥3 in ≥12/24 main worlds AND exceeds random_12 on paired world-survival by positive mean, lower bound of descriptive paired bootstrap 95% >0, and positive-vs-negative world majority.
2. **Externally learned numerical policy (SECONDARY):** report same three gates for imitation_12 versus random_12; also report imitation_2 versus random_2. Never replace a failing primary engineered-witness gate with a more attractive secondary.
3. **Causal numerical genotype control (NECESSARY):** engineered_12 must beat scrambled_12 on main paired survival with positive bootstrap lower bound and positive-vs-negative sign majority, otherwise numerical genotype specificity is not supported on that endpoint.
4. **Shifted robustness (secondary):** report survival out of 12 for all arms; evaluate ≥6/12 worlds survived for engineered_12, and separately imitation_12. No retuning after shifted evaluation.
5. Independent action-head test must demonstrate changing numerical output bias can change a feasible motor decision into a legal birth decision, and directed food/energy test must verify engineered logit ordering. Exact original M5.0 action/physics identity remains intact when numerical genome is transplanted.

**If engineered_12 FAILS:** do not announce this architecture is impossible; report that **this particular constructed witness** is inadequate under the full ecology, and propose fresh non-leaking representational test. If engineered_12 PASSES but imitation does not, we established *an expressible phenotype that the specified external SGD failed to learn*. If both pass, the architecture has viable representational capacity, but whether unguided in-world heredity can discover it remains unresolved.

## Audit and safety

One file new study only, no changes to historical M5.0–M5.2B laws/conclusions. Energy ledger checkpoints zero, deterministic replay, no illegal walls/overlaps, faithful clonal inheritance/parentage. Full read-only GitHub Actions with archived raw CSV/JSON/SVG; permanently archive small per-world table and nulls. No tool, network, source code editing or executable self-replication by organisms. Never describe supplied numerical weights as *emergent natural selection*.
