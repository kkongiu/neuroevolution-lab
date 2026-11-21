# M5.3A — Numerically encoded neural foraging is viable, but fragile under ecological shift

**Completed 21 November 2026.** This is a positive **representational-capacity** result, **NOT** a result of autonomous in-world neuroevolution.

- [Frozen preregistration](../docs/M5_3A_PROTOCOL.md), committed **before implementation or final world evaluation** as `05eb3d16b529cb45c463abd6cc36184af0585b7f`
- [Original successful read-only Actions run #44922015463](https://github.com/kkongiu/neuroevolution-lab/actions/runs/44922015463), **147 regression tests passed**, complete sealed campaign
- [Implementazione](../neuroevolution/m5_3a.py), [test](../tests/test_m5_3a.py), [PR #35](https://github.com/kkongiu/neuroevolution-lab/pull/35)
- [Permanent per-world CSV: 36 worlds × six genotypes](M5_3A_paired_worlds_2026-11-21.csv)
- Original Actions artifact **`m5-3a-numeric-neural-capacity`, ID 14713169853**: `summary.json`, `world_metrics.csv` (216 world-arm outcomes), `population_checkpoints.csv`, `genotypes.json` (exact engineered, externally trained, scrambled and random numeric genomes), six representative genetic traces for each split, and `population_curves.svg`. Archive retention 30 days; protocol, implementation, mean results and selected world CSV retained in repository.

## 1. Question

M5.2A established a **behavioral motor bottleneck**: scripted local scent-following movement together with neural reproductive decisions rescued all ordinary worlds, whereas fully random neural policies went extinct. M5.2B found no autonomous neural ecological survival using 51-weight recurrent genomes and sparse/dense/clone/random inheritance treatments. Does this failure arise because a neural genotype *cannot represent* effective movement, or because random mutations and selection cannot discover a useful genotype quickly enough?

To test representation separately, M5.3A **suspends mutation**: a **fixed genotype** is transplanted into every one of the original 20 founders and **cloned exactly** upon real in-world reproduction. The unmodified M5.0 `act` still calculates all five numerical action logits and performs only physically possible movement or reproduction. **No teacher script supplies runtime decisions and no weight evolves inside the ecological world.**

Three kinds of numeric genotype were evaluated:
1. **Researcher-engineered positive control** (`engineered_12`), a 12-hidden/401-weight fixed RNN with zero recurrent weights, four human-designed directional sensor responses (adjacent food, odor, blocked flags) and a human-designed energy-sensitive reproductive output;
2. **Externally supervised imitation** (`imitation_12` and `imitation_2`), 401- and 51-parameter networks with all action outputs numerical, trained on scripted-teacher development experiences. Recurrent matrices fixed to zero, with weights selected after 12 epochs according to independent teacher validation classification, **not ecological final survival**;
3. Fixed random 401/51-weight controls and **`scrambled_12`** (engineered witness's exact multiset of numeric coefficients, randomly permuted among neural positions).

**Critical:** the engineered controller has hand-written competence **encoded in weights**; externally trained students receive supervised target actions. Neither is an organism that discovered food navigation through selection/mutation. Clonal inheritance transmits competence but does not generate novelty.

## 2. Frozen training and test partitions

**Teacher development:** six worlds 79000–79005, 200 ticks, at most 4500 sampled local 15-channel observation/legal-action/teacher-action pairs. **Teacher validation:** two separate worlds 79100–79101, 200 ticks, at most 1500 pairs. Training: deterministic online masked cross-entropy SGD, 12 epochs, learning rate 0.03; recurrent weights locked at zero, original action outputs all trainable, best epoch selected on **teacher validation action accuracy**.

Training results:
- 401-weight imitation: best validation action agreement **60.93%**, epoch **11**.
- 51-weight imitation: **59.40%**, epoch **11**.

The teacher sometimes uses private physical age to decide reproduction; the neural observation does not explicitly include age, although the unchanged physical birth-eligibility mask does. Good survival with moderate teacher imitation agreement shows exact reproduction of the teacher's individual action choices is **not necessary** for persistent population viability; this is not proof of generalization.

**Final study, no reused worlds:** 24 MAIN world seeds **80000–80023**, and 12 shifted worlds **81000–81011**, 400 ticks each. All six numerical genotypes are evaluated against each new world with identical physical food candidate schedules and 20 clonal founders. The original world conditions and all birth-energy costs remain unchanged from M5.0. In shifted worlds, obstacle density rises to 0.20, odor radius drops to 2, and regeneration probability to 0.40.

## 3. MAIN — preregistered survival and genealogical outcomes

| Fixed numerical genotype | Surviving worlds /24 | Average alive at tick 400 | Mean births/world | Mean food collected/world | Mean maximum generation |
|---|---:|---:|---:|---:|---:|
| **engineered_12** (human-designed numerical witness) | **24** | 14.33 | **125.29** | 402.04 | **12.08** |
| **imitation_12** (externally supervised) | **24** | 12.33 | **100.96** | 403.67 | **10.92** |
| **imitation_2** (externally supervised) | **17** | 3.58 | 50.08 | 342.21 | 6.46 |
| random_12 (one fixed random genotype) | **0** | 0 | 49.25 | 114.54 | 5.08 |
| random_2 (one fixed random genotype) | **0** | 0 | 0.25 | 25.38 | 0.25 |
| scrambled_12 (engineered weights permuted) | **0** | 0 | 8.71 | 20.63 | 1.04 |

**Predeclared primary representational gate PASS:** engineered_12 survives **24/24**, reaches generation ≥3 in **24/24**, and outperforms random_12 survival **24/24 paired worlds**, mean **+100 percentage points**, world-level descriptive paired bootstrap 95% **[+100,+100] pp**.

**Causal genotype-scramble gate PASS:** engineered_12 versus scrambled_12 is also **24/24 favorable paired worlds**, difference **+100 pp**, bootstrap descriptive **[+100,+100] pp**. Permuting weights while preserving their value distribution destroys this genotype's viability under the ordinary ecology.

**Externally learned 401-weight gate PASS:** imitation_12 survives **24/24**, reaches generation≥3 in **24/24**, and outperforms random_12 in **24/24** paired worlds by +100 pp (bootstrap descriptive [100,100] pp). Therefore a *numerically learned* motor/reproduction policy can sustain the original world, even when a separately initialized, heritably mutated random population from M5.0 could not.

**Exploratory compact 51-weight gate PASS:** imitation_2 survives **17/24**, reaches gen≥3 in **22/24**, and outperforms random_2 **17/24** main worlds; paired survival difference **+70.83 pp**, descriptive bootstrap 95% **[+54.17,+87.50] pp**. This is direct representational evidence for the 51-weight network: its M5.2B extinction was **not caused solely by an absolute inability of the small architecture to express a viable policy in ordinary worlds**.

**Statistical caveat:** only one fixed random genotype was preregistered for each architecture and one fixed scrambled genotype. The world-level bootstraps represent variability **across worlds for those chosen fixed genotypes**, not the entire distribution over independently random genomic initializations. Likewise, these positive controls compare fixed clones against fixed clones, not evolving population fitness.

## 4. SHIFTED — striking failure of robustness

**Every engineered and externally supervised genotype goes extinct in all 12 more hostile worlds**:

| Fixed numerical genotype | Surviving worlds /12 | Mean births/world | Mean food collected/world |
|---|---:|---:|---:|
| engineered_12 | **0** | 17.92 | 40.75 |
| imitation_12 | **0** | 18.17 | 78.08 |
| imitation_2 | **0** | 19.17 | 62.83 |
| random_12 | **1** | 38.33 | 86.00 |
| random_2 | **0** | 0.08 | 24.42 |
| scrambled_12 | **0** | 7.08 | 28.58 |

**Predeclared shifted-transfer gates FAIL** for engineered_12 and imitation_12: each 0/12 rather than ≥6/12. The one random_12 survival outlier is a single world result for one fixed random genotype, not evidence of better universal random-controller generalization.

These shifted worlds simultaneously alter walls, olfactory range, and food regeneration; no individual change is causally isolated by this experiment. The human-designed witness cannot be assumed to have represented obstacle avoidance or distant food access well enough. We must not silently retune its weights on these formerly sealed shifted seeds.

For context, M5.1's local programmed greedy+visit control did persist on all eight of its own shifted worlds, but **different seeds, a richer programmed policy and different test samples** prevent treating that cross-campaign difference as a controlled neural-vs-script comparison.

## 5. Scientific interpretation and decision

- **YES:** at least one fixed 401-weight numerical neural genotype represents sufficient ecological behavior to survive and reproduce across all 24 ordinary worlds. A fixed externally supervised 401-weight genotype independently replicates that representational result; a 51-weight numerical policy also survives most ordinary worlds.
- **NO:** these results do **not** show that mutation and in-world selection *discovered* those genotypes. Births are real simulated events but cloned genomes have no adaptive genetic variation by design. We deliberately supplied effective weights through engineering or supervised teaching.
- **NO:** neither effective fixed neural genotype generalizes to the independently sealed shifted world distribution. This is a real and important null result.
- **NOT RESOLVED:** the original neural populations' inability to evolve effective foraging is likely an **evolutionary accessibility/search problem** in ordinary worlds, rather than absolute lack of neural expressivity. But M5.3A alone does not prove what mechanism causes that accessibility barrier, nor guarantee the search could succeed with any particular alternative mutation scheme.

**Next independently preregistered milestone: M5.3B.** Test *only* true in-world genetic search with random (not teacher-trained, not engineered) founders on NEW worlds. A plausible scientific comparison is spatially symmetric/weight-shared neural inductive bias versus same-parameter budget generic RNN, with clonal and independent offspring controls and matched resources, birth/death and organism-tick budgets. Test both ecological survival and inherited-vs-random advantage; require genotype-causal transfer to new worlds before claiming adaptive phenotype discovery. Separate new shifted environments should assess generalization after survival, without retroactive tuning M5.3A.

No organism in this work was allowed tool/network/code access, source modification or executable self-replication.
