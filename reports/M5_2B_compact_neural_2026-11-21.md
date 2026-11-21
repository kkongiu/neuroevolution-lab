# M5.2B — Compact neural genotypes still go extinct; no demonstrated heritable gain

**Completed 21 November 2026.** This is a **negative result** under a protocol [preregistered before any implementation](../docs/M5_2B_PROTOCOL.md), original protocol commit fb32799bcb3d36a9a49127166013db605b13951c.

Evidence: [successful sealed Actions workflow #44921004288](https://github.com/kkongiu/neuroevolution-lab/actions/runs/44921004288), 138 regression tests pass; [code](../neuroevolution/m5_2b.py); [tests](../tests/test_m5_2b.py); [36-world permanent CSV](M5_2B_world_survival_births_2026-11-21.csv). Actions artifact 'm5-2b-neural-heredity', ID **14713532707**, stores 'summary.json', 144 full 'world_metrics.csv' rows, time series, an SVG chart and newborn birth/genotype events, retained for 30 days.

## 1. Question and engineering

The [completed M5.2A study](M5_2A_causal_factorial_2026-11-21.md) had shown the motor/food-search bottleneck causally: replacing only movement with a hand-programmed odor controller rescued 20/20 worlds while leaving neural reproductive decisions unchanged. In M5.2B **there is no scripted movement and no scripted reproduction**: ALL five actions are selected by each agent's own neural genotype using the unchanged original M5.0 action handler.

The original 12-neuron recurrent network has 401 mutable weights. We instead freeze a recurrent tanh network with **2 hidden neurons and exactly 51 mutable numerical parameters**, still reading **all 15 original sensors** and outputting 4 movement directions plus REPRODUCE. The possibility that this network is too small to represent reliable navigation remains an important interpretation limit.

Four matched genetic-birth treatments with otherwise IDENTICAL ecology:
- **sparse_inherited:** exactly 3 randomly selected of 51 weights mutated by Gaussian SD 0.12 at each valid in-world birth;
- **dense_inherited:** all 51 child weights mutated with the same SD;
- **clone_inherited:** exact unchanged parental genome;
- **random_offspring:** child genome independently redrawn Gaussian SD 0.45.

All founder genomes and positions, world obstacles, resource schedules, energy laws, parental birth costs, age limits, neural action legality and scheduling are matched for the same seed. The sparse adapter replaces only the new child's genotype **after a physical cloned birth and before the child acts**; exact parentage, numerical digest, cosine and resource ledger are audited. The other three arms reproduce original M5.0 behavior for their genetic operators with hidden=2. No weights are trained within an organism's lifetime; no external researcher chooses an elite or selects optimal neural actions.

**Preregistered new worlds:** MAIN 77000–77023 (24), SHIFTED 78000–78011 (12), 400 ticks each, four arms. The old M5.0/M5.1/M5.2A results and seeds were not modified or reused.

## 2. Preregistered primary endpoint: actual ecological survival at tick 400

| Genetic mechanism | Main survival /24 | Shifted survival /12 | Mean births MAIN | Mean food collected MAIN | Mean max generation MAIN |
|---|---:|---:|---:|---:|---:|
| **Sparse mutation of three genes** | **0** | **0** | 11.92 | 32.83 | 1.58 |
| Dense mutation of all 51 genes | **0** | **0** | 12.42 | 33.42 | 1.71 |
| Exact clonal inheritance | **0** | **0** | 11.83 | 32.88 | 1.54 |
| Independent random offspring | **0** | **0** | 13.21 | 38.75 | 1.88 |

All 144 world-arm trials completed without simulator or CI errors. There is **no** survival-at-400 advantage for sparse inheritance, sparse vs dense, sparse vs clone or dense vs independent offspring. For all main survival contrasts: **mean difference 0, bootstrap descriptive 95% [0, 0], 0 positive, 0 negative and 24 ties**. In shifted worlds all arms were likewise extinct, with 12/12 tied survival outcomes for each comparison.

**All gates FAIL:** sparse survival >=12/24, sparse generation 3 in >=12/24, positive paired inherited-versus-random persistence, added novelty-vs-clone gate, shifted survival >=6/12 and inherited-vs-random shifted transfer. Sparse inherited worlds reached generation >=3 in **0/24** main worlds. In the main test dense mutations reach generation 3 in 2/24, exact clones in 1/24, independent random offspring in 3/24, but **no lineage persists**.

At the end of 400 ticks **all arms average 32/32 food units still available**. A lack of food in the world is not the most direct observed cause. No positive mutated advantage is visible in this primary ecological endpoint. Random offspring had slightly more births/food collected on average than sparse, but these exploratory count differences do not establish a robust advantage or general mechanism.

## 3. Shifted generalization

Harder worlds (wall fraction 0.20, odor radius 2, regeneration probability 0.40): **0 survivors in each of 12 worlds for every genetic arm**.

Mean births and food collected respectively:
- sparse 12.50 / 29.08;
- dense 12.42 / 30.33;
- clone 12.08 / 28.50;
- random offspring 12.08 / 31.67.

The shifted data do not rescue any hypothesis, and test seeds are separate from the main seeds.

## 4. What this does — and does NOT — establish

**Established:** shortening the neural genotype from 401 to 51 weights and restricting neonatal mutations to three parameters does **not** by itself produce sustained ecological survival or a measurable inherited advantage under the frozen M5.0 physical conditions. There were real in-world births and neuron-controlled actions; no winning offspring lineage emerged. Original full-suite regression (138 tests) passed, and energy and birth invariants remained intact.

**Not established:** this does not prove an evolutionary neural forager cannot emerge. It also does NOT isolate whether the limiting factor is (i) the **capacity** of a two-neuron RNN to represent good navigation, (ii) the accessibility of such genotypes by sparse mutation from random founders, (iii) ecological exposure time/selection intensity, or (iv) the fitness landscape induced by local sensors/argmax actions. We have shown a joint negative outcome for this architecture and selection model, not a theorem about all neuroevolution.

An additional genotype transplant/causal-phenotype test was specified only if there was evidence of stable evolution, which was not achieved; no successful genotype was available for that confirmation. Do not select individual attractive birth-count outliers post-hoc to claim adaptation.

## 5. Decision

**Preserve M5.2B as a negative scientific result.** Do not repeat the same 51-weight experiment with a few tuned Gaussian SD values and call that progress.

If research continues, design a **new preregistered M5.3 with two separate questions**:

1. **Representational positive control:** Can a numeric neural controller with these sensors and output channels express food-directed behavior and reliable birth decisions at all? Use externally engineered or separately trained controller weights only as a *diagnostic capability upper bound*, not as proof they evolved within the ecology. Report inability or difficulty to represent useful behavior openly.
2. **Evolutionary accessibility:** On fresh seeds and without injecting those reference weights as evolved agents, test whether genotype-native mutations and actual birth/death selection discover comparable behavior. Prefer an independently specified compact *equivariant* neural architecture that exploits spatial symmetry through weight sharing without setting positive odor-following weights manually, plus direct controls for clonal standing variation, independent random newborns and altered-resource worlds. Predefine selection budgets and genotype transfer to truly new worlds.

M5.2A established **how the animals could survive** when movement was supplied. M5.2B has **not yet established a way that survival competency evolves** from random neural genotypes. That is the next frontier, and it should remain falsifiable.
