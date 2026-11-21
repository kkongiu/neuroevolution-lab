# M5.1 — Ecological viability and foraging diagnostics, without retuning M5.0

**Frozen preregistration:** commit this document before implementing M5.1, running a new scientific campaign or inspecting any M5.1 final-world scores. M5.0 physics and its negative published results remain unchanged. New test seeds are disjoint from all M5.0 seeds.

## Question

M5.0 generated real in-world newborn organisms and inherited genotypes but **every world became extinct**, although food supply often saturated its 32-item cap. Is the same ecological physics capable of supporting a population if an organism uses a deliberately **hand-programmed local food-search policy**? Does an elementary local odor-following policy forage more effectively than legal random motion, and where do the neural organisms lose energy?

**M5.1 is a feasibility/diagnosis study, not a new evolution optimization trial.** It must not inject hand-coded competence into the evolved neural genome, train on test worlds, select the best post-hoc control, or call scripted performance emergent intelligence.

## World held fixed from M5.0

Use the original M5.0 code as an unchanged imported **physical engine**: 15×15 maze, 20 founders with matched initial random RNN genomes, independent local bodies/ages/IDs, finite lifetime 100 ticks, 1 energy per tick, food energy +18 capped at 54, initial energy 36, 20 initial foods, 32-food capacity, two food renewal candidates per tick each accepted with probability 0.55, wall density 0.12, local odor radius 5, reproduction allowed at age ≥4 and energy ≥34 and population <64, parent pays 20 plus basal cost and child receives 12. No resource subsidy, corpse calories, hidden global map, predator, unscheduled renewal or extra reproduction opportunity. Fixed 400-tick horizons.

Shifted worlds (separate, no retuning): wall density 0.20, local odor radius 2, renewal probability 0.40, as M5.0.

The same world seed determines identical initial founders, walls and a pre-sampled stream of food-placement candidate events **for every arm**. Different agents may eat/spawn at different places so state trajectories legitimately diverge. Spawn failures remain failures, not relocated bonus food.

## Four independent active-policy arms

1. **neural_heritable**: exactly M5.0's original `act` on newly seeded environments, heritable mutation SD .12 at births. The neural network independently chooses REPRODUCE when legal. This is a *fresh replication*, not retuned M5.0.
2. **greedy_odor**: hand-coded reactive baseline uses ONLY local blocked neighbor flags, adjacent food, directional odor; movement chooses unblocked adjacent food first, else the maximum odor-valued legal direction, with deterministic RNG tie breaking; when every odor is zero, uniformly random legal direction. It chooses REPRODUCE when legal, age≥4, **energy≥46** and an adjacent free cell exists. This energy threshold is scripted prior to evaluation, and is MORE DIRECT INFORMATION than the neural policy's original input (the script reads age). The greedy controller is not evolved, carries no useful genotype and may have strong advantages unrelated to selection.
3. **greedy_novelty**: identical reproductive rule and local odor channels; movement chooses among free directions using score `4×adjacent_food + odor + 0.35/(1+prior_visits_to_destination)` and RNG-seeded tie break. Each agent remembers only how many times **itself** visited each cell, not the global map. This is deliberately a **stronger programmed feasibility reference**, not a neural discovery.
4. **random_legal**: uniformly random among unblocked cardinal moves, **the same scripted reproduction threshold** (energy≥46, age≥4), no food sensor use. Distinguishes environmental economics/reproduction from foraging logic.

All programmed agents' genotype bookkeeping can follow the same birth genetics as M5.0 but **weights are not read to choose scripted actions**. This makes the population and reproduction physical, not the behavioral policy heritable in scripted arms. Never call generations of scripted organisms a neural evolutionary success.

**Additional negative controls on the first four main world seeds only:**
- **greedy_no_reproduction**: greedy_odor motion but reproductive action disabled, same food regeneration. By M5.0 max age 100 and no newborns, all founders MUST be absent by tick 100 (physics check). This is not an adaptive comparison.
- **greedy_no_regrowth**: greedy_odor and reproduction unchanged, same 20 initial foods but subsequent food spawn probability forced to zero, to test dependence on **renewable** resources. Report extinction even if no literal zero food at every tick; no assumption of automatic failure unless established by physics.
- Existing M5.0 four no-food worlds are a historical physics control and are not reused as final scientific data. No direct modifications of M5.0 simulations.

## Fixed seeds and metrics

- **Main 16 independently initialized worlds:** seeds **73000–73015**; 400 ticks each, all four active arms.
- **Shifted 8 worlds:** **74000–74007**, harder physical environment, all four active arms; never used for parameter search.
- Extra no-reproduction/no-regrowth arms: 73000–73003 only (4 each), outside primary comparison.
- **Smoke seeds 700–701** at 40 ticks before main are non-scientific engineering tests.
- All arms receive the same initial genomes/spawn schedule. World seeds are independent experimental units; individuals, actions, births and time steps within a world are NOT independent replicates.

**Primary physical feasibility gate:** the predeclared strongest scripted `greedy_novelty` arm should have **≥1 living organism at tick 400 in ≥8/16 main worlds**, AND should have reached at least **generation depth 3 in ≥8/16 worlds**. Success demonstrates *this* environment can sustain a simple programmed organism, not that neuroevolution works.

**Necessary for the foraging-skill hypothesis:** `greedy_odor` must beat `random_legal` on **food consumed per founder-world over 400 ticks** with positive mean paired difference, paired descriptive percentile-bootstrap 95% lower bound >0, and ≥12/16 positive paired worlds. Also report novelty vs greedy and all active arms vs neural, but do not substitute any secondary for the primary gate.

**Secondary at end:** population alive and extinct fraction at tick 100/200/300/400, births, food consumed, generation depth, last extinction tick, cumulative energy from food, available food stock, genotype-independent scripted policy effects. No scripted survival should be interpreted as adaptive inherited gain.

**Behavior diagnostics:** 
- Fraction of eligible reproduction opportunities and actual births, illegal reproduction wishes for neural controllers, blocked/illegal preferred actions and motor-mask corrections, choices aligned with best local food odor when detectable.
- Number of movements, food eaten per 100 movement actions, adjacent-food opportunities and subsequent successful ingestion, mean energy at action, number of organism visits to a cell, fraction of action times with no odor signal and remaining food in a saturated world.
- Per-world population checkpoints at fixed times 0/50/100/150/200/250/300/350/400; early extinction tick and food stock are primary explanatory curves. All comparisons use **world-paired** data; descriptive causes are hypotheses unless experimentally manipulated.
- Exact closed **energy ledger** every checkpoint, births/offspring parentage and distinct genealogy, no teleportation through walls, no two living bodies occupying the same cell, no birth without parent cost, no food consumed twice.

**Interpretation and follow-up:** If BOTH scripted policies die with abundant resources, investigate reachability/odor dynamics and ecology parameters in a separately preregistered M5.2; do not "repair" the results by trying alternative thresholds on the same final seeds. If programmed agents survive but the unmodified neural ones die, this is evidence of a **behavioral competence gap**, not spontaneous evolutionary innovation. A later M5.2 may preregister sensor/mutation/curriculum changes and test inherited gain with unseen seeds.

No simulated organism may call tools, edit executable code, access GitHub, files or network, change the laws of reproduction or choose its own objectives.
