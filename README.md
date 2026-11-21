# NeuroEvolution Lab

Scientific, CPU-only experiments in adaptive neural systems. **No claim of autonomous will or consciousness.** Python 3.11+, no third-party dependencies.

## M1 — Structural neuroevolution

Two populations start with identical recurrent networks. One can modify numeric weights only; the other can also modify topology between generations. Their held-out performance is compared under identical evaluation budgets.

Run:
~~~bash
python -m unittest discover -s tests -v
python -m neuroevolution --generations 12 --population 12 --out results/m1
~~~

M1 writes a resumable `checkpoint.json`, `history.jsonl`, `summary.json` and `scores.svg`. The [first M1 campaign](reports/M1_campaign_2026-11-20.md) was exploratory, with just three paired seeds.

## M1-P — Continual learning and plasticity

Unlike M1, this experiment **updates the neural weights during each task** using SGD. Four arms solve exactly the same task sequences: fully continual weights; reinitialize from scratch; low-activity unit renewal; random unit renewal. It measures within-task adaptation curves and forgetting across successive tasks.

~~~bash
python -m neuroevolution.plasticity --seeds 8 --tasks 8 --steps 128 --out results/plasticity
~~~

This writes `measurements.csv`, `summary.json`, and `curves.svg`. Run with a new output directory for each campaign. No GitHub write operations or arbitrary code modifications are performed by the agents.

Read [the preregistered plasticity protocol](docs/PLASTICITY_PROTOCOL.md) before interpreting the results.

## M2 — Bounded meta-learning of renewal decisions

A five-weight meta-controller is **evolved** on one set of task sequences, selected on a separate validation set, then tested on completely disjoint seeds. It can decide **0 / half / maximum (20%)** neural unit replacements at task boundaries using training data only. No code generation or arbitrary self-modification is possible.

Its independent test compares six approaches: learned decisions, no reset, fixed 20% low-activity renewal, **random units with the exact same reset counts per task**, a **shuffled schedule with the same total number** of low-activity resets, and a freshly initialized network.

Run the whole predefined M2 protocol:

~~~bash
python -m neuroevolution.meta --out results/m2
~~~

Outputs: meta-training history, per-seed metrics, per-task metrics, self-contained SVG, and paired bootstrap confidence intervals. Read the [M2 frozen protocol](docs/M2_PROTOCOL.md). **A winning average against no-reset alone is not enough**; dose-matched and timing-matched controls are primary.

## M2B — Learn *when* to renew at a fixed total budget

M2 selected **zero resets in 112/112 transitions** ([negative result](reports/M2_adaptive_2026-11-21.md)). That made its dose-matched timing comparisons vacuous. M2B tests a different prespecified hypothesis without changing M2's code or data.

Every dose-matched policy now **must spend exactly 10 unit renewals over seven task transitions**, with a maximum of three renewals at any one transition. Two controllers are independently evolved:

- **Adaptive:** sees current and past task *training* error, hidden-unit activity and the task index.
- **Time-only:** sees only task index.

We test both against uniform, shuffled and random-unit choices using independent meta-train, meta-validation and final-test seeds. The report distinguishes **requested vs forced** actions, because mandatory late-budget consumption is not evidence of learned adaptability.

~~~bash
python -m neuroevolution.meta_budget --out results/m2b
~~~

Read [M2B protocol](docs/M2B_PROTOCOL.md) before interpreting results. Equal reset counts do not imply equal CPU/energy overhead.

## M3 — Unpredictable novelty, returns, and causal state ablation

The world now introduces tasks in an **unpredictable order**, sometimes revisiting an **exact old latent task** with newly sampled examples. The task identity and novelty labels are logged but *not disclosed* to the controller.

The core test pits an evolved state-aware renewal scheduler against an independently evolved **time-only** scheduler, at exactly the same intervention quota. Final tests use both fresh seeds from familiar task families and a **partially shifted** world with previously unseen checker/wedge task families. The same learned controller is evaluated with its state observations ablated to establish whether the signals change decisions and outcomes.

~~~bash
python -m neuroevolution.m3 --out results/m3
~~~

Outputs: an auditable frozen summary, meta-training trace, final per-seed and per-task CSVs, separate measures for novel and returning tasks. See the [M3 protocol](docs/M3_PROTOCOL.md).

## M3.1 — Causal value of structural renewal

A **preregistered, deliberately engineered positive-control environment** creates two paired contexts from exactly the same pretrained network: intact representations, and a controlled disruption of influential hidden units. Both interventions—preserve and renew—are run counterfactually from the same starting state on the same data, so the causal effect of renewal can be measured instead of guessed.

A small learned threshold uses *current labelled training-probe accuracy* to decide whether to replace selected hidden units. Independent comparison policies use only episode position, reset-all, preserve-all, and a **shuffled action sequence with the same reset count**. An oracle based on heldout counterfactual outcomes serves solely as a non-deployable upper bound. Main and shifted task-family tests use sealed independent seeds.

~~~bash
python -m neuroevolution.m3_1 --out results/m3_1
~~~

Read [the frozen M3.1 protocol](docs/M3_1_PROTOCOL.md). A success here would **not** show general meta-learning or self-generated goals: the damage operator and its possible repair are deliberately constructed and externally specified.

## M3.2 — Natural lifelong interference (no injected damage)

M3.1 proved that a pre-engineered artificial neural failure could be detected and repaired. **M3.2 removes that external corruption.** One persistent ReLU network learns 12 supervised tasks sequentially. Some tasks recur and others arrive for the first time, causing representational interference solely via normal SGD.

An externally evolved policy can choose *when* to renew three low-activation units, with **exactly four intervention events per lifetime**. Independent controllers see (i) all training-only diagnostics including output-gradient conflict, (ii) accuracy/time only, or (iii) time only. Exact-budget controls shuffle action times, spread them evenly, or select random neurons.

For the adaptive policy, at each final-test boundary we also clone the **actual persistent network** and evaluate both one-step interventions on identical training data; the holdout-based oracle is **never shown to the policy**. We report learning AUC, old-task forgetting, and the narrow one-step counterfactual effect separately. Partial OOD tests use checker/wedge problems absent from training.

~~~bash
python -m neuroevolution.m3_2 --out results/m3_2
~~~

Read the [M3.2 frozen protocol](docs/M3_2_PROTOCOL.md). A negative outcome will be published without retuning final seeds.

## M3.3 — Memory becomes logically possible with context

M3.2's hidden task rules could contradict each other on the same x,y input. **M3.3 introduces an explicit context cue**: four context labels, two pairs of exact opposing classification rules, and the same x,y coordinates in paired test panels. A six-input MLP sees x, y, and a one-hot context code. A blind control gets four zero context inputs with identical capacity, seeds, SGD steps and examples, and therefore has a mathematical 50% balanced-accuracy ceiling.

The experiment also tests a bounded **training-only episodic replay buffer** (replacing one in four SGD batches, not adding extra compute) and whether an externally evolved controller can schedule **three equal-dose neuron renewals** better than independently evolved accuracy-only or clock-only controllers. All methods face the same sequence, and their last-stage memory is measured on heldout panels from all four contexts.

~~~bash
python -m neuroevolution.m3_3 --out results/m3_3
~~~

Read [frozen M3.3 protocol](docs/M3_3_PROTOCOL.md) before results. A positive cue result validates *solvability given a supplied task ID*, not spontaneous context discovery; memory and renewal results are evaluated separately.

## M3.4 — Recognize hidden context from experience, without an oracle ID

In M3.3 the experimenter provided the correct one-hot context identifier. **M3.4 removes that privilege.** A simple **externally coded k-nearest-neighbour associative memory** allocates up to four anonymous context slots by comparing a small labelled support set with training-only samples from previous episodes. A persistent six-input ReLU network learns to use the inferred one-hot slot cue rather than the true context ID.

The scientific tests are explicitly separated: compare correct support with **shuffled final support on the identical trained network**, no-support and zero-context controls; compare online inferred cues with a nondeployable oracle-ID reference; and test fixed-buffer **FIFO vs reservoir replay** plus no replay under identical SGD budgets. Further sealed tests reduce support to six labelled examples with one corrupted label, with/without new stripe/checker task families.

~~~bash
python -m neuroevolution.m3_4 --out results/m3_4
~~~

See the frozen [M3.4 protocol](docs/M3_4_PROTOCOL.md). The resolver is an **algorithm we designed**, not spontaneous neural invention of a cognitive strategy. Labelled support examples are available: the system cannot infer contradictory context identities with no informative observations.

## M3.5 — Spend an equal information budget where feedback seems needed

M3.4's kNN resolver could infer returning latent contexts from labelled support, but performed poorly with **six noisy examples** and new rule families. M3.5 now tests **active information requests**: after six labelled support points with one corrupted label, choose **four of twelve episodes** in which to request **two fresh labelled points**, a total of **eight extra labels per sequence**.

A bounded, externally programmed **uncertainty-margin** scheduler competes against an independently meta-selected entropy heuristic, random request episodes, periodic requests (all at identical information cost), no requests, and requests at every episode (higher-cost diagnostic). No policy chooses which x,y points are labelled: the two extra IID points arrive from an independent sampling stream **only if** an intervention was requested.

All policies share M3.4's fixed 0.84 context-admission threshold, kNN k=7 prototypes, one persistent ReLU network, FIFO replay memory, equal SGD steps, same task streams, and independent meta-train/validation/sealed test seeds. Final outcomes are evaluated separately for familiar line/XOR tasks, unseen stripe/checker families, and an added error in the supplementary feedback.

~~~bash
python -m neuroevolution.m3_5 --out results/m3_5
~~~

See the [preregistered M3.5 protocol](docs/M3_5_PROTOCOL.md). Successful request timing would indicate only the value of this **programmed information-budget policy**, not a spontaneously invented active-learning method or autonomous objective.

## M4.0 — An embodied, evolving organism in a survival grid

**A new research track.** Earlier M1–M3.5 experiments trained classifiers and controlled small neural-reset or memory actions. M4.0 introduces a genuine **2D spatial environment**: mazes with walls, finite food, energy expenditure, collisions and starvation. An organism starts with 24 energy units, consumes one unit per move and can recover energy by collecting food. Food is nonrenewable in this first experiment.

The organism controls its four cardinal movements using a fixed-topology **12-hidden-unit recurrent neural network**. Its inputs are strictly local: immediate walls, nearby food, a limited directional odor gradient, current energy and recent consequences. The odor gradient is deliberately programmed by the experimenter and must not be mistaken for an emergent navigational sense.

An external genetic algorithm evolves only numerical network weights **between separate lifetimes**, selecting a population over eight generations on training mazes; a champion is selected on validation worlds. **No synaptic learning occurs within an organism's lifetime**, and no simulated organism can edit code or generate goals.

On **32 sealed unseen mazes**, compare one frozen evolved champion with a random unevolved network, wall-avoiding random movement, a programmed greedy odor follower, and the same evolved genome with odor disabled. Evaluate another **16 shifted mazes** (more walls, less food, shorter scent range). Report food collected, survival time, wall collisions, visited cells, paired uncertainty and genetic-search history without cherry-picking.

~~~bash
python -m neuroevolution.m4_0 --out results/m4_0
~~~

Results also include CSV, a reproducible saved champion, example world SVG maps and JSON movement traces. Read [the M4.0 frozen protocol](docs/M4_0_PROTOCOL.md) and the [M4.0 Actions campaign](.github/workflows/m4-0-survival-world.yml). **Predators, within-world reproduction and within-lifetime weight learning are NOT yet present.**

## M4.1 — Correct illegal moves, then test evolved food-search skill

M4.0's evolved neural agent hit walls ~18 times per lifetime and lost against a wall-avoiding random walker. M4.1 **isolates physical motor legality from neural planning**. A new decoder takes the recurrent network's four action logits and chooses the highest-scoring **locally legal** direction. The network still receives only M4.0's local wall, food odor and energy sensors. Every blocked preferred action is recorded separately from actual collisions.

For causal comparisons, M4.1 **evolves two networks from exactly the same initial genomes** for each of **eight independent genetic initializations**: one training with the motor mask, the other without it. Each genome is later tested **with and without** the decoder on the same mazes. Evolved masked genomes are also compared against independently initialized masked random networks, the wall-avoiding random walk, an intentionally strong greedy odor follower, and a same-genome odor ablation.

Each training generation uses newly seeded maze maps, and each map is rotated with a fresh reachable starting position. The genomes are selected only on independent validation worlds and then evaluated unchanged on **24 unseen main mazes** and **12 shifted mazes** (more walls, fewer food items, shorter odor range). Use the full two-level genetic-run/world-seed bootstrap and do not treat repeated worlds as independent evolutionary replicates.

~~~bash
python -m neuroevolution.m4_1 --out results/m4_1
~~~

Read the [M4.1 frozen protocol](docs/M4_1_PROTOCOL.md) and [campaign workflow](.github/workflows/m4-1-masked-navigation.yml). This is still **externally driven numeric evolution between lifetimes**, not in-lifetime learning, emergent reproduction, or autonomous choice of objectives.

## M4.2 — Can a recurrent organism *remember* where to turn?

M4.1 showed that an evolved RNN did not beat a wall-avoiding random walker at finding food, even after motor masking. **M4.2 isolates one narrower ability**: remembering a fleeting direction cue through several turns of motor control until the time comes to choose between **two symmetric irreversible fork branches**.

Every heldout world seed provides **two paired worlds with identical walls, start and fork**, but opposite target branches. The environment shows a cardinal direction cue during only the first **two actions**; after this, cue and food odor are zero. The 19-channel neural input also has its energy channel held constant (the *actual* physical energy still decreases), preventing a feedforward network from storing cue information in elapsed time or energy. A one-way branch gate gives food for the correct choice and terminates either way. This is a deliberately simplified **causal memory benchmark**, not an open-world ecology.

Two **separately evolved** populations per genetic start are tested: a recurrent **8-tanh** network (260 numeric weights) and a stateless **11-tanh** feedforward network (268 weights, 3.1% larger), with equal 6-generation × 12-candidate training budgets. The study uses **8 independent genetic starts**, **32 unseen paired main worlds**, **16 longer-delay shifted pairs**, and a diagnostic untrained transfer to **12 open reactive food mazes**.

Controls on the **same trained recurrent genome** reset its recurrent state immediately before the fork, flip the early cue, or remove the early cue entirely. Also compare with a uniformly random legal mover and a simple **scripted finite-state memory policy**; report all controls even if the evolved network fails. Primary is paired RNN minus feedforward food success with hierarchical bootstrap over independent genetic runs and shared paired worlds, plus necessary causal reset and cue-flip effects.

~~~bash
python -m neuroevolution.m4_2 --out results/m4_2
~~~

Read [M4.2 preregistration](docs/M4_2_PROTOCOL.md) and the [sealed GitHub campaign workflow](.github/workflows/m4-2-delayed-cue.yml). **There is no within-lifetime synaptic learning, predator, biological reproduction, autonomous objective invention or source-code editing by simulated organisms.**

## M4.3 — Isolate actual recall from motor control

M4.2's RNN beat a feedforward net partly because it reached the fork more often, while deleting its memory at the fork did not conclusively damage food success. **M4.3 removes that confound entirely.** A simulated circuit sees a two-tick symbolic direction cue, then **2–6 identically neutral observations** (or 8–10 at shifted test), and finally **exactly the same two-choice fork input** for opposite food goals. Neural output actions during cue and waiting are discarded: **every model reaches the decision at the same time and with the same information**.

A stateless feedforward model, the same RNN with state reset at the decision, and the RNN with all cues removed must therefore obtain **exactly 50% per paired world**; a programmed one-bit memory control must obtain 100%. These are tested *analytically in unit tests*, with actual code execution. The train-only fixed curriculum gradually increases cue-to-decision delay (0/1/2 → 1/2/4 → 2/4/6). Cue-at-decision delay 0 is a training scaffold, **never a final-test condition**.

Eight separately evolved populations for an 8-hidden tanh recurrent network (260 weights) and 11-hidden stateless feedforward (268 weights) share **identical candidate × paired-world evaluation budgets**. Final main test uses **64 new paired worlds** and the OOD test **32 longer-delay pairs**, with resampling over genetic starts and world pairs. No parameters may be retuned on final worlds.

On each RNN champion evaluate (1) normal, (2) hidden state **zeroed immediately before decision**, (3) initial target cue **flipped**, (4) initial cue removed. Only a positive main advantage over FF **AND** the same-genome reset/cue controls counts as evidence of functional memory. An independently reported 16-world scaffold vs free-corridor transfer measures movement separately.

~~~bash
python -m neuroevolution.m4_3 --out results/m4_3
~~~

Read [M4.3 frozen protocol](docs/M4_3_PROTOCOL.md) and the [read-only GitHub Actions campaign](.github/workflows/m4-3-causal-recall.yml). Even a positive result is a narrow artificial cue-memory result, **not** synaptic learning in life, autonomous goals or ecological reproduction.

## M5.0 — A persistent in-world ecology, not another externally selected task

This is a **separate project track** motivated by M4.0–M4.3: training neural nets to solve experimenter-written tasks is not the same as observing adaptive organisms. M5.0 introduces a **continuous, energy-limited spatial world** in which real simulated organisms must forage, survive, and explicitly **choose to reproduce** to make descendants.

The environment is a 15×15 obstacle grid with renewable food and real resource limits. An organism has a small fixed-topology 12-tanh recurrent neural controller with 15 **local sensor channels** and five possible actions: four movements and REPRODUCE. Its genome is its numerical neural weight vector. Inheritance/mutation occurs **at in-world birth**, not in an external generational algorithm. Each offspring has limited energy, a recorded parent and a lineage. The animal dies from energy exhaustion or maximum age. **No external fitness function ranks networks and no elite/validation champion is selected**; reproductive outcomes alone influence which weights persist.

Three treatments share exact world seeds and **identical founder genomes**:
- **heritable_mutation**: child gets parent weights + small mutation.
- **random_offspring**: child has a fresh random genome; same birth cost and opportunity, but **no genotype heritability**.
- **heritable_no_mutation**: child copies parent weights exactly; tests inherited founder selection without new genetic variation.

The frozen main campaign runs **12 independent worlds ×400 ticks**; shifted experiment **8 worlds** with more walls, shorter food odor and scarcer renewal. Four worlds without any food/renewal test the energy bookkeeping and starvation. The scientific primary is whether inherited mutation increases **actual in-world birth counts** against random offspring; additional lineage, food, population, genome similarity and energy ledger metrics separate "births occurred" from "adaptive multi-generational selection".

~~~bash
python -m neuroevolution.m5_0 --out results/m5_0
~~~

Outputs include per-world CSV, 50-tick population checkpoints, detailed birth/lineage/food event JSON, SVG world snapshots and paired bootstrap comparisons. Read the [frozen M5.0 protocol](docs/M5_0_PROTOCOL.md) and [dedicated GitHub Actions workflow](.github/workflows/m5-0-continuous-ecology.yml).

**Boundary:** This is a designed artificial ecosystem. Energy/reproductive thresholds, food dynamics, neuron topology, odor sensors and mutational rules are **still written by the researcher**. It does NOT yet exhibit within-life synaptic learning, network rewiring, predators, consciousness, independent goal invention, or executable self-replication. Null or extinct populations must be reported honestly.

## M5.1 — Can our existing ecology sustain any sensible forager?

In M5.0, every population died out despite plentiful uncollected food. **M5.1 diagnoses this failure without changing M5.0's physical laws** and without inserting hand-coded competence into neural genotypes.

On **16 new main world seeds** (73000–73015) and **8 harder new shifted worlds** (74000–74007), compare the SAME 15×15 ecology, founding genomes, food events, energy and reproduction physics under four behavior policies: (i) the **unaltered original M5.0 neural action choice** with in-world heritable mutation, (ii) scripted greedy local food-odor follower plus fixed reproductive energy threshold, (iii) a *stronger programmed reference* using local food and a small personal visit-count memory, and (iv) a legal random walker with the exact same scripted reproduction rule.

The three scripted policies are **not evolved**: child genomes are tracked for continuity but do not determine their scripted actions. Scripted policies use age and in the novelty version their own visited-cell counts, which the neural controller does **not** receive; they are **diagnostic upper references**, not fair tests of emergent cognitive intelligence.

The fixed primary question is whether the programmed novelty forager leaves ≥1 living organism at tick 400 in ≥8/16 worlds and reaches generation depth ≥3 in ≥8/16 worlds. A separate necessary forage check requires a positive seed-paired food-consumption advantage for the purely reactive greedy follower over legal random motion. Four no-reproduction and four no-food-renewal controls test environmental limits; all inherit the unmodified M5.0 metabolism and birth-energy rules. Track food eaten, 50-tick population changes, actual births, starvation, odor alignment, food-adjacent opportunities and energy conservation.

~~~bash
python -m neuroevolution.m5_1 --out results/m5_1
~~~

Read the [preregistered protocol](docs/M5_1_PROTOCOL.md) and [read-only GitHub Actions campaign](.github/workflows/m5-1-ecological-viability.yml). Results include world-by-arm CSV, checkpoint CSV, population curve SVG, sample world maps, and event genealogies. Null findings are valuable: if even simple scripted agents die despite abundant food, the physical ecology itself needs further diagnosis before any new genetic search.

## Actions

[Validation workflow](.github/workflows/validation.yml): tests and brief smoke runs on pull requests/pushes.

[Original M1 research campaign](.github/workflows/research-campaign.yml): original structural neuroevolution, manually or weekly.

[Plasticity campaign](.github/workflows/plasticity-campaign.yml): 24 paired seeds, manually started or explicitly triggered with a kickoff marker. Archives generated metrics and SVG for later scientific review.

[M2 campaign](.github/workflows/m2-adaptive-renewal.yml): isolated training/validation/test and archived metrics; manually launched or explicitly triggered by a kickoff marker.

[M2B fixed-budget timing campaign](.github/workflows/m2b-budgeted-timing.yml): manually triggered or launched by an explicit kickoff marker, with new sealed seed sets.

[M3 unpredictable recurrence campaign](.github/workflows/m3-novelty-recurrence.yml): preregistered meta-training and independent regular/shifted final tests. Launch manually or by an explicit kickoff marker.

[M3.1 causal intervention workflow](.github/workflows/m3-1-causal-repair.yml): explicit kickoff/manual GitHub run, paired potential outcomes, archived metrics.

[M3.2 natural interference campaign](.github/workflows/m3-2-natural-interference.yml): isolated train/validation/final tests, explicit GitHub kickoff, archiving complete counterfactual data.

[M3.3 contextual memory campaign](.github/workflows/m3-3-contextual-memory.yml): isolated train/validation/sealed tests, paired conflicting context panels and replay experiments.

[M3.4 latent context inference campaign](.github/workflows/m3-4-latent-context-inference.yml): preregistered threshold selection and three independent test splits, with paired support ablations and complete CSV metrics.

[M3.5 fixed-information-budget campaign](.github/workflows/m3-5-active-information.yml): blinded support requests with identical query counts and independent shifted/noisy tests.

**Experimental status:** a passing workflow proves that code executed, *not* that a scientific hypothesis succeeded. Preserve negative findings and uncertainty.

## M5.2A — Finding the limiting ability by causal interventions

The [M5.2A preregistered 2×2 experiment](docs/M5_2A_PROTOCOL.md) asks whether original neural populations die because they cannot forage or because they cannot time reproduction. The [completed 30-world report](reports/M5_2A_causal_factorial_2026-11-21.md) and [per-world CSV](reports/M5_2A_world_survival_births_2026-11-21.csv) show: neural movement + neural reproduction **0/20** main worlds survive to tick 400; neural movement + programmed birth **1/20**; programmed odor movement + neural birth **20/20**; both programmed **20/20**. Harder shifted worlds: 0/10, 0/10, 9/10, 10/10. **Causal diagnosis: movement/food-search is the bottleneck in this world; programming an adequate motor policy is NOT the same as evolving it.**

Run deterministic engineering smoke:

~~~bash
python -m neuroevolution.m5_2a --seed-start 700 --seeds 2 --shifted-start 900 --shifted-seeds 2 --negative-control-worlds 1 --ticks 40 --size 11 --initial-food 9 --food-cap 14 --initial-population 8 --max-population 18 --checkpoint-every 10 --hidden 4 --out results/m5-2a-smoke
~~~

The full sealed study is archived in [GitHub Actions #44920201061](https://github.com/kkongiu/neuroevolution-lab/actions/runs/44920201061); never retune it after reading the test results. M5.2B will test compact autonomous neural foraging and genuine heritable selection.

## M5.2B — Testing genuine neural mutation, with no scripted skills

A [preregistered 51-weight recurrent genome](docs/M5_2B_PROTOCOL.md) keeps all five behaviors under actual inherited neural control. Compare exactly three neonatal mutated parameters, all-parameter mutation, strict cloning and fully random offspring. **Negative result:** [36-world M5.2B study](reports/M5_2B_compact_neural_2026-11-21.md) found **0/24** main-world survival for every arm after 400 ticks, and **0/12** in each shifted arm, despite correct physical births and [successful Actions validation](https://github.com/kkongiu/neuroevolution-lab/actions/runs/44921004288). [Per-world outcomes](reports/M5_2B_world_survival_births_2026-11-21.csv).

This genuinely tests inherited **neural decisions** unlike the scripted motor controls in M5.2A, but does not show adaptive selection. Reducing genome dimensionality and mutation scope alone did not overcome the foraging bottleneck. Before adjusting mutation hyperparameters, a separate experiment should verify if the chosen neural architecture can *represent* viable foraging and whether mutations can discover that region from uninformed initial weights.

Run small deterministic smoke:

~~~bash
python -m neuroevolution.m5_2b --seed-start 700 --seeds 2 --shifted-start 900 --shifted-seeds 2 --ticks 40 --size 11 --initial-food 9 --food-cap 14 --initial-population 8 --max-population 18 --checkpoint-every 10 --hidden 2 --sparse-mutated-weights 3 --out results/m5-2b-smoke
~~~

## M5.3A — Neural numerical capacity proved, evolutionary search still open

[Preregistered 36-world representational experiment](docs/M5_3A_PROTOCOL.md): with unchanged original M5.0 ecology and neural actor, a manually **engineered numerical** 401-weight genotype survived **24/24** ordinary worlds; independently **externally supervised** 401-weight genotype also **24/24**, and a 51-weight supervised genotype **17/24**, compared with fixed random or shuffled controls **0/24**. All numerical weights were *cloned unchanged* into genuine children. The result proves these numerical neural architectures can express viable ecological behavior, **not** that the organism's genome evolved it through natural selection.

**Negative OOD result:** all three engineered/externally trained genotypes failed in **12/12 shifted worlds** with increased walls and shorter food odor range; do not retroactively tune these sealed outcomes. [Full scientific report](reports/M5_3A_neural_representation_2026-11-21.md), [per-world data](reports/M5_3A_paired_worlds_2026-11-21.csv), [completed Actions run](https://github.com/kkongiu/neuroevolution-lab/actions/runs/44922015463).

Engineering smoke (separate from sealed study):

~~~bash
python -m neuroevolution.m5_3a --seed-start 700 --seeds 2 --shifted-start 900 --shifted-seeds 2 --teacher-seed-start 1100 --teacher-seed-count 2 --validation-teacher-start 1200 --validation-teacher-count 1 --teacher-ticks 35 --train-cap 150 --validation-cap 90 --epochs 3 --ticks 40 --size 11 --initial-food 9 --food-cap 14 --initial-population 8 --max-population 18 --checkpoint-every 10 --negative-control-worlds 1 --no-food-controls 0 --hidden 12 --out results/m5-3a-smoke
~~~

**Remaining challenge:** whether an initially *random* heritable genotype can evolve food-seeking neural behavior on its own, and maintain it when the ecology changes.

## M5.3B — First measurable inheritance signal from random neural founders

The [fully preregistered genetic experiment](docs/M5_3B_PROTOCOL.md) uses **independently random ten-gene founders**; genes mechanically expand into the native 401-weight neural actor using a **researcher-specified spatial weight-sharing architecture**. Unlike M5.3A, there is **no engineered or supervised founder genome** and no runtime scripted movement; real in-world births copy, mutate, clone or redraw numerical genotypes, and all 5 actions come from the original RNN actor. The architecture has strong designer-supplied inductive bias, so we are NOT evolving new wiring or discovering the sensor definitions.

On 24 main worlds, **sparse mutation 5/24 survived**, dense mutation 5/24, exact clone 3/24, independently redrawn offspring 0/24. Sparse versus random offspring survival **+20.83 pp**, world-paired descriptive bootstrap [+4.17,+37.50] pp; **an inheritance signal**. **But** novel-mutation benefit versus clones is unconfirmed, overall ecological viability misses the preregistered 12/24 gate, and all arms go extinct on **12/12 harder shifted worlds**. Keep negative conclusions. [Full scientific report](reports/M5_3B_de_novo_genetic_selection_2026-11-21.md), [all world outcomes](reports/M5_3B_paired_worlds_2026-11-21.csv), [sealed GitHub run](https://github.com/kkongiu/neuroevolution-lab/actions/runs/44922843440).

Run separate engineering smoke:

~~~bash
python -m neuroevolution.m5_3b --seed-start 700 --seeds 2 --shifted-start 900 --shifted-seeds 2 --ticks 40 --size 11 --initial-food 9 --food-cap 14 --initial-population 8 --max-population 18 --checkpoint-every 10 --no-food-controls 0 --hidden 12 --sparse-gene-count 2 --initial-gene-sd 0.65 --gene-mutation-sd 0.16 --out results/m5-3b-smoke
~~~

## M5.3C — Genetically selected descendants are viable, but not consistently better than founders

The prospective [genotype transplant study](docs/M5_3C_PROTOCOL.md) recovers all five previously successful original M5.3B random-founder evolutionary populations, selects the most-generation final living descendant per family by a **frozen deterministic rule**, and then clones the *unchanged* genes of each descendant, its actual ancestor, final-step mutation reverted, shuffled genes, and a separately sampled random control into **24 NEW ordinary and 12 NEW hard worlds**. All actions remain numerical native M5.0 neural; no new mutation or scripted action during transplant.

Among the 120 heldout **ordinary** world×family evaluations: **110 descendants survive, 104 actual ancestors, 113 last-step mutation reverted, zero shuffled and zero sampled random controls**. Descendant-minus-founder +5 percentage points, crossed-source-family/world bootstrap descriptive **[−20,+30]**, 3 families improve/2 worsen; **preregistered novel-mutation improvement gate FAILS**. Undoing the latest mutation is not harmful (113 vs 110). Shuffled genes fail 0/120, demonstrating numerical configuration matters, but not that natural mutation improved it. **All descendants and ancestors 0/60 in new harder worlds**. These families were *selected because they survived previously*; high homogeneous cloned-genotype transplant viability does not establish broad success of random-founder neuroevolution.

[Full honest scientific report](reports/M5_3C_genotype_ancestor_transplants_2026-11-21.md), [all 900 world outcomes](reports/M5_3C_world_transplants_2026-11-21.csv), [exact verified historical birth-by-birth source genealogies](reports/M5_3C_full_verified_source_genealogies_2026-11-21.json), [original complete 900-trial Actions run](https://github.com/kkongiu/neuroevolution-lab/actions/runs/44924136791) (165 tests green).

## M5.4A — The short-range odor bottleneck has been causally identified

After M5.3C showed that genetically selected neural descendants and their **actual random founders** remain highly viable as frozen clones in ordinary worlds but die under harder combined conditions, a [preregistered 2×2×2 factorial study](docs/M5_4A_PROTOCOL.md) tested the three ecological stressors separately. All maps, original food and founder positions, food candidate cells and random event marks were carefully paired: walls .12/.20, food odor radius5/2, regeneration chance .55/.40. The wall-density contrast explicitly holds the food candidate support set fixed to cells reachable under the high-wall map; do not extrapolate it uncritically to independently resampled complete original worlds. Ten fixed original source-descendant/founder numerical genotypes and a separate human-programmed novelty forager were tested across **16 new MAIN +8 new VALIDATION environments ×8 factor combinations = 2112 completed 400-tick trials**, unchanged original native M5.0 physical and neural actions.

**Result:** reducing **odor radius 5→2** causes **−81.25 percentage points** MAIN neural-descendant survival (descriptive crossed family/world bootstrap [−92.81,−63.75]), versus **−7.50 pp** for reduced regeneration and **−0.625 pp** for more obstacles. The protocol's unique dominant-factor criterion **PASSES**, and independent validation confirms odor −76.25 pp. With only odor shortened in otherwise benign ecology, survival collapses **73/80→4/80**. Original random founders show a similar odor vulnerability. The **programmer-written** greedy+novelty positive-control survives all eight environmental combinations in **all 16 MAIN and all 8 VALIDATION worlds**, confirming physical viability. This programmed controller is never a claim of neural evolution.

**M5.4A is a fixed-genotype causal ecology diagnostic, not an evolution result.** The next *untested* research question is whether random founders can evolve improved short-range exploration through actual inherited birth mutations versus clonal and independent newborn controls after a within-world sensory change.

[Full scientific report](reports/M5_4A_ecological_factorial_2026-11-21.md) · [ALL 2112 world outcomes CSV](reports/M5_4A_all_2112_world_factorial_outcomes_2026-11-21.csv) · [full preregistered factorial statistics](reports/M5_4A_factorial_stats_2026-11-21.json) · [original sealed 174-test GitHub run](https://github.com/kkongiu/neuroevolution-lab/actions/runs/44926123502).

## M5.4B0 — Short-odor numeric neural capacity: a rigorous negative finding

Following M5.4A's strong causal result that the short-radius-2 food signal is the main reason selected neural populations go extinct, [M5.4B0 preregistered a fixed external numeric search](docs/M5_4B0_PROTOCOL.md) of the exact original 10-gene, 401-weight neural subspace. A **researcher-controlled optimizer** evaluated **96 signed numerical candidate genomes** on four distinct developmental worlds (384 400-tick trials), selected the top six on eight new validation worlds (48 trials), then froze the highest-ranked candidate before **24 NEW ordinary odor2 plus 12 NEW harder odor2 held-out worlds**, compared with an equal-size gene scramble, four random numeric gene controls, and separately a **HUMAN-SCRIPTED** local greedy+novelty ecological benchmark (252 final trials).

**Result: the externally selected numeric neural witness survives 0/24 ordinary and 0/12 harder worlds; ALL other fixed numerical control genes likewise 0/24 and 0/12.** No one of the 96 candidates survived any development 400-tick world, nor did any of six finalists survive validation. The human-written physical forager survived **24/24 ordinary and 12/12 harder worlds**. The original ten-gene mapping has **zero recurrent connections** and cannot encode multi-step visit-memory, though this experiment has not proven no possible ten-gene vector could ever persist. **Numerical short-odor representability under the frozen budget FAILS; no within-world neural evolution was tested, no spontaneous adaptation claimed.**

[Full report](reports/M5_4B0_numeric_short_odor_capacity_2026-11-21.md) · [ALL 252 sealed world outcomes](reports/M5_4B0_sealed_252_world_outcomes_2026-11-21.csv) · [complete 96 original external numerical gene vectors](reports/M5_4B0_all_96_external_numeric_genes_and_ancestry_2026-11-21.json) · [432 preheldout development/validation world trials](reports/M5_4B0_external_432_development_validation_worlds_2026-11-21.csv) · [original GitHub Actions 182-test scientific run](https://github.com/kkongiu/neuroevolution-lab/actions/runs/44930989280).

**Next, not yet implemented:** independently preregister a small recurring numerical-memory architecture and compare with memory-off controls on NEW seeds; then only if representational viability is confirmed attempt actual random-founder genetic adaptation vs clones and random newborns. Never insert a programmer-selected external numeric winner into an allegedly uninformed evolved population.

## M5.4B0-R — Eight real bounded recurrent numeric genes added to the neural organism

The [prospectively preregistered recurrent architecture](docs/M5_4B0R_PROTOCOL.md) augments the exact old 10-gene numerical RNN with **8 inherited REAL neural recurrent weights**, making **18 signed numeric genes** total but retaining the original **401-weight, 12-hidden, 15 local-input, 5-action** M5.0 native neural processor, original full physical costed births, local food and 400-tick ecology. For each of the four directional hidden neurons i, `self_feedback=0.65*tanh(g10+i)`, `lateral_from_next=0.35*tanh(g14+i)`; no other weights change. When these eight weights are clamped to zero, ALL 401 weights return **bit-for-bit** to the original M5.3B 10-gene mapping. True numerical recurrent state now affects later motor/reproduction actions; this is not a programmer-scripted visited-cell lookup. All native genetic offspring are exact CLONES, so this representability study is NOT in-world evolution.

Both 18-gene recurrent and original ten-gene feedforward scaffolds were INDEPENDENTLY EXTERNALLY optimized at **identical frozen budgets** (96 numeric candidate vectors each ×4 original-radius2 development worlds, top6×8 independent validation), then compared on **24 NEW ordinary and 12 NEW harder** odor2 worlds against the SAME selected 18 genes recurrence-clamped, original feedforward external winner, scrambled genes, fixed random 18/10 controls, and a **separate HUMAN-SCRIPTED** novelty forager. Original [1,152 complete 400-tick physical experiments](https://github.com/kkongiu/neuroevolution-lab/actions/runs/44932841187), **193 regression tests green**, fixed original protocol SHA `7ffd8e06510a4c08a80c7610d7e0ff2617aec1a7`.

**Results: rare genuine numeric recurrence signal, preregistered robust viability FAILS.** Selected recurrent 18-gene neural survive **2/24** normal short-odor worlds (seeds 89203,89206), versus **0/24** same genes with recurrence OFF and independently optimized old 10-gene neural, and all random/scrambled numerical arms **0/24**. Main paired survival advantage **+8.33 percentage points**, 20k paired world-bootstrap descriptive 95% **[0,+20.83] pp**, **lower bound zero**, hence NO preregistered superiority. Mean food eaten/world recurrent **131.67** vs same-gene clamp **46.29**, mean native births **26.42** vs **11.83** (descriptive secondary gains, not proof of robust survival). On 12 separate harder odor2 worlds ALL numerical arms **0/12**. The HUMAN-programmed local forager survives **24/24 ordinary and 12/12 hard**, not evidence of naturally evolved neural intelligence.

[Full report](reports/M5_4B0R_eighteen_gene_recurrent_neural_capacity_2026-11-21.md) · [all 288 sealed per-world numerical and scripted outcomes](reports/M5_4B0R_all_288_original_sealed_neural_world_outcomes_2026-11-21.csv) · [full selected recurrent18 and feedforward10 signed genes and paired interval statistics](reports/M5_4B0R_frozen_recurrent_genes_and_preregistered_statistics_2026-11-21.json) · [all 192 external candidate genetic vectors/ancestry](reports/M5_4B0R_all_192_external_numeric_genomes_and_ancestry_2026-11-21.json).

**Next scientific question (NOT yet implemented):** understand whether the incomplete viability comes from insufficient numerical exploration state, birth timing or missing source information; any new sensor, architecture, reward or enlarged search must be preregistered on completely fresh worlds. Only after an independently convincing representational result should true random-founder birth-mutation-vs-clone adaptation under mid-world odor loss be attempted, with no researcher externally trained founder privileged in the actual evolution study.

[All 864 independently replayed development and validation world outcomes](reports/M5_4B0R_all_864_reproduced_external_development_validation_world_outcomes_2026-11-21.csv) (original 192 source-genome match verified in [read-only independent Actions audit #44933450090](https://github.com/kkongiu/neuroevolution-lab/actions/runs/44933450090), **no final-world access**).

## M5.4C — Causal foraging-vs-reproduction diagnostic (complete)

The prospectively frozen [M5.4C protocol](docs/M5_4C_PROTOCOL.md), preregistration commit `c351085a15ebb9a56af5f3d198c2f3939905ab80` BEFORE implementation/world results, isolates **NEURAL vs HUMAN-scripted movement** and **NEURAL vs HUMAN-scripted birth choice** in a matched **2×2** original physical ecology. The same previously externally selected 18-gene recurrent numerical genotype #83, original M5.0 401-weight digest `c5e5795332725de4`, original 15 local sensors, five native action preferences and real food/birth energy costs remain unchanged; every one of 144 full 400-tick world-policy simulations inherits EXACT CLONES at real births and **NO within-world genetic evolution** occurs. Programmed action alternatives are explicit causal diagnostics, not evolved skill.

Original completed [GitHub Actions #44938198849](https://github.com/kkongiu/neuroevolution-lab/actions/runs/44938198849), **203 tests GREEN**; 24 fresh original odor-radius2 MAIN worlds 91000–91023 plus 12 fresh SHIFTED hard worlds 91100–91111. Tick400 survival: **NEURAL movement+NEURAL birth 4/24 MAIN,0/12 hard; NEURAL movement+SCRIPTED birth 5/24,0/12; SCRIPTED M5.1 novelty+odor/food movement+NEURAL birth 23/24,12/12; SCRIPTED movement+SCRIPTED birth 24/24,12/12**. Preregistered marginal survival gain of HUMAN-scripted movement **+79.17 percentage points**, 20k paired world bootstrap 95% **[+66.67,+89.58]**, compared with birth policy script alone marginal **+4.17 pp [−8.33,+14.58]**; matched movement-minus-birth advantage **+75 pp [+58.33,+91.67]**. Registered movement-dominant causal bottleneck gate **PASS**. No proof that specifically visit memory alone causes the rescue: programmed movement simultaneously changes food/odor weighting, personal visit count and tie-breaking.

Behavioral mechanism: original neural-move agents revisit their personal already visited cells on **81.9%** of moves (mean 9.56 distinct lifetime cells), while scripted-move+native-neural-birth organisms revisit only **20.9%** (mean **40.18** distinct lifetime cells); mean physical food eaten per world 146.67→387.50. Neural reproduction can sustain 23/24 ordinary and 12/12 hard populations **when a human-coded actor chooses movement**, so the dominant failure is local exploration rather than birth physics. Critically the original genotype was previously researcher-externally selected, never randomly evolved here; the favorable script is NOT a learned neural capability.

[Detailed report](reports/M5_4C_foraging_vs_reproduction_diagnostic_2026-11-21.md) · [ALL 144 complete per-world original neural/script survival, food, energy and visit diagnostics](reports/M5_4C_all_144_paired_world_full_diagnostics_2026-11-21.csv) · [ALL 1296 original 50-tick physical population checkpoints](reports/M5_4C_all_1296_population_checkpoints_2026-11-21.csv) · [complete exact 18 numeric genes, 20k world-bootstrap stats, and prereg gates](reports/M5_4C_exact_source_genes_and_all_preregistered_effects_2026-11-21.json) · [all matched world geometry/food schedule SHA256](reports/M5_4C_36_exact_paired_world_genome_and_food_schedule_sha256_2026-11-21.json).

**Next scientific proposal, NOT yet demonstrated:** M5.4D independently preregister individual movement-rule ablations (explicit personal novelty memory, odor/adjacent food weighting, stochastic exploration) with ORIGINAL neural birth left intact, followed by testing whether a genuinely NUMERICAL neural motor with source-independent heldout worlds can express the successful behavior BEFORE ever claiming actual inherited mutation-driven adaptation.

## M5.5A — Eight truly inherited genes regulate human-programmed imperfect instincts (original real birth mutation, 180-world study completed)

This is a deliberately **new alternative organism architecture** following [M5.4C's causal movement bottleneck diagnosis](reports/M5_4C_foraging_vs_reproduction_diagnostic_2026-11-21.md), **not** a successful naturally learned 401-weight native neural controller. The *PROGRAMMER supplies* the underlying lawful reflexes: physical obstacle avoiding, original local food/odor sensors, bodily hunger modulation, personal per-life visit memory and heading, stochastic motor search, and physically eligible reproductive urge. **Eight signed real inherited numeric genes** modulate food weighting, odor sensitivity, spatial novelty strength, inertial heading, random exploration, energy hunger gain, birth threshold and birth attempt probability. Every genuine original physical birth preserves exact native M5.0 energy/age/food/population legal costs; sparse two-gene or dense eight-gene **actual parent→newborn mutations** are applied at real births before the child can act, along with exact clones and random newborn gene controls. No externally optimized prior original 18-gene genotype is used: each source world starts from **20 independently RANDOM uninformed eight-gene founders**.

The [prospective protocol](docs/M5_5A_PROTOCOL.md) was frozen at commit `a3aa39e4f68acc38d17a4292b4ebefbd4b72b10b` **BEFORE code or outcomes**. Original complete [scientific GitHub Actions #44940687397](https://github.com/kkongiu/neuroevolution-lab/actions/runs/44940687397): **213 tests passed**, exactly **five arms × (24 new MAIN+12 new harder SHIFTED) original short-odor2 worlds =180 complete 400-tick original M5.0 original-physics trials**, **1,620** checkpoints, **16,134** physically costed births across all arms. True within-birth genetic mutation occurred **7,195 times** (3,610 sparse 2/8, 3,585 dense8/8). Random-genotype newborns occurred in a distinct 3,595 births, and both clone arms had **0** new genetic mutation.

**Main 24 worlds tick400 survival:** innate exact clones **24/24**, genuine sparse genetic mutation **24/24**, dense mutation **24/24**, independent random newborns **24/24**, pure random-legal-movement negative motor control **14/24**. **Hard 12 worlds:** innate clones **12/12**, sparse **12/12**, dense **11/12**, random newborns **12/12**, pure random legal movement **2/12**. Real 20k world-paired bootstrap clone-vs-random-motor survival effect +41.67pp [20.83,62.50] MAIN and +83.33pp [58.33,100] HARD shows the **PREPROGRAMMED** instincts' ecological viability. **Actual mutations did NOT show a demonstrated advantage**: sparse vs clonal 400-tick survival delta 0pp [0,0], mean checkpoint population −0.0278 organisms [−0.3287,+0.3102] MAIN and +0.0093 [−0.6019,+0.5185] HARD, intervals include0. This is a **100% survival ceiling**, not evidence of mutation-driven adaptation. Independent genuine offspring-vs-its-own-original-untrained-founder lineage transfer was **not** tested. Genes genuinely change, but no automatically learned new instinct and no verified beneficial alleles.

[Full original candid scientific report](reports/M5_5A_innate_scaffold_with_actual_birth_mutation_2026-11-21.md) · [all 180 original world-genetic outcomes](reports/M5_5A_all_180_original_genetic_trait_world_outcomes_2026-11-21.csv) · [all 1,620 original physiological/gene-mean population checkpoints](reports/M5_5A_all_1620_original_population_and_trait_checkpoints_2026-11-21.csv) · [all 20k preregistered real heredity vs clones paired differences/95% intervals and gates](reports/M5_5A_all_preregistered_world_paired_mutation_statistics_2026-11-21.json). Original complete 36 exact random founder full-precision genotype banks and full 16,134 original physical newborn gene genealogies are preserved in the original Actions artifact ID14720899006, retention30 days; original deterministic code and all aggregate source trial/checkpoint data stay versioned. Next scientific study must introduce genuinely informative **NEW preregistered selection pressure** and independently test descendant alleles against own exact uninformed founder before claiming adaptation; the separate prior original 18-gene NEURAL exploration issue #46 remains independent.

## M5.5B — Physically embodied enemies create selective survival pressure

A new **separate** preregistered study, preserving published M5.5A, adds mobile one-step predators that can actually kill organisms, a local enemy detector, and a ninth **real inherited numeric gene** tuning the strength of programmer-supplied flight. The original 15 local wall/food/odor inputs, food and energy budget, genealogy, legal movement/reproduction, and physics ledger remain unchanged. This is **programmed fear, not evolved neural intelligence**.

The original full [180-world M5.5B study](reports/M5_5B_physical_predator_selection_original_2026-11-21.md) passed **221 tests**, 20k world-paired bootstrap comparisons and original energy balance. In 8 MAIN worlds, original innate clones survive **8/8 with no predators**, **8/8 with one**, and **3/8 with three**; removing the preprogrammed fear detector leaves **0/8** with three. Sparse real two-gene mutation survives **4/8**, but its relative benefit versus clone remains statistically **inconclusive**. Three predators in HARD cause universal extinction, not useful evolutionary improvement. The complete original [180 physical outcomes](reports/M5_5B_all_180_original_predator_world_outcomes_2026-11-21.csv) and [all paired statistics](reports/M5_5B_original_180world_paired_bootstrap_statistics_2026-11-21.json) are permanently archived. Original raw full action ledger available in [GitHub Actions](https://github.com/kkongiu/neuroevolution-lab/actions/runs/44948142490) through 21 December 2026.

Run the full frozen exploratory study:

~~~bash
python -m unittest discover -s tests -v
python -m neuroevolution.m5_5b --out results/m5-5b
~~~

**Scientific status:** a real nontrivial physical selection environment has been found; spontaneous beneficial mutation and transferable evolutionary adaptation have **not** been demonstrated. See the [prospective protocol](docs/M5_5B_PROTOCOL.md). Reserved independent confirmation and ancestor/descendant-gene-reversion tests are next.
