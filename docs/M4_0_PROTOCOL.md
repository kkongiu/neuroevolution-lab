# M4.0 — From classification tasks to a minimal survival world

**Preregistration** — committed before implementation and the first M4.0 evaluation. M1–M3.5 code and published findings remain unchanged.

## Scientific question

Can **selection and mutation of numerical parameters** in a tiny recurrent neural controller produce a reproducible advantage in survival-related **food acquisition** in a spatial world containing impassable obstacles, limited energy and scarce edible resources? Does any advantage persist in mazes that were not used for training, and in an altered environment?

**This is not a test of consciousness or autonomous goals.** Every organism's objective and fitness are set by the experimenter. A recurrent hidden state is internal short-term memory, **not** synaptic learning within a lifetime. No predators and no reproduction *inside* the world in M4.0; those belong to separate milestones.

## World

- Discrete **13 × 13** grid; outside boundary walled, inside walls randomly distributed by an independent world RNG.
- Starting cell at the centre, never a wall, and four adjacent cells are kept clear.
- Each candidate map must contain at least 40 reachable interior cells under 4-neighbour connectivity; finite deterministic retries if needed. Food is placed **only on reachable empty cells**.
- **Training/validation/main test:** obstacle probability 0.16, 12 food pellets and odor/sensing radius 5.
- **Shifted test:** obstacle probability 0.26, 8 food pellets and odor/sensing radius 2. The same controller is evaluated without retraining.
- Each agent starts with **24 energy units**, movement costs 1 energy (including failed collision moves), hitting a wall adds 0.30; food restores 16 energy up to a maximum of 42. Reaching zero energy ends the episode. Horizon **100 steps**. Food is consumed once; no regrowth.
- Four actions: north, east, south, west. A collision leaves position unchanged but consumes energy.
- 15 local sensor values: 4 adjacent blocked-cell flags, 4 bounded directional food-odor intensities from food within the prescribed sensing radius, 4 immediately adjacent food flags, normalized energy, last collision flag, and last-step food feedback. **No** absolute map, true food coordinates, optimal route, future values or world seed.
- Odor is a **designed environmental signal**, potentially simplifying navigation. The same odor is offered to the greedy baseline. No claim of visual-only navigation.

## Neural controller and between-generation evolution

- One-layer Elman-style RNN: **12 tanh recurrent units**, 15 inputs, 4 action logits, deterministic argmax. Inherited weights change **between generations only**. Hidden activations reset at each episode; **no SGD or weight updates during an episode**.
- Initial population **12 random controllers**, weights normally initialized with SD 0.45. Two elite controllers retained unchanged each generation; their offspring receive independent Gaussian mutations SD 0.28; two random immigrants per generation. Numeric weights only, fixed topology.
- **8 generations × 12 candidates × 8 train worlds = 768** candidate-world evaluations; all candidates face exactly the same training worlds.
- Training fitness: 2 × food_collected + 0.03 × steps_alive − 0.05 × wall_collisions. Each generation's train winner is evaluated on **four independent validation worlds**; a single overall champion is selected on validation fitness. Sealed final test seeds are not accessed before that choice.

## Frozen seed partitions

- Training worlds **41000–41007** (8); validation worlds **42000–42003** (4).
- **Main independent test 43000–43031** (32).
- **Shifted independent test 44000–44015** (16).
- Genome, world and stochastic-baseline RNG streams use separate deterministic seeds. One frozen selected genome is deployed unchanged to all final worlds.

## Controls on the same worlds

1. Evolved recurrent neural policy.
2. Unevolved neural policy: a single predeclared random RNN from fixed independent seed **73001**, same architecture/sensors.
3. Wall-avoiding random walker: uniform action among nonblocked immediate neighbours, independent per-world RNG.
4. Greedy odor follower: highest odor among nonblocked directions; when no odor, uniform random among nonblocked directions. Strong *programmed* heuristic, not learning.
5. No-odor ablation: exact same selected evolved genome with four odor channels zeroed, without retraining.

All controls face the same maze, food, energy and horizon. A neural controller, greedy heuristic and random policy **do not have identical CPU costs**.

## Prespecified outcomes

**Primary:** on **32 independent main worlds**, paired difference in foods collected: evolved neural minus wall-avoiding random, with descriptive seed-paired percentile bootstrap 95% and positive/negative/tied seed counts. To call this favorable evidence, require mean > 0 and bootstrap interval excluding zero. This is **not** proof of beating a hand-coded food-following algorithm.

**Necessary secondary:** evolved must also exceed the unevolved random RNN on paired food collection. Report greedy performance regardless of who wins. For positive food-seeking claims, ≥70% of worlds must have at least one eaten pellet.

Additional: steps survived, food eaten, collisions, end energy, distinct visited cells, food-reachability checks, train and validation history, no-odor ablation. Report food dependence on the designed odor sensor explicitly.

**Shifted world:** more walls, less food, shortened odor radius; no retuning. Results are secondary and must not be substituted for the primary.

Bootstrap over **world seeds**; steps and food pellets are not independent experimental units. Preserve null/negative results.

## Integrity and interpretation

Test deterministic seeding, identical starting worlds, no food placed inside walls or unreachable regions, food removed after consumption, no walking through walls, energy conservation except eating, termination on starvation and horizon, and no hidden map access to the neural controller.

M4.0 studies **evolutionary search over inherited neural weights**, NOT within-life learning, self-goals, autonomous structural modification, reproduction inside the ecology or predators. Later milestones can add food regeneration, reproduction, learned spatial memory and eventually predators **one variable at a time**.

Simulated organisms cannot access files, the shell, network, GitHub, source code or external tools.
