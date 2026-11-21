# M4.1 — Legal-action masking versus evolutionary food-seeking

**Preregistered before M4.1 code and all final-seed results.** M4.0 source, champion, seeds and negative findings remain unchanged.

## Question and hypothesis

When every neural controller is prevented from selecting an immediately blocked move, can replicated neuroevolution discover a reproducible food-search advantage over a wall-avoiding random walker? Can improvements survive new mazes and genetic population initializations rather than only one lucky evolving run?

Masking is an **experimenter-defined motor constraint**, not a cognitive achievement. Neural organisms still only evolve inherited numeric weights BETWEEN lifetimes; no online learning, birth, predators, or autonomous goals.

## Frozen world

Reuse M4.0 world physics and sensors: 13x13 grid, impassable walls, finite reachable food, 24 starting energy, movement cost 1, collision surcharge 0.30, food energy +16 (cap 42), 100-step maximum; 15 sensors including blocked neighbors, food in adjacent cells, bounded local directional odor (radius 5), energy and recent outcomes; four cardinal actions. Main density 0.16 with 12 food; shifted density 0.26 with 8 food and odor radius 2. Do not change fitness: 2×food + 0.03×steps alive −0.05×wall collisions.

**New spatiotemporal variation without changing physics:** for each world seed, start from the deterministic reachable M4.0 maze, apply a seed-specific rotation by 0/90/180/270 degrees to walls and food, and choose an RNG-specified starting cell among reachable, unoccupied cells with ≥2 legal adjacent moves. The starting location is sampled from the same connected component as every resource. All policies face the **identical** map, start, resources and energy. The same transformation applies in training, validation, main and shifted worlds. No world test seed influences genome choice.

## Controller and action mask

The network is M4.0's 12-unit tanh recurrent policy (15 inputs, 4 action logits), with identical genome encoding, initialization and Gaussian mutation. When masking is ON, compute logits as usual, but choose the highest-logit **legal** action when the network's preferred action hits a wall. Only the **four existing local blocked flags** decide legality. Record every masked illegal preference as an **attempt corrected**. When masking is OFF, use the original unmodified argmax and actual collisions are counted.

Compare two independently evolving populations per genetic seed, both **initially identical**, with equal evolution budget:
- **mask-trained**: all training and validation episodes execute with the legal-action decoder.
- **unmasked-trained**: all training and validation episodes use original argmax.
No oracle food direction, shortcut to food, new task objective, reward change, or extra SGD.

At final test evaluate **both genotypes** using both decoders (2×2 cross-over) to separate decoder effects from learning effects. Also evaluate masked evolved without the **four odor channels**, keeping genome and decoder fixed.

## Evolution budget and independent partitions

- Genetic initialization seeds: **81000–81007**, exactly **8 independent evolution replicates**.
- For EACH genetic seed and decoder mode: **6 generations × 10 genomes × 6 training worlds/generation = 360 candidate-world training episodes**; **4 fixed validation worlds per generation**, selecting the generation leader using validation fitness only.
- Training world seeds: **45000–45035** (6 NEW worlds per generation, same sequence for every candidate and genetic run). Validation: **46000–46003**.
- Sealed independent main test worlds: **47000–47023** (24).
- Sealed shifted test: **48000–48011** (12), more obstacles, less food and shorter odor.
- For each run and decoder mode, fixed initial population from random.Random(genetic_seed) normal(0,0.45), two untouched elites, two random immigrants, Gaussian offspring mutation SD 0.28. Evolutionary searches use equal fitness evaluations, same training maps, and validation maps. No exchange of evolved genomes between treatments.
- Total training evaluations: 8 genetic runs × 2 training modes × 360 = **5760** candidate-world episodes, plus **384** validation leader-world episodes. All 8 completed runs must be included; no picking the best replicate using final test.

## Prespecified world controls

Across the same held-out world seeds and random episode starts:
1. **evolved_masked**: masked-trained champion with legal-action decoder (primary).
2. **evolved_masked_unmasked**: same masked-trained genome deployed WITHOUT decoder (mask-specific causal ablation).
3. **evolved_unmasked**: unmasked-trained champion deployed without decoder.
4. **evolved_unmasked_masked**: same unmasked-trained genome deployed WITH decoder (factorial cross-over).
5. **random_neural_masked**: one fresh, *predeclared independently generated* 12-unit random neural genome for EACH genetic run, with legal decoder (8 different random genomes).
6. **wall_avoiding_random**: uniform choice among locally legal directions, not informed of food.
7. **greedy_odor**: non-neural, highest directional odor among locally legal actions, tie-random using fixed per-world RNG. Programmed strong heuristic.
8. **evolved_masked_no_odor**: same mask-trained champion, legal decoder, odor channels set to zero. Identical model elsewhere.

No-food negative control: on a fixed subset of **8 main worlds**, remove all food but keep walls and starting state; no mechanism can yield food or energy from food. Report trial lengths and any corrected preferences, do not optimize on this control.

## Frozen outcomes and gate

**Preregistered primary:** across 8 genetic-run replicates and the same 24 main worlds, mean food difference **evolved_masked − wall_avoiding_random**. Quantify uncertainty using **hierarchical paired bootstrap**: independently resample entire genetic seeds (8) and world seeds (24), each with replacement, keeping shared worlds paired for all arms; percentile 95% descriptive bounds. Report mean per genetic seed and mean per world, not 192 independent evolved searches. Positive evidence requires mean >0, lower hierarchical 95% bound >0, and ≥6/8 independently evolved champions beating random in their average main worlds.

**Necessary secondary:** evolved_masked must outperform random_neural_masked on the same split with hierarchical paired inference. Report the comparison to greedy_odor even if unfavorable; a win over random alone is not "intelligence".

**Causal separation:** for the same mask-trained genome compare *masked versus unmasked decoder* on identical worlds (food, illegal attempts, actual collisions). For the same unmasked-trained genome compare both deployments. Between training regimes compare masked-trained/masked deployed vs unmasked-trained/masked deployed. Explicitly separate **mask effect** from **evolutionary search**; do not attribute wall avoidance to neural learning.

**Transferred evaluation** on 12 shifted worlds and all 8 genetic runs, frozen policies. Also report navigation length, visited unique cells, food/steps, deaths, total blocked-action preferences (corrected versus actual collisions), use of odor in the ablation, time-budget proxy and validation/train fitness trajectories.

Worlds are the unit for environmental variation and independently initialized evolution runs the unit for genetic-search variation. A two-level bootstrap is descriptive, not proof of universal generalization. Seed numbers and all hyperparameters are immutable once full holdout results are exposed; publish negative results.

## Engineering safety and interpretation

No in-world reproduction, predators, food regrowth or synaptic updates within a life. The source and its GitHub permissions cannot be modified by simulated organisms. No hidden global map, food coordinates, test seeds or future data available to the policy; the legal-action decoder only sees blocked-adjacent sensor flags. M4.0's negative result remains the baseline, not overwritten or retuned.
