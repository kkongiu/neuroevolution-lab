# M5.0 — A persistent ecology: reproduction, heritability and selection WITHOUT external candidate fitness

**Pre-registration**, committed before any M5.0 implementation or outcomes. M4.0–M4.3 remain untouched. This is a NEW research track, not M4.4.

## Question and limits

Can numerical recurrent neural genomes persist, reproduce and undergo selection **INSIDE one continuous spatial ecology**, without an outside genetic optimizer selecting champions on a shaped supervised score? Does **heritable transmission** outperform offspring whose weights are redrawn at every birth, when the *same reproduction rules, environmental energy accounting and ecological opportunity* apply?

This is artificial natural selection, **not proof of new intelligence, autonomous invention of life's laws, consciousness, online synaptic learning, open-ended adaptation or independent goals**. The experimenter sets the environment, actions, resource dynamics and reproductive energetics.

## World and organisms, frozen configuration

- Finite **15×15** grid with walls, outside border impassable. Interior wall density **0.12** (normal), **0.20** (shifted) using the existing deterministic BFS-reachable M4 world generator. Each world has **20 initial food pellets**, spawned only on connected reachable unoccupied cells; all subsequent food spawns use a **precomputed world-seed-dependent schedule of candidate cells** that does not depend on genetic mode, agent choices or outcomes. An attempted spawn into a wall, existing food or occupied cell is skipped, not replaced with a more favorable hidden target.
- **Food renewable**: at each tick, **two candidate resource-spawn events**, each occurring with independent fixed probability **0.55** (expected 1.10 candidate additions/tick). Food capacity **32** units; each food item adds **18 energy** to the first organism to enter it and is then removed exactly once. No spontaneous nutrition otherwise.
- **20 founding organisms**, initial energy **36** each, maximum energy **54**, maximum population **64**. Each founder starts on distinct reachable free cells. Founder genomes drawn from the same fixed seed-driven normal(0,0.45) distribution across matched arms. All founders have unique lineage IDs; no artificially inserted immigrants after initial population.
- 12 tanh recurrent neurons, **15 inputs**: four blocked-neighbor flags (wall OR organism), four local directional food-odor channels (radius **5** main, **2** shifted), four adjacent-food flags, energy fraction, previous failed action flag, and previous food-consumption flag. Controller actions: four cardinal moves **and a fifth explicit REPRODUCE action**. No true food locations, absolute coordinates, world seed, reward values, unobserved states or privileged map input.
- **Movement** costs **1.0 energy per organism per tick**. Choosing reproduction (or an invalid reproduction, or no legal movement) costs **1.0 basal energy** regardless of success. Motor mask makes blocked motion illegal, as in M4.1, but does not choose an optimal route; if reproduction is ineligible it is removed from the legal-action set. If all four movements are blocked and reproduction is ineligible, organism stays and pays basal energy. The controller chooses between available legal actions.
- **Reproduction requires** age >= **4** ticks, parent energy >= **34**, an unoccupied adjacent traversable cell, and population below capacity. If a REPRODUCE action is selected, a newborn is placed in a randomly chosen free adjacent cell (separate deterministic per-world/mode-independent event stream keyed by parent ID and time). **Parent pays 20 energy** plus its one-unit basal cost; **offspring starts at 12 energy**. The remaining 8 energy in the reproduction transfer is dissipated: **energy is not created by birth**. Genotype is inherited with Gaussian mutation SD **0.12** per weight (clipped ±5); newborn recurrent activations reset to zero. Parent and child IDs and ancestry recorded.
- All organisms pay basal energy for each tick in which alive and scheduled. They die on energy <=0 or age >= **100 ticks**. Death removes occupied location; no carrion. No predator, combat, agent networking or interaction outside resource competition and spatial blocking. Organisms **cannot execute code**.
- **400 ticks per world**. Every tick uses an RNG-derived deterministic **shuffle of existing organism IDs** before acting; newborns begin acting on the **following** tick (cannot reproduce in the same tick as birth). Food regeneration occurs after agent actions, using precomputed candidates; collision/spatial exclusion is the only competition. Events, births/deaths, energy balance, mutation and positions are auditable.
- **Selection is only ecological**: reproduction occurs when organisms choose the reproduction action and can afford it. There is **no outside fitness ranking, elite copying, scheduled generation or external champion selection**. Genotype survival/reproductive success emerges from trajectories and resource limitation.

## Scientific arms and controlled inheritance intervention

All arms use exactly the same world-seed schedule and initial founding genotype distribution; compare independent replicate worlds (not individual children as independent data):

1. **heritable_mutation**: at birth, inherit **actual parent's** genome plus Gaussian mutation (SD 0.12).
2. **random_offspring**: same parent/child event/energy action, but newborn genome redrawn independently from the founding normal distribution. The lineage genealogy is recorded but phenotype is **not transmitted**.
3. **heritable_no_mutation**: newborn exact clone of parent's genome; heritable selection among initial genotypes but **no new variation**, separates inheritance from de novo mutational search.

All three arms use the same RNN structure, 15 input channels, action count, physical resource and death parameters. Models start with **identical founders** on identical maps at each seed. Reproduction can be entirely unsuccessful; **do not rescue populations artificially** after starvation.

**No-food negative control**: in a small independent subset of four main world seeds, run a fourth **heritable_mutation** arm with zero initial food AND zero regeneration, verifying that total available energy does not increase and all finite founders eventually die. This is diagnostic and not a cost-matched adaptive policy.

## Seed partitions, evaluation and audit

No meta-tuning against M5.0 final seeds. Before final run, validate physics via unit tests and an explicit small **smoke seed** range 700–701 at 40 ticks; smoke results have no scientific inferential role.

- **Main independent ecosystem seeds:** **71000–71011** (12 worlds), 400 ticks, normal map/food/odor.
- **Shifted independent seeds:** **72000–72007** (8 worlds), 400 ticks, obstacle density 0.20, local odor radius 2 and food-regeneration event probability **0.40** rather than 0.55. No genome reuse from main across worlds; **evolution happens online inside each independent world**, not from a separately trained champion.
- **No-food control:** first four main seeds 71000–71003 under same topology and founders, food and renewal disabled.
- No independent training/validation champion selection is needed: the simulation's endogenous selection is the object of study.

## Primary hypothesis and preregistered statistical gate

Primary descriptive outcome: **cumulative in-world births over 400 ticks**, compare **heritable_mutation − random_offspring** in the 12 paired main worlds. Report seed-paired bootstrap percentile 95% interval, sign counts and mean. Evidence favoring inherited variation requires mean positive, lower descriptive interval >0, and ≥9/12 worlds positive. These are not independent births: the **world seed** is the experimental unit.

**Necessary secondary:** at least one lineage in ≥9/12 heritable_mutation worlds must reach **generation depth ≥3** (founder generation 0); compare end-state survival and mean lineage depth with random_offspring. Passing only the primary births count is **not proof of cognitive adaptation**: energy ecology and a programmed reproduction action may suffice.

Additional outcomes:
- Population sizes at ticks **0/50/100/200/300/400**, extinction time if any, cumulative births/deaths, total food eaten and generated, food stock and end energy; fraction of births vs deaths as a resource balance.
- Lineage depth, surviving founding lineages, distribution and top-lineage share of descendants, intergenerational **genome cosine similarity**, genome parameter change across generations and motor/reproduction action frequencies. No genotype is evaluated by an external food classification score or preselected on a hidden test.
- Ecological energetic accounting: food energy is the **only** environmental input; basal maintenance, capped food restoration and reproduction dissipation are losses. Track cumulative injected food energy and total organism energy; verify per-action invariants and nonnegative energy.
- **heritable_no_mutation** isolates selection of founder genotypes from ability to produce new variants. Without heritable_mutation outperforming this baseline, claim only inheritance-mediated selection, not beneficial de novo genetic innovation.
- **Shifted test** as independent ecological robustness check, not a source of retuned parameters. If all populations go extinct quickly, report it explicitly.
- Full per-seed and 50-tick checkpoint CSV, event/lineage traces (at least one main and shifted world), population-size plots SVG and hashed genome snapshots, determinism checks.

## Interpretation and safety

An organism whose neural weights control movement and *whether to reproduce* is more embodied/autonomous than a classifier externally bred by a validation fitness. Still, energy costs, reproduction eligibility, fixed topology, genetic mutation and the instinct to favor reproduction are all researcher-specified. A fifth action enables choice but does not entail independent intention.

Food odor is a researcher-designed directional cue; the controller is not learning a map. Heredity/selection alone cannot demonstrate cognition. Even if multi-generational descendants and improved birth counts appear, environmental exploitation can arise from crude fixed locomotion/reproduction biases. **Do not call this "new life".** No simulated agent gets code execution, file access, network access, source editing, tool calling, or self-replicating executable capabilities.

Publish positive, negative and null findings identically. Do not alter the frozen main-world seeds, energy costs, fitness definition (there is none), or reproduction parameters after observing final results. A later M5.1 can introduce new ecological challenges / true learning and independent tests.
