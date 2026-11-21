# M5.4C — PRE-REGISTERED 2×2 causal motor-foraging vs birth-policy factorial

**Protocol FROZEN before any M5.4C implementation or any fresh-world outcomes.** Date 21 November 2026. Research foundation: [M5.4B0-R original report](../reports/M5_4B0R_eighteen_gene_recurrent_neural_capacity_2026-11-21.md), [fully frozen externally selected recurrent18 genotype and statistics](../reports/M5_4B0R_frozen_recurrent_genes_and_preregistered_statistics_2026-11-21.json), both in prior main commit `387b7abe0bbe97526cfcb82a7d8050ed2c16f52f`. The previous selected recurrent18 numerical genome is a **researcher-externally trained**, **selection-conditioned** fixed genotype, NOT a de novo evolutionary origin. No further genotype searches, retuning or model selection in M5.4C.

## 1. Primary question and intervention

Previously the externally selected 18-gene numerical RNN survived only 2/24 original-radius2 normal and 0/12 hard habitats, whereas the **HUMAN-PROGRAMMED** M5.1 local greedy+visit-novelty forager survived all 36 matched hard+normal worlds. That comparison confounds **movement/foraging** with **when to reproduce**.

Test **BOTH intervention axes independently** using exactly the SAME fixed recurrent neural 18-gene genotype, initial founder positions and food schedules. Switch the responsible **policy choice** only, leaving **ALL physical M5.0 native legal move masks, physical birth eligibility, basal energy, newborn cost, food, age, movements, population caps and 400-tick horizon** unchanged.

2×2 arms, names frozen:
- `neural_move_neural_birth` (**NN**): original unmodified M5.0 `act` chooses between four legal moves and legal reproduction from the original native 18-gene recurrent RNN's five action logits. Original honest neural control.
- `neural_move_scripted_birth` (**NS**): if eligible and energy>=**46.0**, **HUMAN-PROGRAMMED** birth is forced, regardless of neural birth logit. Otherwise native RNN chooses highest-logit LEGAL movement among four directions (never a neural birth). Movement still truly numerical neural; birth is explicitly researcher-programmed. When no legal move, physical no-op still charged as original.
- `scripted_move_neural_birth` (**SN**): birth can occur **ONLY IF** native RNN's REPRODUCE logit is the maximum among ORIGINAL legal moves+birth AND original age/energy/capacity birth eligible; otherwise use the separately HUMAN-PROGRAMMED M5.1 `greedy_novelty` **movement-only** policy (4×adjacent food + directional local odor +0.35/(1+personal-neighbor visit count), deterministic tie-break RNG same M5.1 original tag37). No scripted birth choice is used; neural birth logits still evaluate once/turn. Movement is programmed.
- `scripted_move_scripted_birth` (**SS**): exact **HUMAN-PROGRAMMED** M5.1 `greedy_novelty` action: scripted energy-based birth if physically eligible and energy>=46.0, else precisely same scripted movement-only policy as SN. Both motor and reproduction are explicitly programmed; NEVER call this neural, learned, evolved or organic intelligence.

Every action begins with the SAME original 15 local sensor channels, legal moves, physical age/energy, and original RNN weight vector. When neural decisions are used, evaluate original unmodified `m5_0.network_logits` with original recurrent hidden state EXACTLY ONCE per action; if original NN control must temporarily shadow the forward pass for per-action diagnostics, immediately restore hidden and then call untouched `m5_0.act` so net forward state is EXACTLY once. NN must reproduce bitwise original `m5_4b0r.simulate_numeric` for same seed, genotype, source, physical counters and energy.
In hybrid/script-only controls, the RNN may be run for diagnostic/birth logic, but its logits are NEVER allowed to silently control a policy dimension assigned to the programmer.

Physical act implementation: `m5_1._act_scripted` adapter uses the ORIGINAL `m5_0` physical energy/food/birth physics and physical legal mask but accepts a specific preselected action, setting `eco.arm="heritable_no_mutation"` so **every newborn in ALL FOUR arms is an exact numeric clone of the SAME 401-weight recurrent numerical genotype**. The script-only arm's numeric genotype is INERT under scripted choice, but still recorded for paired fairness; no M5.1 mutable 401-weight random bookkeeping agent or new genotype. Verify same physical law for a forced legal birth and movement as native M5.0, actual energy ledger residual≤1e−6, exact original birth digest. Human program only selects existing native legal actions; cannot bypass energy or obstacle checks.

The M5.1 original local scripted movement score and tie-break RNG **must be numerically IDENTICAL** in SN and SS when both choose to move, and validated against `m5_1.scripted_choice(arm="greedy_novelty")` in cases when no birth can occur. No global food coordinates, map/omniscience or additional sensors in neural arms.

## 2. Frozen source genotype and 36 NEW worlds

**One fixed genotype**, exact prior M5.4B0-R externally trained recurrent 18-gene winner candidate #83 from documented original `reports/M5_4B0R_frozen_recurrent_genes_and_preregistered_statistics_2026-11-21.json`, `selected.recurrent.genes` and **original full 401-weight digest `c5e5795332725de4`**. Verify live digest against pinned literal. **DO NOT reselect, re-train, perturb, mutate, hand-edit, scramble or adapt the vector** based on M5.4C worlds. No native within-world genetic mutation in ANY arm.

- **24 independent unseen MAIN world seeds `91000–91023`**, original wall density0.12, original food renewal probability0.55, odor radius2 from initial tick, original physical world generator, 20 original food/20 genetically identical founders, 15×15 cell world, max64 population, 400 ticks, max age100; food energy18, action cost1, original birth threshold/age/legal/capacity, researcher scripted birth trigger energy46.
- **12 independent unseen SHIFTED world seeds `91100–91111`**, original wall density0.20, renewal probability0.40, odor radius2 unchanged, all other physical constants identical. NOT previous B0R development 89000–89003, validation89100–89107 or heldout89200–89223/89300–89311; NOT previous B0 world seeds88000/88100/88200/88300 or M5.4A 86000/87000.
- Four arms per seed, **36×4=144 FULL 400-tick world-policy evaluations**. Exact map, 20 founder positions, 20 food origins, seeded random food regeneration events, initial source genotype, actor action ordering and RNG substreams shared per world across policy arms. Biology diverges only after interventions. No world reused for parameter search; no development tuning set needed because old genotype and script are already completely fixed.
- Full independent fresh-world factorial and analysis MUST run ONCE after successful deterministic tests, not preinspect new sealed worlds.

## 3. Frozen per-action and lineage diagnostics

For all four arms, track without affecting policy:
- `actions_observed`, `movement_actions`, `birth_actions`, `birth_eligible_opportunities`, `births_by_neural_decision` vs `births_by_scripted_decision`, `mean_energy_when_birth`, `mean_energy_at_action`, `neural_birth_requested_while_eligible`, and `birth_opportunity_fraction` to separate reproductive timing and actual natality.
- `odorless_action_fraction` based ONLY on original local odor channels 4..7, `food_adjacent_opportunities`, `food_eaten`, `food_per_100_moves`, `odor_aligned_moves`/alignable movements; do not infer hidden intent from observations.
- `unique_cells_visited_per_agent_life`, `repeated_visits_per_move`, `distinct_cells_visited_by_any_agent`, `total_distinct_founder_lineages_with_live_descendants` at tick400, `max_generation`, `extinct_at`, `population at ticks 0/50/100/150/200/250/300/350/400`, true physically costed births, physical energy balance.
- **Per-agent diagnostic visit counts MUST NOT influence neural movement or neural birth choices**. They may influence ONLY explicitly scripted movement arms, which must be identified and labeled as such. Birth inherits **no ancestor visit map**: each new organism gets its own blank visit history with newborn position counted once, same M5.1 original behavior.
- Diagnostic point-of-move counters must handle agents dying at end of movement rather than silently losing that final visit; the implemented movement action and expected action must be audited against original move-actions delta. For NN arms, purely shadow forward logits are restored before original physical actor runs.
- Record exact genome and map digests for each world, all eight major genotype-related event and checkpoint properties; original M5.4B0R selected genome digest pinned above.

## 4. Frozen statistics and causal decision tree

Primary outcome is **survived tick400 (binary)**, not merely food, births, visual story or number of visited cells. MAIN and SHIFTED **separately** report all four arm survival counts and all diagnostics.

Within each world, define NN,NS,SN,SS survival in {0,1}:
- **Movement-script intervention marginal effect** `E_M = ((SN−NN)+(SS−NS))/2`.
- **Birth-script intervention marginal effect** `E_R = ((NS−NN)+(SS−SN))/2`.
- **Interaction** `I = SS−SN−NS+NN` (nonadditivity on survival probability scale).
- **Movement dominance contrast** `E_M−E_R = SN−NS` (balanced factorial difference).
- **Simple motor rescue** `SN−NN` (script movement only, neural birth retained).
- **Simple birth rescue** `NS−NN` (script reproduction only, neural movement retained).

For these seven preregistered world-paired survival contrasts plus **descriptive** food and birth factorial marginal effects, calculate **20,000 world-level paired bootstrap draws** per split (MAIN RNG `91301`, SHIFTED RNG `91302`, contrast offset 1,000 for each contrast in fixed order [EM,ER,I,EMminusER,SNminusNN,NSminusNN,SSminusNN], food E_M,E_R and births E_M,E_R after those) and report descriptive 95% percentiles; resample WORLD seeds, NOT individual ticks/births/agents, avoiding pseudoreplication. Claims of uncertainty apply only to this source-selected one fixed neural genotype on sampled worlds, not broad 18-gene numeric genotype distributions.

Primary causal movement-bottleneck judgment only if `E_M≥0.25` AND paired descriptive bootstrap 95% lower >0 AND `E_M−E_R` lower >0; reproduction-bottleneck judgment only if `E_R≥0.25` AND lower>0 AND `E_R−E_M` lower>0. Otherwise label `mixed / interaction / underpowered` from predeclared contrasts and full 4-arm patterns without inventing a sole cause. Additional preregistered **both-script-required pattern** if SS≥18/24 MAIN (positive ecological benchmark) and BOTH individual single-script-only arms SN and NS fail to clear 12/24 survival: report synergy/combined bottleneck hypothesis, but do not assert formal interaction unless `I` bootstrap excludes 0. Report whether scripted SS physical positive benchmark survives ≥18/24 MAIN; if it fails, revisit even physiology equivalence. OOD comparisons reported separately with no cross-environment pooling.

A strong movement rescue could still be a **researcher-coded exploration effect**, NOT emergence of exploration in the network. A strong birth rescue could be a **researcher-coded reproductive rule**, NOT learning of reproduction. NN remains externally sourced recurrent genotype, not naturally evolved de novo. This experiment **does not create genuine within-world mutation**, no source ecology is changed retrospectively. Save full 144 numerical trial rows, fixed original genotype provenance, per-action diagnostics, world-paired bootstrap JSON, population checkpoints and first-world event ledgers to permanent repo in addition to GitHub Action artifacts; keep negative results.

## 5. Next scientific action

Choose next representational architecture/sensor/reproduction experiment **only AFTER** reading this preregistered factorial (new protocol/new independent worlds). Do NOT jump to actual in-world mutation-vs-clone evolutionary adaptation without genuine independent numerical survival skill; never seed the researcher-optimized original M5.4B0R winner as a purported independent random founder in a real evolution experiment.

All organisms remain numerically bounded simulations, no arbitrary self-executing program, filesystem, network, tools or credentials.
