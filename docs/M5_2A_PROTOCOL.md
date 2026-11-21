# M5.2A — Preregistered factorial diagnosis of neural ecological failure

**Frozen before implementation and before full final-seed evaluation.** Independent stage A of [issue #31](https://github.com/kkongiu/neuroevolution-lab/issues/31). This stage diagnoses mechanisms; it does **not** evolve new controller architectures, claim an evolution success, or alter historic M5.0/M5.1 analyses.

## Hypothesis and experimental unit

M5.1 proved programmed odor-directed animals can persist while the original heritable RNN fails. But programmed policies change *both* motor search and the reproductive decision. A 2×2 factorial intervention disentangles these two changes. **One world seed** (not animals, births or ticks) is an independent experimental unit. All four arms start from the **same** 20 random founder neural genomes and world positions, same deterministic candidate resource schedule and the unchanged M5.0 dynamics. Divergent realized food placement is an expected consequence of different agent actions.

## Interventions (fixed, not fitted to evaluation seeds)

- `NN` = **neural movement + neural reproduction**. Original M5.0 policy: evaluate its 5-output 12-hidden-unit RNN, select maximum legal logit among 4 motor directions and optional REPRODUCE. This is the unchanged M5.0 baseline.
- `NT` = **neural movement + threshold reproduction**. Advance the *same* 5-output RNN each turn, but replace REPRODUCE choice with an intervention: if physically eligible, age≥4 and energy≥46, force REPRODUCE; otherwise select the highest-logit legal *motor* action. A neural birth preference never bypasses the threshold.
- `GN` = **greedy-odor movement + neural reproduction**. Advance the same RNN each turn. Determine whether it prefers REPRODUCE using the original maximum-over-legal-five-logits procedure; if so, reproduce. Otherwise choose movement with M5.1's local, current-sensor-only `greedy_no_reproduction` behavior: adjacent food first, else local strongest odor, else uniformly random legal motor. No private visit history.
- `GT` = **greedy-odor movement + threshold reproduction**. Exact programmed `greedy_odor` behavior from M5.1, retained as the positive feasibility reference, and genotype has no causal effect on motor or reproductive choice.

For neural-reproduction arms, eligibility, masking, logits and tie-breaking are exactly as in M5.0. For threshold-reproduction arms the condition uses current energy/age, which are **privileged researcher-supplied controls**; they are *not* suddenly learned by the RNN. Both types of movement use exactly the same physical legal-neighbor mask. The RNN always advances its hidden state on NN/NT/GN. All four arms receive identical physical birth costs and genuine inherited random-mutation offspring genomes. Only NN/NT/GN neural decisions are genotype-dependent; GT's numeric genome records do not generate behavior. No numerical fitness scoring or external elite replacement occurs in any arm.

## Frozen world physics and evaluation

Copy the unchanged M5.0 parameters: 15×15, 20 founder agents, max 64, 20 initial food items, max 32; two predetermined food candidates/tick with acceptance probability .55 main / .40 shifted, odor radius 5 main / 2 shifted, wall fraction .12 main / .20 shifted; energy 36 initially, 54 max, +18 for food, −1 per living organism's tick; age limit 100; physical birth eligibility age≥4 and energy≥34, parental birth cost 20, newborn gets 12, population cap and adjacent unoccupied cell mandatory; 12 tanh RNN units, 401 initially random weights with SD .45, child Gaussian mutation SD .12 applied per weight. Freeze tick horizon 400 and action ordering. **No weights are modified during an organism's life.**

- Smoke / engineering seeds 700–701 (never used for scientific result). May use 11×11, 40 ticks and hidden=4 to exercise interfaces.
- **Sealed MAIN**: 20 paired worlds, seeds **75000–75019**, each 400 ticks, four arms.
- **Sealed shifted OOD**: 10 paired worlds, seeds **76000–76009**, each 400 ticks, four arms.
- Never adjust these seeds, thresholds, energy, physics or policy choices in response to observed results. Previous M5.0/M5.1 data remain intact.

## Predeclared outcomes and interpretation

**Primary diagnosis:** four paired main-world contrasts on end-of-horizon **world survival** (0/1), with mean paired differences, 95% **descriptive** paired bootstrap intervals, counts positive/negative/tied:

1. NT−NN isolates the reproduction-threshold intervention *given neural movement*.
2. GN−NN isolates the greedy motor intervention *given neural reproduction*.
3. GT−NT isolates the greedy motor intervention *given threshold reproduction*.
4. GT−GN isolates the reproduction-threshold intervention *given greedy movement*.

**Necessary context:** report GT−NN combined contrast and difference-in-differences (GT−GN)−(NT−NN), explicitly dependent on the chosen endpoint scale; do not automatically infer independent additive mechanisms. A mechanistic difference is considered **supported** only where a positive bootstrap lower bound accompanies a favorable paired sign majority (excluding ties), and an absolute improvement is visible. Otherwise report inconclusive or negative.

**Secondary (exploratory, not replacement gates):** per-world final population, births, highest generation, food eaten, food/100 motor actions, organism deaths, remaining stock at 100 and 400, declared neural birth preferences versus eligible opportunities, time of extinction. Plot population checkpoints at 0/50/100/150/200/250/300/350/400. Report all shifted arms with no reselection or threshold changes. Compare paired worlds as paired units, never pseudo-replicate agent observations.

## Engineering and causal audit

- **Exact outcome identity:** NN matches the unchanged M5.0 simulator under the same seeds/settings; GT matches M5.1 greedy_odor.
- Separate RNN-pass shadow tests verify under NT and GN that hidden state evolves, while genetically identical initial populations remain matched.
- No illegal wall/agent movement, overlapping individuals or impossible births. Verify newborn IDs/lineages and exact energy ledger closure on all arms/checkpoints.
- Non-functional sensor perturbation and birth-threshold toggles in isolated test states verify interventions actually change the specified decisions.
- Full seed-level CSV, checkpoints CSV, summary JSON, and representative birth/food event traces archived via a **read-only** GitHub Actions job after regression CI passes; save small permanent per-world table to repo for future inspection, even if outcome is null.
- If CI fails, fix reproducibility/engineering only; report errors and reruns. No retuning the experimental design or looking at sealed scores while changing it.

**Limits:** GT/GN use a hand-written motor choice; NT/GT use hand-written reproductive timing. Such improvements cannot prove emergent neural competence, inherited genetic advantage, open-ended evolution, self-directed goals or consciousness. The follow-up M5.2B smaller-genome experiment must be preregistered separately, with fresh seeds and genuine genotype-controlled reproduction and movement.
