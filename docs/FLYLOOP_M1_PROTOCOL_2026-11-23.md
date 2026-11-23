# FlyLoop M1 — Frozen sensor, topology, relocation and synaptic-retention stress protocol

**23 November 2026 | prospective protocol, before the first full M1 GitHub Actions science run.**

## Why and scope

FlyLoop M0's same-14-weight online actor–critic outperformed a frozen
controller in a continuous 2D artificial ecology, but its built-in food
scent (including from behind walls) may have made foraging easy.
M0 **9/10** DEV screen passed; a preregistered post-barrier-crossing
novelty criterion failed. Do not reinterpret it as fly-like intelligence.

M1 does not add neurons, search primitives, a world model, planning
scripts, genetic mutation or a new learning algorithm. It keeps
M0's exact five atomic actions, 8-parameter shared actor, 6-parameter
critic, softmax sampling, TD(0) plasticity, researcher-programmed
intrinsic novelty bonus and physiological physical reward.

**Question:** Does plasticity improve survival/food collection when
the privileged directional food gradient is absent or is obstructed,
and can learned synaptic weights be retained across an actual food
relocation-and-restoration within the SAME lifetime?

**Honest limitations:** the eight candidate-action features continue
to include the *adjacent destination food yes/no* bit, and
hunger/injury, obstacle/hazard signals. The "no scent" condition
eliminates directional food gradient, **not all food perception**.
The path-distance pseudo-smell is a hand-designed proxy, NOT
physically correct air transport. Novelty is still researcher-programmed.
The closed barrier geometry, regrowing food, reward, actuator
repertoire and 14-weight linear policy all arise from the researcher.

## Four fixed, mutually distinct ecological challenges

All worlds are 13×13 with persistent 70 initial energy, 100
integrity, 16 food patches, 9 hazards, five primitive N/E/S/W/WAIT
actions, food regrowth after 17 ticks, and up to 480 ticks.
All worlds and weights are paired for all four controllers within
each scenario and seed. Every world seed is **7300–7331 (32
world seeds)**, never used in M0. No seed- or scenario-dependent
model selection and no post-result hyperparameter retuning.

1. \`no_scent_gate\`: M0 vertical barrier; at tick 240 the
   two open passages change from y=(2,10) to y=(5,8).
   The *two scent-related neural feature entries* are zero,
   for all options; adjacent food still visible.
2. \`occluded_noisy_gate\`: M0 vertical barrier and same
   gate switch; direction scent uses the **shortest accessible
   local path up to 4 movement steps** to available food,
   never crossing impassable cells, with a deterministic
   bounded ±0.08 additive sensor noise. No global coordinates
   or planned paths are passed to the policy.
3. \`novel_layout\`: same obstructed/noisy pseudo-smell,
   but a distinct predetermined topology: gate openings
   at y=(3,9) initially, then y=(4,11); two extra short
   interior wall spurs. The 16 food and 9 hazard patches
   are regenerated *only in free cells* with 8 food sites
   on each side. This means a geometry held out relative to
   M0, **not** a statistical independent family replication.
4. \`food_relocation\`: same occluded/noisy pseudo-smell,
   *no concurrent gate switch*; at tick 240 ALL 16 food
   positions move to previously empty, accessible, nonhazard
   cells (8 each side). At tick 360 those 16 original
   positions are **restored**. Both events are independent
   of the controller and recorded. Resets of food patch
   cooldown at each move are explicit. The model observes
   only currently sensed food, never the event schedule,
   original food coordinates, the replacement set or an
   external goal label.

## Four fixed controller ablations, all scenarios

1. \`frozen\`: fixed initialized actor and critic weights,
   stochastic movement (NOT an inactive dummy).
2. \`plastic\`: identical initialized weights and TD(0)
   learning using actual physical energy/integrity changes.
3. \`plastic_novelty\`: plastic plus M0's fixed, hand-coded
   novelty bonus on sensory patterns (NOT learned curiosity).
4. \`plastic_reset\`: same TD(0) learner, but actor and
   critic weights reset to their **identical initial weights
   at tick 240 in ALL scenarios, and again at tick 360
   in the relocation scenario**, without resetting
   physical state, position, world, or action sampling RNG.
   The reset ablation isolates weight persistence partially,
   but does NOT prove spatial memory, because no absolute
   positions are inputs to the controller. The reset arm
   continues learning after each reset.

All 4 controllers use 14 real-weight policy/critic topology,
exact M0 actor–critic learning rates, reward and temperature.
**16 conditions ×32 seeds = 512 distinct lifetimes**,
up to 245,760 sensor-action-environment transitions.
All evaluated for the entire scheduled horizon or until death;
no survivor-only choice of favorable outcomes.

## Precommitted outcome measurements

Each trajectory will record: legal action and position,
energy, integrity, food obtained, collision/hazard,
local/noisy scent, learned weight updates, real physical
reward, event schedule and full initial/final weights.
On relocation, log old/temporary/restored food sites;
track food in phases 0–239, 240–359 and 360–479,
including **zero for agents dead before a phase**.
The time to first food in each phase is censored and
fixed at \`phase_len+1\` if absent; no false success.
Record proportion surviving tick 240 and tick 360,
terminal survival, visited cells, collisions, damage,
returned-to-previously-harvested-site events and
new-gate crossings when relevant.

**Primary comparisons use exact paired world seed, same
scenario, same initialized 14 real-valued weights;**
bootstrap exploratory 95% intervals resample entire
world seeds, never timesteps. Return frequency is NOT
evidence of positional episodic memory: food is
regenerating, and the actor has no geographic map.

### 14 prospectively fixed DEV gates

1. Full **512 evaluation rows** and exact deterministic
   duplicated second complete run (byte-for-byte).
2. All primitive movements physically legal, no walls
   crossed, health/energy valid, no resurrected dead
   agents; log real physical outcomes.
3. Identical **initial 14 weights** and initial food,
   hazard and wall topology for each scenario×seed
   across all four controller arms.
4. \`no_scent_gate\` has exactly zero directional scent
   for all actions and all tick contexts; adjacent food
   remains a *separate* available sensor.
5. Obstructed pseudo-smell never propagates across a
   wall without a legal path of <=4 steps, with no
   added sensor noise beyond ±0.08.
6. Novel topology is distinct from baseline and its
   food/hazards always occupy free accessible cells.
7. Relocation and restoration happen **at exactly
   240 and 360**; the agent receives no event labels
   or restored food locations through its policy input.
8. Frozen weights never change; plastic agents update;
   reset arm resets old weights at prescribed events
   without moving the organism or resetting its RNG.
9. Every record and full raw trajectory is archived,
   including negative findings, exact seed pairing,
   final weights and disabled evolution flags.
10. No-scent plastic mean total food **>= frozen +15%**.
11. Occluded/noisy plastic mean steps alive **>= frozen +10%**.
12. Novel-layout plastic mean total food **>= frozen +15%**.
13. Food-relocation \`plastic\` mean food during the
    **restored phase** >= \`plastic_reset\` by **10%**.
14. Food-relocation \`plastic_novelty\` mean food
    **during relocated phase** >= plain \`plastic\`
    by **10%**.

ALL 14 must pass for narrow M1 screening success.
The previous M0 results are not M1 baselines or tuning
targets; only newly paired M1 arms are compared.
A successful GitHub Actions workflow tests code
correctness, **not** these scientific goals.

## Counterfactual interpretive limits

If plastic learning wins without scent, that demonstrates
adaptation under this artificial local sensor ontology, not
the spontaneous discovery of the concept "forage".
If the weight-preserving agent outperforms resetting,
that supports a causally useful retained synaptic parameter,
not explicit memory of the earlier food locations. If
not, publish the negative result unchanged.

Do not alter main or sealed M5.6.
No reproduction, evolution or ecological success release.
Always:
\`emergent_intelligence_demonstrated=false\`;
\`spatial_memory_verified=false\`;
\`independent_realworld_replication_complete=false\`;
\`evolution_authorized=false\`;
\`advance_to_evolution=false\`.
