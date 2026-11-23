# FlyLoop M0 — PROSPECTIVE protocol: continuously embodied local reinforcement

**Frozen 2026-11-23 prior to the first full FlyLoop-M0 scientific CI run.**
Stacked **draft** experiment branch based on M5.35, without modifications
to original sealed M5.6, its tests, or \`main\`.

## Motivating observation and deliberately narrow claim

A fly can display economical motion, react to danger, search for
resources, and revisit promising places. This is *not* a proof of
general intelligence or that a few generic laws alone generate a
fly's evolved capabilities.

The first test is not "create a digital fly." It asks:

**Can an existing M5.6 online recurrent learner, without a programmer
supplying a find-food, flee, pathfind or bypass-obstacle policy, gain
measurable embodied food collection and survival from genuine
sensor-action-consequence feedback, relative to the EXACT same
unmodified frozen initial network?**

The relevant M5.6 public primitives are directly reused:
\`neuroevolution.m5_6_online.Learner\` (260-weight Elman RNN,
19 sensory channels, four motor actions, bounded single-step
REINFORCE, recurrent state) and
\`neuroevolution.m5_6_curiosity.IntrinsicReward\` (bounded
per-individual location novelty). Initialization uses the
unchanged \`m4_4a._initial_genome\`, not a newly pretrained brain.
These algorithms were already *programmed by researchers*.
No neural operator invention, general intelligence, or original
M5.6 ecological replication can be claimed.

## Physical micro-world and sensory/motor interface

A strict deterministic **11×11 bounded integer grid**, with
border walls and an interior vertical divider at x=5,
containing exactly one open gate at y=2/3 (source layout)
or at y=7/8 (unseen shifted layout). The actor is at (2,5);
the primary food at (8,5), with a nearer recurring
food source on its starting side at (3,3).
An impassable wall cannot be passed using movement.
An explicit hazard tile adjacent to the divider
subtracts finite energy if entered; this is a researcher-supplied
physical danger, not a learned predator.
Food is replenished on a finite fixed deterministic schedule.

Physical energy:
- initial 80, capped at 95;
- every attempted move costs 0.24 energy;
- attempted collision with wall costs an additional 0.45 energy;
- entering hazard costs 7 energy (logged physical harm);
- food restores 24 energy and counts as one actually obtained meal.
When energy reaches zero, that bout ends.
No automatic "flee", "seek food", "go through gap" or
"navigate to target" action rules exist in any controller.

The 19-channel observation exposes ONLY: four immediate cardinal
wall/border readings; four adjacent hazard readings; four
directional food-smell intensities (defined from the
environment, not a given optimal action); energy fraction;
last attempted wall collision; previous externally observed
energy/reward sign; two coordinates normalized to bounded
range; and constant bias. No target labels, distant exact
wall-map, exact gap coordinate or shortest path are supplied.
The smell field encodes coarse distance to food, so
the agent does receive a strong engineered environmental
guide; report this explicit inductive bias.

Four always-available motor commands: attempt to move one
cell north/east/south/west. Wall collisions are permitted
as attempted commands with negative consequence
rather than removing them from legal action choices.
All three arms have exactly the same action space,
physical world, sensors, metabolism, safe constraints,
initial synapses and random action seed schedule.

## Three preregistered arms and matched causal controls

All share *exactly the same* 260-weight M5.6 RNN
initialization for each seed:
1. \`frozen\`: RNN action selection but no weight update.
2. \`plastic\`: M5.6 single-step REINFORCE on **physical
   feedback alone**, enabled; food +1.5, hazard -1.0,
   collision -0.20, otherwise -0.02. These scalar
   training feedback units are not the physical
   energy units. The consequences must actually
   occur before feedback is applied.
3. \`plastic_novelty\`: same real physical feedback
   and exact neural update, plus M5.6 personal
   location-novelty bonus, capped at +0.10 on an
   actual successful entrance into a cell.
   Novelty is a preprogrammed incentive, explicitly
   NOT a self-discovered curiosity system.

All actual world-action rewards, energy, food,
damage and collision metrics MUST ignore intrinsic
novelty. This prevents double-counting the
researcher-injected exploration incentive as survival.
No selective or cross-arm changes in learning-rate,
initial parameters, action space, energy or world geometry.

For a separate ablation \`plastic_novelty_reset\`,
the same \`plastic_novelty\` arm's parameters are erased
**at the start of the heldout layout**, restoring the
original pre-training 260 synapses while retaining
the same agent's accumulated visit history. It
measures dependence on learned weights, but is an
intervention, NOT an independent fourth original
training strategy.

## Design and budget

Preregister 16 *heldout science seeds*, \`44000..44015\`,
with paired frozen/plastic/plastic_novelty variants;
train only on **two consecutive SOURCE bouts**, each
of 150 maximum moves, divider gate y=2 and y=3,
and evaluation on **two different SHIFTED test bouts**,
each of 150 maximum moves, gate y=7 and y=8.
Every new bout resets physical energy, spawn and
immediate physical world (an explicit standardized
arena, NOT proof of a single biological life).
The SAME RNN synaptic weights persist across
bouts for plastic arms, while M5.6 recurrent
hidden state is reset at each bout; per-agent
exploration visit counts persist across bouts.
Frozen runs choose actions using the same original
260 weights throughout.

**There is no evolution and no heritability, no
reproduction, no selected genomes or population.**
These are four repeated training/testing episodes
with a persistent individual controller, not four
independent lifelong digital organisms.
As a direct measure of transfer, no weights
are updated during the first shifted evaluation
bout (evaluation-only freeze); updating may resume
in the second shifted bout, reported separately.

No hidden-outcome policy training or architecture
tuning on seeds \`44000..44015\`.
Exact alternate development seeds \`43000..43003\`
may be used ONLY for code smoke tests, not for
selecting the reported full-16 scientific outcome.

## Raw evidence and metrics

Record per seed, arm, bout: actual attempted
moves, successful movements, wall collision
count, distinct cells visited, first crossing
to far side (x>=6), far-side food actually
eaten, any food eaten, actual hazard entries,
physical survival ticks, physical final
energy, real-scalar training reward total,
bounded intrinsic bonus, M5.6 update count,
RNN weights-before/after SHA256, and
all causal sensor/action/physical reward
step transitions in raw audit.

Primary:
- **shifted-world far-food acquisition** (food
  physically eaten on far side) mean over
  the first shifted evaluation bout;
- **shifted-world physical survival ticks**;
- **shifted-world collisions and hazard damage**.

Secondary source-world acquisition, crossing,
novelty location count, retained vs reset
synaptic performance, and source-to-shifted
acquisition ratio. Use paired seed-level
bootstrap confidence intervals, *not* every
action as an independent organism. Any absolute
survival gains may be due to chosen world design
and food smell; all statements remain within
synthetic environment.

## Prespecified ten screening checks, all mandatory

Five scientific outcome gates (may legitimately FAIL):
1. \`plastic_novelty\` far-side food in the **first
   shifted frozen-learning bout** averages >=0.30
   per seed.
2. \`plastic_novelty\` minus \`frozen\` shifted
   far-side food >=+0.20 food per seed.
3. \`plastic\` minus \`frozen\` first shifted
   far-side food >=+0.10 per seed.
4. \`plastic_novelty\` minus \`plastic\` shifted
   far-side food >=+0.10 per seed, not just
   more spatial coverage.
5. \`plastic_novelty\` minus \`frozen\` physical
   survival on first shifted bout >=+10
   ticks (not learning reward).

Five integrity gates:
6. Four bouts ×16 science seeds ×3 original
   arms present, exact identical source
   260 synapses across three arms each seed.
7. Same wall/food/hazard definitions and
   source/shifted placements for arms, eight
   verified genuine physical transition
   invariants, no target map, learned path
   or correct-action label in any observation.
8. Actual physical reward, survival, eating and
   damage computed without intrinsic novelty;
   no unearned novelty on blocked moves.
9. Shifted first-bout learning is disabled,
   and matched synaptic reset ablation
   is separately present for 16 seeds.
10. Full raw time-step transitions,
    aggregate scores, bootstrap intervals,
    code+checkpoint hashes, negative results,
    old M5.6 regression and evolution-stopped
    flags archived in GitHub Actions artifact.

A run can have all CI tests pass and still FAIL
the scientific five gates. Report that honestly.
No post-hoc retuning after inspecting full
science outcomes; changes require new protocol
and genuinely unexamined science seeds.

\`source_M5_6_unmodified=true\`
\`learned_new_behavior_established=false\` until
independently substantiated;
\`learned_general_curiosity_policy=false\`,
\`autonomous_high_level_planning=false\`,
\`independent_ecological_replication=false\`,
\`evolution_authorized=false\`,
\`advance_to_evolution=false\`.
