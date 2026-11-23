# FlyLoop M1 — Verified outcome: robust embodied plasticity, not positional memory

**23 November 2026 | GitHub Actions [M1 science run #45271361094](https://github.com/kkongiu/neuroevolution-lab/actions/runs/45271361094) SUCCESS; prospective DEV screen 13/14 PASS (overall FAIL).**

This report documents results on the fixed M1 experimental design
recorded *before* full evaluation in
[FLYLOOP_M1_PROTOCOL_2026-11-23.md](FLYLOOP_M1_PROTOCOL_2026-11-23.md).
We have changed no preregistered threshold, neural optimizer,
action repertoire, seed range or experimental sample based on the
observed outcomes. A pre-acceptance **engineering correction** removed
the agent's current position from environmental resource relocation
sampling, so relocated maps are EXOGENOUS and identical across
controllers in the same world. An additional audit ensures that.
This correction did NOT adjust the hypotheses or scoring gates.

M1 measures artificial embodied adaptation in a researcher-designed
world. It does not replicate a fly brain or prove emergence of
abstract concepts, conscious doubts, a spatial map or new actions.

## Fixed M1 design and causal comparisons

Exactly **32 independent predefined world seeds 7300–7331**,
**4 ecological scenarios × 4 equally initialized controllers =
512 complete independent artificial lifetimes**. An organism has
up to 480 consecutive time steps in a 13×13 bounded grid with
16 regrowing food patches, 9 hazards and a real 70-energy/100-health
physiology. All arms use 5 atomic movements, the SAME
8-parameter shared linear actor +6-parameter critic, physical
energy/integrity feedback, learning rates and softmax sampling
as sealed FlyLoop M0. The research-coded *adjacent-food-bit*
sensor remains present **even with zero directional scent**.

World scenarios:
- **no_scent_gate:** all 5 candidate directional scent entries
  equal ZERO; food still visible one movement away.
- **occluded_noisy_gate:** directional scent from the closest
  *reachable* nearby food via wall-respecting path ≤4, plus
  deterministic bounded ±0.08 sensor noise. This is NOT real
  airborne odor simulation.
- **novel_layout:** different fixed old/new gate positions
  and additional immovable interior wall spurs, with valid
  food/hazard sites and the same occluded/noisy sensor.
  Different topology is not proof of truly general ecology.
- **food_relocation:** food sites are moved completely at
  step 240 and original sites restored at step 360, with
  no simultaneous gate change. The relocated map uses
  only world seed, *never agent location*; all four
  controllers on each seed have the exact same replacement
  food map, regardless of their different paths.

Four transparent controllers per scenario:
- **frozen:** same initial random softmax controller,
  no weight changes, but it still moves and eats.
- **plastic:** ordinary online TD(0) actor/critic.
- **plastic_novelty:** same TD update plus a
  RESEARCHER-SUPPLIED sensory-count novelty bonus.
- **plastic_reset:** same ordinary plastic controller,
  but its 14 weights are reset to their original
  initial weights immediately after exogenous change at
  step 240 and, in relocation, also step 360.
  The physical body and action PRNG are NOT reset.

No explicit forage, avoid wall, escape hazard, switch plan,
store X/Y or spatial map procedure exists in the actor.
However the food representation, physiology, hunger and
reward signals, locomotion primitives and plasticity
are all engineered by the experimenter.

## Results — absolute mean per lifetime (N=32 independent worlds per arm)

| World | Fixed food | Plastic food | Novelty food | Weight-reset food | Fixed mean survival steps | Plastic mean survival steps |
|---|---:|---:|---:|---:|---:|---:|
| No directional smell | 20.41 | **49.34** | 49.03 | 40.50 | 358.50 | **480.00** |
| Reachable noisy scent | 18.38 | **65.28** | 66.22 | 50.78 | 331.25 | **472.50** |
| New barrier geometry | 18.63 | **63.38** | 64.50 | 50.72 | 333.97 | **480.00** |
| Move food then restore | 20.25 | **66.34** | 66.75 | 49.09 | 340.50 | **480.00** |

*Scientific caution: longer-lived agents have more food
opportunities.* The food advantage remains after normalizing
by actual time alive:

| World | Frozen food / 100 living ticks | Plastic food / 100 living ticks |
|---|---:|---:|
| No scent | 5.69 | **10.28** |
| Occluded/noisy scent | 5.55 | **13.82** |
| New geometry | 5.58 | **13.20** |
| Food relocation/return | 5.95 | **13.82** |

In the no-scent condition, keeping the plain TD learner
instead of frozen policy gains **+28.94 foods per lifetime**,
exploratory paired world-seed bootstrap 95% interval
**[+26.03,+31.66]**, and **+121.5 living steps**
[+82.91,+160.72]. It survives 480/480 steps on **all
32 worlds**, while frozen agents survive 480/480 on **10/32
worlds**. Removing directional smell does NOT remove the
researcher-coded near-food sensor or physical TD reward.

In occluded/noisy world, plain plastic yields 65.28 vs18.38
food (**+46.91**, paired CI [42.0,51.84]), and lives
472.50 vs331.25 steps. The novelty arm gains little in
total food relative to plain plastic and no claim of
autonomously learned curiosity can be drawn.

On shifted geometry, plain plastic yields **63.38 foods
vs18.63** (mean paired difference **+44.75**, exploratory
world-seed CI **[+40.63,+48.66]**), survives all 32
lifetimes and visits about 60.75 unique cells versus 45.25
in the frozen policy.

### Relocation and weight preservation

In the 120-step period immediately AFTER the original 16
food sites are restored at tick 360:

- **plastic with original retained weights:** 19.47 food
  per lifetime; 3.59 ticks average until first food.
- **plastic_reset with weights zeroed back to original
  initialized values at ticks 240 AND 360:** 9.94 food;
  15.41 ticks average until first food.
- Paired **+9.53 food** for retaining learned weights,
  exploratory 95% seed bootstrap **[+8.19,+11.06]**.
  This is a robust within-world association/causal
  ablation for general adaptive weights; **NOT explicit
  memory of a food location**: the sensor has no global
  coordinates and the reset arm has the same world and
  intact physical body.

The novelty bonus fails its predefined challenge
DURING the *relocated* 120-step phase:
- Plastic food: **18.15625**.
- Plastic + engineered novelty: **17.875**.
- Difference **−0.28125** food per lifetime;
  exploratory paired 95% CI **[−1.66,+1.41]**.
- Protocol requires novelty >=1.10×plastic
  (**>=19.97**); **FAIL**.

### Important exploratory diagnostic against trivial stationary food camping

We independently inspected full action traces to measure
whether food was collected simply by selecting WAIT on
a regrowing food patch:

- In **no_scent_gate**, 98.0% of food collected by
  plastic agent followed a **movement** action, and
  only 2.0% followed WAIT; an average of **10.59
  distinct food sites** yielded food.
- In **occluded_noisy_gate**, 98.7% of foods were
  collected after movement; average **12.66
  distinct sites** per organism.
- In **novel_layout**, 98.8% moving-food;
  average **10.97 distinct food sites**.
- In **food_relocation**, 99.2% moving-food;
  average **19.97 distinct food sites**
  across the two separate site configurations.

This rules out simple WAIT-on-a-patch camping as the
main explanation of high food scores. It still does NOT
prove deliberate navigation, episodic memory or planning.

## 14 frozen prospective gates — 13 PASS / 1 FAIL

1. Exactly 512 complete paired raw rows, byte-identical
   full second run — **PASS**.
2. Legal movements and wall states, no teleportation,
   physically valid energy/integrity — **PASS**.
3. Identical starting world +14 weights within every
   same-seed four-controller block — **PASS**.
4. Exactly no directional food gradient in no-scent
   while adjacent-food sensor is distinct — **PASS**.
5. Obstructed path pseudo-smell never cheats through
   an unreachable wall, ±0.08 noise cap — **PASS**.
6. Novel layout genuinely distinct and all patches
   physically in valid free cells — **PASS**.
7. Relocation at 240, restoration at 360, environment
   exogenous and identically replaced across controllers
   — **PASS**.
8. Frozen weights unchanged, plastic updates and
   reset ablation at all eligible events — **PASS**.
9. All raw traces + negative effects + disabled
   evolution flags archived — **PASS**.
10. No-scent plastic food ≥frozen +15% — **PASS**.
11. Noisy-scent plastic lifetime ≥frozen +10% — **PASS**.
12. Novel layout plastic food ≥frozen +15% — **PASS**.
13. Relocation restored-phase plastic food ≥
    weight-reset +10% — **PASS**.
14. Relocated-phase novelty food ≥plain plastic +10%
    — **FAIL**.

**Overall preregistered M1 DEV screen = FAIL despite
13/14 gates.** Technical reproducibility and software
CI success are separate from prespecified scientific
performance success.

## Independent archival audit and code provenance

Official completed green CI:
[Actions run #45271361094](https://github.com/kkongiu/neuroevolution-lab/actions/runs/45271361094).

- Artifact: \`flyloop-m1-sensory-memory\`, ID
  **14865217385**, 30-day retention until
  **23 December 2026**.
- Downloaded ZIP extracted/inspected separately:
  **two JSON files, each exactly 170,972,775 bytes,
  byte-identical.**
- Exactly **512 distinct paired lifetime rows**, balanced
  128 per scenario /128 per controller, 32 per each
  scenario-controller combination.
- **227,495 actual organism action transitions plus
  590 logged environmental change events**, including
  physical outcome, pre-action scent/adjacent food,
  sensor reward, all moves and the controllers'
  exact original/final weights and reset checkpoints.
- Official compact raw-row SHA-256:
  \`2d83febf2172e609dd38785740bd595bcc1cad835144c27556d9998f55c2623b\`,
  **independently recomputed and exactly matched**.
- Exogenous relocation test: within every world seed,
  all surviving controllers receive the same exact 16
  replacement food sites regardless of their location.

## Disposition and next discriminating science

**Supported within this synthetic ecology:** retaining
learned weights increases feeding even when the
directional food scent is disabled; changing geometry
does not erase this practical advantage; retained
plastic parameters help rapid exploitation after a
resource relocation/return. The differences are not
solely explained by longer life or stationary WAIT
harvesting.

**NOT yet supported:** positional map memory, independent
real-world/general ecology transfer, spontaneously
invented curiosity, ability to create new motor primitives,
biological fly circuitry or evolutionary selection.
The novelty bonus does not pass the relocation criterion.

**High-value next step, not performed:** ablate the
adjacent food-bit and hazard-lookahead feature as well
as scent, test sensor deprivation and delayed/noisy
physical reward, and score behavior in independent
geometries generated by a disjoint world generator.
Do not add new neurons before isolating which
researcher-supplied cues are doing the work.

All stop flags remain FALSE:
\`emergent_intelligence_demonstrated=false\`,
\`spatial_memory_verified=false\`,
\`independent_realworld_replication_complete=false\`,
\`evolution_authorized=false\`,
\`advance_to_evolution=false\`.

M1 remains a DRAFT PR stacked on M0 PR #86, no merge
into main and original sealed M5.6 untouched.
