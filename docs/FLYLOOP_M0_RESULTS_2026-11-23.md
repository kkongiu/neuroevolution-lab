# FlyLoop M0 — Verified embodied-loop science results

**2026-11-23 | [successful GitHub Actions run #45268354201](https://github.com/kkongiu/neuroevolution-lab/actions/runs/45268354201)**
Scientific verdict: **FAIL overall (6/10 prespecified gates; only 1/5 scientific outcomes passes)**.
Software/integrity verdict: **PASS (5/5 causal-integrity gates, new unit tests and original M5.6 regressions)**.
This record is a genuine negative result, *not* a digital fly or a
demonstration of autonomous spatial reasoning.

The prospective design and success thresholds were published first in
[FLYLOOP_M0_PROTOCOL_2026-11-23.md](FLYLOOP_M0_PROTOCOL_2026-11-23.md).
After the frozen 16-seed results became available,
no sensory design, reward, model hyperparameter, threshold, heldout
world, seed, or predeclared comparison was retuned.

## What actually ran

A new **11×11 grid physics loop** with two food cells,
walls forming a full-height divider with one gate,
a harmful tile and finite persistent energy within each bout:

\`\`\`text
#############  (illustrative, not drawn to scale)
#    #    #
#    #  F #
#    #    #
#  F #    #
# A  #    #
#    #    #
#    #    #
#          #  <- gate at y=8 in one shifted test world
#    #    #
#############
\`\`\`

The implementation itself exactly uses x=5 for the divider,
(2,5) for the source-side spawn, (3,3) for a near food
resource, (8,5) for a far food resource and (4,6) for the
static hazard. Each grid cell is represented by exact
integer coordinates; no shortest-path solution is encoded.

Each step executes the full embodied feedback path:

1. Compute **19 causal local/current sensory channels**:
   immediate wall occupancy, adjacent hazard, pre-engineered
   food-smell field sampled one step away in 4 directions,
   current energy, previous collision/feedback/motion, normalized
   x/y coordinates and bias.
2. Call the **original M5.6 260-weight recurrent
   \`Learner.action\`**, choosing one of N/E/S/W cardinal
   move attempts, even when that direction is blocked.
3. Apply REAL physics **before** reward:
   movement, collisions, energy consumption,
   hazard damage, actual food consumption and replenishment.
4. Reuse **original M5.6 \`Learner.learn\`**, based on
   actual physical feedback with an optional and
   separately accounted M5.6 \`IntrinsicReward.observe_arrival\`.
5. Continue with persistent learned synaptic weights.
   Recurrent hidden state resets between the four
   controlled bouts; body energy and food are standardized
   at the start of each bout. No natural reproduction,
   population selection, or lifelong continuous body
   operation is claimed.

Arms: \`frozen\` (no plasticity),
\`plastic\` (actual food, damage and movement feedback)
and \`plastic_novelty\` (identical learner with bounded
personal novelty bonus only on genuine arrivals), plus
\`plastic_novelty_reset\` weight-erasure ablation in shifted bouts.
Every seed's three starting RNN weight vectors are
**byte-identical**. Each arm gets identical world
rules, physical costs, action RNG seed schedule
and observation schema. Only updates and the separately
accounted internal novelty bonus differ.

Two source layouts had the divider gap at y=2/3.
Two shifted layouts moved that gap to y=7/8;
the **first shifted bout froze ALL learning** and
is the PRIMARY transfer test. The later shifted bout
allows renewed learning and is secondary.
Sixteen paired seeds 44000..44015 per arm, each with
at most 150 physical action steps/bout.

### Primary first-shifted-world results: 16 seeds

| Arm | Far-side food actually eaten, mean | Far-side crossing fraction | Mean collision count | Mean hazard events | Mean physical survival ticks | Mean final physical energy |
|---|---:|---:|---:|---:|---:|---:|
| Frozen RNN | **0.0000** | 18.75% | 41.19 | 2.56 | 141.88 | 36.92 |
| Plastic RNN | **0.1875** | 25.00% | 28.63 | 2.19 | **150.00** | 41.10 |
| Plastic + personal novelty | **0.0625** | 12.50% | **25.50** | 2.38 | 149.06 | **42.21** |
| Plastic + novelty, learned weights ERASED just before shifted layout | **0.0000** | 18.75% | 41.19 | 2.56 | 141.88 | 36.92 |

The untrained/erased weights ablation reproduces the frozen controller's
physical behavior on the shifted evaluation, while its **personal
novelty bonus was still computed separately** (not used for action
until learning resumes). This makes the small behavioral
difference attributable to learned weights, not to
privileged map information or a new handcrafted routing policy.

### Paired effects and uncertainty — do not overinterpret N=16

Primary \`plastic - frozen\` far-side food = **+0.1875** meals
per seed, an exploratory paired-seed bootstrap
95% **[0,+0.50]**; the lower bound includes zero.

\`plastic_novelty - frozen\` far-side food =
**+0.0625** meals [0,+0.1875].

\`plastic_novelty - plastic\` =
**−0.1250** meals [−0.375,0].
The extra researcher-programmed novelty incentive
**did not improve** far-side eating, despite fewer collisions.

\`plastic_novelty - frozen\` survival =
**+7.1875 ticks** [−1.25,+19.6875], below
preregistered +10 threshold and including zero.

\`plastic_novelty - erased-weights\` =
**+0.0625** far-side meals [0,+0.1875].
A modest sign of learned behavior, but NOT convincing
independent evidence of a successful adaptive route.

An additional warning: all 16 seeds reuse **the same
TWO heldout physical topologies** (gap y=7 and y=8).
Paired seed intervals measure controller initialization/
stochastic action variability on these layouts, **not**
uncertainty across independently designed natural worlds.

### Secondary: source training and later shifted adaptation

| Phase | Frozen far food | Plastic far food | Plastic+novelty far food |
|---|---:|---:|---:|
| Source bout A, gap y=2 | 0.2500 | 0.2500 | **0.3125** |
| Source bout B, gap y=3 | 0.1875 | 0.2500 | **0.5000** |
| Shifted first, gap y=7, **no learning allowed** | 0.0000 | **0.1875** | 0.0625 |
| Shifted second, gap y=8, **learning allowed** | **0.3125** | 0.0625 | 0.0625 |

The positive source-bout trend in the novelty arm
does NOT generalize to the new gap location.
In the second shifted world the original frozen
learner unexpectedly eats more far-side food than
either plastic arm; this counterexample must be
preserved, not hidden by reporting only source worlds.

### Prespecified gate-by-gate outcome

1. Science: curiosity far-side food >=0.30: **FAIL** (0.0625).
2. Science: curiosity−frozen far-side food >=+0.20: **FAIL** (+0.0625).
3. Science: plastic−frozen far-side food >=+0.10: **PASS** (+0.1875);
   paired CI still includes zero.
4. Science: curiosity−plastic far-side food >=+0.10: **FAIL** (−0.1250).
5. Science: curiosity−frozen survival >=+10 ticks: **FAIL** (+7.1875).
6. Integrity: 16 science seeds, exact three original arm
   starts, four bouts each and shared identical 260
   weights per seed: **PASS**.
7. Integrity: 19 sensory channels, real physical
   transition recomputation, no target/path label: **PASS**.
8. Integrity: intrinsic reward never physical energy,
   no artificial reward on collision: **PASS**.
9. Integrity: NO neural updates in first shifted evaluation,
   matched 16-seed erased-weights ablation: **PASS**.
10. Integrity: complete per-action logs, source M5.6
    regression and prospective evolution-stop flags: **PASS**.

Overall **6/10 checks**, including only **1/5 outcome checks**.
CI completion is not scientific success.

## Audit and exact reproduction

GitHub Actions official success:
[FlyLoop M0 run #45268354201](https://github.com/kkongiu/neuroevolution-lab/actions/runs/45268354201).
New unit tests and original \`tests/test_m5_6_online.py\`
both PASSED. Full raw artifact:
\`flyloop-m0-embodied-dev-results\`,
ID **14863218224**,
30-day retention until **2026-12-23**.

Independently downloaded, extracted, parsed and rehashed artifact:
- **16 science seeds, 224 total physical bouts**,
  32,754 *actual attempted action* records;
- full sensory readings *before* each motor action,
  action chosen, environment transition, energy before
  and after, actual food, collision, harm, reward
  and separate novelty bonus;
- **0** intrinsic bonuses assigned to blocked movement;
- frozen first shifted plasticity **0 updates** for
  all arms including ablation;
- SHA256 canonical sorted raw bout rows:
  \`884e522e4ebe1d30834b77899fa7160d952be487c136dc72ad6f035b2c5f7ff3\`
  independently recomputed EXACTLY matching run output.

No original M5.6 file changed in PR #85: PR diff adds
**only** \`neuroevolution/flyloop_m0.py\`,
\`tests/test_flyloop_m0.py\`,
\`.github/workflows/flyloop-m0.yml\` and
this new protocol/report/README documentation.

The RNN weights are only hashed at bout boundaries
and recorded as real individual synaptic update counts,
**not** every full 260-weight checkpoint in raw output.
All initial weights are exactly reproducible from
\`m4_4a._initial_genome\` with fixed code and seed.
No pretrained new family M5.34–M5.43 neural component
is invoked in M0. They remain possible modular
later-stage replacements with genuinely matched
ablation, not unearned claims that all were integrated.

## What this does NOT establish

- no learning of an innate fly escape reflex;
- no stable bypass route demonstrated on shifted obstacle;
- no real-world animal sensorimotor transfer;
- no automatic inventing of new motor primitives;
- no robust learned curiosity policy (novelty is scripted);
- no guarantee that any particular spatial
  exploration strategy was not mostly driven by
  *researcher-specified odor gradient*;
- no independent topological replication: just
  two heldout divider gaps and one hazard tile;
- no persistent biological life through bouts:
  standardized energy reset between them;
- no reproduction, natural selection or evolution.

All stop flags are \`false\`, including
\`learned_new_behavior_established\`,
\`learned_general_curiosity_policy\`,
\`autonomous_high_level_planning\`,
\`independent_ecological_replication\`,
\`evolution_authorized\`, \`advance_to_evolution\`.
Original M5.6 sealed ecology and \`main\` untouched.

## Next falsifiable problem (not implemented)

The M0 data show strong collision reduction with
synaptic learning but little far-side food transfer.
Before adding a planning module, diagnose *why*:
does the RNN learn local wall avoidance and therefore
stop exploring the narrow detour? Are food odor
and spatial representation adequate? Add a strict
new-environment challenge that **does not alter
the genome, architecture or reward schedule**
and manipulates only the obstacle geometry, with
a frozen-policy evaluation; compare wall-contact
rates, number of useful detours and actual return
to a once-visited food source. Keep diverse
topologies and one heldout family independent.

FlyLoop M0 is a physical feedback FOUNDATION,
**not** a successful demonstration of fly-like intelligence.
