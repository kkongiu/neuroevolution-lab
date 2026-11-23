# FlyLoop M0 — minimal embodied loop

**Experimental only · new draft PR #85 · no changes to main or sealed M5.6.**

FlyLoop is a deliberately small reorganization of the
existing neuroevolution work around a persistent
**sensation→action→physical consequence→learning** loop.
It DOES NOT imitate a fly's complete neural
anatomy and does not yet produce effective
adaptive bypass behavior on unseen obstacles.

## New files

- \`neuroevolution/flyloop_m0.py\` — 11×11 physical world,
  real energy/damage/food, causal 19-channel sensors and
  original 260-synapse M5.6 recurrent controller.
- \`tests/test_flyloop_m0.py\` — physics and weight-ablation
  tests, no future observation labels, actual step audit.
- \`.github/workflows/flyloop-m0.yml\` — both new
  and sealed original M5.6 regressions, full 16-seed
  science run and raw artifact.
- [Pre-frozen protocol](FLYLOOP_M0_PROTOCOL_2026-11-23.md)
- [Verified outcomes and failures](FLYLOOP_M0_RESULTS_2026-11-23.md)

## Conceptual components

\`\`\`text
World (odor, walls, hazard, food, physics)
    |
    v  [19 local sensory readings, current energy, last feedback]
UNMODIFIED M5.6 Learner.action (260-weight RNN)
    |
    v  [N,E,S,W attempt; NOT an optimal-route action]
World.transition (physical movement, collision, food, damage)
    |
    +--> Body energy / food / real survival telemetry
    |
    +--> M5.6 Learner.learn (only actual scalar feedback)
              ^
              | optional capped individual IntrinsicReward
              | NOT included in energy or real food
\`\`\`

The 260 neural weights are the original M4.4A
architecture reused by the unmodified M5.6 \`Learner\`.
This is a real integration of an existing tested
neural component, not a new hand-coded pathfinding
policy. Note that the food-smell field and the
intrinsic novelty bonus ARE hand-coded environmental
biases/incentives, and no emergent fly-like
intelligence can be inferred from their presence.

## Four standardized bouts

\`\`\`text
source_a     divider opening y=2   (online learning enabled)
source_b     divider opening y=3   (online learning enabled)
shifted_eval divider opening y=7   (all learning FROZEN)
shifted_adapt divider opening y=8  (online learning enabled)
\`\`\`

Each bout starts with fresh body energy at spawn
(2,5) and fresh food ecology. Brain weight updates
persist across bouts, but hidden recurrent state is
explicitly reset for comparable beginnings. For the
curiosity arm, personal visit counts persist.
Thus, this is not yet proof of continuously
surviving individual organisms over entire lives.
It is a reproducible developmental intervention.

Three original matched arms:
\`frozen\`, \`plastic\`, \`plastic_novelty\`.
Additional \`plastic_novelty_reset\` ablation
erases only the learned synaptic weights before
the first shifted test, preserving the original
visit history; no new phenotype/motor rules are
injected.

## Run on branch

Requires Python 3.11 and the existing
\`neuroevolution\` package, no separate new
science library.

\`\`\`bash
python -m unittest discover -s tests -p 'test_flyloop_m0.py' -v
python -m unittest discover -s tests -p 'test_m5_6_online.py' -v
python -m neuroevolution.flyloop_m0 --out results/flyloop-m0/embodied-dev.json
\`\`\`

Science uses precisely \`44000..44015\` as 16
paired controller/action seeds; \`43000..43003\`
are reserved solely for mechanical development
tests. Complete science produces **224 bouts**
and **32,754 physical attempted motor actions**
under the frozen protocol.

Results:
**6/10 prospectively frozen screening gates pass**
(**5/5 integrities; only 1/5 scientific goals**).
Physical-reward learner eats 0.1875 far-side
meals per seed in first shifted frozen-learning
bout vs 0 for frozen brain; curiosity variant
eats only 0.0625, so no compelling new-world
route discovery. All confidence intervals and
negative results in the scientific report.

## How previous work is repositioned

| Component | Current M0 role |
|---|---|
| Original M5.6 online RNN | **DIRECTLY EXECUTED** 260-parameter recurrent controller and causal gradient |
| Original M5.6 IntrinsicReward | **DIRECTLY EXECUTED** bounded location-visit novelty, separately from physical energy |
| Original M5.0/M5.5 ecology | **UNTOUCHED**, not assumed equivalent to the new grid |
| M5.34 cross-task representation | Potential future prediction-model adapter, **not yet executed** |
| M5.36 memory consolidation | Potential future memory ablation, **not yet executed** |
| M5.37 context selection | Potential future action-control mediator, **not yet executed** |
| M5.38–M5.43 critical hypotheses and exploration | Future **optional slow loop**, **not yet executed** |

The goal is not to connect every previous
prototype simultaneously; each mechanism will
enter only after it has a defined interface,
an honest causal test, and a reason to exist.

\`evolution_authorized=false\`.
The original sealed M5.6 ecology and main
are unchanged. This is a draft stacked PR,
not a release.
