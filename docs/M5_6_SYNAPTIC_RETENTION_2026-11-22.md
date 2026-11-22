# M5.6 — Development-only learned-synapse retention test (2026-11-22)

**Outcome: No evidence for a survival benefit from retaining curiosity-trained numerical synapses.**

Source: [successful GitHub Actions run 45013591655](https://github.com/kkongiu/neuroevolution-lab/actions/runs/45013591655), 3 new retention tests + 5 original online regression tests passed, 96 ecological simulations executed, complete original rows archived as `m5-6-development-learned-synapse-retention` (30-day artifact retention). Canonical row digest SHA256 `5c85a9368c4a1939d932e27136e3b861adf1319a805d1e79e480d47bb33fa300`.

## Design
24 new development seeds 1056–1079; ecological environment calibrated on *other development seeds*, 1 predator, 9 initial food, regeneration probability 0.55. Training ticks 1–30; evaluation ticks 31–60. Four fully paired arms:

1. Strong visit-novelty curiosity (4×) during first 30 ticks, then intrinsic reward **off**, numeric policy updates **off**, hidden recurrent activations reset, retain learned synapses.
2. **Exact same training history** as (1), but at tick 31 reset all numeric network synapses to their original weights, while identically clearing hidden state and stopping all reward and learning. Thus the within-seed pair differs ONLY in trained versus initial synaptic coefficients at the cut.
3. Food-only learned synapses, then all updating disabled and recurrent hidden reset.
4. Network weights frozen for whole life.

All four use native movement and predator capture, same scripted reproduction rule and unchanged physical energy accounting. Post-cutoff gradient updates verified zero in every trial. This tests within-world conditional persistence, **not** transfer to a new geometry and not epistemic prediction-progress curiosity.

## Results

| Arm | Alive at tick 30 | Alive at tick 60 | Mean post-cutoff food |
|---|---:|---:|---:|
| Strong curiosity, learned synapses retained | 22/24 | **11/24** | 3.25 |
| Same history, original synapses restored | 22/24 | **12/24** | 3.08 |
| Food-only learned, retained | 23/24 | **10/24** | 3.21 |
| Network frozen all along | 22/24 | **6/24** | 3.46 |

Primary causal contrast, retain vs reset: **−1/24 worlds = −4.17 percentage points**. At the matched branching point both groups survive in exactly 22 of 24 worlds. Relative to resetting synapses, retaining won one world, lost two, and tied 21. Differences are **far too small for a robust advantage claim**. Average births and food differences do not establish retained knowledge.

**Interpretation:** The earlier 15/24 vs 11/24 apparent benefit of strong novelty during training is not supported by this stricter synaptic-retention ablation. It may have arisen from contemporaneous reinforcement behavior and chance rather than useful stored knowledge. The learned policy does not show survival-relevant memory retention under this evaluation. This negative result is scientifically useful, not an integration failure.

## Limits / next requirements
- Preliminary 24-seed **development-only** screening; final/replication cohorts remain sealed.
- Same-world continued survival, not transfer to previously unseen environment.
- Resetting learned synapses to original values removes both food-mediated and curiosity-mediated learning, not just curiosity-specific learning; follow-up needs causal factorization by source reward history.
- Controls have unequal training reward budgets; no validated predictive-learning-progress reward or dynamically learned hunger/curiosity arbitration.
- No merge to `main`, no justification for consuming final held-out worlds.
