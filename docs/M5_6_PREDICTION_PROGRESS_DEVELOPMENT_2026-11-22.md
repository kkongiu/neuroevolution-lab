# M5.6 — Predictive-learning-progress curiosity, development screen, 2026-11-22

**Outcome: negative for retained survival benefit. No confirmatory survival evidence.**

**Executed:** 100 physical simulations = 20 new development seeds 1080–1099 × 5 predeclared screen arms; 30 ticks training + 30 ticks no curiosity or synaptic updates. Full raw rows archived in [CI run #45016357230](https://github.com/kkongiu/neuroevolution-lab/actions/runs/45016357230); row SHA256 `ac03f578a823ea39bb4e4b62709488308c2406839aaba445f5b4390800009203`.

### Mechanism

Each agent contains a numerical 19-output, 24-input, one-step sensory transition predictor, with 19 observed channels plus one-hot physical action and bias. The first **four observed transitions** establish a local, never-trained fixed probe set. Subsequent physically experienced transitions drive predictor gradients. Reward is capped positive reduction in **loss measured on that exact separate fixed probe, before versus after each update**. Raw prediction surprise without measured improvement earns nothing. The policy's 260 neural weights learn independently by one-step policy gradient on actual food plus intrinsic prediction progress. Predictor models remain personal to each agent, nonheritable, and curiosity does not replenish physical calories.

**Limitations:** early fixed probes can bias the model toward familiar contexts; this is a linear one-step predictive model, not long-term world knowledge or planning. Reward magnitude, learning updates and policy exploration are not matched between arms.

### Experimental configuration

One predator, 9 initial food, regeneration probability 0.55, 60 ticks. World difficulty derived from previous **baseline-only** development calibration. All 20 seeds are disjoint from past development screening sets and sealed source/holdout/replication world cohorts.

At tick 31: disable all intrinsic reward and numerical policy/predictor updates; reset recurrent hidden state. The primary retained-vs-reset pair experienced the exact same first 30 ticks. In the reset arm, learned policy synaptic weights revert to their original values. Predictor weights have no direct motor effect after the cut.

| Development condition | Alive at tick 30 | Alive at tick 60 | Mean predictor positive-progress events | Mean total prediction-progress bonus |
|---|---:|---:|---:|---:|
| Progress-driven, synapses retained | 19/20 | **9/20** | 104.3 | 0.7905 |
| Identical progress history, synapses reset | 19/20 | **11/20** | 104.3 | 0.7905 |
| Food only, no intrinsic reward | 19/20 | **11/20** | 0 | 0 |
| Visit novelty 4×, synapses retained | 17/20 | **7/20** | 0 | 0 |
| Frozen policy synapses from birth | 19/20 | **9/20** | 0 | 0 |

**Key contrast:** retained progress synapses versus reset: **−2/20 = −10 percentage points**, two seeds harmed, none helped, 18 ties. This is a small development sample, not an independently confirmed causal generalization. Numerical predictor itself exhibits strictly positive progress on held-aside probes, but the neural *policy* does not turn those gains into a demonstrated physical advantage. Policy weight changes are not a proxy for useful learning.

**Verification:** 3 prediction tests + 3 screen tests + 5 original online neural ecology tests passed; branch-only GitHub Actions success. All after-cutoff numerical weight updates checked zero, energy ledger conserved, exact paired training history tested and final world seeds prevented by runtime guard. Main unchanged.

### Decision

Do not consume 161xxx source, 162xxx heldout, or 163xxx replication world IDs. For scientific progress, shift from unconstrained intrinsic bonus to *predictive information that the survival controller can actually use*, including risk forecasting, hunger-dependent decisions, longer memory, or model-based action selection. These would be NEW hypotheses needing their own preregistration and development testing, rather than retuning results after looking at heldout outcomes.
