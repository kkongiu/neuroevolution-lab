# M5.6 development-only curiosity/survival balance pilot

**Date:** 2026-11-22. **Status:** exploratory developmental evidence, not the preregistered held-out M5.6 experiment.

**Source:** [GitHub Actions run #45010224426](https://github.com/kkongiu/neuroevolution-lab/actions/runs/45010224426), completed SUCCESS. JSON raw paired rows uploaded as `m5-6-development-balance-pilot` artifact (30-day retention). SHA256 of canonical trial rows: `6456b14da0237befb41e3431f0af0baae004f56273341c06e4d926be3d8a9598`.

## Frozen development pilot configuration
- **252** short ecological simulations = 12 matched development seeds (1000–1011) × 3 predator intensities (0, 1, 3) × 7 conditions.
- **36 ticks**, 8 initial organisms, 11×11 map, limited food, original M5.0 movement/energetics and M5.5B predator movement/capture.
- Seven explicitly labeled arms: no curiosity, weak (0.25×), moderate (1×), strong (4×), engineered energy-gated (1×), shuffled novelty placebo, and frozen synapses. Scripted reproduction remains identical across arms. `energy_gated_scripted` is **not spontaneous emergent self-preservation**.
- Training reward is one-step actual food plus novelty on visited cells; no prediction learning-progress reward has yet been integrated. Recurrent synapse updates are one-step truncated policy gradient, not complete episode BPTT.
- Physical initial states and keyed pseudorandom worlds are paired; behaviors can diverge after interventions.

## Complete world-level population-persistence counts at tick 36

| Condition | 0 predators | 1 predator | 3 predators |
|---|---:|---:|---:|
| No curiosity | 12/12 | 12/12 | 0/12 |
| Weak curiosity, 0.25× | 12/12 | 12/12 | 0/12 |
| Moderate curiosity, 1× | 12/12 | 12/12 | 1/12 |
| Strong curiosity, 4× | 12/12 | 11/12 | 1/12 |
| Engineered energy-gated 1× | 12/12 | 12/12 | 1/12 |
| Shuffled novelty placebo | 12/12 | 12/12 | 1/12 |
| Frozen synapses | 12/12 | 12/12 | 1/12 |

### Secondary context (mean per world)
- With **one predator**, average final population: no curiosity 2.75; weak 2.75; moderate 2.58; strong 2.92; energy-gated 2.58; shuffled 2.58; frozen 2.75. The stronger-curiosity world-persistence result was **11/12**, not 12/12; final population and persistence are distinct outcomes.
- With **three predators**, all arms show near-total extinction (0–1/12) and near-zero final population. Any **1/12 versus 0/12** survival difference is too unstable to interpret.
- Without predators, universal persistence (12/12) makes world-level survival unable to discriminate arms (ceiling effect).

### Interpretation and limitations
- **No convincing curiosity survival improvement** has been shown. The small apparent differences are on *single worlds* and are also present in shuffled/frozen controls, undermining causal attribution.
- Physical predator pressure, not intrinsic exploration, dominated short-horizon population persistence in this pilot.
- The likely most informative next experiment needs a calibrated intermediate-difficulty ecology and longer horizon, with independent source and sealed held-out cohorts **frozen before evaluating** them. Do not select final arms or report positive gates from these 12 development worlds.
- The tested script does **not** compare true long-term learned survival strategies or self-regulated energy budgeting; no claim of autonomous instinct-versus-curiosity balancing is appropriate.
- The main repository is unchanged; this remains an open draft PR with source and raw trial artifact available.

## Integrity
The integration and balance-pilot workflows finished SUCCESS. The balance-pilot tests verified paired-grid completeness, zero-novelty equivalence and ecological energy ledger invariants. No sealed/held-out M5.6 worlds were accessed.
