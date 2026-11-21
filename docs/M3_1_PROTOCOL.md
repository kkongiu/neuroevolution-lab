# M3.1 — Counterfactual value of neural restructuring (preregistered)

**Status: protocol frozen BEFORE any full M3.1 held-out experiment.** M1–M3 code, metrics, and reports must remain unchanged.

## Rationale

M2B and M3 showed that a state-aware controller's choices depend on observed neural state, but they did **not** improve final learning AUC over simple schedules. The next question is narrower and falsifiable: **can a controller learn when replacing neural units helps and when preserving a learned representation is better?**

This is a *positive-control causal benchmark*, deliberately constructing both favorable and unfavorable conditions for renewal. Any success shows only diagnostic action selection under these conditions, **not** autonomous goals, general plasticity, or free self-modification.

## Causal design

Each paired experiment starts with a small one-hidden-layer ReLU classifier, pretrained on an explicitly supervised 2-D task. Clone the same pretrained network into two *contexts*:

1. **Intact:** retain the previously learned representation. Renewing its highest-influence units may destroy useful acquired structure.
2. **Disrupted:** selectively reverse and amplify the outgoing weights of its influential hidden units, simulating an exogenous representation failure. Renewing influential units may repair the corruption.

The context is **not revealed to the policy**. The policy can inspect only a bounded labelled *training probe* from the current task and the network's predictions/hidden activations.

For each context, clone the exact same pre-action model and evaluate **both potential interventions**: preserve or renew. Holdout examples, training updates, and their ordering are identical across the two intervention branches, and reset randomness is deterministic. Record **effect = AUC(renew) − AUC(preserve)** for each paired context.

The renewal operator is fixed for every policy: it reinitializes the highest estimated output-contribution hidden units, as measured on the current *training probe*, never the hidden source context label. The disruption and operator are intentionally related: this is a diagnostic challenge, and must not be presented as evidence for a broad adaptive-learning mechanism.

Stable and disrupted contexts are equally represented and their order is shuffled per seed. Individual episodes are independent snapshots, not a continuous lifetime.

## Data and settings (locked)

- Meta-training seeds **12000–12007** (8).
- Meta-validation seeds **13000–13003** (4).
- Main final test seeds **14000–14023** (24).
- Partial out-of-distribution diagnostic seeds **15000–15015** (16), using different supervised task families (XOR and circle instead of line).
- Each seed creates **4 independent pretrained models**, each evaluated in **both contexts** = 8 decisions per seed.
- **12 ReLU units**; top **3** units eligible for renewal; **96** SGD pretraining steps, **48** post-intervention SGD steps, batches of **4** examples, learning rate **0.05**.
- Initial and post-adaptation evaluation: **96 independent holdout examples** at steps 0, 12, 24, 36, 48.
- Training probe: **48 labelled training examples** distinct from holdout. Probe labels are permissible online feedback, not oracle outcome labels.
- Disruption: sign-reverse and multiply by **5** the outgoing weights of the top 3 influential units, using a deterministic criterion on a *training* probe. The full model is not reinitialized.
- Renewal: reinitialize the top 3 influential units' input weights, hidden bias, and output weight with the original architecture's initialization distribution.
- CPU-only, Python standard library; no network, shell, code generation, or GitHub writes from the simulated controller.

## What is learned?

A **single threshold** on *pre-intervention training-probe accuracy* selects *renew* if accuracy is below the threshold; otherwise preserve. Choose threshold from a frozen grid **[0.40, 0.45, ..., 0.95, 1.00]** plus the constant preserve/renew candidates. Optimize train AUC (measured on meta-training examples), choose among its **four** best train performers on separate meta-validation seeds. Never choose on final-test seeds.

To control for learning a trivial action schedule, independently optimize a **time-only threshold on episode index** by the same training/validation procedure and candidate count, including constant actions. This clock policy sees no neural-state probe.

On every final seed compare:
- **Learned:** state-aware probe threshold.
- **Time-only:** independently learned episode-position policy.
- **Shuffled dose-matched:** permute learned actions across all eight contexts, preserving exact renewal count per seed.
- **Preserve-all** and **renew-all**.
- **Oracle (upper bound only):** independently choose the better action from two holdout counterfactuals. Oracle is NOT available to controller.
- The paired potential outcomes are used to calculate causal effect and explain failures, not provided to learned/time-only policies during final test.

## Preregistered outcomes

**Primary:** per-seed mean post-adaptation **learning-curve accuracy AUC** difference: learned minus shuffled-dose-matched, on 24 main final seeds. Report paired bootstrap percentile 95% interval (seed as independent unit) and sign counts.

**Necessary secondary gate:** learned must also outperform independently learned time-only strategy on the main test; otherwise no general adaptation advantage is claimed. Report OOD diagnostic separately without retuning the policy.

**Causal opportunity calibration:** using **meta-training seeds only**, the average treatment effect AUC(renew)−AUC(preserve) must be negative in intact contexts and positive in disrupted contexts. If this fails, the environment has not supplied the hypothesized opposing incentives: report that fact and do not claim an adaptive benefit even if an aggregate controller score is high.

Also report effect distributions by context, balanced accuracy of detecting beneficial renewals, regret against oracle, renewal count and learning AUC, pre/post task accuracy and runtime proxy (numbers of SGD evaluations and renewed units). Equal reset counts are assured only for the shuffled comparator, **not** necessarily for time-only and constant baselines.

Do not count two contexts from one pretrained model as independent biological/experimental replications. Use the seed as the aggregation unit.

## Interpretive limits and falsification

A success would be an **engineered positive control**: sensing training failure helps choose between two fixed interventions, under a deliberately inserted corruption. It would **not** establish learned self-generated goals, a self-rewriting algorithm, open-ended structural evolution or a cure for continual-learning plasticity.

A null or negative result on heldout or OOD is a valid research result. After seeing the final test, change no threshold grid, corruption strength, seed partitions, metrics, or selection code: any follow-up must use a separately versioned protocol with new seeds.

The prior M3 conclusions remain frozen and publicly accessible.
