# Research roadmap

## M1 — Controlled structural neuroevolution (implemented)
- Fixed-topology versus topology-mutable recurrent networks.
- Deterministic changing-cue environment, train/holdout separation.
- Checkpoints, tests, SVG and GitHub Actions campaigns.
- **First result:** 3 paired seed runs; positive mean effect but not robust; see [M1 report](../reports/M1_campaign_2026-11-20.md).
- Outstanding: 20–30 independent seed replications, true out-of-distribution tasks, runtime audit.

## M1-P — Continual plasticity pilot (implemented; independent branch of research)
- **Within-lifetime** SGD on an 8-task sequence; four paired arms: continual, fresh, low-activity unit reset, matched random-unit reset.
- Accuracy learning curves, seed-level AUC, forgetting, dead-unit fraction; controls on train/holdout leakage.
- [Frozen protocol](PLASTICITY_PROTOCOL.md), automated tests, separate GitHub Actions campaign.
- **M1-P outcome:** 24 seed × 8 tasks. Continual network learns slower than fresh by 4.59 pp AUC; low-activity renewal improves 2.35 pp over continual, with forgetting trade-off. See [M1-P report](../reports/M1P_plasticity_2026-11-20.md).
- **M1-P2 independent confirmation:** 24 **new** seed × 16 tasks. No evidence for progressively worsening gap between matched early/late task families (−0.30 pp; broad interval covering zero). Targeted renewal again beats random reset (+1.27 pp AUC on tasks 8–15). See [M1-P2 report](../reports/M1P2_confirmation_2026-11-20.md).
- **Conclusion:** promising constrained renewal benefit, but not cumulative plasticity loss nor autonomous self-modification; preserve the null result.

## M2 — Evolving a bounded renewal policy (implementation in experiment branch)
- A five-parameter meta-controller evolves **when to renew**, and the dose (0 / half / maximum 20%).
- Frozen [M2 protocol](M2_PROTOCOL.md), fully disjoint meta-train/meta-validation/sealed-test seed sets.
- Six comparison arms, including matched unit dose and shuffled matched schedule.
- M2 changes *a constrained adaptation policy*, not the neural learning algorithm or arbitrary Python source.
- Gate: only count real adaptive control if held-out advantage survives schedule-shuffled controls **and** the action histogram shows state-dependent variability.
- M2 follow-up (not implemented): inherited local weight-update rules or a genuine learned neural optimizer with matched cost.

### M2 result: null adaptive policy
The original [M2 campaign](../reports/M2_adaptive_2026-11-21.md) selected zero renewal on all 112 opportunities; its dose-matched timing controls became identical, while fixed renewal improved AUC. This is a negative scientific result **under M2's preregistered cost-weighted objective**, not grounds for retroactive tuning.

### M2B: matched-budget scheduling (implementation in experimental branch)
New [frozen M2B protocol](M2B_PROTOCOL.md): every competitive controller must spend the same number of unit renewals. Compare learned state-aware timing with an independently evolved **time-only** policy, uniform scheduling and shuffled dose-matched actions. Test on new seed sets. Forced late renewals are recorded separately from requested decisions.

### M2B result — state-aware vs clock-only scheduling (21 November 2026)
- [M2B report](../reports/M2B_budgeted_timing_2026-11-21.md) and [raw GitHub run](https://github.com/kkongiu/neuroevolution-lab/actions/runs/44883457787).
- **Primary result not confirmed:** state-aware controller AUC 65.50% vs clock-only 65.75% (paired −0.25 pp; 95% descriptive bootstrap −0.75/+0.20 pp). No advantage under equal budget of 10 unit renewals.
- **Behavioral nuance:** 23/112 requested renewals respond to state input, but 25 interventions are forced by the budget constraint; state-dependent behavior is measurable, not demonstrably beneficial.
- Choosing low-activity units, independent of timing, again appears helpful versus random unit selection (+1.25 pp on average, exploratory).
- Do not retune existing M2B after seeing the final test. Next experiment must specify a new environment and unseen seeds before execution.

## M3 — Unpredictable novel and returning tasks (implemented; awaiting first sealed campaign)
- [Frozen M3 protocol](M3_PROTOCOL.md): stochastic task recurrence, new samples for exact recurring latent tasks, new task families on the shifted final test.
- Controller does not observe task ID or novel/return status; uses only past/current training signals and elapsed index.
- Meta-train/validation/regular-test/shifted-family-test seeds are separate.
- Compare evolved state-aware timing against independent time-only policy, state-blind ablation, uniform/permuted dose and random-unit controls.
- Fixed unit-renewal budget; track requested-vs-forced actions and AUC separately for novel and returning tasks.
- **Gate:** positive paired AUC on shifted-world test, evidence of state-sensitive requested actions, and no unacceptable retention cost.
- This is still an externally evolved, constrained policy, **not** open-ended autonomous discovery of interests.
Richer worlds, curiosity-like incentives with noisy-TV negative controls, and behavior descriptors for quality-diversity analyses.

### M3 first result — null advantage under recurrence and partial distribution shift
- [M3 report](../reports/M3_novelty_recurrence_2026-11-21.md), [GitHub run](https://github.com/kkongiu/neuroevolution-lab/actions/runs/44884705352), [per-seed metrics](../reports/M3_test_seed_auc_2026-11-21.csv).
- On shifted task families, adaptive AUC 65.90% versus independently evolved time-only 66.29%, difference **−0.39 pp**, bootstrap percentile 95% **−1.23 / +0.33 pp**: *primary hypothesis not confirmed*.
- On familiar families, adaptive 73.89% versus time-only 74.66%, difference **−0.77 pp**.
- State signals do change requested actions (59/108 shifted, 75/108 familiar), but their use does not yield demonstrated benefit. Preserving negative findings is mandatory.
- Original summary.json incorrectly counted novel/return events for the first seed only; M3 raw task metrics confirm **73 novel / 47 return** events per 12-seed group (including initial task). The post-run diagnostic fix changes only aggregation and leaves published results unmodified.

## M3.1 — Causal positive control: when preservation vs renewal helps (completed)
- [Frozen M3.1 protocol](M3_1_PROTOCOL.md). Two cloned contexts per pretrained model: intact and controlled hidden-unit disruption.
- Evaluate **both potential outcomes** (preserve / renew) on identical heldout examples and adaptation streams; learn a threshold on a *training-only* diagnostic probe.
- Train, validation, main final test, and shifted final test have disjoint seed ranges.
- Compare learned state-based decisions against an independently optimized time-only rule, a shuffled **exact-dose-matched** control, always preserve, always renew and an oracle upper bound.
- **Gate:** first verify on meta-training seeds that renewal harms intact networks but helps disrupted networks; a pass only demonstrates that the benchmark has two opposing causal incentives.
- Positive-control effect is NOT open-ended self-improvement: the corruption is intentionally engineered, and the action space is explicitly restricted.

### M3.1 empirical result — positive control succeeded, generality still unresolved
- [M3.1 scientific report](../reports/M3_1_causal_repair_2026-11-21.md), [full GitHub run](https://github.com/kkongiu/neuroevolution-lab/actions/runs/44897210983), [per-seed archive](../reports/M3_1_seed_metrics_2026-11-21.csv).
- Main final 24 seed: state-diagnostic threshold yields **90.51% learning AUC** vs **79.69%** for shuffled *exact-dose-matched* controls (+10.82 pp, paired bootstrap 95% +9.28/+12.33 pp), and **85.39%** for an independently optimized time-only policy (+5.12 pp).
- Intact: renewal **harms AUC by 10.24 pp**; deliberately damaged network: renewal **helps by 30.39 pp**. Opposite effects were validated first on meta-training seeds before final tests.
- Shifted XOR/circle 16 seed: smaller advantage (+2.80 pp versus matched shuffle), and oracle action agreement falls **96.88% → 65.63%**. It is a fragile out-of-distribution effect.
- **Critical limitation:** accuracy before intervention practically reveals the deliberately inserted damage on the training family. The renewal operator is intentionally aligned with the damage mechanism. This is *only* a causal **positive-control benchmark**, not an emergent capability to discover how to repair arbitrary structural failures or invent goals.
- Next: pre-register M3.2 for unlabelled, non-artificial learning interference with new seed splits, internal-state diagnostics and true memory preservation measurements.

## M3.2 — Continuous training with natural task interference (completed; primary null)
- [Frozen M3.2 protocol](M3_2_PROTOCOL.md), distinct from the engineered M3.1 positive control.
- One evolving network faces 12 tasks with latent task returns and novelties; **no artificial damage**.
- Fixed total of four renewal actions; three independently evolved policies: full neural diagnostics, accuracy-only, and time-only; fixed uniform/shuffled/random-unit controls.
- Diagnostics derive exclusively from labelled TRAINING data; full final test and shifted-family final test use new seed ranges.
- On final seeds, both interventions are cloned from each adaptive trajectory's pre-decision state, then trained identically to record one-step AUC/forgetting effects.
- Primary gate: held-out seed-level AUC vs independent time-only; necessary secondary: improvement over independent accuracy-only without concealing memory loss.
- Interpretive caution: some latent label rules conflict on the same unlabeled 2D inputs, so old-task retention can be fundamentally limited; this is **not** arbitrary self-programming.

### M3.2 actual result — primary negative, operator-selection signal
- [M3.2 report](../reports/M3_2_natural_interference_2026-11-21.md), [sealed GitHub run](https://github.com/kkongiu/neuroevolution-lab/actions/runs/44899315401), [32-seed archive](../reports/M3_2_seeds_2026-11-21.csv).
- Main 20 test seeds: full-state adaptive AUC **74.46%**, independently optimized time-only **74.77%** (−0.316 pp, paired bootstrap 95% −0.837/+0.100 pp), and accuracy-only **74.84%** (−0.382 pp). **Neither preregistered success gate passed**.
- At exactly the same renewal times/counts, low-activity unit selection versus random-unit selection was **+1.494 pp** on regular tasks (exploratory), but with greater forgetting; effect is **not robust** on shifted checker/wedge tasks.
- One-step potential outcomes were evaluated from **352** real pre-intervention states (no synthetic damage). Main mean renewal effect was +0.439 pp learning AUC, but +0.710 pp forgetting. Local oracle agreement only 46.8% (main) and 53.8% (shifted).
- Fundamental design limitation: changing task rules can produce **mutually incompatible labels for identical 2D inputs without context cues**, making perfect retention impossible. Do not claim a general inability of neural networks to self-adapt from this environment.
- Suggested M3.3: introduce an observable context/task cue or compositional, non-contradictory task relationships, freeze new seed splits, test achievable lifelong memory and adaptation under comparable resource budgets.

## M3.3 — Compatible contextual memory and replay (completed; adaptive-renewal null)
- [Frozen M3.3 protocol](M3_3_PROTOCOL.md) created before experiment implementation.
- Four explicit context IDs (one-hot) resolve two pairs of logically opposed labels for identical 2D points. Context-blind matched-capacity control is provably restricted to 50% balanced accuracy on paired holdout panels.
- One persistent 20-unit ReLU MLP learns 12 contexts episodes, each context seen at least once. Evaluate final memory on all four contexts, plus peak-to-final forgetting.
- Training-only replay buffer replaces one in four SGD mini-batches; compare cue+replay vs cue+no-replay and cue-blind+replay.
- Fixed budget of three neural renewal events; independently meta-trained state-aware, accuracy-only and clock-only policies, with uniform/shuffled/random-unit matched controls.
- Independent seed partitions plus partial OOD stripe/checker-opposition test.
- **Primary:** final four-context balanced accuracy, cue+replay vs matched context blind. **Secondary:** replay ablation. **Exploratory:** does neural state-aware renewal outperform time and accuracy-only on learning AUC without losing contextual retention?
- No autonomous objective invention or neural source-code rewriting. Interpret the provided context cue as an external scaffold, not general reasoning.

### M3.3 empirical result — contextual memory improves, forced renewal not justified
- [M3.3 full report](../reports/M3_3_contextual_memory_2026-11-21.md), [sealed GitHub run](https://github.com/kkongiu/neuroevolution-lab/actions/runs/44900094399), [36-seed archive](../reports/M3_3_seeds_2026-11-21.csv).
- Main **24 seeds**: context + replay, without renewal, **74.33%** final balanced four-context accuracy versus **50.00%** for a matched blind architecture: **+24.33 pp** (paired bootstrap 95% +22.23/+26.45). The 50% bound is **guaranteed by opposing paired labels**, not evidence of general intelligence.
- Replay versus no replay (both with context and zero reset): **+5.03 pp final memory** on familiar tasks (+2.07/+8.06 pp), **−5.88 pp peak forgetting**, but **−1.07 pp current-task learning AUC**: a measurable stability–plasticity trade-off.
- On shifted stripe/checker families, context still helps (+12.43 pp versus blind) but replay offers **no confirmed gain** (−0.62 pp, interval includes zero). Worst-context accuracy is only ~52.8% on main, so memory remains unreliable for some contexts.
- Learned adaptive renewal did NOT beat independent time-only or accuracy-only (+0.03 pp AUC, wide uncertainty); its final memory **70.51%** is below context+replay/no-reset **74.33%**. Do not advance to autonomous self-modification claims.
- Protocol caveat: M3.3 adaptive controller's retention feature includes all previously stored contexts including the current context on return, while the text described other-context memory. Main cue/replay controls are unaffected; any fix needs new seeds.
- Next research issue: [M3.4 latent context inference and bounded episodic memory](https://github.com/kkongiu/neuroevolution-lab/issues/14).

## M3.4 — Online context recognition from labelled experiences (completed)
- [Frozen M3.4 protocol](M3_4_PROTOCOL.md) committed before implementation/full test, [report](../reports/M3_4_latent_context_inference_2026-11-21.md), [original sealed run](https://github.com/kkongiu/neuroevolution-lab/actions/runs/44901460644), [48-seed archive](../reports/M3_4_seed_metrics_2026-11-21.csv).
- The deployable kNN resolver sees **training-only labelled support examples**, never hidden context IDs; it supplies anonymous slots to a persistent six-input ReLU network.
- Main **24 independent test seeds** (16 clean support examples): **100%** context-return consistency and 4/4 distinct first visits. Final inferred-context memory **72.18%**, versus 50.00% no support and 38.95% after adversarial final-support rotation; primary paired +33.24 pp (descriptive bootstrap +28.74/+37.50 pp).
- This main success is in a **deliberately easy/structured task**: the meta-train/meta-validation contexts were also perfectly classified. The adversarial support shuffle is below chance because swapped opposing contexts mislead the model; zero-support 50% is a *mathematical* ceiling.
- With six support examples including one corrupted label: return consistency **62.5%** (below preregistered >65% threshold), final memory **66.37%**. With shifted stripe/checker families: return consistency **30.2%**, final memory **54.30%**, versus privileged oracle ID **59.64%**. **Robust latent-context inference is NOT established.**
- Replay FIFO vs no replay improves main memory by **+2.98 pp**, but descriptive bootstrap interval (−0.02/+6.05 pp) includes zero; FIFO vs reservoir similarly uncertain. No demonstrated replay advantage on shifted families.
- The resolver is **externally written kNN with an offline meta-selected threshold**, not autonomous algorithm invention. Labelled support signals are required; no claim of general unsupervised context discovery.
- [M3.5 issue](https://github.com/kkongiu/neuroevolution-lab/issues/16): test whether querying one more labelled example when uncertain outperforms random/periodic requests *at matched information cost*, with new seed splits.

## M3.5 — Active feedback seeking under an equal global query budget (completed)
- [Frozen M3.5 protocol](M3_5_PROTOCOL.md), created before the full simulation code and any final-seed experiments.
- Six initial labelled support observations per episode, with exactly one wrong label; four of twelve episodes may request two **new IID labelled examples** at a fixed total cost of eight labels per seed.
- Primary: margin-uncertainty scheduler versus **random-request schedule**, identical information budget and persistent M3.4 neural/associative architecture. Necessary secondary versus periodic requests at the same cost; independent entropy heuristic as exploratory.
- Counterfactual controls: no extra information; all episodes requesting extra information (higher cost). No use of hidden context ID or future request labels in request decisions.
- Meta-train/validation thresholds on new disjoint seed groups. Final test: 24 independent known-family seeds, 12 shifted-family seeds, 12 known-family seeds with **one extra corrupted requested label**.
- Audit meta-accuracy separately from final memory: return-context consistency, context-slot clustering, first-visit distinctness, four-context balanced memory, worst-context accuracy, forgetting, learning AUC, query counts, forced actions and kNN overhead.
- Success requires a positive paired main advantage **beyond random at identical information cost**, and support versus periodic; do not relabel poor shifted-family generalization as a success.
- Context resolver and querying policy are externally programmed; simulated neural networks cannot generate or run code or establish independent objectives.

### M3.5 empirical result — direct random-control advantage not confirmed
- [M3.5 scientific report](../reports/M3_5_active_information_2026-11-21.md), [sealed original GitHub run](https://github.com/kkongiu/neuroevolution-lab/actions/runs/44903487190), [48-seed final accuracy archive](../reports/M3_5_seed_accuracy_2026-11-21.csv).
- Main **24 independent seeds**, exact 4 × 2 = 8 extra labels per policy: uncertainty margin **65.30%** final four-context accuracy vs independently random-episode queries **62.92%**; difference +2.38 pp, paired bootstrap descriptive 95% **−1.56/+7.27 pp**: **primary hypothesis NOT confirmed**.
- Preregistered secondary margin vs periodic **+4.22 pp** (+0.68/+8.61 pp), but does not rescue primary null. Return-to-original-slot consistency **77.08%** margin vs **64.06%** random on main.
- On shifted stripe/checker (12 seeds), margin **56.32%**, versus periodic/no query **57.16%**. With corrupted additional feedback, margin **63.74%**, no-query **63.87%**. More labelled queries can worsen retention.
- Actual margin action requests in main: **96/96 executed queries self-triggered by the external uncertainty threshold**, zero quota-forced, while **72 additional post-quota desires were blocked**. Distinguish requests from blocked desires and budget enforcement.
- Externally selected query thresholds on separated meta-seeds: margin **0.70**, entropy **0.75**. This is a bounded, **researcher-written information-request algorithm**, NOT self-generated curiosity or autonomous goals.
- Next preregistration question: [M3.6 — reliability and cost-aware spatial querying](https://github.com/kkongiu/neuroevolution-lab/issues/18), with a non-mandatory query budget and comparably costly baselines; do not retune M3.5 or reuse its sealed test seeds.

## M4.0 — Spatial food-seeking and evolutionary survival (completed; primary negative)
- [M4.0 preregistered protocol](M4_0_PROTOCOL.md), distinct from M3.5 information-seeking work.
- Bounded 13×13 grid with randomly generated obstacles, BFS-reachable edible food, energy expenditure, mortality, 100-step episodes. Sensory inputs: adjacent barriers/food, limited odor, energy, feedback.
- 12-unit Elman RNN controls 4 moves; numeric weights inherited and Gaussian-mutated **between generations**, not updated inside a lifetime.
- Train on 8 deterministic independent mazes across 8 generations ×12 genomes, select generation winner only on 4 separate validation worlds, and evaluate one frozen policy on 32 **held-out** and 16 harder shifted worlds.
- Matched world controls: fixed random RNN, wall-avoiding random movement, deliberately strong greedy odor rule, same champion with no odor. Primary paired food collected versus wall-avoiding random; necessary secondary versus random RNN. Survival, navigation and collisions are diagnostics.
- No predators or organism reproduction inside the ecosystem yet. Future study should add spatiotemporal memory tests, food regeneration and reproduction one variable at a time, with selection and measurement preregistered. No autonomy/consciousness inference.

### M4.0 scientific result — evolved policy fails against wall avoidance
- [Complete M4.0 report](../reports/M4_0_survival_world_2026-11-21.md), [GitHub Actions original campaign](https://github.com/kkongiu/neuroevolution-lab/actions/runs/44904926385), [48-world food dataset](../reports/M4_0_food_by_world_2026-11-21.csv).
- One frozen recurrent organism selected by **validation at generation 2** after 768 evolutionary candidate-world evaluations; final 32 unseen worlds. Evolved food **2.19** vs wall-avoiding random **3.69** (**−1.50 food**, paired bootstrap 95% **−2.53/−0.47**): **preregistered primary hypothesis failed**.
- It collects more than one fixed unevolved random network (**0.56 food**) but far less than a hand-coded greedy odor follower (**8.47 food**). No evidence of useful odor reliance: evolved **2.19**, same genome with odor sensors zeroed **2.59**.
- Evolver averages **17.66 wall collisions per life**, dies of starvation in **all 32** main-world tests. The specific failure mode is poor motor action selection, even when local wall sensors are available.
- Shifted test (16 unseen harder mazes): evolved **1.31** foods vs wall-avoiding random **2.00**, greedy odor **2.69**; primary advantage not generalized.
- Training best fitness improves from **3.29** to **6.90**, but validation peak at generation **2** rather than later improved train leaders suggests possible environment overfitting. Only **one genetic evolution run**: world-level replication is not genome-run replication.
- **Next issue [M4.1](https://github.com/kkongiu/neuroevolution-lab/issues/20):** isolate wall-action masking and replicate evolution across independent search seeds on NEW held-out worlds. Do not retroactively change the M4.0 policy, reward weights or seed splits.
- No predators, in-world organism reproduction, within-lifetime weight training or emergent self-chosen objectives yet. An implemented ecology is not proof of intelligence.

## M4.x — Persistent objectives (speculative, not implemented)
Preregister observable persistence and revision tests. Keep evidence of goal-directed behavior strictly separate from subjective experience and autonomy claims.
Preregister observable persistence and revision tests. Keep evidence of goal-directed behavior strictly separate from subjective experience and autonomy claims.
