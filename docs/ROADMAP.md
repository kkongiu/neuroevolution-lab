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

## M4.1 — Controlled legal movement and replicated navigation evolution (completed; primary null)
- [Preregistered M4.1 protocol](M4_1_PROTOCOL.md), committed before implementation and new heldout tests.
- Same energy, finite food, odor, walls and fitness as M4.0, but variable reachable start and rotated maps. Motor mask depends ONLY on existing adjacent-wall sensor flags; it corrects illegal preferred actions without knowing food location.
- **8 independent genetic starting populations**, each evolved separately with and without action masking, with exactly 6 generations × 10 candidates × 6 new training worlds per generation; select each leader solely on 4 validation worlds.
- On final 24 main/12 shifted mazes, cross-over both evolved genomes with both decoder settings. Add random masked RNN for each genetic seed, strong greedy odor control, matched wall-avoiding random, and no-odor masked-genome ablation.
- Primary: evolved masked minus wall-avoiding random food on 24 sealed worlds, with **hierarchical paired bootstrap over both 8 genetic replicates and shared 24 world seeds**, >0 and lower bound >0 and >=6/8 independently positive replicates. Necessary secondary evolved masked vs random masked RNN.
- Record actual collisions **and corrected preferences**, energy, survival, visited cells, odors and validation histories. Null-no-food worlds check that food is not manufactured by the simulation.
- Safety mask is researcher-coded assistance, not evidence that the network learned to avoid walls. Keep M4.0's negative results unchanged.

### M4.1 scientific result — mask fixes collisions, not food-seeking competence
- [Complete M4.1 scientific report](../reports/M4_1_masked_navigation_2026-11-21.md), [full GitHub Actions run](https://github.com/kkongiu/neuroevolution-lab/actions/runs/44906089697), [food per independent genetic run](../reports/M4_1_food_by_genetic_run_2026-11-21.csv).
- **8 independently initialized genetic runs × 2 evolved training modes**, each with exactly 360 candidate-world training trials. Test 24 new main worlds and 12 harder shifted worlds, randomized reachable starts and rotated grids; **no M4.0 results retuned**.
- Main **preregistered primary NULL**: masked-trained/ masked-deployed RNN **2.245 food**, legal random walker **2.292**; paired hierarchical mean difference **−0.047** (bootstrap 95% **−1.156/+1.057**), **4/8 genetic runs positive**, below the gate of six.
- Mask has an independently established **causal positive effect** on the SAME genotype: **+0.797 food** (**+0.323/+1.266**), 8/8 genetic runs positive; actual collisions zero, but **23.73 proposed illegal moves per life** still corrected by the programmed decoder. Do not call this learned wall avoidance.
- Evolved masked RNN beats masked random RNN by **+1.344 food** (+0.677/+2.042) but loses **−6.630 food** against a hand-coded odor follower (−8.182/−5.010). No beneficial reliance on odor observed: same champion no-odor **2.313** vs intact **2.245**.
- Training with mask versus without, **both deployed masked**, yields **−0.021 food** (bootstrap includes zero). Thus the measured improvement is mainly the assistance of the **motor decoder**, not a learned food-search policy.
- Harder shifted-world test also null versus walker: evolved **1.063**, random legal **1.083**. All **64 no-food negative controls** yield zero food.
- Next proposed experiment: [M4.2 delayed-cue memory vs matched feedforward](https://github.com/kkongiu/neuroevolution-lab/issues/22). Preregister actual information requirements and test recurrence with a hidden-state reset ablation, rather than introducing predators or reproduction too early.

## M4.2 — Spatial delayed-cue memory: completed; causal-memory null
- [M4.2 frozen protocol](M4_2_PROTOCOL.md) committed BEFORE implementation/evaluation. A new T-fork task gives two symmetric goal targets on identical geometry, but only a 2-step initial cardinal-direction cue, disappearing before the fork.
- Trial ends on entry to one of two one-way fork gates (correct branch yields one food); this is a deliberately narrow cognitive-memory diagnostic, NOT an open survival ecology.
- **No scent, neighboring food, global map, exposed time or energy clock at the fork**, avoiding hidden task-label leakage. Real energy still decays physically.
- Evolve 8 separate recurrent and near-parameter-matched feedforward populations (260 vs 268 numeric weights) over exactly equal budgets and paired worlds; one validation-selected champion of each type per genetic seed.
- Main final **32 paired world seeds**, shifted longer-delay **16 paired seeds**; causal reset of the RNN hidden state immediately before fork decision, flipped initial cue, cue blanking, legal random and a programmed scripted-memory reference.
- Primary: evolved RNN minus independent feedforward final binary food success, hierarchical paired bootstrap over 8 genetic starts and shared 32 world-pairs; necessary secondary reset-at-fork effect and flipped cue effect; report train/validation overfit diagnostics.
- Exploratory transfer only: frozen champions face 12 new open M4.1 mazes with full odor alongside greedy odor/random, and with recurrent odor disabled. A learned cue-memory policy is not expected automatically to forage efficiently.
- Not a demonstration of self-chosen goals or within-life weight learning. Null results stay null. Do not rewrite M4.0/M4.1 or reuse their final seeds.

### M4.2 empirical result — architectural gain ≠ demonstrated memory
- [Scientific report](../reports/M4_2_delayed_cue_memory_2026-11-21.md), [sealed GitHub campaign](https://github.com/kkongiu/neuroevolution-lab/actions/runs/44908425038), [per-genetic-run seed archive](../reports/M4_2_success_by_genetic_run_2026-11-21.csv).
- Main **8 independent genetic runs ×32 paired worlds**: recurrent success **39.26%**, near-size-matched feedforward **28.13%** (primary **+11.13 pp**, hierarchical bootstrap 95% **+2.34/+20.90 pp**, 7/8 genetic starts positive). Primary **architecture** criterion passes.
- **Necessary causal-memory gates FAIL**: same RNN with memory reset at fork **33.01%**, Δ+6.25 pp (bootstrap **−1.95/+14.45**); cue inverted **30.47%**, Δ+8.79 pp (**−6.64/+23.44**). RNN also reaches the fork more than FF (72.07% vs56.25%), confounding locomotion and recall. Programmed one-bit cue memory gets 100%; RNN does not beat legal random (34.38%) with clear evidence.
- **Longer delay shifted 16 pairs**: RNN **26.17%** vs FF **25.00%**, no generalization confirmation. In free-roaming open survival mazes, RNN gathers **1.50** food vs legal random **2.58** and odor-greedy **7.92**; odor ablation Δ+0.14 (uncertain).
- All earlier M4.0/M4.1 negative records preserved. **Next issue:** [M4.3 causal recall without motor confound](https://github.com/kkongiu/neuroevolution-lab/issues/24); preregister a pure memory decision with guaranteed fork arrival before reintroducing free navigation.
- Recurrent state is not within-life synaptic learning; numerical weights are externally evolved. No self-chosen objectives, reproduction or predators.

## M4.3 — Motor-independent causal delayed recall (completed; bounded causal success)
- [Frozen M4.3 protocol](M4_3_PROTOCOL.md), registered before implementation and any test-world evolution.
- Paired opposite target directions, one transient cue (two input ticks), then neutral input with **no action-dependent locomotion or external clock**. Two legal choices at the same final observation; **100% guaranteed decision arrival**. This isolates delayed recall from physical motor success.
- Exact memoryless ceiling: FF, same-RNN state reset and blank cue each score **50% per paired seed**; a handcrafted one-bit memory reference scores 100%. Contradictory oracles become an actual test of hidden-state retention.
- Frozen 8-gen curriculum over delays (0,1,2) → (1,2,4) → (2,4,6); 8 independent genetic initializations; RNN 260 parameters vs FF 268 with the same 86,016 total training trials and matching validation selection budget.
- Sealed **64 main paired worlds** with delays 2/4/6; **32 shifted paired worlds** with unseen delays 8/9/10. Separate **16 guided-versus-free corridor pairs** to evaluate movement transfer without corrupting the memory-only primary.
- Primary RNN minus FF paired food-choice success, hierarchical bootstrap by 8 genetic runs and 64 world pairs, plus **necessary same-genome RNN reset and cue-flip causal effects**, all preregistered. Main-only success is not robust delayed recall if shifted performance collapses.
- No source edits/within-life weight learning by simulated organisms. Negative/null controls retained; M4.0–M4.2 immutable.

### M4.3 published evidence: within-life hidden-state memory demonstrated on symbolic cue
- [Full M4.3 scientific report](../reports/M4_3_causal_delayed_recall_2026-11-21.md), [successful sealed GitHub campaign](https://github.com/kkongiu/neuroevolution-lab/actions/runs/44909467363), [per-genetic-run archive](../reports/M4_3_success_by_genetic_run_2026-11-21.csv).
- On **8 independent genetic runs ×64 paired opposite-goal worlds**, evolved recurrent circuit **74.02%** correct after cue disappears, versus feedforward **50%** (**primary +24.02 pp**, hierarchical bootstrap descriptive 95% **+16.50/+31.54 pp**, **8/8** run wins). This is a ***motor-free symbolic memory test*** with all agents guaranteed fork readout.
- The **same RNN** with its hidden state zeroed right before choosing returns **50%** (causal reset Δ+24.02 pp, 95% **+16.89/+31.84 pp**); flipped cue yields **25.98%** (Δ+48.05 pp, 95% **+32.42/+62.89 pp**). All main preregistered gates pass. Blank-cue RNN and analytical FF control remain **50%**, scripted one-bit memory **100%**.
- Caution: the 50% FF/reset/blank outcomes follow mathematically from balanced opposite targets, and flipped-cue results are linked to normal accuracy. The three contrasts are **not independent replications**. Reset remains a direct same-genotype intervention.
- On **32 shifted world pairs with unseen delay 8–10**, RNN **60.55%** vs 50% control; primary interval **−0.39/+21.68 pp**, **not confirmed**. Guided corridor transfer RNN **62.50%**; free autonomous movement **23.83%**, with fork arrival **52.73%**. Memory is not yet a functional free-roaming navigation policy.
- **Next issue:** [M4.4 integrate causal recall with motor planning](https://github.com/kkongiu/neuroevolution-lab/issues/26). Preregister new genetics/world seeds, fully separate arrival and recalled choice, and report both main and out-of-distribution tests. Do NOT overwrite M4.0/M4.1/M4.2 negative findings.
- None of these runs demonstrate within-lifetime synaptic learning, ecological reproduction, map learning, predator avoidance or autonomous objectives.

## M5.0 — Continuous ecological selection and organism reproduction (complete; primary negative)
- [Frozen M5.0 protocol](M5_0_PROTOCOL.md), committed before code and any final-seed run. Leaves all M4.x scientific results unchanged.
- **No external training fitness or generational selection**: a population of 20 recurrent neural organisms physically occupies one continuous 15×15 world; finite energy, movement, walls, competition for renewable food, death and **explicit in-world reproduce choice**.
- Each birth creates an actual child with parent ID, inherited lineage/generation and a separate 12-energy endowment drawn from 20 energy units charged to the parent. Parent and child cannot generate net energy; track exact cumulative food input, basal expenditure and dissipation.
- Main scientific comparisons use identical founding neural weights and world seeds: **heritable mutation** vs **fresh random offspring** (births but no trait heritability) vs **clone without mutation** (selection but no novelty).
- Preregistered **12 main ×400 ticks** and **8 shifted** (scarcer food, denser obstacles, shorter odor) worlds; per-seed paired bootstrap with the world as independent unit. Primary advantage in cumulative births, necessary generation-depth check and persistence/lineage/food-genome diversity diagnostics. Four no-food worlds must go extinct without reproducing.
- Records every food gain, birth parent/child, genome digest, unique founding lineage and population at scheduled checkpoints, with deterministic replays and SVG maps.
- **This is not unlimited open-ended natural evolution**: genotype length/topology, laws of reproduction, birth energy and resources are still designed; no within-life weight learning, predators or independent objectives. A few births do not establish cognitive adaptation.
- Next phase only if data justify it: repeatable multigenerational persistence, perturb environmental resources, separate selection of pre-existing genotypes from beneficial novel mutations, eventually embodied perception/learning with causal controls.

### M5.0 empirical outcome — births and inherited generations, but all worlds go extinct
- [Scientific report](../reports/M5_0_persistent_ecology_2026-11-21.md), [original successful GitHub Actions campaign](https://github.com/kkongiu/neuroevolution-lab/actions/runs/44911495711), [60 ecosystem world/arm outcomes](../reports/M5_0_world_metrics_2026-11-21.csv). Full raw lineage and 50-tick population data archived with Actions artifact.
- **In-world reproduction genuinely executed** by local RNN organisms; parents pay reproductive energy and children receive inherited/genetically mutated weights and have their own age, body, and lineage. **No outside fitness ranking or genetic champion selection** exists in M5.0.
- Main **12 independent worlds ×400 ticks**: inherited+mutated genotypes **23.42 births/world**, newborns with independent random genomes **20.83**, exact clones **20.17**. Primary Δ+2.58 births, paired bootstrap 95% **−2.58/+9.17**, only **6/12** positive worlds (required 9/12): **failed**. Generation≥3 in **6/12**, required 9/12: **failed**. One world reached generation 8 but also died.
- **ALL populations in ALL arms go extinct**, normal and harder shifted. Main mean extinction time **146.9 ticks** inherited, **130.6** random, **126.5** clones. In the shifted 8 worlds, inherited births **17.13** vs random **18.00**, Δ−0.88 with inconclusive paired interval.
- The average inherited run at tick **100** has just **2.92 living organisms** while **31.33/32 food slots** are occupied: failure likely involves foraging/motor policy, not only lack of global resources. **Zero-food control:** four worlds, no births and all founders die at tick 36. Energy bookkeeping exactly balanced.
- Genome cosine parent-child **~0.967** with small mutation, **1.000** for exact clones, **~0.001** for independent random newborns: inheritance interventions verified, but not beneficial adaptation.
- **Next issue [M5.1 ecological viability & behavioral diagnosis](https://github.com/kkongiu/neuroevolution-lab/issues/28):** check whether a simple scripted greedy food follower with the SAME physical birth law can maintain a population; separate locomotion, perception and reproduction bottlenecks. Do NOT modify old frozen M5.0 test seeds or reclassify null as success.
- Still researcher-designed ecology, not proof of independent goal formation, within-life synaptic learning, adaptive structural evolution, predators or digital consciousness.

## M5.1 — Ecological viability: successful programmed populations, failed spontaneous neural populations (completed)
- [Frozen M5.1 protocol](M5_1_PROTOCOL.md) precedes code and sealed results. M5.0 published environment/negative outcomes remain unchanged.
- Compare **four policies** under identical 15×15 M5.0 energy/resource/birth laws, matched founder genomes and candidate food-spawn schedules: original heritable RNN, scripted local greedy odor, scripted local odor+novelty visit memory, and legal random walker. All programmed policies have the same age/energy reproduction threshold (46 energy) and differ only in movement choice; RNN still chooses its own reproduce action.
- Baseline scripts ARE NOT EVOLVED and scripted genotypes have no causal effect on actions. Script reads age and novelty visitation history unavailable to the neural controller: strong feasibility reference, not equal-information learned-skill comparison.
- **16 new main ×400 ticks** worlds and **8 shifted ×400 ticks**. Primary: scripted novelty policy keeps population alive to tick 400 in ≥8/16 worlds AND reaches generation≥3 in ≥8/16. Necessary local-food-search benchmark: greedy odor must beat legal random on seed-paired food consumed with 95% descriptive bootstrap lower bound >0 and ≥12/16 worlds favorable.
- Extra four **no-reproduction** and four **no-food-renewal** worlds isolate physical life-cycle and resource flux. Check exact energy ledger, age death, no arbitrary births or food additions, non-overlapping organisms and full reproducibility.
- Publish per-world food/birth/survival, sensory diagnostic fractions, lineage events and population at fixed 50-tick intervals; no post-hoc parameter changes on final seeds.
- Programmed survival would imply the ecology can support at least a *hand-engineered* population, NOT that artificial selection has found an effective organism. No scripted survival suggests a deeper ecology/motor/food-information bottleneck and should guide separately preregistered M5.2 experiments.

### M5.1 sealed results and next scientific problem
- [Complete report](../reports/M5_1_ecological_viability_2026-11-21.md), [per-world main/shifted data](../reports/M5_1_paired_worlds_2026-11-21.csv), [original successful GitHub Actions #44913351669](https://github.com/kkongiu/neuroevolution-lab/actions/runs/44913351669).
- Frozen M5.0 physics, **16 completely new main worlds ×400 ticks**: original in-world inherited/mutated neural genotype **0/16 alive** and **21.38 births /54.94 food consumed** per world; externally scripted local greedy odor forager **16/16 alive**, **112.50 births /404.19 food**; scripted odor+visits **16/16 alive**, **112.63 births /408.13 food**; legal random movement with scripted birth threshold **10/16 alive**, **63.13 births /224.81 food**.
- Scripted novelty viability gate PASSES (16/16 alive and 16/16 at generation≥3; required 8/16 each). Local odor greedy−random food advantage **+179.38** /world, paired bootstrap descriptive 95% **+130.13/+228.88**, **16/16 world wins**: preregistered forage gate PASSES.
- **8 harder shifted worlds**: neural **0/8**, scripted greedy **7/8**, scripted greedy+visits **8/8**, legal random+scripted reproduction **1/8** alive.
- Neural-controller failure is robust: food saturates all 32 cells in every extinct neural world by tick400. At tick100 neural populations average just **1.31 alive with 31.94/32 food**, vs 9.06 greedy odor survivors and 10.12 greedy+visits survivors. Ecological laws CAN support persistence if **programmed** behavior is competent.
- CRITICAL: the scripted baseline also uses **scripted birth decision (threshold energy≥46) and age**, and novelty can track personal cell visits. Its advantage CANNOT be credited exclusively to food guidance. Scripted genotypes do not control behavior and do NOT demonstrate spontaneously evolved intelligence.
- All 4 no-reproduction controls extinct/no births; all 4 no-regrowth controls extinct; no energy ledger violations across 936 checkpoints. Bracketed original run failed solely at empty-population SVG serialization; replay after [PR #30](https://github.com/kkongiu/neuroevolution-lab/pull/30) succeeded **without any physics/seed/policy change**.
- **Next issue:** [M5.2 distinguish motor from reproduction and test low-dimensional evolvability](https://github.com/kkongiu/neuroevolution-lab/issues/31). First factorial interventions, THEN preregister new controllers with fewer parameters/sparse mutations and ecological persistence/heritable-gain gates on fresh worlds. Do not quietly replace neural behavior with hand-coded foraging and call it emergence.
- No in-world synaptic learning, autonomous rule generation, source modification, consciousness or open-ended reproduction implied.

## M4.x — Persistent objectives (speculative, not implemented)
Preregister observable persistence and revision tests. Keep evidence of goal-directed behavior strictly separate from subjective experience and autonomy claims.
Preregister observable persistence and revision tests. Keep evidence of goal-directed behavior strictly separate from subjective experience and autonomy claims.

## M5.2A — Causal 2×2 movement/reproduction diagnosis (completed; motor bottleneck established)

- [Protocol frozen before implementation](M5_2A_PROTOCOL.md), [full scientific report](../reports/M5_2A_causal_factorial_2026-11-21.md), [per-world archive](../reports/M5_2A_world_survival_births_2026-11-21.csv), [successful sealed 30-world Actions campaign](https://github.com/kkongiu/neuroevolution-lab/actions/runs/44920201061).
- Factorial arms: neural movement/neural reproduction (NN), neural/threshold (NT), scripted local odor/neural birth (GN), scripted/threshold (GT). Original M5.0 actor and M5.1 greedy baseline are preserved exactly; independent units are whole worlds.
- **Main-world survival at tick 400:** NN **0/20**, NT **1/20**, GN **20/20**, GT **20/20**. Motor-only intervention GN−NN **+100 percentage points** (paired descriptive bootstrap [100,100]; 20/20 world wins). Birth-only NT−NN +5 pp (interval [0,15]); **NOT confirmed**. Under harder shifted environments NN 0/10, NT 0/10, GN 9/10, GT 10/10.
- GN succeeded while birth decisions remained genome-controlled but its **motor search was hand-coded**. This is evidence for a motor/food-search bottleneck, **not inherited evolution of navigation**.
- Decision: M5.2B must pre-register a compact genuinely genotype-controlled motor+birth policy and compare sparse mutations / whole-genome mutations / exact cloning / random offspring on NEW independent worlds; require survival AND heritable gain AND causal genotype controls, not just scripted ecological feasibility.

## M5.2B — Compact neural heredity, sparse mutation vs controls (completed; primary negative)

- [Frozen M5.2B protocol](M5_2B_PROTOCOL.md), [report](../reports/M5_2B_compact_neural_2026-11-21.md), [per-world data](../reports/M5_2B_world_survival_births_2026-11-21.csv), [successful sealed workflow](https://github.com/kkongiu/neuroevolution-lab/actions/runs/44921004288).
- Unchanged M5.0 **fully neural** motor and reproduction with 2 hidden recurrent units / 51 numeric genome coefficients, instead of 12 / 401. Four newborn genetic operators: 3-weight sparse inheritance, all-weight mutation, clonal inheritance, independent random genotype. All movement and birth are genotype-dependent, unlike the successful scripted-arm M5.2A.
- **Full result:** 24 main ×4 arms ×400 ticks and 12 shifted ×4 completed, 138 tests pass, **zero surviving world in every single arm and split**. Sparse newborns avg 11.92 births /32.83 food units main; no generation-3 worlds main; random newborns avg 13.21 births /38.75 foods; all 32/32 food slots remain full when extinct.
- All primary heredity, viable population, mutation-over-clone and shifted transfer gates FAIL. This does not distinguish an inaccessible fitness landscape from insufficient representational capacity of a two-unit RNN.
- Next independent experiment should first establish **representational** foraging feasibility for a numeric controller, then independently test **evolutionary accessibility** without injecting hand-coded movement as an evolved genotype, on fresh sealed worlds. Keep negative findings.
