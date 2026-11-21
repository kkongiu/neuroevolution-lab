# M4.2 — Delayed-cue spatial memory versus matched feedforward control

**FROZEN protocol.** Committed before M4.2 implementation, any evolutionary search, and any new final-world results. Preserve M4.0/M4.1 immutable.

## Scientific question

Can a recurrent neural organism **retain a one-time spatial cue across several locomotor steps**, choose a correct branch at a fork, and outperform a comparable memoryless feedforward network trained with exactly the same number of genotype/world evaluations? Does erasing its recurrent state **at the fork** causally remove its advantage?

This isolates *within-lifetime neural state memory*, not within-lifetime weight learning. No subjective awareness, new self-generated objectives, predators or in-world reproduction are claimed.

## A: delayed-cue, symmetrical fork task (primary)

- Deterministic 13×13 **T corridor** with impassable walls, initial energy 24, one-unit movement cost, 0.30 surcharge on wall collisions, legal-action decoder from the four immediate wall flags, and maximum 20 actions. Each paired world has the same geometry, start and initial energy for *two opposing target directions*.
- Episode starts in a one-cell-wide approach corridor **3–5 steps before a fork**; the fork has a left branch and right branch. Orientation is randomized among four rotations and optionally reflected per world seed. A starting cue consisting of four **absolute cardinal direction bits** indicates which branch contains one unit of food. The cue is available on **the first two action observations only**, then becomes exactly zero; the cue never reappears.
- The branches are **one-way commitment gates**: an agent moving from the central fork into either branch terminates the trial. It obtains one food pellet *if and only if* the chosen branch was the signaled target, otherwise zero. This explicit simplified gate isolates remembered direction at the decision and avoids confounding later branch traversal or scent following. The organism moves itself through the approach corridor: no scripted transport.
- Both branch goals are instantiated for each seed with **identical geometry/start** and opposite ephemeral cues. Thus if they reach the fork at the same time, present-time observations there are identical; memoryless inference cannot use an after-cue visual difference, and a fixed choice cannot be correct on both paired variants.
- Observation shape: original M4 spatial sensors 15 (4 adjacent blocked, 4 food odor, 4 adjacent-food, energy, prior collision, prior food) **plus four one-hot cue bits**, 19 inputs total. In delayed-cue worlds, odor and adjacent-food inputs are always zero until the trial ends: no target information leaks from those channels; no absolute x,y coordinates or world seed enter the network.
- Junction success is measured as **mean binary food success across both opposing variants**, over a *world pair*, not the individual side as an independent trial. Other diagnostics: fraction reaching fork, correct branch conditional on reaching, steps, corrected illegal preferences, episode survival and unique visited cells. A failed approach does not receive full credit.
- Training fitness **fixed before final testing**: 12×success + 4×(correct fork branch) + 0.15×maximum progress along the approach corridor − 0.03×actions − 0.08×blocked-action preferences. This shaped fitness can help solve the approach, but is an objective specified by the experimenter, not evidence of intrinsic goals.

## B: open reactive food-search task (separate diagnostic)

- Reuse M4.1's randomized-start and rotated 13×13 food mazes, same legal action decoder, full odor signal, food/energy/horizon mechanics. Feed the original 15 sensors plus four **always-zero cue bits** to the **frozen M4.2 champions**; no retraining on reactive worlds. Compare greedy odor, wall-avoiding random, and **same recurrent champion with odor set to zero**. This is a **transfer diagnostic only**; no claims about effective reactive odor navigation unless the within-genome odor ablation worsens.
- Seeds **58000–58011** (12 new heldout reactive worlds), distinct from all training, validation, delayed main and shift worlds.

## Genomes and evolution

- RNN: one tanh Elman hidden layer of **8 recurrent units**, 19 inputs and four output logits = **260 numeric weights**; hidden state reset at the beginning of each episode only.
- Feedforward: one tanh hidden layer of **11 stateless units**, same 19 inputs, four output logits = **268 weights** (3.1% MORE than RNN, a conservative near-matched capacity). FF recomputes hidden from the current sensory vector; no state, clock, position or agent ID. Identical deterministic argmax and legal-action decoder.
- All weights initially drawn normal(0,0.45); mutation normal(0,0.28); fixed-topology. Each genetic replicate creates initial populations independently from the same RNG seed for each architecture (individual gene lengths differ). All agent training and validation worlds are paired and identical across RNN/FF comparisons.
- Eight independent genetic seeds **85000–85007**. Per seed AND architecture, exactly **6 generations × 12 genomes × 10 paired training worlds/generation × two target sides** = 1,440 training episodes, 11,520 × 2 architectures = **23,040 training agent-world episodes** overall.
- Each generation retains two elites, mutates offspring and inserts two random immigrants. The best *training* genotype of that generation is evaluated on six independent paired validation worlds; the champion for each run/architecture is the generation leader with best mean validation fitness. No test world influences selection.
- Training pairs: **55000–55059**, distinct 10 pairs per generation. Validation: **56000–56005** (same six pairs for every generation). 
- Sealed **main final pairs 57000–57031** (32; approach length 3–5, orientation/reflection randomized).
- Sealed **shifted final pairs 57500–57515** (16; approach length **5–6** so cue-to-fork delay increases; no retraining or rewiring).
- Selected eight RNN and eight FF champions are **all** evaluated on every heldout pair; none is selected using final data.

## Predefined arms on identical paired worlds

1. **evolved_rnn**: RNN evolved with cue; state maintained.
2. **evolved_ff**: feedforward evolved independently, same worlds and evaluation budget.
3. **rnn_reset_at_fork**: **same RNN genome**, clear recurrent hidden values only on the FIRST arrival at the fork, just before deciding; not a retrained or separate model.
4. **rnn_shuffled_cue**: **same RNN genome**, flip the provided initial cue to the opposite paired target while keeping actual branch reward unchanged; no retraining.
5. **legal_random**: uniform random legal move with a predeclared independent RNG stream, no cue.
6. **scripted_memory**: simple finite-state program storing the initial cue and its initial corridor heading, then walking forward and choosing the stored absolute target direction at the fork; an upper reference, not a learned policy.
7. **rnn_no_cue**: same RNN with initial cue channels zeroed, no retraining.

The same legal motor decoder applies to RNN, FF and the scripted controller; random chooses locally legal moves; no hidden map/food oracle is available to the learned policies. Scripted controller reads the supplied initial cue and locally blocked moves but not the hidden true goal directly.

## Frozen statistical gates

**PREREGISTERED PRIMARY:** (evolved_rnn − evolved_ff) **paired food-success fraction** across 8 genetic seed replicates × 32 world pairs. Use a **hierarchical paired bootstrap**, resampling the 8 independent genetic replicates and 32 shared world-pair seeds **separately**, and reporting a descriptive percentile 95% interval and per-genetic-run differences. Require mean >0, lower interval >0, and ≥6/8 independent genetic champion pairs with positive mean difference. **A positive effect is only memory evidence if fork-state-reset ablation also passes.**

**Necessary secondary:** evolved_rnn − rnn_reset_at_fork must show positive mean with lower hierarchical percentile bound >0. The state reset is a causal intervention on the same selected genome, not a proof of general intelligence. A second necessary information check is evolved_rnn − rnn_shuffled_cue positive mean (with descriptive interval); without it a success could simply be a fixed directional reflex.

Report separately **scripted FSM upper reference** and legal random controls, and RNN versus RNN-no-cue. The FF 50% bound is conditional on reaching the junction identically for both paired goal conditions; do not assert unconditional 50% if different histories/failed reaches alter outcomes. On paired worlds with same geometry, FF cannot infer target side from its currently identical fork observation.

Transfer to shifted longer delays is **secondary** and does not substitute for main. Report whether gap worsens as delay grows, plus reactive open-maze behavior and odor/no-odor transfer diagnostic. No re-optimization of hyperparameters on final seeds. All null and negative outcomes receive equal prominence.

## Tests and limits

Assert paired geometry identity, opposing cue/food truth, fork current observation equality after cue turns off (for identical positions/energy/history), no scent/adjacent-food leakage, deterministic reproducibility, legal movement, exact gate reward and termination, selection independence from final seeds, equality of genotype-evaluation budgets and bootstrap independent sampling of genetic/world dimensions. All agents must be able to reach the fork under legal moves; the scripted program should succeed on both variants.

The simple one-way fork is **not an ecosystem**: it provides a clean, limited causal memory test before increasing ecological complexity. The cue is an externally supplied symbolic signal and 8 genetic initializations are still a small sample. Recurrent state may preserve the cue, but synaptic weights do not change within a lifetime; no self-chosen objectives, in-world reproduction, predators, or autonomous source modifications.
