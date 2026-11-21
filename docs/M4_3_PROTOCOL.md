# M4.3 — Causal delayed recall, separated from locomotion

**Frozen preregistration:** commit this protocol before implementing M4.3 or inspecting any M4.3 final-seed results. Do not alter M4.0/M4.1/M4.2 source, protocol, genotypes or historical holdouts.

## Research question

M4.2 found an evolved RNN outperforming a near-size-matched feedforward (FF) network at collecting food in a T-corridor, but RNNs reached the junction more often, and clearing the RNN state did **not** conclusively damage final success. Can a recurrent circuit retain a **transient target-direction cue** when **all models are guaranteed to arrive at the identical binary decision**, with no motor-path failure or external memory?

This is a **programmed/evolved neural recall test**, not biological life, within-lifetime synaptic learning, goal formation, predator evasion or consciousness.

## Part A — pure delayed-recall bench (PRIMARY, no locomotion)

Use the same geometry generator `m4_2.make_pair` for paired goals in a 13×13 T-junction. For each world seed construct TWO worlds with **exactly the same walls, position at junction, legal branch actions, orientation, reflection and sensory inputs**, but opposite true food targets.

An evaluation has three phases, each presented to the neural model:
1. **Cue**: exactly two successive 19-dimensional sensory vectors at the initial corridor point, containing a single cardinal-direction cue bit indicating the correct branch. Other input fields are the same in the two goal variants.
2. **Delay**: exactly D successive 19-dimensional **neutral observations identical across both goals**: all cue bits zero, no food odor or adjacent food, constant energy input 1, constant collision/food feedback 0, and no absolute position or time. **Neural action outputs before the decision are discarded and have no effect on the next observation, energy or target.**
3. **Fork decision**: the exact same 19-dimensional vector for the two opposite goals, derived only from actual local blocked-neighbour flags at the fork; cue, odor, food and external clock all zero. Evaluate four action logits and use the **shared legal-action mask** to choose one of two possible branches (the approach direction is blocked **only at the readout panel**, to guarantee a binary choice, without changing either world's goal). Success=1 iff chosen branch is the target, otherwise 0. **No movement/energy is simulated in Part A**, and **100% of agents reach a decision**. Each paired world contributes the mean of the two opposite-target binary trials, never two independent experimental units.

Important analytic controls: with D>=1, a stateless FF network and a same-RNN genotype whose hidden state is reset **immediately before the final observation** must score **exactly 50% on each paired world**, because their final sensor inputs are literally identical for both targets. The same model with cue omitted must also score 50%. The handcrafted finite-state cue memory should score 100%. Unit tests must assert all these invariants before final experiments.

For D=0 in the **training-only** introductory curriculum, the target cue remains visible at the readout decision; this is a training scaffold, **not** the sealed primary benchmark.

## Models, fair search and frozen curriculum

- Reuse M4.2's numerical genome format and neural forward pass **without modifying M4.2**: Elman RNN **8 tanh hidden, 260 weights**; FF **11 tanh hidden, 268 weights** (+3.1%, to the FF's advantage). Four output logits; the same two-legal-branch mask at the decision.
- **8 independent genetic population seeds 90000–90007**, each evolving **separate RNN and FF populations**. Numeric weights normal SD **0.45**, independent Gaussian mutation SD **0.28**, 14 candidates/generation, 2 unchanged elites and 2 random immigrants.
- **8 generations**, **8 fresh training maze seeds per generation**, every maze always evaluated with both opposing goals AND exactly three curriculum delays:
  - generations 0–1: delays **(0,1,2)**;
  - generations 2–3: delays **(1,2,4)**;
  - generations 4–7: delays **(2,4,6)**.
- Exactly **8 generations ×14 candidates ×8 world pairs ×2 target sides ×3 delays = 5,376 candidate-trials per architecture and genetic seed**, identical budgets; **86,016 training trials** across both architectures and eight genetic replicates. No test-seed data enter evolution.
- Fixed numeric **training fitness per trial:** **0.85×binary_success +0.15×smooth_correct_branch_preference**, with smooth preference = `1/(1+exp(-min(25,max(-25,correct_logit-other_branch_logit))))`. True target is visible only to the **external fitness evaluator**, not the network at inference. Continuous margin term assists evolutionary search without any gradient updates/SGD. All models get exactly the same objective.
- Training world pair seeds **60000–60063** (8 per generation, no reuse). Fixed validation pair seeds **61000–61007**. For each generation evaluate its best training genome on the **same three validation delays (2,4,6)**, both paired target sides. Select **one** champion per architecture/genetic seed using mean validation fitness; tie-break earlier generation. All 16 champions freeze before any final test.
- Genetic RNG streams are deterministic and separate from world generation. No post-hoc change to population, mutation, training fitness, delays, metrics or champion selection.

## Sealed evaluation

- **Main primary**: **64 unseen paired world seeds 62000–62063**. Assign delay **2,4,6** using `[2,4,6][world_seed % 3]`. Frozen architectures, no retraining.
- **Long-delay OOD**: **32 new paired seeds 63000–63031**, delay **8,9,10** by `[8,9,10][world_seed % 3]`, no retuning. Still same orientation/reflection distribution.
- **Guided navigation transfer**: **16 further paired spatial worlds 64000–64015**. Programmatically transport each selected policy down the M4.2 corridor while delivering actual position-specific local-wall observations and initial cue. The controller **cannot choose the locomotor steps**; only its final fork choice counts. Compare this guided arrival with separate **fully free autonomous corridor movement** with the same frozen genome, to expose the motor/navigation gap. This is secondary exploratory transfer, NOT substituted for the pure-memory main.
- Exact same worlds, opposing goals, target cue and FF/RNN fit budgets for all trained arms; varying world orientation cannot give the final input the answer.

## Fixed arms and causal interventions

1. **rnn**: evolved recurrent genotype, state carried through cue and neutral delay.
2. **ff**: independently evolved 11-hidden feedforward genotype, same training and validation budget.
3. **rnn_reset**: **same selected RNN genotype**, state zeroed immediately BEFORE the final decision; not a separate trained model.
4. **rnn_flipped_cue**: same RNN, initial cue changed to point to the **opposite** branch, actual reward remains unchanged.
5. **rnn_no_cue**: same RNN, all cue bits zeroed, identical neutral observations/decision.
6. **random_binary**: a uniformly sampled legal branch using an RNG shared between the two opposite goal variants, guaranteeing a paired mean exactly **50%**.
7. **scripted_one_bit**: stores initial cue direction in an explicit scalar memory and chooses it at decision. A deliberately privileged **programmed memory benchmark**, success **100%**. It does not read the hidden reward label directly.

Do not treat more observations, higher compute, or a scripted memory reference as evidence of spontaneous intelligence.

## Preregistered statistical success criteria

**Primary architecture contrast:** at main heldout, paired mean binary success **rnn − ff**. Use **hierarchical paired bootstrap** resampling whole **8 independent genetic starts** AND whole **64 paired world seeds**, independently, preserving paired goal sides and the same worlds for every arm. Report descriptive percentile 95% range and per-run differences. Require mean >0, lower CI >0, **≥6/8 positive independent genetic runs**, and absolute mean RNN success **>55%**.

**Necessary causal memory contrast:** the **same RNN minus rnn_reset** must independently satisfy mean >0, lower hierarchical CI >0, and **≥6/8 positive genetic runs**, with no re-selection based on final seed. **Necessary cue-control:** same RNN minus rnn_flipped_cue must satisfy mean >0 and lower CI >0. All **three** gates must pass to claim useful causal delayed cue-memory on the bounded task. The reset-and-FF benchmark is mathematically 50%; a positive interval alone does not prove general task understanding.

Additional diagnostic comparison rnn minus rnn_no_cue, paired binary random and scripted reference. Report training/validation curves, mean success by delay, dependency on cue, every reset/cue intervention and genetic-run variability.

Long-delay shifted test reports attenuation and fresh-seed hierarchical intervals **without changing thresholds or models**. Guided/free corridor results measure navigation vs recall, with fork arrival reported separately for free movement. A main-only gain cannot establish robust generalization.

**Interpretation:** even if successful, this would show *externally evolved* short-term hidden-state retention of an explicitly provided symbol. It is NOT synaptic learning during lifetime, long-term map construction, automatic reward discovery, predation, food seeking or independent life.

## Tests and reproducibility

Tests must verify every paired final observation is **identical** for both opposite target goals (including energy/clock); D>=1 FF, reset, no-cue and seeded paired random exactly 0.5 per seed; scripted exactly 1; correct readout only has two legal branches, no path failure; zero prior-network outputs influence readout observations; contrast deterministic over repeated execution; train/validation/final seed splits disjoint and frozen; curriculum and candidate-world budgets exact; no M4.2 module changes.

An explicit GitHub Actions campaign saves settings, all champion numerical weights, per-generation history, world/genetic/delay and all per-arm scores, hierarchical bootstrap summaries, and example paired diagrams. Publish null and negative findings with equal prominence.
