# M5.32 — frozen DEV protocol: two continuously plastic nonlinear synaptic units

**Date: 2026-11-22. This document precedes the first full M5.32 GitHub Actions science campaign.**

## Purpose

M5.31's *soft one-neuron* network generalized to triple-input interactions
better than its hard-pruned representation (82.64% vs 77.60%), yet neither
solved XOR of two disjoint ANDs (42.88% soft accuracy). The next minimal
structural test permits **two trainable soft conjunction units** whose
sensory gates, strengths and a common five-weight product module learn
only from the eight actions and binary rewards the task episode actually
produced. The shared conjunction and multiplicative XOR-style output
operator are still fixed by researchers: this is **not** autonomous
invention of new mathematical or neural operations.

A candidate with two independently plastic units is compared with an
otherwise identically initialized, same-restart/same-steps dual unit
**without the diversity penalty**. This is the strict compute- and
parameter-matched control for the value of diversity-induced
specialization. Other controls (one unit, original M5.30 menu, product
alone) are scientific benchmarks **not** exact matched-compute
comparisons; avoid claiming a compute efficiency advantage from them.
Also report scrambled feedback and a post-fit random-unit ablation
control; learning must be reward dependent.

## Locked algorithm

- Each of two units has exactly four sigmoid-free **clipped [0,1]**
  continuous input gates and a scalar clipped [0,1] residual strength.
  Both project a HUMAN-DEFINED four-sensor conjunction
  \`h_j=prod_k (1-g_jk*(1-x_k))\`. They contribute
  \`(1-2*a_1*h_1)*(1-2*a_2*h_2)\` to the HUMAN-DEFINED
  five-weight product input network. This form can express a parity rule
  with XOR of two distinct conjunction interactions.
- Optimize 15 task-specific float64 scalars per initialization (5 base
  weights, 2 × [4 gates + strength]) by **300** full-batch support-only
  steps, across **eight** deterministic initializations. No learned
  shared neural plasticity, active exploration or prior fitting across
  task seeds. Clips: base weights to [-1,+1], gates/strengths to [0,1].
- Step sizes: 0.30 base, 0.09 gates, 0.08 strengths; L1-style
  penalty 0.01 per strength in both arms.
- Primary dual-neuron candidate uses diversity penalty
  \`0.025 * mean_k(g1k*g2k)\`; strict matched control has coefficient
  exactly zero, with all other optimizer parameters, initializations,
  fit work, architecture, data and numerical type identical.
- Select the best restart **only by support MSE plus relevant support
  penalties**, never exam cues. Final primary prediction uses both
  continuously plastic units; no posttraining hard-pruning or oracle
  topology choice. This deliberately tests the M5.31 pruning failure.
- Both ablations are **post-fit, zero-retraining**: replace separately
  each unit's multiplicative correction by identity 1. Grade only on
  unseen cues *after training*. For every context record change in
  reward accuracy, per-unit contribution, gate overlap, and whether two
  different units genuinely make different causal contributions.
- At inference persistent code is **15×32=480 float32 bits** per
  A/B/C context (1440 bits overall); cast the trained float64 values
  to float32 before scoring. This excludes eight 15-float64 optimizer
  trajectories, gradients, support history, supplied operators,
  computation code and all transient workspace. Not a whole-brain
  memory or energy efficiency result. Single M5.31 soft baseline
  requires 10×32=320 bits/context, and uses 16 instead of 8 restarts.

## Frozen synthetic development split

To permit fair **same-task paired comparisons**, use the exact 52
M5.31 synthetic task seeds and A/B/C rules, actions, and eight unseen
cues from the **already examined M5.31 DEV family**. This is a
**reuse of seen family and seeds** after studying M5.31 results, NOT
fresh independently held-out confirmation. Model-restart seeds:
**977, 983, 991**. All M5.32 candidate and matched control arms see
identical own-action and real own-binary-reward records, plus the
same eight support cue IDs and the same eight unseen cue IDs per
context (not their labels). Explicitly record the absence of leaks.
The original M5.31 DEV scores are not reused as an externally
independent test set.

Cohorts:
1. 12 simple parity tasks.
2. 16 parity + pair-AND tasks.
3. 16 parity + triple-AND tasks.
4. 8 parity + XOR of TWO DISJOINT pair-ANDs, the original
   one-neuron **out-of-class** stress family.

Total **156 row trials, 468 A/B/C contexts, 3744 query evaluations per
arm**. The eight support actions remain chosen at random by the
TEST HARNESS, not adaptively by an organism. The original M5.31
small pilot seeds 8000/9000/10000 were already used in its design
and remain excluded from the M5.32 campaign as well.

Seven arms on the identical episodes:
- **two_soft_diversity**: two learned soft conjunctions plus
  diversity penalty (treatment).
- **two_soft_no_diversity**: identical 15-parameter dual architecture,
  identical eight starts and 300 steps but no diversity penalty
  (**strict matched control**, not retrained post-hoc).
- **two_soft_scrambled_rewards**: exact same treatment algorithm
  fed randomized support rewards, frozen actual exam.
- **one_soft_m531**: one 10-parameter M5.31 soft neuron, 16 × 300
  steps; strong accuracy benchmark but not matched model capacity.
- **fixed_product**: original five-weight M5.29 reward-adapted
  product baseline.
- **m530_fixed_operator_menu**: original seven whole-program
  operator candidates, only a comparative baseline.
- **privileged_true_two_masks**: explicitly UNFAIR diagnostic
  reference with the two researcher-given true masks and
  chosen support-trained base product; no true masks EVER enter
  the trainable candidate, matched control or score selection.
  For tasks with fewer than two interactions, missing true masks
  are empty / identity.

Additionally record both **zero-retraining ablations** of the
primary candidate for each context and a **randomly selected
unit-ablation** control, so two-unit utility and causal
specialization are independently inspectable.

## Prespecified screen (not automatic evolutionary authorization)

1. Two-disjoint-pair accuracy of primary dual network >=55%.
2. Two-disjoint-pair primary dual accuracy >=one-soft-M5.31 +8pp.
3. Two-disjoint-pair primary dual accuracy >=fixed product +10pp.
4. Primary dual accuracy on parity >=90%.
5. Primary dual accuracy on triple-interaction >=75%.
6. Two-disjoint-pair **strictly matched** comparison: diversity
   regularizer performance >=no-diversity performance -3pp,
   so specialization penalty cannot buy apparent structure by
   catastrophic accuracy losses.
7. Within two-pair cohort, at least 10% of contexts show **both**
   frozen ablations drop exam accuracy >=1/8 each. This is a strict
   causal, per-neuron contribution screen, not evidence of
   physiological biological specialization.
8. Primary dual on two-pair family beats reward scrambling
   by >=5pp.
9. Exact 8+8 disjoint support/exam firewall and no true rule
   passed to optimizer; synthetic environment-only truth access.
10. 3 task-context persistent float32 snapshots total 1440
    bits each row, verified on actual dtype and byte count.
11. Unsuccessful out-of-class interaction and all ablation
    failures reported regardless of favorable averages.

No expectation of all gates passing is baked into the implementation.
Results may be negative. No retroactive threshold changes, secret
test tuning, confirmation label or unplanned evolution. Exploratory
paired CIs are for uncertainty context only. No claims about
independent family replication, plastic general intelligence,
self-created neural primitives, catastrophic forgetting resilience
or ecology, because this experiment cannot establish them.

Whatever the outcome:
\`autonomously_invented_neural_operator=false\`,
\`shared_meta_plastic_controller_learned=false\`,
\`independent_task_family_replication_complete=false\`,
\`continual_memory_retention_verified=false\`,
\`evolution_authorized=false\`,
\`advance_to_evolution=false\`.

Original M5.6 sealed, no merge to \`main\`. New draft PR stacked
on M5.31.
