# M5.34 — FROZEN protocol: a SHARED learned representation, not an oracle-installed task solution

**23 November 2026. This prospective development protocol precedes the
first M5.34 full GitHub Actions science evaluation.**

## Motivation, testable question, epistemic firewall

M5.33 showed a paradox: with eight actively chosen support cues, an
engineered human-specified **112-function symbolic hypothesis grammar**
uniquely identified all test functions, but the independently trained
15-parameter M5.32 dual-soft neural model learned poorly on the eight
unobserved cues. Prior family-aware information acquisition alone
does not induce usable neural representation.

**Question:** Can a reusable, gradient-trained *neural representation
learned across distinct earlier functions* improve eight-support
generalization to functions whose full 16-bit truth tables have NEVER
appeared in cross-task neural training, without passing the test
function ID or symbolic posterior into the learner?

This tests **amortized learned conditional prediction**: a shared,
ordinary MLP reads the 16-dimensional partial cue observation table
and 16-dimensional observation mask, then predicts outputs for 16
cues. The MLP is *pretrained on DIFFERENT, researcher-generated
function truth tables*. Its hidden representation is reused/frozen
across test functions; no optimizer/weight adaptation is performed
on heldout functions. This is NOT an organism that learns its own
neuron semantics, decides when to investigate, or continuously
learns without forgetting. It has a researcher-engineered
fixed cue one-hot indexing scheme and a fixed 32→64→16 network.

### Exactly disjoint function-level split — NOT independent family-level

Reuse the complete M5.33 **112 unique 16-bit truth tables** generated
by the known artificial task family (4 affine masks × 2 biases ×
one of 14 effects = 112). Split *whole truth tables* BEFORE neural
training, within each of four family strata using a fixed
\`numpy.default_rng(3401)\` shuffle:

- parity: 8 functions, **1 whole function held out**
- one pairwise AND: 48, **12 held out**
- one triple AND: 32, **8 held out**
- XOR of two disjoint pairwise ANDs: 24, **6 held out**

**85 TRAIN functions vs 27 completely UNSEEN-function evaluation
truth tables**, no literal 16-bit truth signature overlap. This is
a **new function-level within-known-family split**, not independent
*family*-level confirmation. Same parity, AND/XOR *generator* and
four raw sensory bits are shared by training and test. No test
signature or its full target truth vector enters training,
optimizer early stopping or model selection.

Each of 27 heldout functions has **3 preregistered independently seeded
eight-cue episodes**, each evaluated under:
- **random** sensory acquisition (2 shared starting cues + 6 randomly
  selected remaining cues);
- **engineered active 112-hypothesis version-space acquisition**
  (2 shared starting cues + 6 entropy-selected cues using previous
  *actual* action/reward observations). This active planner knows
  the full **human-supplied family catalog** including possible
  test functions, but never the current target ID. It is
  independent of the neural model and NOT counted as learned
  neural curiosity.

Two neural pretraining initializations **1301 and 1303**
(2×27×3×2 = **324 paired evaluation rows**, 162 heldout
episodes per sampling arm). In each episode precisely eight
genuine chosen-action/observed-reward support triples, eight
strictly disjoint frozen unseen cue exam IDs, and the same
observations for ALL compared learners. For any given cue
the actual chosen action is a precommitted 0/1 action, and
the observed success only reveals the corresponding result
under the deterministic binary-complement assumption.

### Neural training contract, hyperparameters frozen

The candidate is a *shared* two-layer MLP, 32-dimensional
partial-observation input (16 labels at observed cue IDs,
zero elsewhere; plus 16 support-mask indicators), **64 tanh
hidden features**, 16 tanh output scores. About
3,152 trainable float32-like scalars, stored as 32-bit floats.
No parity features, AND gates, rule IDs, operator menu,
known symbolic posterior labels, or current test answers
are input features.

For each pretraining initialization: **3,000 Adam steps,
batch 32 episodes, learning rate 0.003**, fixed Adam
\`beta1=0.9, beta2=0.999, eps=1e-8\`; cross-task batch
draws only from the **85 training functions**, with fresh
random 8-of-16 support cue masks. The 16 full truth labels
for TRAIN functions are used to compute squared-error loss
**only on the eight withheld query cues of those TRAIN
episodes**. This is valid cross-task supervised learning,
but requires an explicit RESEARCHER SUPPLIED task
distribution with complete training labels and may
constitute a large supervised training budget.
No test truth table, test function membership, test
cue outcome or test episode used for tuning.

At evaluation, the 3,152-parameter neural representation
is **frozen**: it receives only the eight true chosen
action/reward support observations and the observed mask.
No new per-test-function gradient steps or rule labels.
Output at observed support cue IDs is clamped to the
logical actual support label (deterministic binary action);
output on the eight unseen cues is genuinely predicted.

### Controls and comparisons

All receive IDENTICAL supports and exam cues for each heldout episode:

1. **shared_meta_representation**: MLP honestly pretrained on
   85 training functions, frozen at test.
2. **scrambled_meta_pretraining**: SAME MLP, initialization,
   Adam hyperparameters, 3000 steps, batch size and training
   function distribution, but all 16 query targets in each
   TRAIN episode come from an **independently sampled
   different TRAIN function**. This deliberately destroys the
   support→query relation; the test function is not revealed.
3. **untrained_same_network**: identical starting weights,
   no cross-task pretraining. Parameter-matched but not
   training-compute-matched; not a matched compute claim.
4. **m532_independent_dual_soft**: M5.32 15-parameter two-soft-
   neuron gradient fit separately from eight real supports
   (8 restarts ×300 updates) and diversity OFF. Strong
   within-lifetime reference; **NOT** compute- or
   parameter-matched to the pretrained 3,152-weight MLP.
5. **human_privileged_exact_grammar**: offline 112-function
   posterior majority vote using correct human-known
   family. This reference is deliberately **UNFAIR**:
   it may explicitly enumerate the correct heldout truth
   table as a possible model and must not be called
   autonomous neural understanding.

Also store per-context explicit support8 accuracy,
unseen8 accuracy, and all16 accuracy, plus learned
hidden activation norms as diagnostics. Compare active
vs random *separately* to avoid claiming sampling causes
a gain without observing one.

**Compute honesty:** the meta neural model benefits from
\`3000 × 32 = 96,000\` separate **full-target TRAIN
function episodes** per initialization, vs within-task
dual-soft adaptation on just eight *test* observations.
Representation reuse and resource efficiency must
therefore be measured separately; there is NO matched
total-exposure or energy comparison.

## Exact prespecified narrow gates

1. **two-pair** unseen8 meta accuracy >=60% (pooled two sampling arms).
2. **two-pair** meta unseen8 vs frozen scrambled-meta >=+8pp.
3. **two-pair** meta unseen8 vs untrained same network >=+8pp.
4. **two-pair** meta unseen8 vs M5.32 dual-soft >=+8pp.
5. Meta unseen8 **mean across four family cohorts** >=70%.
6. Meta unseen8 on pair-AND >=70%.
7. Meta unseen8 on triple-AND >=75%.
8. **All 85 training truth signatures disjoint from all 27
   evaluation truth signatures** (exact bytes, no IDs shared).
9. 8 chosen-action/reward support records + eight strictly
   disjoint never-queried test cues, same between learner arms.
10. True test function absent from MLP meta pretraining,
    explicit grammar only in the tagged selection/oracle
    arms; scrambled meta training has identical budget.
11. Record ALL cohorts, both sampling schemes, all arms,
    complete raw observations, model checkpoints/digest,
    negative results and fixed no-evolution flags.

These are **local development screening checks**, NOT claims
of learned independent task-family transfer or life.
Each threshold must be marked FAIL if missed. Exploratory
intervals should cluster over heldout *function identity*;
repeated cue episodes/optimizer initializations are NOT
independent biological samples. Avoid post-hoc retuning
with heldout functions: changes require a new protocol
and a completely new evaluation split.

No claim of:
- an autonomous learned curiosity policy;
- an invented conjunction/parity operator;
- shared plasticity/executive adapting during a test;
- lifelong memory retained without forgetting;
- independent real-world or task-family validation;
- general intelligence or autonomous evolutionary fitness.

The sealed M5.6 ecology is not accessed and \`main\` is
untouched. M5.34 is a stacked draft on M5.33. Regardless
of results:
\`learned_general_curiosity_policy=false\`,
\`autonomously_invented_neural_operator=false\`,
\`shared_meta_plastic_controller_learned=false\`,
\`independent_task_family_replication_complete=false\`,
\`continual_memory_retention_verified=false\`,
\`evolution_authorized=false\`, \`advance_to_evolution=false\`.
