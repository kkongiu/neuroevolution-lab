# M5.30 — Prospectively declared own-feedback operator-selection experiment

**Protocol recorded before running the M5.30 GitHub Actions experiment.** This
is a limited candidate-selection experiment; **not** autonomous neural discovery
of a compositional grammar, ecological survival, or independent replication.

## Scientific question

M5.29 generalizes 97.22% across new complete parity functions, but only because
the researcher gave the network exactly the right multiplication operator. Can
the learner use **only eight personally experienced chosen-action/reward events**
to select among a *fixed researcher-defined menu* of competing compositions,
without consulting unseen exam answers, and perform better than forcing the
parity operator in an altered rule family?

The seven pre-programmed choices are an ordinary product network and the
same product combined with one of six possible pairwise AND sign flips. Each
choice trains an equally sized **five-float** task neural code using the same
M5.29 SGD, 8 restarts × 180 steps, with eight support examples. The controller
chooses the smallest support MSE + 0.025 complexity penalty for non-plain
operators. There is **no cross-task learned executive** and no transfer of
candidate weights: selection is engineered, optimization is real.

A/B/C remains explicitly visible, as in M5.29; A/B have opposite rules and
C differs. Evaluation uses whole-rule masks 5,9,11,14; these are *already exposed*
in the M5.29 development experiment, so this is another **new-seed
development screen**, not a sealed or independent task-family replication.
All new cue exams remain disjoint from the eight support cues.

## Prespecified simulation cohorts and fair controls

- Affine parity (16 seeds 7100–7115): the correct fixed operator is plain
  product, so unnecessary pairwise-selection should be penalized.
- Parity XOR one randomly selected pairwise AND (16 seeds 7200–7215):
  the pair differs by simulated task/context. Any of six exact proposed
  compositions could match. No true pair is supplied to the selecting agent.
- Parity XOR a three-way AND (8 seeds 7300–7307): **not expressible exactly**
  by the supplied one-pair menu; stress tests restricted representational scope.

Each cohort repeats with three optimizer-initialization seeds 817,827,829.
Every task gets 8 personally experienced chosen-action/reward supports,
8 never-seen cue exams. Each context runs six arm scorings: own-feedback
selected operator, fixed product, random operator with real-reward training,
same selection after randomized rewards, **privileged** correct-operator
reference (evaluation only), and unadapted product. The privileged reference
falls back to plain product for outside-menu three-way tasks, because no
candidate can be declared exactly correct there.

The ordinary candidate cannot read target masks, family labels, true pair
operator, correct unchosen action, or evaluation outcomes. The simulator
uses them **only** to generate physical feedback and later grade exam accuracy.
The learner can logically derive a label from its chosen action/reward because
this artificial world has precisely two deterministic complementary actions;
this is an explicit simplifying prior.

**Prespecified narrow development gates (not post hoc):**
1. Pairwise nonaffine unseen mean accuracy >=70%.
2. Gain over fixed product in pairwise family >=8 percentage points.
3. Gain over randomized-reward selection in pairwise family >=15pp.
4. No eight-cue train/exam overlap.
5. True operator not passed to learner.
6. Exactly 489 persistent task-code bits for three contexts (5×float32 +
   3-bit operator index for each).
7. Outside-menu three-way stress recorded (regardless of whether successful).

The primary result is **accuracy on frozen unseen cues** in the pairwise
family, along with all failures and uncertainty. Three optimizer seeds and
small synthetic cohorts give exploratory intervals only. Two tested arms
have substantially unequal compute versus simple baselines, and the entire
seven-candidate plus optimizer training state, interpreter and 8 supports are
excluded from the 489-bit persistent *final code* accounting. Do not use
that figure as a total memory or energy measure.

## Required scientific decision

Even if all narrow development gates pass: report only that the organism
**selected between supplied hypothesis operators from experience**.
Set \`autonomously_discovered_composition_operator=false\`,
\`learned_shared_neural_executive=false\`,
\`independent_task_family_replication_complete=false\`,
\`evolution_authorized=false\`, \`advance_to_evolution=false\`.
Failure is a valid, permanently archived negative result. Do **not** merge
into \`main\`, unlock the original sealed M5.6 datasets, or start ecological
evolution. The next scientific stage would have to **learn new operators**
across genuinely different task families with independently replicated
holdouts.
