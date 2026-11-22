# M5.32 — verified controlled DEV result: two plastic units show limited causal diversity but FAIL the primary learning screen

**2026-11-22 | GitHub Actions [#45072966328](https://github.com/kkongiu/neuroevolution-lab/actions/runs/45072966328) SUCCESS.**
All M5.32 analytic gradient / firewall tests, prior M5.31 tests, and
original M5.6 online-learning regression checks passed, followed by
the complete science run. **Scientific stop/go is NEGATIVE: 8/11
prespecified checks pass and 3 crucial primary checks FAIL.** A
successful CI run is not a successful learning hypothesis.

The protocol was [frozen before CI]
(M5_32_DUAL_SOFT_SYNAPSES_PROTOCOL_2026-11-22.md) and uses exactly
the ALREADY STUDIED M5.31 52 DEV synthetic tasks, not independent
held-out replication. Model initialization seeds 977, 983 and 991;
**156 row trials × three observable A/B/C contexts = 468 learned
context-specific networks**, each seeing 8 own randomized
chosen-action / observed binary reward experiences and 8 disjoint
unseen sensory cues (**3,744 unseen grades per arm**).
The environment's ground-truth interactions never enter candidate,
matched no-diversity, or scrambled optimizer signatures; exam
outcomes are never training data.

## Actual complete-arm unseen accuracy

| Arm | Plain parity | Single pair AND | Single triple AND | TWO disjoint pair ANDs |
|---|---:|---:|---:|---:|
| **Two soft units, trainable gates + diversity penalty** | **97.92%** | 75.17% | 80.03% | **50.52%** |
| Two soft units, exact matched optimizer, NO diversity penalty | 96.76% | 72.66% | **85.94%** | 46.18% |
| Two soft units, randomized support rewards | 50.46% | 48.61% | 51.39% | **52.08%** |
| M5.31 one soft conjunction unit | **100.00%** | 75.00% | 81.60% | 43.23% |
| M5.29 fixed product-only network | 97.92% | 56.94% | 68.58% | 37.85% |
| M5.30 seven supplied complete operators | 98.84% | **84.72%** | 71.61% | 41.32% |
| Privileged researcher-given TRUE two interaction masks | 97.92% | **99.48%** | **98.96%** | **94.79%** |

**The last row is a deliberately unfair structural-knowledge upper
reference** and must never be described as an autonomously discovered
solution. Both nonlinear units, conjunction primitive and multiplicative
(XOR-like) combination rule in the candidate were human supplied.

### Most informative pairwise differences

On **two independent pairwise-AND interactions**:
- Primary two-unit vs single-unit re-evaluated on the SAME contexts:
  **+7.29 percentage points**, exploratory seed-cluster paired
  bootstrap 95% **[+1.74, +11.98] pp**. Useful but below the
  prospectively frozen **+8pp** screen.
- Primary two-unit vs fixed product **+12.67pp**, passes
  +10pp gate, but both models perform below strong learning standards.
- Primary two-unit **50.52% accuracy** misses predeclared
  >=55% gate; 50.52% is essentially chance-like classification.
- Primary two-unit vs strictly parameter- and optimizer-matched dual
  network without diversity **+4.34pp** (exploratory paired interval
  approximately **[0.00, +8.68]pp**). This does not establish robust
  advantage of the regularizer.
- **Crucial NEGATIVE:** primary with real reward **50.52%** versus
  identical two-unit optimizer with scrambled rewards **52.08%**
  (−1.56pp, paired exploratory interval **[−8.85, +6.08]pp**).
  Thus the relevant unfamiliar family offers **NO causal evidence
  of reward-driven successful generalization** in this trial.
  The exact prespecified +5pp vs scrambled gate FAILS.

On **triple interactions**, the no-diversity matched model reaches
**85.94%**, but explicit diversity regularization suppresses accuracy
to **80.03%**, a **−5.90pp** difference with exploratory paired 95%
interval **[−9.98, −2.08]pp**. This is a meaningful negative effect
of our current "force-specialization" penalty on this family. Primary
two-unit itself is no better than original one-soft M5.31 81.60%
under this split. The old menu remains best for simple pairwise
interactions, 84.72% vs primary 75.17%.

### Frozen post-fit causal ablation

For every context, independently set each unit's output factor to
identity **1.0** on each unseen cue, without updating weights. This
is a causal removal operation, not a re-trained alternative model.
Also select a RANDOM unit for ablation using an independent fixed
random seed **before grading** and preserve its score as an audit
control. All original frozen synapse gates/strengths and the 15
float32 task-code values are in the raw JSON.

| Synthetic cohort | Both units each cause >=1/8 accuracy loss | Different nonzero input masks | Mean drop when removing unit 1 / unit 2 |
|---|---:|---:|---:|
| Plain parity | 2.78% | 2.78% | +2.31pp / +1.85pp |
| Pairwise AND | 4.17% | 6.94% | +6.86pp / +6.08pp |
| Triple AND | 0.69% | 2.08% | **−1.48pp / −1.22pp** |
| TWO independent ANDs | **19.44%** | **43.06%** | +4.34pp / +3.47pp |

Only **14/72** outside-class contexts meet the minimum
*both-unit-causal* ablation criterion. Among those 72 contexts,
**9/72** obtain the exact unordered two true conjunction input
masks (12.50%), and **35/72** match at least one true mask.
The 14 both-causal contexts have distinct, nonzero thresholded masks;
nonetheless that evidence alone does NOT establish genuine symbolic
specialization, since unseen function accuracy is chance-like and
reward corruption performs as well or better.

**Important accuracy/ablation interpretation:** negative mean drops
on triple interactions indicate ablating a unit can actually HELP
generalization. The regularizer may encourage an unnecessary second
unit there. A thresholded gate mask used solely for *reporting*
is NOT a hard-pruned inference network in M5.32.

### Persistence and compute honesty

15 float32 trained scalars = 5 original base weights + 2×(4 synapse
gates + 1 strength) = **60 actual bytes / 480 persistent task-code bits
per observable context**, **180 bytes / 1,440 bits across A/B/C**.
This is NOT the entire learner memory: 8 full float64 network restarts,
300 update steps, history, fixed nonlinear math, gradients and task
recognition are excluded. The strict matched comparison is to the
dual network with the same 15 trained parameters, initializations,
8 restarts, 300 steps and same input data but **zero diversity
penalty**. Comparing against M5.31 soft (16 restarts and ten
parameters) is an architecture benchmark **not a matched-FLOP test**.
No energy or compute efficiency conclusion follows.

### Preregistered screening gates

| Gate | Outcome |
|---|---|
| TWO-pair unseen accuracy >=55% | **FAIL**, 50.52% |
| TWO-pair dual gain over one soft >=8pp | **FAIL**, +7.29pp |
| TWO-pair dual gain over fixed >=10pp | PASS, +12.67pp |
| Plain parity retained >=90% | PASS, 97.92% |
| Triple accuracy >=75% | PASS, 80.03% |
| Diversity not worse than matched no-diversity by >3pp on TWO-pair | PASS, +4.34pp |
| Both neurons causally important in >=10% TWO-pair contexts | PASS, 19.44% |
| TWO-pair real reward beats scrambled by >=5pp | **FAIL**, −1.56pp |
| No support/exam overlap or oracle leakage to candidate | PASS |
| Three frozen float32 networks use exactly 1440 task-code bits | PASS |
| Failures, random ablation and both individual ablations reported | PASS |

**8/11, science screen FAIL.** Mechanical implementation checks and
other side conditions cannot compensate for missed primary gates.

### Evidence, exact archive, conclusion

**Full Actions run:** [#45072966328](https://github.com/kkongiu/neuroevolution-lab/actions/runs/45072966328), all jobs passed. Artifact
\`m5-32-dual-soft-synapses-dev\` (ID **14774041535**,
expires **2026-12-22**), containing complete support records,
frozen predictions, 156 rows × 3 contexts, both ablation scores,
random-ablation audit, raw synapse gates, all arm accuracies, full
metrics, and the fixed dataset metadata.

Independent retrieval and parse of this actual ZIP verified exactly
**156 rows, 468 contexts, 8/11 gates**, and recomputed the
canonical SHA-256 of the compact sorted raw JSON:
\`b3f0230a097a2e2906215af86c6a678682093feb355695fcfb9aaaf4078dc899\`.
The recomputed digest matches the stored campaign field exactly.

**Recommended scientific response:** Do NOT scale up neuron count or
claim an organism is learning independent operations. The task has only
eight training examples (some interaction gates rarely activate);
pairwise effects can remain observationally ambiguous and multiple
units overfit the eight examples. Consider a next **separate**
M5.33 controlled investigation varying observation coverage,
experimentally controlled active cue selection, and diversity
*after* evidence of complementary neuron contribution, while holding
train/test independence and accounting for costs. Add genuinely
new externally defined task families before any confirmation claim.
This is a proposal, NOT completed M5.33 work.

Immutable disposition flags:
- \`local_dual_plasticity_screen_pass=false\`
- \`autonomously_invented_neural_operator=false\`
- \`shared_meta_plastic_controller_learned=false\`
- \`independent_task_family_replication_complete=false\`
- \`continual_memory_retention_verified=false\`
- \`evolution_authorized=false\`
- \`advance_to_evolution=false\`
- \`sealed_M5_6_ecology_accessed=false\`

The original M5.6 ecology remains sealed. This PR is DRAFT stacked
on M5.31. No merge to \`main\`, and no evolution authorization.
