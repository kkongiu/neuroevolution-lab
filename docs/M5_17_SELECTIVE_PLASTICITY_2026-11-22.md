# M5.17 — Selective synaptic plasticity versus experience replay and structural remodeling

**Date:** 2026-11-22. **Status:** completed exploratory development experiment. **6/7 declared screening criteria pass; independent replication absent and evolution explicitly BLOCKED.**

## Question

Can a neural organism **protect previously learned competence by regulating how quickly individual parameters may change**, *even without episodic replay*, while simultaneously preserving the ability to acquire new tasks? Is selective growth/pruning better than random growth/pruning at exactly the same event timings and number of active synapses?

The mechanism is a researcher's **EWC-like soft synaptic-consolidation heuristic**: after a learning phase, estimate the per-parameter sensitivity of the reward-prediction logit on actual experienced state–action events, capture its weight anchor, and slow future gradient movement and penalize drift proportional to the parameter's salience across prior phases. All cues and rewarded actions are observed only as chosen-action binary feedback. Importance is NOT true causal ablation, and phase-boundary consolidation is **explicitly triggered by the laboratory curriculum** rather than self-initiated. This is selective plasticity, not established autonomous plasticity regulation.

## Classroom and protocol

- Fresh independent **development-only seeds 4000..4023 (24)**; original M5.6 sealed source/holdout/replication cohorts and all earlier M5.9–M5.16 development seeds untouched.
- Same action-conditioned 16-tanh-hidden-unit two-action reward predictor for each treatment (145 possible learnable parameters). Observations contain two binary input cues **x,y**, an explicit one-hot task identity A/B/C, and the candidate action. Task identity remains laboratory-provided.
- Task A: rewarded for action = `x XOR y`; B: rewarded for inverse XOR; C: rewarded for action = `x`. These three rules conflict in their cue/action mapping to varying degrees, but each world has only four possible binary cue pairs, repeatedly presented.
- Curriculum: **400 A → 400 B → 400 C → 160 A → 160 B**, totaling 1,520 lived episodes per learner. Each 40-trial block includes all four cue pairs ten times. All feedback during training comes from the **chosen action's observed reward**; the environment-only hidden correct action is never passed as a teacher label. No correct-action label is inferred by a privileged API.
- After every learned phase, frozen exams are run on **all contexts previously encountered**, without updating weights, synapse structure, episodic memory or the action RNG. This includes A immediately after B, A/B immediately after C, and all three after the final return to A/B. Frozen exams use new random shuffles of the *same four patterns*, **not novel contexts or out-of-distribution problems**.
- Bounded episodic replay memory = **128 total experienced action–feedback records**, deliberately split across the three known contexts with capacities **43 A / 43 B / 42 C**. This context partition is externally programmed and not an emergent decision of the organism.
- Four recorded variables per structural arm: active parameter counts, deliberate birth events, pruning events, and matched-yoke counts/timings. The structural arms begin at 104/145 active parameters; a hard bound remains 145, no neuron additions. The implementation is dense and parameter occupancy is **not a measured runtime, FLOPs, or energy saving**.

## Nine controls and update-budget fairness

1. `online`: one real-reward gradient per interaction; **1,520 updates**, no replay and no consolidation.
2. `recent_repeat`: five updates of latest actual event; **7,600** updates, no history replay or consolidation.
3. `replay`: one current update + four samples from 128-record context-stratified past experiences; **7,600** updates; no consolidation.
4. `plasticity_only`: one actual-event update plus salience-dependent weight protection at curriculum phase ends, **1,520** updates, **no replay**.
5. `replay_plasticity`: 7,600 updates with the same 128-record replay, plus consolidation (no structural pruning).
6. `replay_plasticity_sparse`: same rule and replay with **104 fixed active** out of 145 possible synapses.
7. `replay_plasticity_growth`: same sparse initialization, but low-progress-triggered activation of up to four previously dormant weights at 80-step checks, and selective pruning of up to two lower-importance active edges every 160 lived steps. Existing important edges protected preferentially; the mechanisms are **heuristics**, not causal gradients of the removal outcome.
8. `replay_plasticity_yoked`: copies precisely the directed-growth condition's **per-seed structural event timing, birth/death counts, and occupancy history**, but randomizes edge identities; equal 7,600 update passes and active-parameter exposure as directed growth.
9. `replay_uncoupled`: same 7,600 update count and balanced replay architecture as replay, but learner feedback is replaced by an independent binary coin; negative learning control, not a strict reward-marginal-matched shuffle.

All nine treatments initialize from the **same exact seed-specific dense genotype**, and sparse groups share exactly the same initial active masks. Experiential learning does not change the heritable representation. No evolution, genetic selection or reproduction.

## Verified numerical results

[Successful GitHub Actions run #45039632202](https://github.com/kkongiu/neuroevolution-lab/actions/runs/45039632202): all five new tests (balanced cues, real-feedback firewall, frozen probes, 128-record memory ceiling, equal/yoked structural events, seed isolation) plus five original M5.6 regression tests pass. **24 seeds × 9 arms = 216 full independent trajectories** of 1,520 training episodes. All raw per-organism metrics, frozen exams and structural event rows are stored as the GitHub Actions artifact `m5-17-selective-plasticity-development` (30-day retention). Raw row SHA-256: `13f7de468240626e4df70c34483ac398b020a2b29bcb38cd43bf47219d188bf1`.

### Frozen performance by arm (%)

| Arm | **A after B** | **A after C** | **B after C** | **C immediately after C** | Final mean A,B,C | Mean active parameters |
|---|---:|---:|---:|---:|---:|---:|
| Online only, no replay | 47.92 | 48.96 | 59.38 | 98.96 | 85.07 | 145.0 |
| Repeat latest experience 5× | 56.25 | 50.00 | 64.58 | 100.00 | 95.83 | 145.0 |
| Replay only | **100.00** | **100.00** | **97.92** | **100.00** | **100.00** | 145.0 |
| **Plasticity only, no replay** | **95.83** | **95.83** | 63.54 | **79.17** | 79.51 | 145.0 |
| Replay + plasticity | 100.00 | 100.00 | 89.58 | 96.88 | 95.49 | 145.0 |
| Replay + plasticity, static sparse | 100.00 | 100.00 | 56.25 | 72.92 | 76.04 | 104.0 |
| **Replay + plasticity, directed birth/prune** | **100.00** | **100.00** | **73.96** | **85.42** | **87.15** | 107.68 |
| Same structural event dose, random identities | 93.75 | 93.75 | 64.58 | 84.38 | 78.13 | **107.68** |
| Replay with random feedback | 48.96 | 50.00 | 53.13 | 46.88 | 50.69 | 145.0 |

All exam scores are means over 24 development seeds on frozen, balanced 80-decision exam sets. Different subgroup and curriculum choices would change these numbers; the experiment is not an independent confirmatory result.

### Strong diagnostic #1: protection works, but new-task plasticity is impaired

Without any replay, the protected-weights learner achieves **95.83% on A after B**, versus **47.92% for the online-only learner**, an exploratory **+47.92 percentage-point** memory benefit using the *same single learning update per lived episode*. Thus protecting learned weights can be effective even without re-presenting older experiences.

But that learner also scores only **72.92% on B after its first B learning** versus **89.58% online**, and **79.17% on C after C training** versus **98.96% online**. This is the central **stability–plasticity dilemma**: preserving existing representations is useful, but overconsolidation makes new learning harder.

### Strong diagnostic #2: combining replay + consolidation did not help this task

**Replay-only** reaches 100.00% on A after C, 97.92% on B after C, 100% on C just after training, and final mean 100%. **Replay+plasticity** reaches 100.00%, 89.58%, 96.88% and final mean 95.49% respectively. Adding the implemented consolidation heuristic therefore **does not outperform replay**; it makes some context acquisition/retention worse. This falsifies the simple hypothesis that replay and fixed-intensity protection necessarily yield additive benefits.

### Structural birth/prune diagnostic: positive within sparse capacity

- Directed birth/prune relative to **same initially sparse, static network**: final three-task mean **87.15% vs 76.04%**, +11.11 pp. Mean active parameters 107.68 versus 104.00, hence this difference alone **mixes increased capacity and edge placement**.
- Directed birth/prune relative to **exactly event-yoked, same-occupancy random identities**: final three-task mean **87.15% vs 78.13%**, +9.03 pp at the **same mean active parameters, event timings and birth/prune dose**. This supports selecting which existing/potential edges to prune or activate versus random turnover in this family. It does not independently establish that the *stalled-progress trigger* is necessary or that growth saves CPU energy.
- Both sparse arms remain inferior to 145-parameter replay-only at 100%; pruning is not yet an accuracy improvement over full capacity.

## Predeclared development screen

1. Replay+plasticity A after A, B after B and C after C each >=85%: **PASS** (100%, 90.63%, 96.88%).
2. Plasticity without replay A after B >= online +8 pp: **PASS**, +47.92 pp.
3. Replay+plasticity A after C >= replay only +5 pp: **FAIL**, both 100% (0 pp).
4. Replay+plasticity B after C >=75%: **PASS**, 89.58%.
5. Replay+plasticity C after later A/B returns >=75%: **PASS**, 96.88%.
6. Directed structural remodeling final average >= static sparse +5 pp: **PASS**, +11.11 pp.
7. Directed structural remodeling final average >= random-yoked +5 pp: **PASS**, +9.03 pp.

**6/7 pass**. Explicit workflow fields: `development_screen_pass=false`, `independent_replication_complete=false`, `evolution_authorized=false`, `advance_to_evolution=false`. (All seven must pass to pass screening; passing six of seven does NOT advance.) No sealed environments accessed, no merge to `main`.

## Critical limitations

- Experimenter supplies known task identity, structured cue representations, explicit curriculum end times **for consolidation**, and explicit 43/43/42 memory segmentation. The organism does not choose when to consolidate, how much memory to allocate, nor when to invent a new task.
- Reward-prediction parameter salience derives from local derivatives on actually experienced actions. It is **not** experimentally verified causal importance or biological synaptic tagging.
- The actor can master these *four-cue, binary-action* problems with replay only; ceiling performance makes some incremental comparison uninformative. Novel continuous cues, three or more actions and hidden task contexts remain necessary.
- Structural weights are formed/pruned inside a **fixed maximum 145-parameter network**, not unbounded neuron growth. Dense arithmetic is still used; number of active parameters is merely a structural occupancy proxy.
- Only development seeds used; no fresh independent replication, no confidence interval for paired differences, no confirmatory claims.

## Next preregistered proposal (M5.18)

1. **Regulate protection strength using measured learning progress:** allow selective relaxation if the current task has poor performance while old memories are still well retained; compare with a fixed protection coefficient at **identical episode/update and replay budgets**. Do not tune on this same 4000–4023 cohort and relabel it independent.
2. Test **task-blind detection**: infer when stable task statistics change from reward prediction error rather than receiving an explicit curriculum-end signal. Retest with hidden/noisy contexts.
3. Evaluate a capacity-aware **memory/plasticity tradeoff** on 3–10 tasks, 3+ actions, more varied rule families and truly unseen sensory combinations.
4. Verify structural placement effects with **matched pruning-only, birth-only, and equal-capacity static ablations** and per-seed paired confidence intervals before claiming benefit from growth itself.
5. Only after a separately preregistered independent successful replication, discuss possible evolution in the physical world. Keep original M5.6 sealed cohorts untouched until explicit authorization.

**No change to `main`, no autonomous evolution run.**
