# M5.21 — Meta-training plasticity control across fresh lifetimes: negative transfer result

**Date:** 2026-11-22. **Status:** completed development experiment with an explicitly separate training/evaluation seed split. **2/10 STOP/GO criteria passed.** Evolution, structural expansion and progression to new task domains remain blocked.

## Research question and motivation

M5.20 demonstrated that within a single organism's lifetime a small tabular Q-learner could update a synaptic protection policy, but it did not outperform random or fixed comparators. It only received about **28 effective metaplasticity decisions**. M5.21 therefore asks whether it is possible to **learn a plasticity policy across independent simulated lifetimes** and transfer *only that policy* to newly initialized organisms, so that their previously acquired task knowledge is not inherited.

This is an explicit model of meta-learning over resets rather than artificial evolutionary selection: lower-level network, synaptic anchors, observed reward records, action-choice random state and training trajectory **always restart from scratch**; only a learned 4-group Q-table is passed between training lifetimes.

## Prespecified data split

- **Meta-training:** 64 independent lower-brain lifetimes, seeds **4400–4463**, to update one shared actual-reward Q policy. In a separate matched 64-life training sequence over the *same training seeds*, a control Q policy learns from independent uniform random pseudo-rewards rather than the real meta-reward. Each Q policy gets **64 × 28 = 1,792 effective Q updates**, no external exam grades.
- **Frozen evaluation:** 24 independent **new** organism seeds **4500–4523**, completely disjoint from the 64 training seeds and earlier M5.6–M5.20 development cohorts. During evaluation, both learned Q tables are **frozen**, with 0 further Q updates. Other arms are evaluated on the identical 24 fresh brain founders.
- **Important limitation:** training and evaluation use the **same simple A→B→C→A→B curriculum grammar**, with task identity explicitly supplied and only four possible binary sensory cue patterns per context. New-seed transfer is **not** new-task-family transfer or open-ended intelligence. The original M5.6 sealed source/holdout/replication cohorts remain untouched.

All 296 entire simulated lifetimes are accounted for: **128 training** (64 genuine and 64 corrupted-reward) and **168 evaluation** (24 new organisms × seven regulators). Each lower brain experiences exactly **1,520 chosen-action outcomes** and makes exactly one gradient parameter update per lived action. Q learning itself entails 1,792 additional scalar Q-table updates per meta-training policy; it is **extra computational cost**, not equal to passive regulator cost.

## Lower-level model and control

Same 16-hidden-unit, 145-weight action-conditioned reward prediction network as M5.20, with independently initialized per-seed weights and own-action scalar success/failure training. Context identities A/B/C are explicitly observed. Tasks: A=XOR (400), B=inverse XOR (400), C=x (400), then 160 A and 160 B revisits, with 40-step balanced four-pattern blocks. After each phase, synaptic importance/consolidation anchors are established by an **externally provided phase boundary**. This is neither self-triggered consolidation nor natural ecology.

All arms hold the same 128-record personal memory ceiling (43 A, 43 B, 42 C), with read-only old-experience Brier evaluations; their per-life reference-forward budget is **3,940** matched evaluations. Each individual stores only real (x,y,context,chosen_action,observed success) records. There is **no supervised correct-action target** and no historic replay weight training. During meta-training, **no frozen examination is even computed**, preventing exam outcomes from entering the Q-policy.

### Meta-controller policy and reward

Every **40 lived decisions**, a controller chooses one of **0.10, 0.30, 0.60, 1.00** synaptic-protection strengths **for exactly one of four 4-neuron groups**; the other three stay at 0.30. The group rotates deterministically, giving more interpretable marginal attribution than M5.20's four simultaneous group choices. The four-state Q policy observes only whether recent success fell below 80% and whether previously observed past-action Brier loss exceeded 0.15; it has **no direct curriculum-phase input**.

One-step Q learning uses `learning_rate=0.20`, `discount=0.70`, and epsilon-greedy exploration annealed with *cumulative* effective decisions across the 64 training lives. All Q updates in the first A phase are disabled because no prior synaptic anchors exist; each subsequent lifetime has **28 causally effective decisions**, 1,792 total per policy.

Its *engineered, non-exam* reward for a 40-action block is:

```
R_meta = 0.8 * (new_accuracy - 0.5)
       + 0.5 * (new_accuracy - previous_block_accuracy)
       - 1.4 * max(0, absolute_old_Brier_after - 0.08)
       - 0.4 * max(0, old_Brier_after - old_Brier_before)
```

Both Brier terms use only the individual's actual past chosen-action records in the other contexts. This corrects M5.20's emphasis solely on newly increased forgetting, by also penalizing **persistently elevated historical prediction error**. The meta-reward remains manually selected: optimizing it does not guarantee strong accuracy/retention on unseen exams.

## Seven frozen-evaluation arms

1. **learned_transfer:** actual-meta-reward policy trained across 64 lives, then Q frozen.
2. **corrupted_transfer:** same Q training budget but independent random pseudo-meta-reward, then frozen.
3. **untrained_q:** never-updated zero Q, greedy action selection, all other conditions equal.
4. **random_controller:** independent uniformly random protection choice for the rotated single group, never trained.
5. **fixed_weak:** protection factor 0.30 for every neuron group; other parameters and memory equal.
6. **fixed_strong:** protection factor 1.00 for every group once anchors exist; 0.30 during the initial causally inert A phase.
7. **heuristic_controller:** a hand-coded one-group regulator driven by the same coarse success/risk bits.

All regulators receive identical 1,520 lived actions, one lower neural-weight update per action, 128 record slots, 3,940 old-experience model predictions and 38 controller opportunities (of which 28 are effective). In evaluation, each Q controller performs **zero** Q updates. Only the Q-table values persist across simulated training lives; the 24 test organisms' actual cognitive state is independently reset.

## Real completed experiment

[Successful GitHub Actions #45052555236](https://github.com/kkongiu/neuroevolution-lab/actions/runs/45052555236) executed 5 new consistency/isolation tests and the 5 original M5.6 neural learner regression tests, all passing. All **128 full 64+64 training lives** and **168 independent frozen-policy test lives** completed. Machine-readable Q tables, per-lifetime training diagnostics, per-arm frozen grades and paired-seed bootstrap uncertainty were archived as artifact `m5-21-meta-training-frozen-transfer` (30-day retention).

Trained-Q SHA256 `d4417ccca1be458ed22b4f7e0b193022c32327869699e3e5a7ddf8fdfd2a385b`; canonical frozen-evaluation raw rows SHA256 `bef1b277ce2ba334d4864ac64ab0d35f1d3852adfab5399ced8dc46425882228`.

### Frozen exam accuracy (%) — 24 new-seed organisms per arm

| Treatment | A after B | **A after C** | B first learned | **B after C** | C first learned | Final A/B/C after revisits | Pre-refresher tradeoff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed weak 0.30 | 91.67 | 93.75 | 85.42 | 70.83 | 92.71 | **85.42** | **85.68** |
| Fixed strong 1.00 | 95.83 | 94.79 | 76.04 | 72.92 | 71.88 | 80.21 | 78.91 |
| Random one-group regulation | 91.67 | 95.83 | 85.42 | 70.83 | 91.67 | 84.38 | 85.94 |
| Untrained zero-Q regulator | 89.58 | 92.71 | 82.29 | 67.71 | **94.79** | 85.07 | 84.38 |
| Corrupted-meta-reward Q | 91.67 | 94.79 | **84.38** | **72.92** | 92.71 | 84.72 | **86.20** |
| **True-reward Q trained on 64 lives** | **91.67** | **96.88** | **81.25** | **68.75** | **89.58** | **83.33** | **84.11** |
| Hand-coded policy | 90.63 | 92.71 | 82.29 | 72.92 | 95.83 | 83.33 | 85.94 |

**Pre-refresher tradeoff** is a fixed equal-weight mean of frozen A-after-C, initial B accuracy, frozen B-after-C and initial C accuracy. It evaluates both retention and acquisition before re-teaching old skills. All numbers are means of the same 24 evaluation seeds.

### Main interpretation

**The frozen learned meta-policy does transfer its Q actions into unseen-seed organisms, but it does NOT improve actual long-term plasticity.** It retains A after C at 96.88%, better than weak fixed protection's 93.75%, but first-time acquisition of B drops to 81.25% from 85.42%, C drops to 89.58% from 92.71%, and B after C remains only 68.75%. Stronger A protection is not enough; the organism must still learn B and C.

The trained policy's pre-refresher tradeoff is **84.11%**, compared to **85.68% fixed weak**, **85.94% random**, **86.20% corrupted-meta-reward trained** and **84.38% untrained Q**. No positive improvement over any matched control.

### Paired bootstrap analysis (24 fresh evaluation seeds)

Comparison is transferred true-reward Q minus named control on per-seed pre-refresher tradeoff. 1,600 fixed-seed paired bootstrap draws; intervals are exploratory, not confirmatory hypothesis tests.

| Comparator | Mean advantage (percentage points) | Paired 95% bootstrap interval (pp) |
| --- | ---: | --- |
| Fixed weak | −1.56 | [−4.43, +0.52] |
| Random controller | −1.82 | [−4.69, +0.26] |
| Corrupted-reward Q | −2.08 | [−5.21, +0.26] |
| Untrained zero Q | −0.26 | [−3.13, +2.34] |

All confidence intervals include 0; the development experiment provides **no credible benefit of real-meta-reward policy training**, despite 64-life exposure and an explicit independent-seed frozen transfer test. The comparison with corrupted-meta-reward Q also cautions that Q changes alone or regularization may be enough to produce similar behavior in this tiny task.

## Predeclared developmental stop/go gate (2/10 pass)

1. A frozen after B ≥90%: **PASS**, 91.67%.
2. A frozen after C ≥90%: **PASS**, 96.88%.
3. B first learning ≥90%: **FAIL**, 81.25%.
4. B frozen after C ≥90%: **FAIL**, 68.75%.
5. C first learning ≥90%: **FAIL**, 89.58%.
6. Final A/B/C mean ≥90%: **FAIL**, 83.33%.
7. Tradeoff vs fixed weak +5 percentage points: **FAIL**, −1.56 pp.
8. Tradeoff vs random +5 pp: **FAIL**, −1.82 pp.
9. Tradeoff vs corrupted-meta-reward Q +5 pp: **FAIL**, −2.08 pp.
10. Tradeoff vs untrained zero Q +5 pp: **FAIL**, −0.26 pp.

**2/10 PASS, 8/10 FAIL**. Code forcibly retains `development_screen_pass=false`, `independent_task_replication_complete=false`, `evolution_authorized=false`, `advance_to_evolution=false`.

## Scientifically important caveats

- This does not show the impossibility of meta-RL for plasticity. It shows this specific coarse-state, constrained four-neuron-group, four-action tabular Q-controller, trained on one known task grammar with an engineered meta-reward, does not beat matched baselines after 64 training lives.
- Only Q values transfer: not individual learned weights, old examples or internal learner states. Yet the training/evaluation *curriculum format is identical*, so there is no transfer to novel structures or ecologies.
- The regulator uses one group decision per block. This reduces but does not eliminate **credit assignment** problems; other group weights have been changed by previous blocks, so 40-step rewards still depend on historical actions.
- The memory loss signal uses **Brier prediction loss on historically selected past actions**. Good prediction of old actions' outcomes need not correspond exactly to maintaining greedy **policy** accuracy on all actions or unseen cases.
- Absolute old loss has a tunable researcher-defined threshold (.08) and weights (1.4 and .4). This split was not optimized via independent validation; testing alternative reward functions on these evaluation seeds would overfit.
- Consolidation still depends on experimenter task boundaries, and context identity is given explicitly.
- Equal lower-brain arithmetic does NOT mean equal total compute, since Q training has 1,792 additional Q-table updates per policy. Historical reference count is the same, as an intentionally generous baseline control.
- The bootstrap confidence intervals are development-only and computed on 24 new seeds of the same task grammar; no independent task-family replication.

## Stop decision and focused next hypothesis

Do **not** expand to new task families, environments or evolution in response to this failure. The primary unresolved problem is the **plasticity–stability tradeoff**. A more direct causal mechanism may be needed instead of increasingly elaborate meta-reward heuristics.

One scientifically cleaner, still plasticity-only diagnostic would verify each candidate synaptic update against old actually experienced prediction losses **before committing** the update. An old-task trust-region proposal can reduce local interference without tuning a global protection-strength reward. Such a mechanism must be held to the same actual-experience budget, the same memory cap, and an explicit accounting for additional candidate-forward evaluations. However, no new experiment has yet been authorized by this outcome.

**No change to `main`, no sealed M5.6 cohorts, no automatic evolutionary activity.**
