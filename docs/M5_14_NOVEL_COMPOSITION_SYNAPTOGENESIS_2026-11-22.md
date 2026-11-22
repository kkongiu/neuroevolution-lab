# M5.14 — Synaptic formation, consolidation/pruning, and new compositional tasks

**Date:** 2026-11-22. **Status:** completed development-only negative compositional-transfer diagnostic. Growth improves training of the new task versus a capacity-limited sparse baseline, but does not show superior zero-shot transfer or learned growth timing. **Foundational operator-learning prerequisite fails.** No evolutionary promotion.

## Question and task grammar

Can a native fixed-eight-neuron, 260-possible-synapse M5.6 RNN form new synapses when learning stalls, preserve useful old edges through pruning, and transfer skills A and B to a novel problem C that algebraically combines those skills?

Designed *observable* task grammar, 880 sequential trials per organism:

| Phase | Decisions | Synaptic learning? | Target held exclusively by environment |
|---|---:|---|---|
| P | 160 | Yes | x XOR y (introduce operator) |
| A | 160 | Yes | x XOR hidden private bit a |
| B | 160 | Yes | y XOR hidden private bit b |
| A return | 40 | Yes | same A |
| B return | 40 | Yes | same B |
| **C zero-shot probe** | **80** | **No** | (x XOR a) XOR (y XOR b) |
| C acquisition | 160 | Yes | same C |
| **A return probe** | **40** | **No** | same A |
| **B return probe** | **40** | **No** | same B |

Cues (x,y) are factorially balanced in each 40-trial block. Each observation includes task identity, the operands and shared operator/operand-role flags, but **not** hidden bits a or b, the correct action, or upcoming transitions. Feedback is one scalar success/failure signal after each chosen action. The operator flag from P also appears in C and operand-role flags from A and B both appear in C. This grammar is engineered by the experimenter and **is not open-domain natural compositional intelligence**. Even with designed grammar, the learner is not given a direct supervision label or a hardcoded XOR calculation.

Both the early C transfer test and the final A/B recall probes are **frozen**: action choices and recurrent hidden-state evolution continue, but no neural weight, structural mask or growth events change. Thus early C scoring measures pre-C-feedback transfer and later probe scores measure old-task retention without retraining during the probe.

Development-only seeds **3700–3723**, fully disjoint from M5.6 sealed SOURCE/HOLDOUT/REPLICATION seeds and previous M5.9–M5.13 development seeds.

## Neural interventions and controls

All non-dense arms initialize from the same 156 active edges and same 260-value native founder genome on each seed, with the same M5.6 8-hidden-unit controller and per-action reward, learning rate 0.04 and stochastic action tickets. All have a hard cap of 260 possible edges; no new neuron or open-ended network growth.

1. `fixed_sparse`: weight updates only, 156 static active edges.
2. `full_dense`: fixed 260 active edges from birth, pure-parameter-capacity control.
3. `scheduled_birth`: activate eight new synaptic slots every 80 training actions, independent of successes; choose candidate locations from local activity/feedback opportunity trace.
4. `adaptive_birth`: after each training window of 40 actions, activate up to eight dormant candidate edges only when last-window accuracy is below 68% and the change from previous window is below +7.5 percentage points. The trigger accesses **only the outcome history**, never task identities/transitions or answers. Candidate edge selection uses local pre/post coactivation coupled with actual reward history; it is not a counterfactual causal-importance calculation.
5. `yoked_random_locations`: exactly the same per-seed synapse-birth times and counts as `adaptive_birth`, drawn from that paired experimental run, but randomized edge identities (audited same mean synaptic occupancy). This isolates *where* new synapses are placed, not the merit of the birth trigger itself. No C targets are transmitted to the policy.
6. `adaptive_birth_prune`: adaptive candidate growth plus pruning of four low fast/slow activity-credit edges every 80 training actions (down to minimum 120). A structural-consolidation heuristic, not proven biological pruning.

Cohort interventions are compared on the same curriculum, hidden rule bits and founder seeds. Actual active synapse counts may vary by treatment, so performance differences versus fixed_sparse are **not evidence of a purely growth-specific effect separate from increased parameter capacity**. The scheduled arm is *not* perfectly dose matched to adaptive birth; use the yoked arm for exact growth-exposure matching.

## Verified actual GitHub Actions results

[Run #45033293352](https://github.com/kkongiu/neuroevolution-lab/actions/runs/45033293352): all 4 new compositional/frozen-probe invariants and 5 original native learner regressions passed; 24 seeds × 6 arms = **144 simulations**, **880 decisions each**. Complete raw JSON plus synapse birth/pruning events archived as `m5-14-compositional-task-synaptogenesis` for 30 days.

Row digest SHA256: `a37c20964fe5e82067ca12461051e4fa88e6f4a38b4afa966c7ca168189e286f`.

| Arm | XOR operator P final-40 | C prelearning frozen zero-shot | C acquisition final-40 | Old A frozen | Old B frozen | Avg active synapses |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed sparse 156 | 48.02% | 49.58% | 56.88% | 68.23% | 65.52% | 156.00 |
| Full dense 260 | 48.23% | 48.33% | 63.02% | 67.71% | 68.54% | 260.00 |
| Scheduled birth | 47.92% | 50.83% | **63.75%** | 69.17% | 72.19% | 193.90 |
| Adaptive birth | 48.02% | 50.00% | 63.44% | 70.73% | 72.29% | 199.27 |
| Birth yoked, random locations | 48.33% | 47.60% | 61.77% | 71.88% | 71.77% | **199.27** |
| Adaptive birth + pruning | 48.13% | **51.61%** | 63.33% | 70.42% | **73.75%** | 182.04 |

Other observations: adaptive birth adds an average of **80.33 synapses**; adaptive birth+prune adds ~**84.67** and removes **36** across 880 trials. Its average active synapses are 182 rather than 260. The placement-yoked control's exact birth time/count and allocated average occupancy are identical to adaptive birth, but early C accuracy is lower (47.60% vs 50.00%); this difference is not by itself a demonstration of abstract composition or statistical significance.

### Prerequisite warning: operator learning fails

The **P XOR operator phase does not produce reliable learning**: final-40 P accuracy is about **48%** across *all six arms*, even the dense network. For 2-choice outputs the chance level is 50%. It is scientifically invalid to conclude from near-chance C zero-shot results that the structural mechanisms prevent conceptual transfer: the prerequisite primitive operation has **not been learned**. This is a stronger bottleneck in the current REINFORCE-based learning algorithm than simple synaptic allocation.

The preliminary C prelearning advantage for adaptive+prune versus static sparse is **+2.03 percentage points** (51.61% vs 49.58%), far short of the declared +5 pp screening threshold and well within plausible random fluctuation. Nobody reaches the absolute 60% C zero-shot threshold.

### Development screening criteria (set before running)

1. Adaptive+prune C zero-shot >=60%: **FAIL**, 51.61%.
2. Adaptive+prune C zero-shot >=5 pp above static sparse: **FAIL**, +2.03 pp.
3. Adaptive+prune C-learning late-window >=65%: **FAIL**, 63.33%.
4. A frozen return >=65%: **PASS**, 70.42%.
5. B frozen return >=65%: **PASS**, 73.75%.

**2/5 pass; `advance_to_evolution=false`.**

## Limits and next experiment

- This development experiment does not show spontaneous generalization from previously solved skills. The operator P is not mastered, so transfer cannot meaningfully be expected.
- Growing **synapses** within a fixed neural graph of 260 possible weights does not realize the open-ended neuron count envisioned earlier. Computational workload still includes dense-size M5.6 operations. Mean active synapse count is an occupancy proxy, not joule/FLOPs savings.
- The role flags and combinational task syntax are experimenter-defined. No external task-change oracle supplies target rules, but C's operator/operand structure is explicitly represented by sensory flags.
- Persistence of A/B recall does not demonstrate long-run unbounded continual learning or rule inference.
- The main scientific next step is a **learning-prerequisite benchmark**: first find a task/learning algorithm on which simple and XOR operator primitives can reliably reach predeclared performance under scalar feedback (or an honest separate supervised-learning comparator), then freeze acquired weights and assess zero-shot combination on genuinely held-out novel pairings; only thereafter revisit synaptic growth and pruning.
- No source/holdout/replication cohorts used and no changes to `main`. This is a stacked draft branch off M5.13.

**Do not tune against the same 3700..3723 cohort and relabel results as fresh confirmation.**
