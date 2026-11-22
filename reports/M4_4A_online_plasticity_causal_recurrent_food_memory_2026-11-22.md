# M4.4A — Real within-lifetime synaptic plasticity and causal RNN short-term recall jointly yield physical food reward (positive preregistered test)

**Date: 22 November 2026.** **Both predeclared principal MAIN and longer-delay OOD transfer gates PASS**, within the strong limitation of an externally **motor-guided physical T-corridor**. Do **not** reframe this as fully autonomous ecological navigation, endogenous predator learning, or natural evolving neural biology.

## 1. Scientific context and preregistration

Earlier M3.3 proved that SGD can update weights and replay helps preserve familiar task performance. Earlier M4.3 proved *the hidden recurrent state* can hold a stimulus after its cue disappears, but only when its 260 RNN weights were genetically **optimized externally before the lifetime**. These are **different capacities in different systems**.

**M4.4A was prospectively frozen [protocol `af9e64e9e2ec3a5f47d27f963c27467661ee1681`](https://github.com/kkongiu/neuroevolution-lab/commit/af9e64e9e2ec3a5f47d27f963c27467661ee1681) BEFORE code, training outcomes and unseen worlds.** It tests both **within-lifetime online recurrent synaptic plasticity** and **causally useful hidden-state memory in the SAME small recurrent neural network**, when the actual terminal reward is the physical food found by the network's own chosen branch.

- Source training [GitHub Actions #45004409330](https://github.com/kkongiu/neuroevolution-lab/actions/runs/45004409330) **SUCCESS**: all unit tests passed, including finite-difference numerical verification of full recurrent BPTT policy gradients and exact equivalence with original M4.2 Controller. 8 independent initialized nets ×3 trained variants ×288 genuine physical reward episodes = **6,912 original rewarded training episodes**, on 24 original training corridor geometries 121000–121023, repeated 6 times with both hidden food goals in every geometry.
- Exact original and learned full numerical 260-RNN/268-FF weights for all eight initializations, all 6,912 training events, and source-only geometry hashes were **committed to Git BEFORE any final evaluation**: frozen source bank [`reports/M4_4A_FROZEN_ONLINE_SOURCE_BANK_2026-11-22.json`](M4_4A_FROZEN_ONLINE_SOURCE_BANK_2026-11-22.json), committed in independent source-only CI commit **`640863e9279950424557284ec8530820a227d168`**, bank SHA256 **`691da96fa9173ad76e3f0402b25e3211a00f4a50091cef1b1b4e2ba167014c7f`** and `heldout_worlds_examined=0`. All original raw source trials permanently archived [here](M4_4A_all_6912_original_rewarded_source_episodes_2026-11-22.csv).
- The physically moving simulated agent spends **one real native movement-cost unit each traversed cell** and physically eats food only when its chosen gate is the actual food-containing gate. It gets terminal feedback `+1` on its OWN actual food, `-1` on its own empty branch, **without correct action label or hidden food identity being provided to the neural learner**. The learner ascends the exact REINFORCE log-probability gradient through ALL previous tanh recurrent states, thereby updating **actual 19-input, recurrent hidden-to-hidden AND output weights** after every experience; no external mutation or genetically evolved champion is used. Stochastic fork choice for source exploration, independent neural initial seeds 124000–124007.
- The cue explicitly tells the robot which absolute direction to prefer for exactly the **first two** observed physical corridor cells. It vanishes before the fork, and opposite food-goal worlds have **exactly identical final fork observation, walls, geometry and original physical motor costs**. Any systematic paired success above chance demands retrieving the old cue or a hidden correlating signal; the physical fork observation SHA is matched for both opposite goals.
- Training lasts 288 reset-location physical episodes per network. Hidden activation and energy restart each episode, but synapses keep prior episode updates. That is **online learning across episodes within the simulated learner's lifetime**. It is **not** uninterrupted multi-day ecological life. The agent is **externally moved along the stem** toward the fork; only its final gate action is its own neural motor control. This scaffold is expressly required for M4.4A and is not hidden.

## 2. All eight frozen arms and their interpretation

Every identical independent new source-seed/world/goal observation is evaluated under:

1. `online_rnn`: original full 260-weight RNN (8 tanh recurrent units) **after 288 own-reward updates**, weights then **frozen** for all independent tests.
2. `initial_rnn`: its OWN exact original 260 numeric weights before ever experiencing rewards (not a different baseline random seed).
3. `online_rnn_reset`: **SAME trained numerical weights** but zero all recurrent hidden states immediately before reading the fork sensors, then perform a normal fork computation. This is a genuine within-genome causal memory intervention.
4. `online_rnn_flipcue`: **SAME trained numerical weights**, input cue swapped to the opposite branch in the first two cells, with physically food-containing gate left unchanged. Because this is a balanced two-choice task, its success is mathematically tied to the correct-cue success; it is not an independent statistical replication.
5. `shuffled_reward_rnn`: the **SAME original random RNN initialization** and same 288 physical training opportunities, but the learning feedback is an independently randomized signed value unrelated to the physically observed food. It then evaluates accurately against physical food in novel worlds, with original trained numerical weights frozen.
6. `online_ff`: 268-weight 11-tanh-unit **feedforward** policy trained with the same physical experience and same REINFORCE feedback, but with no hidden state across corridor observations. It has slightly **MORE** trainable parameters than the RNN.
7. `scripted_one_bit`: explicitly researcher-programmed cue-storage action control, **100% expected**; not a learned neural intelligence.
8. `random_binary`: opposite food goal pairs, same chance action, **50%** expected.

The negative controls `online_ff`, `online_rnn_reset`, `random_binary` all score **exactly 50% over perfectly matched opposite target pairs BY CONSTRUCTION**, not independently discovered neural statistics. The strongly positive comparisons reflect one physical-reward effect plus one hidden-memory causal effect, not four independent scientific discoveries.

## 3. Direct evidence of numerical online synaptic learning: 6,912 original source episodes

All eight source networks exhibited strictly positive **actual recurrent-weight L2 drift**, ranging **0.38965–1.60104**, and strictly positive input-weight and readout-weight numerical drift; original genotype vectors are full precision and permanently archived. Exact BPTT gradient was independently checked by numerical finite differences on old RNN and matched FF original architectures.

A later **read-only audit** of the ORIGINAL archived 6,912 physical stochastic source episodes, [source epoch audit](M4_4A_original_source_training_epoch_audit_2026-11-22.json), reported the physical food collection rate by 48-episode epoch and 8 genetic initializations (NOT fresh independent evaluation; these are sampled stochastic on-policy training events):

| Original training epoch | Actual reward-trained RNN | Same RNN with meaningless random reward | Actual reward-trained feedforward |
|---|---:|---:|---:|
| 0 | 54.17% | 53.39% | 50.26% |
| 1 | 52.60% | 52.86% | 51.30% |
| 2 | 56.77% | 48.96% | 49.22% |
| 3 | 64.06% | 43.49% | 45.83% |
| 4 | 80.21% | 49.22% | 53.39% |
| 5 | **82.55%** | **50.00%** | **44.01%** |

The true-RNN early-to-late increase is **+28.39 percentage points** descriptive source-training rate, with **7/8 independent initializations** improving and one worsening slightly. Training uses random action sampling and is not directly comparable in absolute score to frozen argmax heldout tests. The source epoch audit used no new learning, no new independent worlds, no selection, and **did not change preregistered gates**.

## 4. Independent MAIN physical maze test — both prospective MAIN memory/plasticity gates PASS

[Original independent GitHub Actions #45004555499](https://github.com/kkongiu/neuroevolution-lab/actions/runs/45004555499) **SUCCESS**: all regression tests and frozen original source-bank hash verified; exactly **32 NEVER-SEEN MAIN paired corridor geometries 122000–122031**, two opposite physical hidden food goals, eight INDEPENDENT initial numerical networks, eight diagnostic treatments = **4096** MAIN physical world trials. Complete original data [in the permanently archived 6,144 rows](M4_4A_all_6144_original_guided_final_world_trials_2026-11-22.csv).

Each arm has **512 total nested network×world×goal trials**; inferential independent unit is **one of 8 separately initialized trained RNNs**, not 512 independent learning processes.

| Frozen trained numerical agent | Actual physical food collected | Food success rate |
|---|---:|---:|
| **Reward-trained RNN (8 tanh hidden)** | **463/512** | **90.43%** |
| The SAME initial RNN without lifetime reward training | 249/512 | 48.63% |
| SAME trained RNN, hidden-state memory reset immediately before decision | 256/512 | 50.00% |
| SAME original RNN trained on fake unrelated reward | 260/512 | 50.78% |
| Online-trained ~parameter-matched FF network | 256/512 | 50.00% |
| SAME trained RNN, initially observed cue inverted | 49/512 | 9.57% |
| Uniform random binary baseline | 256/512 | 50.00% |
| Researcher-programmed symbolic cue memory | 512/512 | 100.00% |

The 20k **network-initialization-cluster** bootstrap contrasts, computed on per-initialization paired 64 world-goal examples, are:

| Prespecified MAIN paired contrast | Advantage percentage points | Descriptive 95% cluster bootstrap |
|---|---:|---:|
| Reward-trained RNN − its SAME initial untrained RNN | **+41.80 pp** | **+28.52 to +53.32 pp** |
| Reward-trained RNN − reward-shuffled RNN | **+39.65 pp** | **+26.95 to +49.22 pp** |
| Reward-trained RNN − same RNN with hidden memory reset | **+40.43 pp** | **+26.95 to +50.00 pp** |
| Reward-trained RNN − trained FF | +40.43 pp | +26.95 to +50.00 pp |
| Reward-trained RNN − same RNN with cue reversed | +80.86 pp | +53.91 to +100.00 pp |

The predeclared combined MAIN positive gate required **≥65% correct food acquisition, lower95% >0 vs own initial RNN, lower95% >0 vs meaningless random feedback, lower95% >0 vs memory reset, and gains in ≥6/8 independent network initializations**. All conditions passed. In fact all eight original RNN initializations gained relative to own untrained initial RNN, although some gains were substantially smaller than others.

This is the first **demonstrated combined within-learner reward-based numerical synaptic modification + short-term recurrent hidden memory** under matched original physical energy/food dynamics, not a hard-coded animal strategy.

## 5. Independent longer-delay OOD test: conditional recall generalizes to longer corridor stems

Sixteen NEW shifted physical geometries 123000–123015, stems length **5–6** instead of MAIN 3–5, two opposite food goals, eight neural initializations, eight treatment arms = **2048** original shifted physical trials. Each arm has 256 nested outcomes.

| Agent / causal intervention | Actual food successes | Rate |
|---|---:|---:|
| **Reward-trained RNN with learned recurrent memory** | **223/256** | **87.11%** |
| Own original untrained RNN | 127/256 | 49.61% |
| SAME trained RNN memory reset | 128/256 | 50.00% |
| RNN trained with shuffled reward | 134/256 | 52.34% |
| FF trained on actual reward | 128/256 | 50.00% |
| SAME trained RNN with cue flipped | 33/256 | 12.89% |
| Researcher-coded symbolic memory | 256/256 | 100% |

OOD RNN minus SAME-genome memory-reset advantage **+37.11pp**, independent initialization cluster bootstrap95% **[+22.27,+48.44] pp**. OOD RNN minus own initial RNN **+37.50pp**, 95% **[+21.875,+50.78] pp**. The predeclared OOD positive gate required RNN success ≥60% and strictly positive cluster-bootstrap lower95% over memory reset: both pass. This shows bounded generalization to longer *physical-corridor cue-to-decision delay*, NOT arbitrary temporal memory or complex worlds.

## 6. Distinct autonomous full-motor diagnostic: STILL NOT SOLVED

Original M4.2 `simulate()` with all movement decisions now made by the frozen trained/untrained RNN, without any scaffold, was evaluated on **8 other new world geometries 123100–123107** × 2 food sides ×8 initial network seeds ×2 neural interventions = **256 original fully autonomous physical motor trials**. The inference for this exploratory diagnostic was NOT a predeclared main positive gate.

| Fully unguided transfer, same frozen numerical RNN | Real food success | Actually reaches fork |
|---|---:|---:|
| Reward-trained RNN | **54/128 = 42.19%** | **63/128 = 49.22%** |
| Its own original untrained RNN | **22/128 = 17.19%** | **61/128 = 47.66%** |

The learned RNN makes an apparently more food-effective decision in unassisted travel, but fork arrival remains **below half the trials**, and reward-based online updates are OFF in this diagnostic. It does not satisfy the autonomy requirement of issue #26, nor show that the agent can infer, forage and survive in unconstrained random ecologies. The full 256 original unguided motor trial rows are [archived](M4_4A_all_256_original_autonomous_transfer_trials_2026-11-22.csv).

## 7. Final preregistered verdict and boundaries

- **Within-lifetime reward-dependent numerical synaptic updates:** **CONFIRMED** in network weights: exact online REINFORCE+BPTT gradient verified, recurrent matrices changed in all eight independently initialized models, direct reward-vs-shuffled control.
- **Causally useful learned recurrent memory:** **CONFIRMED** for early explicitly given directional cue → no cue for several physically traversed corridor cells → binary real food decision. Same-trained-RNN hidden-state reset destroys the performance advantage, exact matched opposite-goal fork observations.
- **Joint learned plasticity + memory, heldout new geometry:** **MAIN GATE PASS**, 90.43% versus 48.63% initial / 50% reset / 50.78% shuffled reward.
- **Joint plasticity + memory on unseen longer corridors:** **OOD GATE PASS**, 87.11% versus 50% reset.
- **Autonomous embodied navigation with motor chosen by the network:** **NOT SOLVED** (42.19% overall unguided food success, only 49.22% reach fork; diagnostic only).
- **Within-one-unbroken-life online synaptic learning across changing random-food ecologies:** **NOT TESTED** (training episodes reset spatial position, state and energy, preserving trained synapses).
- **Predator avoidance, birth, ecological generation sorting, heritable acquired memories, emergence of intelligence:** **NOT TESTED / NOT PROVEN**. M5.5B/C physical predator track remains a **separate unmerged branch** with different researcher-coded fear behavior.

Interpretive safeguards:
1. The task supplies a simple explicit directional cue, two known terminal gate actions, deterministic food reward, and **externally scripted motor movement until the fork**. This is a strong research control but an artificially favorable setting for learning. No inference to naturally learned arbitrary map memory is warranted.
2. The final score is deterministic argmax over fork choices after all 288 source training episodes; source training used stochastic exploration. The two success rates have distinct measurement regimes.
3. Eight independent seeded neural initializations are the independent training replicates; the 32 or 16 paired physical geometries per model are nested evaluation worlds. Bootstrap intervals resample the **eight** training initializations, NOT individual agent timesteps, and are descriptive, not population-level significance guarantees.
4. Opposite-goal paired worlds give exactly 50% for memory-off FF, hidden-reset and chance controllers by **mathematical construction**. Cue flip is almost the exact complement of the correctly cued success, so these are **mechanistically important but not statistically independent evidence sources**.
5. This M4.4A protocol was frozen before implementing code, the full-precision source bank was committed before evaluating any independent world, and the full original source and target outputs (including failures) are permanently archived. The historical original source-train epoch curves were audited afterwards but **not** used to select weights or rerun final worlds. No after-the-fact hyperparameter retuning occurred.
6. Exact complete computational reproduction from checkout: `python -m unittest discover -s tests -v`; source replay `python -m neuroevolution.m4_4a --phase source` (must only be used without touching final worlds); independent original evaluation `python -m neuroevolution.m4_4a --phase final` using the permanently frozen source-genome bank.

## 8. Next rigorous proposal (NOT performed here)

M4.4B should incrementally remove the **approach guidance**, without changing its trained genotype or hidden-state mechanics and using **new never-before-examined development / validation / final seeds**. Separate (a) *reaching the fork*, (b) *choosing the correct remembered branch conditional on arriving*, (c) *overall real food/energy return*, and (d) *genuine updates from food during full autonomous continuous episodes*. Introduce physically sparse navigation/reward without privileged labeling, retain matched same-genome memory-reset/feedback-shuffle controls, and only then consider predators and births. Freeze gate, equal resource budgets and independent initial-seed cluster bootstrap before looking at results. Do not use these completed M4.4A new worlds for trial-and-error tuning.
