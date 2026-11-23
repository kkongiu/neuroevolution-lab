# M5.34 verified results — shared learned representation transfers to unseen whole functions, with critical boundaries

**23 November 2026 | Official [GitHub Actions science run #45127191748](https://github.com/kkongiu/neuroevolution-lab/actions/runs/45127191748) SUCCESS.** The full M5.34 learning experiment has finished. Tests checking exact function-level train/eval exclusion, reward-only support encoding, identity and fairness of cue selection, frozen neural inference, old M5.33 no-leak behavior and original M5.6 online plasticity regressions all passed.

**Prospective narrow DEV screening checks: 11 / 11 PASS.** This is meaningful evidence of **conditional amortized neural inference transferring from distinct previous synthetic functions to 27 unseen complete functions** within the SAME familiar mathematically programmed task generator. It is *not* independent task-family transfer, neural invention, active learned curiosity or an ecological organism.

The [M5.34 protocol](M5_34_LEARNED_REPRESENTATION_PROTOCOL_2026-11-23.md) and its thresholds were frozen before the full GitHub Actions scientific results were available. First CI test run failed due to a wrong training-gradient variable name (\`target\` vs \`targets\`), corrected as a genuine code/test fix without changing the frozen cohorts, model architecture, pretraining recipe or gates. Subsequently the completed run is #45127191748.

## The intervention and hard function split

A 32-input →64-tanh-hidden→16-tanh-output MLP has **3,152 genuinely float32 parameters**. Its input is ONLY (a) a length-16 partial observed target vector decoded from the eight actually chosen binary actions and binary rewards using the deterministic two-action complement property and (b) an independent length-16 observation mask. No named parity/AND rule feature, experiment target function, test outcome, symbolic oracle posterior or test ID is supplied to the neural learner. Hidden features and all weights are **shared across tasks**, without any test-function gradient fine-tuning.

**SPLIT_SEED=3401:** all 112 distinct pre-known synthetic full truth tables were stratified into **85 TRAIN function signatures** and **27 EVAL completely disjoint function signatures**:
- plain parity: 7 train / **1 eval**;
- parity + one pairwise conjunction: 36 train / **12 eval**;
- parity + one triple conjunction: 24 train / **8 eval**;
- parity + XOR of two disjoint pairwise conjunctions: 18 train / **6 eval**.

These are unseen **function IDs/truth signatures**, NOT new families or independently sourced ecologies. Pretraining still uses the same researcher-supplied Boolean function generator, so the result remains synthetic, task-distribution-privileged, and cannot be extrapolated to real-world concept formation.

Two meta optimizer seeds **1301, 1303**, exactly **3 support episodes per heldout function** under each of 2 cue-selection policies (random and researcher-designed known-family entropy acquisition): **27×3×2×2 = 324 paired neural evaluation rows**, each with 8 genuine support action/reward experiences and 8 disjoint, never observed cue queries. All compared learners see precisely the same episode supports, exam cue IDs, actions and reward observations. Accuracy below is **only on eight genuinely unseen cues**, not the mixture of support and unseen used as M5.33's primary metric.

### 5 controlled arms

1. **Learned shared meta representation:** 3,000 Adam steps ×32 sampled episodes per step **per initialization** (~96,000 training episodes) drawn only from the 85 TRAIN functions. Training query labels are full labels from TRAIN functions only, which is **substantial supervised prior exposure** and must not be confused with reward-only single-lifetime learning.
2. **Training-corruption null:** EXACT same 3,152-weight architecture, initialization, support-input sampling, 3,000 Adam steps and training budget, but training query targets are independently paired with a DIFFERENT TRAIN function. Breaks support→query relations while leaving the total training exposure and work matched.
3. **Scratch neural null:** identical untrained 3,152-weight model; parameter-matched but NOT pretraining-compute matched.
4. **M5.32 independent dual soft neuron:** original 15-parameter two-conjunction neural model fit for 8×300 SGD iterations separately from each test task's actual support records, no diversity penalty. It is not parameter-, training-data- or compute-matched to the shared pretrained 3,152-weight MLP.
5. **Human-known 112-rule symbolic posterior:** explicitly UNFAIR reference containing the correct family and possibly full heldout function in its available prior. Never passed as candidate neural features, labels or model initialization.

The two sampling policies are held constant *within each episode comparison*. The active cue policy is still human-coded, knows the finite grammar, and is not an autonomous curiosity policy.

## The frozen unseen-eight-cue results

| Distinct heldout function family | Learned shared representation | Scrambled pretraining | Untrained same MLP | M5.32 dual-soft | UNFAIR correct-family symbolic reference |
|---|---:|---:|---:|---:|---:|
| Plain parity (**only 1 function heldout**) | 73.96% | 34.38% | 48.96% | **80.21%** | 87.50% |
| Pairwise AND (**12 functions**) | **71.18%** | 46.79% | 47.92% | 65.97% | 93.75% |
| Triple AND (**8 functions**) | **87.37%** | 43.36% | 47.66% | 76.95% | 95.57% |
| Two disjoint ANDs (**6 functions**) | **61.81%** | 43.92% | 53.47% | 46.70% | 96.18% |

**Macro mean of the four family accuracies for the shared meta network: 73.58%.** Macro mean is unweighted by family size; individual-function uncertainty can be high, especially with only one heldout parity signature. All 27 unseen function signatures were strictly excluded from supervised meta pretraining and null pretraining.

### Primary previously problematic TWO-disjoint-AND cohort

- Shared meta: **61.8056%** unseen.
- M5.32 within-task dual-soft: **46.7014%**. **+15.10 percentage points** for shared meta (exploratory whole-function-cluster bootstrap 95% **[+2.60,+26.39] pp**), based on only **6 heldout functions** and correlated episode/optimizer repeats.
- Identically pretrained *but deliberately uninformative* scrambled mapping: **43.9236%**, hence genuine learned mapping gains **+17.88pp** (exploratory whole-function-cluster bootstrap **[+4.34,+30.90] pp**).
- Same untrained network: **53.4722%**, hence +8.33pp. Exploratory bootstrap **[−3.82,+18.75] pp** crosses zero: **do NOT claim conclusive superiority over untrained random weights**, despite the preregistered point-estimate +8pp gate passing.
- Strong human-programmed family oracle: **96.18%** unseen; a large gap remains.
- Active sampling with the researcher-defined 112-rule grammar: shared meta **61.11%** unseen; random sampling: **62.50%** unseen. **Meta learning did not make active cue sampling superior** on this two-pair cohort. This issue remains unresolved.

### Other families and negative results

- Pair AND: shared meta 71.18% vs M5.32 65.97%, a +5.21pp point gain; exploratory whole-function-cluster bootstrap **[−0.35,+10.85] pp** includes zero.
- Triple AND: shared meta **87.37%** vs M5.32 **76.95%**, +10.42pp, exploratory interval **[+2.21,+19.40]pp**.
- Parity: shared meta **73.96%**, notably *below* M5.32's **80.21%** on the **one heldout whole parity function**. This is a negative result for simple parity and the one-function sample makes inferential uncertainty essentially unidentifiable.
- Shared meta all-16 accuracy with eight supports included is **80.90%** on two-pair tasks, versus only **61.81%** on genuinely unseen cues; the above analysis therefore uses the **unseen-only** number as the primary measure.

The unsupervised/unaligned null and the pretrained representation are both exposed to the same architecture, but results must not be interpreted as a matched-compute comparison against the original local M5.32 dual-soft brain: the shared model had enormous pretraining data and >200× more parameters.

## The 11 prospectively frozen DEV screening checks

1. TWO-pair shared-meta unseen8 >=60%: **PASS**, 61.81%.
2. TWO-pair gain over scrambled meta >=+8pp: **PASS**, +17.88pp.
3. TWO-pair gain over same-architecture untrained MLP >=+8pp: **PASS**, +8.33pp (**bootstrap crosses zero**).
4. TWO-pair gain over M5.32 dual-soft >=+8pp: **PASS**, +15.10pp.
5. Macro mean over four known families >=70%: **PASS**, 73.58%.
6. Pair AND unseen8 >=70%: **PASS**, 71.18%.
7. Triple AND unseen8 >=75%: **PASS**, 87.37%.
8. Exact TRAIN/EVAL complete truth signatures disjoint, 85 vs27: **PASS**.
9. Eight authentic chosen action/reward supports and eight disjoint frozen exam cues: **PASS**.
10. No heldout functions used in supervised meta fit; corrupted target association uses same pretraining steps and inputs: **PASS**.
11. All arms, two sampling schemes, all negatives and model checkpoint weights recorded, no-evolution flags: **PASS**.

**11/11 passes a deliberately narrow exploratory DEV screen, not 11 independent demonstrations of cognition, independent replication or self-evolution.** No threshold was adjusted after seeing the score. Exploratory intervals cluster by test **full truth-function identity**; the 3 episodes and two initialization seeds are repeated tests of those identities, not 324 independent unrelated problems.

## Data and exact reproducibility

**Verified official CI science run:** [#45127191748](https://github.com/kkongiu/neuroevolution-lab/actions/runs/45127191748), all scientific tests and original M5.6 regressions passing.

Full Actions artifact \`m5-34-shared-representation-dev\`,
ID **14796545779**, 30-day retention through **2026-12-23**.
Its downloaded ZIP was independently opened and parsed:
- **324 trial rows** and **exactly 27 distinct never-trained function signatures**;
- two full learned meta representations plus two scrambled training controls, each with all *3,152 actual float32 weights* included in saved checkpoints;
- all heldout functions, complete eight-action/reward supports and eight unseen exam IDs, each learning arm's frozen predictions, full scores and hidden activation norms;
- complete prospective 11 screening checks and all negative results;
- compact canonical raw-row SHA-256 **\`14cc91b0b4716c24de86e20b06e7fa5474f6454292566f1be61849a84e87600b\`**, independently recomputed from the ZIP, EXACT match with run field.

Shared learned model per pretrained initialization consumes **3,152 ×4 =12,608 bytes** (~12.3KiB) in float32 weights alone (excluding workspace and sampled training episodes); the M5.32 within-task model uses only 60 bytes per context *after* its own fit but with separate unreported optimizer memory/work. No claim of equal cost, minimal agent memory or energy advantage.

## Scientific disposition and next worthwhile test

**Supported:** a supervised neural representation trained across **distinct earlier synthetic functions** generalizes to entire excluded test functions more usefully than scrambled-source transfer and an independently adapted earlier dual neuron in hard two-pair and triple-AND tasks. This supports investigating *knowledge transfer / learned representation*, not adding more neurons.

**Not supported:** autonomous operator invention, curiosity learned from environment, direct unassisted one-life self-modification, independent unknown task-family transfer, robust parity ability, resource efficiency or ecological survival.

The following remain hard-coded FALSE even with the positive narrow screen:
- \`learned_general_curiosity_policy=false\`
- \`autonomously_invented_neural_operator=false\`
- \`shared_meta_plastic_controller_learned=false\`
- \`independent_task_family_replication_complete=false\`
- \`continual_memory_retention_verified=false\`
- \`evolution_authorized=false\`
- \`advance_to_evolution=false\`
- \`sealed_M5_6_ecology_accessed=false\`

The PR remains DRAFT stacked on M5.33, original sealed M5.6 ecology untouched and no merge to \`main\`.

**Proposed M5.35, NOT IMPLEMENTED:** a complete family-holdout transfer challenge with completely new synthetic structure generators, explicit matching of pretraining exposure and memory, and an experiment on sequential updating without catastrophic forgetting. Only then should we reconsider statements about a self-learning organism.
