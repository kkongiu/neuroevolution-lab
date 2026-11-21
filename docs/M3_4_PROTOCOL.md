# M3.4 — Infer latent task context from labelled observations, not an oracle ID

**Preregistration:** written before the first complete M3.4 implementation and before all M3.4 final-test evaluation. Keep M3.3 source and results unchanged.

## Hypothesis

Can a bounded, online **associative context resolver** infer which previously learned rule has returned using only a small, separately sampled **labelled support set**, and supply a useful context cue to a persistent network without ever receiving the experimenter's latent task ID?

This is a test of *externally programmed* episodic recognition and replay. It is **not** autonomous discovery of a learning algorithm, human-like self-awareness, or neural source rewriting.

## World and information barrier

Four fixed but unknown latent contexts occur across **12 sequential episodes**. The first four episodes contain each context exactly once, in a randomly permuted order; the next eight select one of those contexts randomly. The learner **does not receive** context ID, novelty/return flag, or the guarantee that the first four are distinct. IDs may appear **only in the test harness** to generate labels and to audit inferred clustering after decisions.

For context pairs 0/1 and 2/3, labels are exact logical inverses on matched x,y panels. In the main world the base rules are **line and XOR**. In the shifted-world test the base rules are **stripe and checker**. Thus given (x,y) alone, a context-blind predictor cannot exceed **50% final balanced accuracy** on the paired held-out panels. A correct hidden task ID is never an input to the deployable context resolver or its neural predictor.

Before every episode, the environment emits a **labelled support set** sampled independently of all SGD training and held-out panels. Support is provided to every resolver and is the *only* new information with which to choose a context slot for the episode. It has the form (x, y, observed label); the labels are ordinary observation/feedback, **not final holdout labels**.

The resolver may allocate up to four anonymous slots, matching support labels to **k-nearest-neighbour predictions derived only from past training examples** stored in slot prototypes. For each candidate slot it predicts support labels using k=7 past nearest neighbours (uniform majority vote); its score is fraction of support labels matched. If best score is below a meta-selected acceptance threshold and a free slot exists, it allocates a new slot; otherwise chooses the best-scoring existing slot (stable numeric tie break). In the absence of any prototype, allocate slot zero. The resolution algorithm **does not know that the first four episodes are distinct** and may merge or split contexts incorrectly.

After episode SGD ends, its first 64 **TRAINING** samples (not support/holdout) update the selected slot's FIFO prototype. These 64 samples are the only prototypes used for recognition. The resolver does **not** access neural weights, final holdout labels, hidden latent context IDs or the intervention/outcome metrics.

For a final memory test on each latent context, independently draw a **new support set** and infer the slot *without changing the prototype bank*. Evaluate the persistent network on a completely independent paired test panel with the inferred slot cue. The final support labels are available as test-time observations, but final holdout panel labels are never used for inference.

## Locked partitions, optimization and world shift

- Meta-training seed sequences: **24000–24005** (6).
- Meta-validation sequences: **25000–25002** (3).
- Sealed main test: **26000–26023** (24).
- Sealed reduced-support same-family test: **27000–27011** (12).
- Sealed family-shift plus reduced-support test: **28000–28011** (12).
- All seed ranges disjoint. Meta-tuning only selects the resolver acceptance threshold from **[0.60, 0.68, 0.76, 0.84]**.
- Meta-score: **balanced pairwise clustering accuracy** on episode assignments (average same-context recall and different-context rejection). The two highest-scoring training thresholds advance to meta-validation; choose the highest meta-validation score, with stable candidate-order tie breaks. For meta-learning *only*, the **true latent IDs may be used for offline fitness**, never for runtime inference or final-test policy selection.
- Persistent neural classifier: 24 hidden ReLU neurons, **6 inputs** (x, y, 4 inferred-slot one-hot bits); same parameter count for blank-context controls.
- Per episode: 48 SGD updates × batch of 4, learning rate 0.04. Heldout panel: 64 x,y points per context, paired and inverse-labelled within each context pair.
- SGD learning-curve accuracy is measured at updates 0/12/24/36/48.
- **Main** meta-training/validation/main final: 16 clean labelled support observations per episode and per final context. 
- **Few-shot** test: the same line/XOR base families, but **6 support observations with exactly one label flipped** per episode/final context. Models, threshold and memory limits unchanged.
- **Shifted** test: stripe/checker base families with the same six-support/one-flip condition. The two additional test groups distinguish support degradation from task-family change, although other differences may remain.
- Supports, SGD examples and holdout panels have disjoint pseudorandom streams and use different sampled points. The first four distinct contexts are part of the *world*, not an explicit signal provided to the resolver.

## Fixed replay budget and comparisons

All deployed arms receive identical task order, support streams, labelled SGD training samples, initial network weights, number of SGD steps and model architecture. A memory bank stores up to **24 training examples per inferred slot**, maximum **96 stored examples**. Every fourth SGD update **replaces** the current-task batch with a batch sampled from a *different inferred slot's* stored training examples, if available, with a deterministic slot rotation. Replacing rather than adding SGD steps makes the optimization step count identical, but reduces exposure to current-task examples; report this tradeoff.

Five independently trained neural trajectories per seed:

1. **inferred_fifo**: slots inferred from support; FIFO buffer, replay.
2. **inferred_reservoir**: **the same inferred slots** as #1, reservoir-sampled buffer of identical size, replay.
3. **inferred_no_replay**: same inferred slots and model, no replay.
4. **oracle_context**: latent ID supplied to neural model and replay grouping **only as a non-deployable reference**; exact same model capacity/SGD steps/buffer limit.
5. **no_context_replay**: neural inputs contain four zero context bits, but replay still groups by the **same inferred slots as #1**, to isolate whether the cue itself carries predictive value.

All inferred arms share a **single deterministic resolver trace** per seed, computed independently of network learning, with no access to future supports. This design deliberately isolates inference and replay: differences among #1–3 cannot come from different context assignments. Offline oracle may exploit privileged IDs and should not be treated as a comparable deployable algorithm.

**Paired inference ablations on the *same* trained #1 neural model, no retraining:**
- **shuffled_support:** cyclically rotate the four *final support sets* between contexts before inferring cues; source and target panels are never revealed to the resolver; fixed one-position rotation.
- **no_support:** replace all four inferred slot cues with zero vectors.
Both share exactly the same learned weights, heldout samples and final evaluation budget as #1. Shuffled support is a test-time intervention on information, not a separately trained model.

**Buffer retention controls**: FIFO vs reservoir at exactly equal slot capacities, replay sampling frequency and SGD counts; this is a *memory selection* comparison, not an adaptive replay controller learned by the network.

## Prespecified outcomes and gates

**Primary:** main sealed test paired **final balanced four-context heldout accuracy** of inferred_fifo minus its *same-network* shuffled_support ablation, with seed-level percentile bootstrap descriptive 95% interval and number of positive/negative seeds. To claim a meaningful context-inference benefit also require:
- inferred_fifo above **50%** balanced no-support/no-context reference, and
- >0.65 return-episode slot-consistency rate **and** >=3.5 distinct slots on average after first four novel episodes (preventing degenerate assign-all-to-one-slot strategies).

**Secondary:** inferred_fifo vs inferred_no_replay and inferred_reservoir on final balanced memory, worst-context accuracy, mean past-context forgetting and within-episode learning AUC. Replay may improve retention at the expense of current-task learning; this tradeoff must be explicit.

**Context inference metrics, separate from predictive performance:** mean number of distinct first-visit slots, return-to-original-slot consistency, balanced pairwise clustering accuracy, final-support slot consistency against first-visit assignments, all computed for audit **after** decisions using true IDs. A high classification accuracy cannot substitute for successful identity inference.

**Transfer:** repeat *the frozen threshold and replay schedules* on the 12 few-shot and 12 shifted seed sets. Do not redefine a primary result after observing a secondary winner. Report all null/negative outcomes and any evidence that few support observations or label noise degrade inference.

**Limits:** this benchmark supplies labelled feedback, uses externally specified kNN and slot allocation, and does **not** make the neural model discover context entirely unaided. With zero support information, opposite label pairs remain logically impossible to distinguish. Inference overhead (kNN prototypes plus replay bank) must not be called compute-equivalent to the oracle or blank-context controls: report storage and decision-probe work separately. All comparisons with equal SGD counts still differ in support-processing and replay overhead.

Do not modify M3.3 or retroactively change the threshold grid, latent task families, noise, metrics, or seed splits after sealed results. No generated code execution or network activity occurs inside the simulated learners.
