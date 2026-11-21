# M3.3 — Contextual memory, logical compatibility and structural renewal

**Frozen before M3.3 code implementation or evaluation on new final-test seeds.** All past milestones remain unchanged. This is a bounded scientific experiment, not evidence of self-generated goals.

## Why

M3.2 asked a single 2D-input network to remember successive, sometimes contradictory classification rules **without a context identifier**. Two different rules can demand different labels for the same input, so the requirement can be mathematically impossible. M3.3 first tests whether providing a context cue makes mutually contradictory but identifiable tasks learnable, then asks whether replay and adaptive unit renewal help maintain acquired skills.

## Observable world

Four stable, externally signalled contexts (IDs 0–3). The *same* x,y points are labelled with **opposing** answers in each pair:
- contexts 0/1: a randomly oriented linear boundary and its logical inverse;
- contexts 2/3: a randomly shifted XOR boundary and its inverse.

The independent shifted-family test replaces these with **stripe and checker** rules, each paired with its exact inverse. Labels are never contradictory *given (x,y,context)*.

Inputs are 2 coordinates plus **four one-hot context bits**; the blinded control receives all context bits set to zero but has the **same six-input architecture and parameter count**, initial weights, labels, SGD budget and examples. Held-out evaluation for paired contexts uses **identical x,y points but inverted labels**, so a deterministic context-blind model cannot exceed **50% balanced accuracy averaged over all four contexts**. This is a mathematical control, not a discovery to be celebrated as intelligence.

Each independent seed has 12 sequential episodes, with the first four contexts presented in a randomized permutation, and the following eight randomly revisit any of the four. A single model persists over the whole sequence; no deliberate weight damage.

## Replay

A bounded memory buffer stores up to 24 **training examples per previously seen context** (never any holdout examples). On one in four SGD update steps, when at least one different context is already in the buffer, **replace** the current-task mini-batch with samples from previously seen contexts, chosen deterministically from the buffer. This does **not increase SGD update count**, but reduces current-context examples, a necessary tradeoff. Replay bank content and sampling must be identical across all replay-enabled arms in a seed. The no-replay ablation uses the full current stream at every step.

The context cue is an externally supplied signal, not inferred by the network. This benchmark tests *cued memory*, not unsupervised context discovery.

## Prespecified sizes and data partitions

- Meta-training seeds **20000–20003** (4), meta-validation **21000–21001** (2).
- Main sealed test **22000–22023** (24 independent seeds).
- Shifted-family sealed test **23000–23011** (12 independent seeds).
- 4 contexts, 12 episodes, hidden width 20, 64 SGD updates/episode, batch size 4, learning rate 0.04.
- 64 independent held-out examples/context, evaluated on paired x,y panels, plus training-only probes.
- Learning-curve AUC measurements at update steps **0, 16, 32, 48, 64**.
- Replay buffer 24 training examples per seen context; replay step frequency **every fourth SGD update**.
- Neural renewal operator: replace **3 least-active hidden units** on the *preceding episode's training inputs*.
- All competitive schedulers must spend **exactly 3 renewal events across 11 boundaries** (9 unit replacements, repeat selections possible).
- Meta-search: 6 policy candidates, 3 generations, numerical mutation RNG seed **971**, separately for state-aware, accuracy-only and time-only policies. The winning generation leader is selected on meta-validation seeds. Fitness: mean within-episode learning-curve AUC (episodes 1–11) **plus 0.20 × mean after-episode accuracy over previously seen contexts**. This memory-weighted fitness is fixed before sealed testing.

## Competing experimental arms

**Core contextual-memory controls** (same capacity and SGD update counts):
1. cue + replay + no structural renewal;
2. **context blind** + replay + no renewal;
3. cue + **no replay** + no renewal.

**Fixed-renewal-budget controls**, all receiving the cue and replay:
4. state-aware evolved renewal timing (sees train accuracy for current context, mean accuracy on other contexts' TRAINING replay buffer, hidden inactive fraction and elapsed episode index);
5. independently optimized **accuracy-only** (current train accuracy + time; no retention or hidden-state features);
6. independently optimized **time-only** (episode index only);
7. evenly spaced fixed 3 renewal events;
8. adaptive events temporally shuffled (exactly 3 events);
9. adaptive event times but three **random** units instead of least-active.

All competitive arms are seeded identically and use identical task order, labels, replay streams, architecture, learning rates and SGD steps. Arms 1–3 have no renewal and are diagnostic; they are **not dose-matched to arms 4–9**.

## Outcomes and preregistered gates

**Primary memory feasibility comparison:** paired seed-level difference in **final balanced four-context held-out accuracy**, cue+replay+no-renewal **minus** blinded+replay+no-renewal, on the 24 main final seeds. Report seed-level paired bootstrap 95% descriptive interval and signs. The blinded arm must perform at exactly 50% on the explicitly contradictory paired panels. Positive context-cue effect establishes only feasibility with a *provided context label*.

**Secondary retention mechanism:** cue+replay+no-renewal vs cue+no-replay+no-renewal on final balanced four-context accuracy and average worst-context accuracy. If replay does not improve retention, report that fact.

**Exploratory adaptive-plasticity gate (declared before test):** on the same 24 seed sequences, state-aware + cue + replay **vs independently trained time-only + cue + replay**, using paired episode learning AUC and final balanced contextual retention. An adaptive-benefit claim additionally requires positive mean AUC over accuracy-only, positive mean vs shuffled matched intervention schedule, and evidence of state-dependent *unforced* renewal requests. If not, adaptive renewal is **unconfirmed** even if cue or replay helps.

Additional diagnostics: final accuracy on each context; balanced accuracy vs chance, minimum individual context accuracy, old-context forgetting measured against its historical peak on heldout panels; learning AUC; novel first-visit vs return episodes; actual/forced actions, unit replacement counts, CPU runtime. All CI intervals resample **entire independent seeds**, not episodes/examples.

**Shifted-family test** checks the *same frozen policy* on stripe/checker oppositions using new seed range. This is a partial distribution shift, not a claim of open-ended generalization. No hyperparameter/threshold tuning after sealed test.

## Safety and interpretive boundaries

The system has **no autonomous code execution, network access, self-editing, GitHub credentials or objective invention**. Policies are numeric vectors evolved externally; context is provided by the experimenter. The blind model's 50% ceiling is designed into the task and does not prove deep learning capability. Conflict-free memory does not imply a general solution to catastrophic forgetting. Any negative or null result must be archived with equal prominence.
