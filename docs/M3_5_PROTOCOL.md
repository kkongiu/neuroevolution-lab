# M3.5 — Value of information: when to request additional feedback

**Preregistered before implementing M3.5 and before looking at M3.5 held-out seeds.** All M3.4 source code, data and conclusions remain unchanged.

## Hypothesis

Can a bounded context resolver, already receiving a small labelled support set, improve **future recognition and prediction** by requesting two extra labelled examples **only at uncertain moments**? Does this outperform random or periodic requests, using an *identical global information budget*?

This is an **externally programmed active-information policy**, not an agent spontaneously inventing goals or code.

## Fixed world and information barrier

Reuse M3.4's **four contradictory latent task contexts** with a single persistent ReLU neural classifier, 12 sequential episodes, and a resolver based on 7-nearest-neighbour matches against **prior training examples only**. The first four episodes visit all contexts once, in hidden random order; the following eight revisit contexts. The learner does not receive the latent context ID or novel/return flag.

Before each episode, all policies receive **six labelled support examples** from a stream disjoint from SGD-training and holdout; **exactly one** support label is intentionally corrupted. The resolver scores each existing anonymous slot by how accurately its kNN predictor classifies those six examples. A policy may request **two fresh IID labelled observations** from a separate stream, independent of training, base support and heldout panels, BEFORE selecting a slot. These two examples are available ONLY to strategies that requested them. The policy cannot choose their x,y positions: **this tests when to ask, not which point to query**.

Each competitive strategy must request on **exactly four of twelve episodes**: **eight supplementary labelled examples in total** per seed, in addition to the same 72 base support examples. Quota completion may force a late query; requests denied because quota is already spent and forced requests are logged separately. The first episode has no reference prototypes: uncertainty based on candidate agreement is undefined and is represented with score zero (a query may still be forced by quota). The prospective extra examples and hidden IDs are **not accessible when deciding whether to ask**.

The resolver may allocate at most four anonymous slots and stores up to **64 TRAINING examples per slot** as identification prototypes. During SGD learning, each neural model may store up to **24 TRAINING examples per inferred slot** for FIFO replay, replacing (never adding) **one out of four** SGD minibatches with old training samples. Extra labelled query observations are used **only for context resolution**, not as additional SGD examples or replay memory.

All matched arms receive identical SGD sample streams, starting weights, architecture, 48 SGD steps × batch 4 per episode, and the same query-count budget. Differences in assigned context slots may lead to different downstream replay groups; this is a genuine consequence of context decisions, not a separate treatment.

## Preregistered querying policies

1. **margin_uncertainty** (primary): compute scores for each candidate slot from current six-label support, before any query. With >=2 candidates use uncertainty = 1 − (top score − runner-up score); with 1 candidate use 1 − top score; with 0 candidates use 0. Request 2 additional labels if uncertainty >= selected threshold and quota permits; force remaining requests when remaining episode opportunities equal remaining quota.
2. **entropy_uncertainty**: normalized Shannon entropy of softmax candidate match scores (temperature fixed at inverse temperature 6), 0 with fewer than 2 candidates. Request if entropy >= independently selected threshold; the same four-request quota and forced completion apply.
3. **random**: independently of context, choose a uniform subset of four episode positions from twelve using an isolated fixed RNG stream per seed.
4. **periodic**: four requests evenly spaced at fixed positions (3rd, 6th, 9th, 12th episodes).
5. **no_query**: never request; **not** matched cost, a diagnostic only.
6. **query_all**: request on all twelve episodes (24 extra labelled examples), a **higher-cost diagnostic**, not a fair competitor.

All six policies use the same M3.4 FIFO neural architecture, capacity, replay budget, and resolver admission threshold **0.84**, frozen from M3.4; do **not** optimize this threshold on M3.5 final data. The 'random' and 'periodic' policies do not see uncertainty. Each policy's resolver and neural network run independently but use the same world/seed.

To measure whether adaptive policies actually exploit uncertainty, log initial scores, original threshold requests, forced/capped queries, return-slot consistency, number of first-visit distinct slots, final-slot consistency, and total extra labels. Hidden task IDs enter **only post-hoc audit and meta-training policy selection**, not runtime decisions.

## Meta-selection, seed splits and fixed sizes

- Meta-train: **29000–29005** (6 independent sequences), meta-validation: **29500–29502** (3).
- Main sealed test: **30000–30023** (24 sequences), familiar line/XOR contexts, six base supports with one flipped label, two **clean** extra labels if requested.
- Shifted-family test: **31000–31011** (12), stripe/checker contexts, same six noisy base supports and clean extras.
- Corrupted-extra test: **32000–32011** (12), line/XOR contexts, same base support, plus one intentionally corrupted label among the two extras when requested.
- Model: 24 ReLU hidden units; six neural inputs x, y, and four inferred anonymous one-hot slot bits. Four latent contexts, 12 episodes; holdout panels with 64 points per context, paired x,y with inverted labels in each opposing pair.
- 48 SGD steps × 4 examples per episode; learning rate 0.04; evaluations at 0/12/24/36/48. Replay every 4 SGD updates using at most 24 training examples per inferred slot.
- Context identification prototypes: FIFO 64 training examples per slot; kNN k=7; **no holdout examples** enter training, querying, or memory.
- Optimize margin trigger from grid **[0.25, 0.40, 0.55, 0.70, 0.85]** and entropy trigger independently from **[0.30, 0.45, 0.60, 0.75, 0.90]**.
- Meta-search objective (computed only on meta-train seeds): mean **balanced pairwise clustering accuracy of episode slot assignments**, using true IDs **offline for fitness only**. Top two thresholds per policy on train advance to independent meta-validation; best meta-validation wins, stable grid-order tie break. **Do not** use main/family-shift/corrupted-extra seeds in selection.
- Each policy always spends exactly four requests in meta-selection too; cost is a constant and cannot be exploited by doing fewer queries.

## Prespecified scientific gates

**PRIMARY:** On the **24 main heldout seeds**, mean paired difference in **final balanced four-context prediction accuracy** `margin_uncertainty − random`, at identical **eight extra labels per seed**. Report percentile paired bootstrap 95% descriptive intervals and signs **resampling entire seeds**, not episodes or individual support points. The gain must be positive and the interval must exclude zero for a clear empirical advantage.

**Necessary secondary:** positive final balanced accuracy versus periodic requests; this must not contradict the main claim. Also report:
- improvement versus no_query (but this is *not* a cost-matched comparison);
- inference **return-slot consistency**, pairwise cluster accuracy, first-visit distinct slots and final-support consistency, measured separately from prediction accuracy;
- **gain per additional labelled query** (differences divided by 8, the fixed extra-label budget), and total support-processing / kNN-vote overhead;
- worst-context accuracy, forgetting and mean within-episode learning AUC, to catch cases where one context's skill is sacrificed for a high mean.

**Entropy policy:** secondary independent heuristic at identical four-request quota; do not select the best-looking policy after seeing heldout seeds.

**Transfer gates:** apply **exactly the same preselected thresholds** without retuning to 12 shifted-family and 12 corrupted-extra seeds. A main-only success is *not* robust adaptive information seeking.

**Ablations and controls:** no additional queries, all episodes queried (higher cost), periodic, random. The experimental ID itself, the label-generating rules, any future query labels, and all holdout panel labels are inaccessible to the runtime resolver. A mutation of future query labels must **not change the current query action**; only a prior requested query can change later prototype states.

An observed difference on this synthetic world only demonstrates a **programmed rule for spending a fixed information budget**. It does not establish free-form active learning, choice of optimal spatial points, autonomous purpose, or consciousness.

No retroactive changes of quota, threshold grid, seed groups, extra-label noise, metrics or baseline definitions after looking at final outcomes. Publish negative/null outcomes with equal prominence.
