# M5.4A — Shortened food-odor perception is the dominant cause of neural ecological extinction

**Research milestone complete, 21 November 2026.** This is a rigorously preregistered **factorial ecological diagnosis**, **NOT** demonstration of new adaptive evolution.

- [Frozen protocol](../docs/M5_4A_PROTOCOL.md), preregistered commit **`d9beaedff7f43f2acbba8e6cd8c29b698b47d42b`** before M5.4A code or any new-world results.
- [Original successful read-only Actions experiment #44926123502](https://github.com/kkongiu/neuroevolution-lab/actions/runs/44926123502), **174 regression tests green**, **2,112 full 400-tick environment×policy outcomes** completed.
- [Permanent COMPLETE 2,112-trial world-level numeric outcome table](M5_4A_all_2112_world_factorial_outcomes_2026-11-21.csv).
- [Permanent verified 24 matched wall-map, food, founder, renewal candidate pool and preselected numeric genotype digests](M5_4A_paired_map_geometry_sha_2026-11-21.json), independently reconstructed in [read-only geometry-only Actions audit #44927320481](https://github.com/kkongiu/neuroevolution-lab/actions/runs/44927320481) with **zero** new organism testing. All 24 environments were accepted on first geometry-generation attempt (max attempt = 0).
- [Full numerical factorial results, preregistered 20k/5k crossed bootstrap, main and validation, scripted and original founder controls](M5_4A_factorial_stats_2026-11-21.json).
- Original GitHub Actions artifact **`m5-4a-paired-ecology-factorial`, ID 14715663924** (30-day retention) contains full `world_metrics.csv`, `summary.json`, original numerical founder genotypes, matched map/food/founder hashes `coupled_world_geometry.json`, checkpoint population trajectories, all first-world event ledgers and `factorial_survival.svg`.

## 1. Research question and design: three truly separable stressors

Previous M5.3C held-out genotype transplants showed **110/120** selected-descendant numerical genomes and **104/120** their actual random founders survive ordinary worlds, but **0/60 descendants and 0/60 ancestors** survive harder shifted worlds. All three ecological axes had changed simultaneously, so that loss could not be attributed to any one factor.

Frozen M5.4A tested the complete **2×2×2 environmental factorial**:

- WALLS: obstacle density **0.12** (benign) vs **0.20** (stressed).
- ODOR: directional food-odor sensor radius **5** vs **2 cells**.
- REGROWTH: per candidate food regeneration success probability **0.55** vs **0.40**.

**Physical pairing is intentional, not the original map generator:** random uniform draws are sampled once for each interior cell to create **nested low/high wall maps**. The common high-wall-reachable component supplies exactly the same 20 initial food sites, 20 distinct initial organism positions and identical presampled candidate resource cells to all 8 treatment combinations. Renewal marks/candidate locations are exactly paired, so .40 candidate events are a literal subset of .55 events. The original unchanged M5.0 food occupancy, physical birth, energy and 15-sensor/5-action neural motor rules apply in every treatment.

**Estimand caveat:** low-wall-only accessible cells are open for movement but never special regeneration candidate cells; this identifies an independently manipulated *wall effect under fixed common food support*, NOT the complete effect of asking the historical world generator to independently resample whole maps/food/founders at .12 vs .20.

### Controls and independence

Ten preexisting numerical genotypes: descendants and their real ancestors for each of five historically source-selected M5.3B families **82006, 82015, 82018, 82019, 82023**. Each numerical 401-weight genotype is held constant in 20 artificial founders and cloned exactly at genuine in-world births, no new mutation or training. All five motor/reproductive action decisions are native **unaltered** M5.0 neural logits; original neural architecture's 10-to-401 fixed directional wiring is researcher-designed, not evolved.

The eleventh **scripted_greedy_novelty** positive physical controller is the original M5.1 human-written local food-odor + visited-cell memory search, with energy-threshold scripted birth. **Explicitly NOT evolved, learned or neural**. It tests whether the changed ecology is in principle occupiable by a bounded-information forager.

- **MAIN** fresh seeds **86000–86015**, 16 world environments × 8 factorial conditions × 11 policy genotypes/controls = **1,408 full world trials**.
- **VALIDATION** new independent seeds **87000–87007**, 8 environments ×8×11 = **704 more full world trials**, run without retuning any numeric genome/physical factor.
- The experimental unit for neural inference is five previously successful **source family genes** crossed with 16/8 new independent world environments, not 1,408 independent evolutionary histories. Survival is measured at tick **400**.
- Primary matched survival factor effects average stressed−benign over all **four levels of the other two factors**, not one opportunistically cherry-picked corner. 20,000 deterministic crossed source-family × world-resampling bootstraps for main, 5,000 independent confirmation resamples. Conditional on five preselected original source-family successes.

## 2. Decisive PRIMARY MAIN result: ODOR SENSORY RADIUS

Neural-descendant marginal survival effects (absolute percentage points, stress−benign across all other factorial combinations):

| Independently manipulated factor | MAIN marginal survival impact | Descriptive 95% crossed bootstrap | Predeclared harmful |
|---|---:|---|---|
| **ODOR radius 5→2** | **−81.25 pp** | **[−92.81,−63.75] pp** | **YES** |
| FOOD REGROWTH .55→.40 | **−7.50 pp** | **[−12.81,−2.81] pp** | YES, secondary |
| WALL density .12→.20 | **−0.625 pp** | **[−6.25,+4.69] pp** | NO |

**Preregistered unique dominant-factor gate PASSES.** The mean odor effect is substantially worse than either independent resource-renewal or wall obstacle effect, **even after** averaging across the other four-factor contexts:

- ODOR minus FOOD REGROWTH marginal harm: **−73.75 pp**, descriptive paired crossed 95% **[−88.75,−53.13] pp**.
- ODOR minus WALL marginal harm: **−80.625 pp**, descriptive paired crossed 95% **[−93.75,−60.63] pp**.

Both comparison confidence intervals have strictly negative upper bounds, exactly as preregistered; no relaxed gates, revised seeds, modified mutation rates or external neural teaching were used. The order of effect magnitude is **odor ≫ regrowth ≫ wall density**, for this precisely controlled matched-resource world design.

### Exact MAIN survival in each factorial corner (five descendants × 16 worlds = 80 per cell)

| Walls stressed | Odor stressed | Regrowth stressed | Original selected descendants survived | Per-cell rate |
|---|---|---|---:|---:|
| No | No | No | **73/80** | **91.25%** |
| No | No | Yes | **63/80** | **78.75%** |
| No | Yes | No | **4/80** | **5.00%** |
| No | Yes | Yes | **0/80** | 0% |
| Yes | No | No | **71/80** | 88.75% |
| Yes | No | Yes | **62/80** | 77.50% |
| Yes | Yes | No | **3/80** | 3.75% |
| Yes | Yes | Yes | **2/80** | 2.50% |

**Clear one-factor intuition:** with resources and walls held at their benign level, simply shrinking olfaction **5→2 cells** collapses survival **73/80→4/80**. By contrast, raising walls alone changes 73/80→71/80, and reducing renewal alone 73/80→63/80. The full factorial averages across the other four contexts as required; these simple corners are descriptive and not substituted for the predeclared estimator.

Corresponding mean resources collected/world:
- Original benign scenario: **374.10 foods**, mean **129.69 births**.
- *Only* odor shortened: **117.14 foods**, mean **46.84 births**.
- *Only* food regrowth lowered: **255.89 foods**, mean **88.50 births**.
- All three stressed: **69.79 foods**, mean **29.61 births**.
This collapse in food intake and reproductive opportunities gives a physical mechanism for the survival outcome, but the study cannot establish any unobserved neural mental state.

## 3. Independent held-out VALIDATION confirms odor as the principal effect

Eight completely new world environments, five fixed descendant genotypes = **40 per factorial cell**.

| Manipulated factor | VALIDATION marginal survival impact | 95% descriptive crossed bootstrap | Harm supported? |
|---|---:|---|---|
| **ODOR 5→2** | **−76.25 pp** | **[−91.25,−55.63] pp** | **YES** |
| REGROWTH .55→.40 | −11.25 pp | [−21.25,+0.625] pp | Not definitive in smaller validation sample |
| WALL .12→.20 | −2.50 pp | [−10.00,+4.38] pp | NO |

**All three effect directions are concordant** in sign between MAIN and validation. ODOR remains the uniquely worst stressor and the ranking criterion also passes on validation, though the **preregistered MAIN** confidence intervals remain the formal primary gate.

Validation survival by factor corner, in the same ordering as above: **35/40, 32/40, 3/40, 0/40, 35/40, 27/40, 4/40, 0/40**. Severe joint stress eliminates all selected-descendant transplanted worlds in validation.

### Actual original ancestor neural genotype control

The exact **random original founder** of each successful M5.3B source lineage exhibits the *same qualitative dominant olfactory vulnerability*:

- MAIN ancestor marginal ODOR effect **−74.06 pp**, REGROWTH **−15.94 pp**, WALLS **−1.56 pp**.
- VALIDATION ancestor marginal ODOR effect **−68.75 pp**, REGROWTH **−10 pp**, WALLS **−5 pp**.
- Original ancestor ordinary benign scenario survives **72/80**, versus selected descendant **73/80**; original ancestor all-hard **0/80**, descendant **2/80** (do not infer a statistically established mutation benefit from this tiny corner contrast).

This demonstrates that the short-odor bottleneck is already present in the favorable **founder genotypes**, not something the descendants newly developed.

## 4. Critical viability control: researcher-scripted local greedy+novelty forager survives ALL treatments

Human-scripted M5.1 `greedy_novelty` policy on the same paired original physical costs:

- **MAIN: 16/16 living populations at tick400 on each of all EIGHT factor combinations**, including **all-stressed 16/16**.
- **VALIDATION: 8/8 survivors at tick400 in each of all EIGHT combinations**, including **all-stressed 8/8**.
- **Predeclared hardest-ecology viability gate PASSES**, 8/8 ≥4/8.

Despite shortened odor radius2 and lower resource regrowth .40, it collects **287.75 foods per all-hard MAIN world** and **290.63 per all-hard VALIDATION world**, compared with ~398–395 under benign renewal and odor. The world remains **physically inhabitable** with legal movement, local bounded sensory information, a visit-memory heuristic, and unchanged birth/energy costs.

**Essential separation:** this controller has **programmer-supplied exploration/novelty tie-break and programmed birth thresholds**. It is an ecological **positive reference**, NOT evolved neural intelligence, not comparable parameter-for-parameter to the fixed numerical RNN, and its success must not be sold as emergent intelligence.

## 5. Effects interact; do not extrapolate beyond the controlled estimand

On MAIN descendant survival, preregistered factorial point interactions (descriptive, not separately powered):
- ODOR×REGROWTH: **+8.75 pp**;
- WALL×ODOR: **+2.50 pp**;
- WALL×REGROWTH: **+2.50 pp**;
- Three-way interaction: **+2.50 pp**.

Because survival is near zero with already-short odor, the outcome can exhibit floor effects, making additional stressors appear to have smaller marginal effects. Do **not** interpret the positive interaction sign as evidence that shortened odor helps under lower renewal. The primary marginal factor effect already averages the other axes, and all adverse odor effects remain profound.

The nested-map matching is an intervention on **obstacle density conditional on common food support**, not an unconditional distributional effect over all possible wall/food generators. The fixed selected genotypes have a strong predesigned directional odor/food bias and no recurrent learning in this numerical neural subspace. Thus the result identifies their **short-signal ecological vulnerability**, not a universal law that odor always dominates for other neural architectures or animals.

Descriptive crossed bootstrap uncertainty is computed across source-genome families and new worlds, but **only 5 previously selected successful source genetic histories** exist. Nothing in the study demonstrates a new mutation/selection benefit, a hidden planning strategy, evolved memory, or adaptive improvement when the environment changes.

## 6. Scientific conclusion and next preregistered milestone

**Strong positive causal finding:** shrinking the local food-odor observation horizon from 5 cells to 2 is by far the **principal cause of survival collapse** of the selected fixed numerical neural genotypes, even controlling for wall density and food renewal. The same severe stress is survivable under a separate human-scripted bounded-local search heuristic. The neural phenotype **lacks competent short-odor exploration**.

**M5.4A is NOT evolution.** No genotype, weights or child mutations evolved during these transplant worlds; every descendant was an exact clone, and a scripted benchmark used hand-written motor and birth choices. Improved behavior under this stress still must emerge **de novo** in an independently preregistered *in-world evolutionary* study with random founders, mutation-vs-clone-vs-random child controls and independent post-shift validation. Researchers must not manually supply the M5.1 novelty-memory strategy into original founder weights and then label it evolved.

**Recommended independently frozen follow-up M5.4B:** use the same original neural 10-gene numerical physiology, stage a mid-world drop in odor radius from 5→2 (with walls/regrowth fixed), and compare populations with sparse/dense heritable mutation, exact founder clones and independent random newborn genotypes on BRAND NEW world seeds. A strong conclusion would require true POST-SHIFT survival/reproduction and a source-family/world genotype-mutant superiority over **exact clonal founders**, ideally confirmed by later independent genotype transfer; no M5.4A main/validation worlds may be reused to retune it. Consider adding an externally scripted positive reference but never count its behavior as evolved.

All simulated agents remain bounded numeric code-free entities without file, tool or network access.
