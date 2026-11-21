# M5.4C — Foraging-movement control, not reproductive timing, causes extinction of the source-selected 18-gene neural organism

**Preregistered causal 2×2 ecological diagnostic COMPLETED — 21 November 2026.** Strong fixed-genotype finding: substituting HUMAN-PROGRAMMED local exploration while retaining ORIGINAL neural reproduction rescues **23/24** normal and **12/12** hard 400-tick worlds, whereas substituting HUMAN-PROGRAMMED reproduction while keeping ORIGINAL neural navigation rescues only **5/24** normal and **0/12** hard worlds. **This is NOT demonstration that the neural organism evolved the scripted policy.**

## 0. Source integrity, prospective protocol and original auditable evidence

- [Prospectively FROZEN full scientific protocol](../docs/M5_4C_PROTOCOL.md) in **`c351085a15ebb9a56af5f3d198c2f3939905ab80` BEFORE M5.4C code or any of its 36 new world outcomes**. Four intervention arms, original selected numeric genome, birth threshold46, seeds, physical 400 ticks, detailed behavior metrics, all marginal/interaction contrasts, 20k world-paired bootstrap confidence intervals and bottleneck decision criteria all frozen.
- [Original one-time read-only GitHub Actions scientific run #44938198849](https://github.com/kkongiu/neuroevolution-lab/actions/runs/44938198849) **SUCCESS**: **203 historical and new tests PASS**, **36 NEW independent worlds × 4 arms = 144 complete original-physics 400-tick trials**, original 1,296 checkpoints, all 20k per-world paired-bootstrap calculations completed.
- [ALL original 144 world-policy numerical outcomes and all per-lifetime diagnostic measures](M5_4C_all_144_paired_world_full_diagnostics_2026-11-21.csv), permanently stored in repo; [ALL 1,296 native 50-tick population, birth, food, energy, traversal, odor and lineage checkpoints](M5_4C_all_1296_population_checkpoints_2026-11-21.csv), permanently stored in repo.
- [Exact fixed original externally selected 18 numeric genes, human policy provenance, complete preregistered seven survival contrasts, 4 auxiliary food/birth contrasts, all per-world deltas, full 20k paired bootstrap distributions' 95% bounds, gates, uncertainty limits](M5_4C_exact_source_genes_and_all_preregistered_effects_2026-11-21.json).
- [All 36 matched physical wall+food+founder+complete 400-tick food-RNG schedule SHA256s, identical across the four policies](M5_4C_36_exact_paired_world_genome_and_food_schedule_sha256_2026-11-21.json).
- Original full archived `m5-4c-neural-vs-programmed-motor-birth-factorial` Actions artifact ID **14720288367** (30-day retention), containing original `summary.json`, individual full event genealogy ledgers for one ordinary+one shifted world per policy, source frozen numeric vector, full 144 original per-world metric rows, original 1,296 physical snapshots and matched geometry digests. Crucial original 144 world and 1,296 checkpoint rows are permanently retained in repository, independently verified against source-run JSON log rows.
- Historical unmodified source M5.4B0-R original main SHA **`387b7abe0bbe97526cfcb82a7d8050ed2c16f52f`**, externally optimized 18-gene fixed candidate **#83**, original expanded 401-weight neural genotype digest **`c5e5795332725de4`** and original 18 signed numeric gene coordinates pinned. No new genotype optimization performed.

## 1. What was manipulated — NOT learned or naturally evolved

The original numeric 18-gene recurrent RNN was previously selected by a HUMAN-CONTROLLED EXTERNAL optimizer in M5.4B0-R. It supplies the SAME original 401 weights to every one of the 20 founders and all descendants across all arms and worlds. These numeric genomes **never mutate inside the simulated ecology**; all real physically energy-costed births produce exact clones. This study does not test or demonstrate natural evolution, true genetic adaptation, plasticity or emergence of a human-coded search policy.

A real-world-policy **2×2 switch** separates actions into the choice of four cardinal legal movement steps vs the choice of physically eligible birth:

| Registered arm | Movement decision policy | Physically eligible reproduction decision |
|---|---|---|
| **NN** `neural_move_neural_birth` | Original unmodified native M5.0 RNN's direction-logit maximum | Original unmodified native M5.0 RNN's birth logit competes with four legal movements |
| **NS** `neural_move_scripted_birth` | Original recurrent RNN's highest-legal movement logit | **HUMAN-PROGRAMMED** energy≥46 threshold and original M5.0 eligibility |
| **SN** `scripted_move_neural_birth` | **HUMAN-PROGRAMMED** M5.1 original local greedy food+odor+personal visit-count novelty | Original native recurrent RNN's birth preference vs legal movements and original M5.0 eligibility |
| **SS** `scripted_move_scripted_birth` | **HUMAN-PROGRAMMED** original M5.1 local greedy food+odor+personal visits | **HUMAN-PROGRAMMED** original threshold46 birth when physically eligible |

The M5.1 scripted movement uses current local original visible adjacent food (4×), local directional food smell, and a researcher-provided `0.35/(1+personal visits to target neighbor)` bonus; tie-breaking uses the old deterministic seed/tick/agent-ID substream. This personal visit dictionary is ***not inherited by the original neural genome and not an evolved neural internal memory***. Original finite 15 local sensory channels are unchanged. The HUMAN-PROGRAMMED birth alternative changes only the preference to choose an already physically allowed birth at energy≥46; it cannot bypass original M5.0 energy, max age, basal action cost, newborn cost, neighbor availability or population capacity.

**Physical reference audits:** dedicated tests verify NN action/food/birth event ledger is exactly identical to previous original `m5_4b0r.simulate_numeric` for the same seeds/genome; SS has exactly identical physical food/birth event ledger to original M5.1 human-coded local greedy+novelty reference (only M5.1's *inert mutable bookkeeping-genome hashes* differ, intentionally excluded from physical comparison). Both hybrid cases use original native M5.0 physical action law through tested script adapter and all newborns clone the original pinned numeric RNN. Personal movement-visit counters are diagnostic for NN movements but may guide ONLY human-scripted movements. Even an organism that starves on its final movement has that physical final cell counted in diagnostic traversal.

The 36 fresh ORIGINAL M5.0 worlds never appeared in any previous source optimization, model selection or test world: MAIN **91000–91023** (walls density.12, regrowth probability.55, odor radius2 from tick0), SHIFTED **91100–91111** (density.20, regrowth.40, odor radius2). Every world has original 15×15 grid, 20 initial food and 20 matched founder positions, 400 ticks, initial energy36, food18, basal movement cost1, original age/birth law, max64 population. Paired wall/initial-food/positions and all random future food candidate draws were verified byte-identical between all four policies before the first actor moved.

## 2. MAIN — exceptionally clear 2×2 survival rescue

**24 NEW independent normal short-odor radius2 worlds. Original population survival to tick400.**

| Policy: movement × reproduction | Worlds surviving tick400 | Survival | Mean native physically real births/world | Mean food eaten/world |
|---|---:|---:|---:|---:|
| **NN** neural × neural | **4/24** | 16.7% | 28.50 | 146.67 |
| **NS** neural × HUMAN-scripted birth | **5/24** | 20.8% | 61.71 | 186.46 |
| **SN** HUMAN-scripted movement × neural birth | **23/24** | 95.8% | 83.00 | 387.50 |
| **SS** HUMAN-scripted movement × HUMAN-scripted birth | **24/24** | 100% | 115.25 | 398.75 |

This is a much stronger mechanistic result than the earlier B0-R original comparison because we changed **ONE major decision axis at a time**, keeping the original same numeric recurrent neural reproduction with scripted movement (SN).

### Registered main factorial effects on survival

Let each row be one independent world seed with four binary endpoints NN,NS,SN,SS.

- Marginal **movement-script rescue `E_M=½[(SN−NN)+(SS−NS)]`**: **+79.17 percentage points**, preregistered 20,000-replicate WORLD-PAIRED 95% bootstrap **[+66.67,+89.58] pp**; exceeds registered ≥25pp criterion with strictly positive lower95%.
- Marginal **reproduction-script rescue `E_R=½[(NS−NN)+(SS−SN)]`**: **+4.17 pp**, 95% **[−8.33,+14.58] pp**; confidence interval includes ZERO. No robust survival evidence for reproduction-only rescue.
- Strong direct **movement dominance `E_M−E_R=SN−NS`**: **+75.00 pp**, 95% **[+58.33,+91.67] pp**; lower bound decisively positive.
- **Movement-only simple rescue `SN−NN`**: +79.17 pp, 95% **[+62.50,+95.83] pp**.
- **Birth-only simple rescue `NS−NN`**: +4.17 pp, 95% **[−16.67,+25.00] pp**.
- **Full human-coded control vs original NN `SS−NN`**: +83.33 pp, 95% **[+66.67,+95.83] pp**.
- Binary survival-scale **interaction `I=SS−SN−NS+NN`**: 0 pp, 95% **[−25.00,+20.83] pp**; no preregistered evidence interaction differs from zero. This does **not** imply zero interaction in all neural genotypes or hard worlds.

**All three preregistered conditions for MOVEMENT-DOMINANT programmed rescue PASS; reproduction-dominance FAILS, double-script-only requirement FAILS, and full script positive-control PASS**.

These are conditional comparisons for **one old source-selected recurrent genotype** on 24 new worlds; the world-bootstrap cannot quantify variation across other genotypes/training methods.

## 3. SHIFTED — independent confirmation under harder ecology

**12 OTHER entirely unseen hard short-odor radius2 worlds**; no reoptimization or modification after main-world results.

| Policy | Survived 400 ticks | Mean real births | Mean food eaten |
|---|---:|---:|---:|
| **NN** neural movement × neural birth | **0/12** | 12.92 | 64.92 |
| **NS** neural movement × programmed birth | **0/12** | 23.92 | 64.50 |
| **SN** HUMAN-programmed movement × neural birth | **12/12** | 64.33 | 296.58 |
| **SS** HUMAN-programmed movement × scripted birth | **12/12** | 88.67 | 297.08 |

Independent hard-world marginal programmed-movement rescue **+100 percentage points**, descriptive WORLD-PAIRED bootstrap 95% **[+100,+100]** in THESE twelve paired test-worlds; reproduction-only marginal effect **0 pp [0,0]**. This is an observed complete binary separation on these world seeds, not a universal theorem for all possible environments.

NN physical numeric recurrent reproduction *works sufficiently to propagate generations* when movement search is externally scripted; it cannot support the population in ANY hard new world with the old native neural movement even when reproduction is scripted.

## 4. Behavior explains the rescue: repeated moves, real food and real births

The source-selected neural movement repeatedly visits the same cells rather than covering the habitat efficiently. These preregistered per-lifetime diagnostics are observational mechanisms, not a perfect causal isolation of the ***visit-count memory term*** itself:

| MAIN average (per world) | NN original both-neural | NS neural movement / scripted births | SN programmed movement / neural births | SS both scripted |
|---|---:|---:|---:|---:|
| Fraction of all actual movement steps entering a cell **already visited by the same individual** | **81.9%** | 80.1% | **20.9%** | 20.2% |
| Mean distinct cells ever visited **per organism lifetime** (including deceased organisms) | **9.56** | 8.59 | **40.18** | 35.25 |
| Mean original physical food units eaten per world | 146.67 | 186.46 | **387.50** | 398.75 |
| Mean actual costed births per world | 28.50 | 61.71 | **83.00** | 115.25 |
| Average fraction of actions with no local food odor | 78.7% | 80.0% | 84.7% | 85.6% |

The scripted movement policies visit substantially more distinct cells and avoid repeated positions, even though they often experience **more odorless sensor observations**. The problem is **not simply absence of local food odor**; it is how to choose a productive legal step when scent is unavailable. The old native neural recurrent controller's sparse directional inputs and feedback may yield near-deterministic repetitive trajectories, despite genuine recurrent state, whereas the HUMAN-coded M5.1 local scoring explicitly biases away from personal revisit loops.

Changing only birth policy substantially increases real births (NN 28.5 → NS 61.71) **without** reliably saving the population (4/24 →5/24). When movement search is scripted, original native neural birth timing suffices for 23/24 normal and 12/12 hard world survival despite fewer births than human-coded SS. Thus the narrow diagnosis that **birth timing is the main reason for failure** is falsified for this particular original source-selected 18-gene controller in these original new worlds.

Factual limitation: the human-coded movement intervention changes multiple properties at once — explicit adjacent-food prioritization, directional odor, personal visit-memory bonus and stochastic tie-breaking. Therefore M5.4C establishes a **dominant causal movement-policy bottleneck**, but **does NOT yet prove that specifically the missing visited-cell memory term alone is the sole mechanism**, as opposed to the relative weighting of local food, an exploration policy or tie randomness. Further proof requires separate programmatic movement-factor ablations or newly expressive numerical neural controllers with new independent blinded worlds.

## 5. Immediate scientific decision, not an evolution claim

### What M5.4C DOES establish
1. A frozen previously researcher-selected recurrent numeric 18-gene organism **can physically reproduce sufficiently to sustain populations** even in original difficult radius2 worlds, **provided** the movement direction is chosen by a separate human-coded local exploration policy.
2. Causal intervention of HUMAN-coded movement choice rather than its native RNN movement choice produces a **large, independently reproduced, preregistered survival advantage**, and significantly outperforms switching only birth choice.
3. The difference is strongly consistent with deficient **spatial coverage and repeated loops**, as measured by independent action diagnostics and food consumption. The physical habitat and birth machinery are not intrinsically fatal.

### What M5.4C does NOT establish
1. NO numerical neural network evolved or learned the HUMAN-coded movement strategy. Every original within-world physical birth clones the same **externally researcher-selected** 18-gene numeric vector. ZERO true genetic mutations, no natural-selection adaptation.
2. It does NOT prove that personal visit memory, specifically, is the only missing neural information. Programmed movement changes local food reward weighting, deterministic preference structure and tie-breaking as well as visit novelty.
3. It does NOT establish that an arbitrary random uninformed founder or alternative 18-gene numerical RNN could achieve the same survival without the human-coded motor intervention.
4. The world-bootstrap confidence interval only describes the 36 new original physical test worlds of this particular fixed selected genotype, not 36 independent natural evolutionary origins; no `p`-value generalization over genetic architectures is warranted.

### Precommitted next-science recommendation

**M5.4D (NEW PROSPECTIVELY FROZEN protocol)**: in the now proven movement axis, isolate **visit-novelty vs local food/odor weighting vs randomized exploration/tie-breaking** with all native neural birth decisions unchanged, strictly new disjoint world seeds and matched physical map+food RNG. Controls should include human-coded original M5.1 greedy+novelty, same greedy score with NO personal visit novelty bonus, odor/adjacent food greedy, and randomized legal exploration only when no food scent is available, plus original native full recurrent NN movement. This would identify which **part of the programmed movement rescue** mattered most without confusing it with original neural ability. Then test whether numerical neural recurrence or a documented explicitly programmer-defined local novelty sensor can *represent* the effective movement subpolicy, using fresh unused development and truly heldout worlds, consistent numerical action/birth physics and external-optimum-vs-genuine-evolution separation.

**Actual de novo neuroevolution under mid-world odor5→2 stress remains M5.4B1 and MUST NOT be declared achieved by a human-coded movement intervention or by seeding these externally selected old source genes.** Only after original numerical intrinsic short-odor competence is independently demonstrated should truly new random founder populations undergo mutation-vs-clone-vs-random-child and strict original-founder-vs-descendant transfer, all on disjoint new worlds.

The synthetic organisms remain fully numerical simulated states with no external tool, file, network or arbitrary executable code access.
