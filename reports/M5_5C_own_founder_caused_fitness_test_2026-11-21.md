# M5.5C — Source-born mutant descendants do NOT outperform their OWN original founders on independent predator worlds

**Candid preregistered null/negative result, 21 November 2026.** The goal was to make a clean causal genotype comparison, **not** to guarantee improvement. No original M5.5B ecology or M5.5A finding has been rewritten.

## Prospective integrity and source artifacts

1. The [M5.5C protocol](../docs/M5_5C_PROTOCOL.md) was committed **`5f8d2b803b3a30143027483c4e071265b5c7851e` BEFORE implementation, source genotype selection or looking at transfer-world outcomes**. It specified exactly eight source families, six interventions, independent transfer world seeds, 1,728 original ecological trials, family-cluster bootstrap and all result gates.
2. [Original source-only candidate freeze](https://github.com/kkongiu/neuroevolution-lab/actions/runs/44951823080) **SUCCESS: 226 tests passed**. Source replays used **only the eight existing M5.5B MAIN SOURCE worlds 94000–94007**, sparse 2/9 in-world genuine mutations under three physical enemies. Each source replay exactly matched the permanent original M5.5B source CSV on geometry SHA, founder SHA, births, predation deaths, alive final, actual mutated births and population AUC. Not one independent transfer map was examined.
3. **BEFORE heldout evaluation**, the eight exact full-precision candidate descendant/own founder pairs were committed separately as [`78b48cdc6a3645126c329f8897f91e8523da0dfc`](https://github.com/kkongiu/neuroevolution-lab/commit/78b48cdc6a3645126c329f8897f91e8523da0dfc) in the [frozen source gene bank](M5_5C_FROZEN_SOURCE_CANDIDATES_2026-11-21.json), SHA256 **`d4cb17b20ec35e7d05474dd04dff464eb0c197106b2a63c669b69c66e7284eab`**. Every selected g8-mutated newborn had actually become a parent in its SOURCE environment (their direct offspring counts were **1,6,3,2,3,5,9,4**). They were source-reproductive-success-selected rather than random population samples; this conditioning is disclosed.
4. [Independent heldout action #44952005010](https://github.com/kkongiu/neuroevolution-lab/actions/runs/44952005010) **SUCCESS**. All source bank SHA checks and 226 historical/novel tests passed before any target world. Run exactly **8 independently selected source families × (24 fresh MAIN 94200–94223 + 12 fresh HARD 94300–94311) × 6 frozen genotype interventions × 400 original-physics ticks = 1,728 physical trials** and **15,552 original checkpoints**.
5. [Every original world-level transplant trial](M5_5C_all_1728_original_causal_transplants_2026-11-21.csv), [all 15,552 original physical checkpoints](M5_5C_all_15552_original_physical_checkpoints_2026-11-21.csv), and [all original 20,000 SOURCE FAMILY resampled statistics](M5_5C_original_preregistered_family_bootstrap_statistics_2026-11-21.json) have been **permanently committed to git** from the successful original job, not reconstructed from a handpicked second run. Full original output also exists in the job's time-limited 30-day artifact.

## What these six transplanted genotypes actually represent

Each NEW independent world starts with 20 native physically embodied agents whose nine numeric inherited genes are **identically initialized to one frozen candidate arm genotype**. Every new native birth is a physical, energy-costed **exact clone** (no further variation in transfer), so genetic differences cause only the replicated changes to the externally supplied same eight original foraging/reproduction traits and ninth local fear gain. Every world has the same three real physically moving enemies, original original-physics native local wall/food/odor observation, energy and reproduction costs. For each source family × transfer world, walls, food schedule, founder positions and enemy initial coordinates are EXACT matched across all six interventions.

The six prospectively fixed arms are: (1) **ACTUAL mutant descendant** born in the 400-tick M5.5B source experiment; (2) **OWN ORIGINAL founder**, not a generic different founder; (3) **all nine genes fully reverted to that original founder**, identical to arm2; (4) mutant with only **fear gene g8 reverted** to its original founder; (5) original founder with only the mutant's **fear gene g8**; (6) a fixed **unselected random nine-gene** source-family control. Different source families are evaluated on shared matched independent transfer worlds, but **eight source families**, NOT 192 per arm, are the independently resampled genetic replicates.

## Independent MAIN: eight source families × 24 fresh worlds = 192 nested world/family trials per arm

These counts are **surviving worlds at tick400 out of 192 FAMILY–WORLD evaluations**, not 192 independent genetic-evolution events.

| Transfer genotype | 400-tick survivor worlds | Mean population at native checkpoints |
|---|---:|---:|
| OWN original uninformed founder | **104/192** (54.2%) | **7.8519** |
| EXACT full nine-gene reversion (mechanics audit) | **104/192** (54.2%) | **7.8519** |
| Actual physical source-born mutant descendant | **95/192** (49.5%) | **7.6892** |
| Mutant with only fear gene returned to founder | **102/192** (53.1%) | **7.8090** |
| Original founder with only mutant g8 | **100/192** (52.1%) | **7.8113** |
| Independently unselected nine-gene random control | **91/192** (47.4%) | **7.2211** |

**Primary actual descendant vs OWN founder:**
- Population survival (per-family 24-world-mean paired difference) **−4.6875 percentage points**, preregistered 20k SOURCE-FAMILY paired bootstrap95% **[−14.0625,+3.125] pp**. Zero is inside.
- Mean native population checkpoints **−0.16262 organisms**, family bootstrap95% **[−0.64236,+0.29745]**. Zero is inside. Only **4/8 families** have positive AUC transfer effect; 3 negative and one exact tie.
- Source-born mutant vs **unselected random nine-gene control** mean AUC **+0.46817**, bootstrap95% **[−0.61632,+1.67650]**. Not a demonstrated win over this essential counterfactual.
- **Fear-gene-specific effect:** source-born mutant vs same genotype with g8 restored to original **−0.11979** population-AUC units [−0.53183,+0.15625], no positive effect. Original founder with only changed mutant g8 vs own original founder **−0.04051** [−0.39931,+0.27257]. Both include zero.
- **Full original genotype reversion identity:** *exactly 0* difference for EVERY source family, EVERY transfer world, every survival and population-AUC comparison. Different labels give identical source geno hash and physical outcomes. The ancestral baseline, environmental matching and intervention isolation passed an unusually strong control.

Interpretation: **The actual selected source-born genotype does not transfer a systematic fitness advantage to independent physical predator ecologies.** A negative group mean does not prove mutations are inherently detrimental; eight source-family replicates and the wide bootstrap intervals leave many small effects unresolved. This is nonetheless a failed preregistered positive evolutionary-adaptation gate, not a success hidden behind a favorable subgroup.

## HARD OOD: 8 genetic families × 12 fresh HARD worlds = 96 nested trials/arm

| Source genotype | Survivor worlds at tick400 | Mean native population checkpoints |
|---|---:|---:|
| OWN original founder | **6/96** | **4.0544** |
| Full original-gene reversion | **6/96** | **4.0544** |
| Actual source-born mutant descendant | **7/96** | **3.8819** |
| Only fear-gene reverted mutant | **9/96** | **3.9826** |
| Only mutant fear gene in founder | **5/96** | **3.9468** |
| Unselected random genotype | **2/96** | **3.6620** |

Mutant-vs-own-founder hard AUC delta **−0.17245**, SOURCE-family bootstrap95% **[−0.62037,+0.25926]**, again fails the positive gate. Both genotype types mostly go extinct. The **seven vs six** surviving family–world combinations do not establish better transferable adaptation under much harder habitat constraints.

## Prospective gates and interpretation

- **Independent source freeze before transfers:** PASS.
- **Exact original physical clonal birth, predator damage, energy conservation, matched world geometry and enemy spawn:** PASS, all original physical ledgers checked every checkpoint.
- **Whole-genome own-founder reversion identity in every matched trial:** PASS, exact.
- **New ninth-gene fear trait's specific adaptive advantage:** **FAIL**, uncertainty intervals include zero.
- **True new mutant descendant vs OWN founder, positive MAIN survival and population AUC AND hard AUC AND independently random control with >=6 positive source families:** **FAIL**. None of the jointly required gains is demonstrated.
- **Spontaneously learned danger awareness/new predator-sensing machinery:** **NOT IMPLEMENTED**. The physical enemy sensor and food/fear/motor instigations are explicitly coded by researchers. The genome numerically tunes them.

**Research verdict:** M5.5B established a genuinely dangerous survival environment and the causal value of researcher-programmed fleeing; M5.5C confirms that descendants actually born from real mutations and selected by source-family reproductive success **do not show a reproducible transferred beneficial genetic advantage** in this preregistered independent transplant study. We cannot claim Darwinian novel genetic adaptation on this evidence. This is a scientifically useful negative result and should remain permanently published.

## Practical next research path (not yet implemented or tested)

Do not retune the completed heldout worlds to recover a positive result. A separate M5.5D should prospectively test whether the **evolutionary mechanism** is blocked by weak phenotype selection, high drift, low number of effective independent generations, or saturation of programmer-fixed instincts. Prefer multi-generation **prospective competition with real lineage frequencies** and repeated genotype reversion, more independent source families, genuinely new untouched environment seeds, at least a predeclared non-extinction balanced predation regime, and explicit controls distinguishing source founder sorting vs de-novo heritable mutants. Genetic susceptibility to actual predation must be separated from neutral source-specific adaptation. No claim of successful emergent learning without an explicit new tested learning mechanism.
