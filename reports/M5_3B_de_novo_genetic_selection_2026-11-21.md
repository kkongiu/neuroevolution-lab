# M5.3B — First within-world heritable survival signal, but mutation advantage and robustness unproven

**Frozen independent campaign: 21 November 2026.** [Preregistered protocol](../docs/M5_3B_PROTOCOL.md) was committed **before implementation and before inspecting test-world scores**, original protocol commit `22970a71a5e7ac25cc5621b3e9e1b2eba6e437ee`.

- [Original successful read-only Actions run #44922843440](https://github.com/kkongiu/neuroevolution-lab/actions/runs/44922843440), **156 regression tests passed**, all 36 worlds × 4 genetic transmission arms ×400 ticks completed.
- [Code](../neuroevolution/m5_3b.py), [tests](../tests/test_m5_3b.py), [development PR #36](https://github.com/kkongiu/neuroevolution-lab/pull/36).
- [Permanent CSV of all 36 paired world survival/birth outcomes](M5_3B_paired_worlds_2026-11-21.csv).
- GitHub Actions 30-day artifact `m5-3b-de-novo-genetic-selection`, ID **14714855150**, includes `summary.json`, 144 rows `world_metrics.csv`, `population_checkpoints.csv`, representative birth/genealogy traces and `population_curves.svg`.

## 1. Meaningful departure from M5.3A

M5.3A demonstrated numerically encoded foraging competence in ordinary worlds (human-engineered 401-weight genotype 24/24 surviving; separately supervised 401-weight 24/24; supervised 51-weight 17/24), but all externally supplied policies failed in shifted worlds. **It did not test evolutionary discovery**, because it supplied skilled weights and cloned them without mutation.

**In M5.3B every founder's ten genes are sampled at random, independently. No external teacher, trained genome, human-designed founder value, genomic champion selection or global fitness score.** All five actions are selected by the original M5.0 numerical RNN actor. Births/offspring/mutations occur only inside real persistent ecological worlds.

**Important architectural qualification:** the ten numerical genes expand into a 401-weight neural network using a researcher's predesigned *shared direction/odor/food/energy wiring*. This is an intentionally strong architecture prior that shrinks the search space, **not evolved sensor discovery or autonomous invention of rules**. A positive result can show genetic selection **within this architecture**, not open-ended evolution of network architecture.

The ten numerical genes, each independently Gaussian N(0,0.65) at founder creation, encode shared food/odor/blocked gain, four motor biases, and three energy/birth coefficients. The map and fixed output weights are frozen, recurrent couplings all zero. At true offspring births, four treatments mutate two genes (SD .16), mutate all ten (SD .16), clone exactly, or redraw all ten independently (SD .65), with perfectly matched founder genes and food candidate schedule across treatments. Parents transfer energy to genuine offspring under **unchanged M5.0 physics**; original newborn birth/energy/legal action handler remains the source of physical events.

## 2. Preregistered MAIN survival after 400 ticks — 24 fresh worlds

Seeds **82000–82023**, all independent of earlier M5 campaigns.

| In-world genetic mechanism | Surviving /24 worlds | Mean living at end | Births mean/world | Food eaten mean/world | Mean max generation |
|---|---:|---:|---:|---:|---:|
| **Sparse mutation: two genes at birth** | **5** | 2.54 | 38.54 | 137.46 | 4.88 |
| Dense mutation: all ten genes | **5** | 2.25 | 37.50 | 129.58 | 4.13 |
| Exact inheritance with no new mutation | **3** | 1.46 | 32.42 | 119.54 | 4.25 |
| Each child independently redrawn random | **0** | 0.00 | 16.42 | 85.21 | 2.17 |

Specific successful worlds for sparse mutation: **82006, 82015, 82018, 82019, 82023**; for dense: **82012, 82015, 82018, 82021, 82023**; for clones: **82018, 82019, 82023**. These are **observed** frozen scores, not world seeds to reuse for the next primary evaluation.

The sparse treatment achieved generation ≥3 in **12/24** main worlds, satisfying that secondary generational threshold. But it sustained an actual population in **only 5/24**, below the explicitly preregistered primary requirement of ≥12/24. No claim of robust ecological viability.

## 3. Preregistered paired survival contrasts, whole world as unit

| Main-world contrast | Absolute survival difference | Descriptive paired bootstrap 95% | Positive / negative / tie worlds | Scientific interpretation |
|---|---:|---|---|---|
| Sparse two-gene mutation − random newborn genotype | **+20.83 percentage points** | **[+4.17,+37.50] pp** | 5 / 0 / 19 | **Paired inheritance comparison SUPPORTED** |
| Dense mutation − random newborn genotype | **+20.83 pp** | **[+4.17,+37.50] pp** | 5 / 0 / 19 | Secondary comparison supported |
| Sparse mutation − exact clones | **+8.33 pp** | **[0,+20.83] pp** | 2 / 0 / 22 | **NOT supported**; lower limit includes zero |
| Sparse mutation − dense mutation | **0 pp** | **[−16.67,+16.67] pp** | 2 / 2 / 20 | No confirmed difference |

The central outcome is **a genuine in-world survival advantage for hereditary numeric genotypes over fresh random child genotypes**, within this architecture and these 24 independent worlds. This is not yet evidence that **novel mutations** produced a new foraging competence: unmutated **clonal inheritance already survives 3/24 worlds**, consistent with **selection among favorable random founder variants**. Sparse beats clones in only two worlds with a bootstrap lower bound exactly zero.

**All preregistered success gates considered together:**
- Neural sparse ecological survival ≥12/24: **FAIL** (5).
- Max generation≥3 in ≥12/24: **PASS** (12).
- Positive paired survival versus random offspring with descriptive bootstrap lower >0 and favorable signs: **PASS**.
- **Combined primary gate: FAIL**, because sustained ecological persistence is too rare.
- Mutation-versus-clone novel-adaptation gate: **FAIL**, cannot support invention of competence by mutation.
- Robust shifted-world transfer: **FAIL**.

The descriptive bootstrap does **not** constitute a guarantee of high-power inference for a novel evolutionary process. There are only five positive versus random worlds; all world-arm outcomes and nulls are preserved rather than selectively reporting favorable lineages.

## 4. SHIFTED worlds — all extinct

Twelve new shifted-world seeds **83000–83011** with more obstacles, shorter odor radius and weaker food regeneration, no overlap with previous shifted seeds.

| Treatment | Surviving /12 | Mean births | Food collected |
|---|---:|---:|---:|
| Sparse | **0** | 13.50 | 44.33 |
| Dense | **0** | 12.00 | 44.08 |
| Clone | **0** | 13.58 | 46.17 |
| Independent random child | **0** | 14.17 | 44.42 |

All OOD survival comparisons are exactly tied at 0, descriptive paired intervals [0,0]. Thus **no ecological adaptability under environmental change was established**. Each extirpated world retained mean 32/32 food stock; local successful foraging is absent in the more hostile geography.

## 5. Engineering integrity and limitations

All 156 tests and complete frozen campaign passed. Genes influence neural logits through fixed numerical mapping; neural action output can reverse by changing numeric gene signs, with no scripted motor. Ten-gene mutation changes occur only on genuine births, with exact lineage/parental cost, genotype digests and variable gene distances recorded. Energy and physical collisions remain governed by the unchanged original ecology. No historical protocol, test seeds or conclusions were modified.

**Novelty distinction:** this study genuinely allows birth/death selection to operate on varied numeric genotypes in a persistent world; the architecture, sensory features, output wiring and reproduction laws are *researcher designed*. Ecological persistence in 5 worlds is real simulated survival, but it is sparse and may depend on favorable founder variation rather than acquired mutations. Nothing demonstrates consciousness, open-ended adaptation, self-chosen goals, self-modifying software or executable replication.

## 6. Next research decision: controlled genotype transplant and standing-variation audit

Do NOT simply retune sparse mutation SD after looking at the final seeds. Instead, a separately preregistered **M5.3C** could:
1. deterministically recover complete ten-gene founder and descendant vectors from the **already published successful-world runs** (replay using their frozen seed and original M5.3B code, no changes to scientific primary outcomes);
2. prospectively select a descendant per world via a **fixed non-test-based rule** (e.g. highest-generation living agent at tick 400, then lowest ID) and compare it against the matched originating **founder** and a same-world **clonal-only** genotype in independent unseen worlds;
3. test genotype transplant without mutation as a causal phenotype assay, plus spatially shifted and randomized/scrambled gene controls, whole-world statistics;
4. distinguish *new mutation benefit* from mere founder luck, while retaining the strong architectural bias as an explicit limitation.

Such a test is useful only as a new separate study with fresh independent evaluation seeds; it must never reclassify M5.3B's preregistered failed primary gate as a success. If descendant genotypes do not generalize better than founders, the apparent inheritance advantage may reflect simple retention of already capable founder variants rather than cumulative adaptation.
