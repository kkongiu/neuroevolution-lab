# M5.30 — Completed 22 November 2026: support-selected human-supplied operators

**Run:** [GitHub Actions #45068913156](https://github.com/kkongiu/neuroevolution-lab/actions/runs/45068913156) — **SUCCESS**. Five M5.30 scientific unit tests and five original M5.6 online-plasticity regression tests passed, followed by the completed development simulation. The seven previously published narrow screening checks all pass. This is **NOT** autonomous operator invention, a learned reusable neural executive, independent new-family replication or ecological evolution.

The protocol, criteria and cohort seed lists were recorded in
[the preregistered M5.30 design](M5_30_OPERATOR_SELECTION_PROTOCOL_2026-11-22.md)
before the completed scientific Actions run. The CI workflow initially had
a YAML quoting defect (a colon in a step label); it was fixed without
changing the scientific simulation or thresholds. Its initially invalid
workflow runs are **not** experiment results.

## Actual development cohort experiment

- Three optimizer initialization seeds (817, 827, 829), 16 parity seeds,
  16 parity-plus-pairwise-AND seeds and 8 out-of-menu triple-AND seeds:
  **120 evaluated trial rows** (3 × [16 + 16 + 8]).
- Each row scores three externally visible A/B/C contexts: **360 actual
  task contexts**. Each task has eight own chosen-action/reward experiences,
  and eight disjoint unseen-cue queries. The three rule codes are separate.
- For each task, seven **researcher-written** operator hypotheses are
  separately fitted from the same eight support experiences by genuine
  task-specific M5.29 neural SGD, and chosen by support-only MSE plus
  frozen pairwise complexity penalty 0.025. The identical menu is also
  fitted on corrupted support feedback for the negative control.
- Candidate can see neither the true operator, family label, generator
  coefficients, unchosen-action outcome nor the held-out exam reward.
  Explicitly label the human-privileged reference as an oracle.
- Candidate final task storage is **3 × (5 float32 × 32 bits +
  3-bit menu index) = 489 bits**. This is NOT total organism memory:
  the 14 temporary candidate model fits, eight parallel optimizer
  restarts per candidate, support observations, structural interpreter
  and all compute are excluded. Models with different inference compute
  are not equal-budget or speed-matched.

### Frozen accuracy on NEW cues (eight never experienced per context)

| Arm | Plain parity | Parity + pair AND | Out-of-menu triple AND |
|---|---:|---:|---:|
| **Support-selected operator, real feedback** | **98.09%** | **78.56%** | **73.26%** |
| Always plain product, real feedback | 97.40% | 54.17% | **77.43%** |
| Random menu operator, real-feedback fit | 61.98% | 59.38% | 61.81% |
| Selected operator, scrambled support feedback | 48.52% | 50.95% | 49.65% |
| **Human-privileged true-operator reference** | 97.40% | **98.44%** | 77.43%* |
| Plain product, no gradient adaptation | 52.60% | 50.52% | 46.35% |

*Out-of-menu has no true matching operator. In that column, the
'privileged reference' uses the plain product as a declared fallback.
It is not an actual oracle for three-way-AND tasks.

Within the specifically supplied pairwise-composition family, real
support-selection beats fixed product by **+24.39 percentage points**,
exploratory two-level paired bootstrap 95% **[+18.75, +29.86] pp**.
It beats corrupted-feedback menu selection by **+27.60 pp**
[+21.35, +33.07] pp, but **loses 19.88 pp** against a
privileged correct-menu-operator reference [-24.91, -15.28] pp.
These confidence intervals are **development/exploratory**, derived
from only three optimizer seeds and correlated repeated task contexts;
they do not represent independent, broad new-family replication.

**Correct structural choice frequency:** 56.94% of the 144 pairwise
task contexts. On plain parity the correct plain-product selection
rate is 96.53% (139 of 144 contexts). In the pairwise family the
selected plain-product count is 19/144, and other picks are distributed
over the six supplied pair operators. Eight support experiences often
cannot identify the true pair among competing candidates, even when
a privileged choice of the correct operator scores 98.44%.

**Critical negative transfer:** for three-way AND outside the
hypothesis menu, selection scores **73.26%**, worse than fixed product
**77.43%**, difference **−4.17 pp**, exploratory bootstrap interval
[−7.12, −1.39] pp. This is direct evidence that the menu still
constrains generalization; it is not proof all neural architectures
would fail. The exact same candidate is used without tailoring to this
cohort and the decline is preserved as a negative result.

### Prespecified checks

All **7/7** declared narrow checks are TRUE:
nonaffine-pair unseen accuracy ≥70% (actual 78.56%);
gain versus fixed product ≥8pp (actual +24.39pp);
gain versus corrupted-feedback selection ≥15pp (actual +27.60pp);
support/exam separation;
no access to the true operator in the candidate;
489-bit final task-specific codes;
out-of-menu stress explicitly present.

These are **not** seven claims of independent scientific breakthroughs.
Four are implementation, accounting, leakage or completeness checks.

### Data integrity, permanent result and limits

- The GitHub Actions artifact
  \`m5-30-operator-selection-development\` contains all 120 raw rows,
  actual code values, per-context selected and offline true operator IDs,
  support selection margins and all six-arm outcomes. GitHub Action
  artifact retention is 30 days. The simulation source + deterministic
  seeds are retained in this branch for rerunning after expiration.
- **Canonical raw-rows SHA-256:**
  \`17444596a513b3e3764f1da5b023c12b39bc003a81b2fc354793b3c118dad8f6\`.
  Independently recomputed from downloaded artifact:
  \`sha256(json.dumps(raw_rows, sort_keys=True, separators=(",", ":")).encode())\`
  exactly matches the reported digest.
- All four familiar rule masks (5,9,11,14) were already visible in the
  previous M5.29 development sequence. This is **fresh synthetic seed
  replication**, not a wholly unforeseen task-family test. Menus contain
  handcoded AND terms and the eight binary rewarded actions imply
  complementary deterministic labels. No organism discovers AND,
  invents a composition operation, or meta-learns a controller over
  multiple lives.
- No original M5.6 sealed worlds were accessed; original ecology,
  evolution and main branch remain untouched.

**Scientific stop/go flags:**
\`limited_support_operator_selection_screen_pass=true\`;
\`autonomously_discovered_composition_operator=false\`;
\`learned_shared_neural_executive=false\`;
\`independent_task_family_replication_complete=false\`;
\`evolution_authorized=false\`;
\`advance_to_evolution=false\`.

**Next hypothesis, NOT yet validated:** Allow the neural controller to
*construct/parameterize an interaction* from raw experience, rather than
select from six researcher-supplied pair indices; train its shared
controller on a family of tasks and test unseen operator classes with
strictly heldout independent task families and resource-matched controls.
Do not automatically proceed to evolutionary survival from this gate.
