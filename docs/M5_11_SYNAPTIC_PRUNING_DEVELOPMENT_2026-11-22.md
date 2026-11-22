# M5.11 — Structural synaptic pruning, birth and consolidation: development results

**2026-11-22. Status:** fully implemented and executed **development-only** experiment. Some positive relative evidence against **matched random turnover**, but no evidence yet that synaptic turnover outperforms a stable sparse brain at competence retention. No reproduction, no evolution, no physical ecology integration.

## Research question

Does remodeling synaptic connectivity (creating, consolidating, pruning) with fixed neuron count improve retention after intervening problems, while maintaining fewer synapses than the full dense controller?

The same native M5.6 controller has **8 recurrent hidden neurons** and exactly **260 possible weighted synapses**. Synapse birth/death is *effective edge activation/removal in these 260 slots*; there are no new neurons, no unbounded graph/neuronal growth, and no realistic synapse metabolism. All non-dense arms start with the **same 156 active slots** and initial synapse weights on a matched seed. The full-density capacity control has all 260 active from birth.

The environment presents cue/context A → B → A → C → B, **160 action-feedback episodes per context**, 800 in total. Binary cue and context are observable; hidden cue XOR rule bits are not. All variants use the same 260-value initialized genotype, visible-context encoding, action RNG tickets, learning rate .04 and reward structure. There is no engineered context-to-module routing. Recurrent state and synapse weights persist across phases. The environment alone determines success; no correct-action label is fed to the policy.

All structural updates occur on a fixed 40-action interval **without providing the phase transition time**. The `grow_prune` variant removes synapses with lowest historical activity/outcome credit (plus a small weight-magnitude term), while forming new dormant edges when a running EMA of observed errors exceeds a fixed threshold. Pruned weights are set to zero; dormant weights remain exactly zero through policy-gradient updates until new edge activation. The utility proxy is heuristic and could protect task-specific associations imperfectly, especially over long periods of inactivity.

## Seven arms

- `fixed_sparse`: 156 active synapses, immutable topology; baseline for benefits of rewiring versus pure weight learning.
- `prune_only`: activity/credit-based pruning, down to at least 110, no birth.
- `grow_only`: open dormant edges upon error pressure, no pruning; cap at 260.
- `grow_prune`: credit-protected pruning and error-triggered edge birth; both active.
- `turnover_random`: fixed 12 randomly removed/replaced per interval, **not turnover-dose matched** to `grow_prune`, so its deficits cannot alone prove beneficial intelligent synapse selection.
- `yoked_random_turnover`: for each matched seed, receives the same exact series of synapse addition/pruning counts as its `grow_prune` counterpart, but randomly chooses *which* synapses to turn over. Audited **exact equality of total birth/death counts AND temporal allocated-synapse trajectory** for each seed. Its intervention schedule is sourced from the paired adaptive run, not independently decided by the placebo; it receives no correct-action labels or privileged future state for decisions.
- `full_dense`: fixed 260-synapse network, necessary capacity counterfactual.

All arms include both action choices and the same possible synaptic slots, with identical starting 156 active slots except dense.

## Real executed result

[Successful GitHub Actions run #45030604839](https://github.com/kkongiu/neuroevolution-lab/actions/runs/45030604839) passed four new structural unit test cases plus five original native-controller regressions; **24 seeds × 7 arms = 168 full organism trajectories** evaluated. Full JSON artifact archived as `m5-11-structural-synapses-and-pruning-development` (30-day retention). Canonical JSON row digest `9d219d18f5218fead20e494dccd0063dc957833899bfc1e82d9a5277f6531ccd`.

| Arm | Full-trial accuracy | First-window return A | First-window return B | Mean active synapses | Mean births | Mean prunes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed sparse | 68.57% | 76.77% | 70.00% | 156.00 | 0 | 0 |
| Prune only | 68.56% | 76.67% | 70.73% | **120.00** | 0 | 46 |
| Grow only | 70.58% | 75.10% | 73.54% | 213.97 | 79.5 | 0 |
| Grow + prune | 69.36% | 76.25% | **71.04%** | 168.16 | 119.08 | 107.92 |
| Random turnover (unmatched) | 62.51% | 67.19% | 62.19% | 156.00 | 240 | 240 |
| **Yoked random turnover (matched)** | **66.79%** | **70.21%** | **65.21%** | **168.16** | **119.08** | **107.92** |
| Full dense | **73.95%** | **80.31%** | **75.83%** | 260.00 | 0 | 0 |

### Primary comparisons

- **Prune-only vs fixed-sparse:** almost indistinguishable global accuracy (**68.56% vs 68.57%**), while mean allocated active edges decrease **156 → 120 (~23.1%)**. This is an allocation proxy, not direct computational joules; the implementation still computes fixed-size dense policy operations, so no demonstrated wall-clock or FLOP savings.
- **Grow+prune vs yoked event-dose matched random topology:** return B **71.04% vs 65.21% = +5.83 percentage points**, with identical per-seed and per-interval number of rewired edges. This exploratory comparison suggests the *identity of synapses retained/pruned* matters, not just turnover frequency or allocated capacity. Not an independent confirmatory generalization.
- **Grow+prune vs fixed sparse:** return B **71.04% vs 70.00% = +1.04 pp**, failing the preregistered +10 pp competence gain threshold.
- **Grow+prune vs full dense:** still lower mean accuracy **69.36% vs 73.95%**. Extra capacity remains a substantial explanation.
- **Retention after inactivity** isn't universally better: return B minus first-visit late B window is negative in all arms (grow+prune **−5.10 pp**, sparse **−3.85 pp**, full **−3.02 pp**), despite respectable absolute recall. Do not claim solved catastrophic forgetting.

### Screening gates

1. Return B grow+prune >=65%: **PASS** (71.04%)
2. Synapse-ticks less than full dense: **PASS** (168.16 <260 active edges on average)
3. Grow+prune return B exceeds fixed sparse by >=10 pp: **FAIL** (+1.04 pp)
4. Grow+prune exceeds matched random turnover return B by >=5 pp: **PASS** (+5.83 pp)

**3/4 pass. `advance_to_evolution=false`.** The exploratory outcomes must not be retroactively selected as an independent final validation.

## Limitations / next protocol

1. This is a **fixed 260-edge topology mask**, not an unbounded structural brain. The stated task was *synaptic pruning*, not a biologically literal pruning model.
2. Critically, **pruning-only has similar performance to fixed sparse**, and the benefit is smaller edge allocation rather than better flexibility.
3. The heuristic credit preserves and prunes synapses based on aggregated action-feedback and activity. It does NOT causally establish each synapse's unique contribution; require counterfactual edge-drop probes during **development only**, never providing oracle true target labels, and measure old-context retention.
4. Task contexts are explicitly sensory-visible; no open-ended unsignaled rule discovery. Novel problem-family validation is still needed.
5. Edge count `synapse-ticks` is not currently a measured compute-energy saving because the implementation still loops over all 260 weight slots for actions/gradients. Optimize sparse kernels only after scientific benefit is confirmed.
6. A true matched synaptic pruning study should vary retention intervals and turnover pressure, and compare against equal-density static masks (not solely full density).
7. No reproduction, growth-rule heredity or ecological energy ledger. Maintain M5.6 SOURCE/HOLDOUT/REPLICATION cohorts untouched; `main` unchanged.

**Recommended next step:** investigate why credit-based consolidation is better than random turnover but still not superior to fixed sparse; test controlled connection-ablation importance, longer A/B inactivity, and preserve old circuit competence before introducing reproduction.
