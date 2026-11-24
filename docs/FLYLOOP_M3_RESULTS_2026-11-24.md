# FlyLoop M3 — Audited result: movement-history helps modestly, outcome memory does NOT causally establish blind foraging
**24 November 2026. Prospective DEV science screen = FAIL (11/18 gates). All software tests + deterministic replay PASS.**

- Official completed GitHub Actions [run #45355392710](https://github.com/kkongiu/neuroevolution-lab/actions/runs/45355392710) succeeded. M0/M1/M2 regression tests and new M3 tests passed.
- Preregistered protocol BEFORE full run: [FLYLOOP_M3_PROTOCOL_2026-11-24.md](FLYLOOP_M3_PROTOCOL_2026-11-24.md).
- No criteria, seed sets, actor/critic learning rate or neural inputs changed on the basis of M3 outcomes.
- GitHub Actions artifact: **flyloop-m3-causal-temporal-memory**, ID **14902471288**, archived to **24 December 2026**. It contains **two complete 223,415,999-byte JSON results, BYTE IDENTICAL**, plus all raw live histories. GitHub archive retained 30 days.
- A *separate independent Python audit*, based only on the downloaded GitHub archive, independently recomputed complete and raw-row SHA256 and every motion/physiology/memory event; all validation counts were **zero errors**.

## Why M3, and what it ACTUALLY did

M2 failed to improve resource acquisition per living tick once direct food and hazard lookahead were removed. M3 tests whether a minimal trace of last action, collision, and **already-experienced** physical reward/damage is enough to overcome this, without giving the organism a better food sensor or more neural weights.

All tested brains use the EXACT previous M0 **8-action weight actor +6-weight critic = 14 real weights**; the original TD rule, physiology and five N/E/S/W/WAIT primitives are not altered. Only 5 PREVIOUSLY ZERO input-feature slots are reused to represent motor direction alignment with the preceding movement, preceding wall collision, previous actual positive physical reward, actual injury, and a decaying feedback trace. Critic has two reused zero slots. The form of these temporal features is **hand-designed** and has no positional map. It does NOT invent a new exploration policy, neural topology, goal or action.

The original M2 **BLIND** sensor is used in every arm. All action and critic raw sensory inputs contain zero directional food smell, zero adjacent food indicator and zero future hazard lookahead. An actual food encounter increases energy, and that consequence may be used on LATER time steps. Local wall sensing and body energy/injury remain.

**Three environments** (each 32 independently initialized world seeds 7900–7931): baseline gate switch; M2's separately generated connected random mazes with ten wall posts; and no gate switch but all 16 food sites moved at t240 and restored at t360. Seven original-weight controls: frozen, uniformly moving random, original blind TD plastic, motor-only memory with TD, motor + TRUE last physiological feedback with TD, identical motor + SHAM older decoupled physiological feedback with TD, motor + TRUE feedback but FROZEN neural weights. Full experiment **3 ×7 ×32 =672 persistent lives**. Original M0/M1/M2 and sealed M5.6 unchanged; PR stacked on M2 and DRAFT.

## Principal outcome: pooled food per 100 ACTUALLY lived steps

| Arm | Baseline gate | Procedural maze | Food relocation/return |
|---|---:|---:|---:|
| Original blind FROZEN | 5.285 | 4.869 | 5.580 |
| Uniform movement, no learning | 5.738 | **6.337** | 6.082 |
| Original blind TD plastic | 5.523 | 5.347 | 5.842 |
| Previous movement/collision memory only +TD | 5.924 | **6.331** | **6.452** |
| True recent consequence feedback +motor +TD (target) | **6.174** | 5.720 | 6.256 |
| Old/shuffled own consequence feedback +motor +TD | **6.249** | 5.690 | 6.306 |
| True feedback inputs but FROZEN neural weights | 5.207 | 5.107 | 5.430 |

In original baseline, true feedback raises food-per-100 from **5.523 to 6.174** vs original blind TD (+0.651, **+11.8%**). Paired world-seed exploratory bootstrap 95% interval of pooled-rate difference **[+0.192,+1.123]**. This is a real narrow improvement relative to original TD, but fails preregistered **+15%** gate. It also misses +10% relative to uniform movement (6.174 vs5.738, **+7.6%**, paired bootstrap [−0.277,+1.081]).

**The crucial negative result is on procedural geometry.** True feedback makes **5.720 per 100 living ticks**, versus **6.331** with motor-only history and **6.337** with uniform random movement. True-vs-motor-only Δ **−0.611**, paired seed bootstrap95 **[−1.123,−0.115]**. The addition of explicit recent reward feedback may thus be harmful relative to just keeping motor context for these mazes. True-vs-uniform Δ **−0.617**, bootstrap [−1.323,+0.099]; not a demonstrated advantage. True-vs-same-motor-with-OLDER-SHUFFLED-feedback Δ **+0.031**, bootstrap **[−0.654,+0.638]**: no discriminable evidence that the correctly time-aligned consequence is better than a causally unpaired older signal.

In the food-relocation phase itself (ticks 240–359), average food per full 32-seed cohort including zero for dead agents:
- Original blind plastic: **6.5625**.
- True temporal feedback +motor: **6.3125**, **−3.8%**, not the preregistered +15% improvement.
- Motor-only history: **6.71875**.
There is no demonstrated new-food search strategy in the blind temporal-memory system. The system cannot store previous food coordinates.

### Lifespan and risk

All control arms remain biologically primitive with highly variable survival. In procedural worlds, mean lifetime of true-feedback is **318.5** of 480 ticks (vs plain plastic **292.8**, motor-only **343.5**, random motion **332.9**); true-feedback complete survival only **9.4% of worlds** (3/32). True-feedback experiences more hazard contacts than random baseline in procedural mazes (24.63 vs22.53) and does not display an emergent autonomous escape mechanism. None of these are evidence of fly-level abilities.

## Preregistered gates, 11/18, overall FAIL

| Gates | Result |
|---|---|
| 1–11: 672 balanced lives, repeat-byte identity, physics, no hidden-food sensor, past-only memory, older-only sham, exactly 14 original weights, M2 baseline compatibility, 32 novel-world maps, position-independent food relocation, full negative/stop-flag audit | **11/11 PASS** |
| 12: baseline true feedback ≥ plain TD ×1.15 | **FAIL** |
| 13: baseline true feedback ≥ random motion ×1.10 | **FAIL** |
| 14: procedural true feedback ≥ random motion ×1.10 | **FAIL** |
| 15: procedural true feedback ≥ plain TD ×1.15 | **FAIL** |
| 16: procedural true feedback ≥ motor-only ×1.10 | **FAIL** |
| 17: procedural true feedback ≥ shuffled-feedback ×1.10 | **FAIL** |
| 18: relocated phase true feedback food ≥ plain TD ×1.15 | **FAIL** |

**11/11 integrity criteria pass; 0/7 prospective scientific thresholds pass.** The development-science screen is unequivocally **FAIL**, despite successful GitHub Actions, legitimate temporal states and byte-exact repeat.

## Independent audit, no original model code imported

The GitHub artifact was independently downloaded and inspected, with all 672 life histories and original/final weights. Audit independently recomputed and validated:
- **672** logged lives / **96** distinct world-seed×scenario paired maps.
- **235,906** physical action transitions; **698** environmental change events; **13,679** actual food consumption transitions.
- **100,874** checks that a memory source was an actually observed PAST experience; all sham sources at least two ticks old and never future.
- **3,398,800** independent reconstructed action-feature comparisons (allowed memory slots only); no discrepancy.
- Physics independently rechecked: action ↔ actual movement, all wall constraints, no extra actions after death, positions continuous, reward exactly reconstructable from physical energy and integrity; 0 violations.
- All seven matched arms start with same world map and 14 parameters within scenario+seed, no unpermitted sensor leakage into raw blind matrix.
- All complete result files identical, **SHA256**:
  6a72d767c52325ee7f53c04f412dfd363b19bb3553fe55ae20aa47d5a5581dcb
- Independently recomputed compact canonical RAW-ROWS SHA256:
  62345a93f1d5b80b0f176457561dace48b217bcc709f44788aa387a72ea8703b
  exactly matches the M3 program's reported checksum.
- Independent audit errors: **0**.
- The independent audit script and machine-readable report are supplied separately with the downloaded dataset; the GitHub Actions artifact contains original trace-by-trace evidence.

## Interpretation and next discriminating question

This experiment does **not** support the proposition that a scalar recent-reward trace is enough to create efficient BLIND foraging. The slightly better baseline performance is not robust to an independently generated maze, not superior to movement-only, and is almost indistinguishable from a sham feedback stream. The strong contrast between M2's benefits WITH informative food sensors and M3's weak gains WITHOUT them points to a **representational/credit assignment limitation**, not simply insufficient plasticity.

The **motor-only** trace does better than plain TD in the procedural maze (6.331 vs5.347 per 100 lived ticks), but is matched by uniform random movement (6.331 vs6.337); it is not evidence of innovative goal-directed behavior. The engineered alignment feature is a prior that encourages correlated motion, not genuine spatial learning.

A useful next step is NOT blindly adding a planner or copying a fly's labeled action list. One discriminating experiment would give the organism only **self-motion and changes in locally available sensory fields**, with delayed outcomes and a spatially stable environment, then test whether it discovers a transferable structure. Keep the random-moving and motor-only controls; test real world transfer and causal vs shuffled sensor traces. Explicit negative result must remain immutable.

**All STOP FLAGS false:** general_intelligence_demonstrated, spatial_memory_verified, autonomous_operator_invention_verified, evolution_authorized, advance_to_evolution. No direct modification to main or M5.6, and the previous three FlyLoop experiments remain intact.
