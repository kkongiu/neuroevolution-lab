# FlyLoop R11 — Learning real contact recovery before fast motor teaching

**25 November 2026 — COMPLETED scientific screen: FAIL 11/19 gates.** All **10/10 integrity** tests PASS; only **1/9 scientific efficacy** gates PASSES (actual fast response timing). Original R1–R10 regression suites PASS. Two complete R11 campaigns are byte-identical and an external independently implemented raw physics/teaching audit reports zero anomalies within disclosed tolerances.

[Prerecorded R11 protocol](FLYLOOP_R11_PROTOCOL_2026-11-25.md) committed BEFORE R11 implementation/data. **No post-result changes** to cohort, original R9 motor plant, learning rates, time horizons, real recovery definition, threshold gates, sample matching or conclusions. Main and R1–R10 remain untouched. Draft stacked on R10 PR #100.

## The key question and implemented teacher

R9 demonstrated that an independent four-neuron rapid circuit can physically intervene within a genuine 0.05-second microstep but made collision handling WORSE. R10 proved a *functional slow→fast teaching pathway* but nearly all of the slow teacher's approved actions were ordinary safe cruising rather than successful collision recovery.

R11 explicitly implements **real collision → future ACTUALLY EXECUTED movements → verified four contact-free moving microsteps → teach the successful slow action to the fast motor circuit, retrospectively**, without any scripted direction or fabricated physical future.

- Original R1 **16 tanh recurrent slow neurons**, 8 local sensors, 434 seeded neural coefficients FROZEN and 2 continuous analog actuators. Same R9 physical 4×.05 s integrations per .20 s macro and original R9 motor inertia; speed actuator still lacks stop/back/reverse ability.
- A new SLOW researcher-specified **3×28 = 84-weight** online action-outcome critic, initialized all zero, learns actual physical macro progress/contact and **delayed real episode success/failure**. Uses original 16 neuron states, local three wall rays and actual motors only, no global body position/map or unrealized motor futures. Critic fixed learning rate .014, weights clipped [-2,2], frozen at testing.
- Slow motor chooses original 5 continuous R7 local analog candidate alternatives; every fourth training macro exogenously explores the local menu. After a REAL contact onset, up to 12 subsequent slow macros explore uniform [-1,+1] commands on BOTH analog motors without pre-programmed turning direction. This exogenous exploration/critic's progress preference is HUMAN ENGINEERED, not a learned reflex.
- Real recovery detector opens one window of **32 actual 0.05-s microsteps** per contact episode. A recovery lesson exists only if **four consecutive later REAL physical microsteps** each have zero collision and their combined physically observed movement is ≥**.035 world units**. Target = exact slow-issued motor actually executed in the first of those four steps. Input = local fast four-neuron hidden state physically observed at the first microstep AFTER the source contact. A single unresolved sustained wall jam is NOT silently reclassified as success.
- The SLOW critic updates its delayed recovery head **only after the actual event resolves**. The four-neuron FAST pathway receives ONLY the slow-issued analog motor teacher label in retrospective consolidation after all 160 training macros; it NEVER receives injury/reward/contact as direct training targets. The original R9 4-neuron /7 genuinely local-sensor fast state persists; only 10 fast motor output readout coefficients can change under supervised LMS .025; all student coefficients freeze during evaluation.
- **STRICT update-count matched** comparisons: true contact-state/recovery-motor association, motor labels cyclically shuffled among the very SAME actual recoveries of the SAME organism, regular actually safe slow actions taught at the SAME contact-state input samples, and synthetic uniform random motor labels. Each gets exactly the SAME number of supervised fast updates and SAME contact states; when only zero or one successful recovery exists, the shuffled control legitimately has no alternative label and cannot establish useful ablation separation. Also a fixed fast circuit and slow-only arm. Fast priority triggered by the actual R9 immediate last-contact or ray-proximity input; no handcoded physical escape/attack maneuver.

## Complete paired physical cohort

**12 unused original world seeds 12100–12111** × original R1 obstacles / R2 nearfield exogenous ten-rectangle maps × paired static/removal regime (exact five predetermined walls removed at **macro t=160**, after identical prechange worlds) × six controllers = **288 independent natural synthetic organism lives**.

All **288 ×260 macros ×4 real microsteps =299,520 physically executed microintegrations**, **74,880 macro episodes**, with no death, omitted seed or resampling. All six arms in a matched world have **exactly identical 160-macro REAL training physical trajectories**, identical slow states and teacher experience; divergence begins exclusively after fast student consolidation and test-time fast motor override. Every real physical motor, body, salience sensor and learning event retained in original and repeated raw archives.

## PRIMARY preregistered nearfield + five-wall removal, test macros160–259 (12 independent seeds)

| Controller | Real contact microframes / actual traveled meter | Contact persists into next microframe | Unique 1×1 world cells | Average actual distance |
|---|---:|---:|---:|---:|
| **Slow critic only, NO fast motor** | **1.2924** | **94.92%** | **6.75** | **11.41** |
| Fixed 4-neuron fast pathway, never trained | 7.5110 | 98.89% | 5.83 | 9.00 |
| Fast trained on count-matched ordinary safe slow actions | 7.4720 | 98.88% | 5.83 | 9.01 |
| **Fast trained on REAL verified slow recovery actions** | **7.4536** | **98.88%** | **5.83** | **9.01** |
| Fast count-matched shuffled genuine recovery actions | 7.4541 | 98.88% | 5.83 | 9.01 |
| Fast count-matched synthetic random motor labels | 7.6432 | 98.90% | 5.83 | 8.95 |

**Strong negative finding**: true-recovery-fast actually increases contact microframes per meter by a factor of **5.77× relative to slow only** (7.4536/1.2924), and still repeats **795 of 804** physically observed contact-followup opportunities. It retains only **86.4%** of slow covered cells and **78.95%** of true physical distance; therefore cannot be called safer or more competent. The fast motor is physically active on **27.96%** of postchange microsteps, and during the same original slow macro physically overrides on **100%** of eligible immediate after-contact next microsteps. The **timing and priority mechanism works; useful defensive action selection does not**.

At the same time, there is **almost NO measurable difference** between fast true, count-matched shuffled and ordinary lessons: their contact densities cluster around 7.45–7.47. This is NOT evidence that true vs shuffled is equivalently effective in a well-trained model: the actual teacher had almost no genuine recovery experience to distill.

Among 12 UNIQUE nearfield seed training lives (static/removal and all six arms intentionally duplicate training, counted ONCE), SLOW physical experience contained only **FOUR** validated contact→real recovery episodes, across **TWO** seed organisms; only **ONE seed** had ≥2 real successful recovery lessons. Original preregistered gate demands ≥12 episodes across ≥6 seed organisms and ≥6 organisms with ≥2 lessons. Both hard gates FAIL without changing their values.

**Only 4 physically earned supervised updates** across all 12 test nearfield organisms per supervised arm; the other ten seed organisms effectively keep the original fixed fast readout. This is a fundamental sample starvation bottleneck. Broad undirected contact-conditioned exploration increased action variability but did not overcome persistent collision and one-way motor plant limitations.

Actual four-microstep contact recovery after test-time contact: slow-only **6/177 =3.39%** versus true-fast **5/798 =0.63%**, a secondary physically observed negative result.

### Replication in the original alternative obstacle-family generator

In obstacles+removal, slow-only covers **6.42** grid cells at **2.7223** real contact microframes/meter. Genuine recovery-fast covers only **5.25** cells at **14.1067** contacts/meter, approximately **5.18× WORSE**. Thus a poor result is **not restricted to the nearfield generator**. Static conditions and all six controls remain archived.

## All 19 prospective gates: NO RECLASSIFICATION

**10/10 INTEGRITY PASS**: balanced 288 bodies and full twice-run byte identity; genuine R9 .05-s physical motion/walls/injury/energy; original 16/8/434 slow neurons fixed; four-fast-neuron/7-local/10-output system frozen except authorized teaching; new zero-start 3×28 slow real/delayed outcome heads; exact six-arm paired training and exogenous world change; every recovered experience has real collision/actual four contact-free movement proofs; exact true/generic/shuffled/random learning count and input-state match; actual next-microstep fast salience and priority; full raw positive and negative data, no false autonomy/biology/evolution claim.

**1/9 SCIENTIFIC PASS**:
11. ≥12 actual recovered episodes in ≥6/12 nearfield worlds: **FAIL** (4, 2 worlds).
12. ≥6 seed worlds with ≥2 real distinct recovery episodes: **FAIL** (1 world).
13. True fast contacts per real meter ≤80% slow only: **FAIL** (7.454 vs1.292).
14. True fast contacts per real meter ≤85% fixed fast: **FAIL** (7.454 vs7.511).
15. True fast contacts per real meter ≤85% count-matched generic instruction: **FAIL** (7.454 vs7.472).
16. True fast contacts per real meter ≤85% count-matched genuine shuffled recovered lessons: **FAIL** (7.454 vs7.454).
17. True fast contact-followup repeat fraction ≤85% slow-only: **FAIL** (98.88% vs94.92%).
18. True retains ≥90% slow grid coverage and travel and ≥85% RMS spread: **FAIL** (cells 86.4%, real distance 78.95%; spread itself passes).
19. True actual rapid override ≥10% and same-macro next .05-s contact response ≥50%: **PASS** (27.96%, 100%).

**Overall R11 preregistered science SCREEN FAIL = 11/19**. No single efficacy metric or software GREEN rescues this.

Exploratory paired 12-independent-seed bootstrap for recovery-fast minus slow-only covered cells: mean **−0.9167 cells**, 95% exploratory bootstrap **[−2.9167,+0.3333]**; heterogeneous small sample and most student arms have no learned lessons.

## Full reproducibility + independent mathematical and physical audit

- [Successful GitHub Actions full run #45574956380](https://github.com/kkongiu/neuroevolution-lab/actions/runs/45574956380) passed ALL unchanged R1–R10 regression suites, new R11 four-free-step real recovery and teacher provenance/paired-controls tests, first FULL 288-life run and a FULL exact byte-identical second run.
- Official archive GitHub Actions artifact ID **15003971586**, title **flyloop-r11-contact-recovery-teacher-results**, expires **2026-12-25**. Original AND repeated complete JSONs each **524,262,159 bytes**, each exactly SHA-256 **0cee7b8f682c4c93da2b57e727481744308d30abd9c3fd7f79d2672da7b26aad**. Independently reconstructed canonical all 288 raw lives SHA-256 **4ed69f7dd83205408889f219522a649436f2d9efc117e28ada61b490e606bbbc**.
- A **standalone Python auditor importing ZERO original FlyLoop/neuroevolution code** verified ALL **299,520** real microphysics integrations in all 288 organisms, four independently reconstructed fast recurrent states per frame, **18,720** independently ray-cast full 8-local-sensor samples (1/16 physical frames), **46,080** real aftermacro slow progress/contact model updates, **1,236** delayed physical recovery head updates, all **72** complete source-row VERIFIED recovery lessons (duplicated six arms×two regimes across 6 unique original successful experiences), and all **48** physically sourced student supervised weight updates in all worlds. The latter includes original 4 genuine nearfield and 2 obstacles unique seed experiences across matched regimes/arms. **ZERO differences above stated numerical tolerance**; independent full raw canonical SHA and source byte equality match. The audit loads one raw life at a time because each serialized source dataset exceeds 500 MB.
- Transparency/limits: auditor uses its own R9 physical dynamics and slow critic/fast learner math, but original frozen 16-neuron hidden-state generation itself is checked via original regression tests and recorded states, NOT independently regenerated in the audit. Full eight-local wall-ray calculation was sampled at **1/16** actual frames; body motion, collision/energy/injury and fast recurrence were independently reconstructed for ALL real frames. Physical/critic tolerance ≤5e-6/8e-5, reflecting some 9-decimal rounded source diagnostic traces. This is an audit of an artificial numerical world, not biological fly escape behavior.

[Standalone independent audit JSON](FLYLOOP_R11_INDEPENDENT_AUDIT.json) is committed beside this report; complete raw archive and exact standalone independent auditor Python source are also available as conversation artifacts.

## Scientific next decision — should not add more fast neurons yet

The architecture now DOES connect a physically experienced, delayed slow evaluation of an actual contact-recovery sequence to a real fast four-neuron motor instruction, but only two of twelve organisms ever discover positive recovery examples. This is not sufficient training data and the fast route is actively harmful.

A better next **separately preregistered physical causal experiment** should first isolate the motor plant's reachability: from identical physical collision states, experimentally apply full steering/throttle commands with (i) unchanged R9 forward-only sigmoid motor physics versus (ii) a uniformly changed brake/reverse-capable actuator plant; record when actual release within 32 microsteps is physically attainable, with matched time/inertia and no hand-scripted "flee" policy in organism. If reaching free movement is impossible or extremely rare in (i), additional hierarchical learning cannot solve the physical problem. Only after reachability is characterized should we test a learned slow-to-fast recovery teacher again, keeping source R11 negative evidence immutable.

**Claim/stop flags all false**: autonomous escape, autonomous attack, autonomous goal invention, food seeking, general intelligence, biological Drosophila replication, evolution authorization and advance to evolution. A working engineered slow→fast knowledge transfer is NOT acquired natural escape.

R11 remains DRAFT stacked on R10 PR #100; main and all earlier experiment sources intact.
