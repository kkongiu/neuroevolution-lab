# FlyLoop R10 — Slow-to-fast teaching is real, but not yet specific enough to produce reliable defense

**25 November 2026. Complete preregistered result: SCIENCE SCREEN FAIL, 14/18 hard gates (10/10 integrity PASS; 4/8 efficacy PASS).** Original science was run TWICE, all R1–R10 regressions passed and outputs were byte-identical. An independently implemented physical and slow/fast learning audit reconstructed all 276,480 actual microintegrations and 13,172 supervised fast-student updates within stated roundoff tolerances.

[Original R10 preregistration](FLYLOOP_R10_PROTOCOL_2026-11-25.md) was committed **BEFORE the first R10 code/results**; no scientific threshold, random seed, motor setting, learner rate or arm was changed in response to outcome. This report cannot convert a partial behavioral gain into full success.

## What changed conceptually from R9
The user identified that the fast motor pathway should not learn independently from its own injuries: **the slow circuit should teach the fast one what it has learned**.

R10 implements this hierarchy genuinely, with explicit limitations:
- The original original R1 **16 recurrent tanh neurons, 8 local sensory inputs, 434 frozen neural coefficients and 2 analog motor outputs** produce the slow core state and base command.
- A NEW **slow critic with 56 learnable coefficients** (two outputs predicting physically experienced progress and contact from the original slow neural state, actual local rays and actual motor) is the only teacher that learns from physical consequences. It uses online LMS 0.014, exclusively AFTER complete real four-microstep episodes, no position/map/physical oracle. The slow critic itself is a RESEARCHER-PROGRAMMED evaluation rule, NOT independently emergent intent.
- The slow teacher selects from five locally perturbed analog motor commands. It explores other real actions every fourth training macro. Only after executing one real complete macro does it determine whether the example is an **eligible lesson**: actual normalized movement ≥0.15, ZERO physical contact and no exogenously forced exploration.
- The fast 4-neuron recurrent student with 7 local senses and **ONLY 10 motor output coefficients** learns to IMITATE those teacher-issued two analog motor commands, with supervised delta LR=.025. **It NEVER receives reward, contact, injury or progress as direct labels**. Its readout is fixed during evaluation; fast response acts on actual last-microcontact or local ray>.82 in the next real 0.05s substep.
- Fast naive imitation gets EVERY slow motor command including bad lessons; fast validated teacher gets only successful slow-critic-selected commands; fast shuffled receives a REAL validated slow command of the SAME organism at least 8 earlier macros, mispaired to current fast sensory state; fast fixed never learns; direct comparison learns physically as R9 only after the training phase; slow-only never enables fast motors.
- The six arms have precisely **identical real trajectories during ALL 120 training macros**. All fast output physical overrides are locked until macro120. The slow learned critic and supervised fast students are FROZEN for test macros120..239. All real R9 microphysics remains the SAME in all six arms (4×0.05s physical integration per .20s slow macro), retaining inertia, damage and the original inability to stop/back up.

No attack stimulus, scripted flee(), learned genetic reflex, food, evolved goals or biological organism claims.

## Full fixed cohort and sound experimental contrast

**12 new seeds 11800–11811** ×2 R1/R2 procedural 10-obstacle geometry generators ×2 matched static / removal-of-five-odd-index-obstacles conditions before macro120 ×6 controller arms = **288 independent continuous natural lives**. Every life has **240 complete original slow macros and 960 genuinely executed physical 0.05-s microsteps**, yielding **69,120 experienced original macro episodes and 276,480 real physical integrations**. Zero deaths, no original life replacement. Across all cases the independently reimplemented audit found **34,560 real slow critic training updates, 13,172 student supervised readout updates and 5,224 separate direct-reward control readout updates**.

The preregistered statistical inference unit is the **12 independent world seeds**, not hundreds of thousands of correlated physical frames. Both world generators and static/removal conditions remain visible in raw evidence.

## Primary nearfield / removal: physical testing ONLY AFTER teacher→student learning (macros120–239)

| Controller | Actual micro-contact events / traveled meter | Actual repeated contact fraction | Average 1×1 visited cells | Actual mean distance |
|---|---:|---:|---:|---:|
| Original slow teacher, NO fast pathway | 3.9370 | 98.23% | 5.83 | 13.17 |
| Fast circuit INITIAL weights FIXED | 2.8353 | 96.15% | 6.33 | 13.78 |
| Fast with direct physical feedback, R9-style control | 3.0609 | 96.20% | 6.17 | 13.64 |
| Fast copies **all** slow actions indiscriminately | 4.1427 | 98.30% | 5.33 | 13.04 |
| **Fast learns only slow-validated actions (R10 hypothesis)** | **2.4377** | **96.75%** | **6.50** | **13.71** |
| Fast learns mismatched ≥8-macros-OLD validated actions | **2.2227** | 96.74% | 6.08 | 13.83 |

**Positive (but limited):** fast teacher selective training has **41.2% lower collision density** than indiscriminate slow imitation, which passes the preregistered ≥15% advantage gate; it retains ≥90% baseline slow physical movement and exploration, and at runtime actually overrides the base motor in **14.48%** of real post-change microsteps with **100%** of eligible next microstep same-macro after-contact interventions. The slow→fast signal is causally implemented, not simulated only on paper.

**Key negative:** the same teacher-validated fast head is **9.7% WORSE** in collision density than the historical-mismatched teaching control (2.4377 vs2.2227). It is **14.0% better** than the fixed untrained fast pathway (2.8353), just below the **15%** fixed efficacy threshold, so this gate FAILS. The fast teacher's repeated contact fraction **96.75%** is only modestly below slow-only **98.23%** (rather than ≥10% relative improvement), so that gate FAILS too. The fast still remains in contact after **387 of 400** physically verified previous-collision followup opportunities. The reduction in *collision density* largely comes from how the body moves, not from consistently breaking contact.

**Important confound, disclosed rather than hidden:** fast-shuffled gets **867 real supervised updates** versus **963** for fast-teacher, because a mismatched label is skipped until an at-least-8-macro-old genuine validated lesson is available. Thus this control differs in both temporal motor association and effective lesson count; it is a meaningful negative control but NOT a perfectly update-count-matched ablation of state/label alignment. An R11 should separate this.

### Why teaching does not produce a defensive reflex yet

The slow critic validated **963 physical teaching examples** in the twelve unique nearfield training histories (static/removal duplicates counted ONCE); but only **35 of 963 = 3.63%** occurred in any macro where the fast route detected a real collision or wall-proximity event. The original preregistration demanded **≥50% threat-relevant validated lessons** and at least one real teacher lesson in each seed. Despite plentiful valid general locomotion examples, nearly all are **ordinary safe movement**; the fast student isn't receiving enough state-specific demonstrations of resolving a harmful event.

This is a direct limitation of the current teacher definition: "successful safe progress" is not the same as "successful escape from an ongoing collision". The 16-neuron slow core itself has not demonstrated reliably competent long-horizon obstacle avoidance, so more rapid imitation of its ordinary behavior cannot create a successful defensive policy automatically.

### Generalization sanity check: OTHER ordinary R1 obstacle generator
On \`obstacles\`+removal, fast validated teacher produces **4.498 contact microframes/meter** vs slow-only **4.651** (only ~3.3% reduction), and **5.67 cells vs slow-only 5.75**. The older-shuffled teacher is substantially better here (**2.455 microcontacts/meter, 6.75 cells**). Therefore the nearfield-selective improvement does **not** show robust generalized defensive learning across environmental generators.

## ALL 18 prospective hard gate verdicts — NO score manipulation

**10/10 integrity PASS:**
1. 288 balanced worlds, complete source study twice byte-identical;
2. Same actual R9 four real substep motor physics in every comparison;
3. Correct paired original 16-unit /8-local /434 frozen coefficients and exogenous slow motor noise;
4. Slow 56-weight critic zero-initialized, trained only from actual experienced episode outcomes 0..119, frozen in evaluation;
5. Six arms identical actual training trajectories and correctly matched geometry pre-change;
6. Fast 4-neuron /7-sensory /10-output circuitry, NO physical actuation before teaching complete;
7. Physical experience judged by slow critic and supervised fast updated only AFTER teacher labels exist, no fast direct physical targets;
8. Every shuffled fast instruction comes from genuine older same-body validated slow action, students frozen during test;
9. Identical physical local contact/proximity fast gate and segregated R9 direct-reward control;
10. Full real raw logs, unchanged R1–R9/main, no false autonomy or biology claims.

**4/8 SCIENCE PASS:**
11. ≥60 real nearfield unique slow validated demonstrations: **PASS** (963).
12. Validated fast student lesson coverage each nearfield seed AND ≥50% actually salient physical episodes: **FAIL** (35/963=3.63%).
13. Fast teacher microcontacts/meter ≤85% naive imitation: **PASS** (2.438 vs4.143).
14. Fast teacher microcontacts/meter ≤85% older-shuffled teaching: **FAIL** (2.438 vs2.223).
15. Fast teacher microcontacts/meter ≤85% fixed untrained fast: **FAIL**, 2.438 vs2.835 gives only 14.0% reduction.
16. Fast teacher repeated-contact fraction ≤90% slow-only: **FAIL**, 96.75% vs98.23%.
17. Fast teacher retains ≥90% slow baseline visited cells and physical travel: **PASS** (6.50 vs5.83; 13.71 vs13.17).
18. Fast teacher really overrides in ≥10% physical microsteps AND reacts in next .05s inside same macro to ≥50% actual earlier contacts: **PASS** (14.48%; 100%).

**Overall preregistered R10 screen FAIL, 14/18.** Science code working is not enough; the teacher still doesn't provide the appropriate danger-specific motor competence.

## Two complete source runs + independent physical and model audit

- [GREEN GitHub Actions run #45566005957](https://github.com/kkongiu/neuroevolution-lab/actions/runs/45566005957) passed original R1–R9 regressions + all R10 teacher/student/provenance/causality tests and ran the full R10 science experiment TWICE. Complete original/repeat JSON files **470,776,097 bytes EACH** and exact same SHA256 **ac85cf82c9b94e69fd3b09d80dbc352d7a6abfd6028f94480f8c83007dea97f3**.
- GitHub Actions artifact **flyloop-r10-slow-teacher-fast-student-results**, ID **15000062866**, expires **2026-12-25**.
- Full canonical SHA256 of all **288 raw lives**: **57fc8e2800b505711867cb3ea827ac428b4d97ea43e5bb66a71345fb110a2de9**; independently recomputed and verified.
- Separate downloadable [independent audit summary](FLYLOOP_R10_INDEPENDENT_AUDIT.json), plus standalone script and raw data archive. Standalone auditor uses its own actual R9 physics integration and contact/injury rules, independently rebuilds 17,280 actual full local eight-sensor wall ray observations, all original four-recurrent-neuron fast states, slow 56-weight critic online learning, real validated/naive/old-shuffled teaching action provenance, ALL 13,172 supervised student updates and 5,224 segregated direct-reward updates. **0 numerical or causal provenance discrepancies at declared tolerances**. Every one of 276,480 physical microintegrations independently recomputed.
- Limitations of independent audit: source slow original 16-neuron activation dynamics were not independently rederived (original neurons' coefficients frozen and their states recorded), and full local sensory ray geometry was sampled once every **16** physical microsteps, not all 276,480 (all physical motor/position states were reconstructed). Some source trace values are printed rounded to 9 decimals, and corresponding numerical tolerances are disclosed. The independent audit establishes synthetic computational consistency, not biological validity.

## Next proper hypothesis, preserving the negative data

To test the user's actual hypothesis that **slow control trains fast defense**, R11 should change *what constitutes a teacher-worthy episode*, not add handcoded flee():
1. Identify actual physical contact episodes from the slow system's history and evaluate **successful recovery OVER SUBSEQUENT experienced movements**, requiring the prior collision to end and actual progress to resume, rather than only successful no-contact cruising.
2. Let the slow system retain and learn these **temporally extended real recovery episodes**, credit a preceding slow motor response only when measured recovery really occurs, and transfer that action to fast for a matching **pre-contact or first-contact** local state. No fictional counterfactual actions or maps.
3. Make the shuffled-teacher ablation **lesson-count matched**, preserving exactly the same number of supervised fast updates with only motor/state pairing changed.
4. Test with same R9 motor plant first, then with an equally changed brake/reverse-enabled plant across ALL arms as a separate causal experiment. The original motor lacks zero speed and backwards movement; teaching better turns may remain physically ineffective until the body can disengage.

No claims of learned biological escape, attack, autonomously invented survival goal, open-ended intelligence or evolution. All stop flags FALSE.

R10 remains a DRAFT PR stacked on R9 #99; main and all prior experiment branches unchanged.
