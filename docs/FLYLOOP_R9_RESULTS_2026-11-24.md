# FlyLoop R9 — Rapid defensive motor bypass responds in 0.05 s but makes obstacle avoidance WORSE

**24 November 2026 | COMPLETE. Precommitted developmental science screen: FAIL, 10/16 gates (9/9 integrity, 1/7 efficacy).** The completed GitHub Actions workflow is GREEN, two entire science runs byte-identical, and a separate independent implementation reconstructs all physical/neural/motor and learning events within tight floating-point tolerances. Do NOT confuse a successful substep intervention with useful defense, acquired escape or attack.

[Original preregistered 16-gate protocol](FLYLOOP_R9_PROTOCOL_2026-11-24.md), committed BEFORE model implementation and simulation. No post-result tuning of motor physics, random seeds, weights, learning coefficient, salience trigger or efficacy thresholds.

## Why we tested this

R8's slow 16-neuron recurrent body learned physically correct injury/contact outcomes but could still remain in contact with a wall repeatedly. Biology suggests some defensive sensorimotor routes can bypass slow deliberation. R9 separates two distinct hypotheses:
1. Can a local 4-neuron event-gated pathway actually INTERRUPT the slow motor DURING one .20-second slow macroframe, with a .05-second response to physical contact? **YES, by construction and measured on real simulated microphysics.**
2. Does this response, and ONLINE learning of motor consequences rather than a fixed or old-feedback fast circuit, make the body reduce repeated collisions **without losing exploration**? **NO, decisively fails all six preregistered behavioral efficacy gates.**

### CRITICAL PHYSICAL DISCRETIZATION DIFFERENCE FROM R8
ALL five R9 arms use a NEW common microphysical integrator: 4 real .05-second integrations per original .20-second slow macro, with \`micro_alpha = 1-(1-.18)^(1/4)\`, R1 actuator targets/inertia, obstacle/block logic, energy=.02/4 per real microstep and injury=.25/4 per real collision microstep. Old R8 integrated only once per .20 s; **do NOT compare R8 vs R9 absolute collision counts per real meter directly**. Comparisons WITHIN R9 are physically matched.

## Locked experiment

- **12 NEW independent seeds 11500–11511** ×2 unchanged exogenous R1/R2 procedural ten-rectangle worlds (\`obstacles\` and \`nearfield\`) ×2 paired regimes (static world or remove exactly five odd-index rectangles before perception at macro t=140) ×5 controllers = **240 independent lives**. **ALL 240 bodies complete 280 original slow macroframes ×4 actual microframes = 268,800 real motor/physics micro-integrations.** No animals resampled, skipped or resurrected.
- Original 16-recurrent-unit /8 local sense /434 seeded synapse slow motor policy remains FROZEN. Its motor command and seeded R2 Gaussian noise update ONCE each .20 s. All control arms share exact initial body, slow and fast neural weights, geometry and slow noise common prefix.
- New independent **4-neuron recurrent FAST circuit**, 7 LOCAL sensor inputs sampled every .05 s: 3 real proximity wall rays, previous physically measured contact, physical speed and angular speed, internal energy depletion. Event gating when last real contact occurs or any local ray >.82. No global position, target or pre-scripted choice of left/right.
- At or after macro20, a fast event can OVERWRITE the current two continuous motor commands during the CURRENT .20-s slow brain interval. Four hidden fast neurons update every .05 s; only **ten fast output coefficients** may learn. Actual output perturbation noise seeded independently. Motor acceleration and angular inertia still follow real physics, so neural override cannot instantaneously reposition the body.
- Same rapid trigger and world physics across five arms:
  1. \`slow_only\`: never activates the fast motor (original slow motor maintained over microframes).
  2. \`fast_fixed\`: event-gated fast four-neuron circuit, output coefficients permanently frozen.
  3. \`fast_real\`: same initial fast circuit but updates ten output weights AFTER actual fast motor movement from researcher-coded physical feedback \`0.7*normalized actual movement−1.2*actual contact\`; online eligibility LR=.002.
  4. \`fast_old_sham\`: same learner/noise, but training from a real measured same-body feedback event **at least 12 real microsteps in the past**, independently sampled.
  5. \`fast_random\`: same event gate, random continuous [-1,1] alternative motors.
- First 20 slow macrosteps same actual motion for all controls; static/removal regimes physically identical through macro139. Both exactly verified.

## MAIN preregistered physical outcome: nearfield worlds after removal (macros 140–279; 12 independent seeds)

| Controller | Mean postchange 1×1 cells | Collision microframes / actual traveled meter | Contact followed by another contact | Mean real distance | Mean spatial RMS radius |
|---|---:|---:|---:|---:|---:|
| **Slow-only original recurrent** | **9.50** | **3.2141** | **93.77%** | **15.0118** | 1.2609 |
| Fast 4-neuron circuit, fixed output | 7.75 | 7.2092 | 95.96% | 13.2008 | 1.1509 |
| **Fast 4-neuron real-feedback LEARNER** | **8.00** | **8.6089** | **96.83%** | **12.8549** | 1.2627 |
| Fast circuit, old REAL feedback ≥12 microsteps old | 7.83 | 8.4506 | 97.43% | 12.6717 | 1.1629 |
| Fast triggering randomly chosen motor | 4.50 | 30.4338 | 99.68% | 8.6308 | 0.7283 |

**THE LEARNED FAST PATH ACTUALLY INTERVENES:** on **38.47%** of physically available postchange microframes, and in **993/993=100%** of contact-followup opportunities **within the SAME original slow macroframe**. Thus the causal faster bypass is functional, with as little as one real .05-second microstep delay after measured contact.

**But it makes the body WORSE:**
- Real-contact microframes per actual meter **2.68×** slow-only, not 20% lower as preregistered.
- Contact followup rate **96.83%**, WORSE than slow-only **93.77%** (and worse than fixed fast **95.96%**; only a small advantage vs old-feedback sham **97.43%**, nowhere near the required ≥15% reduction).
- Coverage **8.00 vs9.50 cells**, only **84.2%** of slow baseline, below mandatory 90% preservation threshold.
- Actual travel distance **12.85 vs15.01**, only **85.6%** of slow baseline (required ≥90%). RMS spread 1.2627 is comparable to slow's 1.2609, but this alone cannot rescue poor movement and contact metrics.
- More physically random movement is much WORSE still: 30.43 contact microframes/m and only 4.50 cells. Thus the reduced exploration of fast policy is not a useful safety effect.

Physical **four-microstep (.20s) contact recovery** fractions, on eligible observed actual collisions: slow-only **23.48%** (135/575), fast fixed **14.02%** (159/1,134), fast actual-feedback learner **12.12%** (160/1,320), old-feedback sham **10.29%** (131/1,273), fast random **1.22%** (38/3,124). Note that repeated contacts within an extended jam generate multiple eligible contact trials; this is a diagnostic, not independent scientific samples.

**Generalization NOT rescued in other geometry:** \`obstacles\`+removal fast learner explores **7.33 vs9.25 cells** with **9.178 vs4.158 contact microframes/meter**, worse. Static environments also worse and all raw outcomes retained.

Paired exploratory bootstrap over only the 12 truly independent nearfield world seeds: fast REAL minus slow-only mean explored-cell difference **−1.50** cells; exploratory 95% interval **[−3.9167,+0.4167]**. This interval is uncertain on coverage, but collision and behavioral gates already fail.

## Frozen 16 preregistered gate verdict

**9/9 integrity PASS**, covering full balanced and identical replay, correct original physical R1-inspired micro-kinematics and proportional damage, original fixed slow neurons and paired Gaussian noise, exact 4-fast-neuron/7-local-sensor network with 10 trainable motor coefficients, after-physical-event-only one-microstep overrides, genuine real reward and historical ≥12-microstep-old sham labels, identical warmup/pre-change worlds, complete raw histories and old code untouched/stop flags false.

**1/7 science PASS:**
10. Fast real repeat contact rate ≤80% slow: **FAIL**.
11. Fast real contacts per actual meter ≤80% slow: **FAIL**.
12. Fast real repeat contact rate ≤85% fixed fast: **FAIL**.
13. Fast real repeat contact rate ≤85% old-feedback sham: **FAIL**.
14. Fast real preserves ≥90% coverage versus BOTH slow and random: **FAIL** (vs slow only 84.2%).
15. Fast real preserves ≥85% spread AND ≥90% real movement distance versus slow: **FAIL** (distance 85.6%).
16. Genuine fast motor override ≥10% postchange microframes AND response at next physical microstep within same slow frame ≥50%: **PASS** (38.47%, 100%).

**Overall prospective DEV scientific screen FAIL =10/16**, despite software GREEN. This is a negative result on the *researcher-designed* rapid reward-learning rule and current motor plant, not a proof that fast biological reflexes are useless.

## Reproducibility, archive and separate independent verification

- Completed GREEN original [GitHub Actions run #45486534766](https://github.com/kkongiu/neuroevolution-lab/actions/runs/45486534766), unchanged R1–R8 regression suites, R9 microphysics and neural/sham tests and two FULL physically simulated studies pass. Both full JSON science outputs are **BYTE-IDENTICAL**, each has SHA256 \`771bec3535f1abc4319fffffb122d4b32c820f629789ee435972e38559b52a7f\`.
- GitHub artifact \`flyloop-r9-fast-defensive-microphysics-results\`, id **14963649952**, retained until **2026-12-24**. 240 complete per-step full physical/neural/motor traces and separate identical rerun. Original raw-rows SHA256 \`226ae26d75ac12694686e486b78084ea7467b3918531c50550771cbe3000d061\`.
- An independent Python auditor, with NO FlyLoop/neuroevolution source imports (only stdlib + orjson parsing), regenerated both original randomized ten-rectangle geometries, all 434 original slow neural coefficients, 16 tanh slow units and seeded slow Gaussian noise, all FOUR FAST recurrent states and seven real local sensors, original fast randomly initialized readouts, all real **268,800 .05-second motor/body physical integrations**, **35,065 fast neural readout weight updates** and **17,327 actually sourced old real reward targets**, including full nearfield physical contact/followup/position statistics. Audit verified numeric physical positions/angles within 2e-8 and terminal head coefficients within 2e-12, with maximum absolute head difference ~1.89e-12. During the completely independent floating-point recomputation, **2,674 intermediate exact textual coefficient hashes differed at last-bit precision**; this is explicitly disclosed rather than falsely claiming bit-for-bit independent arithmetic. The TWO source full scientific datasets themselves remain byte-identical, and no physically meaningful numerical discrepancy was observed at the stated tolerances. [Independent audit summary](FLYLOOP_R9_INDEPENDENT_AUDIT.json) and standalone script downloadable with full original archive.
- Biological and epistemic limitation: a coherent synthetic physical/neural software simulation does NOT establish actual nervous-system physiology, innate fear, autonomous goal discovery, real insect escape, attack ability or evolutionary reproduction.

## The unexpectedly important motor-plant limitation

Original R1 uses \`desired_forward_speed = 1.2 * sigmoid(1.6 * throttle)\`, with \`throttle∈[-1,+1]\`. Hence even maximum negative throttle commands a strictly POSITIVE forward speed of about **0.202 world units/s** rather than full stop or reversal; current code's motors **cannot brake to zero or back away**. Steering is continuously inertia-limited by \`1.9*tanh(1.6*steer)\`, reaching only about 1.75 rad/s at its strongest command, and the motor signal is low-pass filtered at each microstep. A circuit that can intervene at .05 s may still spend many subsequent physical steps pushing forward against the obstacle, especially if its random output makes the rotation less useful.

This is a **mechanistic hypothesis derived from actuator equations**, NOT experimentally demonstrated as the sole cause of R9's failure. A principled NEXT diagnostic, without rewriting R9, is to test reachability from identical real collision states using experimentally FORCED motor commands under the locked original unidirectional and a separately defined bidirectional/braking motor plant, apply the same plant to ALL relevant arms, and then ask whether learning still has unique benefits beyond mere physical escape availability. Do not add a fixed \`flee()\` action and call that learned defense.

**STOP FLAGS all FALSE:** autonomous_escape_demonstrated, autonomous_attack_demonstrated, autonomous_goal_invention_demonstrated, food_search_demonstrated, general_intelligence_demonstrated, biological_fly_replication_demonstrated, evolution_authorized, advance_to_evolution. \`fast_motor_pathway_implemented=true\` for the physical bypass itself, but \`useful_fast_defense_demonstrated=false\`.

PR #99 stays DRAFT stacked on R8 #98; main and all previous experimental branches source untouched.
