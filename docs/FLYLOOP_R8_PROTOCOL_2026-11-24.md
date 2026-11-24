# FlyLoop R8 — preregistered within-life learning from collision, bodily injury and actual displacement
**24 November 2026. Commit before implementation, execution or results.** Draft stacked on R7 #97. NO modifications to R1–R7, M0–M4, sealed M5.6 or main.

## Research question and causal limit
R7 established a **functional causal feedback loop**: 16 recurrent neurons move, their real perceptions train a short-horizon sensor predictor, and that predictor changes subsequent analog motor commands. R7 explored 61% more 1×1 cells than the frozen circuit in changed nearfield geometry, but did **not** reduce physical collision density and appeared to repeat contacts. Can the same organism, **during its own continuous life**, use actually experienced failed movement, collisions and injury to learn to choose *less harmful* motor variations without a scripted escape action?

The **physical-outcome targets, five candidate motor commands, fixed action score and numerical coefficients are researcher designed**. R8 does NOT independently invent a survival goal. Original physics makes integrity loss 0.25 exactly when a collision occurs; therefore "collision" and "injury" are **redundant channels**, not independent evidence. Learn only from **real post-action measurements**, no precomputed map/collision oracle or simulated future candidate physics.

## Locked physical environment and paired cohorts
- **16 new seeds 11200–11215**, separate from prior M/R series. Original 22×22 R1 world and 10 exogenous rectangles per \`obstacles\` and \`nearfield\` R2 generators. Original 16 recurrent tanh neural units, 8 local senses (three wall-proximity rays, motion, last collision, energy) and original 434 initial neural coefficients, seeded R2 Gaussian motor noise (σ=.12). The recurrent/input/hidden weights NEVER change. The two analog actuators follow unchanged R1 speed/turn inertia, physical damage 0.25/contact and energy 0.02/step.
- Two regimes per seed/geometry: \`static\` retains 10 walls; \`removal\` starts exactly identically and removes original rectangles 1,3,5,7,9 **just BEFORE local perception at t=160**. No environment-change flag is provided to the organism. Prechange trajectories across regimes **must be identical** for each seed/geometry/controller (unless actual death earlier).
- **Six arms** per seed/geometry/regime = **16×2×2×6 =384 lives**, max **320 actual movement frames each**, potential 122,880. The life terminates physically if energy/integrity reaches zero; no resurrection or resampling. Three original R4 84-coefficient *sensory* heads may train from birth on actual next three rays, identical to R7. Only R2 positive control may change existing 34 neural motor readout coefficients; all others preserve the full original 434 initial neural parameters.
- Exactly five local continuous candidate commands from the *original noisy recurrent motor*: base, ±0.35 steering, ±0.20 throttle (clamped [-1,1]); SAME for R7 sensory-only, true-physical, older-physical-sham, and random-menu arm. All arms physically follow original base for **first 50 frames** (R2 begins its own existing learning immediately). No prewritten wall-following/turn-after-contact manoeuvre.

## NEW real physical-outcome heads, online only
Two independent **3×28 =84-coefficient** linear LMS heads per living organism, \`true\` and \`older_sham\`, all coefficients initially exactly ZERO, with the **same R4 28 real local pre-action features**: 16 current recurrent neuron activations, current local three wall-proximity rays, current executed TWO-component analog command, six wall-ray×command interactions and constant bias. Outputs are clipped [0,1] from hardcoded starting offsets [0.25,0.50,0.25] plus learned weighted features. They predict actual next-step:
1. wall collision occurrence, 0 or 1;
2. **real physical displacement divided by absolute maximum single-step distance 1.2×0.2**, in [0,1];
3. actual integrity loss divided by 0.25, 0 or 1 (redundant with channel 1 in this R1 world).
Head learning rate **0.025**; every coefficient clipped [-2,2], one update/real after-action step, NO external training or head transfer.
- TRUE trains only on the just-observed REAL three-outcome vector, after the action and with its actual executed command.
- OLDER_SHAM is given the otherwise-identical original features/executed command but trains on an **actual physically measured outcome** of the SAME organism from source t−6 or EARLIER, drawn with a separate deterministic RNG; before six actual earlier outcomes it gets NO update. Both heads are evaluated each frame against the SAME REAL current physical outcome, not against sham label, thus physical-prediction accuracy can be compared along exactly identical frozen-controller trajectories. The independent sham RNG never advances original motor-noise RNG.
- Both heads learn their own lifetime experience from time zero; no cross-episode information, no absolute x/y, global map or ray casting in the head, no future target before action. The R7 sensory head still learns local future wall sensors from real movement in all non-R2 arms and influences R7 novelty score exactly as before.

## Locked researcher-engineered candidate scoring
At t≥50, for each of five *possible analog motor VALUES*, calculate current learned head predicted collision, relative physical progress, normalized integrity damage **before physical execution**. Separately calculate R7 predicted three-wall-ray change from the base candidate. R8's fixed, untrained **physically informed decision score** is:

\`score = +0.25*sqrt(R7 predicted three-ray MSE vs base) +0.60*predicted_normalized_displacement −0.50*predicted_collision −0.50*predicted_integrity_loss\`

Choose the candidate with largest score; ties favor base (index 0). No hand-coded "if hit wall then turn" or concrete route target. It may still underexplore, freeze or exploit prediction errors. The SHAM policy uses its OLDER_SHAM head in that same score; the TRUE policy uses the TRUE head. Models receive the REAL outcome of the actually chosen motor only, and update strictly AFTER movement. No alternate candidate world clone or actual collision information before action is supplied.

**Six controls:**
1. \`frozen\`: original R7 frozen noisy recurrent motor; train both physical heads passively for *identical-trajectory true vs delayed-sham predictor tests*, but never alter action.
2. \`r7_novelty\`: original R7 learned sensory-change-only selection, same menu, no physical head guidance; physical heads may train passively; primary behavioral comparator.
3. \`r8_true_physics\`: same R7 sensory novelty + online TRUE collision/displacement/damage head as the fixed physical outcome score above.
4. \`r8_older_sham\`: identical rule/action menu/learner except physically outcome HEAD trained against the organism's own ≥6-frame-OLDER real physical outcomes.
5. \`random_menu\`: independent seeded uniform choice from same five candidates after warmup, physical/sensory models may train passively but do not select.
6. \`r2_progress\`: original R2 physical 12-step displacement minus contact penalty, only 34 motor readout coefficients adaptive, no candidate action selection. This explicitly contains a hand-engineered physical reward and is NOT the same motor action menu.

## Metrics and meaningful safeguards
For PRIMARY \`nearfield\` + \`removal\`, evaluate only physical frames **t=160..319**. For every life save every real step, collision, proposed/actual motor, original and next eight/three local sensors, 16 hidden units, all five pre-action candidate predictions/scores, source time and REAL older sham training label, three bodily consequence targets, integrity/energy/position, risk-model true+sham updates/hashes, 3×84 original sensory head coefficients/hashes and 2×84 new physical head final coefficients. Full R7 menu and physics can be audited and repeated byte-identically; no hidden future state.
- **Physical contact density**: pooled collisions divided by actual meters traveled, not by nominal lives.
- **Repeated collision fraction**: count cases in which a real collision at t−1 is followed by another real collision at t (both within evaluation window), divided by the number of prior actual collisions with an actual next step; also record absolute repeated-collision counts; if denominator zero, original gate fails, no division-by-zero victory.
- Mean distinct 1×1 postchange cells, actual meters, alive steps, and postchange RMS trajectory radius to avoid being safe only by not moving.
- **Zero-displacement fraction** after wall hits and among ALL real frames, because collision effects can be trivially avoided by stopping if speed permits.
- Physical prediction error (3-channel MSE) on EXACT same physical trajectory for TRUE vs older-sham in the frozen arm, all actual after-action targets. Do not treat frames as independent seed replicates.
- Per-seed bootstrap95 over **16 world seeds**, 2 geometry/2 regimes nested when relevant, exploratory and not a substitute for locked effect-size gates. Never hide unfavorable world family, static results, or seed-level negative outcomes. No food, predators, agency or evolutionary claims.

## Nineteen immutable gates (10 integrity, 9 scientific)
**Integrity:**
1. Exactly **384 balanced physical lives**, and TWO complete serialized JSON outputs must be BYTE-IDENTICAL in GitHub CI; no lost seed, arm or geometry.
2. Original R1 continuous position/motor inertia/energy/integrity, collision damage .25, no wall penetration or post-death movement; genuinely dead episodes retained.
3. Matched original 10-rectangle map, body pose and **434 initial coefficients** per seed×geometry×regime across all six arms; original exogenous motor noise identical over shared alive prefixes.
4. Exactly original 16 persistent neural units, 8 local sensory channels, original recurrent/input/hidden weights unchanged. No secret map/absolute destination/future rays or collision oracle.
5. Two physical heads EACH exactly 84 weights all zero at birth, 3 predicted outputs clipped [0,1] with fixed [0.25,0.5,0.25] baselines and same physical action features; only real after-action labels; original sensory heads 3×84 correct and trained after actual sensory experience.
6. TRUE head source target is physical post-step *current* real collision/displacement/injury; never knows future target before physical integration, no training on other candidate outcomes.
7. SHAM target from the same organism's actual previous physical outcomes at source ≤t−6, with independent RNG and not by fabricating experiences; both heads scored on true current target.
8. Same exact five analog candidate commands, fixed pre-action score, no world lookahead; all motor decisions actually executed before outcome learning, 50 common warmup steps.
9. Static vs removal environment and same-arm body trajectories identical through t=159; exogenous removal is EXACT five predefined walls from t=160 without alerting actor; frozen and other original 434 weights remain identical, except original R2 34 readout changes.
10. Complete unedited raw records, exact full deterministic replay, original R1–R7 and main unchanged, honest negative results, all food/escape/goal/intelligence/evolution stop flags FALSE.

**Scientific efficacy (all required):**
11. On SAME frozen physical trajectories, TRUE physical head mean postchange real-target 3-outcome MSE ≤**0.85×** delayed-actual-SHAM head MSE (denominator zero FAIL).
12. In nearfield removal, TRUE repeated-collision fraction ≤**0.80×** R7 sensory-only policy's fraction (and R7 denominator ≥50 actual transitions).
13. In nearfield removal, TRUE pooled real collision events per meter ≤**0.75×** R7 sensory-only collision density (R7 actual meters >0).
14. In nearfield removal, TRUE mean postchange distinct 1×1 grid cells ≥**0.90×** R7; don't win by withdrawing from exploration.
15. TRUE mean postchange distinct grid cells ≥**0.90×** matched random-menu baseline; don't merely become motionless.
16. TRUE mean postchange spatial trajectory RMS radius ≥**0.85×** R7; detect confined orbiting.
17. TRUE repeated-collision fraction ≤**0.85×** older-actual-SHAM-guided motor controller's repeated fraction (sham denominator >0).
18. TRUE policy must **really override** the original neural base command for ≥**10%** of alive postchange frames; otherwise it isn't an active learned physical consequence selector.
19. TRUE total actual postchange meters traveled ≥**0.90×** R7's mean physical distance; protect against merely reducing contacts by failing to move.

**All 19 must pass.** A green CI is only software-correctness. Reduction in repeated collisions alone does not establish smart navigation. A successful narrow R8 would mean a RESEARCHER-GIVEN harm-sensitive motor scoring rule benefits from real sensorimotor learning, NOT autonomous invented survival or independent biological fitness.

**STOP FLAGS all FALSE:** \`food_search_demonstrated\`, \`escape_reflex_learned\`, \`autonomous_goal_invention_demonstrated\`, \`general_intelligence_demonstrated\`, \`biological_fly_replication_demonstrated\`, \`evolution_authorized\`, \`advance_to_evolution\`. Set \`closed_loop_physical_prediction_guides_action_implemented\` true if and only if implementation verifies this causal mechanism; \`effective_collision_learning_demonstrated\` true ONLY if ALL scientific gates pass.
