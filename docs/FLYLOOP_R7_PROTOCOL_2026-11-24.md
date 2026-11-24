# FlyLoop R7 — Prospective protocol: online sensory learning actually changes the next action
**24 November 2026. Freeze BEFORE first R7 code run or outcome.** DRAFT stacked on R6 branch; R1–R6, M0–M4, sealed M5.6 and main untouched.

## Research question and major caveat

R6 showed experimentally externally trained action-conditioned sensory heads generalize absolute sensor values across worlds, but failed preregistered precision of action-caused SENSOR DIFFERENCES. R7 for the first time feeds an **online learned local sensory model back into action selection DURING THE SAME artificial organism's life**, rather than transplanting heads from a different virtual life. Can closed-loop *predicted controllable sensor change* produce physically broader navigation WITHOUT merely circling, or does it select spurious model artifacts?

**The preference "maximize predicted short-horizon sensory difference" and the five local candidate motor perturbations are RESEARCHER PROGRAMMED, not a discovered animal goal.** This is not a self-invented navigation drive, escape reflex, biological fly, evolved organism, or proven open-ended intelligence. R7 uses model-based action selection with five locally proposed continuous commands, NOT a new end-to-end learned actor or a rule "when obstacle then turn". It still has the original 16 recurrent neurons generating the baseline analog motor and ORIGINAL R2 physical motor perturbation, which is the reference for five temporary candidate motions. We do NOT provide absolute map locations, routes, food or other global information to the agent.

## Immutable experimental cohort

- **16 wholly new seeds 10900–10915**, entirely separate from all earlier FlyLoop M/R cohorts.
- **Two 10-obstacle geometry families**: original R1 procedural and R2 near-spawn generator.
- **Two experimentally paired regimes**: \`static\` retains its exact 10 obstacles throughout; \`removal\` starts identical to static and removes ORIGINAL rectangles with indices 1,3,5,7,9 immediately **BEFORE perception at t=240**. Both regimes have identical physical layouts and starting organism states before t=240. Removing walls cannot trap the body in new obstacles; it is an exogenous physical environmental change, not a reward-dependent adaptation. Both retain the same arena outer walls. Inform the agent ONLY through its three actual physical ray sensors; no switch flag or global map or position.
- **Six controller arms** ×16 seeds×2 geometry families×2 regimes=**384 independent continuous lives**, up to 480 actual R1 physical frames per life, dt=.2. All share same seed's **434 randomly initialized R1 coefficients**, initial body pose, original exogenous map and R2 Gaussian noise on common live prefixes. Physical energy/integrity and early death are real, NEVER resurrected; do not discard poor worlds or resample. Compare **16 INDEPENDENT SEEDS**, not 480 correlated frames.
- 16 original recurrent tanh neurons, 8 LOCAL original R1 sensors and continuous dual actuators; original initialized recurrent/input/hidden/motor coefficients frozen in ALL arms other than \`r2_progress\` positive control. R2 progress control may adapt **ONLY the 34 original motor readout coefficients** using its unchanged R2 implementation.
- Online predictor heads are R4's original **3×84 coefficient** true/blind/yoked LMS heads initialized ALL ZERO at life start and updated from the organism's own REAL physical next three wall rays **ONLY AFTER each action** at original fixed LMS learning rate .014. NO transfer from R6, no external clone interventions/labels, no future ray, no mutation, no cross-life fitted weights or hidden offline training. The yoked head trains using the same organism's actual motor ≥6 steps old (independent RNG); the actual next rays are still genuine and shared. All heads see the original 16 recurrent state and current local three rays; command features differ as in R4.
- **First 60 frames t=0–59 warmup**: EVERY arm uses exactly the same R2 noisy recurrent commanded motor, except \`r2_progress\` which can already learn R2 output readout synapses. The passive, actual and yoked models learn online in this warmup but DO NOT steer. Model-driven selection can begin at t=60 ONLY.
- Five EXACT researcher-defined temporary candidate continuous commands at each later step, computed exclusively from the organism's just-produced R2 real noisy base motor \`(throttle,steer)\`:
  1. base motor;
  2. (base throttle, clamp(base steer−**0.35**,−1,+1));
  3. (base throttle, clamp(base steer+**0.35**,−1,+1));
  4. (clamp(base throttle−**0.20**,−1,+1), base steer);
  5. (clamp(base throttle+**0.20**,−1,+1), base steer).
  No trajectory lookahead via actual world physics, no global information, no immediate-wall/escape if/else reflex, no prewritten route; these are only continuous local MOTOR candidate values. Clipping can cause duplicate commands, a limitation disclosed.
- For model-guided arms at t≥60 compute the model's PRE-action next three-ray prediction for each candidate with its CURRENT learned weights, original current local rays and 16 recurrent state. The **ENGINEERED intrinsic score** is mean squared difference between each candidate's three predicted next rays and the model's prediction using the base command, with tie-breaker favoring candidate zero (base). The candidate with highest score is physically executed. No fabricated future observation is accepted as training: all heads then learn ONLY from the real next three ray readings of the chosen motor. If the head still has all-zero weights, all choices equal base.
- IMPORTANT: Because the head does not predict distances to walls explicitly as an objective, this "maximize predicted sensory variation" can increase collisions or induce repetitive circling and should not be falsely described as learned obstacle avoidance. The 5-way action scoring policy is RESEARCHER-SPECIFIED, not itself learned.

### Six preregistered arms, all paired
1. \`recurrent_frozen\`: original noisy recurrent motor and physics; head stays zero, no candidate action selection.
2. \`passive_predictor\`: online true/blind/yoked head learns on actual sensations, but never changes original noisy recurrent action. This MUST have exactly the same physical trajectory as \`recurrent_frozen\`, demonstrating that model training alone cannot cause the outcome.
3. \`true_closed_loop\`: true online motor-conditioned head influences next continuous action via the fixed 5-candidate score after warmup; all heads update on real consequences.
4. \`yoked_closed_loop\`: same candidate scorer, uses OLD-MOTOR mispaired \`yoked\` head to choose; all heads update on real next sensors. The same action menu and motor noise as true, but relevant learned action conditioning is wrong.
5. \`random_menu\`: at t≥60 select uniformly from the identical five candidate motor commands with an independent seeded RNG, without predictive guidance; online heads can train but do NOT guide. Controls for movement simply becoming more varied due to the menu.
6. \`r2_progress\`: original R2 motor-readout plasticity with physical net displacement/contact reward and unchanged R2 seeded noise; may train heads passively but NEVER uses the candidate selector. A benchmark for researcher-engineered physical motor reward, NOT a matched action menu. Interpret results accordingly.

## Logged data & physically meaningful endpoints
Record all 384 lives and up to 184,320 actual R1 motor integrations; every frame includes original 8 local sensor readings, original 16 hidden states, neural base motor, EXACT R2 injected Gaussian perturbation, all 5 candidate commands, head predictions for each candidate (actual & yoked arms), chosen candidate index, actual physically executed motor, actual next 3 wall rays, physical collision/pose/energy/integrity, pre-update head MSE, historical motor source tick ≥6 older, update events, model coefficient hash summary (every 20 frames) and complete three 84-head final weights. Record original map, removed map, true switch frame, original/final 434 core weights, exact all-frame visited grid cell positions and traveled distance. No more data than exists after early death.

Primary **nearfield+removal** outcome:
- total **unique 1×1 cells** reached **during t=240..479 only** (not cumulative lifetime);
- pooled actual collisions PER meter traveled during same post-change phase, with numerator/denominator aggregated over actual alive movement (do NOT bias by death);
- per-life post-change trajectory RMS radius around the POST-phase centroid to detect circular confinement;
- mean number of novel post-change cells not seen before the environment change (ecological adaptation diagnostic, secondary).
- separate report for \`static\` nearfield, \`obstacles\` both regimes, and whole-life. Compare model predictive late MSE true vs time-mismatched yoked on exactly SAME \`passive_predictor\` trajectory.
- Count how often an actual model-dependent action choice differs from base and its causal source (model updates, not external command). No hidden simulation of candidate future outcomes.

Per-seed bootstrap95 over 16 independent seeds of postchange real visited cell delta (TRUE minus frozen, yoked, random, original R2) exploratory only; do not claim significance of actions as independent samples. Repeat whole experiment twice, require serialized JSON BYTE-IDENTICAL.

## Exactly 17 prospective gates, 10 integrity +7 efficacy, all required
**Integrity:**
1. Exactly 384 balanced lives with correct 16 seeds×2 geometries×2 regimes×6 controllers and two full byte-identical repeat outputs in GitHub Actions.
2. TRUE original R1 continuous physics: actual motor inertia, no inflated-obstacle/boundary penetration, real injury and energy accounting, no resurrection or motion after death.
3. Exactly identical original 434 coefficient initialization and initial body/map for all matched seed×geometry×regime arms; same motor noise stream on the shared alive prefix, independent of mode.
4. Original 16 recurrent tanh neural states, 8 local sensory channels, no global position, map hint, future sensors, meal/food, predator or hidden planner input.
5. Three 84-coefficient heads all-zero at birth; train ONLY after real next-ray observations with original R4 LMS; no head updates before physical action, no offline training/weight transfer; frozen comparator heads do not change.
6. The actual sensor-model arm's candidate scoring uses ONLY pre-action local sensors, current hidden and actual 5 candidate motors, and head weights BEFORE learning from current step; candidate menu exactly as protocol, same menu for yoked and random.
7. YOKED input comes only from SAME individual's genuinely executed motor at most t−6, with independent RNG and no future motor history; actual-training and sham-training labels always the SAME physically observed next rays.
8. \`passive_predictor\` and \`recurrent_frozen\` actually follow EXACT same body and real motor trajectory on each paired world despite the former's head learning, and all head-adapted intervention feedback after outcome only.
9. Static and removal worlds/subjects match identically for **ALL pre-change frames t=0..239** within each matched seed×geometry×arm; the \`removal\` world removes exactly 5 predefined rectangles at t=240 before perception and never adds obstacles or traps a body.
10. Full raw neuron/motor/model/physical frame histories, all prospective gates and explicit unchanged code and false autonomy/food/evolution stop flags archived.

**Scientific efficacy:**
11. Online true head trained in the passive matched-life control has late (t≥240) real-target next-ray MSE **≤0.95 × its own yoked head MSE**, evaluated on identical true physical behavior; if yoked error zero, FAIL.
12. Model-guided \`true_closed_loop\` postchange nearfield+removal **mean 1×1 visited cells ≥1.10 × recurrent_frozen** postchange cells.
13. Same TRUE postchange nearfield+removal **pooled collision events per real meter ≤0.90 × recurrent_frozen** (if denominator zero, FAIL).
14. Same TRUE postchange nearfield+removal mean visited cells ≥**1.10 × yoked_closed_loop**.
15. Same TRUE postchange nearfield+removal mean visited cells ≥**1.10 × random_menu**.
16. Same TRUE postchange nearfield+removal trajectory RMS radius ≥**0.90 × recurrent_frozen**; collision reductions alone do not qualify if the organism just circles in a tiny patch.
17. TRUE model-guided controller must ACTUALLY override original base command after warmup on **≥10% of its alive post-change physical actions** (across all worlds; otherwise it is not a closed action loop even if unrelated metrics pass).

All 17 are mandatory. A green GitHub test suite does NOT imply scientific success. Even all seven efficacy gates passing would only demonstrate usefulness of a RESEARCHER-DEFINED sensor-variation selection rule, not spontaneous goal/escape/foraging invention. No retrospective tuning of seed/map/removal index/command step/head learning rate/scoring objective or thresholds after outcome.

**Stop flags permanently FALSE:** \`food_search_demonstrated=false\`, \`escape_reflex_learned=false\`, \`autonomous_goal_invention_demonstrated=false\`, \`general_intelligence_demonstrated=false\`, \`biological_fly_replication_demonstrated=false\`, \`evolution_authorized=false\`, \`advance_to_evolution=false\`. A narrow \`closed_loop_action_selection_implemented\` flag may be TRUE as a code fact, while \`effective_adaptive_navigation_demonstrated\` must reflect the preregistered science screen.
