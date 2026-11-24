# FlyLoop R4 — preregistered action-conditioned sensor prediction and motor controllability
**24 November 2026. This protocol was committed BEFORE any full R4 experimental evaluation.**

## Question / honest boundary
R3 demonstrated 1-step *wall-sensor prediction* better than persistence and temporally shuffled sensory labels, but optimizing negative prediction error did not improve movement. R4 tests whether knowing **which sensory consequences actually depend on one's motor commands** helps. It separates the ability to learn a motor-conditioned predictor from the ability to exploit it.

This is an experiment on a **researcher-engineered objective of sensorimotor influence**. Prediction models, model-difference intrinsic reward, update rule and physical sensors are entirely specified by the experimenter. It cannot prove that the organism *spontaneously* invented seeking, fleeing, exploring, controllability or purpose.

## World and fixed existing organism
Reuse **unchanged R1 physics and R2 local motor exploration**: 22×22 continuous world, body radius .3, 16 persistent recurrent tanh neurons with 8 local physical sensor inputs, 2 continuous speed/turn motors with inertia, current wall proximity rays only (front / ±.70 rad), actual speed, angular velocity, prior collision and energy state; max 600 steps dt=.20. No food, predator, route/mapping sensors, absolute world coordinates in neural input, evolution or preprogrammed avoidance. Both arena types are original independent ten-obstacle generators from R1 and R2, no modifications.

Exactly **24 new seeds 9950–9973**, entirely separate from M0–M4/R1–R3. Two environments: \`obstacles\` (R1's existing random rectangles) and \`nearfield\` (R2's existing nearer-spawn rectangles). Seven arms, **2×24×7 = 336 independent persistent lives**, no preselection, 600-step maximum. Same exact seeded world, initial body and **434 original R1 neural coefficients**, and same R2 seeded Gaussian (σ=.12) motor-noise stream on the common live prefix of all neural arms. All original input/recurrent/hidden coefficients stay frozen. Only 34 existing readout/motor coefficients can change in 3 active predictive arms and R2 progress control. No transfer of weights across lives, no weight inheritance/evolution.

## Three prediction heads; controlled action channel
Each head predicts **the three actual wall-proximity readings at t+1** by adding a learned offset to the current three rays, clipped to [0,1]. Head uses the exact current hidden 16-vector, current rays 3-vector, its nominated command 2-vector, six pairwise ray×command interactions, and constant 1: 16+3+2+6+1=**28 features**, three separate readouts ×28=**84 trainable coefficients per head**. All predictor coefficients start at ZERO (meaning persistence). LMS learning rate **.014**, update only AFTER real physical motion with actual next wall-ray observation, with coefficients clipped [-2,2]. Both true and shuffled heads receive **the same true post-action target** (unlike R3's older-label control); only the **motor-command training input** is manipulated.

Heads:
- \`true\`: uses **actually executed motor command** in prediction, training and pre-action error scoring.
- \`blind\`: has all 2 motor-command features and all 6 ray×command interactions forced to zero, leaving state/ray only. Same 84 coefficient slots but fewer active inputs, not a parameter-matched active-complexity comparison.
- \`yoked\`: uses a motor command **previously actually executed by the same living organism, at least SIX completed steps earlier**, randomly selected with an independent RNG. Before six past steps, input (0,0). The logged source tick must satisfy source ≤ t−6. The sham RNG MUST NEVER perturb the real motor noise stream. Labels are always real current post-action target, so the only causal confound being ablated is the mapping from chosen action to actual sensory outcome.
- In the \`passive_probes\` arm, all THREE heads train on **the EXACT same physical trajectory** with frozen motors. Score late true/blind/yoked head MSE on real next-sensor targets, plus a counterfactual **same-trained-true-head** prediction MSE when its command is replaced by the yoked command before the action. This within-life check tests whether the TRUE head genuinely depends usefully on actual motor commands rather than merely extra coefficients or state correlation. No "counterfactual ground truth" is asserted: an action not executed has no observed target in that world.

## Seven locked controller arms
1. \`recurrent_frozen\`: original R1 recurrent policy with same R2 seeded motor perturbations, no head/motor learning.
2. \`r2_progress\`: exact R2 local motor rule rewarded by physical 12-tick net displacement and wall-contact cost (researcher-specified positive-control objective), no prediction heads.
3. \`passive_probes\`: frozen recurrent motor + all three head learners, true/blind/yoked, no motor updates. All head predictions scored against SAME real next sensory observations.
4. \`agency_gain\`: same heads, learns original 34 motor output coefficients from local **action-conditioning advantage** \`g = clip((MSE_blind−MSE_true)/(MSE_blind+0.01),−1,+1)\` computed only AFTER real movement, update via same R2 eligibility-trace correlation with motor perturbations and EMA baseline (LR=.0018, eligibility leak .85, baseline .97). g is a researcher-engineered intrinsic objective, NOT self-invented agency.
5. \`sham_gain\`: same heads/motor update, but advantage substitutes \`MSE_yoked\` for \`MSE_true\`, thus a feedback baseline about older, noncausally paired motor commands.
6. \`prediction_error\`: same true head/motor update as R3, learning from \`-MSE_true\`, to compare maximized predictability vs action-specific prediction improvement; researcher objective.
7. \`inertial_random\`: R1 correlated random motor with same physical actuators; no head or motor learning, no wall sensing.

All prediction/motor updates occur AFTER the executed motor has produced observed real body/wall sensor changes. No action ranking or planning routine is provided.

## Metrics
Archive all 336 raw trajectories, per-step 8 sensors, 16 hidden states, raw/perturbed/executed motor commands, selected yoked command and its old source tick, real post-action 3-ray observation, all THREE pre-action head predictions (and same-true-head command-swap prediction), all four raw MSEs, true and yoked training features, motor feedback, actual legal displacement/collision, full initial/final 434 core coefficients + three 84-coefficient heads. Full deterministic rerun must be **byte-identical**.

Primary late-phase predictor metric: pooled MSE on steps **300–599**, on the SAME \`passive_probes\` physical trajectories. Primary behavioral outcome: unique 1×1 cells and pooled collisions per actual distance in \`nearfield\`, averaged over 24 independent world seeds. Supplementary: steps lived, total distance, mean signed/absolute curvature, RMS spatial trajectory radius, terminal energy/health, original \`obstacles\` generalization, exploratory paired-world bootstrap95. Do NOT mistake frames for independent worlds. Do NOT interpret lower collision rates without coverage/spatial spread.

## 18 immutable gates (11 integrity, 7 science) — all must pass for narrow DEV success
**Integrity:**
1. Exactly 336 balanced lives, complete byte-identical repeat in CI.
2. Legal unchanged original R1 position/heading/actuator physics, no wall penetration or post-death movement.
3. Identical seed/scenario world, initial pose and original 434 weights across all seven arms; neural motor noise same on common live prefix.
4. Exactly 16 recurrent hidden neurons, 8 true LOCAL sensor channels, no map, food, target vector or future input.
5. Three heads each EXACTLY 84 weight positions initially zero; only head weights and 34 existing motor output weights can change in permitted arms; original 400 nonmotor coefficients never change.
6. All head predictions computed PRE movement and trained POST movement on actual next wall rays; metrics on actual same trajectory and no synthetic future labels.
7. TRUE head receives actual executed 2D motor command, BLIND head receives zero command and interactions; independently audited by reconstructing pre-action predictions/feature matrices from logs.
8. YOKED head uses same animal's actually executed historical motor command at tick ≤ t−6, independent PRNG, no future or fabricated history at early steps.
9. Passive head control has original frozen motor and exact physically matched current/next sensors for all three heads; the command-swapped evaluation uses the same trained TRUE head *before its current update*.
10. Two exogenous ten-rectangle map generators, ≥20 distinct valid maps out of 24 seeds EACH, no overlap spawn or one another.
11. Complete raw negative and positive outcomes, two exact replicas, intact R1/R2/R3 and M0–M4/main, all intelligence/food/evolution stop flags false.

**Science:**
12. \`passive_probes\` true head **late MSE ≤ 0.85 × motor-BLIND head late MSE**, measured on same physical actions and targets.
13. \`passive_probes\` true head late MSE ≤ **0.85 × YOKED head** late MSE, same physical actions/targets.
14. Same trained TRUE head's late MSE with actual command ≤ **0.85 × with HISTORIC YOKED command**, same actual future target (command specificity, not counterfactual causal proof).
15. \`nearfield\` \`agency_gain\` mean UNIQUE 1×1 cells ≥ **1.10 × recurrent_frozen**.
16. \`nearfield\` \`agency_gain\` pooled collisions per actual meter ≤ **0.90 × recurrent_frozen**.
17. \`nearfield\` \`agency_gain\` mean unique cells ≥ **1.10 × sham_gain**.
18. \`nearfield\` \`agency_gain\` mean unique cells ≥ **1.10 × prediction_error**.

All 18 are required; CI green does not imply science success. No post-hoc gate/seed/sensor/learning parameter tuning. If prediction gains but motor does not, explicitly report limited predictive skill without motor competence. Even all gates passing would NOT demonstrate autonomous goal invention, neural evolution or actual fly behavior.

**Stop flags permanently false:** \`food_search_demonstrated=false\`, \`escape_reflex_learned=false\`, \`autonomous_goal_invention_demonstrated=false\`, \`general_intelligence_demonstrated=false\`, \`biological_fly_replication_demonstrated=false\`, \`evolution_authorized=false\`, \`advance_to_evolution=false\`.
