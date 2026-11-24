# FlyLoop R3 — prospective protocol: sensorimotor prediction, not a scripted escape strategy

**24 November 2026 | frozen before the first full R3 experimental run.** Stacked draft branch on FlyLoop R2. R1, R2, M0–M4, sealed M5.6 and main remain unchanged.

## Research question

R1's continual recurrent circuit mainly circles. In R2, externally rewarded net displacement+contact reduced collisions but also reduced exploratory coverage. Could a recurrent organism learn *what its next wall sensors will report* based on its actual action and internal state, without receiving a hand-coded movement plan, global position, route, food scent or a "turn away" action? And does converting that learned predictor's error into local motor plasticity produce control benefits that persist against matched ablations?

**Important epistemic limits:** The self-supervised predictor, the squared-error objective, its motor-error eligibility update, and even the representation of the sensor rays are **researcher supplied**. Better prediction alone is NOT better navigation. Minimizing prediction error can trivially encourage circling or remaining in predictable areas; specifically test for this failure mode. This is not a fly brain, spontaneous curiosity, reward invention or autonomously discovered movement primitive.

## Locked physical and neural setup

- 22×22 continuous 2D arena; point-body radius 0.30; motor inertia, two continuous forward-speed/turn commands, 8 R1 local raw sensor values, 3 short-range wall rays within 3.5 world units, 16 recurrent tanh units, 2 analog motor outputs, 600 frames max, dt=.2, energy depletion, real wall contact/integrity.
- The neural core is **exact R1** (434 seeded initial coefficients). R2's researcher-supplied seeded Gaussian motor perturbations of SD 0.12 are shared on the same live prefix across all neural control arms. Only existing 34 motor-output weights/biases can adapt in a motor-plastic arm; 16 recurrent units and upstream weights remain fixed.
- A **new 66-weight linear sensor prediction head** (3 sensors ×22 coefficients per sensor) is added in arms with a predictor. It predicts the THREE wall-proximity readings at **time t+1**, from the actually observed current three rays, current 16 recurrent states, executed perturbed 2D motor command, and a unit bias. Specifically each prediction \`clip(current_ray + dot(w, [h16, current_ray3, command2,1]), 0, 1)\`; all 66 initial head weights are ZERO, thus the frozen-head control predicts persistence exactly. Head training after the action is simple clipped LMS: \`w += .014 × (actual_next_ray - predicted_ray) × feature\`, each head weight bounded [-2,2]. No future or neighboring ray sensor is available before movement, and no map coordinate can enter the head. The learner is NOT trained to steer left/right; it is trained only to predict observable consequences.
- Predictor true training label is the **actually observed** three rays at t+1 after the body moves. A sham head instead trains with a randomly selected observation of the SAME living organism from **≥6 completed frames earlier** (independent RNG, never future/current/previous frame); before sufficient history, sham head gets no update. The real next-frame sensor is still retained for honest evaluation of sham prediction accuracy.
- Prediction error \`MSE = mean((pre-action prediction - actual next observed three rays)^2)\` is recorded for every actual step in every arm. Naive persistence MSE is computed on the SAME trajectory and same next-ray target (paired within steps); cannot confuse policy-dependent easy worlds with prediction improvements.
- For motor plasticity, **a policy perturbation correlation update** uses scalar \`-MSE\` minus an EMA baseline (0.97 prior +0.03 newest), with motor eligibility leak 0.85, learning rate .0018, and Gaussian motor noise from the same exogenous seed. It updates motor output weights **AFTER** real sensory consequence. As with R2, this update is a researcher-coded, clipping-biased estimator; no self-invented plasticity rule. It may select self-predictable circling instead of exploration. The control objective never directly mentions collisions, covered cells, speed, direction or food.
- Cohort evaluation: **24 entirely new seeds 9700–9723**, disjoint from all earlier R/M FlyLoop studies, 2 physical world families (\`obstacles\` uses R1's ten independently generated rectangles, \`nearfield\` uses R2's separate ten-rectangle generator), 8 exact controllers = **384 continuous lifetimes** ≤600 frames (up to 230,400 actual movement integrations).
- Every seed/scenario has same exogenous world, starting body, initial 434 coefficients and neural perturbation RNG stream for all applicable arms. No cross-lifetime optimizer, seed re-selection, genetic evolution or parameter retuning from results.
- Two complete serialized result files must match byte-for-byte; logs include every pre/post wall observation, 16 hidden units, raw and executed motors, injected perturbations, predicted next sensor readings, actual next sensor readings, true and sham label source times, predictor errors, motor feedback, collisions, exact initial/final weights, body/world state. Save fixed sample sizes including dead lives.

## Eight specified arms

1. \`recurrent_frozen\`: same R2 noisy recurrent controller, NO prediction head training or motor updates.
2. \`r2_progress\`: R2's exact same physical progress/contact reward and motor plasticity, to check whether prediction can improve beyond earlier hand-shaped reward.
3. \`predict_passive\`: train 66 head weights on REAL next sensory observation, but freeze the motor policy.
4. \`predict_motor\`: train same true predictor, and learn motor readout from **actual prediction surprise** (negative MSE).
5. \`predict_sham\`: train sensor head with genuinely OLDER unrelated observations; update the motor with the **actual current prediction MSE**, using same motor noise and rule. Controls the contribution of correctly paired predictive learning.
6. \`predict_frozen\`: frozen zero head = simply "the next sensor will be unchanged", but motor readout learns from actual persistence prediction error. Tests whether a learned head is needed vs trivial direct sensory change.
7. \`leaky_predict_motor\`: same true prediction+motor learning but recurrence ablated, retaining neural leakage and same motor inertia; counts of *active* network parameters differ after ablation.
8. \`inertial_random\`: R1 smooth random motor without predictor/learning or neuron use; genuinely different-policy baseline.

No action picker, path planner, sensory destination map, reward for visits or escape maneuver is introduced. Even the learned head cannot be called a world model in the strong sense (short-horizon wall-proximity predictor only).

## Outcome metrics

On whole life and on frames 300–599 (predeclared late phase):
- average real prediction MSE and same-trajectory persistence MSE, including sham/frozen controls;
- exploratory 24-paired-world-seed 95% bootstrap for MSE changes and for unique 1×1 cells and collisions;
- total physical steps, unique grid cells, traveled distance, pooled wall contacts per meter, trajectory RMS radius (detect circular confinement), surviving fraction, terminal integrity, mean abs angular rate;
- coefficients before/after; number of head and motor updates.
- Head accuracy may improve from training but this is separate from **control efficacy**.

## Sixteen preregistered gates (all for narrow DEV success)

**Integrity (10):**
1. Exactly 384 balanced world×seed×arm rows, and **entire second run byte-identical** (GitHub Actions).
2. Legal original R1 physics, continuous body positions/motor, no wall penetration or resurrection, physical energy/integrity.
3. Identical exogenous 10-rectangle map, initial body pose and 434 initial neural coefficients per world×seed across all eight controls; motor noise identical on the shared live prefix across seven neural arms. `predict_passive` and `predict_sham_passive` must follow EXACTLY identical physical trajectories (their heads never affect the motor).
4. 16 original recurrent-neuron states and original 8 local sensor channels (wall ray, speed, turn, previous collision, energy), no hidden map, food, reward lookup or future senses.
5. Predictor has exactly 66 learnable coefficients, all initially ZERO, and only these + up to 34 original motor coefficients can change in allowed arms. Frozen and random arms retain their coefficients; original 400 nonmotor neural coefficients never update.
6. Predictions are computed from real t sensory values and executed motor signal BEFORE physical World.integrate; labels use actual t+1 readings AFTER physical movement; MSE calculated against unaltered actual next readings on the same real trajectory.
7. Sham training labels come only from same organism's recorded t−6 or OLDER actual previous wall-ray samples, sampled with an **independent RNG**. No fabricated sham labels in initial steps. The two passive predictor controls must have identical actual motor, sensor and physical outcome trajectories.
8. Persistence reference is exactly current three real wall values and evaluated on the same true future observations, regardless of policy; frozen-head predictions equal this reference.
9. Two exogenous map generators have 10 valid nonoverlapping rectangles and at least 20 different maps over 24 world seeds each.
10. All per-step logs, coefficient hashes, reproducibility and explicit false evolution/intelligence stop flags archived; R1/R2 original source unmodified.

**Scientific efficacy (6):**
11. In \`nearfield\`, TRUE passive online predictor's **late-phase** mean MSE ≤ **0.85 × SAME-TRAJECTORY persistence MSE**. If persistence loss is exactly zero, automatic FAIL.
12. In \`nearfield\`, TRUE passive predictor's late-phase mean MSE ≤ **0.85 × `predict_sham_passive` predictor's** late-phase mean real-target MSE on EXACTLY THE SAME TRAJECTORIES. Never treat time frames as independent replicates.
13. In \`nearfield\`, \`predict_motor\` mean unique 1×1 cells ≥ **1.10 × recurrent_frozen**.
14. In \`nearfield\`, \`predict_motor\` pooled collisions/meter ≤ **0.90 × recurrent_frozen**.
15. In \`nearfield\`, \`predict_motor\` mean unique cells ≥ **1.10 × predict_sham**.
16. In \`nearfield\`, \`predict_motor\` mean unique cells ≥ **1.10 × predict_frozen**.

All 16 must pass. A green software workflow is not scientific success. A predictor could succeed while navigating worse, so report them separately. On any failure preserve full negative findings and thresholds. STOP FLAGS permanently false:
\`goal_invention_demonstrated=false\`,
\`self_learned_planning_demonstrated=false\`,
\`successful_obstacle_avoidance_demonstrated=false\`,
\`food_search_demonstrated=false\`,
\`biological_fly_replication_demonstrated=false\`,
\`evolution_authorized=false\`,
\`advance_to_evolution=false\`.
