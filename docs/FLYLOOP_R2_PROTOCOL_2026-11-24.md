# FlyLoop R2 — prospectively frozen physical-feedback motor plasticity protocol
**24 November 2026. This protocol precedes the first complete R2 science evaluation.**

## Research question
R1's 16-neuron recurrent loop produces continuous movement, but often turns in tight circles. The wall-blind controller collided less because it moved in an even smaller area. Thus collisions alone are a misleading outcome.

**Does online local motor plasticity, driven by real physical *displacement over time* and actual wall contact, improve broad exploration beyond the exact same recurrent weights with no learning, a temporally decoupled feedback control, and a nonrecurrent plastic control?**

The only biological inspiration is a continuously active embodied circuit. The **physical-progress objective and motor perturbation learning rule are explicitly researcher designed**. Even if R2 succeeds, nothing here proves the spontaneous invention of goals, natural fly motor learning, intelligence, food foraging, flee reflex or evolution.

## Fixed architecture and environment
- R1 is reused *unchanged*: 22×22 continuous 2D world, radius 0.3, motor inertia, walls, exact 8 local sensors (constant, three wall rays, speed, turn, last collision, current energy deficit), 16 tanh hidden units, 2 analog motors, 600 simulation frames with dt=0.2.
- Exactly R1's original 434 initial real coefficients (input 16×8 + recurrence 16×16 + output 2×16 + two biases), deterministically seeded per world. Full recurrence at every step except leaky ablation. **Only 34 original output-motor coefficients (2×16+2 biases) may change during one lifetime**. All input/recurrent synapses and 16 hidden biases remain strictly fixed. No structural rewiring, additional neurons, map, route planner, explicit turn-away macro, food or evolution.
- **24 NEW world seeds 9400–9423**, disjoint from R1 and M0–M4 seeds. Two scenes:
  1. \`obstacles\`: original R1 world generator, 10 rectangles, with new seeds.
  2. \`nearfield\`: a separate exogenous deterministic 10-rectangle generator, placing obstacles nearer the spawn but never overlapping body or one another; all controllers in a world receive the same rectangles.
- Identical initial body and neural matrices per world/seed across **six** controller arms; the same physics and actuators. 24×2×6 = **288 persistent lives**, up to 600 frames each. Full rerun byte-for-byte.

## Online learning and controls (parameters frozen BEFORE outcomes)
- Neural commands use R1's recurrent \`Circuit.forward\`, then an **additive Gaussian exploratory perturbation** of both continuous motor commands with fixed SD=0.12, independently seeded from the same world seed; commands clipped to [-1,1]. Every neural learning arm, including frozen, receives *the same exogenous motor perturbation stream per world seed*. This isolates learned weights as well as possible, though trajectories and sensor states diverge as learning changes motion. Clipping introduces bias into the estimated gradient; this is a proof-of-concept local policy perturbation, **not** an unbiased REINFORCE estimator.
- A homeostatic physical feedback scalar is computed **AFTER** the action:
  \`progress = distance(body[t+1], body[max(0,t−11)]) / (1.2×0.2×min(12,t+1))\`.
  It is net physical displacement across an at-most-12-step window normalized by the maximum linear travel possible. It does **NOT** encode absolute destination or exploration cells; a sustained straight segment can be rewarded even in uninteresting empty space. Contact incurs a fixed -1.2. \`real_feedback = progress - 1.2×collision\`. These are researcher-chosen objectives, not spontaneous desires.
- Exponentially decaying scalar reward baseline (0.97 old +0.03 latest actual feedback) and **two 17-coefficient eligibility traces** using the actually drawn exploration perturbations, last 16 neural activations and a unit bias feature. Eligibility trace leak=0.85. Each live motor weight increments by \`learning_rate × clipped(feedback − EMA baseline, ±2) × eligibility\`, with fixed learning rate 0.0018. All updated motor parameters clipped to [-3,3] for numerical safety; never change input/recurrent/hidden parameters. Plasticity occurs only after physical action.
- This rule is a researcher-coded stochastic local correlation update. It does NOT discover the learning rule, fitness objective or sensing repertoire.

**Six paired arms:**
1. \`recurrent_frozen\`: original recurrent controller, same additive exploration perturbations, output weights fixed.
2. \`recurrent_plastic\`: recurrent + correct actual progress/contact feedback and eligibility learning.
3. \`recurrent_sham\`: recurrent + identical learner but **training feedback taken from the same organism's real history at least FIVE frames older** (with separate RNG); the most recent feedback is not allowed. Older values may remain autocorrelated with current physical state, so this is a *weak* causal decoupling, not random oracle information.
4. \`collision_only\`: recurrent + same learner, training feedback consists ONLY of contemporaneous -1.2×actual collision; tests whether engineering net progress contributes beyond avoidance shaping.
5. \`leaky_plastic\`: R1 nonrecurrent leaky neural state + identical motor plasticity and progress/contact feedback; tests role of recurrent connections, but active hidden parameter count differs because recurrence is ablated.
6. \`inertial_random\`: original R1 correlated random commands and motor physics, no neural training or map access; no added perturbation, labeled a *different-policy* baseline.

All six initialize from the same 434 coefficients; only three plastic/one sham/one leaky arms update motor readout. No intergeneration inheritance; reset between seeds.

## Measurements and 13 preregistered DEV gates
Per life archive full per-frame neural 16-vector, 8 sensor vector, raw and perturbed motor commands, exact injected exploration noise, before/after body pose/heading/speed/turn, collision, physical reward and training reward with source timestep, 17×2 eligibility traces, motor weights/updates, plus initial/final 434 coefficients, original map and actual physics. Report pooled collisions/meter, unique 1×1 cells reached, total meters, net displacement, time alive, terminal integrity, mean turning, path efficiency and mean trajectory radius around its centroid (compact circling indicator). Paired seed bootstrap over *independent worlds*, not action frames.

**Integrity gates:**
1. Exactly 288 balanced rows, two full output files byte-identical (Actions).
2. Exact original R1 physical motor dynamics, legal steps, no penetration of inflated obstacles, no post-death actions.
3. Same initial environment/pose/434 neural coefficients across six arms per scene×seed.
4. Correct continuous recurrent/leaky equations on raw 8 input channels; no map, global position, next-step lookahead or suggested move in the 16-neuron input.
5. Only 34 motor output coefficients ever change, hidden/input/recurrent/bias parameters unchanged; frozen/random arms never update.
6. Correct real physical feedback computed only AFTER action, at-most-12 step odometric net displacement and true contact, no food/hidden task reward.
7. Same independently seeded Gaussian perturbation sequence across five neural arms; sham feedback source tick ≤ current tick−5, from the same life, with separate random stream.
8. Exogenous obstacles have valid geometry and identical paired layouts; >=20 unique maps in each of two scenes, independent of motor trajectories.
9. Full frame traces, weights, objective coefficients, failed results and all evolution/intelligence claims marked false.

**Prospective efficacy gates (all required, may FAIL):**
10. In nearfield, recurrent plastic **mean unique cells** ≥ frozen recurrent ×1.10.
11. In nearfield, recurrent plastic **pooled collisions per actual meter** ≤ frozen recurrent ×0.90.
12. In nearfield, recurrent plastic **mean unique cells** ≥ time-shuffled-feedback controller ×1.10.
13. In nearfield, recurrent plastic **mean unique cells** ≥ collision-only controller ×1.10.

All 13 must pass for narrow R2 development-science success. A software test/CI success is not a scientific success; negative results must be retained. *Post-hoc descriptive* differences in the ordinary-obstacles scenario, against leaky_plastic, and trajectory radius inform next hypothesis but do not change the gates. No seed/threshold selection after outcome. A positive result is **conditional evidence for this particular physical-progress reward and eligibility update**, not for autonomous goal invention or real fly-like navigation.

Stop flags always FALSE:
\`food_search_demonstrated=false\`,
\`avoidance_reflex_learned=false\`,
\`unprogrammed_objective_emerged=false\`,
\`general_intelligence_demonstrated=false\`,
\`biological_fly_replication_demonstrated=false\`,
\`evolution_authorized=false\`,
\`advance_to_evolution=false\`.
