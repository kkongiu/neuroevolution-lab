# FlyLoop M0 — Frozen experimental protocol, 23 Nov 2026

**Status: preregistered before the first complete FlyLoop M0 Actions science run.**

## The actual scientific question

Can continuous sensory-motor interaction in a small 2D ecology
produce a measurable and transferable improvement in survival and
resource gathering **without injecting a complete action plan**?

M0 is deliberately **not** an implementation of a fly brain. It is a
small artificial embodied agent, deliberately omitting M5.40–43
hypothesis committees, privilege-laden symbolic grammar, and explicit
"seek food", "escape hazard" or "go around the wall" scripts.

M0 uses three versions of EXACTLY the same shared-weight,
action-conditional linear policy and the same primitive movement
actions; only the learning/exploration ablation changes.
The inherited sealed M5.6 ecology remains untouched.
This is a NEW experimental harness on a stacked DRAFT PR.
No changes to main, no merge, and no evolution.

## World, sensors, and primitive actuators

- Discrete 13×13 bounded 2D area, persistent from timestep to
  timestep for **480** steps, with a vertical barrier at x=6.
- There are initially **two openings** (y=2,10); at timestep
  **240** these CLOSE and **two other openings** (y=5,8)
  OPEN. The environment changes independently of the agent;
  an agent cannot teleport through solid obstacles.
- Per seed, **16 food locations** and **9 damaging hazards**
  in open cells, seeded before the experiment and unchanged
  within the lifetime; patches regrow after **17 steps**.
- Organism starts at x=2, y=6 with **70 energy and 100
  integrity**, both persistent and bounded in [0,100].
  A time step drains energy 0.25; a collision adds a 0.5
  energy penalty; a food encounter restores up to 17.0
  energy; hazard contact drains 1.0 energy and 4 integrity.
  Death when energy<=0 or integrity<=0.
  There is no hidden stage reward for traversing the
  barrier, returning to food, novelty or surviving.
- Exactly five elementary outputs: move N,E,S,W or WAIT.
  No "flee", "forage", "follow wall" or "climb" action.
- Every candidate direction is represented by 8 real-valued
  local input features: constant, hunger, injury, food in
  adjacent destination, scalar directional smell from food
  within radius 4, obstacle in destination, hazard in
  destination, and whether the action is WAIT. Sensing uses
  only the current world state, never future rewards or the
  private route planner. No absolute x/y coordinates, map
  ID, precomputed path or new-gate schedule supplied to policy.
- Policy is a shared **8-weight linear actor**, scoring each
  candidate action using the SAME eight weights (rotational
  generalization). Critic is a **6-weight linear baseline**
  using current coarse sensor summary. Parameters = 14
  floats, not a claimed biological synaptic implementation.
  All three arms start with identical per-seed initialized
  actor and critic weights.

## Arms and credit assignment

1. **frozen:** identical seeded stochastic softmax
   policy, no actor or critic updates. Exploration is the
   normal stochasticity of softmax, NOT zero exploration.
2. **plastic:** same policy with online actor–critic TD(0):
   delta = (observed physical reward + gamma*Vnext−Vnow)
   and weighted action feature gradient, clipped.
   Physical reward is computed from actual changes in
   energy and integrity, NOT food/wall labels.
3. **plastic_novelty:** same online update plus a pre-coded
   intrinsic novelty bonus on first encounters with an
   observation signature (and on subsequent rare
   observations with inverse-square-root decay). This is
   an **engineered curiosity prior**, not an organically
   discovered goal or a learned curiosity network.
   All arms receive the same raw sensors and action set.
   The novelty arm uses no special extra actuator,
   pathfinding routine or direct food seeking heuristic.
   Its extra internal sensor-signature counts are explicitly
   disclosed and accounted for.

The neural weights adapt within a single lifetime and are
reset between seeds. No genetic algorithm, offspring or
cross-life transfer is run. No open-loop list of correct
moves is available in the code.

## Precommitted run and measurements

Run **32 independent world seeds**: integers **7100–7131**.
For each world seed: same world topology, initial food/hazard
map and initial weights for all three arms; stochastic action
sequences diverge naturally with experience. Exact seeds
and hyperparameters in code, not tuned to any observed
evaluation outcome. The world change always at tick 240.

Per (seed,arm) archive:
- steps alive, time of death, mean energy, terminal energy,
  mean/terminal integrity, food consumed, hazard encounters,
  collision count, unique cells, distinct food locations
  visited, number of successful crossings through x=6 before
  and after gate change, and contact with previous food
  locations again (return events).
- All time/position/action/reward/energy/health traces,
  actual world map seeds and configs, and the final actor
  and critic weights.
- Post-change food, survival and crossing metrics measured
  **only** when an agent survives past the change; report
  dead-before-change proportion, do not assign fabricated
  successful recoveries to dead organisms.

Also run hard-coded QA ablations on separately seeded
sanity cases: frozen actor weights must remain EXACTLY
unchanged, plastic weights should update on a real nonzero
TD event, plastic+zero novelty must be identical to plastic,
all actions and rewards valid, map change traceable,
no wall crossing, deterministic same-run SHA.

### 10 frozen scientific and integrity gates

1. Deterministic exact 32×3=96 complete rows and identical
   bytes/hash on a repeated full run.
2. Every real step has one valid primitive action, legal
   position, physical energy/damage state and no wall
   traversal/teleportation.
3. All three arms share the exact same initial neural
   weights and seed world map; the episode is persistent.
4. Frozen weights unchanged; plastic and novelty perform
   at least one nonzero online weight update in surviving
   episodes.
5. Novelty coefficient 0 reproduces EXACTLY the plastic
   learner trajectory with the same random seeds.
6. Physical ecology switch actually closes original
   barrier openings and opens distinct new ones at tick
   240, with no agent privileged gate-map input.
7. Mean **food consumed**, plastic > frozen by >=10%.
8. Mean **time alive**, plastic > frozen by >=5%.
9. Novelty arm post-switch mean food >= plastic post-switch
   mean food by 10% (unconditional population measure,
   including zero for dead agents).
10. Mean successful post-switch barrier crossings, novelty
    >= plastic by at least 0.5 absolute crossings.

**All ten must pass** for a narrow M0 development screen.
CI correctness success is distinct from science success.
Report all outcomes, negative gates and per-seed effects.
The 32 evaluation worlds are not independent validation
of open-ended biological evolution.

## Evidence boundaries

- The geometry, survival need, hunger feature, food smell,
  learning rule, reward definition and novelty bonus are all
  researcher-coded. They might suffice to make adaptive
  competence easy; document that and measure controls.
- A linear policy does NOT invent new movement primitives,
  abstract goals or structural cognition.
- One new barrier schedule, two fixed openings per phase,
  and only 32 shared synthetic world seeds is NOT ecological
  generality. Even passing this pilot does not establish
  causal emergent intelligence.
- Paired seed differences should be reported, including
  bootstrap intervals over WORLD SEED (never steps).
- The old frozen M5.6 study stays sealed. No retroactive
  claims about the previous M5.42/43 negative results.
- Stop flags remain false:
  \`general_intelligence_demonstrated=false\`,
  \`autonomous_operator_invention_verified=false\`,
  \`independent_ecology_replication_complete=false\`,
  \`evolution_authorized=false\`,
  \`advance_to_evolution=false\`.
