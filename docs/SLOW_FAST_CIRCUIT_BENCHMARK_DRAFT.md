# Slow/fast circuit benchmark — preregistration draft (2026-11-25)

Status: PROTOCOL ONLY. No experimental outcomes have been generated. Freeze implementation, code revision, seeds, and analysis before executing sealed tests.

## Question
Does a fast action pathway distilled from a slow deliberative controller improve survival and hazard response over the slow controller alone and over a capacity-matched untrained fast pathway?

## Arms
A: slow controller only.
B: same slow controller + randomly initialized, frozen fast controller.
C: same slow controller + fast controller trained ONLY to imitate slow-controller decisions on training-state observations (teacher distillation). No direct reward, damage, or outcome-gradient updates to fast weights.

Match sensory inputs, movement legality, energy accounting, baseline slow weights, starting conditions and compute budget. Define a fixed hazard-gating rule and a fixed fallback to slow actions. Log whether each action came from fast or slow. Distillation targets must be computed from slow decisions, never labels derived from direct collision outcomes. The slow controller may itself learn from its own training experiences, but all three arms must share its frozen checkpoint at evaluation.

## Design
Use 30 independent, sealed held-out environment seeds; replay exactly the same seeds for A/B/C. Separate training/validation/test seed namespaces; no tuning after test inspection. Include ordinary and shifted environments as independently reported strata. Predefine hazard frequency and scenario generator. Do not call the experiment a test of fast escape unless hazards, danger sensors, damage and timing are actually implemented and validated in code.

## Primary outcomes
1. Survival: fraction alive at a prespecified fixed tick horizon; count deaths and censoring.
2. Hazard-response latency: simulation steps between first detectable hazard and first valid evasive action; specify an upper bound for non-response and report its fraction.
3. Damage per exposure: total health/energy damage attributable to hazards divided by number of independently defined hazard exposures.

## Secondary outcomes
Food units collected per lifetime; food per 100 live ticks; actual wall collisions; attempted illegal moves; distinct cells visited; repeated-cell fraction; energy at end; number and share of fast-path interventions; fast-vs-teacher agreement on an independent observation set; per-action decision compute (instrumented deterministic operation count) and measured wall-clock latency (environment/hardware documented). Never equate simulation ticks with CPU latency.

## Event log schema (JSONL)
One record per action:
run_id, protocol_version, git_sha, arm, world_seed, organism_id, tick, x, y, energy_before, energy_after, food_before, food_after, hazard_detected, hazard_id, hazard_distance, damage, collision, illegal_attempt, slow_action, fast_action, selected_action, action_source, fast_gate, fast_confidence, decision_ops, decision_ns, alive_after.
Unavailable fields must be null, not fabricated. Include a separate hazard-exposure event stream with first_detection_tick, first_evasion_tick, and exposure_outcome; avoid treating correlated frames as independent exposures.

## Analysis
Compute paired per-world C-A and C-B differences, report all 30 paired observations, mean/median effects, paired bootstrap 95% uncertainty intervals (resampling WORLD seeds, not frames), and exact number of wins/ties/losses. For multiple organisms per world, aggregate to world first. No cherry-picked seeds or post-hoc threshold changes. An advantage requires improved hazard latency AND non-inferior survival/damage, with thresholds and alpha specified before final test. If hazards are absent, explicitly mark primary hazard endpoints NOT TESTABLE and do not claim a positive result.

## Implementation acceptance gates
- Inspect current simulator and controller interfaces before integrating.
- Verify slow-only matches baseline deterministically.
- Verify fast weights are updated exclusively via teacher-target distillation; fail tests if hazard/damage rewards update them.
- Verify fast-path ablation and random-fast arm differ only in intended pathway.
- Verify paired seeds generate identical world initial conditions.
- Verify all logs, artifacts, analysis code, environment configuration, and version hashes are reproducible.
- Run unit tests and small development smoke before any sealed evaluation.
- Freeze final protocol + code and publish negative outcomes without retuning.

This is a new experiment; historical M5 findings are motivation, not evidence for its outcome.
