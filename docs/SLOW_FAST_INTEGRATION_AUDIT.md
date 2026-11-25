# Integration audit — 2026-11-25

Reviewed current default-branch source files `neuroevolution/m5_0.py`, `neuroevolution/m5_5a.py`, `tests/test_m5_5a.py`, and README.

## Verified interfaces and constraints
- M5.0 uses a persistent 15x15 ecology with renewable food, 15 local neural inputs, 5 actions (4 movement + reproduction), 12-hidden neural controller, physical birth costs and starvation. This is the natural environment baseline, but it has no documented fast/slow teacher-distillation path.
- M5.5A explicitly replaces the 401-weight neural actor with **programmer-designed** movement/reproduction instincts modulated by eight heritable numbers. Therefore M5.5A is **not** a valid slow neural teacher or an already implemented fast neural circuit. Using it as teacher would answer a different question.
- Existing M5.5A tests check inherited genes and ecological physics, not teacher-to-student distillation.
- README reports M5.4C movement exploration failure and the strong programmed-movement rescue, but these are historical motivation only.

## Integration requirements (before sealed study)
1. Identify the currently active slow/fast controller implementation and its invocation points; if absent, implement the minimal two-path controller without changing M5.0 physical rules.
2. Instrument hazard detection/damage: M5.0 has walls and energy depletion but no documented predators or health damage. A blocked wall move is not automatically a hazard-exposure episode. Either add a separately specified danger scenario on new seeds or relabel the benchmark as fast navigation/obstacle response; do not report nonexistent damage or escape latency.
3. Collect slow teacher action targets from TRAINING worlds only. Freeze slow teacher, train fast student on these targets, freeze fast weights before heldout testing.
4. Use a fixed gate with identical trigger conditions for B/C. Keep slow fallback and all world random streams matched; avoid consuming extra shared RNG in B/C.
5. Add deterministic unit tests for teacher-only gradient provenance, disabled-fast equivalence to baseline, paired world initial state, and event-log schema.
6. Freeze implementation SHA, final seed lists and success thresholds; then run development smoke and sealed 30-world paired campaign.

## Scientific status
INTEGRATION NOT YET VERIFIED. No new test has been executed. Do not interpret the protocol document as a completed benchmark. The next code change should be the minimal controller adapter plus deterministic tests, not a scientific result claim.
