# M5.6 — Confirmatory preregistration safety gate (2026-11-22)

The **252-world developmental pilot** is successful technically but not informative enough for confirmatory survival inference: pressure 0 gave universal survival, pressure 3 near-total extinction, and pressure 1 gave near-universal survival. Reusing the current simulator unchanged on 32 held-out worlds would likely be a weak and scientifically inappropriate test.

**Action taken now:** Added `neuroevolution/m5_6_prereg_gate.py` to reserve distinct, deterministic source (161000–161031), holdout (162000–162031), and independent replication (163000–163031) world IDs, each with 32 worlds, all separate from development seed IDs 1000–1099. All cohorts and intended controls are serialized in a hashed manifest. These seed IDs have not been evaluated in M5.6. **The manifest is provisional; not an executable final freeze.**

**Predeclared primary endpoint:** between-policy *paired world-level survival difference* for curiosity versus no curiosity under the same initial predators, with favorable scientific criterion improvement at least 10 percentage points AND paired bootstrap 95% lower bound above zero (20,000 bootstrap samples) on the final held-out cohort; independent replication required before a positive claim. Do not treat organism descendants/ticks as independent units.

**Required factorial arms (unchanged from prospective protocol):** novelty+RNN+predators, zero-novelty+RNN+predators, memory-reset+novelty+predators, novelty+RNN+zero predators, shuffled reward+RNN+predators, and programmed exploration baseline+predators. The development-only `energy_gated_scripted` reference is clearly an engineered policy, not learned survival instinct.

**Nonnegotiable gate before opening holdout:** calibrate an informative world difficulty using *development seeds only*; implement all six preregistered controls; match source training, reward budgets and weights; write and freeze the complete numerical source-weight bank; create the retention/transfer test with learning/novelty disabled; pin simulation implementation, horizon, settings and seed/hash manifest in a separately reviewable commit; pass CI. The gate deliberately raises an error until these conditions are met. Do not bypass the gate by manually passing `True` without evidence.

**Scientific honesty:** The current code learns movements using one-step policy-gradient updates, while reproduction is still a scripted baseline. It has not implemented validated predictive learning-progress curiosity. Full autonomous, internal hunger-versus-curiosity arbitration is unproven. No final cohort has been executed.

This safeguards the scientific validity of the test the user authorized: complete the preconditions, then open the sealed evaluation; do not use final-world outcomes to tune the system.
