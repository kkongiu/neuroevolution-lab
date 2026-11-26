# R13 original results — 2026-11-26
Actions 45783130093, job 139783198167, 313 tests passed, two complete deterministic 192-life runs identical by cmp. Artifact 15099916022 expires 2027-02-24. R13 screen FAILED (3/5); no thresholds changed. No fast circuit, recurrent weights frozen; learned action-outcome critic only.
## Original test aggregates
| geometry | arm | distance | contact microframes | injury | sum cells | recovery / eligible | survivors | density |
|---|---|---:|---:|---:|---:|---|---:|---:|
| obstacles | full | 1136.937851376 | 113 | 7.0625 | 155 | 10/11 | 24 | .09938977743 |
| obstacles | original | 1071.213969472 | 1156 | 72.25 | 180 | 26/29 | 24 | 1.07914948175 |
| obstacles | delayed | 999.465732886 | 567 | 35.4375 | 188 | 36/42 | 24 | .56730309139 |
| obstacles | fixed | 286.257421455 | 2985 | 186.5625 | 115 | 40/58 | 22 | 10.42767724528 |
| nearfield | full | 1072.875052333 | 650 | 40.625 | 132 | 35/37 | 24 | .60584874127 |
| nearfield | original | 1008.598002326 | 1466 | 91.625 | 207 | 30/65 | 24 | 1.45350277972 |
| nearfield | delayed | 1057.874343051 | 555 | 34.6875 | 156 | 23/28 | 24 | .52463697947 |
| nearfield | fixed | 299.150967682 | 2604 | 162.75 | 104 | 34/46 | 20 | 8.70463505493 |
Primary gates: density vs original PASS, distance PASS, cells FAIL, survival PASS, density vs delayed FAIL. Paired bootstrap density difference full-original [-2.4661575201,+.7191473791], full-delayed [-.4935169230,+.7824557967]. Both span zero. Recovery rates are descriptive, conditional on observed eligible episodes, not matched counterfactuals. Cells are sums of per-life unique grid cells, not global union. Collision counts are contact microframes, not unique collisions. Distance is simulation world units. Technical deterministic replication is not independent statistical replication.
Hypothesis for follow-up: safe repetitive movement may explain reduced cells despite distance gain; not established by aggregates. Inspect per-life trajectories, signed velocity, turn, action occupancy, and sensory-group ablations before changing rewards or advancing to rapid circuit teaching. The archive contains sensor/action/event traces but not absolute trajectories, so exact physical replay would be necessary to inspect spatial paths. No direct training of recurrent sensory afferents occurred. No claim of autonomous body schema or reliable real-time sensory benefit.
