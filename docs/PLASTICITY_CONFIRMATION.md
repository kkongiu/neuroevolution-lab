# M1-P2 — Independent longer-sequence confirmation (protocol frozen before run)

## Question

Does the gap in within-task learning AUC between the continually trained network and a fresh network **become more negative with more previous tasks**, when comparing the **same task families**?

This directly separates *negative transfer on a single transition* from *cumulative loss of plasticity*. Secondary: does selective low-activity reset still beat matched random reset on new seeds?

## Locked design

Use unchanged code, architecture and learning operators from M1-P. Change **only**:
- Independent seed sequences **300 through 323** (never seen in M1-P);
- **16 consecutive tasks** instead of 8.

Same four arms, 24-unit hidden layer, learning rate 0.04, 128 updates/task, batch size 2, evaluation size 96, reset 20% at each switch.

Task sequence cycles through four families (line, XOR, circle, stripe) with independently seeded parameters.

### Primary comparison (prespecified)

For each seed and each task, compute `gap = AUC_continual − AUC_fresh`.

To match identical task families and exclude the first-task initialization effect, compare:
- **Early**: average `gap` for tasks **4, 5, 6, 7** (one of each family);
- **Late**: average `gap` for tasks **12, 13, 14, 15** (same families).

For each seed compute `Δ = late − early`. **Negative Δ** suggests deterioration beyond the immediate transfer gap; zero/positive Δ fails to support progressive worsening at this horizon. Report paired mean and a descriptive percentile bootstrap interval across the 24 seed-level differences.

### Secondary comparison

For each seed compare average AUC `low_activity_reset − random_reset` across tasks **8..15**. Report sign counts and paired bootstrap interval.

Report retention/forgetting as a separate measure. Repeatability and modest model size do **not** establish significance outside this task family.

## Failure conditions

- No progressive gap: classify as **negative evidence for progressive loss**, even if the continued network is always worse than fresh.
- Low-activity reset not better than random: classify targeted renewal as **unconfirmed**.
- Retention collapse with reset: report the trade-off, not a pure benefit.

No choices about architecture, learning rate, task set, evaluation points, metrics or seeds may be tuned after seeing this campaign's results. Further confirmation requires new seed ranges and new task families.
