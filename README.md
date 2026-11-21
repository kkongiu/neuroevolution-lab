# NeuroEvolution Lab

Scientific, CPU-only experiments in adaptive neural systems. **No claim of autonomous will or consciousness.** Python 3.11+, no third-party dependencies.

## M1 — Structural neuroevolution

Two populations start with identical recurrent networks. One can modify numeric weights only; the other can also modify topology between generations. Their held-out performance is compared under identical evaluation budgets.

Run:
~~~bash
python -m unittest discover -s tests -v
python -m neuroevolution --generations 12 --population 12 --out results/m1
~~~

M1 writes a resumable `checkpoint.json`, `history.jsonl`, `summary.json` and `scores.svg`. The [first M1 campaign](reports/M1_campaign_2026-11-20.md) was exploratory, with just three paired seeds.

## M1-P — Continual learning and plasticity

Unlike M1, this experiment **updates the neural weights during each task** using SGD. Four arms solve exactly the same task sequences: fully continual weights; reinitialize from scratch; low-activity unit renewal; random unit renewal. It measures within-task adaptation curves and forgetting across successive tasks.

~~~bash
python -m neuroevolution.plasticity --seeds 8 --tasks 8 --steps 128 --out results/plasticity
~~~

This writes `measurements.csv`, `summary.json`, and `curves.svg`. Run with a new output directory for each campaign. No GitHub write operations or arbitrary code modifications are performed by the agents.

Read [the preregistered plasticity protocol](docs/PLASTICITY_PROTOCOL.md) before interpreting the results.

## M2 — Bounded meta-learning of renewal decisions

A five-weight meta-controller is **evolved** on one set of task sequences, selected on a separate validation set, then tested on completely disjoint seeds. It can decide **0 / half / maximum (20%)** neural unit replacements at task boundaries using training data only. No code generation or arbitrary self-modification is possible.

Its independent test compares six approaches: learned decisions, no reset, fixed 20% low-activity renewal, **random units with the exact same reset counts per task**, a **shuffled schedule with the same total number** of low-activity resets, and a freshly initialized network.

Run the whole predefined M2 protocol:

~~~bash
python -m neuroevolution.meta --out results/m2
~~~

Outputs: meta-training history, per-seed metrics, per-task metrics, self-contained SVG, and paired bootstrap confidence intervals. Read the [M2 frozen protocol](docs/M2_PROTOCOL.md). **A winning average against no-reset alone is not enough**; dose-matched and timing-matched controls are primary.

## M2B — Learn *when* to renew at a fixed total budget

M2 selected **zero resets in 112/112 transitions** ([negative result](reports/M2_adaptive_2026-11-21.md)). That made its dose-matched timing comparisons vacuous. M2B tests a different prespecified hypothesis without changing M2's code or data.

Every dose-matched policy now **must spend exactly 10 unit renewals over seven task transitions**, with a maximum of three renewals at any one transition. Two controllers are independently evolved:

- **Adaptive:** sees current and past task *training* error, hidden-unit activity and the task index.
- **Time-only:** sees only task index.

We test both against uniform, shuffled and random-unit choices using independent meta-train, meta-validation and final-test seeds. The report distinguishes **requested vs forced** actions, because mandatory late-budget consumption is not evidence of learned adaptability.

~~~bash
python -m neuroevolution.meta_budget --out results/m2b
~~~

Read [M2B protocol](docs/M2B_PROTOCOL.md) before interpreting results. Equal reset counts do not imply equal CPU/energy overhead.

## M3 — Unpredictable novelty, returns, and causal state ablation

The world now introduces tasks in an **unpredictable order**, sometimes revisiting an **exact old latent task** with newly sampled examples. The task identity and novelty labels are logged but *not disclosed* to the controller.

The core test pits an evolved state-aware renewal scheduler against an independently evolved **time-only** scheduler, at exactly the same intervention quota. Final tests use both fresh seeds from familiar task families and a **partially shifted** world with previously unseen checker/wedge task families. The same learned controller is evaluated with its state observations ablated to establish whether the signals change decisions and outcomes.

~~~bash
python -m neuroevolution.m3 --out results/m3
~~~

Outputs: an auditable frozen summary, meta-training trace, final per-seed and per-task CSVs, separate measures for novel and returning tasks. See the [M3 protocol](docs/M3_PROTOCOL.md).

## Actions

[Validation workflow](.github/workflows/validation.yml): tests and brief smoke runs on pull requests/pushes.

[Original M1 research campaign](.github/workflows/research-campaign.yml): original structural neuroevolution, manually or weekly.

[Plasticity campaign](.github/workflows/plasticity-campaign.yml): 24 paired seeds, manually started or explicitly triggered with a kickoff marker. Archives generated metrics and SVG for later scientific review.

[M2 campaign](.github/workflows/m2-adaptive-renewal.yml): isolated training/validation/test and archived metrics; manually launched or explicitly triggered by a kickoff marker.

[M2B fixed-budget timing campaign](.github/workflows/m2b-budgeted-timing.yml): manually triggered or launched by an explicit kickoff marker, with new sealed seed sets.

[M3 unpredictable recurrence campaign](.github/workflows/m3-novelty-recurrence.yml): preregistered meta-training and independent regular/shifted final tests. Launch manually or by an explicit kickoff marker.

**Experimental status:** a passing workflow proves that code executed, *not* that a scientific hypothesis succeeded. Preserve negative findings and uncertainty.
