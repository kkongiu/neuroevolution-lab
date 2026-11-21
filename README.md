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

## Actions

[Validation workflow](.github/workflows/validation.yml): tests and brief smoke runs on pull requests/pushes.

[Original M1 research campaign](.github/workflows/research-campaign.yml): original structural neuroevolution, manually or weekly.

[Plasticity campaign](.github/workflows/plasticity-campaign.yml): 24 paired seeds, manually started or explicitly triggered with a kickoff marker. Archives generated metrics and SVG for later scientific review.

[M2 campaign](.github/workflows/m2-adaptive-renewal.yml): isolated training/validation/test and archived metrics; manually launched or explicitly triggered by a kickoff marker.

**Experimental status:** a passing workflow proves that code executed, *not* that a scientific hypothesis succeeded. Preserve negative findings and uncertainty.
