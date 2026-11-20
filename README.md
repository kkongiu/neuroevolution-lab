# NeuroEvolution Lab

**M1: Does mutable neural topology improve adaptation over weight-only evolution?**

Small, reproducible, CPU-only experiment. Two populations start from **identical genomes** and are evaluated under the same changing environments. One population mutates only numeric weights; the other can also mutate connections and active hidden units.

This is a research instrument, **not** a demonstration of autonomy, consciousness, open-ended intelligence, or self-generated interests.

## Start

Requires Python **3.11+** and **no third-party dependencies**.

~~~bash
python -m unittest discover -s tests -v
python -m neuroevolution --generations 12 --population 12 --episodes 3 --steps 120 --out results/demo
~~~

Read **results/demo/summary.json**, **history.jsonl** and **scores.svg**. Every generation creates a complete **checkpoint.json** (genomes, random generator state, config and history).

Continue a campaign with an increased total number of generations:

~~~bash
python -m neuroevolution --resume results/demo/checkpoint.json --generations 25
~~~

To change other settings, start a new campaign. Fixed and mutable populations are never mixed. The holdout environment seeds are **not used for selection**.

## Environment

A recurrent neural controller sees a binary cue plus previous cue, action and reward. A hidden regime changes every N steps, reversing which action is rewarded. An agent can benefit from using memory to infer the regime, but the regime is never given as an observation.

We log training accuracy, **held-out accuracy**, accuracy immediately after regime reversals, and active connections. Genetic mutations are constrained to numeric weights and topology bits; **there is no executable-code generation, self-execution, or network access**.

## Automation

- **Validation** runs tests and a small campaign on pushes and pull requests.
- **Research campaign** can be launched manually and runs weekly with an identifiable seed. Results are retained as GitHub Actions artifacts for a limited time; download checkpoints you want to keep.

See [experimental protocol](docs/PROTOCOL.md) and [roadmap](docs/ROADMAP.md).

**Important:** A single campaign does not establish a causal advantage. Replicate over independent seeds and report failures as well as successes.
