"""M4.3 — Causal delayed cue recall with zero locomotion confound.

Eight genetic replicates of near-parameter-matched RNN and stateless FF
networks. Every agent reaches the very same fork observation; actions during
the cue and delay are ignored and cannot encode the target in the world.
No weight change within an agent's lifetime, no file/network capabilities.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from .m4_1 import hierarchical_bootstrap
from .m4_2 import (
    Controller, Settings as PreviousSettings, TrialWorld, make_pair,
    observation as corridor_observation, fresh_genome, mutate,
    genome_size, direction, simulate as free_corridor_trial,
)

CUE_TICKS = 2
MAIN_DELAYS = (2, 4, 6)
SHIFTED_DELAYS = (8, 9, 10)
VALIDATION_DELAYS = (2, 4, 6)
ARMS = (
    "rnn", "ff", "rnn_reset", "rnn_flipped_cue",
    "rnn_no_cue", "random_binary", "scripted_one_bit",
)
CONTRASTS = (
    ("rnn", "ff", "primary_rnn_vs_ff"),
    ("rnn", "rnn_reset", "reset_effect"),
    ("rnn", "rnn_flipped_cue", "cue_flip_effect"),
    ("rnn", "rnn_no_cue", "no_cue_effect"),
    ("rnn", "random_binary", "vs_random"),
    ("rnn", "scripted_one_bit", "vs_scripted"),
)


@dataclass(frozen=True)
class Settings(PreviousSettings):
    train_start: int = 60000
    train_worlds: int = 8
    validation_start: int = 61000
    validation_worlds: int = 8
    test_start: int = 62000
    test_worlds: int = 64
    shifted_start: int = 63000
    shifted_worlds: int = 32
    genetic_seed_start: int = 90000
    genetic_runs: int = 8
    hidden: int = 8
    ff_hidden: int = 11
    population: int = 14
    generations: int = 8
    elites: int = 2
    immigrants: int = 2
    cue_ticks: int = CUE_TICKS
    horizon: int = 20
    guided_start: int = 64000
    guided_worlds: int = 16

    def validate(self) -> None:
        super().validate()
        if self.cue_ticks != CUE_TICKS:
            raise ValueError("The preregistered cue lasts exactly two observations")
        if self.genetic_runs < 2 or self.population <= self.elites + self.immigrants:
            raise ValueError("Independent genetic starts and viable population required")
        ranges = (
            (self.train_start, self.train_worlds * self.generations),
            (self.validation_start, self.validation_worlds),
            (self.test_start, self.test_worlds),
            (self.shifted_start, self.shifted_worlds),
            (self.guided_start, self.guided_worlds),
        )
        for i, (start, size) in enumerate(ranges):
            if start < 0 or size < 1:
                raise ValueError("Invalid world seed partition")
            group = set(range(start, start + size))
            for other_start, other_size in ranges[i + 1:]:
                if group.intersection(range(other_start, other_start + other_size)):
                    raise ValueError("Overlapping world training/final seed splits")
        if (genome_size("rnn", self) != 260 or
                genome_size("ff", self) != 268):
            # Prevent a silently changed task/parameterization in final campaign.
            if self.hidden == 8 and self.ff_hidden == 11:
                raise AssertionError("Frozen model parameter count changed")


def curriculum_delays(generation: int) -> tuple[int, int, int]:
    if generation < 0:
        raise ValueError("Generation cannot be negative")
    if generation <= 1:
        return (0, 1, 2)
    if generation <= 3:
        return (1, 2, 4)
    return (2, 4, 6)


def selected_delays(seed: int, shifted: bool) -> int:
    return (SHIFTED_DELAYS if shifted else MAIN_DELAYS)[seed % 3]


def fork_observation(world: TrialWorld, *, show_cue: bool = False,
                     flip_cue: bool = False,
                     no_cue: bool = False) -> tuple[float, ...]:
    """Binary-only fork: block the approach action at READOUT in both worlds."""
    # Physical energy is still depleted in guided trials. The pure bench has no
    # energy dynamics, and both tests hide any elapsed-time/energy signal.
    initial = corridor_observation(
        world, world.fork, CUE_TICKS if not show_cue else 0,
        24., 0, 0, Settings(),
        flip_cue=flip_cue, no_cue=no_cue,
    )
    blocked = list(initial[:4])
    branch_actions = {direction(world.fork, gate) for gate in world.gates}
    if len(branch_actions) != 2:
        raise AssertionError("Need exactly two separate branch actions")
    for action in range(4):
        blocked[action] = 0. if action in branch_actions else 1.
    values = tuple(blocked) + initial[4:]
    if set(i for i in range(4) if values[i] == 0.) != branch_actions:
        raise AssertionError("Readout leaks an additional motor branch")
    return values


def neutral_observation() -> tuple[float, ...]:
    # All samples in an episode (and both opposite targets) get exactly this
    # zero-information wait input. Constant energy channel prevents clock leaks.
    return (0.,) * 12 + (1., 0., 0.) + (0.,) * 4


def cue_observation(world: TrialWorld, *, flip: bool = False,
                    blank: bool = False) -> tuple[float, ...]:
    return corridor_observation(
        world, world.start, 0, 24., 0, 0, Settings(),
        flip_cue=flip, no_cue=blank,
    )


def inputs_for_trial(world: TrialWorld, delay: int, *, flipped: bool = False,
                     no_cue: bool = False) -> tuple[tuple[float, ...], ...]:
    if delay < 0:
        raise ValueError("Delay must be nonnegative")
    cue = cue_observation(world, flip=flipped, blank=no_cue)
    fork = fork_observation(
        world, show_cue=(delay == 0),
        flip_cue=flipped, no_cue=no_cue,
    )
    # Actions at all earlier stages are discarded; time, position, energy and
    # final observation do not depend on those actions.
    return (cue, cue) + (neutral_observation(),) * delay + (fork,)


def _logits(controller: Controller) -> list[float]:
    n = controller.n
    stride = 19 + (n if controller.architecture == "rnn" else 0) + 1
    offset = n * stride
    return [
        sum(w * h for w, h in zip(
            controller.genome[
                offset + action * (n + 1):offset + action * (n + 1) + n
            ], controller.hidden
        )) + controller.genome[offset + action * (n + 1) + n]
        for action in range(4)
    ]


def run_trial(world: TrialWorld, settings: Settings, arm: str,
              genome: tuple[float, ...] | None, delay: int,
              *, save_sequence: bool = False) -> dict:
    if arm not in ARMS:
        raise ValueError(arm)
    input_seq = inputs_for_trial(
        world, delay,
        flipped=arm == "rnn_flipped_cue",
        no_cue=arm == "rnn_no_cue",
    )
    legal = sorted(direction(world.fork, gate) for gate in world.gates)
    correct = world.target_direction
    other = next(a for a in legal if a != correct)
    sequence_actions: list[int] = []
    if arm == "scripted_one_bit":
        # Script copies only the observed cue, never the hidden true reward.
        memory_bit = input_seq[0][15:].index(1.)
        chosen = memory_bit
        if chosen not in legal:
            raise AssertionError("Scripted cue points outside legal branch set")
        soft_preference = float(chosen == correct)
    elif arm == "random_binary":
        # Same random draw for both opposite goals in a paired world seed:
        # exactly one correct decision over the paired target sides.
        chosen = random.Random(world.seed * 1117 + 23).choice(legal)
        soft_preference = 0.5
    else:
        if genome is None:
            raise ValueError("A neural arm needs a frozen genome")
        architecture = "ff" if arm == "ff" else "rnn"
        controller = Controller(genome, architecture, settings)
        for obs in input_seq[:-1]:
            sequence_actions.append(controller.choose(obs))
        if arm == "rnn_reset":
            controller.reset_memory()
        chosen = controller.choose(input_seq[-1])
        logits = _logits(controller)
        margin = min(25., max(-25., logits[correct] - logits[other]))
        soft_preference = 1. / (1. + math.exp(-margin))
    success = int(chosen == correct)
    return {
        "world_seed": world.seed,
        "side": world.side,
        "delay": delay,
        "arm": arm,
        "success": success,
        "choice": chosen,
        "target_direction": correct,
        "fork_reached": 1,  # Pure recall has no motor control.
        "cue_at_readout": int(delay == 0 and arm != "rnn_no_cue"),
        "fitness": 0.85 * success + 0.15 * soft_preference,
        "soft_correct_preference": soft_preference,
        "predecision_actions": sequence_actions if save_sequence else [],
        "decision_observation": list(input_seq[-1]) if save_sequence else [],
    }


def evolve_one(settings: Settings, genetic_seed: int,
               architecture: str) -> tuple[tuple[float, ...], list[dict], dict]:
    if architecture not in ("rnn", "ff"):
        raise ValueError(architecture)
    rng = random.Random(genetic_seed)
    population = [
        fresh_genome(rng, settings, architecture)
        for _ in range(settings.population)
    ]
    validation_pairs = [
        w for seed in range(settings.validation_start,
                           settings.validation_start + settings.validation_worlds)
        for w in make_pair(seed, settings)
    ]
    validation_delays = VALIDATION_DELAYS
    leaders = []
    history = []
    evaluated = 0
    for generation in range(settings.generations):
        delays = curriculum_delays(generation)
        train_pairs = [
            w for seed in range(
                settings.train_start + generation * settings.train_worlds,
                settings.train_start + (generation + 1) * settings.train_worlds
            ) for w in make_pair(seed, settings)
        ]
        scored = []
        for genome in population:
            trials = [
                run_trial(world, settings, architecture, genome, d)
                for world in train_pairs for d in delays
            ]
            evaluated += len(trials)
            scored.append((
                statistics.mean(x["fitness"] for x in trials),
                genome,
                statistics.mean(x["success"] for x in trials),
            ))
        scored.sort(key=lambda t: t[0], reverse=True)
        leader = scored[0]
        valid = [
            run_trial(world, settings, architecture, leader[1], d)
            for world in validation_pairs for d in validation_delays
        ]
        valid_fit = statistics.mean(x["fitness"] for x in valid)
        valid_success = statistics.mean(x["success"] for x in valid)
        leaders.append((valid_fit, generation, leader[1]))
        history.append({
            "genetic_seed": genetic_seed, "architecture": architecture,
            "generation": generation,
            "curriculum_delays": ",".join(map(str, delays)),
            "train_fitness_best": leader[0],
            "train_fitness_population_mean":
                statistics.mean(x[0] for x in scored),
            "train_success_best": leader[2],
            "validation_fitness": valid_fit,
            "validation_success": valid_success,
            "cumulative_training_trials": evaluated,
        })
        print(f"M4.3 evolution gene={genetic_seed} arch={architecture}"
              f" gen={generation} train_success={leader[2]:.3f}"
              f" validation_success={valid_success:.3f}", flush=True)
        elites = [x[1] for x in scored[:settings.elites]]
        population = list(elites)
        while len(population) < settings.population - settings.immigrants:
            population.append(mutate(rng.choice(elites), rng, settings))
        while len(population) < settings.population:
            population.append(fresh_genome(rng, settings, architecture))
    selected_fitness, selected_gen, chosen = max(
        leaders, key=lambda t: (t[0], -t[1])
    )
    expected = settings.generations * settings.population * (
        settings.train_worlds * 2 * 3
    )
    if evaluated != expected:
        raise AssertionError("Training budget violated")
    return chosen, history, {
        "genetic_seed": genetic_seed,
        "architecture": architecture,
        "validation_selected_generation": selected_gen,
        "validation_selected_fitness": selected_fitness,
        "genome_parameters": len(chosen),
        "candidate_training_trials": evaluated,
        "validation_trials":
            settings.generations * settings.validation_worlds * 2 * 3,
    }


def guided_corridor_trial(world: TrialWorld, settings: Settings,
                          arm: str, genome: tuple[float, ...]) -> dict:
    """External scripted movement to fork; neural actions before fork discarded.

    This is separately reported and not included in the pure delayed recall.
    """
    controller = Controller(genome, "ff" if arm == "ff" else "rnn", settings)
    position = world.start
    for step in range(world.stem_length):
        senses = corridor_observation(
            world, position, step, 24. - step, 0, 0, settings
        )
        controller.choose(senses)  # motor choice ignored under scaffold
        dx, dy = ((0, -1), (1, 0), (0, 1), (-1, 0))[world.forward]
        nxt = (position[0] + dx, position[1] + dy)
        if nxt in world.walls:
            raise AssertionError("Scaffold entered wall")
        position = nxt
    if position != world.fork:
        raise AssertionError("Scripted motor did not reach fork")
    final = fork_observation(world)
    choice = controller.choose(final)
    return {"success": int(choice == world.target_direction),
            "fork_reached": 1, "choice": choice}


def summarize(records: list[dict], genetic_seeds: list[int],
              world_seeds: list[int], split: str) -> dict:
    by_key = {
        (r["genetic_seed"], r["world_seed"], r["side"], r["arm"]): r
        for r in records
    }
    mean = {}
    for arm in ARMS:
        mean[arm] = statistics.mean(
            by_key[g, w, side, arm]["success"]
            for g in genetic_seeds for w in world_seeds for side in (0, 1)
        )
    diffs = {}
    for a, b, name in CONTRASTS:
        matrix = [
            [
                statistics.mean(
                    by_key[g, w, side, a]["success"] -
                    by_key[g, w, side, b]["success"]
                    for side in (0, 1)
                ) for w in world_seeds
            ] for g in genetic_seeds
        ]
        per_run = [statistics.mean(row) for row in matrix]
        ci = hierarchical_bootstrap(
            matrix, genetic_seeds[0] * 123 + world_seeds[0] +
            [item[2] for item in CONTRASTS].index(name)
        )
        diffs[name] = {
            "delta_success": statistics.mean(per_run),
            "hierarchical_bootstrap_95": list(ci),
            "per_genetic_run": per_run,
            "positive_genetic_runs": sum(x > 0 for x in per_run),
            "negative_genetic_runs": sum(x < 0 for x in per_run),
            "tied_genetic_runs": sum(x == 0 for x in per_run),
        }
    primary = diffs["primary_rnn_vs_ff"]
    reset = diffs["reset_effect"]
    flipped = diffs["cue_flip_effect"]
    primary_gate = bool(
        mean["rnn"] > .55 and primary["delta_success"] > 0
        and primary["hierarchical_bootstrap_95"][0] > 0
        and primary["positive_genetic_runs"] >= 6
    )
    reset_gate = bool(
        reset["delta_success"] > 0
        and reset["hierarchical_bootstrap_95"][0] > 0
        and reset["positive_genetic_runs"] >= 6
    )
    cue_gate = bool(
        flipped["delta_success"] > 0
        and flipped["hierarchical_bootstrap_95"][0] > 0
    )
    delays = sorted(set(r["delay"] for r in records))
    by_delay = {
        str(delay): {
            arm: statistics.mean(
                r["success"] for r in records
                if r["arm"] == arm and r["delay"] == delay
            )
            for arm in ARMS
        } for delay in delays
    }
    return {
        "split": split,
        "independent_genetic_runs": len(genetic_seeds),
        "world_pairs": len(world_seeds),
        "paired_goal_trials_per_arm":
            len(genetic_seeds) * len(world_seeds) * 2,
        "means": mean,
        "by_delay": by_delay,
        "contrasts": diffs,
        "gates": {
            "primary_rnn_vs_ff": primary_gate,
            "causal_reset": reset_gate,
            "causal_cue_flip": cue_gate,
            "all_memory_gates": bool(primary_gate and reset_gate and cue_gate),
        },
        "analytical_controls_exact": {
            "ff_50_percent": mean["ff"] == .5,
            "reset_50_percent": mean["rnn_reset"] == .5,
            "no_cue_50_percent": mean["rnn_no_cue"] == .5,
            "random_binary_50_percent": mean["random_binary"] == .5,
            "scripted_100_percent": mean["scripted_one_bit"] == 1.,
        },
    }


def transfer_summary(records: list[dict], genetics: list[int],
                     worlds: list[int]) -> dict:
    metrics = ("guided_success", "free_success", "free_fork_reached")
    result = {
        arch: {
            k: statistics.mean(
                r[k] for r in records if r["architecture"] == arch
            ) for k in metrics
        } for arch in ("rnn", "ff")
    }
    result["world_pairs"] = len(worlds)
    result["genetic_runs"] = len(genetics)
    result["exploratory_only"] = True
    return result


def diagram(world: TrialWorld, path: list[tuple[int, int]]) -> str:
    cell = 22
    shapes = []
    for y in range(world.size):
        for x in range(world.size):
            c = (x, y)
            color = "#526173" if c in world.walls else "#e7f1e7"
            shapes.append(
                f'<rect x="{x*cell}" y="{y*cell}" width="{cell-1}" '
                f'height="{cell-1}" fill="{color}" rx="2"/>'
            )
    for side, gate in enumerate(world.gates):
        x, y = gate
        color = "#2e9b58" if side == world.side else "#b7c7b8"
        shapes.append(
            f'<circle cx="{x*cell+11}" cy="{y*cell+11}" '
            f'r="6" fill="{color}"/>'
        )
    xy = " ".join(f"{x*cell+11},{y*cell+11}" for x, y in path)
    shapes.append(f'<polyline points="{xy}" stroke="#e97720" '
                  'stroke-width="3" fill="none"/>')
    x, y = world.start
    shapes.append(f'<circle cx="{x*cell+11}" cy="{y*cell+11}" '
                  'r="7" fill="#286cd1"/>')
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {world.size*cell} {world.size*cell}" '
        f'width="{world.size*cell}" height="{world.size*cell}">'
        + "".join(shapes) + "</svg>\n"
    )


def run(settings: Settings, out: Path) -> dict:
    settings.validate()
    out.mkdir(parents=True, exist_ok=True)
    genetics = list(range(settings.genetic_seed_start,
                          settings.genetic_seed_start + settings.genetic_runs))
    main_seeds = list(range(settings.test_start,
                            settings.test_start + settings.test_worlds))
    shifted_seeds = list(range(settings.shifted_start,
                               settings.shifted_start + settings.shifted_worlds))
    guided_seeds = list(range(settings.guided_start,
                              settings.guided_start + settings.guided_worlds))
    main_worlds = [
        w for s in main_seeds for w in make_pair(s, settings)
    ]
    shifted_worlds = [
        w for s in shifted_seeds for w in make_pair(s, settings, shifted=True)
    ]
    guided_worlds = [
        w for s in guided_seeds for w in make_pair(s, settings)
    ]
    history, champions, selections = [], {}, []
    records, guided_records = [], []
    for genetic_seed in genetics:
        rnn, history_rnn, selection_rnn = evolve_one(
            settings, genetic_seed, "rnn"
        )
        ff, history_ff, selection_ff = evolve_one(
            settings, genetic_seed, "ff"
        )
        champions[str(genetic_seed)] = {
            "rnn": list(rnn), "ff": list(ff)
        }
        history.extend(history_rnn + history_ff)
        selections.extend((selection_rnn, selection_ff))
        for split, worlds in (("main", main_worlds),
                              ("shifted", shifted_worlds)):
            for world in worlds:
                delay = selected_delays(world.seed, split == "shifted")
                for arm in ARMS:
                    genome = (
                        ff if arm == "ff" else
                        None if arm in ("random_binary", "scripted_one_bit")
                        else rnn
                    )
                    r = run_trial(world, settings, arm, genome, delay)
                    records.append({
                        "split": split, "genetic_seed": genetic_seed,
                        **{k: v for k, v in r.items()
                           if k not in ("predecision_actions",
                                        "decision_observation")}
                    })
            print(f"M4.3 FINAL genetic_seed={genetic_seed} "
                  f"split={split} pairs={len(worlds)//2}", flush=True)
        for world in guided_worlds:
            for arch, genome in (("rnn", rnn), ("ff", ff)):
                guided = guided_corridor_trial(world, settings, arch, genome)
                free = free_corridor_trial(
                    world, settings,
                    "evolved_rnn" if arch == "rnn" else "evolved_ff",
                    genome
                )
                guided_records.append({
                    "genetic_seed": genetic_seed,
                    "world_seed": world.seed,
                    "side": world.side,
                    "architecture": arch,
                    "guided_success": guided["success"],
                    "free_success": free["success"],
                    "free_fork_reached": free["fork_reached"],
                })
    selected_main = [r for r in records if r["split"] == "main"]
    selected_shifted = [r for r in records if r["split"] == "shifted"]
    summary = {
        "protocol": "M4.3 preregistered causal delayed recall, motor-free",
        "settings": asdict(settings),
        "selection": selections,
        "main": summarize(selected_main, genetics, main_seeds, "main"),
        "shifted": summarize(selected_shifted, genetics, shifted_seeds, "shifted"),
        "corridor_transfer": transfer_summary(
            guided_records, genetics, guided_seeds
        ),
        "limits": [
            "Goal is a researcher-supplied cardinal cue, not discovered",
            "No locomotion before primary decision; a toy causal memory bench",
            "Long delays require extrapolation beyond training curriculum",
            "Weights evolve between lifetimes only, no within-life SGD",
            "A handcrafted one-bit memory baseline has 100% by design",
            "The same genetic run sees all paired worlds; bootstrap both run and world",
            "No network, shell, source edit or other external agent capabilities",
        ],
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8"
    )
    (out / "champions.json").write_text(
        json.dumps({
            "genomes": champions,
            "selections": selections,
            "model_parameters": {"rnn": genome_size("rnn", settings),
                                 "ff": genome_size("ff", settings)}
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (out / "train_validation.csv").open("w", newline="",
                                             encoding="utf-8") as f:
        keys = list(history[0])
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)
    with (out / "recall_trial_metrics.csv").open("w", newline="",
                                                encoding="utf-8") as f:
        keys = ("split", "genetic_seed", "world_seed", "side", "delay",
                "arm", "success", "choice", "target_direction",
                "fork_reached", "cue_at_readout", "fitness",
                "soft_correct_preference")
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows({k: r[k] for k in keys} for r in records)
    with (out / "corridor_transfer.csv").open("w", newline="",
                                             encoding="utf-8") as f:
        keys = ("genetic_seed", "world_seed", "side", "architecture",
                "guided_success", "free_success", "free_fork_reached")
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(guided_records)
    for label, seed in (("main", main_seeds[0]),
                        ("shifted", shifted_seeds[0])):
        world = make_pair(seed, settings, label == "shifted")[0]
        genome = tuple(champions[str(genetics[0])]["rnn"])
        delay = selected_delays(seed, label == "shifted")
        trace = run_trial(world, settings, "rnn", genome,
                          delay, save_sequence=True)
        path = [world.start]
        for _ in range(world.stem_length):
            x, y = path[-1]
            dx, dy = ((0, -1), (1, 0), (0, 1), (-1, 0))[world.forward]
            path.append((x+dx, y+dy))
        (out / f"example_{label}.svg").write_text(
            diagram(world, path), encoding="utf-8"
        )
        (out / f"example_{label}_trace.json").write_text(
            json.dumps({
                "world_seed": seed, "delay": delay,
                "goal": list(world.target),
                "fork": list(world.fork),
                "start": list(world.start),
                "selected_rnn_genetic_seed": genetics[0],
                "model_steps": len(inputs_for_trial(world, delay)),
                "trial": trace,
            }, indent=2) + "\n", encoding="utf-8"
        )
    print("M4.3 PRIMARY", summary["main"]["contrasts"]["primary_rnn_vs_ff"],
          flush=True)
    print("M4.3 RESET", summary["main"]["contrasts"]["reset_effect"],
          flush=True)
    print("M4.3 CUE", summary["main"]["contrasts"]["cue_flip_effect"],
          flush=True)
    print("M4.3 GATES", summary["main"]["gates"], flush=True)
    print("M4.3 SHIFTED", summary["shifted"]["contrasts"]["primary_rnn_vs_ff"],
          flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = Settings()
    for fld in fields(Settings):
        val = getattr(defaults, fld.name)
        parser.add_argument("--" + fld.name.replace("_", "-"),
                            type=type(val), default=val)
    parser.add_argument("--out", type=Path, default=Path("results/m4_3"))
    args = parser.parse_args()
    run(Settings(**{
        fld.name: getattr(args, fld.name) for fld in fields(Settings)
    }), args.out)


if __name__ == "__main__":
    main()
