"""M5.9 isolated first test: evolve *initial neural predisposition*, not memory.

This is a minimal ABSTRACT two-choice problem arena, NOT a physical M5.8
ecology integration. There is no energy or predation in this initial proof of
concept. A valid puzzle solution is a real observed correct response to a
private, randomly selected contextual rule; only actual success permits
selection. No correct action is ever given to the learner as a supervised label.

Genotype: exact 260 initial synapses of M5.6's neural controller. Every child
starts a new Learner.fresh(genotype), with ZERO recurrent state, zero visits,
zero forward-predictor training, and no inherited in-life weight changes.
The only heredity is the mutated ORIGINAL genotype.

Arms: success-driven reproduction + plastic synapses, success-driven
reproduction with synapses frozen, random-parent reproduction + plastic
synapses. Reproduction slots and population size matched; selection alone
differs between successful and random parent controls.

Training puzzle family uses two observable cues, a hidden binary contextual
rule and rule changes between episodes. The XOR mapping is STRUCTURAL,
engineered and tiny. The problem has very limited generalization scope and
cannot prove open-ended cognition or physical problem solving.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path

from .m5_6_online import Learner

EVOLUTION_SEEDS = tuple(range(3000,3012))
EVALUATION_SEEDS = tuple(range(3050,3062))
ARMS = ("solution_plastic", "solution_frozen", "random_parent_plastic")
POPULATION = 10
GENERATIONS = 10
TRAIN_EPISODES = 4
TRIALS_PER_EPISODE = 10
EVAL_EPISODES = 5
MUTATION_SD = .035
GENOME_LENGTH = 260
LEARN_BONUS = 1.0


@dataclass(frozen=True)
class Puzzle:
    """Private rule is hidden, observable cue/phase provided, not the answer."""
    phase: int
    rule: int
    cues: tuple[int,...]

    def observation(self, cue: int, trial: int) -> tuple[float,...]:
        signals = [0.]*19
        signals[0] = float(cue == 0)
        signals[1] = float(cue == 1)
        signals[2] = float(self.phase == 0)
        signals[3] = float(self.phase == 1)
        signals[4] = trial/max(1,len(self.cues)-1)
        # Feature 5 is previous feedback furnished to agent separately.
        return tuple(signals)

    def successful(self, cue: int, action: int) -> bool:
        return action == (cue ^ self.rule ^ self.phase)


def puzzles(seed: int, generation: int, episodes: int,
            *, evaluation: bool = False) -> tuple[Puzzle,...]:
    if (seed not in (EVALUATION_SEEDS if evaluation else EVOLUTION_SEEDS)
            or episodes < 1 or episodes > 10):
        raise ValueError("M5.9 separate guarded training/evaluation cohorts")
    rng = random.Random(seed*1_000_003+generation*33_731+int(evaluation)*99_017)
    out=[]
    for e in range(episodes):
        phase=e%2
        rule=rng.randrange(2)
        cues = tuple(rng.randrange(2) for _ in range(TRIALS_PER_EPISODE))
        out.append(Puzzle(phase,rule,cues))
    return tuple(out)


def founder_genomes(seed: int) -> tuple[tuple[float,...],...]:
    if seed not in EVOLUTION_SEEDS:
        raise ValueError("No final or evaluation seed may initialize selection")
    rng = random.Random(seed*179_999+39)
    return tuple(tuple(rng.gauss(0.,.26) for _ in range(GENOME_LENGTH))
                 for _ in range(POPULATION))


def evaluate_individual(genome: tuple[float,...], tasks: tuple[Puzzle,...],
                        seed: int, *, plastic: bool) -> dict:
    agent = Learner.fresh(genome, (5,5))
    attempts = 0
    successful = 0
    problems_solved = 0
    early = 0
    late = 0
    changed_weights = 0
    # New puzzle: reset transient hidden state but preserve in-life synaptic
    # modifications. No prior episode cue sequence or rule is copied.
    for e,puzzle in enumerate(tasks):
        agent.hidden = [0.]*8
        hits=0
        prev_feedback=0.
        for t,cue in enumerate(puzzle.cues):
            obs = list(puzzle.observation(cue,t))
            obs[5]=prev_feedback
            obs=tuple(obs)
            rng = random.Random(seed*100_009+e*10_007+t*503)
            action,cache=agent.action(obs,(0,1),rng)
            if action is None:
                raise AssertionError("No legal action in binary arena")
            outcome=int(puzzle.successful(cue,action))
            attempts += 1
            successful += outcome
            hits += outcome
            if t < len(puzzle.cues)//2:
                early += outcome
            else:
                late += outcome
            changed_weights += int(agent.learn(
                cache, LEARN_BONUS if outcome else 0.,
                enabled=plastic) > 0.)
            prev_feedback = float(outcome)
        # Explicit, observed episode-level problem-solvers have achieved
        # >=70% response accuracy; chance baseline is 50%. Fixed in advance.
        problems_solved += int(hits >= 7)
    return {"accuracy": successful/attempts,
            "solved_episodes":problems_solved,
            "early_accuracy":early/(attempts/2),
            "late_accuracy":late/(attempts/2),
            "learn_updates":changed_weights,
            "learned_weight_drift":math.sqrt(math.fsum(
                (a-b)**2 for a,b in zip(agent.weights,genome))),
            "final_hidden_norm":math.sqrt(math.fsum(h*h for h in agent.hidden)),
            "personal_predictor_updates":agent.predictor.learned_transitions}


def mutate(genome: tuple[float,...], rng: random.Random) -> tuple[float,...]:
    if len(genome)!=GENOME_LENGTH:
        raise ValueError("Invalid neural genotype")
    return tuple(max(-3., min(3.,w+rng.gauss(0.,MUTATION_SD)))
                 for w in genome)


def generation_step(genomes: tuple[tuple[float,...],...],
                    tasks: tuple[Puzzle,...], seed: int, generation: int,
                    arm: str) -> tuple[tuple[tuple[float,...],...],dict]:
    if arm not in ARMS or len(genomes)!=POPULATION:
        raise ValueError("Invalid evolution arm")
    plastic=arm!="solution_frozen"
    results=[evaluate_individual(g,tasks,
             seed*31_003+generation*201+i*47,plastic=plastic)
             for i,g in enumerate(genomes)]
    # Selection factor is ONLY the actual number of solved episodes.
    # All arms have exactly POPULATION offspring slots. Random-parent
    # controls draw founders without looking at the puzzle outcomes.
    scores = [r["solved_episodes"] for r in results]
    selection_rng=random.Random(seed*307_009+generation*1013+55)
    weights = ([1. + score for score in scores]
               if arm!="random_parent_plastic" else [1.]*POPULATION)
    parents=selection_rng.choices(range(POPULATION),weights=weights,k=POPULATION)
    children=tuple(mutate(genomes[p],random.Random(
        seed*973_817+generation*101_773+slot*301+17))
        for slot,p in enumerate(parents))
    if any(child == genomes[parent] for child,parent in zip(children,parents)):
        raise AssertionError("Genome not mutated")
    return children,{
        "mean_solved":statistics.mean(scores),
        "mean_accuracy":statistics.mean(r["accuracy"] for r in results),
        "mean_learning_drift":statistics.mean(
            r["learned_weight_drift"] for r in results),
        "mean_late_minus_early":statistics.mean(
            r["late_accuracy"]-r["early_accuracy"] for r in results),
        "total_solved":sum(scores),
        "parent_indices":parents,
        "unique_selected_parents":len(set(parents)),
        "all_children_inherit_initial_genotypes":True,
    }


def run(seed: int, arm: str, *, generations: int = GENERATIONS) -> dict:
    if seed not in EVOLUTION_SEEDS or arm not in ARMS or (
            type(generations) is not int or not 1<=generations<=GENERATIONS):
        raise ValueError("Development-only evolution trial")
    genomes=founder_genomes(seed)
    history=[]
    for generation in range(generations):
        tasks=puzzles(seed,generation,TRAIN_EPISODES)
        genomes,summary=generation_step(genomes,tasks,seed,generation,arm)
        history.append({"generation":generation+1,**summary})
    # No transfer of the individual's during-life modifications, only
    # descendants of the genetic starter weights.
    fresh_tasks=puzzles(EVALUATION_SEEDS[seed-EVOLUTION_SEEDS[0]],
                        777,EVAL_EPISODES,evaluation=True)
    assessments=[evaluate_individual(
        genome,fresh_tasks,
        91_000+seed*111+i*271,plastic=arm!="solution_frozen")
        for i,genome in enumerate(genomes)]
    # Also evaluate zero-adaptation outcome for genetic predisposition.
    frozen_assessments=[evaluate_individual(
        genome,fresh_tasks,91_000+seed*111+i*271,plastic=False)
        for i,genome in enumerate(genomes)]
    return {
        "seed":seed,"arm":arm,"generations":generations,
        "population":POPULATION,"final_population":len(genomes),
        "final_genotype_sha256":hashlib.sha256(json.dumps(
            genomes,separators=(",",":")).encode()).hexdigest(),
        "history":history,
        "novel_problems":len(fresh_tasks),
        "mean_novel_accuracy":statistics.mean(x["accuracy"] for x in assessments),
        "mean_novel_solved":statistics.mean(
            x["solved_episodes"] for x in assessments),
        "mean_novel_late_minus_early":statistics.mean(
            x["late_accuracy"]-x["early_accuracy"] for x in assessments),
        "mean_novel_accuracy_no_lifelong_learning":statistics.mean(
            x["accuracy"] for x in frozen_assessments),
        "mean_novel_solved_no_lifelong_learning":statistics.mean(
            x["solved_episodes"] for x in frozen_assessments),
        "mean_novel_learning_weight_drift":statistics.mean(
            x["learned_weight_drift"] for x in assessments),
        "mean_novel_memory_or_predictor_transferred":0,
    }


def experiment(seeds: tuple[int,...]=EVOLUTION_SEEDS) -> dict:
    if not seeds or len(seeds)!=len(set(seeds)) or any(
            type(s) is not int or s not in EVOLUTION_SEEDS for s in seeds):
        raise ValueError("Only unique M5.9 evolution development seeds")
    rows=[run(seed,arm) for seed in seeds for arm in ARMS]
    summary={}
    for arm in ARMS:
        x=[r for r in rows if r["arm"]==arm]
        summary[arm]={
            "mean_new_problem_accuracy":statistics.mean(
                r["mean_novel_accuracy"] for r in x),
            "mean_new_problems_solved_per_organism":statistics.mean(
                r["mean_novel_solved"] for r in x),
            "mean_new_problem_accuracy_with_learning_disabled":statistics.mean(
                r["mean_novel_accuracy_no_lifelong_learning"] for r in x),
            "mean_within_lifetime_late_minus_early_accuracy":statistics.mean(
                r["mean_novel_late_minus_early"] for r in x),
            "mean_weight_change_during_test":statistics.mean(
                r["mean_novel_learning_weight_drift"] for r in x),
        }
    canon=json.dumps(rows,sort_keys=True,separators=(",",":"))
    return {"status":"isolated_abstract_development_evolution_not_physical",
            "world_seed_guard":"3000-3011 evolution; 3050-3061 distinct test tasks",
            "evolution_seeds":list(seeds),"physical_worlds":0,
            "arms":list(ARMS),"generations":GENERATIONS,
            "population":POPULATION,"task_family":"binary-contextual-rule-reversal",
            "new_test_worlds_are_not_independent_of_repeated_model_selection":True,
            "raw_rows":rows,"summary":summary,
            "rows_sha256":hashlib.sha256(canon.encode()).hexdigest()}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path,default=Path(
        "results/m5-9/initial-genotype-selection-development.json"))
    args=ap.parse_args()
    result=experiment()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",
                        encoding="utf-8")
    print("M5_9_EVOLUTION "+json.dumps({
        "status":result["status"],
        "generations":result["generations"],
        "population":result["population"],
        "runs":len(result["raw_rows"]),
        "summary":result["summary"],
        "rows_sha256":result["rows_sha256"]},sort_keys=True),flush=True)


if __name__=="__main__":
    main()
