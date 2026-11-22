"""M5.9b: multistep delayed-outcome problems with evolving *initial* synapses.

This is a STANDALONE cognitive arena, not a claim about embodied survival.
Three consecutive binary gates must all be passed for a solution (random
one-attempt baseline exactly 1/8 for action-unbiased policies); there is no
reward or correctness signal until all three actions have occurred.
Different episodes have newly chosen hidden rule bits, unobservable directly.
The action-conditioned cue and gate index are observed, but never the answer.

The inherited genome is EXACTLY the founder's initial 260 weights, mutated,
never synapses changed by within-life REINFORCE or transient hidden state.
Terminal feedback is delivered after the full 3-action sequence: all three
policy gradients are updated in one batch after the delayed outcome.

Matched arms: solver selection + plastic; solver selection + frozen;
random-parent selection + plastic. Same genomic founders and identical
episode corpus within each seed/generation; exactly equal offspring slots.

A >chance association does NOT by itself establish an evolved ability to
adapt: separately inspect within-episode speed, frozen descendant behavior,
and fresh problems; selection can favor static biases.
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
from .m5_9_selection import mutate

EVOLUTION_SEEDS = tuple(range(3100,3112))
NOVEL_SEEDS = tuple(range(3150,3162))
ARMS = ("solver_plastic","solver_frozen","yoked_random_plastic")
GENS = 8
POP = 8
EPISODES = 3
ATTEMPTS = 12
GATES = 3
CHANCE_SOLVE = (1/2)**GATES  # one-shot 12.5%, not 50%
SOLVED_THRESHOLD = 4 # P(Binomial(12,1/8)>=4) ~ 5% for i.i.d. random policy
UPDATE_REWARD_SUCCESS = 1.
UPDATE_REWARD_FAILURE = -.15


@dataclass(frozen=True)
class Problem:
    hidden_rule: int
    context: int
    cues: tuple[tuple[int,...], ...]

    def observed(self, attempt: int, gate: int, last_result: float) -> tuple[float,...]:
        values=[0.]*19
        cue=self.cues[attempt][gate]
        values[cue]=1.
        values[2+gate]=1.
        values[5+self.context]=1.
        values[7]=last_result
        values[8]=attempt/(ATTEMPTS-1)
        # All rule/target data are deliberately omitted.
        return tuple(values)

    def correct(self, attempt:int, gate:int, action:int)->bool:
        cue=self.cues[attempt][gate]
        return action == (cue ^ self.hidden_rule ^ self.context ^ (gate%2))


def tasks(seed:int, generation:int, count:int, *, novel:bool=False)->tuple[Problem,...]:
    domain=NOVEL_SEEDS if novel else EVOLUTION_SEEDS
    if type(seed) is not int or seed not in domain or not 1<=count<=5:
        raise ValueError("M5.9b training and novel-world seed cohorts are sealed")
    rng=random.Random(seed*1_000_003+generation*9901+int(novel)*811)
    out=[]
    for k in range(count):
        rule=rng.randrange(2)
        context=k%2
        cues=tuple(tuple(rng.randrange(2) for _ in range(GATES))
                   for _ in range(ATTEMPTS))
        out.append(Problem(rule,context,cues))
    return tuple(out)


def founders(seed:int)->tuple[tuple[float,...],...]:
    if seed not in EVOLUTION_SEEDS:
        raise ValueError("Only evolution development seeds may produce founders")
    rng=random.Random(seed*79_993+117)
    return tuple(tuple(rng.gauss(0,.26) for _ in range(260))
                 for _ in range(POP))


def solve(genome:tuple[float,...], problems:tuple[Problem,...],
          experience_seed:int, *, plastic:bool)->dict:
    learner=Learner.fresh(genome,(5,5))
    before_weights=tuple(learner.weights)
    solved_episodes=0
    successes=0
    first_half=0
    last_half=0
    update_count=0
    for e,p in enumerate(problems):
        # Clear only transient recurrent state for a new hidden-world rule.
        # Within-life learned parameters persist across problem episodes.
        learner.hidden=[0.]*8
        prior_feedback=0.
        episode_successes=0
        for attempt in range(ATTEMPTS):
            cached=[]
            passed=True
            # Three actions must be chosen before ANY feedback/gradient update.
            for gate in range(GATES):
                obs=p.observed(attempt,gate,prior_feedback)
                rng=random.Random(experience_seed*103_103+e*19_991
                                  +attempt*1103+gate*71)
                action,cache=learner.action(obs,(0,1),rng)
                if cache is None or action is None:
                    raise AssertionError("No action in gate task")
                cached.append(cache)
                passed = bool(passed and p.correct(attempt,gate,action))
            outcome=int(passed)
            successes+=outcome
            episode_successes+=outcome
            if attempt < ATTEMPTS//2:
                first_half+=outcome
            else:
                last_half+=outcome
            # A failed attempt returns terminal failure only; *not* which
            # step failed. No future information or answer leaks.
            terminal_reward=(UPDATE_REWARD_SUCCESS if passed
                             else UPDATE_REWARD_FAILURE)
            for cache in cached:
                update_count+=int(learner.learn(
                    cache,terminal_reward,enabled=plastic)>0)
            prior_feedback=float(outcome)
        solved_episodes+=int(episode_successes>=SOLVED_THRESHOLD)
    total=len(problems)*ATTEMPTS
    return {
        "attempt_success":successes/total,
        "solved_episodes":solved_episodes,
        "first_half_success":first_half/(total/2),
        "last_half_success":last_half/(total/2),
        "within_episode_improvement":(last_half-first_half)/(total/2),
        "weight_drift":math.sqrt(math.fsum(
            (a-b)**2 for a,b in zip(before_weights,learner.weights))),
        "policy_weight_updates":update_count,
        "predictor_memory_retained":learner.predictor.learned_transitions,
    }


def evolve(seed:int,arm:str,generations:int=GENS)->dict:
    if type(seed) is not int or seed not in EVOLUTION_SEEDS or (
            arm not in ARMS or type(generations) is not int
            or not 1<=generations<=GENS):
        raise ValueError("M5.9b registered development evolution only")
    genomes=founders(seed)
    plastic=arm!="solver_frozen"
    history=[]
    for generation in range(generations):
        problems=tasks(seed,generation,EPISODES)
        trained=[solve(g,problems,
                       experience_seed=seed*3037+generation*977+i*211,
                       plastic=plastic)
                 for i,g in enumerate(genomes)]
        scores=[r["solved_episodes"] for r in trained]
        selection_rng=random.Random(seed*13001+generation*907+13)
        weights=([1+s for s in scores] if arm!="yoked_random_plastic"
                 else [1]*POP)
        parents=selection_rng.choices(range(POP),weights=weights,k=POP)
        next_genomes=tuple(mutate(genomes[parent],random.Random(
            seed*71_991+generation*8831+slot*137+33))
            for slot,parent in enumerate(parents))
        history.append({
            "generation":generation+1,
            "mean_train_success":statistics.mean(
                r["attempt_success"] for r in trained),
            "mean_train_late_minus_early":statistics.mean(
                r["within_episode_improvement"] for r in trained),
            "total_problem_episodes_solved":sum(scores),
            "offspring_slots":len(next_genomes),
            "parents":parents,
            "inherited_only_genetic_initial_weights":True,
        })
        genomes=next_genomes
    novel_seed=NOVEL_SEEDS[seed-EVOLUTION_SEEDS[0]]
    novel_problems=tasks(novel_seed,777,5,novel=True)
    outcome=[solve(g,novel_problems,
                   experience_seed=seed*6061+i*273,plastic=plastic)
             for i,g in enumerate(genomes)]
    frozen=[solve(g,novel_problems,
                  experience_seed=seed*6061+i*273,plastic=False)
            for i,g in enumerate(genomes)]
    return {
        "seed":seed,"arm":arm,"generations":generations,
        "genotype_sha256":hashlib.sha256(json.dumps(
            genomes,separators=(",",":")).encode()).hexdigest(),
        "history":history,
        "novel_attempt_success":statistics.mean(
            r["attempt_success"] for r in outcome),
        "novel_episode_solves_per_individual":statistics.mean(
            r["solved_episodes"] for r in outcome),
        "novel_late_minus_early":statistics.mean(
            r["within_episode_improvement"] for r in outcome),
        "novel_success_when_plasticity_disabled":statistics.mean(
            r["attempt_success"] for r in frozen),
        "mean_lifetime_weight_drift":statistics.mean(
            r["weight_drift"] for r in outcome),
        "transferred_episodic_memory":0,
    }


def experiment(seeds:tuple[int,...]=EVOLUTION_SEEDS)->dict:
    if (not seeds or len(seeds)!=len(set(seeds)) or any(
            type(s) is not int or s not in EVOLUTION_SEEDS for s in seeds)):
        raise ValueError("M5.9b development evolutionary lineage seed fence")
    rows=[evolve(s,a) for s in seeds for a in ARMS]
    stats={}
    for arm in ARMS:
        subset=[r for r in rows if r["arm"]==arm]
        stats[arm]={
            "mean_novel_three_gate_success":statistics.mean(
                r["novel_attempt_success"] for r in subset),
            "mean_novel_episodes_solved_out_of_five":statistics.mean(
                r["novel_episode_solves_per_individual"] for r in subset),
            "mean_novel_late_minus_early":statistics.mean(
                r["novel_late_minus_early"] for r in subset),
            "mean_novel_success_with_learning_disabled":statistics.mean(
                r["novel_success_when_plasticity_disabled"] for r in subset),
        }
    digest=hashlib.sha256(json.dumps(
        rows,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return {
        "status":"development_abstract_delayed_credit_assignment_not_embodied",
        "lineage_seeds":list(seeds),"novel_seeds_provenance":"3150-3161",
        "runs":len(rows),"generations":GENS,"population":POP,
        "puzzle_gate_count":GATES,"chance_single_attempt":CHANCE_SOLVE,
        "solved_threshold":SOLVED_THRESHOLD,"summary":stats,
        "rows_sha256":digest,"raw_rows":rows,
    }


def main()->None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,default=Path(
        "results/m5-9/delayed-three-gate-development.json"))
    args=parser.parse_args()
    report=experiment()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n",
                        encoding="utf-8")
    print("M5_9_DELAYED_EVOLUTION "+json.dumps({
        k:report[k] for k in (
            "status","runs","generations","population","puzzle_gate_count",
            "chance_single_attempt","solved_threshold","summary","rows_sha256")
    },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
