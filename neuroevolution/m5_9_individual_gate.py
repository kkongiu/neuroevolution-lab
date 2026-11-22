"""M5.9c: individual-learning prerequisite gate, BEFORE any more evolution.

A single M5.6 recurrent learner experiences a stationary two-action cue rule,
then an unannounced complete rule reversal. One reward after each observed
action, never a target-action label. Compare identical starter genotypes and
action RNG with plasticity enabled, disabled, and reward-shuffled.

This is diagnostic abstract cognition, not physical ecology, inheritance,
or a final test on sealed worlds. Do not advance to harder curricula unless
predeclared success criteria hold. No fitness-based parent selection occurs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from pathlib import Path

from .m5_6_online import Learner

PROBE_SEEDS = tuple(range(3200, 3224))
ARMS = ("plastic", "frozen", "shuffled_feedback")
TRIALS_PER_PHASE = 160
WINDOW = 40
REWARD_CORRECT = 1.0
REWARD_INCORRECT = -0.15
# These thresholds are fixed before running the development diagnostic.
MIN_ABOVE_FROZEN = .10
MIN_REVERSAL_RECOVERY = .10
MIN_ABSOLUTE_LATE_ACCURACY = .65


def genome(seed: int) -> tuple[float,...]:
    if type(seed) is not int or seed not in PROBE_SEEDS:
        raise ValueError("M5.9c individual probe only 3200..3223")
    rng = random.Random(seed * 71_911 + 59)
    return tuple(rng.gauss(0., .26) for _ in range(260))


def sensory(cue:int, previous_feedback:float) -> tuple[float,...]:
    """Only cue and previous success feedback; never hidden rule or target."""
    obs = [0.]*19
    obs[cue] = 1.
    obs[2] = previous_feedback
    return tuple(obs)


def schedule(seed:int)->tuple[int,...]:
    """Exogenous, balanced binary cues in every 40-trial window."""
    rng = random.Random(seed*97_009 + 173)
    out = []
    for _ in range((2*TRIALS_PER_PHASE)//WINDOW):
        cues = [0]*(WINDOW//2)+[1]*(WINDOW//2)
        rng.shuffle(cues)
        out.extend(cues)
    return tuple(out)


def run(seed:int, arm:str)->dict:
    if seed not in PROBE_SEEDS or arm not in ARMS:
        raise ValueError("Registered seed and learning-control arm required")
    learner = Learner.fresh(genome(seed),(5,5))
    cues = schedule(seed)
    rng_rule = random.Random(seed*1009+91)
    rule = rng_rule.randrange(2)
    previous_feedback=0.
    correct=[]
    updates=0
    # Yoked feedback is independent of this learner's own current actions,
    # but has a matched rate by alternating balanced binary outcomes.
    feedback_rng = random.Random(seed*8191+77)
    scrambled = list(cues)
    feedback_rng.shuffle(scrambled)
    for tick,cue in enumerate(cues):
        if tick == TRIALS_PER_PHASE:
            rule ^= 1
        obs = sensory(cue,previous_feedback)
        # Same action random ticket within each paired arm; later actions
        # may diverge as synapses/hidden activation diverge.
        decision_rng = random.Random(seed*997_007+tick*1069+21)
        action,cache=learner.action(obs,(0,1),decision_rng)
        if action is None:
            raise AssertionError("No action produced")
        did_solve=int(action==(cue^rule))
        correct.append(did_solve)
        feedback=float(did_solve)
        if arm=="shuffled_feedback":
            feedback=float(scrambled[tick])
        reward=REWARD_CORRECT if feedback else REWARD_INCORRECT
        updates+=int(learner.learn(cache,reward,enabled=arm!="frozen")>0)
        # For fairness, the observation reveals real own previous feedback
        # in all arms. The shuffled intervention ONLY scrambles learning
        # reward, not the environmental observation.
        previous_feedback=float(did_solve)
    def accuracy(values):
        return sum(values)/len(values)
    a=correct[:TRIALS_PER_PHASE]
    b=correct[TRIALS_PER_PHASE:]
    metrics={
        "initial_first_window":accuracy(a[:WINDOW]),
        "initial_last_window":accuracy(a[-WINDOW:]),
        "reversal_first_window":accuracy(b[:WINDOW]),
        "reversal_last_window":accuracy(b[-WINDOW:]),
        "initial_accuracy":accuracy(a),
        "reversal_accuracy":accuracy(b),
        "initial_learning_gain":accuracy(a[-WINDOW:])-accuracy(a[:WINDOW]),
        "reversal_recovery":accuracy(b[-WINDOW:])-accuracy(b[:WINDOW]),
        "overall_accuracy":accuracy(correct),
        "nonzero_policy_updates":updates,
        "private_weight_drift_l2":sum(
            (x-y)**2 for x,y in zip(learner.weights,genome(seed)))**.5,
        "memory_predictor_updates":learner.predictor.learned_transitions,
    }
    return {"seed":seed,"arm":arm,**metrics}


def experiment(seeds:tuple[int,...]=PROBE_SEEDS)->dict:
    if (not seeds or len(seeds)!=len(set(seeds)) or any(
            type(s) is not int or s not in PROBE_SEEDS for s in seeds)):
        raise ValueError("Unique M5.9c development probes only")
    rows=[run(s,arm) for s in seeds for arm in ARMS]
    index={(r["seed"],r["arm"]):r for r in rows}
    summary={}
    for arm in ARMS:
        subset=[index[(seed,arm)] for seed in seeds]
        summary[arm]={key:statistics.mean(x[key] for x in subset)
                      for key in (
        "initial_first_window","initial_last_window",
        "reversal_first_window","reversal_last_window",
        "initial_learning_gain","reversal_recovery","overall_accuracy",
        "nonzero_policy_updates","private_weight_drift_l2")}
    paired_advantages=[
        index[(seed,"plastic")]["reversal_last_window"]-
        index[(seed,"frozen")]["reversal_last_window"] for seed in seeds]
    shuffled_advantages=[
        index[(seed,"plastic")]["reversal_last_window"]-
        index[(seed,"shuffled_feedback")]["reversal_last_window"]
        for seed in seeds]
    # Explicit gate: group-level average thresholds all must hold. This is
    # a DEVELOPMENT screening rule, not a formal statistical significance test.
    gate={
        "initial_gain_above_frozen": (
            summary["plastic"]["initial_learning_gain"]-
            summary["frozen"]["initial_learning_gain"]>=MIN_ABOVE_FROZEN),
        "reversal_recovery_at_least_10pp":(
            summary["plastic"]["reversal_recovery"]>=MIN_REVERSAL_RECOVERY),
        "reversal_final_accuracy_above_65pct":(
            summary["plastic"]["reversal_last_window"]>=MIN_ABSOLUTE_LATE_ACCURACY),
        "reversal_better_than_frozen_by_10pp":(
            statistics.mean(paired_advantages)>=MIN_ABOVE_FROZEN),
        "reversal_better_than_scrambled_reward_by_10pp":(
            statistics.mean(shuffled_advantages)>=MIN_ABOVE_FROZEN)
    }
    digest=hashlib.sha256(json.dumps(
        rows,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return {
        "status":"development_individual_learning_diagnostic_not_final",
        "seeds":list(seeds),"trials_per_phase":TRIALS_PER_PHASE,
        "rule_reversal_tick":TRIALS_PER_PHASE,
        "selection_or_reproduction":False,
        "arms":list(ARMS),"summary":summary,
        "paired_mean_plastic_minus_frozen_reversal_final":
            statistics.mean(paired_advantages),
        "paired_mean_plastic_minus_scrambled_reversal_final":
            statistics.mean(shuffled_advantages),
        "preregistered_screening_criteria":gate,
        "advance_to_delayed_feedback":all(gate.values()),
        "raw_rows":rows,"rows_sha256":digest
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,default=Path(
        "results/m5-9/individual-learning-gate-development.json"))
    args=parser.parse_args()
    report=experiment()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",
                        encoding="utf-8")
    print("M5_9_INDIVIDUAL_LEARNING_GATE "+json.dumps({
        k:report[k] for k in (
            "status","trials_per_phase","summary",
            "paired_mean_plastic_minus_frozen_reversal_final",
            "paired_mean_plastic_minus_scrambled_reversal_final",
            "preregistered_screening_criteria",
            "advance_to_delayed_feedback","rows_sha256")},
        sort_keys=True),flush=True)


if __name__=="__main__":
    main()
