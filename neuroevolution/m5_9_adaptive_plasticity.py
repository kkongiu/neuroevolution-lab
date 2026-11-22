"""M5.9d: multiple hidden-rule reversals; adaptive plasticity pilot.

Preregistered DEVELOPMENT-only diagnostic on 24 seeds not used previously
(3224..3247). Four paired arms, identical initial genotypes, cue order,
hidden rule reversals, per-action RNG, and feedback budget. Never access
reversal timestamps in any policy or regulator.

Adaptive arm tracks *unexpected increase in error rate* (fast EMA minus
slow EMA); the detector observes ONLY realized successes/failures, not
hidden target or future changes. Rate modulation is bounded and nonheritable.
Random arm receives independently generated rate changes with identical
range and roughly the same typical rate, no outcome information. Constant
arm reproduces original M5.6 rate .02; decay arm uses prescribed gradual
decay. Fixed trial length, 4 phases, 3 unannounced rule reversals.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from dataclasses import dataclass
from pathlib import Path

from .m5_6_online import Learner
from .m5_9_individual_gate import sensory

SEEDS = tuple(range(3224,3248))
ARMS = ("constant","decay","adaptive","random_rate","yoked_shuffled_rate")
PHASES = 4
PHASE_LENGTH = 160
WINDOW = 40
RATE_BASE = .02
RATE_MIN = .005
RATE_MAX = .08
FAST_ALPHA = .20
SLOW_ALPHA = .025
SURPRISE_GAIN = 2.5
CRITERIA = {
    "min_recovery_delta_over_constant": .10,
    "min_final_adaptive_accuracy": .65,
    "min_recovery_delta_over_random": .10,
    "min_initial_phase_gain": .08,
}
# Same seed per organism across the four arms; none of these are M5.6
# source/holdout/replication or M5.9c exploratory seeds.


@dataclass
class ErrorMonitor:
    fast_error: float = .5
    slow_error: float = .5
    changes: int = 0
    max_surprise: float = 0.

    def update(self, success: int) -> float:
        error=float(1-success)
        self.fast_error += FAST_ALPHA*(error-self.fast_error)
        self.slow_error += SLOW_ALPHA*(error-self.slow_error)
        surprise=max(0.,self.fast_error-self.slow_error)
        self.max_surprise=max(self.max_surprise,surprise)
        # Rate computed only from OBSERVED feedback on last attempt.
        if surprise > .12:
            self.changes+=1
        level=min(1.,SURPRISE_GAIN*surprise)
        return RATE_MIN+(RATE_MAX-RATE_MIN)*level


def genotype(seed:int)->tuple[float,...]:
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError("Only M5.9d development seeds 3224–3247")
    rng=random.Random(seed*71_911+59)
    return tuple(rng.gauss(0.,.26) for _ in range(260))


def cues_for(seed:int)->tuple[int,...]:
    if seed not in SEEDS:
        raise ValueError("Not a development seed")
    rng=random.Random(seed*97_009+173)
    out=[]
    for _ in range(PHASES*PHASE_LENGTH//WINDOW):
        group=[0]*(WINDOW//2)+[1]*(WINDOW//2)
        rng.shuffle(group)
        out.extend(group)
    return tuple(out)


def run(seed:int,arm:str, *, yoked_rates:tuple[float,...]|None=None)->dict:
    if seed not in SEEDS or arm not in ARMS:
        raise ValueError("Unknown M5.9d development seed/arm")
    if arm == "yoked_shuffled_rate":
        if yoked_rates is None or len(yoked_rates) != PHASES*PHASE_LENGTH or any(
                not RATE_MIN <= v <= RATE_MAX for v in yoked_rates):
            raise ValueError("Must provide complete bounded frozen-rate schedule")
    elif yoked_rates is not None:
        raise ValueError("Yoked rates only valid in matched placebo")
    learner=Learner.fresh(genotype(seed),(5,5))
    cues=cues_for(seed)
    first_rule=random.Random(seed*1009+91).randrange(2)
    actual_rule=first_rule
    prev_correct=0.
    monitor=ErrorMonitor()
    events=[]
    correct=[]
    lr=[]
    updates=0
    for tick,cue in enumerate(cues):
        # Reversal is an exogenous ground-truth evaluation condition.
        # Neither Learner nor ErrorMonitor is told when it occurs.
        if tick and tick%PHASE_LENGTH==0:
            actual_rule^=1
        obs=sensory(cue,prev_correct)
        action,cache=learner.action(
            obs,(0,1),random.Random(seed*997_007+tick*1069+21))
        if action is None:
            raise AssertionError("Missing action")
        success=int(action==(cue^actual_rule))
        correct.append(success)
        # All arms get identical REAL feedback given their actual choice.
        # Disconnected placebo rate changes never inspect success.
        rate=RATE_BASE
        if arm=="decay":
            rate=RATE_MIN+(RATE_BASE-RATE_MIN)*(1-tick/(len(cues)-1))
        elif arm=="random_rate":
            # Precommitted independent stochastic rate, no outcome-based tuning.
            rate=random.Random(seed*601_013+tick*1097+791).uniform(
                RATE_MIN,RATE_MAX)
        elif arm=="adaptive":
            # The detector uses only this realized feedback, not hidden rule.
            rate=monitor.update(success)
        elif arm=="yoked_shuffled_rate":
            # Independent of this placebo agent's observations; a rate
            # multiset from the paired adaptive counterfactual is shuffled
            # up front, preserving EXACT dose but removing temporal alignment.
            rate=yoked_rates[tick]
        if arm!="adaptive":
            # Audit-only observation; never used to select an action/rate.
            monitor.update(success)
        updates+=int(learner.learn(
            cache,1. if success else -.15,learning_rate=rate)>0)
        lr.append(rate)
        prev_correct=float(success)
    for p in range(PHASES):
        values=correct[p*PHASE_LENGTH:(p+1)*PHASE_LENGTH]
        events.append({
            "phase":p,
            "first_window":sum(values[:WINDOW])/WINDOW,
            "last_window":sum(values[-WINDOW:])/WINDOW,
            "overall":sum(values)/len(values),
            "recovery":(sum(values[-WINDOW:])-sum(values[:WINDOW]))/WINDOW,
        })
    return {
        "seed":seed,"arm":arm,"phase_windows":events,
        "mean_learning_rate":statistics.mean(lr),
        "min_learning_rate":min(lr),"max_learning_rate":max(lr),
        "weight_drift_l2":sum((a-b)**2 for a,b in
                             zip(learner.weights,genotype(seed)))**.5,
        "policy_updates":updates,
        "rate_history":lr,
        "error_monitor_peak":monitor.max_surprise,
        "transferred_memory":0,
    }


def experiment(seeds:tuple[int,...]=SEEDS)->dict:
    if not seeds or len(set(seeds))!=len(seeds) or any(
            type(s) is not int or s not in SEEDS for s in seeds):
        raise ValueError("Development-only distinct seeds required")
    rows=[]
    for seed in seeds:
        adaptive=run(seed,"adaptive")
        rates=adaptive["rate_history"][:]
        random.Random(seed*770_147+1241).shuffle(rates)
        for arm in ARMS:
            rows.append(
                adaptive if arm=="adaptive" else
                run(seed,arm,yoked_rates=tuple(rates)
                    if arm=="yoked_shuffled_rate" else None))
    idx={(r["seed"],r["arm"]):r for r in rows}
    for seed in seeds:
        a=idx[(seed,"adaptive")]["rate_history"]
        b=idx[(seed,"yoked_shuffled_rate")]["rate_history"]
        if sorted(a)!=sorted(b):
            raise AssertionError("Matched plasticity dose not conserved")
    index={(r["seed"],r["arm"]):r for r in rows}
    summary={}
    for arm in ARMS:
        records=[index[(s,arm)] for s in seeds]
        summary[arm]={
            "mean_rate":statistics.mean(r["mean_learning_rate"] for r in records),
            "mean_policy_updates":statistics.mean(r["policy_updates"] for r in records),
            "phase_windows":[{
                "phase":p,
                "first":statistics.mean(r["phase_windows"][p]["first_window"]
                                        for r in records),
                "last":statistics.mean(r["phase_windows"][p]["last_window"]
                                       for r in records),
                "recovery":statistics.mean(r["phase_windows"][p]["recovery"]
                                           for r in records),
            } for p in range(PHASES)],
        }
    def mean_reversal_recovery(arm:str)->float:
        return statistics.mean(
            summary[arm]["phase_windows"][p]["recovery"]
            for p in range(1,PHASES))
    adaptive=summary["adaptive"]
    constant=summary["constant"]
    random_arm=summary["random_rate"]
    delta_const=mean_reversal_recovery("adaptive")-mean_reversal_recovery("constant")
    delta_rand=mean_reversal_recovery("adaptive")-mean_reversal_recovery("random_rate")
    delta_yoked=mean_reversal_recovery("adaptive")-mean_reversal_recovery("yoked_shuffled_rate")
    gate={
        "better_reversal_recovery_than_constant":
            delta_const>=CRITERIA["min_recovery_delta_over_constant"],
        "better_reversal_recovery_than_yoked_shuffled_rate":
            delta_yoked>=CRITERIA["min_recovery_delta_over_random"],
        "final_reversal_accuracy_at_least_65pct":
            adaptive["phase_windows"][-1]["last"]>=
            CRITERIA["min_final_adaptive_accuracy"],
        "initial_stable_phase_gain_at_least_8pp":
            adaptive["phase_windows"][0]["recovery"]>=
            CRITERIA["min_initial_phase_gain"],
    }
    return {
        "status":"development_plasticity_diagnostic_not_final",
        "seeds":list(seeds),"arms":list(ARMS),
        "phase_length":PHASE_LENGTH,"phase_count":PHASES,
        "unannounced_rule_reversals":PHASES-1,
        "decision_oracle":False,"fitness_selection":False,
        "rules_do_not_depend_on_actions":True,
        "summary":summary,
        "adaptive_minus_constant_mean_reversal_recovery":delta_const,
        "adaptive_minus_random_mean_reversal_recovery":delta_rand,
        "adaptive_minus_dose_matched_yoked_recovery":delta_yoked,
        "rate_dose_matched":True,
        "screening_gate":gate,"advance_to_evolution":all(gate.values()),
        "rows_sha256":hashlib.sha256(json.dumps(
            rows,sort_keys=True,separators=(",",":")).encode()).hexdigest(),
        "raw_rows":rows,
    }


def main()->None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,default=Path(
        "results/m5-9/adaptive-plasticity-reversal-development.json"))
    args=parser.parse_args()
    report=experiment()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n",
                        encoding="utf-8")
    print("M5_9_ADAPTIVE_PLASTICITY "+json.dumps({
        k:report[k] for k in (
            "status","summary",
            "adaptive_minus_constant_mean_reversal_recovery",
            "adaptive_minus_random_mean_reversal_recovery",
            "adaptive_minus_dose_matched_yoked_recovery",
            "screening_gate","advance_to_evolution","rows_sha256")},
        sort_keys=True),flush=True)


if __name__=="__main__":
    main()
