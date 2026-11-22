"""M5.10: individual brain growth vs preallocated equivalent modular capacity.

No reproduction, no adaptive mutation, no environmental change oracle.
Development task: contexts A/B/A/C/B with binary cues and hidden per-context
action rules. Context features ARE visible (real exogenous sensory context),
but hidden rule and correct action are NOT. This tests CONTEXTUAL RETENTION,
not detection of unmarked hidden-rule reversals or causal planning.

Conditions:
  small_shared: 1 native 8-hidden-unit 260-weight M5.6 recurrent expert.
  large_preallocated: 3 native 8-unit experts (24 units, 780 weights)
    instantiated from birth, context-routed by the same fixed feature rule.
  growing_modules: begin with 1 native expert (8 units, 260 weights);
    acquire additional experts on encounter with a new observable context,
    up to exactly 3 modules; recall original expert when context recurs.

Crucial: the routing is explicit programmer-specified context lookup in BOTH
multimodule arms. Claim ONLY growth mechanics and retention in this visible-
context arena; not emerging novel routing or learned change detection.
The large and growing experts have exactly the same seeded initial genomes,
action random sequences, learning budgets and per-context recurrent state;
their trajectories SHOULD be bit-identical except allocation cost.

The body-wide size/cost proxy is *allocated hidden-unit ticks*. This is
not an energy-conserving physical ecology experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from pathlib import Path

from .m5_6_online import Learner

DEV_SEEDS=tuple(range(3300,3324))
ARMS=("small_shared","large_preallocated","growing_modules")
CONTEXT_ORDER=(0,1,0,2,1)
CONTEXT_COUNT=3
MODULE_HIDDEN=8
MODULE_WEIGHTS=260
MAX_MODULES=3
TRIALS_PER_PHASE=160
WINDOW=40
# Original single-step REINFORCE: no teacher/answer label and no oracle reward.
LR=.04
SUCCESS_REWARD=1.
FAIL_REWARD=-.15


def initial_genotype(seed:int, context:int)->tuple[float,...]:
    if type(seed) is not int or seed not in DEV_SEEDS or context not in range(CONTEXT_COUNT):
        raise ValueError("Only M5.10 registered development seeds and contexts")
    rng=random.Random(seed*127_003+context*8819+17)
    return tuple(rng.gauss(0.,.26) for _ in range(MODULE_WEIGHTS))


def context_rules(seed:int)->tuple[int,...]:
    if seed not in DEV_SEEDS:
        raise ValueError("Unregistered M5.10 seed")
    rng=random.Random(seed*10_007+51)
    # Balanced across contexts with no deterministic relationship between
    # perceived context index and correct answer. Draw independent rule bits.
    return tuple(rng.randrange(2) for _ in range(CONTEXT_COUNT))


def cue_schedule(seed:int)->tuple[int,...]:
    if seed not in DEV_SEEDS:
        raise ValueError("Unregistered seed")
    rng=random.Random(seed*211_001+271)
    out=[]
    for _ in range(len(CONTEXT_ORDER)*TRIALS_PER_PHASE//WINDOW):
        values=[0]*(WINDOW//2)+[1]*(WINDOW//2)
        rng.shuffle(values)
        out.extend(values)
    return tuple(out)


def observation(context:int,cue:int,previous_feedback:float)->tuple[float,...]:
    if context not in range(CONTEXT_COUNT) or cue not in (0,1):
        raise ValueError("Invalid exogenous sensor signals")
    values=[0.]*19
    values[cue]=1.
    values[2+context]=1.     # visible context, no rule bit
    values[5]=previous_feedback
    return tuple(values)


def run(seed:int,arm:str)->dict:
    if type(seed) is not int or seed not in DEV_SEEDS or arm not in ARMS:
        raise ValueError("M5.10 development seed / arm only")
    # Start-of-life initial brain. Preallocated has equal peak to growing.
    initial=(0,1,2) if arm=="large_preallocated" else (0,)
    modules={k:Learner.fresh(initial_genotype(seed,k),(5,5))
             for k in initial}
    assigned={k:k for k in initial} if arm=="large_preallocated" else {0:0}
    rules=context_rules(seed)
    cues=cue_schedule(seed)
    correct=[]
    phase_metrics=[]
    births=[]
    accessed=[]
    sizes=[]
    used_per_context={}
    active_context=None
    previous_feedback=0.
    for tick,cue in enumerate(cues):
        phase=tick//TRIALS_PER_PHASE
        context=CONTEXT_ORDER[phase]
        if arm=="growing_modules" and context not in assigned:
            if len(modules)>=MAX_MODULES:
                raise AssertionError("Exceeded maximum neuronal capacity")
            # Growth is triggered ONLY by first appearance of an observable
            # context feature; the hidden rule is not consulted.
            modules[context]=Learner.fresh(initial_genotype(seed,context),(5,5))
            assigned[context]=context
            births.append({"tick":tick,"observed_context":context,
                           "hidden_neurons_added":MODULE_HIDDEN})
        expert_id=0 if arm=="small_shared" else assigned[context]
        expert=modules[expert_id]
        if active_context!=context:
            # Clear common previous-feedback sensory channel on ANY observed
            # context switch. Do not reset or modify stored expert weights.
            previous_feedback=0.
            active_context=context
        obs=observation(context,cue,previous_feedback)
        action,cache=expert.action(
            obs,(0,1),random.Random(seed*997_007+tick*1069+21))
        if action is None:
            raise AssertionError("Native learner failed to act")
        outcome=int(action==(cue^rules[context]))
        correct.append(outcome)
        expert.learn(cache,SUCCESS_REWARD if outcome else FAIL_REWARD,
                     learning_rate=LR)
        previous_feedback=float(outcome)
        used_per_context[context]=expert_id
        accessed.append(expert_id)
        sizes.append(len(modules)*MODULE_HIDDEN)
    for phase,context in enumerate(CONTEXT_ORDER):
        start=phase*TRIALS_PER_PHASE
        values=correct[start:start+TRIALS_PER_PHASE]
        phase_metrics.append({
            "phase":phase,
            "context":context,
            "first_window":sum(values[:WINDOW])/WINDOW,
            "last_window":sum(values[-WINDOW:])/WINDOW,
            "gain":(sum(values[-WINDOW:])-sum(values[:WINDOW]))/WINDOW,
            "accuracy":sum(values)/len(values),
        })
    return {
        "seed":seed,"arm":arm,"phase_metrics":phase_metrics,
        "rules_sha256":hashlib.sha256(bytes(rules)).hexdigest(),
        "overall_accuracy":statistics.mean(correct),
        "return_A_accuracy":phase_metrics[2]["first_window"],
        "initial_A_accuracy":phase_metrics[0]["first_window"],
        "learned_A_accuracy":phase_metrics[0]["last_window"],
        "A_recall_ratio":phase_metrics[2]["first_window"]-
                         phase_metrics[0]["last_window"],
        "return_B_accuracy":phase_metrics[4]["first_window"],
        "learned_B_accuracy":phase_metrics[1]["last_window"],
        "B_recall_ratio":phase_metrics[4]["first_window"]-
                         phase_metrics[1]["last_window"],
        "new_context_C_final":phase_metrics[3]["last_window"],
        "modules_at_birth":len(initial),
        "modules_at_end":len(modules),
        "peak_allocated_hidden_units":max(sizes),
        "allocated_hidden_unit_ticks":sum(sizes),
        "new_module_events":births,
        "answer_trace_sha256":hashlib.sha256(bytes(correct)).hexdigest(),
        "active_expert_trace_sha256":hashlib.sha256(bytes(accessed)).hexdigest(),
        "inherited_learning_or_memory":False,
        "fitness_selection":False,
    }


def experiment(seeds:tuple[int,...]=DEV_SEEDS)->dict:
    if not seeds or len(set(seeds))!=len(seeds) or any(
            type(s) is not int or s not in DEV_SEEDS for s in seeds):
        raise ValueError("M5.10 only distinct development seeds")
    rows=[run(seed,arm) for seed in seeds for arm in ARMS]
    idx={(r["seed"],r["arm"]):r for r in rows}
    # The fair head-to-head: preallocation and growth share EXACTLY
    # the same context-specific models, route and experience.
    equivalence={
        "all_seed_answer_traces_equal":all(
            idx[(seed,"large_preallocated")]["answer_trace_sha256"]==
            idx[(seed,"growing_modules")]["answer_trace_sha256"]
            for seed in seeds),
        "all_seed_expert_routing_equal":all(
            idx[(seed,"large_preallocated")]["active_expert_trace_sha256"]==
            idx[(seed,"growing_modules")]["active_expert_trace_sha256"]
            for seed in seeds)
    }
    summary={}
    for arm in ARMS:
        records=[idx[(s,arm)] for s in seeds]
        keys=("overall_accuracy","learned_A_accuracy","return_A_accuracy",
              "A_recall_ratio","learned_B_accuracy","return_B_accuracy",
              "B_recall_ratio","new_context_C_final",
              "allocated_hidden_unit_ticks",
              "modules_at_birth","modules_at_end")
        summary[arm]={k:statistics.mean(r[k] for r in records) for k in keys}
        summary[arm]["mean_phase_accuracy"]=[
            statistics.mean(r["phase_metrics"][p]["accuracy"] for r in records)
            for p in range(len(CONTEXT_ORDER))]
        summary[arm]["mean_phase_gain"]=[
            statistics.mean(r["phase_metrics"][p]["gain"] for r in records)
            for p in range(len(CONTEXT_ORDER))]
    # Development-only progression gate checks that growth helps versus
    # capacity-limited shared network AND is not misrepresented as better
    # than identical preallocation. Never blindly advance to evolution.
    gate={
        "growing_return_A_at_least_65pct":
            summary["growing_modules"]["return_A_accuracy"]>=.65,
        "growing_return_B_at_least_65pct":
            summary["growing_modules"]["return_B_accuracy"]>=.65,
        "growing_beats_small_return_A_by_10pp":
            summary["growing_modules"]["return_A_accuracy"]-
            summary["small_shared"]["return_A_accuracy"]>=.10,
        "large_and_growing_same_observable_behavior":
            all(equivalence.values()),
    }
    return {
        "status":"development_contextual_neural_growth_not_evolution",
        "seeds":list(seeds),"arms":list(ARMS),
        "context_order":list(CONTEXT_ORDER),
        "hidden_rule_is_visible_to_organism":False,
        "observable_context_label":True,
        "context_routing_is_engineered":True,
        "reward_supervision_is_action_label":False,
        "phase_trials":TRIALS_PER_PHASE,
        "max_hidden_units":MAX_MODULES*MODULE_HIDDEN,
        "max_modular_weights":MAX_MODULES*MODULE_WEIGHTS,
        "equivalence_control":equivalence,
        "summary":summary,"screening_gate":gate,
        "advance_to_evolution":all(gate.values()),
        "raw_rows":rows,
        "rows_sha256":hashlib.sha256(json.dumps(
            rows,sort_keys=True,separators=(",",":")).encode()).hexdigest(),
    }


def main()->None:
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,default=Path(
        "results/m5-10/contextual-growth-development.json"))
    args=p.parse_args()
    result=experiment()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",
                        encoding="utf-8")
    print("M5_10_GROWTH "+json.dumps({
        k:result[k] for k in (
            "status","summary","equivalence_control",
            "screening_gate","advance_to_evolution","rows_sha256")},
        sort_keys=True),flush=True)


if __name__=="__main__":
    main()
