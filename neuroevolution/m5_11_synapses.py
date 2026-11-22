"""M5.11 developmental synaptic structural plasticity (NOT neuron genesis).

Same fixed 8-unit, 260-possible-edge M5.6 controller across arms. All begin
with precisely 156 active synapses and identical weights, except full_dense
control, which begins with all 260 edges. The remaining synaptic *slots*
exist but are disconnected (zero effective weight) until formed.

Sparse structural interventions occur every 40 actions using only PAST
observed reward and aggregate synaptic utilization. No correct-action
oracle and no context-specific hardcoded modules. Task schedule A,B,A,C,B,
identical to M5.10, with explicit context cue but no active-context routing.
Five arms: fixed_sparse, prune_only, grow_only, grow_prune, full_dense.
Sixth arm: turnover_random with random pruning/growth and same nominal
slot budget (mechanistic control, not dose-matched for every realized run).

This model approximates structural synaptic remodeling by masking native
RNN edges. It is not an unlimited growing biological graph and does not
model the energy of synapse formation. Pruning protects long-run importance
against temporary inactivity. Growth/pruning policy is engineered, not
learned or evolved.

Development seeds 3400..3423, held separate from all prior cohorts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path

from .m5_6_online import Learner
from .m5_10_growth import CONTEXT_ORDER, CONTEXT_COUNT, TRIALS_PER_PHASE, WINDOW

SEEDS = tuple(range(3400,3424))
ARMS = ("fixed_sparse","prune_only","grow_only","grow_prune",
        "turnover_random","full_dense")
CAPACITY=260
INITIAL_ACTIVE=156
STRUCTURAL_INTERVAL=40
GROW_BATCH=12
PRUNE_BATCH=6
MAX_PRUNED_PER_EVENT=10
MIN_ACTIVE=110
WEIGHT_RATE=.04
CREDIT_DECAY=.99
ERROR_DECAY=.95
TARGET_SUCCESS=.72
# Stable global structural policy, no phase-boundary access.


def genotype(seed:int)->tuple[float,...]:
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError("Only M5.11 unused development seeds 3400..3423")
    rng=random.Random(seed*107_113+17)
    return tuple(rng.gauss(0.,.26) for _ in range(CAPACITY))


def initial_active_set(seed:int)->set[int]:
    rng=random.Random(seed*17_033+227)
    return set(rng.sample(range(CAPACITY),INITIAL_ACTIVE))


def schedule(seed:int)->tuple[tuple[int,int],...]:
    if seed not in SEEDS:
        raise ValueError("Only M5.11 development schedule")
    rng=random.Random(seed*211_001+271)
    out=[]
    for context in CONTEXT_ORDER:
        for _ in range(TRIALS_PER_PHASE//WINDOW):
            cues=[0]*(WINDOW//2)+[1]*(WINDOW//2)
            rng.shuffle(cues)
            out.extend((context,c) for c in cues)
    return tuple(out)


def rules(seed:int)->tuple[int,...]:
    if seed not in SEEDS:
        raise ValueError("Unregistered seed")
    rng=random.Random(seed*10_007+51)
    return tuple(rng.randrange(2) for _ in range(CONTEXT_COUNT))


def observed(context:int,cue:int,prev_success:float)->tuple[float,...]:
    obs=[0.]*19
    obs[cue]=1.
    obs[2+context]=1.
    obs[5]=prev_success
    return tuple(obs)


class Synapses:
    def __init__(self,seed:int,arm:str):
        self.seed=seed
        self.arm=arm
        self.active=(set(range(CAPACITY)) if arm=="full_dense"
                     else initial_active_set(seed))
        self.initial_count=len(self.active)
        self.age=[0]*CAPACITY
        self.credit=[0.]*CAPACITY
        self.recent_error=.5
        self.events=[]
        self.total_new=0
        self.total_removed=0
        self.rng=random.Random(seed*97_337+783)
        self.original=genotype(seed)
        self.agent=Learner.fresh(self.original,(5,5))
        for i in range(CAPACITY):
            if i not in self.active:
                self.agent.weights[i]=0.

    def masked_update(self,cache:tuple,success:int)->None:
        """Reward only updates active synapses; dormant weights remain zero."""
        self.agent.learn(cache,1. if success else -.15,learning_rate=WEIGHT_RATE)
        obs,prev,hidden,probs,chosen=cache
        # Credit proxy uses *local presynaptic activity* plus observed
        # outcome (not external target labels) for all active edges.
        features=list(obs)+list(prev)+[1.]
        for i in self.active:
            if i<224:
                act=abs(features[i%28])
            else:
                act=abs(hidden[(i-224)%9]) if (i-224)%9<8 else 1.
            self.credit[i]=self.credit[i]*CREDIT_DECAY + (
                .01*act*(.2+.8*float(success)))
            self.age[i]+=1
        for i in range(CAPACITY):
            if i not in self.active:
                self.agent.weights[i]=0.
        self.recent_error=ERROR_DECAY*self.recent_error+(
            1-ERROR_DECAY)*(1-float(success))

    def remodel(self,tick:int)->None:
        if (tick+1)%STRUCTURAL_INTERVAL:
            return
        if self.arm in ("fixed_sparse","full_dense"):
            return
        # No forward view or phase index; use realized EMA errors only.
        stressed=self.recent_error>1-TARGET_SUCCESS
        removed=[]
        added=[]
        can_prune=self.arm in ("prune_only","grow_prune","turnover_random")
        can_grow=self.arm in ("grow_only","grow_prune","turnover_random")
        if can_prune and len(self.active)>MIN_ACTIVE:
            # Structural consolidation proxy: preserve high accumulated
            # credit, low utility edges eligible. This is intentionally
            # simple and not guaranteed to preserve old tasks.
            amount=min(PRUNE_BATCH if self.arm!="turnover_random"
                       else GROW_BATCH, len(self.active)-MIN_ACTIVE)
            if self.arm=="turnover_random":
                removed=self.rng.sample(sorted(self.active),amount)
            else:
                ranked=sorted(self.active,key=lambda i:(
                    self.credit[i]+.025*abs(self.agent.weights[i]),i))
                removed=ranked[:amount]
            for i in removed:
                self.active.remove(i)
                self.agent.weights[i]=0.
                self.age[i]=0
                self.credit[i]=0.
            self.total_removed+=len(removed)
        # Grow only when rolling errors justify more resources, except
        # turnover placebo (growth is exogenous and blind to outcomes).
        if can_grow and (stressed or self.arm=="turnover_random"):
            candidates=sorted(set(range(CAPACITY))-self.active)
            amount=min(GROW_BATCH,len(candidates))
            added=self.rng.sample(candidates,amount)
            for i in added:
                self.active.add(i)
                # New connection initialized without inheriting learned
                # weight, prior memory, or any hidden correct answer.
                self.agent.weights[i]=self.rng.gauss(0.,.12)
                self.credit[i]=0.
                self.age[i]=0
            self.total_new+=len(added)
        if removed or added:
            self.events.append({
                "tick":tick+1,"active":len(self.active),
                "new":len(added),"removed":len(removed),
                "observed_error_ema":self.recent_error,
            })


def run(seed:int,arm:str)->dict:
    if seed not in SEEDS or arm not in ARMS:
        raise ValueError("Unregistered M5.11 development seed/arm")
    syn=Synapses(seed,arm)
    oracle_rules=rules(seed) # environment only, never passed to syn
    series=schedule(seed)
    wins=[]
    allocated_ticks=0
    prev_success=0.
    prev_context=None
    all_success=[]
    for tick,(context,cue) in enumerate(series):
        if context!=prev_context:
            prev_success=0.
            prev_context=context
        action,cache=syn.agent.action(observed(context,cue,prev_success),
            (0,1),random.Random(seed*997_007+tick*1069+21))
        if action is None:
            raise AssertionError("No legal action")
        success=int(action==(cue^oracle_rules[context]))
        all_success.append(success)
        syn.masked_update(cache,success)
        allocated_ticks+=len(syn.active)
        syn.remodel(tick)
        prev_success=float(success)
    for p,ctx in enumerate(CONTEXT_ORDER):
        values=all_success[p*TRIALS_PER_PHASE:(p+1)*TRIALS_PER_PHASE]
        wins.append({"phase":p,"context":ctx,
                     "first":sum(values[:WINDOW])/WINDOW,
                     "last":sum(values[-WINDOW:])/WINDOW,
                     "overall":sum(values)/len(values)})
    return {
        "seed":seed,"arm":arm,
        "initial_synapses":syn.initial_count,
        "final_synapses":len(syn.active),
        "new_synapses":syn.total_new,
        "pruned_synapses":syn.total_removed,
        "mean_synapses":allocated_ticks/len(series),
        "synapse_ticks":allocated_ticks,
        "overall_accuracy":statistics.mean(all_success),
        "return_A_first":wins[2]["first"],
        "return_B_first":wins[4]["first"],
        "learned_A_last":wins[0]["last"],
        "learned_B_last":wins[1]["last"],
        "A_retention":wins[2]["first"]-wins[0]["last"],
        "B_retention":wins[4]["first"]-wins[1]["last"],
        "phase_windows":wins,
        "remodeling_events":syn.events,
        "answer_sha256":hashlib.sha256(bytes(all_success)).hexdigest(),
        "routing_is_programmed":False,
        "policy_received_correct_action":False,
        "reproduction":False,
    }


def experiment(seeds:tuple[int,...]=SEEDS)->dict:
    if not seeds or len(set(seeds))!=len(seeds) or any(
            type(s) is not int or s not in SEEDS for s in seeds):
        raise ValueError("Guarded independent M5.11 development seeds")
    rows=[run(s,a) for s in seeds for a in ARMS]
    index={(r["seed"],r["arm"]):r for r in rows}
    keys=("overall_accuracy","return_A_first","return_B_first",
          "A_retention","B_retention","new_synapses","pruned_synapses",
          "mean_synapses","synapse_ticks","final_synapses")
    summary={a:{k:statistics.mean(index[(s,a)][k] for s in seeds)
                for k in keys} for a in ARMS}
    # Fair exact comparison: all non-full arms begin at 156 synapses;
    # full-density baseline controls a possible pure capacity effect.
    gate={
        "growth_prune_improves_B_return_by_10pp_over_fixed_sparse":
           summary["grow_prune"]["return_B_first"]-
           summary["fixed_sparse"]["return_B_first"]>=.10,
        "growth_prune_B_return_over_65pct":
           summary["grow_prune"]["return_B_first"]>=.65,
        "growth_prune_outperforms_random_turnover_5pp":
           summary["grow_prune"]["return_B_first"]-
           summary["turnover_random"]["return_B_first"]>=.05,
        "growth_prune_fewer_synapse_ticks_than_full":
           summary["grow_prune"]["synapse_ticks"]<
           summary["full_dense"]["synapse_ticks"],
    }
    return {
        "status":"development_sparse_synaptic_remodeling_no_genesis_no_evolution",
        "seeds":list(seeds),"arms":list(ARMS),
        "neurons_hidden_fixed":8,"possible_edges":CAPACITY,
        "initial_active_edges":INITIAL_ACTIVE,
        "contexts_are_observable":True,
        "context_router_is_explicit":False,
        "pruning_trigger_is_engineered":True,
        "real_synapse_birth_death":True,
        "metric_not_actual_energy":"synapse-ticks",
        "summary":summary,"screening_gate":gate,
        "advance_to_evolution":all(gate.values()),
        "raw_rows":rows,
        "rows_sha256":hashlib.sha256(json.dumps(
            rows,sort_keys=True,separators=(",",":")).encode()).hexdigest(),
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,default=Path(
        "results/m5-11/synaptic-remodeling-development.json"))
    args=parser.parse_args()
    result=experiment()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",
                        encoding="utf-8")
    print("M5_11_SYNAPSES "+json.dumps({
        k:result[k] for k in ("status","summary","screening_gate",
             "advance_to_evolution","rows_sha256")},sort_keys=True),flush=True)


if __name__=="__main__":
    main()
