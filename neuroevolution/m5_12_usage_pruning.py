"""M5.12 — Probabilistic pruning of underused synapses.

Development-only experiment, NOT neuron death and not yet organism
reproduction. Fixed eight-unit/260-slot native M5.6 RNN; each non-dense
condition initially has SAME 200 active synapses and identical weights.
Every 40 decisions prune exactly four edges until 120 remain.

A stochastic weighted sample selects weakly used edges preferentially.
Usage is measured by synapse-local presynaptic/post-synaptic activation
magnitudes, never the hidden correct answer. "Consolidated" arm also
tracks slow lifetime activation to protect dormant older connections.

Controls: no pruning at initial 200, uniform random matched dose, low
weight magnitude matched dose, fast usage matched dose, fast+slow lifetime
usage matched dose, full fixed 260. All conditions face same task
A->B->A->C->B and same 800 actions, same cue and context observations,
same random action tickets and learning budget.

Counts of deleted edges and their timing are identical in all pruning
arms BY CONSTRUCTION; no future state/teacher label enters the policy.
Use independent new development seeds 3500..3523.
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
from .m5_10_growth import CONTEXT_ORDER, TRIALS_PER_PHASE, WINDOW

SEEDS=tuple(range(3500,3524))
ARMS=("static_200","random_prune","magnitude_prune",
      "usage_prune","usage_consolidated","full_260")
TOTAL_EDGES=260
INITIAL_EDGES=200
MIN_EDGES=120
REMOVALS_PER_EVENT=4
PRUNE_PERIOD=40
FAST_DECAY=.94
SLOW_DECAY=.996
PROTECTION_FRACTION=.50
TEMPERATURE=.025
LEARNING_RATE=.04


def initial_genome(seed:int)->tuple[float,...]:
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError("M5.12 uses only distinct development seeds 3500..3523")
    rng=random.Random(seed*107_113+17)
    return tuple(rng.gauss(0.,.26) for _ in range(TOTAL_EDGES))


def initial_mask(seed:int)->set[int]:
    rng=random.Random(seed*17_033+227)
    return set(rng.sample(range(TOTAL_EDGES),INITIAL_EDGES))


def task(seed:int)->tuple[tuple[int,int],...]:
    if seed not in SEEDS:
        raise ValueError("Unregistered M5.12 development seed")
    rng=random.Random(seed*211_001+271)
    result=[]
    for ctx in CONTEXT_ORDER:
        for _ in range(TRIALS_PER_PHASE//WINDOW):
            binary=[0]*(WINDOW//2)+[1]*(WINDOW//2)
            rng.shuffle(binary)
            result.extend((ctx,c) for c in binary)
    return tuple(result)


def rules(seed:int)->tuple[int,...]:
    rng=random.Random(seed*10_007+51)
    return tuple(rng.randrange(2) for _ in range(3))


def observation(context:int, cue:int, previous:int)->tuple[float,...]:
    obs=[0.]*19
    obs[cue]=1.
    obs[2+context]=1.
    obs[5]=float(previous)
    return tuple(obs)


def sample_prunes(active:set[int], importance:list[float], n:int,
                  rng:random.Random)->list[int]:
    """Select without replacement, inverse importance, no deterministic cutoff."""
    pool=sorted(active)
    selected=[]
    for _ in range(n):
        # Shift by minimum for robust exponent; temperature is declared.
        low=min(importance[i] for i in pool)
        relative=[math.exp(-min(60.,(importance[i]-low)/TEMPERATURE))
                  for i in pool]
        chosen=rng.choices(pool,weights=relative,k=1)[0]
        selected.append(chosen)
        pool.remove(chosen)
    return selected


class SparseBrain:
    def __init__(self,seed:int,arm:str):
        self.seed=seed
        self.arm=arm
        self.genome=initial_genome(seed)
        self.active=set(range(TOTAL_EDGES)) if arm=="full_260" else initial_mask(seed)
        self.agent=Learner.fresh(self.genome,(5,5))
        self.fast=[0.]*TOTAL_EDGES
        self.slow=[0.]*TOTAL_EDGES
        self.events=[]
        self.rng=random.Random(seed*97_337+783)
        self._mask()

    def _mask(self):
        for edge in range(TOTAL_EDGES):
            if edge not in self.active:
                self.agent.weights[edge]=0.

    def observe_and_learn(self,cache:tuple, success:bool):
        """Compute local coactivation from already observed neural state."""
        obs,prev,hidden,probs,chosen=cache
        pre=list(obs)+list(prev)+[1.]
        for i in self.active:
            if i<224:
                source=abs(pre[i%28])
                post=abs(hidden[i//28])
            else:
                a=(i-224)//9
                j=(i-224)%9
                source=abs(hidden[j]) if j<8 else 1.
                # Policy-output strength of the output unit is proxied by
                # its current actual selection probability; no labels.
                post=probs.get(a,0.)
            usage=source*post
            self.fast[i]=FAST_DECAY*self.fast[i]+(1-FAST_DECAY)*usage
            self.slow[i]=SLOW_DECAY*self.slow[i]+(1-SLOW_DECAY)*usage
        self.agent.learn(cache,1. if success else -.15,
                         learning_rate=LEARNING_RATE)
        self._mask()

    def prune(self,tick:int)->None:
        if (tick+1)%PRUNE_PERIOD or self.arm in ("static_200","full_260"):
            return
        n=min(REMOVALS_PER_EVENT,len(self.active)-MIN_EDGES)
        if n<=0:
            return
        if self.arm=="random_prune":
            removed=self.rng.sample(sorted(self.active),n)
        else:
            if self.arm=="magnitude_prune":
                importance=[abs(w) for w in self.agent.weights]
            elif self.arm=="usage_prune":
                importance=self.fast
            elif self.arm=="usage_consolidated":
                importance=[(1-PROTECTION_FRACTION)*x+
                            PROTECTION_FRACTION*y
                            for x,y in zip(self.fast,self.slow)]
            else:
                raise AssertionError("Unknown pruning strategy")
            removed=sample_prunes(self.active,importance,n,self.rng)
        for edge in removed:
            self.active.remove(edge)
            self.agent.weights[edge]=0.
            self.fast[edge]=0.
            self.slow[edge]=0.
        self.events.append({"at_tick":tick+1,"pruned":len(removed),
                            "edges":removed,"remaining":len(self.active)})


def run(seed:int,arm:str)->dict:
    if seed not in SEEDS or arm not in ARMS:
        raise ValueError("Unknown M5.12 development seed/arm")
    brain=SparseBrain(seed,arm)
    observed_task=task(seed)
    hidden_rules=rules(seed) # environment only; NEVER given to learner
    previous_context=None
    previous_feedback=0
    successes=[]
    occupancy=0
    for tick,(context,cue) in enumerate(observed_task):
        if context!=previous_context:
            previous_feedback=0
            previous_context=context
        action,cache=brain.agent.action(
            observation(context,cue,previous_feedback),
            (0,1),random.Random(seed*997_007+tick*1069+21))
        if action is None or cache is None:
            raise AssertionError("No legal action")
        outcome=int(action==(cue ^ hidden_rules[context]))
        successes.append(outcome)
        brain.observe_and_learn(cache,bool(outcome))
        occupancy+=len(brain.active)
        brain.prune(tick)
        previous_feedback=outcome
    windows=[]
    for phase,context in enumerate(CONTEXT_ORDER):
        data=successes[phase*TRIALS_PER_PHASE:(phase+1)*TRIALS_PER_PHASE]
        windows.append({"phase":phase,"context":context,
                        "first":sum(data[:WINDOW])/WINDOW,
                        "last":sum(data[-WINDOW:])/WINDOW,
                        "accuracy":sum(data)/len(data)})
    return {
        "seed":seed,"arm":arm,
        "overall":statistics.mean(successes),
        "initial_A_late":windows[0]["last"],
        "return_A_early":windows[2]["first"],
        "initial_B_late":windows[1]["last"],
        "return_B_early":windows[4]["first"],
        "A_retention":windows[2]["first"]-windows[0]["last"],
        "B_retention":windows[4]["first"]-windows[1]["last"],
        "mean_active_synapses":occupancy/len(successes),
        "initial_active_synapses":TOTAL_EDGES if arm=="full_260" else INITIAL_EDGES,
        "final_active_synapses":len(brain.active),
        "deleted_synapses":sum(e["pruned"] for e in brain.events),
        "pruning_events":brain.events,
        "phase_windows":windows,
        "outcomes_sha256":hashlib.sha256(bytes(successes)).hexdigest(),
        "structural_changes_use_future_rules":False,
        "reproduction":False,
    }


def experiment(seeds:tuple[int,...]=SEEDS)->dict:
    if not seeds or len(set(seeds))!=len(seeds) or any(
            type(s) is not int or s not in SEEDS for s in seeds):
        raise ValueError("Only unique M5.12 development seeds")
    rows=[run(s,a) for s in seeds for a in ARMS]
    idx={(r["seed"],r["arm"]):r for r in rows}
    for s in seeds:
        controls=[idx[(s,a)] for a in ARMS if a.endswith("_prune") or a=="usage_consolidated"]
        signatures=[tuple((e["at_tick"],e["pruned"],e["remaining"])
                          for e in row["pruning_events"]) for row in controls]
        if len(set(signatures))!=1:
            raise AssertionError("Pruning arms have unequal dose/timing")
    fields=("overall","return_A_early","return_B_early",
            "A_retention","B_retention","mean_active_synapses",
            "deleted_synapses")
    summary={arm:{key:statistics.mean(idx[(s,arm)][key] for s in seeds)
                  for key in fields} for arm in ARMS}
    gate={
        "usage_recall_B_5pp_over_random":
            summary["usage_consolidated"]["return_B_early"]-
            summary["random_prune"]["return_B_early"]>=.05,
        "usage_recall_B_5pp_over_static":
            summary["usage_consolidated"]["return_B_early"]-
            summary["static_200"]["return_B_early"]>=.05,
        "usage_global_accuracy_95pct_of_static":
            summary["usage_consolidated"]["overall"]>=
            .95*summary["static_200"]["overall"],
        "uses_fewer_synapses_than_static":
            summary["usage_consolidated"]["mean_active_synapses"]<
            summary["static_200"]["mean_active_synapses"],
    }
    return {
        "status":"M5.12_development_probabilistic_synaptic_pruning",
        "seeds":list(seeds),"arms":list(ARMS),
        "possible_synaptic_slots":TOTAL_EDGES,
        "neurons_added":0,"genetic_selection":False,
        "pruning_quota_per_interval":REMOVALS_PER_EVENT,
        "pruning_period":PRUNE_PERIOD,
        "matched_pruning_dose_and_timing":True,
        "context_identity_sensory_visible":True,
        "summary":summary,"screening_gate":gate,
        "advance_to_evolution":all(gate.values()),
        "raw_rows":rows,
        "rows_sha256":hashlib.sha256(json.dumps(
            rows,sort_keys=True,separators=(",",":")).encode()).hexdigest(),
    }


def main()->None:
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,default=Path(
        "results/m5-12/usage-pruning-development.json"))
    args=p.parse_args()
    r=experiment()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(r,indent=2,sort_keys=True)+"\n",
                        encoding="utf-8")
    print("M5_12_USAGE_PRUNING "+json.dumps({
        k:r[k] for k in ("status","summary","screening_gate",
                        "advance_to_evolution","rows_sha256")},
        sort_keys=True),flush=True)


if __name__=="__main__":
    main()
