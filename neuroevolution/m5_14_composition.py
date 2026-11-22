"""M5.14 development: combine independently learned primitives in a new task.

Novel compositional test, online synapse birth and optional pruning. Native
8-hidden-unit M5.6 RNN with at most 260 possible edges; NO neuron growth,
open-ended graph, genetic reproduction or claim of autonomous task invention.

Pretrain XOR operator (P); learn A(x) and B(y), each with an independently
hidden rule bit. A novel task C has target A(x) XOR B(y). Observation carries
the same operator flag encountered during P, the A/B operand-role indicators
encountered separately during A/B, and a newly visible C-context identity.
The policy receives NO target labels, hidden rule bits or transition signals.
Because task grammar/role flags are designed by experimenters, transfer is
only within this structured task family, not open-world compositional thought.

Frozen zero-shot C test PRECEDES any feedback-based learning about C; hidden
state may evolve on test observations but synapses and structural masks never
change. C-learning performance is scored separately. Afterward probe old
A/B without learning to check catastrophic forgetting.

Arms:
 fixed_sparse             156 active slots and synaptic weight learning only
 full_dense               260 active from birth, size control
 scheduled_birth          add eight dormant edges at preset training ticks
 adaptive_birth           add eight high-candidate-score dormant edges if
                          consecutive feedback windows indicate no progress
 yoked_random_locations   EXACT birth event timing/count copied from matched
                          adaptive arm, but randomly selected edge identities
 adaptive_birth_prune     same adaptive birth + credit-based consolidation/
                          pruning four low-credit active edges per 80 steps

24 new DEV seeds 3700..3723, disjoint from all earlier cohorts.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import statistics
from pathlib import Path

from .m5_6_online import Learner

SEEDS=tuple(range(3700,3724))
ARMS=("fixed_sparse","full_dense","scheduled_birth","adaptive_birth",
      "yoked_random_locations","adaptive_birth_prune")
CAPACITY=260
INITIAL_ACTIVE=156
GROW_COUNT=8
PRUNE_COUNT=4
MIN_ACTIVE=120
LR=.04
WINDOW=40
# Task grammar, duration, learning-enabled flag.
PHASES=(("P",160,True),("A",160,True),("B",160,True),
        ("A",40,True),("B",40,True),("C",80,False),
        ("C",160,True),("A",40,False),("B",40,False))
TRAINING_STEPS=sum(n for _,n,learn in PHASES if learn)
TOTAL_STEPS=sum(n for _,n,_ in PHASES)
GATE={
    "zero_shot_above_chance":.60,
    "zero_shot_advantage_over_sparse":.05,
    "C_learned_late_absolute":.65,
    "old_A_return_accuracy":.65,
    "old_B_return_accuracy":.65,
}


def starter(seed:int)->tuple[float,...]:
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError("M5.14 unique development seeds 3700..3723 only")
    rng=random.Random(seed*107_113+17)
    return tuple(rng.gauss(0.,.26) for _ in range(CAPACITY))


def start_edges(seed:int)->set[int]:
    if seed not in SEEDS:
        raise ValueError("Invalid seed")
    return set(random.Random(seed*17_033+227).sample(
        range(CAPACITY),INITIAL_ACTIVE))


def hidden_rules(seed:int)->tuple[int,int]:
    if seed not in SEEDS:raise ValueError("Invalid seed")
    rng=random.Random(seed*10_007+51)
    return (rng.randrange(2),rng.randrange(2))


def curriculum(seed:int)->tuple[tuple[str,int,int,bool],...]:
    if seed not in SEEDS:raise ValueError("Invalid seed")
    rng=random.Random(seed*211_001+271)
    trials=[]
    for task,n,learn in PHASES:
        for _ in range(n//WINDOW):
            # All four (x,y) combinations appear equally often.
            pairs=[(x,y) for x in (0,1) for y in (0,1) for _ in range(WINDOW//4)]
            rng.shuffle(pairs)
            trials.extend((task,x,y,learn) for x,y in pairs)
    return tuple(trials)


def target(task:str,x:int,y:int,hidden:tuple[int,int])->int:
    """Simulator-only hidden correct answer, never agent observation."""
    if task=="P":return x^y
    if task=="A":return x^hidden[0]
    if task=="B":return y^hidden[1]
    if task=="C":return (x^hidden[0])^(y^hidden[1])
    raise ValueError("Unknown task")


def sensory(task:str,x:int,y:int,previous_feedback:int)->tuple[float,...]:
    if task not in ("P","A","B","C") or x not in (0,1) or y not in (0,1):
        raise ValueError("Invalid task signals")
    obs=[0.]*19
    obs[("P","A","B","C").index(task)]=1.
    obs[4+x]=1.
    obs[6+y]=1.
    obs[8]=float(previous_feedback)
    obs[9]=float(task in ("P","C"))  # operator seen during P
    obs[10]=float(task in ("A","C")) # reusable A operand role
    obs[11]=float(task in ("B","C")) # reusable B operand role
    return tuple(obs)


class GrowingBrain:
    def __init__(self,seed:int,arm:str,*,yoked_plan=None):
        self.seed=seed
        self.arm=arm
        self.agent=Learner.fresh(starter(seed),(5,5))
        self.active=(set(range(CAPACITY)) if arm=="full_dense"
                     else start_edges(seed))
        self.birth_rng=random.Random(seed*97_337+783)
        self.random_location_rng=random.Random(seed*873_331+199)
        self.opportunity=[0.]*CAPACITY
        self.fast=[0.]*CAPACITY
        self.slow=[0.]*CAPACITY
        self.recent_blocks=[]
        self.outcomes_for_block=[]
        self.train_clock=0
        self.events=[]
        self.yoked_plan=yoked_plan or {}
        self._mask()

    def _mask(self):
        for i in range(CAPACITY):
            if i not in self.active:
                self.agent.weights[i]=0.

    def _coactivity(self,cache):
        obs,prev,hidden,probs,selected=cache
        inputs=list(obs)+list(prev)+[1.]
        result=[]
        for i in range(CAPACITY):
            if i<224:
                result.append(abs(inputs[i%28])*abs(hidden[i//28]))
            else:
                action=(i-224)//9
                j=(i-224)%9
                source=abs(hidden[j]) if j<8 else 1.
                result.append(source*probs.get(action,0.))
        return result

    def learn(self,cache,success:int):
        coactive=self._coactivity(cache)
        for i,activity in enumerate(coactive):
            # Prospective connection opportunity is observed locally,
            # including potential contacts that are still dormant.
            self.opportunity[i]=.985*self.opportunity[i]+.015*(
                activity*(.25+.75*float(success)))
            if i in self.active:
                self.fast[i]=.95*self.fast[i]+.05*activity
                self.slow[i]=.995*self.slow[i]+.005*activity
        self.agent.learn(cache,1. if success else -.15,
                         learning_rate=LR)
        self._mask()
        self.outcomes_for_block.append(success)
        self.train_clock+=1

    def _grow(self,tick:int,count:int,*,random_locations:bool=False):
        available=sorted(set(range(CAPACITY))-self.active)
        n=min(count,len(available))
        if n<=0:return
        if random_locations:
            indices=self.random_location_rng.sample(available,n)
        else:
            # Local coactivity/feedback-based candidate formation, NOT
            # an edge gradient or causal importance estimate.
            indices=sorted(available,key=lambda i:(
                -self.opportunity[i],i))[:n]
        for i in indices:
            self.active.add(i)
            self.agent.weights[i]=self.birth_rng.gauss(0.,.12)
            self.fast[i]=0.
            self.slow[i]=0.
        self.events.append({"tick":tick+1,"kind":"birth","count":n,
                            "active":len(self.active),"edges":indices})

    def _prune(self,tick:int):
        n=min(PRUNE_COUNT,len(self.active)-MIN_ACTIVE)
        if n<=0:return
        ranked=sorted(self.active,key=lambda i:(
            .35*self.fast[i]+.65*self.slow[i]+.015*abs(self.agent.weights[i]),i))
        chosen=ranked[:n]
        for i in chosen:
            self.active.remove(i)
            self.agent.weights[i]=0.
            self.fast[i]=0.
            self.slow[i]=0.
        self.events.append({"tick":tick+1,"kind":"prune","count":n,
                            "active":len(self.active),"edges":chosen})

    def remodel(self,tick:int,learning_enabled:bool):
        if not learning_enabled:return
        # Both histories and event triggers read only PREVIOUS outcomes.
        # Do not receive task identity, task boundaries or correct labels.
        if self.train_clock%WINDOW==0:
            block=sum(self.outcomes_for_block)/len(self.outcomes_for_block)
            self.recent_blocks.append(block)
            self.outcomes_for_block.clear()
            if len(self.recent_blocks)>=2:
                prior,now=self.recent_blocks[-2:]
                stalled=(now<.68 and now-prior<.075)
            else:stalled=False
            if self.arm in ("adaptive_birth","adaptive_birth_prune") and stalled:
                self._grow(tick,GROW_COUNT)
            elif self.arm=="scheduled_birth" and self.train_clock%80==0:
                self._grow(tick,GROW_COUNT)
            elif self.arm=="yoked_random_locations":
                n=self.yoked_plan.get(tick+1,0)
                if n:self._grow(tick,n,random_locations=True)
        if self.arm=="adaptive_birth_prune" and self.train_clock%80==0:
            self._prune(tick)


def run(seed:int,arm:str,*,yoked_plan=None)->dict:
    if type(seed) is not int or seed not in SEEDS or arm not in ARMS:
        raise ValueError("Only registered M5.14 development seed/arm")
    if arm=="yoked_random_locations" and yoked_plan is None:
        raise ValueError("A valid yoked event plan is required")
    if arm!="yoked_random_locations" and yoked_plan is not None:
        raise ValueError("Yoking only to placebo control")
    brain=GrowingBrain(seed,arm,yoked_plan=yoked_plan)
    rules=hidden_rules(seed)
    samples=curriculum(seed)
    prev_task=None
    previous_feedback=0
    correct=[]
    phase_scores=[]
    occupancy=0
    frozen_checks=[]
    pos=0
    for task,n,learning_enabled in PHASES:
        prev_feedback=0
        frozen_before=(tuple(brain.agent.weights),frozenset(brain.active))
        phase=[]
        for _ in range(n):
            tick=pos
            listed_task,x,y,flag=samples[tick]
            if listed_task!=task or flag!=learning_enabled:
                raise AssertionError("Out of alignment with curriculum")
            action,cache=brain.agent.action(
                sensory(task,x,y,previous_feedback),(0,1),
                random.Random(seed*997_007+tick*1069+21))
            if action is None or cache is None:raise AssertionError("No action")
            # Environment knows target; learner receives just own reward.
            success=int(action==target(task,x,y,rules))
            correct.append(success)
            phase.append(success)
            if learning_enabled:
                brain.learn(cache,success)
                brain.remodel(tick,True)
            occupancy+=len(brain.active)
            previous_feedback=success
            pos+=1
        if not learning_enabled:
            is_frozen=(frozen_before[0]==tuple(brain.agent.weights) and
                       frozen_before[1]==frozenset(brain.active))
            frozen_checks.append(is_frozen)
            if not is_frozen:raise AssertionError("Probe changed synapses")
        phase_scores.append({
            "task":task,"learning_enabled":learning_enabled,
            "n":n,"first":sum(phase[:40])/40,
            "last":sum(phase[-40:])/40,
            "accuracy":sum(phase)/len(phase),
        })
        prev_task=task
    if pos!=TOTAL_STEPS:raise AssertionError("Incomplete curriculum")
    count_birth=sum(e["count"] for e in brain.events if e["kind"]=="birth")
    count_prune=sum(e["count"] for e in brain.events if e["kind"]=="prune")
    return {
        "seed":seed,"arm":arm,"phase_scores":phase_scores,
        "P_operator_late":phase_scores[0]["last"],
        "A_learned_late":phase_scores[1]["last"],
        "B_learned_late":phase_scores[2]["last"],
        "C_zero_shot_frozen":phase_scores[5]["accuracy"],
        "C_acquired_first":phase_scores[6]["first"],
        "C_acquired_last":phase_scores[6]["last"],
        "A_return_frozen":phase_scores[7]["accuracy"],
        "B_return_frozen":phase_scores[8]["accuracy"],
        "overall_accuracy":statistics.mean(correct),
        "initial_synapses":CAPACITY if arm=="full_dense" else INITIAL_ACTIVE,
        "final_synapses":len(brain.active),
        "mean_active_synapses":occupancy/TOTAL_STEPS,
        "synapses_born":count_birth,"synapses_pruned":count_prune,
        "events":brain.events,
        "all_probe_synapses_unchanged":all(frozen_checks),
        "outcomes_sha256":hashlib.sha256(bytes(correct)).hexdigest(),
        "rule_bits_hashed":hashlib.sha256(bytes(rules)).hexdigest(),
        "inherited_individual_learning":False,
    }


def experiment(seeds:tuple[int,...]=SEEDS)->dict:
    if not seeds or len(set(seeds))!=len(seeds) or any(
            type(s) is not int or s not in SEEDS for s in seeds):
        raise ValueError("Unique independent M5.14 development seeds")
    rows=[]
    for seed in seeds:
        adaptive=run(seed,"adaptive_birth")
        plan={e["tick"]:e["count"] for e in adaptive["events"]
              if e["kind"]=="birth"}
        for arm in ARMS:
            r=(adaptive if arm=="adaptive_birth" else
               run(seed,arm,yoked_plan=plan
                   if arm=="yoked_random_locations" else None))
            rows.append(r)
        yoked=next(r for r in rows[-len(ARMS):]
                   if r["arm"]=="yoked_random_locations")
        yoke_events=[(e["tick"],e["count"]) for e in yoked["events"]
                     if e["kind"]=="birth"]
        if yoke_events!=sorted(plan.items()):
            raise AssertionError("Identical synapse birth dose/timing required")
        if yoked["mean_active_synapses"]!=adaptive["mean_active_synapses"]:
            raise AssertionError("Matched growth occupancy required")
    idx={(r["seed"],r["arm"]):r for r in rows}
    summary={}
    keys=("P_operator_late","A_learned_late","B_learned_late",
          "C_zero_shot_frozen","C_acquired_first","C_acquired_last",
          "A_return_frozen","B_return_frozen","overall_accuracy",
          "synapses_born","synapses_pruned","mean_active_synapses")
    for arm in ARMS:
        summary[arm]={k:statistics.mean(idx[(seed,arm)][k] for seed in seeds)
                      for k in keys}
    main=summary["adaptive_birth_prune"]
    sparse=summary["fixed_sparse"]
    gate={
        "C_zero_shot_at_least_60pct":
            main["C_zero_shot_frozen"]>=GATE["zero_shot_above_chance"],
        "C_zero_shot_exceeds_sparse_by_5pp":
            main["C_zero_shot_frozen"]-sparse["C_zero_shot_frozen"]>=
            GATE["zero_shot_advantage_over_sparse"],
        "C_learned_final_65pct":
            main["C_acquired_last"]>=GATE["C_learned_late_absolute"],
        "old_A_frozen_return_65pct":
            main["A_return_frozen"]>=GATE["old_A_return_accuracy"],
        "old_B_frozen_return_65pct":
            main["B_return_frozen"]>=GATE["old_B_return_accuracy"],
    }
    return {
        "status":"M5.14_development_structural_growth_composition_no_evolution",
        "seeds":list(seeds),"arms":list(ARMS),
        "curriculum":list(PHASES),"training_steps":TRAINING_STEPS,
        "total_steps":TOTAL_STEPS,
        "max_synaptic_slots":CAPACITY,"initial_sparse_edges":INITIAL_ACTIVE,
        "frozen_pre_C_learning_probe":True,
        "composition_grammar_is_experimenter_supplied":True,
        "correct_actions_never_given_as_learner_labels":True,
        "matched_yoked_birth_counts_times_and_occupancy":True,
        "selection_or_reproduction":False,
        "summary":summary,"screening_gate":gate,
        "advance_to_evolution":all(gate.values()),
        "raw_rows":rows,
        "rows_sha256":hashlib.sha256(json.dumps(
            rows,sort_keys=True,separators=(",",":")).encode()).hexdigest(),
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,default=Path(
        "results/m5-14/compositional-synaptogenesis-development.json"))
    args=p.parse_args()
    out=experiment()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(out,sort_keys=True,indent=2)+"\n",
                        encoding="utf-8")
    print("M5_14_COMPOSITION "+json.dumps({
        k:out[k] for k in ("status","summary","screening_gate",
            "advance_to_evolution","rows_sha256")},sort_keys=True),flush=True)


if __name__=="__main__":
    main()
