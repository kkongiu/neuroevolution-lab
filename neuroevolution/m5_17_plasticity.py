"""M5.17 developmental school: plasticity, synaptic consolidation, and growth.

Same action-conditioned reward-prediction architecture in each arm. A -> B
-> C -> A -> B tasks; four familiar binary cues, task identity observable.
The agent gets only feedback from *its own action*, never correct labels.
Frozen exams occur before any new teaching after a task switch.

Novel interventions:
- Consolidation: streaming parameter sensitivity to successful experience,
  captured at task boundaries as a soft weight-importance/anchor snapshot.
  Future gradient steps are slowed and softly regularized toward anchors
  for past competencies (an EWC-like heuristic).
- Autobiographical replay: 128 total examples allocated explicitly across
  A/B/C context bins (43,43,42). This is NOT autonomously managed memory.
- Structural adaptation: up to 145 possible weights, sparse 104 initially;
  low-progress-triggered activation of dormant weights, pruning of weak
  unprotected weights. Matched per-seed random identity/identical birth-death
  event schedule is the structural control. No new neurons.
- Genomes are newly seeded, but there is no genetic selection/reproduction.

All training uses newly disjoint M5.17 development seeds 4000..4023.
Never unlock original M5.6 sealed source/holdout/replication worlds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from collections import deque
from pathlib import Path

SEEDS=tuple(range(4000,4024))
ARMS=(
    "online",                  # one online outcome gradient
    "recent_repeat",           # five newest-outcome gradients
    "replay",                  # five updates, balanced episodic replay
    "plasticity_only",         # one update, consolidated old synapses
    "replay_plasticity",        # five updates, both mechanisms
    "replay_plasticity_sparse", # sparse fixed structural size
    "replay_plasticity_growth", # sparse growing/pruned
    "replay_plasticity_yoked",  # identical birth/death dose random edges
    "replay_uncoupled",        # outcome-independent reward control
)
PHASES=(("A",400),("B",400),("C",400),("A",160),("B",160))
CONTEXT={"A":0,"B":1,"C":2}
NUM_CONTEXTS=3
HIDDEN=16
DIM=8                 # x,y,3 context,2 action,1 bias
PARAMS=HIDDEN*DIM+HIDDEN+1  # 145
MEMORY_LIMIT=128
TRAIN_STEPS=sum(n for _,n in PHASES)
PROBE_STEPS=80
UPDATES=5
LR=.067
CONSOLIDATE_STABILITY=3.2
CONSOLIDATE_ANCHOR=.85
INITIAL_SPARSE=104
MIN_ACTIVE=96
BIRTH_BATCH=4
PRUNE_BATCH=2
GROW_INTERVAL=80
PRUNE_INTERVAL=160

GATES={
    "three_primitives_mastered":.85,
    "A_after_B_no_replay_advantage":.08,
    "A_after_C_replay_consolidation_advantage":.05,
    "B_after_C_absolute":.75,
    "C_after_return_absolute":.75,
    "structural_vs_sparse_5pp":.05,
    "structural_vs_yoked_5pp":.05,
}


def require_seed(seed:int)->None:
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError("Only independent M5.17 development seeds")


def curriculum(seed:int,phase_index:int,n:int):
    require_seed(seed)
    if n%40:raise ValueError("Balanced phases only")
    rng=random.Random(seed*211_001+phase_index*271+71)
    pairs=[]
    for _ in range(n//40):
        block=[(x,y) for x in (0,1) for y in (0,1) for _ in range(10)]
        rng.shuffle(block)
        pairs.extend(block)
    return tuple(pairs)


def sensory(x:int,y:int,ctx:int,action:int):
    if (x not in (0,1) or y not in (0,1)
            or ctx not in (0,1,2) or action not in (0,1)):
        raise ValueError("Binary cues and action, three explicit contexts")
    return (2.*x-1.,2.*y-1.)+tuple(float(ctx==i) for i in range(3))+(
        float(action==0),float(action==1),1.)


def environmental_reward(ctx:int,x:int,y:int,action:int)->int:
    """Privileged environment target MUST NOT be given to agent."""
    if ctx==0:desired=x^y
    elif ctx==1:desired=1^(x^y)
    elif ctx==2:desired=x
    else:raise ValueError("Unknown environment")
    return int(action==desired)


def is_sparse_arm(arm:str):
    return arm in (
        "replay_plasticity_sparse","replay_plasticity_growth",
        "replay_plasticity_yoked")


def has_plasticity(arm:str):
    return arm in ("plasticity_only","replay_plasticity",
                   "replay_plasticity_sparse",
                   "replay_plasticity_growth",
                   "replay_plasticity_yoked")


def has_replay(arm:str):
    return arm in ("replay","replay_plasticity",
                   "replay_plasticity_sparse",
                   "replay_plasticity_growth",
                   "replay_plasticity_yoked",
                   "replay_uncoupled")


class Learner:
    def __init__(self,seed:int,arm:str,*,yoked_events=None):
        require_seed(seed)
        if arm not in ARMS:raise ValueError("Unknown M5.17 learning arm")
        if (arm=="replay_plasticity_yoked")!=(yoked_events is not None):
            raise ValueError("Yoked event list required only for control")
        self.seed=seed
        self.arm=arm
        founder=random.Random(seed*103_973+7)
        self.weights=[
            founder.gauss(0.,.45)
            for _ in range(HIDDEN*DIM)
        ]+[
            founder.gauss(0.,.13)
            for _ in range(HIDDEN)
        ]+[0.]
        self.original_weights=tuple(self.weights)
        self.active=set(range(PARAMS))
        if is_sparse_arm(arm):
            rng=random.Random(seed*17_033+227)
            self.active=set(rng.sample(range(PARAMS-1),INITIAL_SPARSE-1))
            self.active.add(PARAMS-1)
            self._mask()
        self.action_rng=random.Random(seed*93_773+83)
        self.replay_rng=random.Random(seed*873_331+197)
        self.noise_rng=random.Random(seed*67_231+193)
        self.structural_rng=random.Random(seed*97_337+783)
        self.context_memory={
            0:deque(maxlen=43),1:deque(maxlen=43),2:deque(maxlen=42)}
        self.anchors=[]      # prior-context (snapshot, relative importance)
        self.phase_sensitivity=[0.]*PARAMS
        self.phase_samples=0
        self.opportunity=[0.]*PARAMS
        self.usage=[0.]*PARAMS
        self.current_block=[]
        self.past_block_accuracy=[]
        self.events=[]
        self.yoked_events=yoked_events or ()
        self.yoked_cursor=0
        self.updates=0
        self.lived=0
        self.active_sum=0
        self.previous_context=None
        self.rehearsals=0

    def _mask(self):
        for i in range(PARAMS):
            if i not in self.active:self.weights[i]=0.

    def snapshot(self):
        return (tuple(self.weights),tuple(sorted(self.active)),
                self.updates,self.lived,
                tuple(tuple(self.context_memory[c]) for c in range(3)),
                tuple(self.events),tuple(self.usage),
                tuple(self.opportunity),tuple(self.current_block),
                tuple(self.past_block_accuracy))

    def _gradients(self,x:int,y:int,context:int,action:int):
        inp=sensory(x,y,context,action)
        h=[
            math.tanh(math.fsum(self.weights[j*DIM+k]*feat
                                for k,feat in enumerate(inp)))
            for j in range(HIDDEN)]
        logit=self.weights[-1]+math.fsum(
            self.weights[HIDDEN*DIM+j]*h[j] for j in range(HIDDEN))
        prob=1./(1.+math.exp(-max(-35.,min(35.,logit))))
        grad=[0.]*PARAMS
        for j,hj in enumerate(h):
            grad[HIDDEN*DIM+j]=hj
            coeff=self.weights[HIDDEN*DIM+j]*(1-hj*hj)
            for k,feat in enumerate(inp):
                grad[j*DIM+k]=coeff*feat
        grad[-1]=1.
        return prob,grad

    def choose(self,x:int,y:int,context:int,train_tick:int,explore:bool):
        values=[self._gradients(x,y,context,a)[0] for a in (0,1)]
        best=int(values[1]>values[0])
        if explore:
            fraction=min(1.,train_tick/TRAIN_STEPS)
            eps=.25*(1-fraction)+.07*fraction
            if self.action_rng.random()<eps:
                return self.action_rng.randrange(2)
        return best

    def _importance(self):
        strength=[0.]*PARAMS
        for _,importance in self.anchors:
            for i,v in enumerate(importance):
                strength[i]+=v
        return strength

    def _update(self,record):
        x,y,context,action,reward=record
        prob,g=self._gradients(x,y,context,action)
        delta=reward-prob
        # Protect significant previously-used weights against drift.
        old=self.weights[:]
        protection=self._importance() if has_plasticity(self.arm) else None
        for i in self.active:
            movement=LR*delta*g[i]
            if protection is not None and protection[i]:
                movement/=1.+CONSOLIDATE_STABILITY*protection[i]
                penalty=math.fsum(
                    imp[i]*(old[i]-anchor[i])
                    for anchor,imp in self.anchors)
                movement-=LR*CONSOLIDATE_ANCHOR*penalty
            self.weights[i]=max(-5.,min(5.,old[i]+movement))
        self._mask()
        self.updates+=1

    def observe(self,x:int,y:int,ctx:int,action:int,real_feedback:int,
                tick:int):
        if real_feedback not in (0,1):
            raise ValueError("Only actual reward from chosen action")
        self.lived+=1
        if self.previous_context is None:
            self.previous_context=ctx
        elif ctx!=self.previous_context:
            # A new class starts; adapt from the feedback without explicit
            # scripted context switch hooks beyond observable task identity.
            self.previous_context=ctx
            self.current_block.clear()
            self.past_block_accuracy.clear()
        received=self.noise_rng.randrange(2) if (
            self.arm=="replay_uncoupled") else real_feedback
        record=(x,y,ctx,action,received)
        _,grad=self._gradients(x,y,ctx,action)
        for i,der in enumerate(grad):
            if i in self.active:
                # Functional importance is a local *sensitivity* heuristic,
                # not a causal deletion test or exact synaptic Fisher matrix.
                self.phase_sensitivity[i]+=der*der*(
                    .25+.75*float(real_feedback))
                self.usage[i]=.995*self.usage[i]+.005*abs(der)
            else:
                self.opportunity[i]=.985*self.opportunity[i]+.015*(
                    abs(der)*(.25+.75*float(real_feedback)))
        self.phase_samples+=1
        self._update(record)
        if has_replay(self.arm):
            self.context_memory[ctx].append(record)
            for _ in range(UPDATES-1):
                nonempty=[c for c in range(NUM_CONTEXTS)
                          if self.context_memory[c]]
                c=self.replay_rng.choice(nonempty)
                selected=self.replay_rng.choice(tuple(self.context_memory[c]))
                self._update(selected)
                if c!=ctx:self.rehearsals+=1
        elif self.arm=="recent_repeat":
            for _ in range(UPDATES-1):self._update(record)
        self.current_block.append(real_feedback)
        if len(self.current_block)==40:
            self.past_block_accuracy.append(sum(self.current_block)/40)
            self.current_block.clear()
        self.active_sum+=len(self.active)

    def consolidate(self,context:int):
        if not has_plasticity(self.arm):return
        if self.phase_samples<40:raise AssertionError("Need real experiences")
        average=[v/self.phase_samples for v in self.phase_sensitivity]
        # Rank-scale to 0..1; do not protect every parameter equally.
        maximum=max(average) or 1.
        normalized=[min(1.,(v/maximum)**.5)
                    if i in self.active else 0.
                    for i,v in enumerate(average)]
        self.anchors.append((tuple(self.weights),normalized))
        self.phase_sensitivity=[0.]*PARAMS
        self.phase_samples=0

    def remodel(self,tick:int):
        if self.arm not in ("replay_plasticity_growth",
                            "replay_plasticity_yoked"):return
        if self.arm=="replay_plasticity_yoked":
            while (self.yoked_cursor<len(self.yoked_events)
                   and self.yoked_events[self.yoked_cursor]["tick"]==tick):
                e=self.yoked_events[self.yoked_cursor]
                self._alter(e["kind"],e["count"],tick,random_locations=True)
                self.yoked_cursor+=1
            return
        # No lookahead into upcoming task/correct answers; statistics
        # computed solely from previous lived outcomes.
        if (tick%GROW_INTERVAL==0
                and len(self.past_block_accuracy)>=2):
            prev,now=self.past_block_accuracy[-2:]
            if now<.85 and now-prev<.05:
                self._alter("birth",BIRTH_BATCH,tick,random_locations=False)
        if tick%PRUNE_INTERVAL==0:
            self._alter("prune",PRUNE_BATCH,tick,random_locations=False)

    def _alter(self,kind:str,count:int,tick:int,random_locations:bool):
        if kind=="birth":
            pool=sorted(set(range(PARAMS-1))-self.active)
            n=min(count,len(pool))
            if not n:return
            indices=(self.structural_rng.sample(pool,n) if random_locations
                else sorted(pool,key=lambda i:(-self.opportunity[i],i))[:n])
            for i in indices:
                self.active.add(i)
                self.weights[i]=self.structural_rng.gauss(0.,.09)
                self.opportunity[i]=0.
                self.usage[i]=0.
                # Reinforcement of *old* phantom anchors on newly grown
                # synapses would be incorrect: erase that old importance.
                for _,imp in self.anchors:imp[i]=0.
        elif kind=="prune":
            pool=sorted(self.active-set([PARAMS-1]))
            n=min(count,max(0,len(self.active)-MIN_ACTIVE))
            if n<=0:return
            protected=self._importance()
            indices=(self.structural_rng.sample(pool,n) if random_locations
                else sorted(pool,key=lambda i:(
                    protected[i]+.2*self.usage[i]+.015*abs(self.weights[i]),i))[:n])
            for i in indices:
                self.active.remove(i)
                self.weights[i]=0.
                self.usage[i]=0.
                self.opportunity[i]=0.
                for _,imp in self.anchors:imp[i]=0.
        else:raise ValueError("Only birth/prune")
        self.events.append({"tick":tick,"kind":kind,"count":len(indices),
                            "active":len(self.active),"edges":indices})


def frozen_exam(brain:Learner,seed:int,ctx:int,phase:int):
    state=brain.snapshot()
    score=0
    for x,y in curriculum(seed,phase,PROBE_STEPS):
        action=brain.choose(x,y,ctx,0,False)
        score+=environmental_reward(ctx,x,y,action)
    if brain.snapshot()!=state:raise AssertionError(
        "Frozen exam changed synaptic weights/memory/plasticity")
    return score/PROBE_STEPS


def simulate(seed:int,arm:str,*,yoked_events=None):
    require_seed(seed)
    brain=Learner(seed,arm,yoked_events=yoked_events)
    scores={}
    learning_scores={}
    tick=0
    for phase,(name,n) in enumerate(PHASES):
        ctx=CONTEXT[name]
        recent=[]
        for x,y in curriculum(seed,phase,n):
            choice=brain.choose(x,y,ctx,tick,True)
            result=environmental_reward(ctx,x,y,choice)
            brain.observe(x,y,ctx,choice,result,tick)
            tick+=1
            brain.remodel(tick)
            recent.append(result)
        learning_scores[str(phase)+"_"+name]=sum(recent[-80:])/80
        # Consolidate each learned context including post-return revision.
        brain.consolidate(ctx)
        # All encountered contexts tested frozen BEFORE next phase.
        for old in range(3):
            if old<=max(CONTEXT[p[0]] for p in PHASES[:phase+1]):
                scores["phase_"+str(phase)+"_"+("ABC"[old])]=frozen_exam(
                    brain,seed,old,20+phase*3+old)
    if tick!=TRAIN_STEPS:raise AssertionError("Unfair experience count")
    expected_updates=TRAIN_STEPS*(1 if arm in (
        "online","plasticity_only") else UPDATES)
    if brain.updates!=expected_updates:
        raise AssertionError("Unmatched learning budget")
    size=sum(len(b) for b in brain.context_memory.values())
    if size>MEMORY_LIMIT:raise AssertionError("Exceeded replay memory limit")
    return {
        "seed":seed,"arm":arm,
        "scores":scores,"learning_scores":learning_scores,
        "A_after_A":scores["phase_0_A"],
        "A_after_B_before_return":scores["phase_1_A"],
        "B_after_B":scores["phase_1_B"],
        "A_after_C_before_return":scores["phase_2_A"],
        "B_after_C_before_return":scores["phase_2_B"],
        "C_after_C":scores["phase_2_C"],
        "A_after_A_relearning":scores["phase_3_A"],
        "B_after_A_relearning":scores["phase_3_B"],
        "C_after_A_relearning":scores["phase_3_C"],
        "A_after_B_relearning":scores["phase_4_A"],
        "B_after_B_relearning":scores["phase_4_B"],
        "C_after_B_relearning":scores["phase_4_C"],
        "mean_prior_retention_after_C":statistics.mean((
            scores["phase_2_A"],scores["phase_2_B"])),
        "mean_final_three_task_retention":statistics.mean((
            scores["phase_4_A"],scores["phase_4_B"],scores["phase_4_C"])),
        "lived_experiences":brain.lived,
        "parameter_updates":brain.updates,
        "mean_active_parameters":brain.active_sum/TRAIN_STEPS,
        "initial_active_parameters":(
            INITIAL_SPARSE if is_sparse_arm(arm) else PARAMS),
        "final_active_parameters":len(brain.active),
        "historical_rehearsals":brain.rehearsals,
        "memory_at_end":size,
        "consolidation_snapshots":len(brain.anchors),
        "events":brain.events,
        "frozen_probes":True,
        "reproduction":False,
    }


def experiment(seeds:tuple[int,...]=SEEDS):
    if not seeds or len(seeds)!=len(set(seeds)) or any(
        type(s) is not int or s not in SEEDS for s in seeds):
        raise ValueError("Fresh unique M5.17 development seeds only")
    rows=[]
    for seed in seeds:
        growth=simulate(seed,"replay_plasticity_growth")
        planned=[{k:e[k] for k in ("tick","kind","count")}
                 for e in growth["events"]]
        for arm in ARMS:
            r=(growth if arm=="replay_plasticity_growth"
               else simulate(seed,arm,yoked_events=planned
                             if arm=="replay_plasticity_yoked" else None))
            rows.append(r)
        yoked=next(r for r in rows[-len(ARMS):]
                   if r["arm"]=="replay_plasticity_yoked")
        for lhs,rhs in zip(growth["events"],yoked["events"]):
            if any(lhs[k]!=rhs[k] for k in ("tick","kind","count","active")):
                raise AssertionError("Yoked remodeling event dose mismatch")
        if len(growth["events"])!=len(yoked["events"]) or (
                growth["mean_active_parameters"]!=yoked["mean_active_parameters"]):
            raise AssertionError("Yoked remodeling timing/capacity mismatch")
    by={(r["seed"],r["arm"]):r for r in rows}
    metrics=("A_after_A","A_after_B_before_return","B_after_B",
             "A_after_C_before_return","B_after_C_before_return",
             "C_after_C","A_after_A_relearning","B_after_A_relearning",
             "C_after_A_relearning","A_after_B_relearning",
             "B_after_B_relearning","C_after_B_relearning",
             "mean_prior_retention_after_C",
             "mean_final_three_task_retention","parameter_updates",
             "memory_at_end","mean_active_parameters",
             "final_active_parameters","historical_rehearsals")
    summary={
        arm:{key:statistics.mean(by[(seed,arm)][key] for seed in seeds)
             for key in metrics}
        for arm in ARMS
    }
    p=summary["replay_plasticity"]
    g=summary["replay_plasticity_growth"]
    gate={
        "A_B_C_each_mastered_before_returns":
            min(p["A_after_A"],p["B_after_B"],p["C_after_C"])>=
            GATES["three_primitives_mastered"],
        "plasticity_without_replay_reduces_A_forgetting":
            summary["plasticity_only"]["A_after_B_before_return"]-
            summary["online"]["A_after_B_before_return"]>=
            GATES["A_after_B_no_replay_advantage"],
        "adding_consolidation_to_replay_preserves_A":
            p["A_after_C_before_return"]-
            summary["replay"]["A_after_C_before_return"]>=
            GATES["A_after_C_replay_consolidation_advantage"],
        "B_retained_through_C":
            p["B_after_C_before_return"]>=GATES["B_after_C_absolute"],
        "C_retained_through_A_and_B_returns":
            p["C_after_B_relearning"]>=GATES["C_after_return_absolute"],
        "guided_remodel_outperforms_static_sparse":
            g["mean_final_three_task_retention"]-
            summary["replay_plasticity_sparse"][
                "mean_final_three_task_retention"]>=
            GATES["structural_vs_sparse_5pp"],
        "guided_remodel_outperforms_matched_random":
            g["mean_final_three_task_retention"]-
            summary["replay_plasticity_yoked"][
                "mean_final_three_task_retention"]>=
            GATES["structural_vs_yoked_5pp"],
    }
    return {
        "status":"M5.17_dev_selective_plasticity_replay_synapse_remodel",
        "seeds":list(seeds),"arms":list(ARMS),"curriculum":list(PHASES),
        "active_contexts_explicitly_visible":True,
        "frozen_probes_no_weight_or_memory_updates":True,
        "reward_only_for_actual_action":True,
        "memory_cap_all_replay_arms":MEMORY_LIMIT,
        "context_allocation_is_researcher_programmed":True,
        "matched_replay_update_counts":True,
        "matched_growth_prune_events_and_occupancy":True,
        "plasticity_importance_is_sensitivity_heuristic":True,
        "no_new_neurons_or_unbounded_growth":True,
        "no_original_sealed_cohorts":True,
        "summary":summary,"screening_gate":gate,
        "development_screen_pass":all(gate.values()),
        "independent_replication_complete":False,
        "evolution_authorized":False,
        "advance_to_evolution":False,
        "raw_rows":rows,
        "rows_sha256":hashlib.sha256(json.dumps(
            rows,sort_keys=True,separators=(",",":")).encode()).hexdigest(),
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,default=Path(
        "results/m5-17/selective-plasticity-development.json"))
    args=p.parse_args()
    report=experiment()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",
                        encoding="utf-8")
    print("M5_17_PLASTICITY "+json.dumps({
        k:report[k] for k in ("status","summary","screening_gate",
              "development_screen_pass","advance_to_evolution",
              "rows_sha256")},sort_keys=True),flush=True)


if __name__=="__main__":
    main()
