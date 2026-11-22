"""M5.18 — prediction-error-regulated metaplasticity, development ONLY.

STOP-GATE experiment following M5.17: the objective is to learn new skills
WITHOUT forgetting prior ones, rather than add more tasks or evolve brains.
Controlled 3-context A-B-C-A-B curriculum on fresh development seeds.
Same neural model, founder weights, tasks, training exposure and learning
objective for all arms; compare fixed consolidation, deliberately weaker
constant consolidation, prediction-error-dependent consolidation and replay.
The gate updates exclusively from the agent's own observed action outcome
versus its *pre-update* success estimate. No teacher targets or knowledge of
upcoming tasks. Curriculum phase ends still externally trigger synaptic
consolidation (NOT autonomous consolidation).
All metrics are development-only. Evolution and sealed world access blocked.
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

SEEDS=tuple(range(4100,4124))
ARMS=(
    "online",
    "replay",
    "plasticity_only",
    "loose_plasticity_only",
    "adaptive_plasticity_only",
    "replay_plasticity",
    "adaptive_plasticity_replay",
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
    "adaptive_A_retained_after_B":.85,
    "adaptive_A_retained_after_C":.80,
    "adaptive_B_acquisition":.85,
    "adaptive_C_acquisition":.85,
    "adaptive_final_all":.85,
    "improve_B_over_fixed":.08,
    "improve_C_over_fixed":.08,
    "A_not_worse_than_fixed_by_more":.10,
    "replay_B_not_worse_by_more":.05,
    "replay_C_not_worse_by_more":.05,
}
ADAPT_FAST_DECAY=.92
ADAPT_SLOW_DECAY=.995
ADAPT_MAX_RELAX=7.
MIN_PROTECTION_FRACTION=.10
LOOSE_PROTECTION_FRACTION=.25


def require_seed(seed:int)->None:
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError("Only independent M5.18 development seeds")


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
    return False


def has_plasticity(arm:str):
    return arm in (
        "plasticity_only","loose_plasticity_only",
        "adaptive_plasticity_only","replay_plasticity",
        "adaptive_plasticity_replay")


def has_replay(arm:str):
    return arm in (
        "replay","replay_plasticity",
        "adaptive_plasticity_replay")


def is_adaptive(arm:str):
    return arm in ("adaptive_plasticity_only","adaptive_plasticity_replay")


class Learner:
    def __init__(self,seed:int,arm:str,*,yoked_events=None):
        require_seed(seed)
        if arm not in ARMS:raise ValueError("Unknown M5.18 learning arm")
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
        self.total_importance=[0.]*PARAMS
        self.weighted_anchor=[0.]*PARAMS
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
        self.protection_fraction=1.0
        self.error_fast=None
        self.error_slow=None
        self.protection_trace=[]

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
        # Preaggregated at curriculum boundaries, not recomputed for every
        # stochastic gradient or episodic replay update.
        return self.total_importance

    def _update(self,record):
        x,y,context,action,reward=record
        prob,g=self._gradients(x,y,context,action)
        delta=reward-prob
        # Protect significant previously-used weights against drift.
        old=self.weights[:]
        protection=self.total_importance if has_plasticity(self.arm) else None
        factor=(self.protection_fraction if is_adaptive(self.arm)
                else LOOSE_PROTECTION_FRACTION
                if self.arm=="loose_plasticity_only" else 1.)
        for i in self.active:
            movement=LR*delta*g[i]
            if protection is not None and protection[i]:
                movement/=1.+CONSOLIDATE_STABILITY*protection[i]*factor
                penalty=(protection[i]*old[i]-self.weighted_anchor[i])
                movement-=LR*CONSOLIDATE_ANCHOR*factor*penalty
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
        pre_update_probability,grad=self._gradients(x,y,ctx,action)
        if is_adaptive(self.arm):
            # This scalar is calculated exclusively from the actual lived
            # outcome, BEFORE any synapse update. No C/A/B target or phase
            # boundary is used to decide how much protection to relax.
            squared_surprise=(real_feedback-pre_update_probability)**2
            if self.error_fast is None:
                self.error_fast=squared_surprise
                self.error_slow=squared_surprise
            else:
                self.error_fast=(ADAPT_FAST_DECAY*self.error_fast+
                                 (1-ADAPT_FAST_DECAY)*squared_surprise)
                self.error_slow=(ADAPT_SLOW_DECAY*self.error_slow+
                                 (1-ADAPT_SLOW_DECAY)*squared_surprise)
            elevated=max(0.,self.error_fast-self.error_slow)
            persistent=max(0.,self.error_fast-.08)
            pressure=elevated+.7*persistent
            self.protection_fraction=max(
                MIN_PROTECTION_FRACTION,
                min(1.,1./(1.+ADAPT_MAX_RELAX*pressure)))
        elif self.arm=="loose_plasticity_only":
            self.protection_fraction=LOOSE_PROTECTION_FRACTION
        else:
            self.protection_fraction=1.
        self.protection_trace.append(self.protection_fraction)
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
        for i,value in enumerate(normalized):
            self.total_importance[i]+=value
            self.weighted_anchor[i]+=value*self.weights[i]
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
                self.total_importance[i]=0.
                self.weighted_anchor[i]=0.
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
        "average_protection_fraction":statistics.mean(brain.protection_trace),
        "minimum_protection_fraction":min(brain.protection_trace),
        "fraction_episodes_relaxed":sum(
            g<.8 for g in brain.protection_trace)/len(brain.protection_trace),
    }


def experiment(seeds:tuple[int,...]=SEEDS):
    if not seeds or len(seeds)!=len(set(seeds)) or any(
        type(s) is not int or s not in SEEDS for s in seeds):
        raise ValueError("Fresh unique M5.18 development seeds only")
    rows=[simulate(seed,arm) for seed in seeds for arm in ARMS]
    by={(r["seed"],r["arm"]):r for r in rows}
    metrics=(
        "A_after_A","A_after_B_before_return","B_after_B",
        "A_after_C_before_return","B_after_C_before_return",
        "C_after_C","A_after_A_relearning","B_after_A_relearning",
        "C_after_A_relearning","A_after_B_relearning",
        "B_after_B_relearning","C_after_B_relearning",
        "mean_prior_retention_after_C","mean_final_three_task_retention",
        "parameter_updates","memory_at_end",
        "average_protection_fraction","minimum_protection_fraction",
        "fraction_episodes_relaxed")
    summary={
        arm:{key:statistics.mean(by[(seed,arm)][key] for seed in seeds)
             for key in metrics} for arm in ARMS
    }
    adaptive=summary["adaptive_plasticity_only"]
    fixed=summary["plasticity_only"]
    replay_adaptive=summary["adaptive_plasticity_replay"]
    replay=summary["replay"]
    gate={
        "adaptive_A_retained_after_B_85pct":
            adaptive["A_after_B_before_return"]>=
            GATES["adaptive_A_retained_after_B"],
        "adaptive_A_retained_after_C_80pct":
            adaptive["A_after_C_before_return"]>=
            GATES["adaptive_A_retained_after_C"],
        "adaptive_B_learned_85pct":
            adaptive["B_after_B"]>=GATES["adaptive_B_acquisition"],
        "adaptive_C_learned_85pct":
            adaptive["C_after_C"]>=GATES["adaptive_C_acquisition"],
        "adaptive_all_final_average_85pct":
            adaptive["mean_final_three_task_retention"]>=
            GATES["adaptive_final_all"],
        "adaptive_B_vs_fixed_plus8pp":
            adaptive["B_after_B"]-fixed["B_after_B"]>=
            GATES["improve_B_over_fixed"],
        "adaptive_C_vs_fixed_plus8pp":
            adaptive["C_after_C"]-fixed["C_after_C"]>=
            GATES["improve_C_over_fixed"],
        "adaptive_retention_A_within10pp_of_fixed":
            adaptive["A_after_B_before_return"]>=
            fixed["A_after_B_before_return"]-
            GATES["A_not_worse_than_fixed_by_more"],
        "adaptive_plus_replay_B_within5pp_of_replay":
            replay_adaptive["B_after_B"]>=replay["B_after_B"]-
            GATES["replay_B_not_worse_by_more"],
        "adaptive_plus_replay_C_within5pp_of_replay":
            replay_adaptive["C_after_C"]>=replay["C_after_C"]-
            GATES["replay_C_not_worse_by_more"],
    }
    for seed in seeds:
        for arm in ARMS:
            expect=TRAIN_STEPS*(UPDATES if has_replay(arm) else 1)
            if by[(seed,arm)]["parameter_updates"]!=expect:
                raise AssertionError("Gradient update dose mismatch")
    return {
        "status":"M5.18_dev_error_gated_metaplasticity_no_evolution",
        "seeds":list(seeds),"arms":list(ARMS),
        "curriculum":list(PHASES),
        "active_contexts_explicitly_visible":True,
        "phase_boundaries_still_trigger_consolidation":True,
        "plasticity_gate_uses_only_pre_update_reward_prediction_error":True,
        "fixed_and_adaptive_same_architecture_experiences_updates":True,
        "low_constant_protection_controls_mere_weakening":True,
        "frozen_probes_no_weight_or_memory_updates":True,
        "reward_only_for_actual_action":True,
        "memory_cap_all_replay_arms":MEMORY_LIMIT,
        "context_allocation_is_researcher_programmed":True,
        "no_structural_growth_or_pruning":True,
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
        "results/m5-18/metaplasticity-development.json"))
    args=p.parse_args()
    report=experiment()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",
                        encoding="utf-8")
    print("M5_18_METAPLASTICITY "+json.dumps({
        k:report[k] for k in ("status","summary","screening_gate",
              "development_screen_pass","advance_to_evolution",
              "rows_sha256")},sort_keys=True),flush=True)


if __name__=="__main__":
    main()
