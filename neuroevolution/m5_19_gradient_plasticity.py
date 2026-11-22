"""M5.19: gradient-conflict constraints as a plasticity STOP/GO diagnostic.

A → B → C → A → B with only own-action success/failure feedback.
Five same-size neural controllers have identical initial weights, 1520
environment interactions, ONE parameter update per interaction, and a
128-experience context-balanced buffer (43/43/42). No replay weight
training: earlier experiences are consulted only to calculate old-task
gradient constraints. All arms compute equal reference-gradient workloads;
only projection arms APPLY the constraints. This is memory-assisted
plasticity, NOT memory-free learning. Observed context labels are supplied
by the lab (no context discovery). All exams are frozen and use the same
familiar four binary cue pairs, not OOD generalization. Growth, pruning,
selection, ecological evolution and sealed seeds are forbidden.

Projection follows GEM-style first-order non-increasing historical loss:
if an UPDATE GRADIENT g is opposed to a remembered gradient m, remove its
component antiparallel to m. This is only a first-order approximation; not
a guarantee for a nonlinear function after a finite step. Local arm
additionally gates only coordinates whose update would oppose remembered
per-coordinate gradients, then applies the same global projections.

All hyperparameters and ten stop/go gates are frozen before DEV cohort
4200..4223 (disjoint from M5.6 sealed and previous development cohorts).
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

SEEDS=tuple(range(4200,4224))
ARMS=("online","fixed_protection","weak_protection",
      "gradient_projection","projection_local")
PHASES=(("A",400),("B",400),("C",400),("A",160),("B",160))
LABELS={"A":0,"B":1,"C":2}
MEMORY_CAP=128
SLOTS=(43,43,42)
HIDDEN=16
DIM=8  # two bipolar cues + three known contexts + two actions + bias
PARAMS=HIDDEN*DIM+HIDDEN+1
LR=.067
TRAIN_STEPS=sum(n for _,n in PHASES)
EXAM_STEPS=80
CONSOLIDATE_STIFFNESS=3.2
CONSOLIDATE_ANCHOR=.85
WEAK_FACTOR=.25
REFERENCE_PATTERNS=((0,0),(0,1),(1,0),(1,1))
LOCAL_CONFLICT_SCALE=.25
PROJECTION_PASSES=3
DOT_FLOOR=1e-12
GATES={
    "A_retained_after_B":.90,
    "A_retained_after_C":.90,
    "B_learned":.90,
    "C_learned":.90,
    "B_retained_after_C":.90,
    "final_three_mean":.90,
    "absolute_advantage_over_weak":.05,
}


def valid(seed:int)->None:
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError("Fresh development seeds 4200..4223 only")


def trials(seed:int,phase:int,count:int):
    valid(seed)
    if count%40:raise ValueError("Only complete balanced blocks")
    rng=random.Random(seed*211_001+phase*271+71)
    out=[]
    for _ in range(count//40):
        block=[(x,y) for x in (0,1) for y in (0,1) for _ in range(10)]
        rng.shuffle(block)
        out.extend(block)
    return tuple(out)


def inputs(x:int,y:int,c:int,a:int):
    if x not in (0,1) or y not in (0,1) or c not in (0,1,2) or a not in (0,1):
        raise ValueError("Two binary cues, known context, binary chosen action")
    return (2.*x-1.,2.*y-1.)+tuple(float(c==i) for i in range(3))+(
        float(a==0),float(a==1),1.)


def environment_reward(c:int,x:int,y:int,a:int):
    if c==0:correct=x^y
    elif c==1:correct=1^(x^y)
    elif c==2:correct=x
    else:raise ValueError("Unknown world")
    return int(a==correct)


def dot(a,b):
    return math.fsum(x*y for x,y in zip(a,b))


def projected_gradient(new, old_vectors):
    """Project intended ascent direction onto remembered half-spaces."""
    g=list(new)
    corrections=0
    for _ in range(PROJECTION_PASSES):
        for m in old_vectors:
            squared=dot(m,m)
            if squared<=DOT_FLOOR:continue
            overlap=dot(g,m)
            if overlap<0:
                amount=overlap/squared
                g=[a-amount*b for a,b in zip(g,m)]
                corrections+=1
    return g,corrections


class Brain:
    def __init__(self,seed:int,arm:str):
        valid(seed)
        if arm not in ARMS:raise ValueError("Unknown M5.19 arm")
        self.seed=seed
        self.arm=arm
        rng=random.Random(seed*103_973+7)
        self.weights=(
            [rng.gauss(0.,.45) for _ in range(HIDDEN*DIM)]
            +[rng.gauss(0.,.13) for _ in range(HIDDEN)]+[0.])
        self.original=tuple(self.weights)
        self.action_rng=random.Random(seed*93_773+83)
        self.memory=tuple(deque(maxlen=n) for n in SLOTS)
        self.anchor_importance=[0.]*PARAMS
        self.anchor_weighted=[0.]*PARAMS
        self.phase_sensitivity=[0.]*PARAMS
        self.phase_samples=0
        self.lived=0
        self.updates=0
        self.reference_gradients_evaluated=0
        self.reference_constraints_evaluated=0
        self.projection_corrections=0
        self.local_conflicting_coordinates=0
        self.local_opportunities=0
        self.memory_used_for_weight_replay=0

    def state(self):
        return (tuple(self.weights),tuple(tuple(b) for b in self.memory),
                tuple(self.anchor_importance),tuple(self.anchor_weighted),
                tuple(self.phase_sensitivity),self.phase_samples,
                self.lived,self.updates,self.reference_gradients_evaluated,
                self.reference_constraints_evaluated,
                self.projection_corrections,self.local_conflicting_coordinates,
                self.local_opportunities,self.action_rng.getstate())

    def prediction_gradient(self,x:int,y:int,c:int,a:int):
        z=inputs(x,y,c,a)
        h=[math.tanh(math.fsum(
                self.weights[j*DIM+k]*v for k,v in enumerate(z)))
           for j in range(HIDDEN)]
        raw=self.weights[-1]+math.fsum(
            self.weights[HIDDEN*DIM+j]*h[j] for j in range(HIDDEN))
        p=1./(1.+math.exp(-max(-35.,min(35.,raw))))
        dp=[0.]*PARAMS
        for j,hj in enumerate(h):
            dp[HIDDEN*DIM+j]=hj
            coeff=self.weights[HIDDEN*DIM+j]*(1-hj*hj)
            for k,v in enumerate(z):
                dp[j*DIM+k]=coeff*v
        dp[-1]=1.
        return p,dp

    def choose(self,x:int,y:int,c:int,tick:int,explore:bool):
        p0=self.prediction_gradient(x,y,c,0)[0]
        p1=self.prediction_gradient(x,y,c,1)[0]
        action=int(p1>p0)
        if explore:
            t=min(1.,tick/TRAIN_STEPS)
            epsilon=.25*(1-t)+.07*t
            if self.action_rng.random()<epsilon:
                action=self.action_rng.randrange(2)
        return action

    def _historical_references(self,current_ctx:int):
        """Consult remembered outcomes; old gradients recomputed at CURRENT weights.

        Exactly four representative past experienced action records for each
        other previously seen context, one for each x/y pattern if present.
        Never generate missing observations, actions, or target labels.
        Same reference-gradient workload for all arms.
        """
        constraints=[]
        for previous,buf in enumerate(self.memory):
            if previous==current_ctx or not buf:continue
            samples=[]
            for x,y in REFERENCE_PATTERNS:
                item=next((r for r in reversed(buf)
                           if r[0]==x and r[1]==y),None)
                if item is not None:samples.append(item)
            if not samples:continue
            total=[0.]*PARAMS
            for x,y,c,a,outcome in samples:
                prob,g=self.prediction_gradient(x,y,c,a)
                residual=outcome-prob
                for i,v in enumerate(g):
                    total[i]+=residual*v
                self.reference_gradients_evaluated+=1
            constraints.append([v/len(samples) for v in total])
            self.reference_constraints_evaluated+=1
        return constraints

    def experience(self,x:int,y:int,c:int,action:int,actual_success:int):
        if actual_success not in (0,1):
            raise ValueError("Only observed chosen-action success/failure")
        tick=self.lived
        self.lived+=1
        # New-task gradients and past-task gradients are both based only
        # on outcomes personally experienced by THIS simulated individual.
        p,grad=self.prediction_gradient(x,y,c,action)
        direction=[(actual_success-p)*g for g in grad]
        previous=self._historical_references(c)
        updated=direction
        if self.arm in ("gradient_projection","projection_local"):
            if self.arm=="projection_local" and previous:
                # Local gating suppresses just the coordinates conflicting
                # with prior learned tasks, rather than all connections.
                updated=[]
                for j,value in enumerate(direction):
                    self.local_opportunities+=1
                    incompatible=any(value*m[j]<0 for m in previous)
                    if incompatible:self.local_conflicting_coordinates+=1
                    updated.append(value*(LOCAL_CONFLICT_SCALE if incompatible else 1.))
            updated,n=self._project(updated,previous)
            self.projection_corrections+=n
        elif self.arm in ("fixed_protection","weak_protection"):
            factor=1. if self.arm=="fixed_protection" else WEAK_FACTOR
            protected=[]
            for i,g in enumerate(direction):
                imp=self.anchor_importance[i]
                if imp:
                    g=g/(1.+CONSOLIDATE_STIFFNESS*imp*factor)
                    penalty=(imp*self.weights[i]-self.anchor_weighted[i])
                    g-=CONSOLIDATE_ANCHOR*factor*penalty
                protected.append(g)
            updated=protected
        for i,value in enumerate(updated):
            self.weights[i]=max(-5.,min(5.,self.weights[i]+LR*value))
        # Consolidation salience also uses actual feedback, no task target.
        if self.arm in ("fixed_protection","weak_protection"):
            for i,value in enumerate(grad):
                self.phase_sensitivity[i]+=(value*value)*(
                    .25+.75*float(actual_success))
            self.phase_samples+=1
        self.memory[c].append((x,y,c,action,actual_success))
        self.updates+=1
        if self.updates!=self.lived:
            raise AssertionError("More than one weight update per lived action")

    def _project(self,direction,old_vectors):
        return projected_gradient(direction,old_vectors)

    def consolidate(self):
        # Researcher-supplied phase end: same as M5.17/M5.18 baseline.
        if self.arm not in ("fixed_protection","weak_protection"):return
        if self.phase_samples<40:raise AssertionError("No training interval")
        mean=[v/self.phase_samples for v in self.phase_sensitivity]
        peak=max(mean) or 1.
        importance=[min(1.,(v/peak)**.5) for v in mean]
        for i,value in enumerate(importance):
            self.anchor_importance[i]+=value
            self.anchor_weighted[i]+=value*self.weights[i]
        self.phase_sensitivity=[0.]*PARAMS
        self.phase_samples=0


def frozen_exam(brain:Brain,seed:int,ctx:int,phase:int):
    before=brain.state()
    passed=0
    for x,y in trials(seed,phase,EXAM_STEPS):
        a=brain.choose(x,y,ctx,0,False)
        passed+=environment_reward(ctx,x,y,a)
    if before!=brain.state():
        raise AssertionError("Exam changed weights, memory or sampling RNG")
    return passed/EXAM_STEPS


def run(seed:int,arm:str):
    valid(seed)
    brain=Brain(seed,arm)
    scores={}
    training={}
    tick=0
    for phase,(name,n) in enumerate(PHASES):
        c=LABELS[name]
        successes=[]
        for x,y in trials(seed,phase,n):
            a=brain.choose(x,y,c,tick,True)
            observed=environment_reward(c,x,y,a)
            brain.experience(x,y,c,a,observed)
            successes.append(observed)
            tick+=1
        training[f"phase_{phase}_{name}"]=sum(successes[-80:])/80
        brain.consolidate()
        for prev in range(max(LABELS[p[0]] for p in PHASES[:phase+1])+1):
            scores[f"phase_{phase}_{'ABC'[prev]}"]=frozen_exam(
                brain,seed,prev,20+phase*3+prev)
    assert tick==TRAIN_STEPS and brain.updates==TRAIN_STEPS
    assert sum(map(len,brain.memory))<=MEMORY_CAP
    # Every arm (including baselines) allocates the exact same number
    # of memory slots, and computes reference gradients on own history.
    return {
        "seed":seed,"arm":arm,
        "A_after_A":scores["phase_0_A"],
        "A_after_B_before_relearning":scores["phase_1_A"],
        "B_after_B":scores["phase_1_B"],
        "A_after_C_before_relearning":scores["phase_2_A"],
        "B_after_C_before_relearning":scores["phase_2_B"],
        "C_after_C":scores["phase_2_C"],
        "A_final":scores["phase_4_A"],
        "B_final":scores["phase_4_B"],
        "C_final":scores["phase_4_C"],
        "mean_final_three":statistics.mean((
            scores["phase_4_A"],scores["phase_4_B"],scores["phase_4_C"])),
        "mean_stability_and_new_learning":statistics.mean((
            scores["phase_2_A"],scores["phase_1_B"],scores["phase_2_C"])),
        "train_last80":training,
        "parameter_updates":brain.updates,
        "reference_gradient_evaluations":brain.reference_gradients_evaluated,
        "reference_constraint_vectors":brain.reference_constraints_evaluated,
        "reference_buffer_capacity":MEMORY_CAP,
        "reference_buffer_size":sum(map(len,brain.memory)),
        "projection_corrections":brain.projection_corrections,
        "local_conflicting_coordinates":brain.local_conflicting_coordinates,
        "local_opportunities":brain.local_opportunities,
        "replay_weight_updates":brain.memory_used_for_weight_replay,
        "frozen_exams":True,
        "phase_end_consolidation_externally_triggered":True,
        "original_sealed_cohort":False,
    }


def experiment(seeds:tuple[int,...]=SEEDS):
    if not seeds or len(seeds)!=len(set(seeds)) or any(
        type(seed) is not int or seed not in SEEDS for seed in seeds):
        raise ValueError("Only unique M5.19 development seeds permitted")
    rows=[run(seed,arm) for seed in seeds for arm in ARMS]
    per={(r["seed"],r["arm"]):r for r in rows}
    keys=("A_after_A","A_after_B_before_relearning","B_after_B",
          "A_after_C_before_relearning","B_after_C_before_relearning",
          "C_after_C","A_final","B_final","C_final","mean_final_three",
          "mean_stability_and_new_learning","parameter_updates",
          "reference_gradient_evaluations","reference_constraint_vectors",
          "projection_corrections","local_conflicting_coordinates")
    summary={arm:{
        k:statistics.mean(per[(seed,arm)][k] for seed in seeds)
        for k in keys} for arm in ARMS}
    loose=summary["weak_protection"]
    local=summary["projection_local"]
    proj=summary["gradient_projection"]
    checks={
        "local_A_after_B_at_least90":
            local["A_after_B_before_relearning"]>=GATES["A_retained_after_B"],
        "local_A_after_C_at_least90":
            local["A_after_C_before_relearning"]>=GATES["A_retained_after_C"],
        "local_B_first_learned_at_least90":
            local["B_after_B"]>=GATES["B_learned"],
        "local_C_first_learned_at_least90":
            local["C_after_C"]>=GATES["C_learned"],
        "local_B_after_C_at_least90":
            local["B_after_C_before_relearning"]>=GATES["B_retained_after_C"],
        "local_final_A_B_C_mean_at_least90":
            local["mean_final_three"]>=GATES["final_three_mean"],
        "local_tradeoff_vs_weak_plus5pp":
            local["mean_stability_and_new_learning"]-
            loose["mean_stability_and_new_learning"]>=
            GATES["absolute_advantage_over_weak"],
        "projection_tradeoff_vs_weak_plus5pp":
            proj["mean_stability_and_new_learning"]-
            loose["mean_stability_and_new_learning"]>=
            GATES["absolute_advantage_over_weak"],
    }
    for seed in seeds:
        counts={per[(seed,a)]["parameter_updates"] for a in ARMS}
        memory_cap={per[(seed,a)]["reference_buffer_capacity"] for a in ARMS}
        reference_evals={per[(seed,a)]["reference_gradient_evaluations"]
                         for a in ARMS}
        if counts!={TRAIN_STEPS} or memory_cap!={MEMORY_CAP}:
            raise AssertionError("Unmatched gradient update/memory budget")
        # Balanced context exposures ensure the same number of stored
        # x/y patterns, hence reference gradient computations per arm.
        if len(reference_evals)!=1:
            raise AssertionError("Unmatched counterfactual compute work")
        if any(per[(seed,a)]["replay_weight_updates"]!=0 for a in ARMS):
            raise AssertionError("Old experiences were directly replay-trained")
    return {
        "status":"M5.19_dev_constrained_gradient_local_plasticity",
        "seeds":list(seeds),"arms":list(ARMS),
        "curriculum":list(PHASES),
        "frozen_exam_steps":EXAM_STEPS,
        "actual_action_outcomes_only":True,
        "past_experiences_used_as_reference_constraints_only":True,
        "no_experiential_replay_weight_updates":True,
        "same_gradient_update_and_reference_compute_budget":True,
        "same_reference_memory_capacity":MEMORY_CAP,
        "task_context_is_explicit":True,
        "only_four_experienced_cue_pairs_per_context":True,
        "consolidation_phase_end_researcher_triggered":True,
        "nonlinear_projection_only_first_order_not_safety_guarantee":True,
        "no_structural_growth_prune_or_evolution":True,
        "original_M5_6_sealed_cohorts_unaccessed":True,
        "summary":summary,"screening_gate":checks,
        "development_screen_pass":all(checks.values()),
        "independent_replication_complete":False,
        "evolution_authorized":False,
        "advance_to_evolution":False,
        "raw_rows":rows,
        "rows_sha256":hashlib.sha256(json.dumps(
            rows,sort_keys=True,separators=(",",":")).encode()).hexdigest(),
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,default=Path(
        "results/m5-19/gradient-conflict-plasticity-development.json"))
    args=parser.parse_args()
    result=experiment()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(
        result,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    print("M5_19_GRADIENTS "+json.dumps({
        k:result[k] for k in (
            "status","summary","screening_gate",
            "development_screen_pass","advance_to_evolution",
            "rows_sha256")},sort_keys=True),flush=True)


if __name__=="__main__":
    main()
