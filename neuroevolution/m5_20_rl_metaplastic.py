"""M5.20 exploratory meta-RL controller for the stability–plasticity dilemma.

A small tabular Q-learning regulator chooses 4 protection strengths separately
for four neuron groups. An RL decision is made every 40 *lived* environment
actions. It receives NO exam results, correct-action target, privileged
world model, task boundary, or future trial. Its reward combines recent
observed action success and increased prediction error over actual archived
past action/outcome records. This is genuine bootstrapped Q learning, but
still an engineered meta-reward and externally triggered phase-end synapse
consolidation (not self-evolving autonomous metaplasticity).

Seven matched treatments: ordinary online, fixed strong/weak weight protection,
random controller, heuristic controller, learned Q controller, and a Q control
that receives independently shuffled/signed meta-reward. Each has exactly
1,520 lived actions, one neural weight update per action, four protection
choices per 40-action block, same 145-parameter brain, same 128-record old
experience memory and reference-score computation. Q-table learning itself
adds small extra operations; this cost is explicitly counted separately.
All reward history for meta-control is feedback on action outcomes only.

This is DEVELOPMENT cohort 4300..4323 only; sealed original M5.6 world
cohorts unaccessed. No synapse growth/prune, ecological expansion, evolution,
or promotion to main. The 90% gate must pass retention AND acquisition.
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

SEEDS=tuple(range(4300,4324))
ARMS=("online","fixed_strong","fixed_weak","random_controller",
      "heuristic_controller","q_learning_controller",
      "scrambled_reward_q")
PHASES=(("A",400),("B",400),("C",400),("A",160),("B",160))
CONTEXT={"A":0,"B":1,"C":2}
HIDDEN=16
DIM=8
PARAMS=145
GROUPS=4
WEIGHTS_PER_GROUP=4
BLOCK=40
EXAM=80
TRAIN_STEPS=sum(n for _,n in PHASES)
SLOTS=(43,43,42)
MEMORY_LIMIT=sum(SLOTS)
STRENGTHS=(.10,.30,.60,1.00)
LR=.067
CONSOLIDATION_STIFFNESS=3.2
CONSOLIDATION_ANCHOR=.85
Q_LEARNING_RATE=.34
Q_DISCOUNT=.70
Q_EPSILON_START=.40
Q_EPSILON_END=.12
OLD_LOSS_COEFF=1.2
NEW_SUCCESS_COEFF=.8
NEW_PROGRESS_COEFF=.5
OLD_RISK_THRESHOLD=.15
NEW_SUCCESS_THRESHOLD=.80
GATES={
    "retention_A_after_B":.90,
    "retention_A_after_C":.90,
    "acquire_B":.90,
    "retain_B_after_C":.90,
    "acquire_C":.90,
    "final_three":.90,
    "tradeoff_gain_over_fixed_weak":.05,
    "tradeoff_gain_over_random":.05,
    "tradeoff_gain_over_scrambled":.05,
}


def valid(seed:int):
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError("M5.20 new development cohort 4300..4323 only")


def trials(seed:int,phase:int,n:int):
    valid(seed)
    if n%BLOCK:raise ValueError("Only complete 40-action blocks")
    rng=random.Random(seed*211_001+phase*271+71)
    samples=[]
    for _ in range(n//BLOCK):
        four=[(x,y) for x in (0,1) for y in (0,1) for _ in range(10)]
        rng.shuffle(four)
        samples.extend(four)
    return tuple(samples)


def inputs(x:int,y:int,ctx:int,a:int):
    if x not in (0,1) or y not in (0,1) or ctx not in (0,1,2) or a not in (0,1):
        raise ValueError("Only valid binary cues, observed context/action")
    return (2.*x-1.,2.*y-1.)+tuple(float(ctx==i) for i in range(3))+(
        float(a==0),float(a==1),1.)


def environment_reward(ctx:int,x:int,y:int,a:int):
    """Environment-only correct response; never passed to the regulator."""
    if ctx==0:correct=x^y
    elif ctx==1:correct=1^(x^y)
    elif ctx==2:correct=x
    else:raise ValueError("Invalid task")
    return int(a==correct)


def group(i:int):
    if i==PARAMS-1:return GROUPS-1
    if i < HIDDEN*DIM:return (i//DIM)//WEIGHTS_PER_GROUP
    return ((i-HIDDEN*DIM)//WEIGHTS_PER_GROUP)


def brier(p:int|float,actual:int):
    return (p-actual)**2


class Brain:
    def __init__(self,seed:int):
        valid(seed)
        r=random.Random(seed*103_973+7)
        self.w=[r.gauss(0.,.45) for _ in range(HIDDEN*DIM)]+[
            r.gauss(0.,.13) for _ in range(HIDDEN)]+[0.]
        self.original=tuple(self.w)
        self.action_rng=random.Random(seed*93_773+83)
        self.memory=tuple(deque(maxlen=c) for c in SLOTS)
        self.importance=[0.]*PARAMS
        self.anchor_sum=[0.]*PARAMS
        self.phase_sensitivity=[0.]*PARAMS
        self.phase_n=0
        self.lived=0
        self.updates=0
        self.reference_forward_evaluations=0
        self.absolute_weight_drift=0.

    def snapshot(self):
        return (tuple(self.w),tuple(tuple(m) for m in self.memory),
                tuple(self.importance),tuple(self.anchor_sum),
                tuple(self.phase_sensitivity),self.phase_n,
                self.lived,self.updates,self.reference_forward_evaluations,
                self.absolute_weight_drift,self.action_rng.getstate())

    def prob_grad(self,x,y,c,a):
        z=inputs(x,y,c,a)
        h=[math.tanh(math.fsum(self.w[j*DIM+k]*v
                 for k,v in enumerate(z))) for j in range(HIDDEN)]
        output=self.w[-1]+math.fsum(
            self.w[HIDDEN*DIM+j]*h[j] for j in range(HIDDEN))
        p=1./(1.+math.exp(-max(-35.,min(35.,output))))
        grad=[0.]*PARAMS
        for j,hj in enumerate(h):
            grad[HIDDEN*DIM+j]=hj
            coefficient=self.w[HIDDEN*DIM+j]*(1-hj*hj)
            for k,v in enumerate(z):
                grad[j*DIM+k]=coefficient*v
        grad[-1]=1.
        return p,grad

    def choose(self,x,y,c,tick,train:bool):
        s0=self.prob_grad(x,y,c,0)[0]
        s1=self.prob_grad(x,y,c,1)[0]
        a=int(s1>s0)
        if train:
            e=.25*(1-min(1.,tick/TRAIN_STEPS))+.07*min(1.,tick/TRAIN_STEPS)
            if self.action_rng.random()<e:
                a=self.action_rng.randrange(2)
        return a

    def old_brier(self,current_ctx):
        """Read-only test on past *experienced* actions, not exam observations.

        No current-context sample is used. Reward is the historical result
        obtained by the agent for its chosen action; negative observations
        are as informative as positive ones for reward prediction.
        """
        values=[]
        for c,mem in enumerate(self.memory):
            if c==current_ctx:continue
            for x,y,ctx,a,observed in mem:
                p,_=self.prob_grad(x,y,ctx,a)
                values.append(brier(p,observed))
                self.reference_forward_evaluations+=1
        return statistics.mean(values) if values else None

    def live(self,x,y,c,choice,outcome,protections):
        if outcome not in (0,1) or len(protections)!=GROUPS:
            raise ValueError("Only observed 0/1 chosen-action outcome")
        if any(v not in STRENGTHS and v!=0. for v in protections):
            raise ValueError("Unregistered plasticity factor")
        p,gradient=self.prob_grad(x,y,c,choice)
        direction=outcome-p
        old=self.w[:]
        for i in range(PARAMS):
            value=direction*gradient[i]
            protection=protections[group(i)]
            imp=self.importance[i]
            if imp>0 and protection:
                value/=(1.+CONSOLIDATION_STIFFNESS*imp*protection)
                value-=CONSOLIDATION_ANCHOR*protection*(
                    imp*old[i]-self.anchor_sum[i])
            new=max(-5.,min(5.,old[i]+LR*value))
            self.w[i]=new
            self.absolute_weight_drift+=abs(new-old[i])
            self.phase_sensitivity[i]+=gradient[i]**2*(
                .25+.75*float(outcome))
        self.phase_n+=1
        self.memory[c].append((x,y,c,choice,outcome))
        self.lived+=1
        self.updates+=1

    def consolidate(self):
        """Lab phase boundaries: shared baseline limitation, not agent choice."""
        if self.phase_n<40:raise AssertionError("Not enough experiences")
        mean=[v/self.phase_n for v in self.phase_sensitivity]
        peak=max(mean) or 1.
        for i,value in enumerate(mean):
            imp=min(1.,(value/peak)**.5)
            self.importance[i]+=imp
            self.anchor_sum[i]+=imp*self.w[i]
        self.phase_sensitivity=[0.]*PARAMS
        self.phase_n=0


class Controller:
    def __init__(self,seed:int,arm:str):
        valid(seed)
        if arm not in ARMS:raise ValueError("Unknown controller arm")
        self.arm=arm
        self.rng=random.Random(seed*31_337+467)
        self.noise=random.Random(seed*79_793+207)
        self.q=[[[0.]*len(STRENGTHS) for _ in range(4)]
                for _ in range(GROUPS)]
        self.steps=0
        self.q_updates=0
        self.meta_reward_sum=0.
        self.actual_actions=[]
        self.last_rewards=[]
        self.meta_scores=[]
        self.protection_sums=[0.]*GROUPS
        self.exploratory_actions=0

    def state(self,success:float,old_loss:float|None):
        # Two-bit state: poor recent learning, and risk of old-memory error.
        # No external task identity or curriculum boundary enters policy.
        return int(success<NEW_SUCCESS_THRESHOLD)+2*int(
            old_loss is not None and old_loss>OLD_RISK_THRESHOLD)

    def choose(self,state:int,old_loss:float|None):
        out=[]
        for g in range(GROUPS):
            if self.arm=="online":idx=-1
            elif self.arm=="fixed_strong":idx=3
            elif self.arm=="fixed_weak":idx=1
            elif self.arm=="random_controller":idx=self.rng.randrange(4)
            elif self.arm=="heuristic_controller":
                # Engineered nonlearning comparator using same current risk.
                idx=(0 if state&1 else 1 if state&2 else 2)
            else:
                fraction=min(1.,self.steps/38.)
                eps=Q_EPSILON_START*(1-fraction)+Q_EPSILON_END*fraction
                if self.rng.random()<eps:
                    idx=self.rng.randrange(4)
                    self.exploratory_actions+=1
                else:
                    values=self.q[g][state]
                    idx=max(range(4),key=lambda k:(values[k],-k))
            out.append(idx)
        self.steps+=1
        self.actual_actions.append(tuple(out))
        for g,choice in enumerate(out):
            self.protection_sums[g]+=(
                0. if choice==-1 else STRENGTHS[choice])
        return tuple(0. if i==-1 else STRENGTHS[i] for i in out),tuple(out)

    def feedback(self,state,choices,next_state,reward):
        self.last_rewards.append(reward)
        self.meta_reward_sum+=reward
        if self.arm not in ("q_learning_controller","scrambled_reward_q"):
            return
        learned=(self.noise.uniform(-1.,1.)
                 if self.arm=="scrambled_reward_q" else reward)
        # One-step off-policy bootstrapped Q update after the 40-action block.
        for g,k in enumerate(choices):
            future=max(self.q[g][next_state])
            prediction=self.q[g][state][k]
            self.q[g][state][k]+=Q_LEARNING_RATE*(
                learned+Q_DISCOUNT*future-prediction)
            self.q_updates+=1

    def q_checksum(self):
        return math.fsum(x for g in self.q for row in g for x in row)


def frozen_exam(brain,seed,ctx,phase):
    before=brain.snapshot()
    success=0
    for x,y in trials(seed,phase,EXAM):
        a=brain.choose(x,y,ctx,0,False)
        success+=environment_reward(ctx,x,y,a)
    if before!=brain.snapshot():
        raise AssertionError("Frozen exam leaked learning to organism")
    return success/EXAM


def run(seed:int,arm:str):
    valid(seed)
    brain=Brain(seed)
    controller=Controller(seed,arm)
    scores={}
    training_curves={}
    tick=0
    for phase,(name,n) in enumerate(PHASES):
        context=CONTEXT[name]
        pair_sequence=trials(seed,phase,n)
        for block_index in range(n//BLOCK):
            pairs=pair_sequence[block_index*BLOCK:(block_index+1)*BLOCK]
            current_success=(controller.meta_scores[-1]
                             if controller.meta_scores else .5)
            start_old=brain.old_brier(context)
            state=controller.state(current_success,start_old)
            protections,choices=controller.choose(state,start_old)
            start_drift=brain.absolute_weight_drift
            rewards=[]
            for x,y in pairs:
                a=brain.choose(x,y,context,tick,True)
                actual=environment_reward(context,x,y,a)
                brain.live(x,y,context,a,actual,protections)
                tick+=1
                rewards.append(actual)
            accuracy=statistics.mean(rewards)
            end_old=brain.old_brier(context)
            increase=(max(0.,end_old-start_old)
                      if start_old is not None and end_old is not None else 0.)
            progress=accuracy-current_success
            # Fixed engineered meta objective, not any frozen exam signal.
            # Prevent exploiting archival observations by not updating their
            # stored outcomes or allowing the controller to choose samples.
            meta_reward=(NEW_SUCCESS_COEFF*(accuracy-.5)
                         +NEW_PROGRESS_COEFF*progress
                         -OLD_LOSS_COEFF*increase)
            next_state=controller.state(accuracy,end_old)
            controller.feedback(state,choices,next_state,meta_reward)
            controller.meta_scores.append(accuracy)
        brain.consolidate()
        training_curves[f"phase_{phase}_{name}"]=(
            statistics.mean(controller.meta_scores[-2:]))
        for old in range(max(CONTEXT[p[0]] for p in PHASES[:phase+1])+1):
            scores[f"phase_{phase}_{'ABC'[old]}"]=frozen_exam(
                brain,seed,old,20+phase*3+old)
    if brain.lived!=TRAIN_STEPS or brain.updates!=TRAIN_STEPS:
        raise AssertionError("Unfair synaptic-update or experience budget")
    if sum(map(len,brain.memory))>MEMORY_LIMIT:
        raise AssertionError("Memory exceeds registered 128 records")
    if controller.steps!=TRAIN_STEPS//BLOCK:
        raise AssertionError("Mismatched meta decision budget")
    return {
        "seed":seed,"arm":arm,
        "A_after_A":scores["phase_0_A"],
        "A_after_B":scores["phase_1_A"],
        "B_after_B":scores["phase_1_B"],
        "A_after_C":scores["phase_2_A"],
        "B_after_C":scores["phase_2_B"],
        "C_after_C":scores["phase_2_C"],
        "A_final":scores["phase_4_A"],
        "B_final":scores["phase_4_B"],
        "C_final":scores["phase_4_C"],
        "mean_final":statistics.mean((
            scores["phase_4_A"],scores["phase_4_B"],scores["phase_4_C"])),
        "balanced_tradeoff":statistics.mean((
            scores["phase_2_A"],scores["phase_1_B"],
            scores["phase_2_B"],scores["phase_2_C"])),
        "lived_experiences":brain.lived,
        "synaptic_updates":brain.updates,
        "reference_forward_evaluations":brain.reference_forward_evaluations,
        "memory_records":sum(map(len,brain.memory)),
        "memory_capacity":MEMORY_LIMIT,
        "meta_decision_blocks":controller.steps,
        "meta_q_updates":controller.q_updates,
        "meta_q_checksum":controller.q_checksum(),
        "meta_reward_mean":statistics.mean(controller.last_rewards),
        "meta_success_mean":statistics.mean(controller.meta_scores),
        "mean_protection":statistics.mean(controller.protection_sums)/
                           controller.steps,
        "group_protection_means":[v/controller.steps for v in
                                  controller.protection_sums],
        "q_explorations":controller.exploratory_actions,
        "frozen_exams":True,
        "meta_reward_uses_exam":False,
        "explicit_context_id_visible_to_organism":True,
        "lab_consolidation_boundaries":True,
        "weight_replay_updates":0,
        "no_evolution":True,
    }


def experiment(seeds:tuple[int,...]=SEEDS):
    if not seeds or len(seeds)!=len(set(seeds)) or any(
        type(s) is not int or s not in SEEDS for s in seeds):
        raise ValueError("Unique new M5.20 development cohort only")
    rows=[run(seed,arm) for seed in seeds for arm in ARMS]
    ix={(r["seed"],r["arm"]):r for r in rows}
    metrics=("A_after_A","A_after_B","B_after_B","A_after_C",
             "B_after_C","C_after_C","A_final","B_final","C_final",
             "mean_final","balanced_tradeoff","synaptic_updates",
             "reference_forward_evaluations","meta_decision_blocks",
             "meta_q_updates","meta_reward_mean",
             "mean_protection","meta_success_mean","q_explorations")
    summary={arm:{k:statistics.mean(ix[(seed,arm)][k] for seed in seeds)
                  for k in metrics} for arm in ARMS}
    learned=summary["q_learning_controller"]
    checks={
        "rl_A_after_B_90":learned["A_after_B"]>=GATES["retention_A_after_B"],
        "rl_A_after_C_90":learned["A_after_C"]>=GATES["retention_A_after_C"],
        "rl_B_acquisition_90":learned["B_after_B"]>=GATES["acquire_B"],
        "rl_B_after_C_90":learned["B_after_C"]>=GATES["retain_B_after_C"],
        "rl_C_acquisition_90":learned["C_after_C"]>=GATES["acquire_C"],
        "rl_final_mean_90":learned["mean_final"]>=GATES["final_three"],
        "rl_vs_constant_weak_joint_plus5pp":
            learned["balanced_tradeoff"]-
            summary["fixed_weak"]["balanced_tradeoff"]>=
            GATES["tradeoff_gain_over_fixed_weak"],
        "rl_vs_random_joint_plus5pp":
            learned["balanced_tradeoff"]-
            summary["random_controller"]["balanced_tradeoff"]>=
            GATES["tradeoff_gain_over_random"],
        "rl_vs_scrambled_meta_reward_plus5pp":
            learned["balanced_tradeoff"]-
            summary["scrambled_reward_q"]["balanced_tradeoff"]>=
            GATES["tradeoff_gain_over_scrambled"],
    }
    for seed in seeds:
        if len({ix[(seed,a)]["synaptic_updates"] for a in ARMS})!=1:
            raise AssertionError("Unequal weight updates")
        if len({ix[(seed,a)]["reference_forward_evaluations"]
                for a in ARMS})!=1:
            raise AssertionError("Unequal old-memory feedback computations")
        if any(ix[(seed,a)]["memory_capacity"]!=MEMORY_LIMIT
               for a in ARMS):
            raise AssertionError("Unequal memory storage budget")
    return {
        "status":"M5.20_dev_tabular_Q_metaplasticity_no_evolution",
        "seeds":list(seeds),"arms":list(ARMS),
        "curriculum":list(PHASES),
        "meta_action_protection_strengths":list(STRENGTHS),
        "meta_decisions_every":BLOCK,
        "meta_q_discount":Q_DISCOUNT,
        "meta_reward_excludes_exam_and_oracle_target":True,
        "meta_uses_lived_action_feedback_and_past_observed_outcomes":True,
        "context_identity_is_explicit":True,
        "meta_context_state_is_from_own_progress_and_memory":True,
        "phase_consolidation_externally_triggered":True,
        "all_arms_evaluate_same_old_reference_memory":True,
        "all_arms_same_synaptic_update_budget":True,
        "meta_Q_update_cost_not_equal_to_nonlearning_arms":True,
        "no_direct_historical_weight_replay":True,
        "no_structural_growth_prune_reproduction_or_evolution":True,
        "no_sealed_source_or_replication_seeds":True,
        "screening_gate":checks,"summary":summary,
        "development_screen_pass":all(checks.values()),
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
        "results/m5-20/rl-metaplastic-controller-development.json"))
    args=p.parse_args()
    result=experiment()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(
        result,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    print("M5_20_META_RL "+json.dumps({
        k:result[k] for k in ("status","summary","screening_gate",
            "development_screen_pass","advance_to_evolution","rows_sha256")
    },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
