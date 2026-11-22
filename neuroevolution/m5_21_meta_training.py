"""M5.21 — Train the metaplasticity REGULATOR across many independent lives.

64 development lives teach a single shared meta-Q policy; lower-level brains,
weights, experiences, observations and anchors are ALWAYS reset between lives.
24 completely separate seeds evaluate the FROZEN meta-policy against
hand-tuned constant, random, untrained and corrupted-reward meta-controls.
Meta-training never consults any frozen examination score. A/B/C context
identity and curriculum consolidation boundaries remain researcher-supplied:
this is NOT open-ended autonomy or a novel task family. All original M5.6
sealed cohorts and main stay untouched; evolution explicitly impossible.

One neuron group changes its protection factor per 40-decision block,
three others stay at a registered constant 0.30. This design avoids the
per-group credit ambiguity of M5.20's four simultaneous Q decisions.
Regulator learns over 64 lives (28 effective choices per life = 1792).
Meta reward now penalizes ABSOLUTE old experience Brier loss, not just
within-block forgetting. The policy is learned on seeds 4400..4463 and
FROZEN for unseen-seed examination 4500..4523. No lower-level brain/episode
state is inherited across lives; only the four-group 4-state Q table travels.
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

TRAIN_SEEDS=tuple(range(4400,4464))
EVAL_SEEDS=tuple(range(4500,4524))
SEEDS=TRAIN_SEEDS+EVAL_SEEDS
TRAIN_ARMS=("train_real","train_corrupted")
EVAL_ARMS=("learned_transfer","corrupted_transfer","untrained_q",
           "random_controller","fixed_weak","fixed_strong",
           "heuristic_controller")
ARMS=TRAIN_ARMS+EVAL_ARMS
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
BASE_PROTECTION=.30
LR=.067
CONSOLIDATION_STIFFNESS=3.2
CONSOLIDATION_ANCHOR=.85
Q_LEARNING_RATE=.20
Q_DISCOUNT=.70
Q_EPSILON_START=.30
Q_EPSILON_END=.08
OLD_ABSOLUTE_COEFF=1.4
OLD_DELTA_COEFF=.4
NEW_SUCCESS_COEFF=.8
NEW_PROGRESS_COEFF=.5
OLD_RISK_THRESHOLD=.15
NEW_SUCCESS_THRESHOLD=.80
GATES={
    "retention_A_after_B":.90,
    "retention_A_after_C":.90,
    "acquisition_B":.90,
    "retention_B_after_C":.90,
    "acquisition_C":.90,
    "final_three":.90,
    "tradeoff_gain":.05,
}

def valid(seed:int):
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError("M5.21 development 4400..4463 or unseen evaluation 4500..4523 only")


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



def empty_q():
    return [[[0.]*len(STRENGTHS) for _ in range(4)]
            for _ in range(GROUPS)]


def copy_q(q):
    if q is None:return empty_q()
    if len(q)!=GROUPS or any(len(x)!=4 for x in q):
        raise ValueError("Mismatched meta-policy architecture")
    return [[list(action_values) for action_values in group_states]
            for group_states in q]


def q_digest(q):
    return hashlib.sha256(json.dumps(q,sort_keys=True,
        separators=(",",":")).encode()).hexdigest()


class Controller:
    """Frozen or cross-lifetime learned Q policy; never receives exam labels."""

    def __init__(self,seed:int,arm:str,*,q=None,train_meta:bool=False,
                 meta_training_step:int=0):
        valid(seed)
        if arm not in ARMS:raise ValueError("Unknown metaplastic arm")
        if train_meta and (arm not in TRAIN_ARMS or seed not in TRAIN_SEEDS):
            raise ValueError("Only development lives may train Q")
        if not train_meta and (seed not in EVAL_SEEDS or arm not in EVAL_ARMS):
            raise ValueError("Frozen policy only in unseen evaluation cohort")
        self.arm=arm
        self.train_meta=train_meta
        self.q=q if train_meta else copy_q(q)
        self.rng=random.Random(seed*31_337+467)
        self.scramble_rng=random.Random(seed*79_793+207)
        self.steps=0
        self.q_updates=0
        self.meta_training_step=meta_training_step
        self.action_trace=[]
        self.meta_reward_sum=0.
        self.meta_rewards=[]
        self.accuracies=[]
        self.group_totals=[0.]*GROUPS
        self.random_exploration=0
        self.effective_blocks=0

    def state(self,success:float,old_loss:float|None):
        return int(success<NEW_SUCCESS_THRESHOLD)+2*int(
            old_loss is not None and old_loss>OLD_RISK_THRESHOLD)

    def choose(self,state:int,active:bool):
        # During first learning stage there is no anchored knowledge.
        # The regulator has no causal effect, so do not exploit/train
        # on an action it cannot truly control.
        g=self.steps%GROUPS
        chosen=1
        if active:
            if self.arm=="random_controller":
                chosen=self.rng.randrange(len(STRENGTHS))
            elif self.arm=="fixed_strong":
                chosen=3
            elif self.arm=="fixed_weak":
                chosen=1
            elif self.arm=="heuristic_controller":
                chosen=(0 if state&1 else 1 if state&2 else 2)
            else:
                explore=False
                if self.train_meta:
                    fraction=min(1.,self.meta_training_step/(64*28))
                    epsilon=(Q_EPSILON_START*(1-fraction)
                             +Q_EPSILON_END*fraction)
                    explore=self.rng.random()<epsilon
                if explore:
                    chosen=self.rng.randrange(len(STRENGTHS))
                    self.random_exploration+=1
                else:
                    vals=self.q[g][state]
                    chosen=max(range(len(STRENGTHS)),
                               key=lambda a:(vals[a],-a))
            self.effective_blocks+=1
        strengths=[BASE_PROTECTION]*GROUPS
        if active:
            if self.arm=="fixed_strong":
                strengths=[1.]*GROUPS
            else:
                strengths[g]=STRENGTHS[chosen]
        self.action_trace.append((g,chosen,active))
        for index,value in enumerate(strengths):
            self.group_totals[index]+=value
        self.steps+=1
        return tuple(strengths),(g,chosen)

    def feedback(self,state,decision,next_state,reward,active:bool):
        self.meta_rewards.append(reward)
        self.meta_reward_sum+=reward
        if not active or not self.train_meta:return
        g,a=decision
        observed=(self.scramble_rng.uniform(-1.,1.)
                  if self.arm=="train_corrupted" else reward)
        old=self.q[g][state][a]
        self.q[g][state][a]+=Q_LEARNING_RATE*(
            observed+Q_DISCOUNT*max(self.q[g][next_state])-old)
        self.q_updates+=1
        self.meta_training_step+=1


def frozen_exam(brain:Brain,seed:int,context:int,phase:int):
    before=brain.snapshot()
    passed=0
    for x,y in trials(seed,phase,EXAM):
        act=brain.choose(x,y,context,0,False)
        passed+=environment_reward(context,x,y,act)
    if brain.snapshot()!=before:
        raise AssertionError("Exam contaminated brain, memory or random state")
    return passed/EXAM


def run_life(seed:int,arm:str,*,q=None,train_meta:bool=False,
             meta_training_step:int=0):
    """Fresh brain every life; only the explicit Q array can be inherited."""
    valid(seed)
    brain=Brain(seed)
    brain_founder_digest=hashlib.sha256(json.dumps(
        brain.original).encode()).hexdigest()
    control=Controller(seed,arm,q=q,train_meta=train_meta,
                       meta_training_step=meta_training_step)
    rewards_so_far=[]
    frozen={}
    tick=0
    for phase,(name,n) in enumerate(PHASES):
        ctx=CONTEXT[name]
        phase_pairs=trials(seed,phase,n)
        for ib in range(n//BLOCK):
            cue_block=phase_pairs[ib*BLOCK:(ib+1)*BLOCK]
            previous=(control.accuracies[-1] if control.accuracies else .5)
            old_before=brain.old_brier(ctx)
            current_state=control.state(previous,old_before)
            protected=any(brain.importance)
            factors,decision=control.choose(current_state,protected)
            outcomes=[]
            for x,y in cue_block:
                act=brain.choose(x,y,ctx,tick,True)
                realized=environment_reward(ctx,x,y,act)
                brain.live(x,y,ctx,act,realized,factors)
                tick+=1
                outcomes.append(realized)
            success=statistics.mean(outcomes)
            old_after=brain.old_brier(ctx)
            absolute_old=(max(0.,old_after-.08)
                          if old_after is not None else 0.)
            deterioration=(max(0.,old_after-old_before)
                           if old_before is not None
                           and old_after is not None else 0.)
            meta_reward=(
                NEW_SUCCESS_COEFF*(success-.5)
                +NEW_PROGRESS_COEFF*(success-previous)
                -OLD_ABSOLUTE_COEFF*absolute_old
                -OLD_DELTA_COEFF*deterioration)
            next_state=control.state(success,old_after)
            control.feedback(current_state,decision,next_state,meta_reward,
                             protected)
            control.accuracies.append(success)
        brain.consolidate()
        # Do not even compute exams for development meta-training lives:
        # their grades could never accidentally enter the Q policy.
        if not train_meta:
            for c in range(max(CONTEXT[x[0]] for x in PHASES[:phase+1])+1):
                frozen[f"phase_{phase}_{'ABC'[c]}"]=frozen_exam(
                    brain,seed,c,20+phase*3+c)
    if brain.lived!=TRAIN_STEPS or brain.updates!=TRAIN_STEPS:
        raise AssertionError("Brain exposure mismatch")
    if sum(map(len,brain.memory))>MEMORY_LIMIT:
        raise AssertionError("Episodic memory overflow")
    if control.steps!=TRAIN_STEPS//BLOCK:
        raise AssertionError("Meta-control opportunity mismatch")
    if control.effective_blocks!=28:
        raise AssertionError("Effective opportunity count mismatch")
    if train_meta and control.q_updates!=28:
        raise AssertionError("Expected 28 Q updates per development lifetime")
    if not train_meta and control.q_updates:
        raise AssertionError("Evaluation meta-policy updated itself")
    output={
        "seed":seed,"arm":arm,
        "meta_training":train_meta,
        "fresh_brain_founder_sha256":brain_founder_digest,
        "fresh_brain_lived_actions":brain.lived,
        "fresh_brain_updates":brain.updates,
        "memory_capacity":MEMORY_LIMIT,
        "memory_records":sum(map(len,brain.memory)),
        "reference_forward_evaluations":brain.reference_forward_evaluations,
        "regulation_blocks":control.steps,
        "effective_meta_blocks":control.effective_blocks,
        "q_updates":control.q_updates,
        "q_parameter_sha256":q_digest(control.q),
        "mean_meta_reward":statistics.mean(control.meta_rewards),
        "mean_train_accuracy":statistics.mean(control.accuracies),
        "mean_protection":statistics.mean(
            control.group_totals)/control.steps,
        "random_meta_explorations":control.random_exploration,
        "frozen_exams_run":not train_meta,
        "exam_feedback_used_in_training":False,
        "explicit_context_identity":True,
        "all_memory_lifetime_local":True,
    }
    if not train_meta:
        output.update({
            "A_after_A":frozen["phase_0_A"],
            "A_after_B":frozen["phase_1_A"],
            "B_after_B":frozen["phase_1_B"],
            "A_after_C":frozen["phase_2_A"],
            "B_after_C":frozen["phase_2_B"],
            "C_after_C":frozen["phase_2_C"],
            "A_final":frozen["phase_4_A"],
            "B_final":frozen["phase_4_B"],
            "C_final":frozen["phase_4_C"],
            "final_three_mean":statistics.mean((
                frozen["phase_4_A"],frozen["phase_4_B"],
                frozen["phase_4_C"])),
            "tradeoff_mean":statistics.mean((
                frozen["phase_2_A"],frozen["phase_1_B"],
                frozen["phase_2_B"],frozen["phase_2_C"])),
        })
    return output


def train_controller(arm:str,seeds:tuple[int,...]=TRAIN_SEEDS):
    if arm not in TRAIN_ARMS or not seeds or any(
            type(s) is not int or s not in TRAIN_SEEDS for s in seeds):
        raise ValueError("Training restricted to disjoint development cohort")
    if len(set(seeds))!=len(seeds):
        raise ValueError("Cannot re-use a training life seed")
    q=empty_q()
    rows=[]
    position=0
    for seed in seeds:
        r=run_life(seed,arm,q=q,train_meta=True,
                   meta_training_step=position)
        if r["fresh_brain_updates"]!=TRAIN_STEPS:
            raise AssertionError("Unexpected inherited brain state")
        rows.append(r)
        position+=r["q_updates"]
    if position!=28*len(seeds):
        raise AssertionError("Meta update count budget incorrect")
    return q,rows


def paired_bootstrap_ci(rows,key_arm="learned_transfer",
                        comparator="fixed_weak",
                        metric="tradeoff_mean",replicates=1600):
    ix={(r["seed"],r["arm"]):r for r in rows}
    diffs=[ix[(s,key_arm)][metric]-ix[(s,comparator)][metric]
           for s in EVAL_SEEDS]
    rng=random.Random(314159)
    draws=[]
    for _ in range(replicates):
        draws.append(statistics.mean(rng.choice(diffs) for _ in diffs))
    draws.sort()
    lo=draws[int(.025*replicates)]
    hi=draws[int(.975*replicates)-1]
    return {"delta":statistics.mean(diffs),"boot95":[lo,hi],
            "n_paired_seed":len(diffs)}


def experiment(training_seeds=TRAIN_SEEDS,eval_seeds=EVAL_SEEDS):
    if tuple(training_seeds)!=TRAIN_SEEDS or tuple(eval_seeds)!=EVAL_SEEDS:
        raise ValueError("M5.21 full preregistered development/holdout only")
    learned,train_rows=train_controller("train_real",training_seeds)
    corrupted,corrupt_rows=train_controller(
        "train_corrupted",training_seeds)
    if q_digest(learned)==q_digest(empty_q()):
        raise AssertionError("Meta-learning did not change its policy")
    q_pre_test=q_digest(learned)
    q_corrupt_before=q_digest(corrupted)
    results=[]
    for seed in eval_seeds:
        for arm in EVAL_ARMS:
            selected=(
                learned if arm=="learned_transfer"
                else corrupted if arm=="corrupted_transfer"
                else empty_q())
            row=run_life(seed,arm,q=selected,train_meta=False)
            results.append(row)
    if q_digest(learned)!=q_pre_test or q_digest(corrupted)!=q_corrupt_before:
        raise AssertionError("Evaluation leaked Q learning")
    ix={(r["seed"],r["arm"]):r for r in results}
    metrics=(
        "A_after_A","A_after_B","B_after_B","A_after_C",
        "B_after_C","C_after_C","A_final","B_final","C_final",
        "final_three_mean","tradeoff_mean","mean_train_accuracy",
        "mean_protection","q_updates","reference_forward_evaluations")
    summary={arm:{k:statistics.mean(ix[(seed,arm)][k]
              for seed in EVAL_SEEDS) for k in metrics}
              for arm in EVAL_ARMS}
    learned_s=summary["learned_transfer"]
    checks={
        "A_after_B_90":learned_s["A_after_B"]>=GATES["retention_A_after_B"],
        "A_after_C_90":learned_s["A_after_C"]>=GATES["retention_A_after_C"],
        "B_new_learning_90":learned_s["B_after_B"]>=GATES["acquisition_B"],
        "B_after_C_90":learned_s["B_after_C"]>=GATES["retention_B_after_C"],
        "C_new_learning_90":learned_s["C_after_C"]>=GATES["acquisition_C"],
        "final_three_mean_90":learned_s["final_three_mean"]>=
                              GATES["final_three"],
        "vs_fixed_weak_tradeoff_plus5pp":
            learned_s["tradeoff_mean"]-
            summary["fixed_weak"]["tradeoff_mean"]>=GATES["tradeoff_gain"],
        "vs_random_tradeoff_plus5pp":
            learned_s["tradeoff_mean"]-
            summary["random_controller"]["tradeoff_mean"]>=
            GATES["tradeoff_gain"],
        "vs_corrupt_reward_tradeoff_plus5pp":
            learned_s["tradeoff_mean"]-
            summary["corrupted_transfer"]["tradeoff_mean"]>=
            GATES["tradeoff_gain"],
        "vs_untrained_tradeoff_plus5pp":
            learned_s["tradeoff_mean"]-
            summary["untrained_q"]["tradeoff_mean"]>=GATES["tradeoff_gain"],
    }
    if any(r["meta_training"] or r["q_updates"] or
           not r["frozen_exams_run"] for r in results):
        raise AssertionError("Evaluation policy was trained")
    if any(r["frozen_exams_run"] for r in train_rows+corrupt_rows):
        raise AssertionError("Training ran a privileged examination")
    for seed in EVAL_SEEDS:
        local=[ix[(seed,arm)] for arm in EVAL_ARMS]
        if len({r["fresh_brain_founder_sha256"] for r in local})!=1:
            raise AssertionError("Brains begin from different founders")
        if len({r["fresh_brain_updates"] for r in local})!=1:
            raise AssertionError("Different brain update budget")
        if len({r["reference_forward_evaluations"] for r in local})!=1:
            raise AssertionError("Memory/computation exposure differs")
    paired={
        compare:paired_bootstrap_ci(results,comparator=compare)
        for compare in (
            "fixed_weak","random_controller",
            "corrupted_transfer","untrained_q")}
    return {
        "status":"M5.21_dev_cross_lifetime_frozen_Q_transfer",
        "training_seed_range":[TRAIN_SEEDS[0],TRAIN_SEEDS[-1]],
        "eval_seed_range":[EVAL_SEEDS[0],EVAL_SEEDS[-1]],
        "trained_lives_per_policy":len(TRAIN_SEEDS),
        "effective_meta_updates_per_policy":len(TRAIN_SEEDS)*28,
        "frozen_eval_lives_per_arm":len(EVAL_SEEDS),
        "training_rollouts_per_policy":len(train_rows),
        "training_exam_never_run":True,
        "evaluation_Q_never_updated":True,
        "brain_genome_memory_restarted_each_life":True,
        "meta_policy_only_transferred_object":True,
        "training_task_grammar_same_as_evaluation":True,
        "explicit_task_labels_and_lab_boundary_consolidation":True,
        "single_neuron_group_changed_per_block":True,
        "same_synaptic_update_and_reference_work_budget":True,
        "fixed_train_test_split_no_tuning_on_test":True,
        "corrupt_reward_is_independent_noise_not_permutation":True,
        "no_structural_plasticity_growth_or_evolution":True,
        "no_sealed_original_M5_6_cohorts":True,
        "Q_trained_SHA256":q_pre_test,
        "Q_corrupt_SHA256":q_corrupt_before,
        "learned_Q_values":learned,
        "corrupt_Q_values":corrupted,
        "summary":summary,
        "paired_95pct_bootstrap":paired,
        "screening_gate":checks,
        "development_screen_pass":all(checks.values()),
        "independent_task_replication_complete":False,
        "evolution_authorized":False,
        "advance_to_evolution":False,
        "training_rows_real":train_rows,
        "training_rows_corrupted":corrupt_rows,
        "frozen_eval_rows":results,
        "rows_sha256":hashlib.sha256(json.dumps(
            results,sort_keys=True,separators=(",",":")).encode()).hexdigest(),
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,default=Path(
        "results/m5-21/meta-training-frozen-transfer-development.json"))
    args=parser.parse_args()
    r=experiment()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(r,sort_keys=True,indent=2)+"\n",
                        encoding="utf-8")
    print("M5_21_META_TRAINING "+json.dumps({
        key:r[key] for key in (
            "status","summary","paired_95pct_bootstrap","screening_gate",
            "development_screen_pass","advance_to_evolution",
            "rows_sha256","Q_trained_SHA256")},sort_keys=True),flush=True)


if __name__=="__main__":
    main()
