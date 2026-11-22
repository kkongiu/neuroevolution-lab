"""M5.16: a school where rules change A -> B -> A.

M5.15 established predictive reward models can learn elementary problems but
bounded historical replay showed no advantage in a stationary environment.
M5.16 tests whether context-stratified autobiographical experience preserves
old competence under *interference* at an EXACTLY equal gradient-update
budget to recent-only repetition and FIFO replay.

The agent observes a context sensory bit (A or B) alongside x,y and a
candidate action; the world privately determines A=x XOR y and B=1-(x XOR y).
Because both contexts reuse the same inputs with opposite target actions,
they can interfere. This is NOT context-discovery or compositional transfer.
All contexts and problem grammars are explicitly provided by researchers.
The hidden correct answer never enters the learner; it sees only scalar
success/failure of the action it actually chose.

After learning A, a FROZEN probe scores A. Then before learning B a frozen
B zero-shot probe is scored. After B, two FROZEN probes score A and B BEFORE
any A refresher. Finally after A return training, FROZEN probes score A,B.
Frozen probes do NOT enter experience buffers, update weights or return reward
to the model. They test familiar cue pairs (new episodes), not novel sensory
families. Context stratification (64 old A / 64 B slots) is engineered
and uses the observed context bit as an indexing key; not emergent memory
allocation. Total buffer size is <=128 in all replay arms.

New disjoint development seeds 3900..3923 (M5.6 sealed sets untouched).
No structural growth/pruning, evolution, predator or ecology experiment.
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

SEEDS=tuple(range(3900,3924))
ARMS=("online","recent_repeat","fifo_replay","context_replay",
      "uncoupled_context_replay")
TRAIN_A=640
TRAIN_B=800
RETRAIN_A=240
PROBE=80
HIDDEN=16
MEMORY_CAP=128
UPDATES=5
LR=.075
EXPLORATION_START=.25
EXPLORATION_END=.07
GATES={
    "baseline_A_learned":.85,
    "baseline_B_learned":.85,
    "A_held_after_B":.75,
    "gain_over_fifo":.10,
    "gain_over_repeat":.10,
    "B_retained_after_A_return":.75,
}


def validate_seed(seed:int)->None:
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError("M5.16 development-only seeds 3900..3923")


def sensory(x:int,y:int,context:int,action:int)->tuple[float,...]:
    if x not in (0,1) or y not in (0,1) or context not in (0,1) or action not in (0,1):
        raise ValueError("Only binary cues/context/action")
    return (2.*x-1.,2.*y-1.,float(context==0),float(context==1),
            float(action==0),float(action==1),1.)


def environment_success(context:int,x:int,y:int,action:int)->int:
    """Ground truth solely in environment, never passed as target action."""
    desired=(x^y) if context==0 else (1^(x^y))
    return int(action==desired)


def cues(seed:int,phase:int,n:int)->tuple[tuple[int,int],...]:
    validate_seed(seed)
    if n%40:raise ValueError("Each phase must have complete 40 cue blocks")
    rng=random.Random(seed*211_001+phase*137+271)
    out=[]
    for _ in range(n//40):
        pairs=[(x,y) for x in (0,1) for y in (0,1)
               for _ in range(10)]
        rng.shuffle(pairs)
        out.extend(pairs)
    return tuple(out)


class SchoolBrain:
    """Same initialized predictor/optimizer in every arm; only replay differs."""

    def __init__(self,seed:int,arm:str):
        validate_seed(seed)
        if arm not in ARMS:raise ValueError("Unknown treatment")
        self.seed=seed
        self.arm=arm
        self.dim=7
        init_rng=random.Random(seed*103_973+7)
        self.w=[[init_rng.gauss(0.,.45) for _ in range(self.dim)]
                for _ in range(HIDDEN)]
        self.v=[init_rng.gauss(0.,.13) for _ in range(HIDDEN)]
        self.bias=0.
        self.action_rng=random.Random(seed*93_773+83)
        self.replay_rng=random.Random(seed*873_331+197)
        self.false_reward_rng=random.Random(seed*67_231+193)
        self.fifo=deque(maxlen=MEMORY_CAP)
        self.context_buffers={
            0:deque(maxlen=MEMORY_CAP//2),
            1:deque(maxlen=MEMORY_CAP//2)}
        self.parameter_updates=0
        self.lived_experiences=0
        self.replayed_old_context_during_B=0

    def digest_state(self):
        return (tuple(v for row in self.w for v in row),
                tuple(self.v),self.bias,self.parameter_updates,
                self.lived_experiences,tuple(self.fifo),
                tuple(self.context_buffers[0]),
                tuple(self.context_buffers[1]))

    def predict(self,x:int,y:int,context:int,action:int)->float:
        z=sensory(x,y,context,action)
        h=[math.tanh(math.fsum(wj*sj for wj,sj in zip(row,z)))
           for row in self.w]
        raw=self.bias+math.fsum(v*h0 for v,h0 in zip(self.v,h))
        raw=max(-35.,min(35.,raw))
        return 1./(1.+math.exp(-raw))

    def choose(self,x:int,y:int,context:int,train_tick:int,explore:bool)->int:
        scores=(self.predict(x,y,context,0),self.predict(x,y,context,1))
        best=int(scores[1]>scores[0])
        if explore:
            frac=min(1.,train_tick/(TRAIN_A+TRAIN_B+RETRAIN_A))
            epsilon=EXPLORATION_START*(1-frac)+EXPLORATION_END*frac
            if self.action_rng.random()<epsilon:
                return self.action_rng.randrange(2)
        return best

    def _update(self,record:tuple[int,int,int,int,int]):
        x,y,context,action,outcome=record
        obs=sensory(x,y,context,action)
        h=[math.tanh(math.fsum(w*feat for w,feat in zip(row,obs)))
           for row in self.w]
        raw=self.bias+math.fsum(w*hv for w,hv in zip(self.v,h))
        prob=1./(1.+math.exp(-max(-35.,min(35.,raw))))
        delta=float(outcome)-prob
        old=self.v[:]
        for j,hj in enumerate(h):
            self.v[j]=max(-5.,min(5.,self.v[j]+LR*delta*hj))
            step=LR*delta*old[j]*(1-hj*hj)
            for k,feat in enumerate(obs):
                self.w[j][k]=max(-5.,min(5.,self.w[j][k]+step*feat))
        self.bias=max(-5.,min(5.,self.bias+LR*delta))
        self.parameter_updates+=1

    def _sample(self,context:int):
        if self.arm=="fifo_replay":
            return self.replay_rng.choice(tuple(self.fifo))
        # Stratify by context, without ground-truth access. The procedure
        # explicitly implements 64+64 episodic memory allocation.
        active=[c for c in (0,1) if self.context_buffers[c]]
        chosen=self.replay_rng.choice(active)
        item=self.replay_rng.choice(tuple(self.context_buffers[chosen]))
        if context==1 and chosen==0:
            self.replayed_old_context_during_B+=1
        return item

    def live(self,x:int,y:int,context:int,action:int,success:int):
        """Only the action's real scalar consequence may enter the learner."""
        if success not in (0,1):raise ValueError("Binary real outcome only")
        self.lived_experiences+=1
        feedback=(self.false_reward_rng.randrange(2)
                  if self.arm=="uncoupled_context_replay" else success)
        record=(x,y,context,action,feedback)
        self._update(record)
        if self.arm in ("fifo_replay","context_replay",
                        "uncoupled_context_replay"):
            self.fifo.append(record)
            self.context_buffers[context].append(record)
        if self.arm=="online":return
        if self.arm=="recent_repeat":
            for _ in range(UPDATES-1):self._update(record)
            return
        for _ in range(UPDATES-1):
            self._update(self._sample(context))


def exam(brain:SchoolBrain,seed:int,context:int,phase:int)->float:
    state=brain.digest_state()
    passed=0
    for x,y in cues(seed,phase,PROBE):
        action=brain.choose(x,y,context,0,False)
        # Grade privately; DO NOT call live or expose feedback to model.
        passed+=environment_success(context,x,y,action)
    if brain.digest_state()!=state:
        raise AssertionError("Frozen memory exam altered personal parameters")
    return passed/PROBE


def train(brain:SchoolBrain,seed:int,context:int,phase:int,n:int,tick:int):
    successes=[]
    for x,y in cues(seed,phase,n):
        action=brain.choose(x,y,context,tick,True)
        reward=environment_success(context,x,y,action)
        brain.live(x,y,context,action,reward)
        successes.append(reward)
        tick+=1
    return tick,sum(successes[-80:])/80


def run(seed:int,arm:str)->dict:
    validate_seed(seed)
    if arm not in ARMS:raise ValueError("Unknown M5.16 intervention")
    brain=SchoolBrain(seed,arm)
    tick=0
    tick,train_A=train(brain,seed,0,0,TRAIN_A,tick)
    pre_A=exam(brain,seed,0,10)
    pre_B=exam(brain,seed,1,11)
    tick,train_B=train(brain,seed,1,1,TRAIN_B,tick)
    A_after_B=exam(brain,seed,0,12)
    B_after_B=exam(brain,seed,1,13)
    buffer_A_after_B=len(brain.context_buffers[0]) if arm in (
        "context_replay","uncoupled_context_replay") else sum(
        r[2]==0 for r in brain.fifo)
    old_rehearsals=brain.replayed_old_context_during_B
    tick,relearn_A=train(brain,seed,0,2,RETRAIN_A,tick)
    A_after_return=exam(brain,seed,0,14)
    B_after_return=exam(brain,seed,1,15)
    if tick!=TRAIN_A+TRAIN_B+RETRAIN_A:
        raise AssertionError("Unexpected hidden training episodes")
    expected=tick if arm=="online" else tick*UPDATES
    if brain.parameter_updates!=expected:
        raise AssertionError("A treatment used an unfair update budget")
    if len(brain.fifo)>MEMORY_CAP or any(
       len(v)>MEMORY_CAP//2 for v in brain.context_buffers.values()):
        raise AssertionError("Exceeded neural memory quota")
    return {
        "seed":seed,"arm":arm,
        "A_train_last80":train_A,"B_train_last80":train_B,
        "A_frozen_after_A":pre_A,
        "B_zero_shot_before_B":pre_B,
        "A_frozen_after_B_before_relearning":A_after_B,
        "B_frozen_after_B":B_after_B,
        "A_frozen_after_A_relearning":A_after_return,
        "B_frozen_after_A_relearning":B_after_return,
        "A_interference_drop":A_after_B-pre_A,
        "B_interference_drop":B_after_return-B_after_B,
        "A_reacquisition_gain":A_after_return-A_after_B,
        "A_record_count_after_B":buffer_A_after_B,
        "old_A_rehearsal_samples_during_B":old_rehearsals,
        "lived_training_steps":brain.lived_experiences,
        "parameter_updates":brain.parameter_updates,
        "buffer_size_at_end":len(brain.fifo),
        "exam_parameters_frozen":True,
        "actions_graded_only_by_environment":True,
        "individual_memory_not_inherited":True,
    }


def experiment(seeds:tuple[int,...]=SEEDS)->dict:
    if not seeds or len(seeds)!=len(set(seeds)) or any(
        type(s) is not int or s not in SEEDS for s in seeds):
        raise ValueError("No reused, sealed or duplicate M5.16 seed")
    rows=[run(seed,arm) for seed in seeds for arm in ARMS]
    ix={(r["seed"],r["arm"]):r for r in rows}
    stats=("A_train_last80","B_train_last80","A_frozen_after_A",
           "B_zero_shot_before_B",
           "A_frozen_after_B_before_relearning","B_frozen_after_B",
           "A_frozen_after_A_relearning","B_frozen_after_A_relearning",
           "A_interference_drop","B_interference_drop",
           "A_reacquisition_gain","A_record_count_after_B",
           "old_A_rehearsal_samples_during_B","parameter_updates")
    summary={arm:{key:statistics.mean(ix[(s,arm)][key] for s in seeds)
                  for key in stats} for arm in ARMS}
    strat=summary["context_replay"]
    fifo=summary["fifo_replay"]
    repeat=summary["recent_repeat"]
    gate={
        "A_preinterference_at_least_85pct":
            strat["A_frozen_after_A"]>=GATES["baseline_A_learned"],
        "B_after_learning_at_least_85pct":
            strat["B_frozen_after_B"]>=GATES["baseline_B_learned"],
        "A_after_B_at_least_75pct":
            strat["A_frozen_after_B_before_relearning"]>=GATES["A_held_after_B"],
        "A_vs_FIFO_plus10pp":
            strat["A_frozen_after_B_before_relearning"]-
            fifo["A_frozen_after_B_before_relearning"]>=
            GATES["gain_over_fifo"],
        "A_vs_recent_repeat_plus10pp":
            strat["A_frozen_after_B_before_relearning"]-
            repeat["A_frozen_after_B_before_relearning"]>=
            GATES["gain_over_repeat"],
        "B_after_A_relearning_at_least_75pct":
            strat["B_frozen_after_A_relearning"]>=
            GATES["B_retained_after_A_return"],
        "context_replay_beats_shuffled_feedback":
            strat["A_frozen_after_B_before_relearning"]>
            summary["uncoupled_context_replay"][
                "A_frozen_after_B_before_relearning"]+.10,
    }
    for seed in seeds:
        for arm in ("recent_repeat","fifo_replay","context_replay",
                    "uncoupled_context_replay"):
            if ix[(seed,arm)]["parameter_updates"]!=(
                TRAIN_A+TRAIN_B+RETRAIN_A)*UPDATES:
                raise AssertionError("Different compute update dose")
        if ix[(seed,"fifo_replay")]["A_record_count_after_B"]!=0:
            raise AssertionError("FIFO still retained stale A after long B")
        if ix[(seed,"context_replay")]["A_record_count_after_B"]!=64:
            raise AssertionError("Stratified replay forgot earlier context")
    return {
        "status":"M5.16_development_context_balanced_autobiographical_replay",
        "seeds":list(seeds),"arms":list(ARMS),
        "curriculum":[("A",TRAIN_A),("B",TRAIN_B),("A",RETRAIN_A)],
        "probe_steps_per_exam":PROBE,
        "context_is_observable":True,"fixed_task_grammar":True,
        "same_four_sensory_patterns_at_exam":True,
        "reward_only_from_own_action":True,
        "frozen_return_probes_before_retraining":True,
        "episodic_memory_capacity":MEMORY_CAP,
        "equal_update_budgets_between_replay_and_repeat":True,
        "no_synapse_birth_death_or_reproduction":True,
        "summary":summary,"screening_gate":gate,
        "advance_to_evolution":all(gate.values()),
        "raw_rows":rows,
        "rows_sha256":hashlib.sha256(json.dumps(
            rows,sort_keys=True,separators=(",",":")).encode()).hexdigest(),
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,default=Path(
        "results/m5-16/continual-learning-school-development.json"))
    args=parser.parse_args()
    result=experiment()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n",
                        encoding="utf-8")
    print("M5_16_CONTINUAL_SCHOOL "+json.dumps({
        k:result[k] for k in ("status","summary","screening_gate",
              "advance_to_evolution","rows_sha256")},sort_keys=True),flush=True)


if __name__=="__main__":
    main()
