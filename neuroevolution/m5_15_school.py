"""M5.15: reward-predictive classroom with bounded autobiographical replay.

This is a diagnostic of the *learning rule*, not an ecology or compositional
intelligence experiment. Trial = sensory pair, action, real binary feedback.
Predictor learns P(success | observation, action) by online backpropagation,
not a correct-action oracle. No action target is ever supplied to learner.

Independent new development seeds 3800..3823. Training covers all four cue
pairs in both benchmark worlds; frozen examination uses new *episodes* but
the same four possible cue pairs, NOT out-of-distribution generalization.
Worlds:
  XOR:     2 actions; success iff action == (x ^ y).
  TERNARY: 3 actions; success iff action == x+y. Negative reward does NOT
           disclose which of the other two actions would have succeeded.

Arms:
  native_reinforce          original M5.6 recurrent policy, historic control
  predict_online            one real-outcome update per interaction
  predict_repeat            five updates of latest experience (compute control)
  predict_replay            latest update + four sampled remembered experiences
  predict_uncoupled         same replay compute but independent coin feedback

The three learning-enabled predictor arms use the SAME initialized MLP,
updates, exploration, episode order and one observation per world step; only
selection of additional training transitions differs. Buffer <= 128 records
of the agent's *actual* (observation, chosen action, observed outcome).
The unsupervised test uses greedy action choice and no learning. Recurrent
state is not reset between training and test for the native arm. No synapse
birth/pruning or genetic selection in this experiment.
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

from .m5_6_online import Learner

SEEDS=tuple(range(3800,3824))
TASKS=("xor","ternary")
ARMS=("native_reinforce","predict_online","predict_repeat",
      "predict_replay","predict_uncoupled")
TRAIN_STEPS=1200
EXAM_STEPS=120
HIDDEN=16
MEMORY_SIZE=128
UPDATES_PER_EXPERIENCE=5
LEARNING_RATE=.075
EXPLORATION_START=.26
EXPLORATION_END=.06
# Limits kept fixed BEFORE running independent seeds.
GATES={
    "xor_exam_min":.85,
    "ternary_exam_min":.75,
    "xor_gain_over_native":.10,
    "ternary_gain_over_native":.10,
    "xor_replay_gain_over_repeat":.04,
    "ternary_replay_gain_over_repeat":.04,
}


def _validate(seed:int,task:str):
    if type(seed) is not int or seed not in SEEDS or task not in TASKS:
        raise ValueError("Independent M5.15 development cohort/task only")


def ticket_stream(seed:int,task:str,steps:int):
    """Exactly balanced x/y combinations per 40-trial exposure block."""
    _validate(seed,task)
    if steps%40:raise ValueError("Only complete 40-trial blocks")
    rng=random.Random(seed*211_001+31*(TASKS.index(task)+1))
    out=[]
    for _ in range(steps//40):
        patterns=[(x,y) for x in (0,1) for y in (0,1)
                  for _ in range(10)]
        rng.shuffle(patterns)
        out.extend(patterns)
    return tuple(out)


def target(task:str,x:int,y:int):
    """Hidden environment consequence only. Never provided to learner."""
    return x^y if task=="xor" else x+y


def num_actions(task:str):
    return 2 if task=="xor" else 3


def features(x:int,y:int,a:int,na:int):
    if a not in range(na) or x not in (0,1) or y not in (0,1):
        raise ValueError("Invalid experienced action/cues")
    return tuple([2.*x-1.,2.*y-1.] +
                 [float(a==i) for i in range(na)] + [1.])


class PredictiveLearner:
    """Small, genuine nonlinear reward predictor trained on own outcomes."""

    def __init__(self,seed:int,task:str,arm:str):
        _validate(seed,task)
        if arm not in ARMS or arm=="native_reinforce":
            raise ValueError("Only predictor arms")
        self.arm=arm
        self.na=num_actions(task)
        self.dim=self.na+3
        rng=random.Random(seed*103_973+TASKS.index(task)*211+7)
        self.w=[[rng.gauss(0.,.45) for _ in range(self.dim)]
                for _ in range(HIDDEN)]
        self.v=[rng.gauss(0.,.13) for _ in range(HIDDEN)]
        self.bias=0.
        self.buffer=deque(maxlen=MEMORY_SIZE)
        self.rng=random.Random(seed*93_773+TASKS.index(task)*107+83)
        self.noise_rng=random.Random(seed*67_231+TASKS.index(task)*137+193)
        self.updates=0
        self.env_experiences=0

    def estimate(self,x:int,y:int,a:int)->float:
        z=features(x,y,a,self.na)
        hidden=[math.tanh(math.fsum(wj*sj for wj,sj in zip(row,z)))
                for row in self.w]
        raw=self.bias+math.fsum(w*h for w,h in zip(self.v,hidden))
        if raw>35:return 1.-1e-12
        if raw< -35:return 1e-12
        return 1./(1.+math.exp(-raw))

    def choose(self,x:int,y:int,episode:int,explore:bool):
        values=[self.estimate(x,y,a) for a in range(self.na)]
        best=max(range(self.na),key=lambda a:(values[a],-a))
        if explore:
            fraction=min(1.,episode/TRAIN_STEPS)
            epsilon=(EXPLORATION_START*(1-fraction)
                     +EXPLORATION_END*fraction)
            if self.rng.random()<epsilon:
                return self.rng.randrange(self.na)
        return best

    def _update_one(self,datum:tuple[int,int,int,int]):
        """Binary cross-entropy gradient from one *experienced* consequence."""
        x,y,a,feedback=datum
        inputs=features(x,y,a,self.na)
        hidden=[math.tanh(math.fsum(wj*sj for wj,sj in zip(row,inputs)))
                for row in self.w]
        raw=self.bias+math.fsum(vj*hj for vj,hj in zip(self.v,hidden))
        prob=1./(1.+math.exp(-max(-35.,min(35.,raw))))
        error=float(feedback)-prob
        if not -1.<=error<=1.:raise AssertionError("Invalid update")
        # Cache old output weights to get the actual simultaneous gradient.
        old_v=self.v[:]
        for j,h in enumerate(hidden):
            self.v[j]=max(-5.,min(5.,self.v[j]+LEARNING_RATE*error*h))
            slope=LEARNING_RATE*error*old_v[j]*(1-h*h)
            for k,feat in enumerate(inputs):
                self.w[j][k]=max(-5.,min(5.,self.w[j][k]+slope*feat))
        self.bias=max(-5.,min(5.,self.bias+LEARNING_RATE*error))
        self.updates+=1

    def experience(self,x:int,y:int,a:int,real_feedback:int):
        if real_feedback not in (0,1):
            raise ValueError("Only actual binary outcomes")
        self.env_experiences+=1
        observed=(self.noise_rng.randrange(2)
                  if self.arm=="predict_uncoupled" else real_feedback)
        datum=(x,y,a,observed)
        self._update_one(datum)
        self.buffer.append(datum)
        if self.arm=="predict_online":
            return
        if self.arm=="predict_repeat":
            for _ in range(UPDATES_PER_EXPERIENCE-1):
                self._update_one(datum)
        else:
            for _ in range(UPDATES_PER_EXPERIENCE-1):
                self._update_one(self.rng.choice(tuple(self.buffer)))


class HistoricalPolicy:
    """Original native REINFORCE learner with same sensory pair/action domain."""
    def __init__(self,seed:int,task:str):
        self.task=task
        rng=random.Random(seed*103_973+TASKS.index(task)*211+7)
        genome=tuple(rng.gauss(0.,.26) for _ in range(260))
        self.agent=Learner.fresh(genome,(5,5))
        self.updates=0
        self.env_experiences=0

    def _observe(self,x:int,y:int):
        raw=[0.]*19
        raw[x]=1.
        raw[2+y]=1.
        return tuple(raw)

    def choice(self,x:int,y:int,episode:int,train:bool,seed:int):
        action,cache=self.agent.action(
            self._observe(x,y),tuple(range(num_actions(self.task))),
            random.Random(seed*997_007+episode*1069+21))
        if not train:
            # Score greedy policy at frozen exam, rather than sampling its
            # softmax (predictors are also greedy during exam).
            return max(cache[3],key=cache[3].get),cache
        return action,cache

    def experience(self,cache,feedback:int):
        self.agent.learn(cache,1. if feedback else -.15,learning_rate=.04)
        self.updates+=1
        self.env_experiences+=1


def run(seed:int,task:str,arm:str):
    _validate(seed,task)
    if arm not in ARMS:raise ValueError("Unregistered treatment")
    predictor=arm!="native_reinforce"
    organism=(PredictiveLearner(seed,task,arm) if predictor
              else HistoricalPolicy(seed,task))
    samples=ticket_stream(seed,task,TRAIN_STEPS+EXAM_STEPS)
    training_accuracy=[]
    training_curve=[]
    for tick,(x,y) in enumerate(samples[:TRAIN_STEPS]):
        if predictor:
            action=organism.choose(x,y,tick,True)
            cache=None
        else:
            action,cache=organism.choice(x,y,tick,True,seed)
        # Correct action stays exclusively in the world. Chosen action
        # and experienced scalar feedback are all the agent receives.
        realized_feedback=int(action==target(task,x,y))
        if predictor:
            organism.experience(x,y,action,realized_feedback)
        else:
            organism.experience(cache,realized_feedback)
        training_accuracy.append(realized_feedback)
        if (tick+1)%120==0:
            training_curve.append(sum(training_accuracy[-120:])/120)
    frozen_params=(tuple(v for row in organism.w for v in row)+
                   tuple(organism.v)+(organism.bias,)
                   if predictor else tuple(organism.agent.weights))
    frozen_updates=organism.updates
    exam_results=[]
    calibration=[]
    for j,(x,y) in enumerate(samples[TRAIN_STEPS:]):
        if predictor:
            choice=organism.choose(x,y,TRAIN_STEPS+j,False)
            estimate=organism.estimate(x,y,choice)
            calibration.append((estimate-float(
                choice==target(task,x,y)))**2)
        else:
            choice,_=organism.choice(x,y,TRAIN_STEPS+j,
                                     False,seed)
        exam_results.append(int(choice==target(task,x,y)))
        # EXAM deliberately does NOT call organism.experience().
    check=(tuple(v for row in organism.w for v in row)+
           tuple(organism.v)+(organism.bias,)
           if predictor else tuple(organism.agent.weights))
    if frozen_params!=check or organism.updates!=frozen_updates:
        raise AssertionError("Frozen school examination updated weights")
    if organism.env_experiences!=TRAIN_STEPS:
        raise AssertionError("Extra environment experience leaked")
    if predictor:
        expected=(TRAIN_STEPS if arm=="predict_online" else
                  TRAIN_STEPS*UPDATES_PER_EXPERIENCE)
        if organism.updates!=expected:raise AssertionError(
            "Replay/repeat computation budget mismatch")
        if len(organism.buffer)!=MEMORY_SIZE:raise AssertionError(
            "Unbounded or missing episodic memory")
    return {
        "seed":seed,"task":task,"arm":arm,
        "train_late":training_curve[-1],
        "train_curve":training_curve,
        "exam_accuracy":sum(exam_results)/EXAM_STEPS,
        "exam_successes":sum(exam_results),
        "exam_brier_on_chosen_action":(
            statistics.mean(calibration) if predictor else None),
        "training_environment_experiences":organism.env_experiences,
        "number_parameter_updates":organism.updates,
        "replay_capacity":MEMORY_SIZE if predictor else 0,
        "frozen_exam":True,
        "exam_sha256":hashlib.sha256(bytes(exam_results)).hexdigest(),
        "individual_memory_not_inherited":True,
        "no_correct_action_labels":True,
    }


def experiment(seeds:tuple[int,...]=SEEDS):
    if not seeds or len(seeds)!=len(set(seeds)) or any(
        type(s) is not int or s not in SEEDS for s in seeds):
        raise ValueError("Only new unique M5.15 development seeds")
    rows=[run(seed,task,arm) for task in TASKS for seed in seeds
          for arm in ARMS]
    index={(r["seed"],r["task"],r["arm"]):r for r in rows}
    summary={}
    for task in TASKS:
        summary[task]={}
        for arm in ARMS:
            group=[index[(seed,task,arm)] for seed in seeds]
            summary[task][arm]={
                "train_late":statistics.mean(r["train_late"] for r in group),
                "exam_accuracy":statistics.mean(r["exam_accuracy"] for r in group),
                "exam_brier_on_chosen_action":(
                    statistics.mean(r["exam_brier_on_chosen_action"]
                        for r in group) if arm!="native_reinforce" else None),
                "parameter_updates":statistics.mean(
                    r["number_parameter_updates"] for r in group),
                "environment_experiences":TRAIN_STEPS,
            }
    xor=summary["xor"]; ternary=summary["ternary"]
    checks={
        "xor_predict_replay_exam_85pct":
            xor["predict_replay"]["exam_accuracy"]>=GATES["xor_exam_min"],
        "ternary_predict_replay_exam_75pct":
            ternary["predict_replay"]["exam_accuracy"]>=
            GATES["ternary_exam_min"],
        "xor_vs_native_10pp":
            xor["predict_replay"]["exam_accuracy"]-
            xor["native_reinforce"]["exam_accuracy"]>=
            GATES["xor_gain_over_native"],
        "ternary_vs_native_10pp":
            ternary["predict_replay"]["exam_accuracy"]-
            ternary["native_reinforce"]["exam_accuracy"]>=
            GATES["ternary_gain_over_native"],
        "xor_replay_vs_equal_compute_repeat_4pp":
            xor["predict_replay"]["exam_accuracy"]-
            xor["predict_repeat"]["exam_accuracy"]>=
            GATES["xor_replay_gain_over_repeat"],
        "ternary_replay_vs_equal_compute_repeat_4pp":
            ternary["predict_replay"]["exam_accuracy"]-
            ternary["predict_repeat"]["exam_accuracy"]>=
            GATES["ternary_replay_gain_over_repeat"],
        "both_predictor_controls_above_uncoupled_feedback":
            all(summary[t]["predict_replay"]["exam_accuracy"]>
                summary[t]["predict_uncoupled"]["exam_accuracy"]+.10
                for t in TASKS),
    }
    return {
        "status":"M5.15_predictive_reward_classroom_development_only",
        "seeds":list(seeds),"tasks":list(TASKS),"arms":list(ARMS),
        "training_steps_per_organism":TRAIN_STEPS,
        "frozen_exam_steps_per_organism":EXAM_STEPS,
        "episodes_same_four_cues_not_unseen_combinations":True,
        "action_count_by_task":{task:num_actions(task) for task in TASKS},
        "bounded_remembered_experiences":MEMORY_SIZE,
        "nominal_predictor_hidden_units":HIDDEN,
        "replay_and_repeat_equal_update_budget":True,
        "feedback_is_own_action_consequence":True,
        "uncoupled_control_receives_independent_bernoulli_feedback":True,
        "structural_growth_or_pruning":False,
        "reproduction_or_evolution":False,
        "summary":summary,"screening_gate":checks,
        "advance_to_evolution":all(checks.values()),
        "raw_rows":rows,
        "rows_sha256":hashlib.sha256(json.dumps(
            rows,sort_keys=True,separators=(",",":")).encode()).hexdigest(),
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,default=Path(
        "results/m5-15/predictive-classroom-development.json"))
    args=parser.parse_args()
    result=experiment()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n",
                        encoding="utf-8")
    print("M5_15_CLASSROOM "+json.dumps({
        k:result[k] for k in ("status","summary","screening_gate",
              "advance_to_evolution","rows_sha256")},sort_keys=True),flush=True)


if __name__=="__main__":
    main()
