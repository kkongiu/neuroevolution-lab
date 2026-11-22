"""M5.23: separate episodic MEMORY, an addressable READER and a DIRECTOR.

Different from M5.22: a frozen subnetwork was NOT a reliably retrievable
memory. Here the archive is explicitly written after experienced reward and
indexed via 4 trainable addressing similarities (context, x, y, action).
A fast 16-unit reward predictor learns online. The director has 10 weights
and learns how to combine the predictor's success estimate with the archive
estimate using feedback from only the organism's chosen action.

Ablations:
  fast_only                Archive can write/read, its output is ignored.
  memory_only              Use only observed archived episode estimates.
  fixed_fusion             Same archive, immutable address and 50/50 gate.
  fixed_address_director   Train the director, fix the archive query.
  learned_director         Train reader addressing and director jointly.
  context_blind_director   Train both, but archive cannot address context.
  corrupted_memory_director Store independent random outcomes in archive.
  replay_reference         Full model + 4 historical gradient updates per
                           lived action (5x budget; NOT a fair comparator).

Except replay_reference, each episode causes ONE weight-update call.
Active gradient allocations differ across ablations and are disclosed.
All arms start with identical parameters; they have 128 memory slots
partitioned by the explicit (researcher-supplied) context signal, including
43 records for A, 43 for B and 42 for C. Capacity and archive-read work
are matched across the non-replay arms. The "write" function stores ONLY
what the agent actually chose and observed (never an oracle answer).

Development seeds 4700..4723 are disjoint from all previous experiments.
Frozen exams A -> B -> C -> A -> B evaluate readout before refresher.
This is a deterministic four-binary-cue school with explicit context
identities, NOT a proof of autonomous context discovery, transfer to
unseen concepts, natural-world cognitive plasticity, or evolution.
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

SEEDS=tuple(range(4700,4724))
ARMS=("fast_only","memory_only","fixed_fusion",
      "fixed_address_director","learned_director",
      "context_blind_director","corrupted_memory_director",
      "replay_reference")
PHASES=(("A",400),("B",400),("C",400),("A",160),("B",160))
CONTEXT={"A":0,"B":1,"C":2}
TRAIN_STEPS=sum(n for _,n in PHASES)
BLOCK=40
EXAM=80
HIDDEN=16
DIM=8
FAST_PARAMETERS=HIDDEN*DIM+HIDDEN+1
ADDRESS_PARAMETERS=4
DIRECTOR_DIM=DIM+2
TOTAL_PARAMETERS=FAST_PARAMETERS+ADDRESS_PARAMETERS+DIRECTOR_DIM
LR_FAST=.067
LR_ADDRESS=.075
LR_DIRECTOR=.11
MEMORY_SLOTS=(43,43,42)
MEMORY_CAP=sum(MEMORY_SLOTS)
PSEUDO_OBSERVATIONS=.3
ADDRESS_INIT=(3.5,2.5,2.5,2.5)
GATES={
    "learn_A":.90,
    "retain_A":.90,
    "learn_B":.90,
    "retain_B":.90,
    "learn_C":.90,
    "archive_retrievable":.90,
    "delta_fast_only":.10,
    "delta_fixed_fusion":.05,
    "delta_memory_only":.05,
    "delta_address_fixed":.05,
    "delta_context_blind":.05,
    "delta_corrupted_memory":.10,
}


def validate_seed(seed:int)->None:
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError("M5.23 development-only seeds 4700..4723")


def cues(seed:int,phase:int,n:int):
    validate_seed(seed)
    if n%BLOCK:raise ValueError("Need complete balanced blocks")
    rng=random.Random(seed*211_001+phase*271+71)
    out=[]
    for _ in range(n//BLOCK):
        samples=[(x,y) for x in (0,1) for y in (0,1)
                 for _ in range(BLOCK//4)]
        rng.shuffle(samples)
        out.extend(samples)
    return tuple(out)


def observations(x:int,y:int,ctx:int,act:int):
    if x not in (0,1) or y not in (0,1) or ctx not in (0,1,2) or act not in (0,1):
        raise ValueError("Invalid observed context, binary cue or action")
    return (2.*x-1.,2.*y-1.)+tuple(float(ctx==i) for i in range(3))+(
        float(act==0),float(act==1),1.)


def environment_reward(ctx:int,x:int,y:int,act:int):
    """Privileged world answer stays here; learner sees only actual reward."""
    if ctx==0:best=x^y
    elif ctx==1:best=1^(x^y)
    elif ctx==2:best=x
    else:raise ValueError("Unknown environment")
    return int(act==best)


def logistic(value:float):
    return 1./(1.+math.exp(-max(-35.,min(35.,value))))


class SpecializedBrain:
    """Slow autobiographical store, trained reader, fast brain, trained director."""

    def __init__(self,seed:int,arm:str):
        validate_seed(seed)
        if arm not in ARMS:raise ValueError("Unsupported M5.23 arm")
        self.seed=seed
        self.arm=arm
        rng=random.Random(seed*103_973+7)
        self.fast=[rng.gauss(0.,.45)
                   for _ in range(HIDDEN*DIM)]
        self.fast+=[rng.gauss(0.,.13) for _ in range(HIDDEN)]
        self.fast.append(0.)
        self.address=list(ADDRESS_INIT)
        self.director=[0.]*DIRECTOR_DIM
        self.director[DIM-1]=-.1
        self.director[DIM]=.5
        self.founder=(tuple(self.fast),tuple(self.address),
                      tuple(self.director))
        self.action_rng=random.Random(seed*93_773+83)
        self.replay_rng=random.Random(seed*873_331+197)
        self.corruption_rng=random.Random(seed*67_231+193)
        self.memory=tuple(deque(maxlen=n) for n in MEMORY_SLOTS)
        self.lived=0
        self.gradient_update_calls=0
        self.fast_gradient_update_calls=0
        self.address_gradient_update_calls=0
        self.director_gradient_update_calls=0
        self.historical_replay_updates=0
        self.read_record_evaluations_training=0
        self.read_record_evaluations_exam=0
        self.director_gate_sum=0.
        self.training_decisions=0

    def frozen_state(self):
        return (tuple(self.fast),tuple(self.address),tuple(self.director),
                tuple(tuple(v) for v in self.memory),
                self.lived,self.gradient_update_calls,
                self.fast_gradient_update_calls,
                self.address_gradient_update_calls,
                self.director_gradient_update_calls,
                self.action_rng.getstate(),self.replay_rng.getstate(),
                self.corruption_rng.getstate(),
                self.director_gate_sum,self.training_decisions)

    def _fast(self,z):
        hidden=[math.tanh(math.fsum(self.fast[j*DIM+k]*v
                  for k,v in enumerate(z))) for j in range(HIDDEN)]
        raw=self.fast[-1]+math.fsum(
            self.fast[HIDDEN*DIM+j]*hidden[j]
            for j in range(HIDDEN))
        return logistic(raw),hidden

    def _reader(self,x:int,y:int,ctx:int,action:int,exam:bool):
        """Similarity-weighted actual outcome estimate and address gradient.

        Matching features are explicitly engineered; their magnitudes can
        learn. Query always scans the same capped memory in every arm.
        """
        weighted=PSEUDO_OBSERVATIONS*.5
        mass=PSEUDO_OBSERVATIONS
        candidates=[]
        blind=self.arm=="context_blind_director"
        for bucket in self.memory:
            for oldx,oldy,oldctx,oldaction,observed in bucket:
                mismatch=(
                    0. if blind else float(ctx!=oldctx),
                    float(x!=oldx),float(y!=oldy),
                    float(action!=oldaction))
                score=-math.fsum(w*m for w,m in zip(self.address,mismatch))
                influence=math.exp(max(-35.,min(0.,score)))
                mass+=influence
                weighted+=influence*observed
                candidates.append((influence,observed,mismatch))
        if exam:
            self.read_record_evaluations_exam+=len(candidates)
        else:
            self.read_record_evaluations_training+=len(candidates)
        prediction=weighted/mass
        derivatives=[0.]*ADDRESS_PARAMETERS
        for influence,observed,mismatch in candidates:
            residual=observed-prediction
            for k in range(ADDRESS_PARAMETERS):
                derivatives[k]-=influence*residual*mismatch[k]/mass
        # Confidence is a soft function of empirical support, not an
        # exact-match oracle or privileged knowledge of correct action.
        confidence=1.-PSEUDO_OBSERVATIONS/mass
        return prediction,confidence,derivatives

    def _components(self,x,y,ctx,action,exam:bool):
        z=observations(x,y,ctx,action)
        fast,h=self._fast(z)
        memory,confidence,daddress=self._reader(x,y,ctx,action,exam)
        # These two summary features are stop-gradient in backprop;
        # no special causal label or task answer enters.
        director_input=z+(confidence,abs(fast-memory))
        gate=logistic(math.fsum(
            w*v for w,v in zip(self.director,director_input)))
        if self.arm=="fast_only":gate=0.
        elif self.arm=="memory_only":gate=1.
        elif self.arm=="fixed_fusion":gate=.5
        prediction=(1.-gate)*fast+gate*memory
        return (prediction,fast,memory,gate,z,h,
                director_input,daddress,confidence)

    def choose(self,x,y,ctx,tick,training:bool,*,readout="policy"):
        c0=self._components(x,y,ctx,0,not training)
        c1=self._components(x,y,ctx,1,not training)
        if readout=="policy":
            p0,p1=c0[0],c1[0]
        elif readout=="archive":
            p0,p1=c0[2],c1[2]
        elif readout=="fast":
            p0,p1=c0[1],c1[1]
        else:raise ValueError("Only policy/archive/fast readout")
        chosen=int(p1>p0)
        if training:
            fraction=min(1.,tick/TRAIN_STEPS)
            epsilon=.25*(1.-fraction)+.07*fraction
            if self.action_rng.random()<epsilon:
                chosen=self.action_rng.randrange(2)
        return chosen

    def _learn(self,record):
        x,y,ctx,action,realized=record
        p,pfast,pmem,gate,z,h,director_input,daddr,_=(
            self._components(x,y,ctx,action,False))
        # Maximize log likelihood of OWN experienced reward, not a
        # supervised correct-action target.
        p=max(1e-7,min(1.-1e-7,p))
        common=(realized-p)/(p*(1.-p))
        fast_scale=common*(1.-gate)*pfast*(1.-pfast)
        old=self.fast[:]
        if self.arm!="memory_only":
            for j,hj in enumerate(h):
                out=HIDDEN*DIM+j
                self.fast[out]=max(-5.,min(5.,
                    old[out]+LR_FAST*fast_scale*hj))
                slope=fast_scale*old[out]*(1.-hj*hj)
                for k,feat in enumerate(z):
                    i=j*DIM+k
                    self.fast[i]=max(-5.,min(5.,
                        old[i]+LR_FAST*slope*feat))
            self.fast[-1]=max(-5.,min(5.,old[-1]+LR_FAST*fast_scale))
            self.fast_gradient_update_calls+=1
        if self.arm in ("learned_director","context_blind_director",
                        "corrupted_memory_director","replay_reference",
                        "fixed_address_director"):
            director_scale=common*(pmem-pfast)*gate*(1.-gate)
            for k,feat in enumerate(director_input):
                self.director[k]=max(-6.,min(6.,
                    self.director[k]+LR_DIRECTOR*director_scale*feat))
            self.director_gradient_update_calls+=1
        if self.arm in ("learned_director","context_blind_director",
                        "corrupted_memory_director","replay_reference"):
            for k in range(ADDRESS_PARAMETERS):
                if self.arm=="context_blind_director" and k==0:
                    continue
                self.address[k]=max(.0,min(7.,
                    self.address[k]+LR_ADDRESS*common*gate*daddr[k]))
            self.address_gradient_update_calls+=1
        self.director_gate_sum+=gate
        self.gradient_update_calls+=1

    def live(self,x,y,ctx,action,success):
        if success not in (0,1):raise ValueError("Only real reward 0 or 1")
        record=(x,y,ctx,action,success)
        self._learn(record)
        self.training_decisions+=1
        self.lived+=1
        # The archive is written only AFTER a real consequence arrived.
        # A corrupter may replace only archived feedback, never the fast
        # brain's true current-action training target.
        stored=(
            self.corruption_rng.randrange(2)
            if self.arm=="corrupted_memory_director" else success)
        self.memory[ctx].append((x,y,ctx,action,stored))
        if self.arm=="replay_reference":
            candidates=[rec for buf in self.memory for rec in buf]
            for _ in range(4):
                self._learn(self.replay_rng.choice(candidates))
                self.historical_replay_updates+=1


def exam(brain,seed,ctx,phase,readout="policy"):
    before=brain.frozen_state()
    passed=0
    for x,y in cues(seed,phase,EXAM):
        a=brain.choose(x,y,ctx,0,False,readout=readout)
        passed+=environment_reward(ctx,x,y,a)
    if before!=brain.frozen_state():
        raise AssertionError("Exam modified parameters/memory/action randomness")
    return passed/EXAM


def run(seed:int,arm:str):
    validate_seed(seed)
    brain=SpecializedBrain(seed,arm)
    scores={}
    archive_scores={}
    fast_scores={}
    tick=0
    for phase,(name,n) in enumerate(PHASES):
        ctx=CONTEXT[name]
        for x,y in cues(seed,phase,n):
            a=brain.choose(x,y,ctx,tick,True)
            reward=environment_reward(ctx,x,y,a)
            brain.live(x,y,ctx,a,reward)
            tick+=1
        for seen in range(max(CONTEXT[x[0]] for x in PHASES[:phase+1])+1):
            key=f"phase_{phase}_{'ABC'[seen]}"
            scores[key]=exam(brain,seed,seen,20+phase*3+seen)
            archive_scores[key]=exam(brain,seed,seen,20+phase*3+seen,
                                     readout="archive")
            fast_scores[key]=exam(brain,seed,seen,20+phase*3+seen,
                                  readout="fast")
    if brain.lived!=TRAIN_STEPS or tick!=TRAIN_STEPS:
        raise AssertionError("Unfair organism lived experience budget")
    if sum(map(len,brain.memory))>MEMORY_CAP:
        raise AssertionError("Autobiographical memory overflow")
    expected=TRAIN_STEPS*(5 if arm=="replay_reference" else 1)
    if brain.gradient_update_calls!=expected:
        raise AssertionError("Gradient update count mismatch")
    if arm=="replay_reference":
        if brain.historical_replay_updates!=4*TRAIN_STEPS:
            raise AssertionError("Unfair replay treatment dose")
    elif brain.historical_replay_updates:
        raise AssertionError("Unregistered old experience weight replay")
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
        "A_archive_retrieval_after_C":archive_scores["phase_2_A"],
        "B_archive_retrieval_after_C":archive_scores["phase_2_B"],
        "A_fast_readout_after_C":fast_scores["phase_2_A"],
        "B_fast_readout_after_C":fast_scores["phase_2_B"],
        "joint_pre_refresher":statistics.mean((
            scores["phase_2_A"],scores["phase_1_B"],
            scores["phase_2_B"],scores["phase_2_C"])),
        "minimum_pre_refresher":min(
            scores["phase_2_A"],scores["phase_1_B"],
            scores["phase_2_B"],scores["phase_2_C"]),
        "final_three_mean":statistics.mean((
            scores["phase_4_A"],scores["phase_4_B"],
            scores["phase_4_C"])),
        "lived_actions":brain.lived,
        "gradient_update_calls":brain.gradient_update_calls,
        "fast_gradient_updates":brain.fast_gradient_update_calls,
        "address_gradient_updates":brain.address_gradient_update_calls,
        "director_gradient_updates":brain.director_gradient_update_calls,
        "training_read_record_evals":brain.read_record_evaluations_training,
        "frozen_read_record_evals":brain.read_record_evaluations_exam,
        "memory_slots":MEMORY_CAP,
        "memory_stored_records":sum(map(len,brain.memory)),
        "old_episode_weight_replay_updates":brain.historical_replay_updates,
        "mean_director_archive_gate":brain.director_gate_sum/
                                    brain.gradient_update_calls,
        "final_address_weights":list(brain.address),
        "frozen_exams":True,
        "archive_write_from_actual_choice_only":
            arm!="corrupted_memory_director",
        "no_oracle_correct_action_targets":True,
        "task_identity_externally_visible":True,
    }


def paired_bootstrap(rows,treat,control,metric="joint_pre_refresher",
                     repetitions=1200):
    ix={(r["seed"],r["arm"]):r for r in rows}
    differences=[ix[(seed,treat)][metric]-ix[(seed,control)][metric]
                 for seed in SEEDS]
    rand=random.Random(102938)
    samples=sorted(statistics.mean(rand.choice(differences)
                   for _ in differences) for _ in range(repetitions))
    return {
        "delta":statistics.mean(differences),
        "boot95":[samples[int(.025*repetitions)],
                  samples[int(.975*repetitions)-1]],
        "paired_seeds":len(SEEDS),
    }


def experiment(seeds=SEEDS):
    if tuple(seeds)!=SEEDS:
        raise ValueError("M5.23 only full frozen development cohort")
    rows=[run(seed,arm) for seed in seeds for arm in ARMS]
    ix={(r["seed"],r["arm"]):r for r in rows}
    metrics=("A_after_A","A_after_B","B_after_B",
             "A_after_C","B_after_C","C_after_C",
             "A_archive_retrieval_after_C","B_archive_retrieval_after_C",
             "A_fast_readout_after_C","B_fast_readout_after_C",
             "joint_pre_refresher","minimum_pre_refresher",
             "final_three_mean","gradient_update_calls",
             "fast_gradient_updates","address_gradient_updates",
             "director_gradient_updates","mean_director_archive_gate",
             "training_read_record_evals","frozen_read_record_evals")
    summary={
        arm:{key:statistics.mean(ix[(seed,arm)][key]
                                 for seed in seeds)
             for key in metrics}
        for arm in ARMS
    }
    good=summary["learned_director"]
    checks={
        "specialization_A_learned_90":
            good["A_after_A"]>=GATES["learn_A"],
        "specialization_A_after_C_90":
            good["A_after_C"]>=GATES["retain_A"],
        "specialization_B_first_learned_90":
            good["B_after_B"]>=GATES["learn_B"],
        "specialization_B_after_C_90":
            good["B_after_C"]>=GATES["retain_B"],
        "specialization_C_first_learned_90":
            good["C_after_C"]>=GATES["learn_C"],
        "stored_A_retrievable_after_C_90":
            good["A_archive_retrieval_after_C"]>=
            GATES["archive_retrievable"],
        "stored_B_retrievable_after_C_90":
            good["B_archive_retrieval_after_C"]>=
            GATES["archive_retrievable"],
        "learned_vs_fast_only_plus10pp":
            good["joint_pre_refresher"]-
            summary["fast_only"]["joint_pre_refresher"]>=
            GATES["delta_fast_only"],
        "learned_vs_fixed_fusion_plus5pp":
            good["joint_pre_refresher"]-
            summary["fixed_fusion"]["joint_pre_refresher"]>=
            GATES["delta_fixed_fusion"],
        "learned_vs_memory_only_plus5pp":
            good["joint_pre_refresher"]-
            summary["memory_only"]["joint_pre_refresher"]>=
            GATES["delta_memory_only"],
        "learned_address_vs_fixed_address_plus5pp":
            good["joint_pre_refresher"]-
            summary["fixed_address_director"]["joint_pre_refresher"]>=
            GATES["delta_address_fixed"],
        "context_query_vs_context_blind_plus5pp":
            good["joint_pre_refresher"]-
            summary["context_blind_director"]["joint_pre_refresher"]>=
            GATES["delta_context_blind"],
        "true_memories_vs_random_contents_plus10pp":
            good["joint_pre_refresher"]-
            summary["corrupted_memory_director"]["joint_pre_refresher"]>=
            GATES["delta_corrupted_memory"],
    }
    for seed in seeds:
        arms=[ix[(seed,arm)] for arm in ARMS
              if arm!="replay_reference"]
        if {r["gradient_update_calls"] for r in arms}!={TRAIN_STEPS}:
            raise AssertionError("Wrong update budget for nonreplay arms")
        if len({r["training_read_record_evals"] for r in arms})!=1:
            raise AssertionError("Different read compute dose across arms")
        if len({r["frozen_read_record_evals"] for r in arms})!=1:
            raise AssertionError("Different frozen read compute dose")
        if len({r["memory_stored_records"] for r in arms})!=1:
            raise AssertionError("Different archived record budget")
    return {
        "status":"M5.23_DEV_memory_reader_director_specialization",
        "seeds":list(seeds),
        "arms":list(ARMS),
        "curriculum":list(PHASES),
        "total_parameters_all_arms":TOTAL_PARAMETERS,
        "fast_hidden_neurons":HIDDEN,
        "director_weights":DIRECTOR_DIM,
        "reader_similarity_weights":ADDRESS_PARAMETERS,
        "memory_max_records":MEMORY_CAP,
        "all_arms_preserve_same_personal_archive_cap":True,
        "archive_context_partition_is_lab_programmed":True,
        "memory_read_and_write_ablations":True,
        "read_budget_matched_across_nonreplay_arms":True,
        "same_lived_experiences_all_arms":True,
        "same_update_calls_except_replay":True,
        "active_gradient_parameter_dose_differs":True,
        "replay_reference_extra_4_updates_per_lived_event":True,
        "frozen_exams_never_train":True,
        "same_four_familiar_input_patterns_per_context":True,
        "context_identity_externally_given":True,
        "no_structural_growth_evolution_or_ecology":True,
        "no_sealed_original_M5_6_cohorts":True,
        "summary":summary,"screening_gate":checks,
        "memory_retrieval_supported":all(checks[k]
            for k in checks if k.startswith("stored_")),
        "executive_added_value_screen":all(checks[k]
            for k in checks if k.startswith("learned_vs_") or
            k.startswith("learned_address_vs_")),
        "development_screen_pass":all(checks.values()),
        "independent_task_replication_complete":False,
        "evolution_authorized":False,
        "advance_to_evolution":False,
        "paired_bootstrap":{
            "specialized_vs_fast_only":paired_bootstrap(
                rows,"learned_director","fast_only"),
            "specialized_vs_memory_only":paired_bootstrap(
                rows,"learned_director","memory_only"),
            "specialized_vs_fixed_fusion":paired_bootstrap(
                rows,"learned_director","fixed_fusion"),
            "specialized_vs_address_fixed":paired_bootstrap(
                rows,"learned_director","fixed_address_director"),
            "specialized_vs_context_blind":paired_bootstrap(
                rows,"learned_director","context_blind_director"),
            "specialized_vs_corrupted_memory":paired_bootstrap(
                rows,"learned_director","corrupted_memory_director"),
        },
        "raw_rows":rows,
        "rows_sha256":hashlib.sha256(json.dumps(
            rows,sort_keys=True,separators=(",",":")).encode()).hexdigest(),
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,default=Path(
        "results/m5-23/memory-reader-director-development.json"))
    args=p.parse_args()
    report=experiment()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(
        report,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    print("M5_23_MEMORY_DIRECTOR "+json.dumps({
        k:report[k] for k in (
            "status","summary","screening_gate",
            "memory_retrieval_supported","executive_added_value_screen",
            "development_screen_pass","advance_to_evolution",
            "paired_bootstrap","rows_sha256")},sort_keys=True),flush=True)


if __name__=="__main__":
    main()
