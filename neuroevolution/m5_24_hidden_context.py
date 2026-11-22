"""M5.24: Hidden-context executive learns WHICH memory bank to retrieve.

The teacher supplies x,y and ONLY success/failure of the organism's chosen
binary action; it DOES NOT supply the current A/B/C world identifier.
The environment switches secretly between the SAME three XOR/opposite-XOR/
copy-x worlds. Durations vary by seed to prevent clock-based stage detection.

Roles:
* fast neural success predictor (12 tanh units, no context input)
* 128-record bounded episodic archive in up to three LATENT banks
* memory reader estimating chosen-action success from own-action outcomes
* executive inferring current bank from recent experienced outcomes
* learned logistic director combining fast and memory predictions

The executive's change detector is a HAND-CODED evidence algorithm, NOT
a learned neural regime recognizer. Latent bank identity is internally
inferred; known-context oracle is an explicit privileged upper-bound arm.
The bank reader exploits the STRUCTURAL ASSUMPTION that the two possible
actions have complementary deterministic binary rewards: a failed trial
on action 0 supports action 1. This is a deduction from an experienced
outcome under the stated world assumptions, not an unseen oracle reward.
It will not generalize unmodified to stochastic/noncomplementary worlds.

Frozen examinations give 16 *real, observed* priming action rewards only to
an EPHEMERAL context inference state, then 80 scored decisions without
feedback. No brain weights or episodic memory update on examinations.
Separate no-priming exams expose the unavoidable ambiguity when the same
x,y observation corresponds to opposite rules in hidden worlds.

No ecological evolution, growth or original M5.6 sealed cohorts. DEV-only
seeds 4800..4823, explicitly disjoint from M5.9..M5.23 cohorts.
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

SEEDS=tuple(range(4800,4824))
ARMS=(
    "latent_director",       # inferred latent banks, learned fusion
    "latent_memory_only",    # inferred latent banks, memory readout only
    "latent_fixed_fusion",   # inferred latent banks, 50/50 mixture
    "latent_short_history",  # 3 history episodes, no effective time evidence
    "no_regime_detection",   # one archive, different worlds mixed together
    "scrambled_detection",   # detector gets independent randomized feedback
    "fast_only",             # no archive readout (equal archive work)
    "oracle_context",        # privileged true label, diagnostic ceiling ONLY
)
CONTEXT={"A":0,"B":1,"C":2}
PHASE_NAMES=("A","B","C","A","B")
BLOCK=40
PROBE=16
EXAM=80
FAST_HIDDEN=12
DIM=5        # bipolar x/y, one-hot action 0/1, bias
FAST_PARAMS=FAST_HIDDEN*DIM+FAST_HIDDEN+1
DIRECTOR_PARAMS=5   # intercept, confidence, disagreement, fast, memory
LR_FAST=.065
LR_GATE=.10
MEMORY_SLOTS=(43,43,42)
MEMORY_LIMIT=sum(MEMORY_SLOTS)
MAX_BANKS=3
HISTORY=14
MIN_EVIDENCE=9
MIN_CONFLICTS=3
SWITCH_MATCHES=6
READ_PRIOR=.12
CHANGE_CONFIDENCE=.78
GATES={
    "A_after_C":.90,
    "B_after_C":.90,
    "B_new_learning":.85,
    "C_new_learning":.85,
    "all_after_probe":.90,
    "same_context_as_oracle_within":.07,
    "latent_vs_mixed_delta":.12,
    "latent_vs_scrambled_delta":.12,
    "learned_vs_memory_only_delta":.05,
    "learned_vs_fixed_fusion_delta":.05,
}


def check_seed(seed:int):
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError("Only new DEV seeds 4800..4823")


def schedule(seed:int):
    check_seed(seed)
    rng=random.Random(seed*337+179)
    # No duration is exposed to any non-oracle controller.
    return tuple((name,rng.choice((320,360,400,440))
                  if phase<3 else rng.choice((160,200,240)))
                 for phase,name in enumerate(PHASE_NAMES))


def observations(seed:int,phase:int,n:int):
    check_seed(seed)
    if n%BLOCK:raise ValueError("Must request complete balanced blocks")
    rng=random.Random(seed*211_001+phase*271+71)
    out=[]
    for _ in range(n//BLOCK):
        block=[(x,y) for x in (0,1) for y in (0,1) for _ in range(10)]
        rng.shuffle(block)
        out.extend(block)
    return tuple(out)


def environmental_success(world:str,x:int,y:int,action:int):
    """Never pass world name into the ordinary learner."""
    if world=="A":correct=x^y
    elif world=="B":correct=1^(x^y)
    elif world=="C":correct=x
    else:raise ValueError("Invalid world")
    return int(action==correct)


def sigmoid(x):
    return 1./(1.+math.exp(-max(-35.,min(35.,x))))


def sensory(x,y,action):
    if x not in (0,1) or y not in (0,1) or action not in (0,1):
        raise ValueError("Binary sensory data and chosen action only")
    return (2.*x-1.,2.*y-1.,float(action==0),float(action==1),1.)


class LatentBank:
    """Inference-created bank; no true world labels, only witnessed episodes."""

    def __init__(self,capacity):
        self.records=deque(maxlen=capacity)

    def append(self,record):
        if (len(record)!=4 or any(v not in (0,1) for v in record)):
            raise ValueError("Only binary lived (x,y,chosen_action,reward)")
        self.records.append(record)

    def response(self,x,y,action):
        # Deterministically complementary binary reward is a laboratory
        # structural assumption. All inference comes from observed outcomes.
        count=0
        reward_zero_total=0
        for oldx,oldy,oldaction,reward in self.records:
            if oldx!=x or oldy!=y:continue
            reward_zero_total+=reward if oldaction==0 else (1-reward)
            count+=1
        p0=(READ_PRIOR+reward_zero_total)/(2*READ_PRIOR+count)
        return (p0 if action==0 else 1.-p0),count

    def evidence(self,history):
        """Supported classification: how many past episodes are predicted.

        Unknown (x,y) pairs contribute no support and cannot establish
        which world is active. Evaluation uses only old witnessed records.
        """
        matches=0
        conflicts=0
        supported=0
        for x,y,chosen,reward in history:
            predicted,n=self.response(x,y,chosen)
            if n<2:continue
            supported+=1
            if (predicted if reward==1 else 1.-predicted)>=CHANGE_CONFIDENCE:
                matches+=1
            else:
                conflicts+=1
        return matches,conflicts,supported


class Organism:
    def __init__(self,seed:int,arm:str):
        check_seed(seed)
        if arm not in ARMS:raise ValueError("Unknown hidden-context arm")
        self.seed=seed
        self.arm=arm
        init=random.Random(seed*103_973+7)
        self.fast=([init.gauss(0.,.45)
                    for _ in range(FAST_HIDDEN*DIM)]
                   +[init.gauss(0.,.13) for _ in range(FAST_HIDDEN)]
                   +[0.])
        self.director=[.0,2.,-.4,0.,0.]
        self.founder=(tuple(self.fast),tuple(self.director))
        self.rng=random.Random(seed*93_773+83)
        self.noise_rng=random.Random(seed*67_231+193)
        self.banks=[LatentBank(MEMORY_SLOTS[0])]
        if arm=="oracle_context" or arm=="scrambled_detection" or arm=="no_regime_detection":
            # Oracle banks are allocated as needed by explicit privileged
            # phase labels; others can only create new banks from evidence.
            pass
        self.active=0
        self.history=deque(maxlen=3 if arm=="latent_short_history" else HISTORY)
        self.lived=0
        self.update_calls=0
        self.fast_updates=0
        self.gate_updates=0
        self.read_record_evaluations=0
        self.inferred_switches=[]
        self.unassigned_surprises=0
        self.gate_sum=0.

    def snapshot(self):
        return (tuple(self.fast),tuple(self.director),
                tuple(tuple(b.records) for b in self.banks),
                self.active,tuple(self.history),self.lived,
                self.update_calls,self.fast_updates,self.gate_updates,
                self.rng.getstate(),self.noise_rng.getstate(),
                tuple(self.inferred_switches),self.unassigned_surprises,
                self.gate_sum)

    def _fast_score(self,x,y,a):
        z=sensory(x,y,a)
        h=[math.tanh(math.fsum(
            self.fast[j*DIM+k]*feat for k,feat in enumerate(z)))
            for j in range(FAST_HIDDEN)]
        raw=self.fast[-1]+math.fsum(
            self.fast[FAST_HIDDEN*DIM+j]*v for j,v in enumerate(h))
        return sigmoid(raw),z,h

    def _bank_score(self,bank:int,x,y,a,*,count_work:bool):
        if bank<0 or bank>=len(self.banks):
            return .5,0
        if count_work:
            self.read_record_evaluations+=len(self.banks[bank].records)
        return self.banks[bank].response(x,y,a)

    def _components(self,x,y,a,bank:int,*,count_work:bool):
        fast,z,h=self._fast_score(x,y,a)
        # All arms perform the same one-bank read work when memory is
        # available; tracked by the ledger, not used for training control.
        p,n=self._bank_score(bank,x,y,a,count_work=count_work)
        confidence=min(1.,n/3.)
        features=(1.,confidence,abs(p-fast),fast,p)
        gate=sigmoid(math.fsum(
            w*v for w,v in zip(self.director,features)))
        if self.arm=="fast_only":gate=0.
        elif self.arm=="latent_memory_only":gate=1.
        elif self.arm=="latent_fixed_fusion":gate=.5
        result=(1.-gate)*fast+gate*p
        return result,fast,p,gate,z,h,features

    def _bank_from_history(self,history,*,novelty_allowed:bool):
        """Score latent bank identity using only lived outcomes.

        This is an engineered evidence selector (NOT neural RL). With no
        usable evidence, the agent retains its previous active estimate.
        """
        if not history:return self.active
        records=tuple(history)
        best=None
        for i,bank in enumerate(self.banks):
            matches,conflicts,support=bank.evidence(records)
            if support==0:continue
            candidate=(matches-2*conflicts,matches,-conflicts,-i)
            if best is None or candidate>best[0]:
                best=(candidate,i,matches,conflicts,support)
        if best is None:return self.active
        _,index,matches,conflicts,support=best
        if (support>=MIN_EVIDENCE and
                matches>=SWITCH_MATCHES and conflicts<=1):
            return index
        if novelty_allowed and len(self.banks)<MAX_BANKS:
            current=self.banks[self.active]
            current_matches,current_conflicts,current_support=(
                current.evidence(records))
            if (len(records)>=MIN_EVIDENCE and
                    current_conflicts>=MIN_CONFLICTS and
                    (best[3]>=MIN_CONFLICTS or best[2]<SWITCH_MATCHES)):
                return len(self.banks)
        return self.active

    def current_bank(self,ctx_for_oracle=None):
        if self.arm=="oracle_context":
            if ctx_for_oracle not in (0,1,2):
                raise ValueError("Oracle requires privileged world index")
            return ctx_for_oracle
        if ctx_for_oracle is not None:
            raise AssertionError("Hidden-world learner cannot receive label")
        if self.arm=="no_regime_detection":return 0
        return self.active

    def choose(self,x,y,tick:int,*,training=True,
               context_for_oracle=None,forced_bank=None):
        selected=(forced_bank if forced_bank is not None else
                  self.current_bank(ctx_for_oracle=context_for_oracle))
        p0=self._components(x,y,0,selected,count_work=False)[0]
        p1=self._components(x,y,1,selected,count_work=False)[0]
        a=int(p1>p0)
        if training:
            frac=min(1.,tick/1520.)
            eps=.25*(1.-frac)+.07*frac
            if self.rng.random()<eps:a=self.rng.randrange(2)
        return a

    def _update_fast_and_gate(self,x,y,a,outcome,bank):
        p,pfast,pmem,gate,z,h,features=(
            self._components(x,y,a,bank,count_work=True))
        p=max(1e-7,min(1.-1e-7,p))
        common=(outcome-p)/(p*(1.-p))
        old=self.fast[:]
        scale=common*(1.-gate)*pfast*(1.-pfast)
        # Fast learner always receives one real gradient per observed
        # action across all arms, including memory-only (ablation for use).
        for j,hj in enumerate(h):
            idx=FAST_HIDDEN*DIM+j
            self.fast[idx]=max(-5.,min(5.,old[idx]+LR_FAST*scale*hj))
            coefficient=scale*old[idx]*(1.-hj*hj)
            for k,feat in enumerate(z):
                pos=j*DIM+k
                self.fast[pos]=max(-5.,min(5.,
                    old[pos]+LR_FAST*coefficient*feat))
        self.fast[-1]=max(-5.,min(5.,old[-1]+LR_FAST*scale))
        self.fast_updates+=1
        if self.arm in (
            "latent_director","latent_short_history",
            "no_regime_detection","scrambled_detection",
            "oracle_context"):
            factor=common*(pmem-pfast)*gate*(1.-gate)
            for k,feat in enumerate(features):
                self.director[k]=max(-6.,min(6.,
                    self.director[k]+LR_GATE*factor*feat))
            self.gate_updates+=1
        self.gate_sum+=gate
        self.update_calls+=1

    def experience(self,x:int,y:int,a:int,real_reward:int,
                   *,context_for_oracle=None):
        if real_reward not in (0,1):
            raise ValueError("Only actual chosen-action binary reward")
        bank_index=self.current_bank(ctx_for_oracle=context_for_oracle)
        if self.arm=="oracle_context":
            while len(self.banks)<=bank_index:
                self.banks.append(LatentBank(
                    MEMORY_SLOTS[len(self.banks)]))
        # Learn the actual outcome BEFORE writing it to personal history.
        self._update_fast_and_gate(x,y,a,real_reward,bank_index)
        self.lived+=1
        if self.arm=="oracle_context":
            self.banks[bank_index].append((x,y,a,real_reward))
            self.active=bank_index
            return
        if self.arm=="no_regime_detection":
            self.banks[0].append((x,y,a,real_reward))
            return
        # The adverse control disrupts only the director's evidence about
        # when worlds changed. The archived actual experience stays honest.
        fed=(self.noise_rng.randrange(2)
             if self.arm=="scrambled_detection" else real_reward)
        episode=(x,y,a,fed)
        current=self.banks[self.active]
        expected,n=current.response(x,y,a)
        conflicting=(n>=2 and (
            (fed==1 and expected<1.-CHANGE_CONFIDENCE) or
            (fed==0 and expected>CHANGE_CONFIDENCE)))
        self.history.append(episode)
        if not conflicting:
            current.append((x,y,a,real_reward))
        else:
            self.unassigned_surprises+=1
        if len(self.history)<MIN_EVIDENCE:return
        candidate=self._bank_from_history(
            self.history,novelty_allowed=True)
        if candidate!=self.active:
            old=self.active
            if candidate==len(self.banks):
                self.banks.append(LatentBank(MEMORY_SLOTS[candidate]))
            target=self.banks[candidate]
            # Admit only recent experience that is consistent with
            # target evidence; prevent updating the old archive with
            # contradictory post-change observations.
            for oldx,oldy,olda,oldr in self.history:
                estimated,confidence_n=target.response(
                    oldx,oldy,olda)
                if (confidence_n==0 or
                    (oldr and estimated>=.5) or
                    (not oldr and estimated<=.5)):
                    target.append((oldx,oldy,olda,oldr))
            self.active=candidate
            self.inferred_switches.append({
                "at":self.lived,"from":old,"to":candidate})
            self.history.clear()
        if sum(len(bank.records) for bank in self.banks)>MEMORY_LIMIT:
            raise AssertionError("Overflowed bounded autobiographical memory")

    def select_from_probe(self,probe,*,oracle_index=None):
        if self.arm=="oracle_context":
            if oracle_index not in range(len(self.banks)):
                return 0
            return oracle_index
        if self.arm=="no_regime_detection":return 0
        if self.arm=="fast_only":
            return 0
        if self.arm=="scrambled_detection":
            # Scrambled evidence affected what banks were created
            # during training; exam evidence remains genuine, for
            # a conservative negative control.
            pass
        if not probe:return 0   # no feedback gives no context information
        best=None
        for i,bank in enumerate(self.banks):
            matches,conflicts,supported=bank.evidence(probe)
            # log-support rationale: conflicts matter more than unknowns;
            # bank selection is based exclusively on probe feedback.
            score=(matches-2*conflicts,matches,-conflicts,-i)
            if best is None or score>best[0]:
                best=(score,i)
        return best[1] if best is not None else 0


def probe_cues(seed,exam_index):
    # Exactly four occurrences of each x/y cue, independently shuffled.
    check_seed(seed)
    rng=random.Random(seed*999_983+exam_index*73+10_001)
    data=[(x,y) for x in (0,1) for y in (0,1) for _ in range(4)]
    rng.shuffle(data)
    return tuple(data)


def frozen_exam(brain:Organism,seed:int,world:str,exam_index:int,
                *,priming:int):
    """Read-only weights/archive; 0 or 16 evaluation outcomes are given.

    Probe actions are chosen by a fixed, stimulus-independent exploration
    coin, not the privileged hidden answer. They are genuine action
    outcomes but update ONLY ephemeral inference, never parameters,
    archive or the original organism's session state.
    """
    if priming not in (0,PROBE):raise ValueError("Only 0 or 16 probe exams")
    original=brain.snapshot()
    rng=random.Random(seed*481_501+exam_index*971+29)
    scratch=[]
    for x,y in probe_cues(seed,exam_index)[:priming]:
        action=rng.randrange(2)
        feedback=environmental_success(world,x,y,action)
        scratch.append((x,y,action,feedback))
    oracle=(CONTEXT[world] if brain.arm=="oracle_context" else None)
    selected=brain.select_from_probe(scratch,oracle_index=oracle)
    passed=0
    for x,y in observations(seed,exam_index,EXAM):
        action=brain.choose(
            x,y,0,training=False,
            context_for_oracle=oracle if brain.arm=="oracle_context" else None,
            forced_bank=selected)
        passed+=environmental_success(world,x,y,action)
    if brain.snapshot()!=original:
        raise AssertionError("Exam mutated any learned state")
    return passed/EXAM,selected


def run(seed:int,arm:str):
    check_seed(seed)
    brain=Organism(seed,arm)
    phases=schedule(seed)
    scores={}
    cold={}
    exam_selected_banks={}
    training={}
    occupancy=[]
    phase_transitions=[]
    tick=0
    for phase,(world,n) in enumerate(phases):
        stream=observations(seed,phase,n)
        rewards=[]
        for x,y in stream:
            chosen=brain.choose(
                x,y,tick,training=True,
                context_for_oracle=CONTEXT[world]
                if arm=="oracle_context" else None)
            reward=environmental_success(world,x,y,chosen)
            brain.experience(
                x,y,chosen,reward,
                context_for_oracle=CONTEXT[world]
                if arm=="oracle_context" else None)
            rewards.append(reward)
            tick+=1
        training[f"phase_{phase}_{world}_first16"]=(
            sum(rewards[:16])/16)
        training[f"phase_{phase}_{world}_last80"]=(
            sum(rewards[-80:])/80)
        occupancy.append([len(b.records) for b in brain.banks])
        phase_transitions.append({
            "phase_index":phase,"actual_world_for_reporting":world,
            "end_timestep":tick,"inferred_active_bank":brain.active,
            "bank_count":len(brain.banks)})
        for target_index in range(max(CONTEXT[w] for w,_ in phases[:phase+1])+1):
            test_world="ABC"[target_index]
            exam_index=20+phase*3+target_index
            result,selected=frozen_exam(
                brain,seed,test_world,exam_index,priming=PROBE)
            exam_selected_banks[f"phase_{phase}_{test_world}"]=selected
            no_probe,_=frozen_exam(
                brain,seed,test_world,exam_index,priming=0)
            scores[f"phase_{phase}_{test_world}"]=result
            cold[f"phase_{phase}_{test_world}"]=no_probe
    if brain.lived!=sum(n for _,n in phases):
        raise AssertionError("Wrong lived experience budget")
    if brain.update_calls!=brain.lived or brain.fast_updates!=brain.lived:
        raise AssertionError("Unmatched fast update calls")
    if sum(len(b.records) for b in brain.banks)>MEMORY_LIMIT:
        raise AssertionError("Personal memory grew without bounds")
    # Purely post-hoc observer diagnosis. True world labels are used
    # ONLY outside the learning/exam pathways, to quantify whether each
    # internally inferred archive bank represents A, B or C.
    bank_content_accuracy={}
    for bank_id,archive in enumerate(brain.banks):
        bank_content_accuracy[str(bank_id)]={
            world:sum(
                environmental_success(
                    world,x,y,
                    int(archive.response(x,y,1)[0]>
                        archive.response(x,y,0)[0]))
                for x in (0,1) for y in (0,1))/4.
            for world in ("A","B","C")}
    return {
        "seed":seed,"arm":arm,
        "posthoc_bank_content_accuracy":bank_content_accuracy,
        "exam_selected_banks":exam_selected_banks,
        "actual_phase_lengths":[n for _,n in phases],
        "A_initial":scores["phase_0_A"],
        "A_after_B":scores["phase_1_A"],
        "B_first_learned":scores["phase_1_B"],
        "A_after_C":scores["phase_2_A"],
        "B_after_C":scores["phase_2_B"],
        "C_first_learned":scores["phase_2_C"],
        "A_final":scores["phase_4_A"],
        "B_final":scores["phase_4_B"],
        "C_final":scores["phase_4_C"],
        "A_after_C_zero_probe":cold["phase_2_A"],
        "B_after_C_zero_probe":cold["phase_2_B"],
        "C_after_C_zero_probe":cold["phase_2_C"],
        "joint_pre_refresher":statistics.mean((
            scores["phase_2_A"],scores["phase_1_B"],
            scores["phase_2_B"],scores["phase_2_C"])),
        "mean_post_probe_after_C":statistics.mean((
            scores["phase_2_A"],scores["phase_2_B"],
            scores["phase_2_C"])),
        "minimum_after_C":min(scores["phase_2_A"],
                              scores["phase_2_B"],
                              scores["phase_2_C"]),
        "mean_final":statistics.mean((
            scores["phase_4_A"],scores["phase_4_B"],
            scores["phase_4_C"])),
        "training":training,
        "lived_experiences":brain.lived,
        "gradient_update_calls":brain.update_calls,
        "fast_gradient_updates":brain.fast_updates,
        "director_gradient_updates":brain.gate_updates,
        "memory_capacity":MEMORY_LIMIT,
        "final_memory_size":sum(len(b.records) for b in brain.banks),
        "final_bank_count":len(brain.banks),
        "memory_occupancy_per_phase":occupancy,
        "inferred_switch_events":brain.inferred_switches,
        "inferred_switch_count":len(brain.inferred_switches),
        "timeline_diagnostics":phase_transitions,
        "training_memory_read_evaluations":brain.read_record_evaluations,
        "average_director_memory_gate":brain.gate_sum/brain.update_calls,
        "no_world_id_passed_to_hidden_arm":arm!="oracle_context",
        "exam_priming_real_feedback_count":PROBE,
        "exam_only_ephemeral_working_state":True,
        "no_teacher_oracle_action_labels":True,
    }


def paired_bootstrap(rows,treat,control,metric="joint_pre_refresher",
                     replicates=1400):
    lookup={(r["seed"],r["arm"]):r for r in rows}
    deltas=[lookup[(seed,treat)][metric]-
            lookup[(seed,control)][metric] for seed in SEEDS]
    rand=random.Random(310483)
    dist=sorted(statistics.mean(rand.choice(deltas) for _ in deltas)
                for _ in range(replicates))
    return {"delta":statistics.mean(deltas),
            "boot95":[dist[int(.025*replicates)],
                      dist[int(.975*replicates)-1]],
            "paired_seed_count":len(SEEDS)}


def experiment(seeds=SEEDS):
    if tuple(seeds)!=SEEDS:
        raise ValueError("Full preregistered development cohort required")
    rows=[run(seed,arm) for seed in seeds for arm in ARMS]
    ix={(r["seed"],r["arm"]):r for r in rows}
    keys=(
        "A_initial","A_after_B","B_first_learned","A_after_C",
        "B_after_C","C_first_learned","A_final","B_final","C_final",
        "A_after_C_zero_probe","B_after_C_zero_probe",
        "C_after_C_zero_probe","joint_pre_refresher",
        "mean_post_probe_after_C","minimum_after_C","mean_final",
        "final_bank_count","inferred_switch_count",
        "gradient_update_calls","training_memory_read_evaluations",
        "average_director_memory_gate")
    summary={
        arm:{key:statistics.mean(ix[(seed,arm)][key]
                                 for seed in seeds) for key in keys}
        for arm in ARMS}
    l=summary["latent_director"]
    oracle=summary["oracle_context"]
    checks={
        "learned_hidden_A_after_C_90":
            l["A_after_C"]>=GATES["A_after_C"],
        "learned_hidden_B_after_C_90":
            l["B_after_C"]>=GATES["B_after_C"],
        "learned_hidden_B_first_85":
            l["B_first_learned"]>=GATES["B_new_learning"],
        "learned_hidden_C_first_85":
            l["C_first_learned"]>=GATES["C_new_learning"],
        "learned_hidden_post_probe_mean_90":
            l["mean_post_probe_after_C"]>=GATES["all_after_probe"],
        "hidden_vs_oracle_within7pp":
            l["joint_pre_refresher"]>=
            oracle["joint_pre_refresher"]-
            GATES["same_context_as_oracle_within"],
        "latent_detection_vs_single_bank_plus12pp":
            l["joint_pre_refresher"]-
            summary["no_regime_detection"]["joint_pre_refresher"]>=
            GATES["latent_vs_mixed_delta"],
        "latent_detection_vs_scrambled_evidence_plus12pp":
            l["joint_pre_refresher"]-
            summary["scrambled_detection"]["joint_pre_refresher"]>=
            GATES["latent_vs_scrambled_delta"],
        "learned_director_vs_memory_only_plus5pp":
            l["joint_pre_refresher"]-
            summary["latent_memory_only"]["joint_pre_refresher"]>=
            GATES["learned_vs_memory_only_delta"],
        "learned_director_vs_fixed_fusion_plus5pp":
            l["joint_pre_refresher"]-
            summary["latent_fixed_fusion"]["joint_pre_refresher"]>=
            GATES["learned_vs_fixed_fusion_delta"],
    }
    # No non-oracle branch ever receives the true task identity.
    for seed in seeds:
        if any(ix[(seed,arm)]["no_world_id_passed_to_hidden_arm"] is not True
               for arm in ARMS if arm!="oracle_context"):
            raise AssertionError("Leaked task identity into learner")
        if len({ix[(seed,arm)]["lived_experiences"]
                for arm in ARMS})!=1:
            raise AssertionError("Unequal number of true action outcomes")
        if any(ix[(seed,arm)]["gradient_update_calls"]!=
               ix[(seed,arm)]["lived_experiences"] for arm in ARMS):
            raise AssertionError("Unmatched fast weight update calls")
        if any(ix[(seed,arm)]["memory_capacity"]!=MEMORY_LIMIT
               for arm in ARMS):
            raise AssertionError("Unequal memory budget")
    return {
        "status":"M5.24_DEV_hidden_context_memory_executive",
        "seeds":list(seeds),"arms":list(ARMS),
        "task_names_not_available_to_nonoracle_agent":True,
        "phase_durations_hidden_and_varied_by_seed":True,
        "binary_action_complementarity_is_assumed":True,
        "new_bank_allocated_by_handcoded_reward_conflict":True,
        "context_selector_is_handcoded_not_learned_RL":True,
        "learned_director_controls_trust_in_memory":True,
        "archive_capped_at_128_records":True,
        "exam_16_feedback_probes_no_memory_or_weight_updates":True,
        "zero_probe_controls_included":True,
        "same_four_cue_patterns_without_out_of_distribution_test":True,
        "control_with_oracle_context_label_is_privileged":True,
        "no_evolution_or_synapse_birth":True,
        "no_original_sealed_M5_6_cohorts":True,
        "summary":summary,"screening_gate":checks,
        "hidden_context_behavior_supported":all(
            checks[k] for k in checks if k.startswith("learned_hidden_")),
        "learned_executive_added_value_supported":all(
            checks[k] for k in checks if k.startswith("learned_director_")),
        "development_screen_pass":all(checks.values()),
        "independent_task_replication_complete":False,
        "evolution_authorized":False,
        "advance_to_evolution":False,
        "paired_bootstrap":{
            label:paired_bootstrap(rows,"latent_director",arm)
            for label,arm in (
                ("against_oracle","oracle_context"),
                ("against_memory_only","latent_memory_only"),
                ("against_fixed_fusion","latent_fixed_fusion"),
                ("against_no_detection","no_regime_detection"),
                ("against_scrambled","scrambled_detection"),
                ("against_short_history","latent_short_history"))
        },
        "raw_rows":rows,
        "rows_sha256":hashlib.sha256(json.dumps(
            rows,sort_keys=True,separators=(",",":")).encode()).hexdigest(),
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,default=Path(
        "results/m5-24/hidden-context-director-development.json"))
    args=p.parse_args()
    report=experiment()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",
                        encoding="utf-8")
    print("M5_24_HIDDEN_CONTEXT "+json.dumps({
        k:report[k] for k in (
            "status","summary","screening_gate",
            "hidden_context_behavior_supported",
            "learned_executive_added_value_supported",
            "development_screen_pass",
            "paired_bootstrap","advance_to_evolution",
            "rows_sha256")},sort_keys=True),flush=True)


if __name__=="__main__":
    main()
