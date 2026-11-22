"""M5.26 — Resource-limited, novelty-coverage autobiographical memory.

The same hidden A→B→C→A→B tiny deterministic world as M5.25, with
new development seeds 5000..5023 and 16 actual diagnostic outcomes
per frozen exam. The LONG-TERM autobiographical archive cap is now only 12
(4+4+4 independent experiences) or 8 (3+3+2). The primary candidate
stores one experienced action/outcome per DISTINCT SENSORY CUE in each
evidence-inferred latent bank, rejecting redundant episodes. Matched
control arms use FIFO and reservoir eviction under the same caps.

The novelty rule is ENGINEERED / handcoded. Bank identity and bank
capacity remain engineered; no RL policy for eviction has been shown.
The memory reader assumes deterministic complementary binary actions,
which permits inference about the unchosen action from a real outcome.
The fast neural predictor learns once per lived experience, same founder
parameters for every arm. No A/B/C labels are passed to any agent.

Reference 128-record memory, readout ablations and the 8-slot stress
case are reported. The 8-slot direct key-value archive cannot hold
all 12 distinct (latent context, x, y) pairs simultaneously without
compression/generalization; it is a diagnostic, not an impossible
promise of perfect recall.
No ecological evolution, main merge or original M5.6 sealed worlds.
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

SEEDS=tuple(range(5000,5024))
ARMS=(
    "novelty12",
    "fifo12",
    "reservoir12",
    "novelty_memory_only12",
    "novelty_fixed_fusion12",
    "novelty_fast_only12",
    "novelty8",
    "fifo8",
    "reservoir8",
    "reference128",
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
MEMORY_LIMIT=128
SLOTS12=(4,4,4)
SLOTS8=(3,3,2)
CAPACITY_BY_ARM={a:(SLOTS8 if a.endswith("8")
                   else MEMORY_SLOTS if a=="reference128"
                   else SLOTS12) for a in ARMS}
MAX_BANKS=3
HISTORY=14
MIN_EVIDENCE=9
MIN_CONFLICTS=3
SWITCH_MATCHES=6
READ_PRIOR=.12
CHANGE_CONFIDENCE=.78
LOCK_AFTER=40
PENDING_EVIDENCE=12
MAX_PENDING=12
MIN_ROUTE_MATCH=.78
MAX_NOVELTY_MATCH=.70
GATES={
    "retain_A":.90,"retain_B":.90,
    "acquire_B":.90,"acquire_C":.90,
    "mean_after_C":.90,
    "gain_over_fifo":.10,
    "gain_over_reservoir":.10,
    "gain_over_fast":.20,
    "learned_director_gain":.05,
    "8_slot_loss":.04,
}


def check_seed(seed:int):
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError("Only new DEV seeds 5000..5023")


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
            if n<1:continue
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
        confidence=min(1.,n)
        features=(1.,confidence,abs(p-fast),fast,p)
        gate=sigmoid(math.fsum(
            w*v for w,v in zip(self.director,features)))
        if self.arm in ("fast_only","guarded_fast_only","novelty_fast_only12"):gate=0.
        elif self.arm in ("latent_memory_only","guarded_memory_only",
                         "novelty_memory_only12"):gate=1.
        elif self.arm in ("latent_fixed_fusion","guarded_fixed_fusion",
                         "novelty_fixed_fusion12"):gate=.5
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
            "latent_director","guarded_director","guarded_mutable",
            "novelty12","fifo12","reservoir12","novelty8",
            "fifo8","reservoir8","reference128",
            "guarded_corrupt_probe",
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


class BudgetOrganism(Organism):
    """Independent resource-bounded memory selector, no true world IDs.

    The bank ID itself comes from 12 lived outcomes and a handcoded
    compatibility threshold, not from a teacher. Only this retention
    policy changes across the matched treatments.
    """

    def __init__(self,seed:int,arm:str):
        super().__init__(seed,arm)
        self.slots=CAPACITY_BY_ARM[arm]
        self.banks=[LatentBank(self.slots[0])]
        self.pending=[]
        self.guard_switches=[]
        self.denied_stable_writes=0
        self.bank_write_counts=[0]
        self.per_bank_seen=[0]
        self.new_cue_admissions=0
        self.duplicate_refusals=0
        self.evictions=0
        self.reservoir_rng=random.Random(seed*987_931+129)
        self.probe_noise=random.Random(seed*199_103+23)

    def snapshot(self):
        return (super().snapshot(),tuple(self.pending),
                tuple(self.guard_switches),
                self.denied_stable_writes,tuple(self.bank_write_counts),
                tuple(self.per_bank_seen),self.new_cue_admissions,
                self.duplicate_refusals,self.evictions,
                self.reservoir_rng.getstate(),self.probe_noise.getstate())

    def _write_to_bank(self,bank_id,episode):
        bank=self.banks[bank_id]
        self.per_bank_seen[bank_id]+=1
        cap=self.slots[bank_id]
        if self.arm in ("novelty12","novelty8",
                        "novelty_memory_only12","novelty_fixed_fusion12",
                        "novelty_fast_only12"):
            if any((oldx,oldy)==episode[:2]
                   for oldx,oldy,_,_ in bank.records):
                self.duplicate_refusals+=1
                return
            if len(bank.records)>=cap:
                self.denied_stable_writes+=1
                return
            bank.append(episode)
            self.bank_write_counts[bank_id]+=1
            self.new_cue_admissions+=1
            return
        if self.arm=="reference128":
            if len(bank.records)>=40:
                self.denied_stable_writes+=1
                return
            bank.append(episode)
            self.bank_write_counts[bank_id]+=1
            return
        if self.arm in ("fifo12","fifo8"):
            self.evictions+=int(len(bank.records)==cap)
            bank.append(episode)
            self.bank_write_counts[bank_id]+=1
            return
        if self.arm in ("reservoir12","reservoir8"):
            seen=self.per_bank_seen[bank_id]
            if len(bank.records)<cap:
                bank.append(episode)
                self.bank_write_counts[bank_id]+=1
            else:
                index=self.reservoir_rng.randrange(seen)
                if index<cap:
                    items=list(bank.records)
                    items[index]=episode
                    bank.records.clear()
                    bank.records.extend(items)
                    self.evictions+=1
                    self.bank_write_counts[bank_id]+=1
            return
        raise AssertionError("Unregistered memory selection policy")

    def _mature(self,bank_id):
        if self.arm=="reference128":
            return len(self.banks[bank_id].records)>=40
        return len(self.banks[bank_id].records)>=self.slots[bank_id]

    def _pick_evidence_target(self,records):
        candidates=[]
        for index,bank in enumerate(self.banks):
            matches,conflicts,supported=bank.evidence(records)
            if supported>=5:
                candidates.append((matches/supported,index))
        if not candidates:return None
        candidates.sort(key=lambda x:(x[0],-x[1]),reverse=True)
        fit,index=candidates[0]
        if fit>=MIN_ROUTE_MATCH:return index
        if fit<=MAX_NOVELTY_MATCH and len(self.banks)<MAX_BANKS:
            return len(self.banks)
        return None

    def experience(self,x:int,y:int,a:int,real_reward:int,
                   *,context_for_oracle=None):
        if context_for_oracle is not None:
            raise AssertionError("Hidden learner cannot receive world identity")
        if real_reward not in (0,1):
            raise ValueError("Only actual action binary success/failure")
        record=(x,y,a,real_reward)
        # All organisms have same fast learner and exactly one update
        # on their own lived reward per environmental interaction.
        self._update_fast_and_gate(x,y,a,real_reward,self.active)
        self.lived+=1
        bank=self.banks[self.active]
        if not self._mature(self.active) and not self.pending:
            self._write_to_bank(self.active,record)
            return
        p,n=bank.response(x,y,a)
        contradict=(n>=1 and (
            (real_reward==1 and p<1.-CHANGE_CONFIDENCE)
            or (real_reward==0 and p>CHANGE_CONFIDENCE)))
        if not self.pending and not contradict:
            self._write_to_bank(self.active,record)
            return
        # Quarantine contradiction: NO policy may write it into an
        # already-addressed bank until actual supporting evidence arrives.
        self.pending.append(record)
        if len(self.pending)<PENDING_EVIDENCE:return
        target=self._pick_evidence_target(self.pending)
        if target is None:
            if len(self.pending)>MAX_PENDING:
                self.pending=self.pending[-PENDING_EVIDENCE:]
            return
        previous=self.active
        if target==len(self.banks):
            self.banks.append(LatentBank(self.slots[target]))
            self.bank_write_counts.append(0)
            self.per_bank_seen.append(0)
        if previous!=target:
            self.guard_switches.append((self.lived,previous,target))
        self.active=target
        for buffered in self.pending:
            self._write_to_bank(target,buffered)
        self.pending.clear()
        if sum(len(b.records) for b in self.banks)>sum(self.slots):
            raise AssertionError("Exceeded strict total memory budget")


def make_organism(seed,arm):
    return BudgetOrganism(seed,arm)

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
    brain=make_organism(seed,arm)
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
    if sum(len(b.records) for b in brain.banks)>sum(brain.slots):
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
        "memory_capacity":sum(brain.slots),
        "quarantine_working_memory_capacity":MAX_PENDING,
        "unused_agent_history_buffer":len(brain.history),
        "memory_slots_per_latent_bank":list(brain.slots),
        "record_admissions":sum(brain.bank_write_counts),
        "unique_cue_admissions":brain.new_cue_admissions,
        "duplicate_refusals":brain.duplicate_refusals,
        "record_evictions":brain.evictions,
        "final_memory_size":sum(len(b.records) for b in brain.banks),
        "final_bank_count":len(brain.banks),
        "memory_occupancy_per_phase":occupancy,
        "final_distinct_cues_per_bank":[len({
            (x,y) for x,y,_,_ in bank.records})
            for bank in brain.banks],
        "inferred_switch_events":brain.guard_switches,
        "guarded_pending_size":len(brain.pending),
        "denied_stable_writes":brain.denied_stable_writes,
        "verified_archive_freezes":None,
        "bank_write_counts":brain.bank_write_counts,
        "inferred_switch_count":len(brain.guard_switches),
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
        raise ValueError("Run entire preregistered DEV cohort 5000..5023")
    rows=[run(seed,arm) for seed in seeds for arm in ARMS]
    ix={(r["seed"],r["arm"]):r for r in rows}
    keys=("A_initial","A_after_B","B_first_learned","A_after_C",
          "B_after_C","C_first_learned","A_final","B_final","C_final",
          "joint_pre_refresher","mean_post_probe_after_C",
          "minimum_after_C","final_bank_count","record_evictions",
          "final_memory_size","memory_capacity","record_admissions",
          "unique_cue_admissions","duplicate_refusals")
    summary={arm:{k:statistics.mean(ix[(seed,arm)][k] for seed in seeds)
                   for k in keys} for arm in ARMS}
    candidate=summary["novelty12"]
    delta=lambda arm:(candidate["joint_pre_refresher"]-
                      summary[arm]["joint_pre_refresher"])
    checks={
        "novelty12_A_remembered_after_C_90":
            candidate["A_after_C"]>=GATES["retain_A"],
        "novelty12_B_remembered_after_C_90":
            candidate["B_after_C"]>=GATES["retain_B"],
        "novelty12_B_first_acquired_90":
            candidate["B_first_learned"]>=GATES["acquire_B"],
        "novelty12_C_first_acquired_90":
            candidate["C_first_learned"]>=GATES["acquire_C"],
        "novelty12_mean_A_B_C_after_C_90":
            candidate["mean_post_probe_after_C"]>=GATES["mean_after_C"],
        "novelty12_vs_FIFO_plus10pp":
            delta("fifo12")>=GATES["gain_over_fifo"],
        "novelty12_vs_reservoir_plus10pp":
            delta("reservoir12")>=GATES["gain_over_reservoir"],
        "novelty12_vs_fast_only_plus20pp":
            delta("novelty_fast_only12")>=GATES["gain_over_fast"],
        "novelty12_director_beats_memory_only_plus5pp":
            delta("novelty_memory_only12")>=
            GATES["learned_director_gain"],
        "novelty12_director_beats_fixed_fusion_plus5pp":
            delta("novelty_fixed_fusion12")>=
            GATES["learned_director_gain"],
        "8_slots_show_real_cost_of_missing_cue_coverage":
            delta("novelty8")>=GATES["8_slot_loss"],
        "novelty12_all_three_memory_banks_cover_four_cues":
            all(ix[(seed,"novelty12")]["final_distinct_cues_per_bank"]
                ==[4,4,4] for seed in seeds),
        "memory_caps_never_exceeded":
            all(r["final_memory_size"]<=r["memory_capacity"] for r in rows),
    }
    for seed in seeds:
        if len({ix[(seed,a)]["lived_experiences"] for a in ARMS})!=1:
            raise AssertionError("Different number of experienced actions")
        for a in ARMS:
            r=ix[(seed,a)]
            if r["gradient_update_calls"]!=r["lived_experiences"]:
                raise AssertionError("Unequal weight update calls")
            if r["memory_capacity"]!=sum(CAPACITY_BY_ARM[a]):
                raise AssertionError("Incorrect per-arm capacity")
            if not r["no_world_id_passed_to_hidden_arm"]:
                raise AssertionError("True task identity leaked")
    return {
        "status":"M5.26_DEV_novelty_based_memory_selection",
        "seeds":list(seeds),"arms":list(ARMS),
        "schedules_hidden_and_same_within_seed":True,
        "only_own_action_binary_reward_observed":True,
        "novelty_selection_handcoded_not_learned_meta_policy":True,
        "bank_sizing_and_context_detection_hand_engineered":True,
        "no_oracle_context_provided":True,
        "two_action_deterministic_complementarity_assumed":True,
        "unique_cue_key_value_capacity_target":12,
        "below_target_stress_budget":8,
        "12_slot_bank_allocation":[4,4,4],
        "long_term_memory_capacity_not_total_working_memory":True,
        "temporary_quarantine_max_records":MAX_PENDING,
        "frozen_exam_ephemeral_probe_records":PROBE,
        "8_slot_bank_allocation":[3,3,2],
        "reference_memory_size":128,
        "same_lived_action_count_and_weight_update_calls":True,
        "archive_write_and_compute_costs_vary":True,
        "frozen_exams_give_16_diagnostic_feedback_outcomes":True,
        "familiar_cues_only_not_novel_world_generalization":True,
        "main_branch_unmodified":True,
        "original_M5_6_sealed_cohort_untouched":True,
        "no_evolution_or_genetic_change":True,
        "summary":summary,"screening_gate":checks,
        "storage_selection_screen_pass":all(v for k,v in checks.items()
            if not k.startswith("novelty12_director_")),
        "learned_executive_added_value_screen":all(v for k,v in checks.items()
            if k.startswith("novelty12_director_")),
        "development_screen_pass":all(checks.values()),
        "independent_task_replication_complete":False,
        "evolution_authorized":False,
        "advance_to_evolution":False,
        "paired_bootstrap":{
            a:paired_bootstrap(rows,"novelty12",a)
            for a in ARMS if a!="novelty12"},
        "raw_rows":rows,
        "rows_sha256":hashlib.sha256(json.dumps(
            rows,sort_keys=True,separators=(",",":")).encode()).hexdigest(),
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,default=Path(
        "results/m5-26/compact-memory-development.json"))
    args=p.parse_args()
    report=experiment()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",
                        encoding="utf-8")
    print("M5_26_MEMORY_SELECTION "+json.dumps({
        k:report[k] for k in (
            "status","summary","screening_gate",
            "storage_selection_screen_pass",
            "learned_executive_added_value_screen",
            "development_screen_pass",
            "paired_bootstrap","rows_sha256")
    },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
