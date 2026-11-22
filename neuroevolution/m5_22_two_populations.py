"""M5.22: memory + director versus alternating neural populations.

STOP/GO development laboratory. Seven arms use the SAME 2x8-hidden-unit
two-expert reward predictor with a trainable 8-parameter soft gate.
Each new organism has 154 parameters (not biologically realistic neurons).
Both populations and the gate see two binary cues and an *explicitly given*
environment context signal. The gate is LEARNED from the consequences of
chosen actions: no correct-action label is provided to the agent.

Arms:
- joint: both populations update, gate learns.
- memory_director: one population self-freezes when two consecutive
  recent 40-trial blocks reach >=90% success, after >=200 actual decisions.
  Second population keeps learning; trainable gate chooses their influence.
- memory_phase_bound: experimenter freezes group 0 precisely after first
  400 A trials: an explicit/oracle-schedule upper-bound *control*.
- swapped_phase_bound: same 400-trial schedule but freeze group 1.
- alternating: after initial 400 experiences, alternate which population
  updates every 40 action steps; gate always updates.
- alternating_yoked_random: same number/timing of single-population steps
  as alternating, but population choice randomized within each known phase.
- replay_reference: both update with four additional actual lived-memory
  samples per step; DIFFERENT update budget, positive reference ONLY.

All except replay make EXACTLY 1520 brain update calls per organism.
Some arms freeze half the parameters per call. We explicitly track the
number of group updates; NOT equivalent gradient-parameter exposure.
Replay uses 7600 calls. Every learner has exactly 128 autobiographical
memory slots; only replay_reference directly weight-trains from past
episodes. Examinations are frozen, before refresher teaching, and
reshuffle the same four binary sensory pairs, NOT out-of-distribution
generalization. Original M5.6 sealed cohorts and evolution untouched.
New development seeds 4600..4623, preregistered screening thresholds.
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

SEEDS=tuple(range(4600,4624))
ARMS=("joint","memory_director","memory_phase_bound",
      "swapped_phase_bound","alternating","alternating_yoked_random",
      "replay_reference")
PHASES=(("A",400),("B",400),("C",400),("A",160),("B",160))
CTX={"A":0,"B":1,"C":2}
BLOCK=40
EXAM=80
TRAIN_STEPS=sum(n for _,n in PHASES)
DIM=8
HIDDEN_PER_GROUP=8
GROUPS=2
PREDICTOR_INPUT_PARAMS=GROUPS*HIDDEN_PER_GROUP*DIM # 128
PREDICTOR_OUTPUT_OFFSET=PREDICTOR_INPUT_PARAMS # 128
EXPERT_BIAS_OFFSET=PREDICTOR_OUTPUT_OFFSET+GROUPS*HIDDEN_PER_GROUP # 144
GATE_OFFSET=EXPERT_BIAS_OFFSET+GROUPS #146
PARAMS=GATE_OFFSET+DIM #154
LR=.07
GATE_LR_MULT=1.5
MEMORY_SLOTS=(43,43,42)
MEMORY_CAP=sum(MEMORY_SLOTS)
AUTOFREEZE_MIN_EXPOSURE=200
AUTOFREEZE_THRESHOLD=.90
SCREEN_THRESHOLD=.90
SCREEN_ADVANTAGE=.05


def check_seed(seed:int):
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError("M5.22 development only: seeds 4600..4623")


def cue_sequence(seed:int,phase:int,n:int):
    check_seed(seed)
    if n%BLOCK:raise ValueError("Balanced blocks only")
    rng=random.Random(seed*211_001+phase*271+71)
    all_pairs=[]
    for _ in range(n//BLOCK):
        block=[(x,y) for x in (0,1) for y in (0,1)
               for _ in range(BLOCK//4)]
        rng.shuffle(block)
        all_pairs.extend(block)
    return tuple(all_pairs)


def sensory(x:int,y:int,c:int,a:int):
    if x not in (0,1) or y not in (0,1) or c not in (0,1,2) or a not in (0,1):
        raise ValueError("Only binary cues and chosen action + explicit context")
    return (2.*x-1.,2.*y-1.)+tuple(float(c==i) for i in range(3))+(
        float(a==0),float(a==1),1.)


def reward_world(c:int,x:int,y:int,a:int):
    """Environment-only truth, never passed as a supervised correct action."""
    if c==0:correct=x^y
    elif c==1:correct=1^(x^y)
    elif c==2:correct=x
    else:raise ValueError("Unknown context")
    return int(a==correct)


def sigmoid(x:float):
    return 1./(1.+math.exp(-max(-35.,min(35.,x))))


class TwoPopulationBrain:
    """Two 8-hidden-unit experts with learned context-conditioned soft gate."""

    def __init__(self,seed:int,arm:str):
        check_seed(seed)
        if arm not in ARMS:raise ValueError("Unknown M5.22 intervention")
        self.seed=seed
        self.arm=arm
        init=random.Random(seed*103_973+7)
        self.w=[init.gauss(0.,.45) for _ in range(PREDICTOR_INPUT_PARAMS)]
        self.w += [init.gauss(0.,.13) for _ in range(GROUPS*HIDDEN_PER_GROUP)]
        self.w += [0.]*GROUPS
        self.w += [0.]*DIM
        assert len(self.w)==PARAMS
        self.founder=tuple(self.w)
        self.policy_rng=random.Random(seed*93_773+83)
        self.replay_rng=random.Random(seed*873_331+197)
        self.random_assignment_rng=random.Random(seed*47_521+911)
        self.memory=tuple(deque(maxlen=n) for n in MEMORY_SLOTS)
        self.lived=0
        self.gradient_update_calls=0
        self.population_update_calls=[0,0]
        self.gate_update_calls=0
        self.freeze_step=None
        self.last_two_block_success=deque(maxlen=2)
        self.frozen_group=None
        self.explored_old_memory_replay=0
        self.last_gates=[]

    def snapshot(self):
        return (tuple(self.w),self.lived,self.gradient_update_calls,
                tuple(self.population_update_calls),self.gate_update_calls,
                self.freeze_step,self.frozen_group,
                tuple(self.last_two_block_success),
                tuple(tuple(mem) for mem in self.memory),
                self.policy_rng.getstate(),self.replay_rng.getstate(),
                self.random_assignment_rng.getstate(),
                tuple(self.last_gates))

    def _predict(self,x:int,y:int,c:int,a:int):
        z=sensory(x,y,c,a)
        h=[]
        logits=[]
        for expert in range(GROUPS):
            current=[]
            for j in range(HIDDEN_PER_GROUP):
                base=expert*HIDDEN_PER_GROUP*DIM+j*DIM
                activation=math.tanh(math.fsum(
                    self.w[base+k]*feature for k,feature in enumerate(z)))
                current.append(activation)
            h.append(current)
            logits.append(self.w[EXPERT_BIAS_OFFSET+expert]+math.fsum(
                self.w[PREDICTOR_OUTPUT_OFFSET+expert*HIDDEN_PER_GROUP+j]
                *current[j] for j in range(HIDDEN_PER_GROUP)))
        gate=sigmoid(math.fsum(
            self.w[GATE_OFFSET+k]*feature for k,feature in enumerate(z)))
        logit=gate*logits[0]+(1.-gate)*logits[1]
        return sigmoid(logit),z,h,logits,gate

    def choose(self,x:int,y:int,c:int,tick:int,training:bool):
        p0=self._predict(x,y,c,0)[0]
        p1=self._predict(x,y,c,1)[0]
        a=int(p1>p0)
        if training:
            frac=min(1.,tick/TRAIN_STEPS)
            eps=.25*(1-frac)+.07*frac
            if self.policy_rng.random()<eps:
                a=self.policy_rng.randrange(2)
        return a

    def _gradient_update(self,record,allowed:tuple[bool,bool]):
        x,y,c,a,outcome=record
        p,z,h,logits,gate=self._predict(x,y,c,a)
        residual=outcome-p
        old=self.w[:]
        proportions=(gate,1.-gate)
        for expert,may_change in enumerate(allowed):
            if not may_change:continue
            scale=residual*proportions[expert]
            for j in range(HIDDEN_PER_GROUP):
                out_idx=(PREDICTOR_OUTPUT_OFFSET+
                         expert*HIDDEN_PER_GROUP+j)
                orig_v=old[out_idx]
                self.w[out_idx]=max(-5.,min(5.,
                    old[out_idx]+LR*scale*h[expert][j]))
                slope=(scale*orig_v*(1.-h[expert][j]**2))
                base=expert*HIDDEN_PER_GROUP*DIM+j*DIM
                for k,feat in enumerate(z):
                    idx=base+k
                    self.w[idx]=max(-5.,min(5.,
                                          old[idx]+LR*slope*feat))
            idx=EXPERT_BIAS_OFFSET+expert
            self.w[idx]=max(-5.,min(5.,old[idx]+LR*scale))
            self.population_update_calls[expert]+=1
        # The soft director has a single action-outcome gradient, and is
        # updated regardless of which expert is currently permitted to learn.
        delta=residual*(logits[0]-logits[1])*gate*(1.-gate)
        for k,feat in enumerate(z):
            idx=GATE_OFFSET+k
            self.w[idx]=max(-5.,min(5.,
                old[idx]+LR*GATE_LR_MULT*delta*feat))
        self.gradient_update_calls+=1
        self.gate_update_calls+=1

    def _allowed(self,tick:int,phase:int,block_index:int,
                 yoked_assignments:dict[int,tuple[int,...]]):
        if self.arm=="joint" or self.arm=="replay_reference":
            return (True,True)
        if self.arm=="memory_director":
            return ((self.frozen_group!=0),True)
        if self.arm=="memory_phase_bound":
            return (tick<400,True)
        if self.arm=="swapped_phase_bound":
            return (True,tick<400)
        if tick<400:return (True,True)
        if self.arm=="alternating":
            active=((tick-400)//BLOCK)%2
        elif self.arm=="alternating_yoked_random":
            active=yoked_assignments[phase][block_index]
        else:raise AssertionError("Bad population schedule")
        return (active==0,active==1)

    def lived_action(self,x,y,c,action,success,allowed):
        if success not in (0,1):
            raise ValueError("Only chosen action's real binary consequence")
        record=(x,y,c,action,success)
        self._gradient_update(record,allowed)
        self.lived+=1
        self.memory[c].append(record)
        if self.arm=="replay_reference":
            combined=[r for buf in self.memory for r in buf]
            for _ in range(4):
                self._gradient_update(
                    self.replay_rng.choice(combined),(True,True))
                self.explored_old_memory_replay+=1

    def update_autonomous_freeze(self,block_success:float):
        if self.arm!="memory_director":return
        self.last_two_block_success.append(block_success)
        if (self.frozen_group is None
                and self.lived>=AUTOFREEZE_MIN_EXPOSURE
                and len(self.last_two_block_success)==2
                and min(self.last_two_block_success)>=AUTOFREEZE_THRESHOLD):
            self.frozen_group=0
            self.freeze_step=self.lived


def frozen_exam(brain,seed,ctx,exam_phase):
    state=brain.snapshot()
    score=0
    gates=[]
    for x,y in cue_sequence(seed,exam_phase,EXAM):
        a=brain.choose(x,y,ctx,0,False)
        score+=reward_world(ctx,x,y,a)
        gates.append(brain._predict(x,y,ctx,a)[-1])
    if state!=brain.snapshot():
        raise AssertionError("Frozen exams must not teach or touch RNG")
    return score/EXAM,statistics.mean(gates)


def schedules(seed:int):
    check_seed(seed)
    rng=random.Random(seed*47_521+911)
    # Exact 50:50 active expert occupancy in EVERY post-A phase;
    # random schedule has same per-phase capacity exposure as alternating.
    out={}
    for phase,(_,n) in enumerate(PHASES):
        if phase==0:continue
        choices=[0]*(n//(2*BLOCK))+[1]*(n//(2*BLOCK))
        rng.shuffle(choices)
        out[phase]=tuple(choices)
    return out


def run(seed:int,arm:str):
    check_seed(seed)
    brain=TwoPopulationBrain(seed,arm)
    planned=schedules(seed)
    scores={}
    gate_exams={}
    tick=0
    for phase,(name,n) in enumerate(PHASES):
        ctx=CTX[name]
        seq=cue_sequence(seed,phase,n)
        for block in range(n//BLOCK):
            values=[]
            for x,y in seq[block*BLOCK:(block+1)*BLOCK]:
                allowed=brain._allowed(tick,phase,block,planned)
                choice=brain.choose(x,y,ctx,tick,True)
                success=reward_world(ctx,x,y,choice)
                brain.lived_action(x,y,ctx,choice,success,allowed)
                values.append(success)
                tick+=1
            brain.update_autonomous_freeze(sum(values)/len(values))
        for prev in range(max(CTX[p[0]] for p in PHASES[:phase+1])+1):
            score,avg_gate=frozen_exam(brain,seed,prev,20+phase*3+prev)
            scores[f"phase_{phase}_{'ABC'[prev]}"]=score
            gate_exams[f"phase_{phase}_{'ABC'[prev]}"]=avg_gate
    if tick!=TRAIN_STEPS or brain.lived!=TRAIN_STEPS:
        raise AssertionError("Unfair training exposure")
    if sum(map(len,brain.memory))>MEMORY_CAP:
        raise AssertionError("Autobiographical memory exceeded")
    expected=TRAIN_STEPS*(5 if arm=="replay_reference" else 1)
    if brain.gradient_update_calls!=expected or brain.gate_update_calls!=expected:
        raise AssertionError("Wrong number of actual gradient update calls")
    if arm=="alternating" or arm=="alternating_yoked_random":
        if brain.population_update_calls!=[960,960]:
            raise AssertionError("Alternating/random are not exposure-yoked")
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
        "pre_refresher_joint_score":statistics.mean((
            scores["phase_2_A"],scores["phase_1_B"],
            scores["phase_2_B"],scores["phase_2_C"])),
        "pre_refresher_worst_skill":min(
            scores["phase_2_A"],scores["phase_2_B"],
            scores["phase_2_C"]),
        "final_three_mean":statistics.mean((
            scores["phase_4_A"],scores["phase_4_B"],
            scores["phase_4_C"])),
        "A_after_C_mean_archive_routing_gate":
            gate_exams["phase_2_A"],
        "B_after_C_mean_archive_routing_gate":
            gate_exams["phase_2_B"],
        "C_after_C_mean_archive_routing_gate":
            gate_exams["phase_2_C"],
        "lived_experiences":brain.lived,
        "gradient_update_calls":brain.gradient_update_calls,
        "population_update_calls":brain.population_update_calls,
        "gate_update_calls":brain.gate_update_calls,
        "autonomous_freeze_step":brain.freeze_step,
        "memory_slots":MEMORY_CAP,
        "memory_usage":sum(map(len,brain.memory)),
        "historical_replay_updates":brain.explored_old_memory_replay,
        "frozen_probes":True,
        "correct_action_labels_not_seen":True,
        "task_identity_explicitly_observed":True,
        "both_experts_have_exactly_8_hidden_neurons":True,
    }


def bootstrap_pair(rows,arm,control,metric="pre_refresher_joint_score",
                   repeats=1200):
    lookup={(r["seed"],r["arm"]):r for r in rows}
    values=[lookup[(seed,arm)][metric]-lookup[(seed,control)][metric]
            for seed in SEEDS]
    sampler=random.Random(123456)
    res=sorted(statistics.mean(sampler.choice(values)
               for _ in values) for _ in range(repeats))
    return {"difference":statistics.mean(values),
            "bootstrap95":[res[int(repeats*.025)],
                           res[int(repeats*.975)-1]],
            "paired_seeds":len(SEEDS)}


def experiment(seeds=SEEDS):
    if tuple(seeds)!=SEEDS:
        raise ValueError("Complete preregistered M5.22 DEV cohort required")
    rows=[run(seed,arm) for seed in seeds for arm in ARMS]
    ix={(r["seed"],r["arm"]):r for r in rows}
    keys=("A_after_A","A_after_B","B_after_B","A_after_C",
          "B_after_C","C_after_C","A_final","B_final","C_final",
          "pre_refresher_joint_score","pre_refresher_worst_skill",
          "final_three_mean","gradient_update_calls",
          "A_after_C_mean_archive_routing_gate",
          "B_after_C_mean_archive_routing_gate",
          "C_after_C_mean_archive_routing_gate")
    summary={arm:{k:statistics.mean(ix[(seed,arm)][k] for seed in seeds)
                   for k in keys} for arm in ARMS}
    autofreeze=[ix[(seed,"memory_director")]["autonomous_freeze_step"]
                for seed in seeds]
    summary["memory_director"]["fraction_froze_at_all"]=(
        sum(v is not None for v in autofreeze)/len(autofreeze))
    summary["memory_director"]["fraction_froze_during_A"]=(
        sum(v is not None and v<=400 for v in autofreeze)/len(autofreeze))
    for arm in ("alternating","alternating_yoked_random"):
        for seed in seeds:
            if ix[(seed,arm)]["population_update_calls"]!=[960,960]:
                raise AssertionError("Population exposure unfair")
    for arm in ARMS:
        for seed in seeds:
            if ix[(seed,arm)]["lived_experiences"]!=TRAIN_STEPS:
                raise AssertionError("Unmatched actual interactions")
    memory=summary["memory_director"]
    alternating=summary["alternating"]
    control=summary["joint"]
    random=summary["alternating_yoked_random"]
    screen={
        "memory_retains_A_after_C_90":memory["A_after_C"]>=SCREEN_THRESHOLD,
        "memory_acquires_B_90":memory["B_after_B"]>=SCREEN_THRESHOLD,
        "memory_retains_B_after_C_90":memory["B_after_C"]>=SCREEN_THRESHOLD,
        "memory_acquires_C_90":memory["C_after_C"]>=SCREEN_THRESHOLD,
        "memory_joint_score_beats_joint_plus5pp":
            memory["pre_refresher_joint_score"]-
            control["pre_refresher_joint_score"]>=SCREEN_ADVANTAGE,
        "alternation_retains_A_after_C_90":
            alternating["A_after_C"]>=SCREEN_THRESHOLD,
        "alternation_acquires_B_90":
            alternating["B_after_B"]>=SCREEN_THRESHOLD,
        "alternation_retains_B_after_C_90":
            alternating["B_after_C"]>=SCREEN_THRESHOLD,
        "alternation_acquires_C_90":
            alternating["C_after_C"]>=SCREEN_THRESHOLD,
        "alternation_joint_score_beats_random_yoked_plus5pp":
            alternating["pre_refresher_joint_score"]-
            random["pre_refresher_joint_score"]>=SCREEN_ADVANTAGE,
    }
    return {
        "status":"M5.22_DEV_two_population_memory_director_and_alternation",
        "seeds":list(seeds),"arms":list(ARMS),
        "curriculum":list(PHASES),
        "new_model_parameters":PARAMS,
        "two_population_hidden_neurons":[8,8],
        "trainable_gate_parameters":DIM,
        "explicit_context_identity_given_to_all_arms":True,
        "manual_phase_archival_has_boundary_oracle":True,
        "autofreeze_uses_only_own_recent_success":True,
        "all_non_replay_same_number_weight_update_calls":True,
        "non_replay_parameter_update_masks_differ":True,
        "alternating_and_random_exactly_matched_population_updates":True,
        "replay_reference_fivefold_weight_update_budget":True,
        "same_memory_cap_all_arms":MEMORY_CAP,
        "no_replay_except_positive_control":True,
        "exams_are_same_four_familiar_cue_pairs":True,
        "no_automatic_synapse_birth_or_pruning":True,
        "independent_task_replication_complete":False,
        "evolution_authorized":False,
        "advance_to_evolution":False,
        "summary":summary,"screening_gate":screen,
        "development_screen_pass":all(screen.values()),
        "memory_architecture_pass":all(screen[k]
            for k in screen if k.startswith("memory_")),
        "alternation_architecture_pass":all(screen[k]
            for k in screen if k.startswith("alternation_")),
        "paired_bootstrap":{
            "memory_vs_joint":bootstrap_pair(
                rows,"memory_director","joint"),
            "alternation_vs_yoked":bootstrap_pair(
                rows,"alternating","alternating_yoked_random"),
            "archive_phase_vs_joint":bootstrap_pair(
                rows,"memory_phase_bound","joint"),
        },
        "raw_rows":rows,
        "rows_sha256":hashlib.sha256(json.dumps(rows,
            sort_keys=True,separators=(",",":")).encode()).hexdigest(),
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,default=Path(
        "results/m5-22/two-population-plasticity.json"))
    args=p.parse_args()
    result=experiment()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",
                        encoding="utf-8")
    print("M5_22_TWO_POPULATIONS "+json.dumps({
        k:result[k] for k in (
            "status","summary","paired_bootstrap","screening_gate",
            "development_screen_pass","memory_architecture_pass",
            "alternation_architecture_pass",
            "advance_to_evolution","rows_sha256")},sort_keys=True),flush=True)


if __name__=="__main__":
    main()
