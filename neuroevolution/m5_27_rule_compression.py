"""M5.27 — Generalize from lived outcomes via compressed concept induction.

This is a deliberately limited SCIENTIFIC TEST: candidate induction uses a
RESEARCHER-PROVIDED hypothesis family (4-variable affine/parity Boolean
functions), not an autonomously learned inductive bias or neural plasticity.
Each truth rule is 5 bits (one bias + four parity coefficients). The agent
learns those 5 bits from real outcomes of its OWN chosen action by streaming
GF(2) Gaussian elimination, retaining only linearly independent evidence
until a rule is uniquely identifiable, then retaining only the 5-bit rule.
It NEVER receives oracle correct-action labels or actual rule coefficients.

Three conflicting observable contexts A/B/C are taught sequentially
A→B→C→A→B. This version exposes context identity intentionally, unlike
M5.26, to isolate *rule generalization* from hidden-context recognition.
For each context, the learner sees ONLY 8 of the 16 possible 4-bit sensory
patterns; ALL OTHER 8 patterns are reserved strictly for frozen exams.
The chosen eight give rank 5 over augmented affine features. Training
experience order, founder neural network and sensory sample streams match
across seven comparators, though experienced actions/rewards differ.

Main DEV cohort 5100..5123 (24 seeds, 7 arms).
Out-of-family stress cohort 5200..5211 (12 seeds, 4 arms) includes an
AND interaction: a representation that assumes affine parity is wrong.
This tests the boundary of the built-in prior, not generalization to
arbitrary unseen task families or fully novel environments.

Comparators: 8-slot *global* novelty episode store, 24-slot contextual
literal episode memory (all eight seen cues in each context), an online
tanh action-reward predictor, an incorrect single-feature concept prior,
a compressed-code learner fed random reward, and a privileged exact-rule
upper-bound. The privileged rule is NEVER passed to non-oracle learners.

No neural memory growing, genetic evolution, sealed M5.6 validation
cohorts or main merge. The 5-bit persistent code per context is *not*
the agent's total memory: transient linear equations use up to 30 bits
per context while being solved, plus code, learned neural parameters and
computation. In the final full-rank affine case, precisely 15 bits of
long-term concept-code state represent all three tasks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from collections import OrderedDict
from pathlib import Path

SEEDS=tuple(range(5100,5124))
STRESS_SEEDS=tuple(range(5200,5212))
ARMS=("concept_compressor","episodic8","episodic24",
      "online_neural","wrong_prior","scrambled_reward","oracle_rule")
STRESS_ARMS=("concept_compressor","episodic24",
             "wrong_prior","oracle_rule")
PHASES=(("A",320),("B",320),("C",320),("A",160),("B",160))
CONTEXT={"A":0,"B":1,"C":2}
BITS=4
RULE_BITS=BITS+1
PATTERNS=1<<BITS
TRAIN_PATTERNS=8
EXAM_REPEATS=10
TRAIN_STEPS=sum(n for _,n in PHASES)
INPUT_DIM=4+3+2+1
HIDDEN=24
NEURAL_PARAMS=HIDDEN*INPUT_DIM+HIDDEN+1
NEURAL_LR=.065
TARGET_MIN=.90
MIN_GAIN_EPISODE=.25
MIN_GAIN_WRONG_PRIOR=.18
MIN_GAIN_SCRAMBLED=.18
MAX_STRESS_UNSEEN=.85


def check_seed(seed:int,stress:bool=False):
    allowed=STRESS_SEEDS if stress else SEEDS
    if type(seed) is not int or seed not in allowed:
        raise ValueError("M5.27 DEV only, explicit disjoint cohorts")


def parity(v:int):
    return v.bit_count()&1


def affine_answer(rule:int,cue:int):
    """Used by environment and by the model's OWN candidate rule.

    Learner cannot read the privileged rule; only the environment contains
    true coefficients.
    """
    if not (0<=rule<32 and 0<=cue<16):
        raise ValueError("Four-bit sensory cue and five-bit rule required")
    return parity(rule & (1|(cue<<1)))


def environment_success(rule:int,cue:int,action:int,stress:bool):
    answer=affine_answer(rule,cue)
    if stress:
        # Deliberate interaction outside the entire affine/parity class.
        answer^=(cue&1) & ((cue>>1)&1)
    if action not in (0,1):raise ValueError("Binary chosen action only")
    return int(action==answer)


def rank_gf2(masks):
    pivots=set()
    for item in masks:
        mask=item
        for p in sorted(pivots,reverse=True):
            if mask&(1<<p):
                mask^=pivots[p]
        if mask:
            pivots[mask.bit_length()-1]=mask
    return len(pivots)


def training_split(seed:int,ctx:int,stress:bool):
    check_seed(seed,stress=stress)
    rng=random.Random(seed*191_911+ctx*331+73)
    # Pick a random 8/8 split, with FULL 5-dimensional affine rank.
    for _ in range(150):
        seen=tuple(sorted(rng.sample(range(PATTERNS),TRAIN_PATTERNS)))
        if rank_gf2(1|(cue<<1) for cue in seen)==RULE_BITS:
            held=tuple(c for c in range(PATTERNS) if c not in seen)
            return seen,held
    raise AssertionError("Unexpected rank-5 split search failure")


def hidden_rule_triplet(seed:int,stress:bool):
    """Lab ground truth NEVER passed to ordinary learner."""
    check_seed(seed,stress=stress)
    rng=random.Random(seed*817_309+103)
    masks=[m for m in range(1,16) if m.bit_count()>=2]
    a_mask=rng.choice(masks)
    c_mask=rng.choice([m for m in masks if m!=a_mask])
    a=(a_mask<<1)|rng.randrange(2)
    b=a^1                 # B negates A on every sensory pattern.
    c=(c_mask<<1)|rng.randrange(2)
    return (a,b,c)


class AffineCompiler:
    """Streaming GF(2) elimination, no correct action shown by instructor."""

    def __init__(self):
        self.pivots={}              # highest pivot -> (feature_mask, outcome)
        self.code=None              # compiled 5-bit functional rule
        self.real_experiences=0
        self.independent_equations=0
        self.violations=0

    def observe(self,cue,chosen_action,actual_reward):
        if actual_reward not in (0,1):
            raise ValueError("Only actually experienced 0/1 feedback")
        self.real_experiences+=1
        # Structural assumption: two actions, one reward=1, the other
        # reward=0. Derive correct action ONLY from chosen action + reward.
        implied_action=chosen_action^(1-actual_reward)
        features=1|(cue<<1)
        if self.code is not None:
            self.violations+=int(parity(features&self.code)!=implied_action)
            return
        rhs=implied_action
        row=features
        while row:
            pivot=row.bit_length()-1
            if pivot not in self.pivots:
                self.pivots[pivot]=(row,rhs)
                self.independent_equations+=1
                break
            mask,target=self.pivots[pivot]
            row^=mask
            rhs^=target
        if not row and rhs:
            self.violations+=1
        if len(self.pivots)==RULE_BITS:
            result=0
            for pivot in sorted(self.pivots):
                mask,outcome=self.pivots[pivot]
                bit=outcome^parity((mask^(1<<pivot))&result)
                result|=bit<<pivot
            self.code=result
            self.pivots.clear()

    def predict(self,cue):
        if self.code is not None:
            return affine_answer(self.code,cue)
        # If the logical system is still underdetermined, avoid
        # pretending that all possible actions are known. Choose an
        # arbitrary zero-free-coefficient solution from known equations.
        result=0
        for pivot in sorted(self.pivots):
            mask,outcome=self.pivots[pivot]
            bit=outcome^parity((mask^(1<<pivot))&result)
            result|=bit<<pivot
        return affine_answer(result,cue)

    def persistent_rule_bits(self):
        return RULE_BITS if self.code is not None else 0

    def transient_basis_bits(self):
        # Each independent row is 5 feature bits + 1 feedback bit.
        return 6*len(self.pivots)


class SingleCuePrior:
    """Deliberately WRONG class: only constants or single-feature rules."""

    def __init__(self):
        self.hypotheses=tuple(
            r for r in range(32) if (r>>1).bit_count()<=1)
        self.errors=[0]*len(self.hypotheses)
        self.events=0

    def observe(self,cue,chosen_action,actual_reward):
        answer=chosen_action^(1-actual_reward)
        for i,model in enumerate(self.hypotheses):
            self.errors[i]+=int(affine_answer(model,cue)!=answer)
        self.events+=1

    def predict(self,cue):
        best=min(range(len(self.hypotheses)),
                 key=lambda i:(self.errors[i],i))
        return affine_answer(self.hypotheses[best],cue)


class EpisodeStore:
    """Literal memory of real chosen-action feedback, no learned rules."""

    def __init__(self,capacity:int):
        self.capacity=capacity
        self.records=OrderedDict()
        self.admitted=0
        self.evictions=0

    def observe(self,ctx,cue,chosen_action,reward):
        if reward not in (0,1):raise ValueError("Real reward only")
        key=(ctx,cue)
        if key in self.records:
            return       # novelty archive preserves first witnessed outcome
        if len(self.records)>=self.capacity:
            self.records.popitem(last=False)
            self.evictions+=1
        self.records[key]=(chosen_action,reward)
        self.admitted+=1

    def predict(self,ctx,cue):
        value=self.records.get((ctx,cue))
        if value is None:return 0     # explicit unknown-input policy
        chosen,reward=value
        return chosen^(1-reward)


class OnlineNeural:
    """One experienced-action reward-predictor gradient per lived step."""

    def __init__(self,seed:int):
        rng=random.Random(seed*103_973+7)
        self.weights=([rng.gauss(0.,.35)
                       for _ in range(HIDDEN*INPUT_DIM)]
                      +[rng.gauss(0.,.13) for _ in range(HIDDEN)]
                      +[0.])
        self.updates=0

    @staticmethod
    def vector(ctx,cue,action):
        return (tuple(2.*((cue>>i)&1)-1. for i in range(BITS))
                +tuple(float(ctx==i) for i in range(3))
                +(float(action==0),float(action==1),1.))

    def calc(self,ctx,cue,action):
        features=self.vector(ctx,cue,action)
        hidden=[math.tanh(sum(
            self.weights[i*INPUT_DIM+j]*v
            for j,v in enumerate(features))) for i in range(HIDDEN)]
        logit=self.weights[-1]+sum(
            self.weights[HIDDEN*INPUT_DIM+i]*h
            for i,h in enumerate(hidden))
        prob=1./(1.+math.exp(-max(-35.,min(35.,logit))))
        return prob,features,hidden

    def predict(self,ctx,cue):
        return int(self.calc(ctx,cue,1)[0]>self.calc(ctx,cue,0)[0])

    def observe(self,ctx,cue,action,reward):
        p,x,h=self.calc(ctx,cue,action)
        error=reward-p
        old=self.weights[:]
        for i,v in enumerate(h):
            out_idx=HIDDEN*INPUT_DIM+i
            self.weights[out_idx]=max(-6.,min(6.,
                old[out_idx]+NEURAL_LR*error*v))
            slope=error*old[out_idx]*(1.-v*v)
            for j,item in enumerate(x):
                idx=i*INPUT_DIM+j
                self.weights[idx]=max(-6.,min(6.,
                    old[idx]+NEURAL_LR*slope*item))
        self.weights[-1]=max(-6.,min(6.,
            old[-1]+NEURAL_LR*error))
        self.updates+=1


class Learner:
    def __init__(self,seed:int,arm:str,stress:bool):
        check_seed(seed,stress=stress)
        if arm not in (STRESS_ARMS if stress else ARMS):
            raise ValueError("Unregistered experimental arm")
        self.seed,self.arm,self.stress=seed,arm,stress
        self.action_rng=random.Random(seed*93_773+83)
        self.noise_rng=random.Random(seed*79_793+207)
        self.compilers=[AffineCompiler() for _ in range(3)]
        self.single=[SingleCuePrior() for _ in range(3)]
        self.episode=EpisodeStore(
            8 if arm=="episodic8" else 24)
        self.neural=OnlineNeural(seed)
        self.lived=0
        self.updates=0

    def select(self,ctx,cue,training:bool):
        if self.arm in ("concept_compressor","scrambled_reward"):
            predicted=self.compilers[ctx].predict(cue)
        elif self.arm=="wrong_prior":
            predicted=self.single[ctx].predict(cue)
        elif self.arm in ("episodic8","episodic24"):
            predicted=self.episode.predict(ctx,cue)
        elif self.arm=="online_neural":
            predicted=self.neural.predict(ctx,cue)
        else:
            raise AssertionError("Oracle prediction must be external")
        if training:
            # Exploration does not give hidden correct action, but generates
            # both failures and successes for learning from real feedback.
            rate=.20*(1.-min(1.,self.lived/TRAIN_STEPS))+.05*min(
                1.,self.lived/TRAIN_STEPS)
            if self.action_rng.random()<rate:
                return self.action_rng.randrange(2)
        return predicted

    def experience(self,ctx,cue,action,real_reward):
        if real_reward not in (0,1):
            raise ValueError("Only chosen action feedback")
        if self.arm=="concept_compressor":
            self.compilers[ctx].observe(cue,action,real_reward)
        elif self.arm=="scrambled_reward":
            self.compilers[ctx].observe(
                cue,action,self.noise_rng.randrange(2))
        elif self.arm=="wrong_prior":
            self.single[ctx].observe(cue,action,real_reward)
        elif self.arm in ("episodic8","episodic24"):
            self.episode.observe(ctx,cue,action,real_reward)
        elif self.arm=="online_neural":
            self.neural.observe(ctx,cue,action,real_reward)
        else:
            raise AssertionError("Oracle is a privileged evaluation only")
        self.lived+=1
        self.updates+=1

    def state(self):
        return (tuple((c.code,tuple(sorted(c.pivots.items())),
                       c.violations,c.real_experiences) for c in self.compilers),
                tuple(tuple(p.errors) for p in self.single),
                tuple(self.episode.records.items()),
                tuple(self.neural.weights),self.neural.updates,
                self.lived,self.updates,
                self.action_rng.getstate(),self.noise_rng.getstate())


def generate_phase_cues(seed:int,ctx:int,phase:int,n:int,stress:bool):
    train,_=training_split(seed,ctx,stress)
    rng=random.Random(seed*211_001+phase*271+107)
    if n%len(train):raise ValueError("Phase balanced eight-cue blocks")
    stream=[]
    for _ in range(n//len(train)):
        block=list(train)
        rng.shuffle(block)
        stream.extend(block)
    return stream


def frozen_exam(learner:Learner,ctx:int,seed:int,which:str,
                stress:bool,oracle_rule:int|None=None):
    seen,held=training_split(seed,ctx,stress)
    group={"seen":seen,"heldout":held}[which]
    initial=learner.state()
    correct=0
    for cue in group:
        for _ in range(EXAM_REPEATS):
            if learner.arm=="oracle_rule":
                assert oracle_rule is not None
                choice=affine_answer(oracle_rule,cue)
                if stress:
                    choice^=(cue&1)&((cue>>1)&1)
            else:
                choice=learner.select(ctx,cue,False)
            correct+=environment_success(
                oracle_rule,cue,choice,stress)
    if learner.state()!=initial:
        raise AssertionError("Frozen exam modified learner/model state")
    return correct/(len(group)*EXAM_REPEATS)


def simulate(seed:int,arm:str,stress:bool=False):
    check_seed(seed,stress=stress)
    if arm not in (STRESS_ARMS if stress else ARMS):
        raise ValueError("Unknown arm/stress pairing")
    rules=hidden_rule_triplet(seed,stress)
    learner=Learner(seed,arm,stress)
    scores={}
    tick=0
    for phase,(name,n) in enumerate(PHASES):
        ctx=CONTEXT[name]
        for cue in generate_phase_cues(seed,ctx,phase,n,stress):
            if arm=="oracle_rule":
                action=affine_answer(rules[ctx],cue)
                if stress:action^=(cue&1)&((cue>>1)&1)
            else:
                action=learner.select(ctx,cue,True)
            reward=environment_success(rules[ctx],cue,action,stress)
            if arm=="oracle_rule":
                learner.lived+=1  # explicitly privileged; zero learning updates
            else:
                learner.experience(ctx,cue,action,reward)
            tick+=1
        for old in range(max(CONTEXT[p[0]] for p in PHASES[:phase+1])+1):
            for which in ("seen","heldout"):
                key=f"phase_{phase}_{'ABC'[old]}_{which}"
                scores[key]=frozen_exam(
                    learner,old,seed,which,stress,oracle_rule=rules[old])
    if learner.lived!=TRAIN_STEPS:
        raise AssertionError("Unmatched lived-action feedback budget")
    if arm!="oracle_rule" and learner.updates!=TRAIN_STEPS:
        raise AssertionError("One online update required per lived trial")
    return {
        "seed":seed,"arm":arm,"stress_nonaffine":stress,
        "A_heldout_after_A":scores["phase_0_A_heldout"],
        "A_heldout_after_B":scores["phase_1_A_heldout"],
        "B_heldout_after_B":scores["phase_1_B_heldout"],
        "A_heldout_after_C":scores["phase_2_A_heldout"],
        "B_heldout_after_C":scores["phase_2_B_heldout"],
        "C_heldout_after_C":scores["phase_2_C_heldout"],
        "A_seen_after_C":scores["phase_2_A_seen"],
        "B_seen_after_C":scores["phase_2_B_seen"],
        "C_seen_after_C":scores["phase_2_C_seen"],
        "A_heldout_final":scores["phase_4_A_heldout"],
        "B_heldout_final":scores["phase_4_B_heldout"],
        "C_heldout_final":scores["phase_4_C_heldout"],
        "joint_unseen_pre_refresher":statistics.mean((
            scores["phase_2_A_heldout"],scores["phase_1_B_heldout"],
            scores["phase_2_B_heldout"],scores["phase_2_C_heldout"])),
        "min_unseen_after_C":min(
            scores["phase_2_A_heldout"],scores["phase_2_B_heldout"],
            scores["phase_2_C_heldout"]),
        "seen_after_C_mean":statistics.mean((
            scores["phase_2_A_seen"],scores["phase_2_B_seen"],
            scores["phase_2_C_seen"])),
        "lived_actions":learner.lived,
        "online_updates":learner.updates,
        "training_cue_count_per_context":[TRAIN_PATTERNS]*3,
        "heldout_cue_count_per_context":[PATTERNS-TRAIN_PATTERNS]*3,
        "compressed_rule_ids":[c.code for c in learner.compilers],
        "rank_or_solved":[(RULE_BITS if c.code is not None
                           else len(c.pivots)) for c in learner.compilers],
        "code_bits_final":sum(c.persistent_rule_bits()
                              for c in learner.compilers),
        "transient_equation_bits_final":sum(
            c.transient_basis_bits() for c in learner.compilers),
        "feedback_inconsistencies":sum(c.violations for c in learner.compilers),
        "episodic_stored_records":len(learner.episode.records),
        "episodic_capacity":learner.episode.capacity,
        "episodic_evictions":learner.episode.evictions,
        "neural_model_parameters":NEURAL_PARAMS,
        "concept_family_affine_parity_supplied_by_researcher":True,
        "current_context_identity_explicitly_available":True,
        "environment_truth_not_passed_to_nonoracle_learner":True,
        "frozen_exams_include_no_feedback_or_updates":True,
    }


def paired_ci(rows,arm,control,metric="joint_unseen_pre_refresher",
              seeds=SEEDS,n=1200):
    ix={(r["seed"],r["arm"]):r for r in rows}
    diffs=[ix[(seed,arm)][metric]-ix[(seed,control)][metric]
           for seed in seeds]
    rng=random.Random(597311)
    boot=sorted(statistics.mean(rng.choice(diffs) for _ in diffs)
                for _ in range(n))
    return {"delta":statistics.mean(diffs),
            "boot95":[boot[int(.025*n)],boot[int(.975*n)-1]],
            "n_paired_seeds":len(seeds)}


def experiment():
    rows=[simulate(seed,arm) for seed in SEEDS for arm in ARMS]
    stress_rows=[simulate(seed,arm,True) for seed in STRESS_SEEDS
                 for arm in STRESS_ARMS]
    lookup={(r["seed"],r["arm"]):r for r in rows}
    keys=("A_heldout_after_A","A_heldout_after_B","B_heldout_after_B",
          "A_heldout_after_C","B_heldout_after_C","C_heldout_after_C",
          "A_seen_after_C","B_seen_after_C","C_seen_after_C",
          "joint_unseen_pre_refresher","min_unseen_after_C",
          "seen_after_C_mean","code_bits_final",
          "transient_equation_bits_final","feedback_inconsistencies",
          "episodic_stored_records","episodic_evictions")
    summary={arm:{key:statistics.mean(
        lookup[(seed,arm)][key] for seed in SEEDS) for key in keys}
        for arm in ARMS}
    st={(r["seed"],r["arm"]):r for r in stress_rows}
    stress_summary={arm:{key:statistics.mean(
        st[(seed,arm)][key] for seed in STRESS_SEEDS)
        for key in ("joint_unseen_pre_refresher",
                    "min_unseen_after_C","feedback_inconsistencies")}
        for arm in STRESS_ARMS}
    candidate=summary["concept_compressor"]
    gain=lambda arm:(candidate["joint_unseen_pre_refresher"]-
                     summary[arm]["joint_unseen_pre_refresher"])
    checks={
        "A_unseen_retained_after_C_90":
            candidate["A_heldout_after_C"]>=TARGET_MIN,
        "B_unseen_retained_after_C_90":
            candidate["B_heldout_after_C"]>=TARGET_MIN,
        "B_unseen_learned_first_90":
            candidate["B_heldout_after_B"]>=TARGET_MIN,
        "C_unseen_learned_first_90":
            candidate["C_heldout_after_C"]>=TARGET_MIN,
        "unseen_joint90":candidate["joint_unseen_pre_refresher"]>=TARGET_MIN,
        "beats_24_literal_records_plus25pp":
            gain("episodic24")>=MIN_GAIN_EPISODE,
        "beats_8_literal_records_plus25pp":
            gain("episodic8")>=MIN_GAIN_EPISODE,
        "beats_wrong_class_plus18pp":
            gain("wrong_prior")>=MIN_GAIN_WRONG_PRIOR,
        "beats_scrambled_reward_plus18pp":
            gain("scrambled_reward")>=MIN_GAIN_SCRAMBLED,
        "all_24_fully_compiled_three_5bit_rules":
            all(lookup[(seed,"concept_compressor")]["code_bits_final"]==15
                for seed in SEEDS),
        "all_24_no_evidence_contradictions_in_affine_family":
            all(lookup[(seed,"concept_compressor")][
                "feedback_inconsistencies"]==0 for seed in SEEDS),
        "out_of_class_stress_not_perfect":
            stress_summary["concept_compressor"][
                "joint_unseen_pre_refresher"]<=MAX_STRESS_UNSEEN,
    }
    for seed in SEEDS:
        if len({lookup[(seed,a)]["lived_actions"] for a in ARMS})!=1:
            raise AssertionError("Different lived school exposure")
        if any(lookup[(seed,a)]["online_updates"]!=TRAIN_STEPS
               for a in ARMS if a!="oracle_rule"):
            raise AssertionError("Expected exactly one online update call")
        if any(lookup[(seed,a)]["episodic_stored_records"]>
               lookup[(seed,a)]["episodic_capacity"] for a in ARMS):
            raise AssertionError("Episode memory cap violated")
        for ctx in range(3):
            seen,held=training_split(seed,ctx,False)
            if set(seen)&set(held) or len(seen)!=8 or len(held)!=8:
                raise AssertionError("Leaked unseen pattern into training")
    return {
        "status":"M5.27_DEV_concept_compression_affine_induction",
        "main_dev_seeds":list(SEEDS),
        "nonaffine_stress_seeds":list(STRESS_SEEDS),
        "main_arms":list(ARMS),"stress_arms":list(STRESS_ARMS),
        "curriculum":list(PHASES),
        "unseen_input_patterns_per_context":8,
        "seen_input_patterns_per_context":8,
        "true_context_label_supplied_to_all_arms":True,
        "true_rule_coefficients_supplied_only_to_environment_and_oracle":True,
        "concept_hypothesis_class_is_programmed_not_neurally_learned":True,
        "concept_encoding_five_bits_per_compiled_context":True,
        "working_GF2_equations_not_included_in_final_code_bits":True,
        "action_complementarity_deterministic_binary_reward_assumed":True,
        "separate_nonaffine_counterexample_included":True,
        "same_training_cue_schedule_per_seed":True,
        "per_arm_update_computational_cost_not_equal":True,
        "not_a_hidden_context_replication_of_M5_26":True,
        "no_original_M5_6_sealed_cohorts":True,
        "main_unchanged":True,"no_evolution":True,
        "summary":summary,"stress_summary":stress_summary,
        "screening_gate":checks,
        "affine_concept_generalization_screen_pass":all(checks.values()),
        "learned_neural_rule_compression_demonstrated":False,
        "independent_task_family_replication_complete":False,
        "evolution_authorized":False,"advance_to_evolution":False,
        "paired_bootstrap":{
            comparator:paired_ci(rows,"concept_compressor",comparator)
            for comparator in ARMS if comparator!="concept_compressor"},
        "raw_rows":rows,"stress_rows":stress_rows,
        "raw_rows_sha256":hashlib.sha256(json.dumps(
            rows,sort_keys=True,separators=(",",":")).encode()).hexdigest(),
        "stress_rows_sha256":hashlib.sha256(json.dumps(
            stress_rows,sort_keys=True,
            separators=(",",":")).encode()).hexdigest(),
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,default=Path(
        "results/m5-27/compact-rule-generalization-development.json"))
    args=p.parse_args()
    report=experiment()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n",
                        encoding="utf-8")
    print("M5_27_RULE_COMPRESSION "+json.dumps({
        k:report[k] for k in (
            "status","summary","stress_summary","screening_gate",
            "affine_concept_generalization_screen_pass",
            "learned_neural_rule_compression_demonstrated",
            "paired_bootstrap","raw_rows_sha256")
    },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
