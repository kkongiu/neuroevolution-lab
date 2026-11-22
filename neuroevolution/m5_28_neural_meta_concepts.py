"""M5.28: Can a NEURAL meta-learner infer transferable rule representations?

An actual 2-layer nonlinear DeepSets support-to-query network is trained
by backpropagation (Adam), using only rewards for actually selected binary
actions, across independently generated past tasks. Eight support examples
are encoded into a 32-float learned latent code; the network decoder must
answer eight other sensory cues that were NEVER seen in that task.

CRITICAL split: 24 affine Boolean RULES (12 masks x 2 biases) may be used
to meta-train neural shared weights. EIGHT ENTIRE RULES (four masks x two
biases) are deliberately held out from all meta-training. Evaluate on:
(1) familiar rules but NEW seed/sensory splits, (2) unseen masks even
though the GENERATOR family is affine, (3) entirely nonaffine interaction
task stress. Each has 24 / 24 / 12 new independent evaluation seeds.
Meta-model weights are frozen during all these exams.

A knowledge-restricted neural model receives no handcoded parity basis,
GF(2) solver or actual rule coefficients. The SYNTHETIC environment is
affine/parity; that family is known to the experimenter, but NOT embedded
as an algorithm in the neural encoder/decoder. The symbolic GF(2) solver
is only a separately labelled human-programmed POSITIVE REFERENCE, not
a component of the candidate. The latent code consists of 32 FLOAT32
values per context (96 floats = 3,072 bits) and is NOT a bit-efficient
compression of eight 5-bit examples per task; we must not claim 15-bit
neural concepts.

Context identity A/B/C is supplied to the code slot selector, isolating
functional concept transfer from world recognition (M5.24–M5.26).
No teacher correct-action target: every label shown in meta-training
or episodic support is derived from a particular CHOSEN ACTION and its
actually experienced binary success, under the stated deterministic
complementary 2-action assumption. Query reward is consumed for network
meta-training ONLY; no query reward in frozen evaluation.

Three independently initialized real-reward meta-models plus three
matched randomized-reward meta-training controls (equal 60k episodes).
Untrained same-initialization and literal episodic controls.
Fresh 5800/5900/6000 evaluation seeds, no original M5.6 sealed worlds.
No ecological evolution, main merge or learned general intelligence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path

import numpy as np

from neuroevolution.m5_27_rule_compression import AffineCompiler

# Original M5.6 sealed cohorts and M5.27 seeds are not used.
META_SEEDS=(711,719,727)
KNOWN_EVAL_SEEDS=tuple(range(5800,5824))
NOVEL_EVAL_SEEDS=tuple(range(5900,5924))
NONAFFINE_EVAL_SEEDS=tuple(range(6000,6012))
HELD_MASKS=(3,6,10,12)
TRAIN_MASKS=tuple(m for m in range(16) if m not in HELD_MASKS)
TRAIN_RULES=tuple((m,b) for m in TRAIN_MASKS for b in (0,1))
HELD_RULES=tuple((m,b) for m in HELD_MASKS for b in (0,1))
ALL_ARMS=("trained_neural","scrambled_meta_reward","untrained_neural",
          "literal_episodic","symbolic_gf2_reference")
COHORTS={"familiar_rule_new_split":KNOWN_EVAL_SEEDS,
         "completely_heldout_rule":NOVEL_EVAL_SEEDS,
         "nonaffine_counterexample":NONAFFINE_EVAL_SEEDS}
INPUTS=4
CUE_COUNT=16
SUPPORT_N=8
QUERY_N=8
CODE_DIM=32
DECODE_HIDDEN=128
PARAM_COUNT=(5*CODE_DIM+CODE_DIM+
             (4+CODE_DIM)*DECODE_HIDDEN+DECODE_HIDDEN+
             DECODE_HIDDEN+1)
META_TRAIN_EPISODES=60000
META_LR=.001
PASS_GATES={
    "familiar_rule_learned":.65,
    "heldout_rule_learned":.75,
    "heldout_rule_gain":.10,
    "familiar_rule_gain":.15,
}

CUES=np.array([
    [2*((cue>>bit)&1)-1 for bit in range(INPUTS)]
    for cue in range(CUE_COUNT)],dtype=np.float32)


def check_eval_seed(seed:int,cohort:str):
    if (cohort not in COHORTS or type(seed) is not int or
            seed not in COHORTS[cohort]):
        raise ValueError("Only preregistered disjoint evaluation seeds")


def parity_rule(mask:int,bias:int,cue:int):
    if (mask not in range(16) or bias not in (0,1) or
            cue not in range(CUE_COUNT)):
        raise ValueError("Incorrect four-bit Boolean task")
    return ((mask&cue).bit_count()&1)^bias


def observed_reward(mask,bias,cue,chosen_action,nonaffine=False):
    if chosen_action not in (0,1):raise ValueError("Actual binary action only")
    answer=parity_rule(mask,bias,cue)
    if nonaffine:
        # A deliberate distribution shift outside the affine/parity
        # representation family; NEVER passed to the neural predictor.
        answer^=(cue&1)&((cue>>1)&1)
    return int(chosen_action==answer)


def support_split(rng):
    # All splits guaranteed full 5-dimensional affine rank.
    while True:
        support=tuple(sorted(int(v) for v in rng.choice(16,8,replace=False)))
        pivots={}
        for cue in support:
            row=1|(cue<<1)
            while row:
                p=row.bit_length()-1
                if p not in pivots:
                    pivots[p]=row
                    break
                row^=pivots[p]
        if len(pivots)==5:
            return support,tuple(i for i in range(16) if i not in support)


def actual_support(mask,bias,support,rng,nonaffine=False):
    """No privileged correct action enters the learner's API."""
    records=[]
    for cue in support:
        action=int(rng.integers(2))
        outcome=observed_reward(mask,bias,cue,action,nonaffine)
        records.append((cue,action,outcome))
    return tuple(records)


def support_encoding(records):
    features=np.empty((len(records),5),dtype=np.float32)
    for i,(cue,action,outcome) in enumerate(records):
        features[i,:4]=CUES[cue]
        # This is a logical consequence of physically CHOSEN action
        # and observed complementary 0/1 reward; not oracle access.
        implied=action^(1-outcome)
        features[i,4]=2*implied-1
    return features


def query_labels_from_real_actions(mask,bias,cues,rng,nonaffine=False,
                                   corrupted=False,corruption_rng=None):
    targets=np.empty(len(cues),dtype=np.float32)
    for i,cue in enumerate(cues):
        chosen=int(rng.integers(2))
        reward=(int(corruption_rng.integers(2)) if corrupted else
                observed_reward(mask,bias,cue,chosen,nonaffine))
        targets[i]=chosen^(1-reward)
    return targets


class NeuralConceptEncoder:
    """Real trainable neural example encoder + query decoder.

    Mean-pool learned tanh embeddings of past chosen-action outcomes;
    decode each unobserved cue using this latent code. Neither an XOR
    computation nor explicit task coefficients appear in the forward or
    gradient functions.
    """
    def __init__(self,seed:int):
        rng=np.random.default_rng(seed)
        self.p={
            "We":rng.normal(0,.35,(5,CODE_DIM)).astype(np.float32),
            "be":np.zeros(CODE_DIM,dtype=np.float32),
            "Wd":rng.normal(0,.2,(4+CODE_DIM,DECODE_HIDDEN)).astype(np.float32),
            "bd":np.zeros(DECODE_HIDDEN,dtype=np.float32),
            "wo":rng.normal(0,.12,(DECODE_HIDDEN,)).astype(np.float32),
            "bo":np.array(0.,dtype=np.float32),
        }
        self.m={k:np.zeros_like(p) for k,p in self.p.items()}
        self.v={k:np.zeros_like(p) for k,p in self.p.items()}
        self.steps=0

    def digest(self):
        import hashlib
        return hashlib.sha256(b"".join(
            self.p[k].tobytes() for k in sorted(self.p))).hexdigest()

    def encode(self,records):
        s=support_encoding(records)
        return np.tanh(s@self.p["We"]+self.p["be"]).mean(0)

    def predict(self,code,query_cues):
        x=np.concatenate((
            CUES[list(query_cues)],
            np.broadcast_to(code,(len(query_cues),CODE_DIM))),axis=1)
        h=np.tanh(x@self.p["Wd"]+self.p["bd"])
        logits=h@self.p["wo"]+self.p["bo"]
        return 1./(1.+np.exp(-np.clip(logits,-20,20)))

    def train_episode(self,records,query_cues,queried_action_targets,lr=META_LR):
        """Outer query outcomes are real observations from META-TRAIN tasks.

        On evaluation tasks, this method is NEVER called and query outcomes
        are withheld from the entire learner.
        """
        s=support_encoding(records)
        eh=np.tanh(s@self.p["We"]+self.p["be"])
        code=eh.mean(0)
        x=np.concatenate((
            CUES[list(query_cues)],
            np.broadcast_to(code,(len(query_cues),CODE_DIM))),axis=1)
        dh=np.tanh(x@self.p["Wd"]+self.p["bd"])
        p=1./(1.+np.exp(-np.clip(
            dh@self.p["wo"]+self.p["bo"],-20,20)))
        y=np.asarray(queried_action_targets,dtype=np.float32)
        delta=(p-y)/len(y)
        grad={}
        grad["wo"]=dh.T@delta
        grad["bo"]=delta.sum()
        inner=delta[:,None]*self.p["wo"]*(1.-dh*dh)
        grad["Wd"]=x.T@inner
        grad["bd"]=inner.sum(axis=0)
        dz=(inner@self.p["Wd"].T)[:,4:].sum(axis=0)
        back=(dz/len(records))[None,:]*(1.-eh*eh)
        grad["We"]=s.T@back
        grad["be"]=back.sum(axis=0)
        self.steps+=1
        for key in self.p:
            g=np.clip(grad[key],-2.,2.)
            self.m[key]=.9*self.m[key]+.1*g
            self.v[key]=.999*self.v[key]+.001*g*g
            self.p[key]-=lr*(
                (self.m[key]/(1.-.9**self.steps)) /
                (np.sqrt(self.v[key]/(1.-.999**self.steps))+1e-8))
        return float(np.mean((p>=.5)==y))


def meta_train(seed:int,corrupt:bool=False,episodes:int=META_TRAIN_EPISODES):
    if seed not in META_SEEDS or type(seed) is not int:
        raise ValueError("Only preregistered meta-training initializations")
    if episodes<1 or episodes>META_TRAIN_EPISODES:
        raise ValueError("Training episode budget outside protocol")
    model=NeuralConceptEncoder(seed)
    rng=np.random.default_rng(seed*19_493+51)
    noise_rng=np.random.default_rng(seed*79_793+207)
    for _ in range(episodes):
        mask,bias=TRAIN_RULES[int(rng.integers(len(TRAIN_RULES)))]
        support,query=support_split(rng)
        real=actual_support(mask,bias,support,rng,False)
        if corrupt:
            records=tuple((x,a,int(noise_rng.integers(2)))
                          for x,a,_ in real)
        else:
            records=real
        target=query_labels_from_real_actions(
            mask,bias,query,rng,False,corrupted=corrupt,
            corruption_rng=noise_rng)
        model.train_episode(records,query,target)
    if model.steps!=episodes:raise AssertionError("Meta budget mismatch")
    return model


def task_triplet(seed,cohort):
    check_eval_seed(seed,cohort)
    masks=(HELD_MASKS if cohort=="completely_heldout_rule"
           else TRAIN_MASKS)
    rng=np.random.default_rng(seed*817_309+103)
    first=int(rng.choice(masks))
    third=int(rng.choice(tuple(m for m in masks if m!=first)))
    bias=int(rng.integers(2))
    c_bias=int(rng.integers(2))
    return ((first,bias),(first,1-bias),(third,c_bias))


def evaluate_one(model:NeuralConceptEncoder,seed:int,cohort:str,
                 arm:str):
    rules=task_triplet(seed,cohort)
    stress=cohort=="nonaffine_counterexample"
    if arm not in ALL_ARMS:raise ValueError("Unregistered control")
    init_digest=model.digest()
    context_codes=[]
    context_support=[]
    context_queries=[]
    for ctx,(mask,bias) in enumerate(rules):
        rng=np.random.default_rng(seed*200_003+ctx*33_131+91)
        seen,held=support_split(rng)
        records=actual_support(mask,bias,seen,rng,stress)
        context_support.append(records)
        context_queries.append(held)
        context_codes.append(
            model.encode(records) if arm not in (
                "literal_episodic","symbolic_gf2_reference") else None)

    grades={}
    seen_grades={}
    compressed=[]
    for ctx,(mask,bias) in enumerate(rules):
        support=context_support[ctx]
        query=context_queries[ctx]
        if arm=="symbolic_gf2_reference":
            solver=AffineCompiler()
            for cue,chosen,reward in support:
                solver.observe(cue,chosen,reward)
            pred=[solver.predict(cue) for cue in query]
            seen_pred=[solver.predict(cue) for cue,_,_ in support]
            compressed.append(solver.code)
        elif arm=="literal_episodic":
            memo={cue:(chosen^(1-reward))
                  for cue,chosen,reward in support}
            pred=[memo.get(cue,0) for cue in query]
            seen_pred=[memo[cue] for cue,_,_ in support]
        else:
            pred=list(model.predict(
                context_codes[ctx],query)>=.5)
            seen_pred=list(model.predict(
                context_codes[ctx],[r[0] for r in support])>=.5)
        grades["ABC"[ctx]]=sum(
            int(int(action)==(
                parity_rule(mask,bias,cue)^(
                    ((cue&1)&((cue>>1)&1)) if stress else 0)))
            for action,cue in zip(pred,query))/len(query)
        seen_grades["ABC"[ctx]]=sum(
            int(int(action)==(
                parity_rule(mask,bias,cue)^(
                    ((cue&1)&((cue>>1)&1)) if stress else 0)))
            for action,(cue,_,_) in zip(seen_pred,support))/len(support)
    # As in M5.27, A/B/C contexts are EXPLICIT. Each compact task code
    # persists unchanged across B and C. This tests representation quality,
    # not autonomous synaptic consolidation or hidden regime discovery.
    if model.digest()!=init_digest:
        raise AssertionError("Evaluation mutated shared neural weights")
    return {
        "meta_seed":model_seed_of(model),
        "seed":seed,"cohort":cohort,"arm":arm,
        "A_unseen_after_C":grades["A"],
        "B_unseen_after_C":grades["B"],
        "C_unseen_after_C":grades["C"],
        "unseen_mean":statistics.mean(grades.values()),
        "seen_mean":statistics.mean(seen_grades.values()),
        "A_unseen_after_B":grades["A"],
        "B_initial_unseen":grades["B"],
        "C_initial_unseen":grades["C"],
        "task_context_explicitly_given":True,
        "never_seen_cue_patterns_in_exam":QUERY_N,
        "actual_support_experiences_per_context":SUPPORT_N,
        "no_query_outcomes_observed_in_evaluation":True,
        "no_neural_weight_updates_in_evaluation":True,
        "neural_code_dim_float32_each_context":(
            CODE_DIM if arm not in (
                "literal_episodic","symbolic_gf2_reference") else 0),
        "per_context_codes":len(context_codes),
        "code_float32_storage_bits":(
            3*CODE_DIM*32 if arm not in (
                "literal_episodic","symbolic_gf2_reference") else
            15 if arm=="symbolic_gf2_reference" else 3*8*5),
        "symbolic_codes":compressed,
        "rules_used_for_offline_reporting_only":list(rules),
        "frozen_model_SHA256":init_digest,
    }


def model_seed_of(model):
    # Initializer seed is model metadata set below; not task identity.
    return getattr(model,"meta_seed",-1)


def clusters_ci(rows,arm,comparator,cohort,n=1200):
    """Two-stage sampling: neural training seed, then evaluation-task seed.

    Repeated tasks under one trained encoder are correlated, so a naive
    72-row iid bootstrap would understate model-training uncertainty.
    """
    lookup={(r["meta_seed"],r["seed"],r["arm"]):r for r in rows
            if r["cohort"]==cohort}
    eval_seeds=COHORTS[cohort]
    rng=random.Random(4049)
    values=[]
    for meta in META_SEEDS:
        for seed in eval_seeds:
            assert (meta,seed,arm) in lookup
    observed=statistics.mean(
        lookup[(meta,seed,arm)]["unseen_mean"]-
        lookup[(meta,seed,comparator)]["unseen_mean"]
        for meta in META_SEEDS for seed in eval_seeds)
    for _ in range(n):
        sampled_meta=[rng.choice(META_SEEDS) for _ in META_SEEDS]
        mean=statistics.mean(
            lookup[(meta,seed,arm)]["unseen_mean"]-
            lookup[(meta,seed,comparator)]["unseen_mean"]
            for meta in sampled_meta
            for seed in [rng.choice(eval_seeds) for _ in eval_seeds])
        values.append(mean)
    values.sort()
    return {"delta":observed,
            "hierarchical_boot95":[values[int(.025*n)],
                                   values[int(.975*n)-1]],
            "meta_replicates":len(META_SEEDS),
            "task_seeds_per_meta":len(eval_seeds)}


def experiment(episodes=META_TRAIN_EPISODES):
    if episodes!=META_TRAIN_EPISODES:
        raise ValueError("Full preregistered meta-training budget required")
    assert set(HELD_MASKS).isdisjoint(TRAIN_MASKS)
    all_rows=[]
    logs=[]
    for meta in META_SEEDS:
        trained=meta_train(meta,False)
        scrambled=meta_train(meta,True)
        initial=NeuralConceptEncoder(meta)
        for model in (trained,scrambled,initial):
            model.meta_seed=meta
        assert trained.steps==scrambled.steps==episodes
        assert initial.steps==0
        logs.append({
            "meta_seed":meta,"training_rule_count":len(TRAIN_RULES),
            "heldout_rule_count":len(HELD_RULES),
            "real_meta_learning_steps":trained.steps,
            "corrupted_meta_learning_steps":scrambled.steps,
            "true_reward_model_sha256":trained.digest(),
            "corrupted_reward_model_sha256":scrambled.digest(),
            "untrained_init_sha256":initial.digest(),
        })
        models={
            "trained_neural":trained,
            "scrambled_meta_reward":scrambled,
            "untrained_neural":initial,
            "literal_episodic":initial,
            "symbolic_gf2_reference":initial,
        }
        for cohort,eval_seeds in COHORTS.items():
            for seed in eval_seeds:
                for arm in ALL_ARMS:
                    all_rows.append(evaluate_one(
                        models[arm],seed,cohort,arm))
    summary={}
    metrics=("A_unseen_after_C","B_unseen_after_C",
             "C_unseen_after_C","unseen_mean","seen_mean")
    for cohort,eval_seeds in COHORTS.items():
        summary[cohort]={
            arm:{
                metric:statistics.mean(r[metric] for r in all_rows
                    if r["cohort"]==cohort and r["arm"]==arm)
                for metric in metrics}
            for arm in ALL_ARMS}
        summary[cohort]["trained_neural"]["individual_meta_replica_means"]=[
            statistics.mean(r["unseen_mean"] for r in all_rows
                if r["cohort"]==cohort and r["arm"]=="trained_neural"
                and r["meta_seed"]==meta)
            for meta in META_SEEDS]
    familiar=summary["familiar_rule_new_split"]
    held=summary["completely_heldout_rule"]
    true_f= familiar["trained_neural"]["unseen_mean"]
    true_h=held["trained_neural"]["unseen_mean"]
    checks={
        "familiar_rules_unseen_cues_at_least65":
            true_f>=PASS_GATES["familiar_rule_learned"],
        "familiar_rules_advantage_over_initial_at_least15pp":
            true_f-familiar["untrained_neural"]["unseen_mean"]>=
            PASS_GATES["familiar_rule_gain"],
        "familiar_rules_advantage_over_scrambled_at_least15pp":
            true_f-familiar["scrambled_meta_reward"]["unseen_mean"]>=
            PASS_GATES["familiar_rule_gain"],
        "novel_rule_masks_unseen_cues_at_least75":
            true_h>=PASS_GATES["heldout_rule_learned"],
        "novel_rules_beat_literal_episode_plus10pp":
            true_h-held["literal_episodic"]["unseen_mean"]>=
            PASS_GATES["heldout_rule_gain"],
        "novel_rules_beat_untrained_neural_plus10pp":
            true_h-held["untrained_neural"]["unseen_mean"]>=
            PASS_GATES["heldout_rule_gain"],
        "novel_rules_beat_scrambled_meta_reward_plus10pp":
            true_h-held["scrambled_meta_reward"]["unseen_mean"]>=
            PASS_GATES["heldout_rule_gain"],
        "known_affine_symbolic_reference_at_least95":
            held["symbolic_gf2_reference"]["unseen_mean"]>=.95,
        "train_masks_and_heldout_masks_disjoint":
            set(HELD_MASKS).isdisjoint(TRAIN_MASKS),
        "no_eval_queries_entered_meta_learning":
            all(r["no_query_outcomes_observed_in_evaluation"]
                for r in all_rows),
        "frozen_eval_neural_parameters":
            all(r["no_neural_weight_updates_in_evaluation"]
                for r in all_rows),
    }
    return {
        "status":"M5.28_DEV_neural_meta_learned_concept_embedding",
        "meta_training_initializations":list(META_SEEDS),
        "meta_training_episodes_per_seed_and_reward_arm":episodes,
        "meta_training_rule_masks":list(TRAIN_MASKS),
        "heldout_rule_masks":list(HELD_MASKS),
        "meta_training_rule_count":len(TRAIN_RULES),
        "completely_heldout_rule_count":len(HELD_RULES),
        "neural_encoder_dim_float32":CODE_DIM,
        "neural_decoder_hidden":DECODE_HIDDEN,
        "shared_model_float32_parameters":PARAM_COUNT,
        "three_task_code_storage_bits":3*CODE_DIM*32,
        "original_eight_support_experience_bits_per_context":8*5,
        "learned_neural_code_bit_efficiency_claimed":False,
        "no_symbolic_parity_formula_in_neural_model":True,
        "symbolic_reference_separate_researcher_hypothesis":True,
        "meta_train_query_reward_from_actually_chosen_actions":True,
        "eval_query_feedback_unavailable":True,
        "deterministic_two_complementary_actions_assumed":True,
        "task_identity_supplied_to_code_slot":True,
        "genuinely_novel_rule_functions_held_out":True,
        "stresses_entirely_different_nonlinear_function":True,
        "total_eval_rows":len(all_rows),
        "no_evolution_or_sealed_M5_6_cohort_access":True,
        "meta_training_logs":logs,
        "summary":summary,
        "screening_gate":checks,
        "transferable_neural_concept_screen_pass":all(checks.values()),
        "independent_task_family_replication_complete":False,
        "evolution_authorized":False,
        "advance_to_evolution":False,
        "paired_hierarchical_bootstrap":{
            cohort:{
                arm:clusters_ci(all_rows,"trained_neural",arm,cohort)
                for arm in ALL_ARMS if arm!="trained_neural"}
            for cohort in COHORTS},
        "raw_rows":all_rows,
        "raw_rows_sha256":hashlib.sha256(
            json.dumps(all_rows,sort_keys=True,
                       separators=(",",":")).encode()).hexdigest(),
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,default=Path(
        "results/m5-28/neural-concept-transfer-development.json"))
    args=parser.parse_args()
    result=experiment()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(
        result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("M5_28_META_NEURAL "+json.dumps({
        k:result[k] for k in (
            "status","summary","screening_gate",
            "transferable_neural_concept_screen_pass",
            "paired_hierarchical_bootstrap",
            "raw_rows_sha256")},sort_keys=True),flush=True)


if __name__=="__main__":
    main()
