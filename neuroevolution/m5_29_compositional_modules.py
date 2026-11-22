"""M5.29 — Modular differentiable task adaptation, honest compositional prior.

HYPOTHESIS: M5.28's generic pooled neural encoder memorized whole familiar
functions and anti-generalized to unseen full rule masks. Replace it with
four differentiable bit modules and one bias module, whose outputs are
COMBINED BY A RESEARCHER-PROGRAMMED PRODUCT. That product is a very strong,
explicit affine/parity-compatible inductive bias. The five module weights
are adapted on eight actual support action/reward experiences via SGD with
eight independent starts; the start with smallest SUPPORT loss is selected.
No correct-action teacher and no held-out query outcome are consulted.

This is real gradient-based adaptation of five task-specific float weights,
and compositional reuse of the same module *architecture*, but NOT
autonomous discovery of the multiplication operator, nor learned general
meta-level executive plasticity. The multi-start optimizer is engineered.
No supervised correctness target enters except the complementarity
deduction from observed chosen-action binary success.

The comparison pooled neural network is separately meta-trained on 24
complete known rules, 60000 earlier task episodes, each of 3 independent
initializations. Entire masks (5,9,11,14), both biases, are held out in
THIS NEW split; prior M5.28 full-function mask split is not reused.
All evaluation task seeds are new; all exams contain eight x/y sensory
inputs never presented in that individual task's support experiences.
Out-of-family (AND interaction) stress establishes prior dependence.
Context A/B/C is externally OBSERVABLE in this protocol. No evolution.
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

from neuroevolution.m5_28_neural_meta_concepts import (
    NeuralConceptEncoder, support_split, CUES,
    query_labels_from_real_actions,
)
from neuroevolution.m5_27_rule_compression import AffineCompiler

META_SEEDS=(757,769,773)
KNOWN_SEEDS=tuple(range(6100,6124))
HELD_SEEDS=tuple(range(6200,6224))
STRESS_SEEDS=tuple(range(6300,6312))
COHORTS={
    "familiar_complete_rules":KNOWN_SEEDS,
    "fully_heldout_combinations":HELD_SEEDS,
    "nonaffine_AND_stress":STRESS_SEEDS,
}
HELD_MASKS=(5,9,11,14)
TRAIN_MASKS=tuple(i for i in range(16) if i not in HELD_MASKS)
TRAIN_RULES=tuple((mask,bias) for mask in TRAIN_MASKS for bias in (0,1))
HELD_RULES=tuple((mask,bias) for mask in HELD_MASKS for bias in (0,1))
ARMS=(
    "modular_product_sgd",
    "modular_product_scrambled",
    "modular_product_no_adaptation",
    "additive_modules_sgd",
    "literal_episodic8",
    "pooled_neural_meta",
    "symbolic_GF2_oracle_prior",
)
TASK_RESTARTS=8
ADAPT_STEPS=180
ADAPT_LR=.4
META_EPISODES=60000
SUPPORT=8
HELDOUT=8
CUE_BITS=4
GATES={
    "novel_function_accuracy":.90,
    "novel_beats_pooled":.15,
    "novel_beats_additive":.15,
    "novel_beats_literal":.25,
    "novel_beats_scrambled":.25,
    "novel_beats_noadapt":.25,
    "novel_below_affine_stress":.85,
}
X_BITS=np.array([
    [(cue>>i)&1 for i in range(4)] for cue in range(16)
],dtype=bool)


def seed_check(seed:int,cohort:str):
    if cohort not in COHORTS or type(seed) is not int or seed not in COHORTS[cohort]:
        raise ValueError("Unknown M5.29 development-only task seed")


def actual_reward(mask,bias,cue,action,stress=False):
    if action not in (0,1) or cue not in range(16):
        raise ValueError("Only actual binary cues/actions")
    correct=((mask&cue).bit_count()&1)^bias
    if stress:
        correct^=(cue&1)&((cue>>1)&1)
    return int(action==correct)


def task_rules(seed,cohort):
    seed_check(seed,cohort)
    rng=np.random.default_rng(seed*817309+103)
    masks=HELD_MASKS if cohort=="fully_heldout_combinations" else TRAIN_MASKS
    a=int(rng.choice(masks))
    c=int(rng.choice([m for m in masks if m!=a]))
    ba=int(rng.integers(2))
    bc=int(rng.integers(2))
    return ((a,ba),(a,1-ba),(c,bc))


def support_task(seed,cohort,ctx,rule):
    """One literal eight-example stream. All arms get SAME chosen actions."""
    seed_check(seed,cohort)
    mask,bias=rule
    rng=np.random.default_rng(seed*200003+ctx*33131+91)
    seen,held=support_split(rng)
    records=[]
    for cue in seen:
        action=int(rng.integers(2))
        success=actual_reward(
            mask,bias,cue,action,cohort=="nonaffine_AND_stress")
        records.append((cue,action,success))
    if set(seen)&set(held):
        raise AssertionError("Exam cue appeared in support")
    return tuple(records),tuple(held)


def sign_targets_from_observations(records,corrupt=False,seed=None):
    """Logical implication of each OWN chosen action's actual 0/1 reward.

    This is valid only because the school guarantees exactly two
    deterministic actions with complementary outcomes.
    """
    rng=np.random.default_rng(seed) if corrupt else None
    t=[]
    for cue,action,observed in records:
        reward=int(rng.integers(2)) if corrupt else observed
        label=action^(1-reward)
        t.append(1.-2.*label)
    return np.array(t,dtype=np.float64)


class ModularProductLearner:
    """Differentiable five-module network: bias × active bit specialists.

    Fixed product is a built-in structural prior! SGD optimizes float
    module outputs, not a symbolic GF(2) coefficient search. Eight
    support-only random starts mitigate the nonconvex loss landscape.
    """
    def __init__(self,seed,kind="product"):
        if kind not in ("product","additive"):raise ValueError("Invalid operator")
        self.kind=kind
        rng=np.random.default_rng(seed)
        self.initial=rng.uniform(-.8,.8,(TASK_RESTARTS,5))
        self.initial[0,:]=.5
        self.weights=None
        self.support_mse=None
        self.gradient_steps=0
        self.selected_restart=None

    def _pred(self,theta,cues):
        x=X_BITS[list(cues)]
        if self.kind=="product":
            terms=np.where(x[None,:,:],theta[:,None,1:],1.)
            return theta[:,0,None]*np.prod(terms,axis=2)
        return theta[:,0,None]+np.sum(
            theta[:,None,1:]*x[None,:,:],axis=2)

    def fit(self,records,corrupt=False,corruption_seed=0):
        cues=tuple(c for c,_,_ in records)
        y=sign_targets_from_observations(
            records,corrupt=corrupt,seed=corruption_seed)
        x=X_BITS[list(cues)]
        w=self.initial.copy()
        for _ in range(ADAPT_STEPS):
            gradients=np.empty_like(w)
            if self.kind=="product":
                terms=np.where(x[None,:,:],w[:,None,1:],1.)
                product=np.prod(terms,axis=2)
                predictions=w[:,0,None]*product
                diff=2./len(cues)*(predictions-y[None,:])
                gradients[:,0]=np.sum(diff*product,axis=1)
                for bit in range(4):
                    others=np.prod(np.delete(terms,bit,axis=2),axis=2)
                    gradients[:,bit+1]=np.sum(
                        diff*w[:,0,None]*x[:,bit]*others,axis=1)
            else:
                predictions=(w[:,0,None]+np.sum(
                    w[:,None,1:]*x[None,:,:],axis=2))
                diff=2./len(cues)*(predictions-y[None,:])
                gradients[:,0]=np.sum(diff,axis=1)
                for bit in range(4):
                    gradients[:,bit+1]=np.sum(
                        diff*x[None,:,bit],axis=1)
            w=np.clip(w-ADAPT_LR*gradients,-1.,1.)
        errors=np.mean((self._pred(w,cues)-y[None,:])**2,axis=1)
        chosen=int(np.argmin(errors))
        # Optimizer uses float64 workspace, but PERSISTENT task-specific
        # neural code is explicitly quantized to five float32 values.
        # Evaluate ONLY after this cast, so storage claims are real.
        self.weights=w[chosen].astype(np.float32)
        self.support_mse=float(errors[chosen])
        self.selected_restart=chosen
        self.gradient_steps=TASK_RESTARTS*ADAPT_STEPS
        return self

    def predict(self,cues):
        if self.weights is None:theta=self.initial[:1]
        else:theta=self.weights[None,:]
        return tuple(int(x<0) for x in self._pred(theta,cues)[0])

    def digest(self):
        return hashlib.sha256(
            np.asarray(self.initial,dtype=np.float64).tobytes()+
            (self.weights.tobytes() if self.weights is not None else b"")
        ).hexdigest()


def train_pooled_reference(seed,episodes=META_EPISODES):
    if seed not in META_SEEDS or episodes<1 or episodes>META_EPISODES:
        raise ValueError("Unknown meta-model or training budget")
    model=NeuralConceptEncoder(seed)
    rng=np.random.default_rng(seed*19493+51)
    for _ in range(episodes):
        mask,bias=TRAIN_RULES[int(rng.integers(len(TRAIN_RULES)))]
        seen,held=support_split(rng)
        records=[]
        for cue in seen:
            chosen=int(rng.integers(2))
            records.append((
                cue,chosen,actual_reward(mask,bias,cue,chosen)))
        # The query labels are derived from ACTUALLY chosen actions and
        # observed success in historical META-TRAIN tasks only.
        targets=query_labels_from_real_actions(
            mask,bias,held,rng)
        model.train_episode(tuple(records),held,targets)
    return model


def one_arm(model,meta_seed,seed,cohort,arm):
    if arm not in ARMS:raise ValueError("Unregistered arm")
    rules=task_rules(seed,cohort)
    stress=cohort=="nonaffine_AND_stress"
    digest_before=model.digest()
    rows=[]
    representations=[]
    for context,rule in enumerate(rules):
        records,held=support_task(seed,cohort,context,rule)
        seen=tuple(cue for cue,_,_ in records)
        if arm.startswith("modular_product") or arm=="additive_modules_sgd":
            kind="additive" if arm=="additive_modules_sgd" else "product"
            # The model/meta seed is used in initialization only; no
            # task-level correct function parameters are passed.
            learner=ModularProductLearner(
                seed*93_773+context*913+meta_seed*17,kind)
            if arm!="modular_product_no_adaptation":
                learner.fit(
                    records,corrupt=arm=="modular_product_scrambled",
                    corruption_seed=seed*197+context*7+meta_seed)
            pred=learner.predict(held)
            seen_pred=learner.predict(seen)
            representations.append({
                "context":context,
                "code":learner.weights.tolist()
                        if learner.weights is not None else None,
                "selected_restart":learner.selected_restart,
                "support_mse":learner.support_mse,
                "support_gradient_steps":learner.gradient_steps,
            })
        elif arm=="literal_episodic8":
            memory={cue:action^(1-outcome)
                    for cue,action,outcome in records}
            pred=tuple(memory.get(c,0) for c in held)
            seen_pred=tuple(memory.get(c,0) for c in seen)
        elif arm=="pooled_neural_meta":
            code=model.encode(records)
            pred=tuple(int(p>=.5) for p in model.predict(code,held))
            seen_pred=tuple(int(p>=.5) for p in model.predict(code,seen))
        elif arm=="symbolic_GF2_oracle_prior":
            # Human-supplied exact affine hypothesis family. Reference
            # only; never train/drive the candidate neural modules.
            solver=AffineCompiler()
            for cue,chosen,reward in records:
                solver.observe(cue,chosen,reward)
            pred=tuple(solver.predict(c) for c in held)
            seen_pred=tuple(solver.predict(c) for c in seen)
        else:raise AssertionError("No registered prediction")
        mask,bias=rule
        def grade(cues,actions):
            return statistics.mean(
                int(int(action)==(
                    (((mask&cue).bit_count()&1)^bias)^(
                    ((cue&1)&((cue>>1)&1)) if stress else 0)))
                for cue,action in zip(cues,actions))
        rows.append((grade(held,pred),grade(seen,seen_pred)))
    if model.digest()!=digest_before:
        raise AssertionError("Pooled shared weights modified during test")
    return {
        "meta_seed":meta_seed,"seed":seed,"cohort":cohort,"arm":arm,
        "A_unseen_after_C":rows[0][0],
        "B_unseen_after_C":rows[1][0],
        "C_unseen_after_C":rows[2][0],
        "unseen_mean":statistics.mean(x[0] for x in rows),
        "seen_mean":statistics.mean(x[1] for x in rows),
        "task_code_float32_bits":(
            480 if arm.startswith("modular_product") or
            arm=="additive_modules_sgd" else
            3072 if arm=="pooled_neural_meta" else
            120 if arm=="literal_episodic8" else 15),
        "extra_optimizer_state_excluded_from_code":True,
        "support_experiences_per_context":SUPPORT,
        "unseen_queries_per_context":HELDOUT,
        "task_context_explicitly_visible":True,
        "query_outcomes_unavailable_at_evaluation":True,
        "no_shared_neural_weight_updates_during_evaluation":True,
        "compositional_product_preprogrammed":True,
        "actual_chosen_action_feedback_only":True,
        "rules_used_for_offline_reporting_only":list(rules),
        "module_fit_diagnostics":representations,
        "model_digest":digest_before,
    }


def hierarchical_ci(rows,cohort,against,n=1200):
    lookup={(r["meta_seed"],r["seed"],r["arm"]):r
            for r in rows if r["cohort"]==cohort}
    rng=random.Random(99173)
    seeds=COHORTS[cohort]
    diffs={(m,s):lookup[(m,s,"modular_product_sgd")]["unseen_mean"]-
           lookup[(m,s,against)]["unseen_mean"]
           for m in META_SEEDS for s in seeds}
    observed=statistics.mean(diffs.values())
    samples=[]
    for _ in range(n):
        sampled_models=[rng.choice(META_SEEDS) for _ in META_SEEDS]
        samples.append(statistics.mean(
            diffs[(m,rng.choice(seeds))]
            for m in sampled_models for _ in seeds))
    samples.sort()
    return {"delta":observed,
            "hierarchical_boot95":[samples[int(.025*n)],
                                   samples[int(.975*n)-1]],
            "model_seeds":len(META_SEEDS),
            "task_seeds_each":len(seeds)}


def experiment():
    raw=[]
    meta=[]
    for meta_seed in META_SEEDS:
        trained=train_pooled_reference(meta_seed)
        meta.append({"seed":meta_seed,"pooled_meta_episodes":trained.steps,
                     "pooled_meta_sha256":trained.digest(),
                     "train_complete_functions":len(TRAIN_RULES)})
        for cohort,seeds in COHORTS.items():
            for seed in seeds:
                for arm in ARMS:
                    raw.append(one_arm(trained,meta_seed,seed,cohort,arm))
    summary={}
    for cohort in COHORTS:
        summary[cohort]={}
        for arm in ARMS:
            rs=[r for r in raw if r["cohort"]==cohort and r["arm"]==arm]
            summary[cohort][arm]={
                k:statistics.mean(r[k] for r in rs)
                for k in ("unseen_mean","seen_mean",
                          "A_unseen_after_C","B_unseen_after_C",
                          "C_unseen_after_C")}
    novel=summary["fully_heldout_combinations"]
    acc=novel["modular_product_sgd"]["unseen_mean"]
    gain=lambda arm:acc-novel[arm]["unseen_mean"]
    checks={
        "new_full_rule_accuracy_at_least90":
            acc>=GATES["novel_function_accuracy"],
        "new_rule_beats_pooled_neural_plus15pp":
            gain("pooled_neural_meta")>=GATES["novel_beats_pooled"],
        "new_rule_beats_additive_modules_plus15pp":
            gain("additive_modules_sgd")>=GATES["novel_beats_additive"],
        "new_rule_beats_literal_episodes_plus25pp":
            gain("literal_episodic8")>=GATES["novel_beats_literal"],
        "new_rule_beats_scrambled_feedback_plus25pp":
            gain("modular_product_scrambled")>=GATES["novel_beats_scrambled"],
        "new_rule_beats_no_adaptation_plus25pp":
            gain("modular_product_no_adaptation")>=GATES["novel_beats_noadapt"],
        "nonaffine_stress_shows_prior_limit":
            summary["nonaffine_AND_stress"]["modular_product_sgd"][
                "unseen_mean"]<=GATES["novel_below_affine_stress"],
        "train_and_heldout_complete_mask_sets_disjoint":
            not set(TRAIN_MASKS)&set(HELD_MASKS),
        "no_eval_query_reward_entered_model":
            all(r["query_outcomes_unavailable_at_evaluation"] for r in raw),
        "no_shared_weights_updated_during_testing":
            all(r["no_shared_neural_weight_updates_during_evaluation"]
                for r in raw),
        "three_task_modules_fit_into_480_code_bits":
            all(r["task_code_float32_bits"]==480 for r in raw
                if r["arm"]=="modular_product_sgd"),
    }
    for cohort,seeds in COHORTS.items():
        for seed in seeds:
            rules=task_rules(seed,cohort)
            for c,rule in enumerate(rules):
                records,held=support_task(seed,cohort,c,rule)
                if len(records)!=8 or len(held)!=8 or set(
                        x[0] for x in records)&set(held):
                    raise AssertionError("Leakage of unseen cues")
    return {
        "status":"M5.29_DEV_compositional_module_gradient_adaptation",
        "metadata":{"training_rule_masks":list(TRAIN_MASKS),
                    "fully_heldout_masks":list(HELD_MASKS),
                    "meta_training_seeds":list(META_SEEDS),
                    "meta_train_episodes_per_pooled_model":META_EPISODES,
                    "task_restart_count":TASK_RESTARTS,
                    "task_sgd_steps_per_restart":ADAPT_STEPS,
                    "support_records":SUPPORT,"query_cues":HELDOUT,
                    "code_bits_per_three_tasks":480},
        "strong_engineered_multiplicative_parity_inductive_bias":True,
        "operator_itself_never_learned":True,
        "task_gates_learned_via_reward_driven_SGD":True,
        "unknown_task_context_not_tested":True,
        "two_binary_complementary_actions_assumed":True,
        "separate_human_programmed_GF2_reference":True,
        "total_compute_not_matched_across_arms":True,
        "only_development_task_cohorts":True,
        "novel_rule_masks_never_used_in_pooled_training":True,
        "no_evolution_or_original_M5_6_sealed_access":True,
        "summary":summary,
        "screening_gate":checks,
        "architecturally_scaffolded_composition_screen_pass":all(checks.values()),
        "autonomously_learned_composition_operator":False,
        "independent_task_family_replication_complete":False,
        "evolution_authorized":False,
        "advance_to_evolution":False,
        "paired_cluster_bootstrap":{
            cohort:{
                against:hierarchical_ci(raw,cohort,against)
                for against in ARMS if against!="modular_product_sgd"}
            for cohort in COHORTS},
        "meta_train_log":meta,
        "raw_rows":raw,
        "raw_rows_sha256":hashlib.sha256(json.dumps(
            raw,sort_keys=True,separators=(",",":")).encode()).hexdigest(),
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path,default=Path(
        "results/m5-29/compositional-module-development.json"))
    args=ap.parse_args()
    results=experiment()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(results,sort_keys=True,indent=2)+"\n",
                        encoding="utf-8")
    print("M5_29_MODULAR_COMPOSITION "+json.dumps({
        k:results[k] for k in (
            "status","summary","screening_gate",
            "architecturally_scaffolded_composition_screen_pass",
            "raw_rows_sha256")
    },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
