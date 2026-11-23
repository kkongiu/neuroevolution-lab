"""M5.35 — New Boolean FAMILY zero-shot, supervised sequential retention.

Old-family representation: same 85 TRAIN functions / 27 previously known
heldout functions as M5.34, full 3000-step supervised pretraining.
New families: MUX and signed EXACT-K, completely distinct truth tables.
Second phase: 1200 full-batch Adam updates, N new-only, H half sampled
novel examples duplicated (NO old data), R 50:50 OLD TRAIN replay and
novel TRAIN, with common novel episode draws.

This is deliberately researcher-controlled supervised data/rehearsal,
NOT autonomous continual learning, invented operators or ecology.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import statistics
from pathlib import Path

import numpy as np

from neuroevolution.m5_34_learned_representation import (
    SharedRepresentation, entire_function_split, offline_score,
    PRETRAIN_STEPS, ADAM_BETA1, ADAM_BETA2, ADAM_EPS, LR,
    OPTIMIZER_SEEDS,
)
from neuroevolution.m5_32_dual_soft_synapses import DualSoftSynapses
from neuroevolution.m5_33_active_cue import CATALOG, CUES, _seed_plan

NOVEL_SPLIT_SEED = 3535
ADAPT_STEPS = 1200
BATCH = 32
EPISODES = 2
NOVEL_COUNTS = {
    "conditional_mux": (40,16,96),
    "signed_exact_k": (20,12,48),
}
ARM_NAMES = (
    "before_new_families",
    "new_only_all_new",
    "half_novel_unique_no_replay",
    "replay_half_old_half_new",
    "task_fitted_m532_reference",
)
GATES = {
    "novel_replay_min_accuracy": .60,
    "novel_replay_gain_before": .08,
    "novel_newonly_gain_before": .10,
    "old_replay_forgetting_floor": -.05,
    "old_replay_gain_vs_newonly": .05,
    "old_replay_gain_vs_half_novel": .04,
    "novel_replay_vs_newonly_floor": -.10,
}
EXPERIMENT_TITLE = "M5.35 new-family zero-shot and sequential retention (synthetic DEV)"


def _signature(fn):
    return "".join(str(int(fn(c))) for c in CUES)


def enumerate_new_functions():
    """Produce exhaustive MUX and signed EXCAT-K truth *functions*.

    Only full 16-bit truth signatures are exposed to split logic.
    This enumerator, as a researcher-defined environment, is never
    available as a learned input feature to SharedRepresentation.
    """
    mux = {}
    for selector in range(4):
        available = [i for i in range(4) if i != selector]
        for true_bit, false_bit in itertools.permutations(available, 2):
            for invert_true, invert_false, invert_output in itertools.product((0,1),repeat=3):
                def decision(c):
                    if ((c >> selector) & 1):
                        return ((c >> true_bit) & 1) ^ invert_true ^ invert_output
                    return ((c >> false_bit) & 1) ^ invert_false ^ invert_output
                s = _signature(decision)
                mux.setdefault(s, {"cohort": "conditional_mux", "signature":s,
                                   "truth":tuple(int(z) for z in s)})
    exact = {}
    for flipped_bits in range(16):
        for k in (1,2,3):
            for invert_output in (0,1):
                s = _signature(lambda c: int((c ^ flipped_bits).bit_count()==k) ^ invert_output)
                exact.setdefault(s, {"cohort": "signed_exact_k", "signature":s,
                                     "truth":tuple(int(z) for z in s)})
    old = {"".join(map(str, row)) for row in CATALOG.tolist()}
    if len(old) != 112 or len(mux) != 96 or len(exact) != 48:
        raise AssertionError("Expected 112 old, 96 MUX and 48 EXACT-K signatures")
    if set(mux) & old or set(exact) & old or set(mux) & set(exact):
        raise AssertionError("New family truth signatures leak old known grammar")
    return {"conditional_mux":mux, "signed_exact_k":exact}


def new_family_split():
    catalog = enumerate_new_functions()
    rng = np.random.default_rng(NOVEL_SPLIT_SEED)
    newtrain, newheld, unused = [], [], []
    for cohort, (train_n, held_n, total) in NOVEL_COUNTS.items():
        ordered = [catalog[cohort][s] for s in sorted(catalog[cohort])]
        if len(ordered) != total:
            raise AssertionError("Unexpected novel family count")
        order = rng.permutation(len(ordered))
        newtrain.extend(ordered[int(i)] for i in order[:train_n])
        newheld.extend(ordered[int(i)] for i in order[train_n:train_n+held_n])
        unused.extend(ordered[int(i)] for i in order[train_n+held_n:])
    oldtrain, oldheld = entire_function_split()
    sets = [
        {f["signature"] for f in values}
        for values in (oldtrain, oldheld, newtrain, newheld, unused)
    ]
    if [len(s) for s in sets] != [85,27,60,28,56]:
        raise AssertionError("Wrong old/new supervised/test/unused counts")
    if len(set().union(*sets)) != sum(map(len,sets)):
        raise AssertionError("Complete truth-function leakage across partitions")
    return oldtrain, oldheld, newtrain, newheld, unused


def chosen_support(f, episode):
    """Pure environment-side reward generator; same episode for every arm."""
    if type(episode) is not int or episode not in range(EPISODES):
        raise ValueError("Only two preregistered sensory episodes")
    key = int(hashlib.sha256(f["signature"].encode()).hexdigest()[:10],16)
    order, actions, _ = _seed_plan(key, episode)
    support_ids = order[:8]
    records = tuple(
        (int(c), int(actions[c]), int(actions[c] == f["truth"][c]))
        for c in support_ids
    )
    exam = tuple(c for c in CUES if c not in support_ids)
    if len(set(support_ids)) != 8 or len(exam) != 8:
        raise AssertionError("Eight support / eight frozen unseen cue split violated")
    return records, exam


def _episode_batch(labels, rng, n):
    """Use complete truth labels of TRAIN functions only."""
    rows = rng.integers(0, len(labels), size=n)
    target = np.asarray(labels[rows], dtype=np.float32)
    choice = np.argsort(rng.random((n,16)), axis=1)[:, :8]
    mask = np.zeros((n,16), dtype=np.float32)
    mask[np.arange(n)[:,None], choice] = 1.
    inputs = np.concatenate(((1.-2.*target)*mask,mask),axis=1).astype(np.float32)
    return inputs, (1.-2.*target).astype(np.float32), mask


def _clone_base(base):
    output = SharedRepresentation(353500)
    output.params = {k:np.copy(v) for k,v in base.params.items()}
    output.steps = base.steps
    output.pretraining_pairs_correct = True
    output.training_source_signatures = base.training_source_signatures
    assert output.checkpoint()["sha256"] == base.checkpoint()["sha256"]
    return output


def continue_supervised(base, oldtrain, newtrain, seed, strategy):
    """Same 1200 updates and base weights; novel sampled data H and R identical.

    strategy: 'N' all 32 novel episodes; 'H' 16 novel episodes DUPLICATED;
    'R' 16 identical novel draws +16 old TRAIN replay episodes.
    No heldout signatures, rule IDs or exam cues accepted.
    """
    if strategy not in ("N","H","R"):
        raise ValueError("Unknown predeclared continuation policy")
    oldsign = {x["signature"] for x in oldtrain}
    newsign = {x["signature"] for x in newtrain}
    if len(oldsign) != 85 or len(newsign) != 60 or oldsign & newsign:
        raise ValueError("Contaminated or incomplete training functions")
    model = _clone_base(base)
    oldlabels=np.asarray([f["truth"] for f in oldtrain],dtype=np.float32)
    newlabels=np.asarray([f["truth"] for f in newtrain],dtype=np.float32)
    rng_new=np.random.default_rng(seed*3535+417)
    rng_old=np.random.default_rng(seed*3535+719)
    adam_m={k:np.zeros_like(v) for k,v in model.params.items()}
    adam_v={k:np.zeros_like(v) for k,v in model.params.items()}
    for step in range(1,ADAPT_STEPS+1):
        newx,newt,newmask=_episode_batch(newlabels,rng_new,BATCH)
        oldx,oldt,oldmask=_episode_batch(oldlabels,rng_old,BATCH//2)
        if strategy=="N":
            x,t,mask=newx,newt,newmask
        elif strategy=="H":
            x=np.concatenate((newx[:16],newx[:16]),axis=0)
            t=np.concatenate((newt[:16],newt[:16]),axis=0)
            mask=np.concatenate((newmask[:16],newmask[:16]),axis=0)
        else:
            x=np.concatenate((newx[:16],oldx),axis=0)
            t=np.concatenate((newt[:16],oldt),axis=0)
            mask=np.concatenate((newmask[:16],oldmask),axis=0)
        hidden,pred=model._predict_batch(x,model.params)
        derivative=(2./(BATCH*8))*(pred-t)*(1.-mask)*(1.-pred*pred)
        grads={
            "w2": hidden.T @ derivative,
            "b2": derivative.sum(axis=0),
        }
        dh=(derivative @ model.params["w2"].T)*(1.-hidden*hidden)
        grads["w1"]=x.T @ dh
        grads["b1"]=dh.sum(axis=0)
        for k in model.params:
            g=grads[k].astype(np.float32)
            adam_m[k]=ADAM_BETA1*adam_m[k]+(1-ADAM_BETA1)*g
            adam_v[k]=ADAM_BETA2*adam_v[k]+(1-ADAM_BETA2)*g*g
            mhat=adam_m[k]/(1-ADAM_BETA1**step)
            vhat=adam_v[k]/(1-ADAM_BETA2**step)
            model.params[k]=(model.params[k]-LR*mhat/(np.sqrt(vhat)+ADAM_EPS)).astype(np.float32)
    model.steps=base.steps+ADAPT_STEPS
    # These metadata distinguish persistent old weights from new supervised
    # exposure; they are NOT input features for the neural predictor.
    model.training_source_signatures=tuple(
        list(base.training_source_signatures)+list(newsign)
    )
    model.pretraining_pairs_correct=True
    if sum(v.nbytes for v in model.params.values())!=12608:
        raise AssertionError("Shared network size changed")
    model.training_phase2={
        "strategy":strategy, "adam_reset":True,
        "old_train_episodes":ADAPT_STEPS*(16 if strategy=="R" else 0),
        "novel_train_episodes":ADAPT_STEPS*(32 if strategy=="N" else 16),
        "new_sample_stream_half_identical_for_H_and_R":True,
        "gradient_update_count":ADAPT_STEPS,
        "batch_size":BATCH,
    }
    return model


def trial(function, seed, episode, snapshots):
    if seed not in OPTIMIZER_SEEDS:
        raise ValueError("Unregistered initial learning seed")
    records,exam=chosen_support(function,episode)
    arms={}
    for name,model in snapshots.items():
        pred,hidden_norm=model.predictions(records)
        arms[name]={
            "scores":offline_score(pred,function["truth"],exam,records),
            "frozen_all16_predictions":list(pred),
            "hidden_norm":hidden_norm,
        }
    key=int(hashlib.sha256(function["signature"].encode()).hexdigest()[:10],16)
    old_dual=DualSoftSynapses(seed*1000003+key*31+episode*163,0.).fit(records)
    pred_old=old_dual.predictions(CUES)
    arms["task_fitted_m532_reference"]={
        "scores":offline_score(pred_old,function["truth"],exam,records),
        "frozen_all16_predictions":list(pred_old),
        "support_fit_updates":old_dual.gradient_updates,
    }
    return {
        "heldout_signature":function["signature"],
        "cohort":function["cohort"],
        "family_group":"novel" if function["cohort"] in NOVEL_COUNTS else "old",
        "episode":episode,"pretraining_seed":seed,
        "actual_chosen_action_reward_support": [list(v) for v in records],
        "frozen_eight_unobserved_cue_ids": list(exam),
        "report_only_truth_all16":list(function["truth"]),
        "unseen_cues_disjoint":not bool(set(exam)&{v[0] for v in records}),
        "same_actual_eight_experiences_for_all_arms":True,
        "shared_networks_frozen_before_test":True,
        "arms":arms,
    }


def _family_avg(rows, family, arm):
    subset=[r for r in rows if (r["cohort"]==family if family in NOVEL_COUNTS
                                 or family in ("plain_parity","pair_interaction",
                                              "triple_interaction","two_disjoint_pairs")
                                 else r["family_group"]==family)]
    return statistics.mean(
        r["arms"][arm]["scores"]["unseen8_accuracy"] for r in subset
    )


def _metrics(rows, families):
    by_group={}
    for family in (*families,"old","novel"):
        if family in families:
            subset=[r for r in rows if r["cohort"]==family]
        else:
            subset=[r for r in rows if r["family_group"]==family]
        by_group[family]={
            "row_count":len(subset),
            "distinct_functions":len({r["heldout_signature"] for r in subset}),
            "unseen8_accuracy":{
                arm:statistics.mean(r["arms"][arm]["scores"]["unseen8_accuracy"]
                                    for r in subset)
                for arm in ARM_NAMES
            },
            "all16_accuracy_includes_support":{
                arm:statistics.mean(r["arms"][arm]["scores"]["all16_accuracy_including_support"]
                                    for r in subset)
                for arm in ARM_NAMES
            },
        }
    return by_group


def _paired_ci(rows, group, after, before, draws=1000):
    if group in ("old","novel"):
        subset=[r for r in rows if r["family_group"]==group]
    else:
        subset=[r for r in rows if r["cohort"]==group]
    fs=sorted({r["heldout_signature"] for r in subset})
    diffs={
        f:statistics.mean(
            r["arms"][after]["scores"]["unseen8_accuracy"]
            -r["arms"][before]["scores"]["unseen8_accuracy"]
            for r in subset if r["heldout_signature"]==f
        ) for f in fs
    }
    rng=random.Random(353522)
    values=sorted(statistics.mean(
        diffs[rng.choice(fs)] for _ in fs
    ) for _ in range(draws))
    return {
        "difference": statistics.mean(diffs.values()),
        "exploratory_function_cluster_boot95":[values[int(.025*draws)],
                                                 values[int(.975*draws)-1]],
    }


def run():
    oldtrain,oldheld,newtrain,newheld,unused=new_family_split()
    eval_functions=oldheld+newheld
    snapshots={}
    for seed in OPTIMIZER_SEEDS:
        base=SharedRepresentation(seed).pretrain(oldtrain)
        snapshots[seed]={
            "before_new_families":base,
            "new_only_all_new":continue_supervised(base,oldtrain,newtrain,seed,"N"),
            "half_novel_unique_no_replay":continue_supervised(base,oldtrain,newtrain,seed,"H"),
            "replay_half_old_half_new":continue_supervised(base,oldtrain,newtrain,seed,"R"),
        }
        start=snapshots[seed]["before_new_families"].checkpoint()["sha256"]
        if any(snapshots[seed][arm].checkpoint()["sha256"] == start
               for arm in ("new_only_all_new","half_novel_unique_no_replay",
                           "replay_half_old_half_new")):
            raise AssertionError("Adaptation left checkpoint unchanged")
    raw=[
        trial(f,seed,episode,snapshots[seed])
        for seed in OPTIMIZER_SEEDS for f in eval_functions
        for episode in range(EPISODES)
    ]
    families=("plain_parity","pair_interaction","triple_interaction",
              "two_disjoint_pairs","conditional_mux","signed_exact_k")
    summary=_metrics(raw,families)
    old=summary["old"]["unseen8_accuracy"]
    novel=summary["novel"]["unseen8_accuracy"]
    deltas={
        "zero_shot_novel_vs_taskfit_dual":
            novel["before_new_families"]-novel["task_fitted_m532_reference"],
        "novel_replay_gain_from_before":
            novel["replay_half_old_half_new"]-novel["before_new_families"],
        "novel_newonly_gain_from_before":
            novel["new_only_all_new"]-novel["before_new_families"],
        "old_newonly_forgetting":
            old["new_only_all_new"]-old["before_new_families"],
        "old_halfunique_forgetting":
            old["half_novel_unique_no_replay"]-old["before_new_families"],
        "old_replay_forgetting":
            old["replay_half_old_half_new"]-old["before_new_families"],
        "old_replay_gain_over_newonly":
            old["replay_half_old_half_new"]-old["new_only_all_new"],
        "old_replay_gain_over_halfunique":
            old["replay_half_old_half_new"]-old["half_novel_unique_no_replay"],
        "novel_replay_vs_newonly":
            novel["replay_half_old_half_new"]-novel["new_only_all_new"],
    }
    oldtrain_signatures={f["signature"] for f in oldtrain}
    oldheld_signatures={f["signature"] for f in oldheld}
    noveltrain_signatures={f["signature"] for f in newtrain}
    novelheld_signatures={f["signature"] for f in newheld}
    unused_signatures={f["signature"] for f in unused}
    disjoint_sets=(oldtrain_signatures,oldheld_signatures,
                   noveltrain_signatures,novelheld_signatures,unused_signatures)
    all_disjoint=(
        [len(s) for s in disjoint_sets]==[85,27,60,28,56]
        and len(set().union(*disjoint_sets))==256
    )
    code_bytes_ok=all(
        all(sum(v.nbytes for v in model.params.values())==12608
            and all(v.dtype==np.float32 for v in model.params.values())
            for model in snaps.values())
        for snaps in snapshots.values()
    )
    checks={
        "complete_28_novel_heldout_function_signatures_absent_in_train_old112":
            all_disjoint,
        "new_96_mux_48_cardinality_fulltruth_tables_no_old_overlap":
            len(enumerate_new_functions()["conditional_mux"])==96
            and len(enumerate_new_functions()["signed_exact_k"])==48,
        "replay_novel_heldout_unseen8_accuracy_ge60":
            novel["replay_half_old_half_new"] >= GATES["novel_replay_min_accuracy"],
        "replay_novel_unseen8_gain_from_zero_shot_ge8pp":
            deltas["novel_replay_gain_from_before"] >= GATES["novel_replay_gain_before"],
        "newonly_novel_unseen8_gain_from_zero_shot_ge10pp":
            deltas["novel_newonly_gain_from_before"] >= GATES["novel_newonly_gain_before"],
        "replay_old_retention_no_worse_than_minus5pp":
            deltas["old_replay_forgetting"] >= GATES["old_replay_forgetting_floor"],
        "replay_old_beats_newonly_old_ge5pp":
            deltas["old_replay_gain_over_newonly"] >= GATES["old_replay_gain_vs_newonly"],
        "replay_old_beats_halfunique_old_ge4pp":
            deltas["old_replay_gain_over_halfunique"] >= GATES["old_replay_gain_vs_half_novel"],
        "replay_novel_no_more_than10pp_below_newonly":
            deltas["novel_replay_vs_newonly"] >= GATES["novel_replay_vs_newonly_floor"],
        "identical_support_exam_architecture_and_adaptation_budgets":
            len(raw)==220 and code_bytes_ok
            and all(r["unseen_cues_disjoint"] and
                    len(r["actual_chosen_action_reward_support"])==8 and
                    len(r["frozen_eight_unobserved_cue_ids"])==8 and
                    set(r["arms"])==set(ARM_NAMES) for r in raw)
            and all(
                all(snaps[arm].training_phase2["gradient_update_count"]==ADAPT_STEPS
                    and snaps[arm].training_phase2["batch_size"]==BATCH
                    for arm in ("new_only_all_new","half_novel_unique_no_replay",
                                "replay_half_old_half_new"))
                and snaps["half_novel_unique_no_replay"].training_phase2["novel_train_episodes"]
                    == snaps["replay_half_old_half_new"].training_phase2["novel_train_episodes"]
                for snaps in snapshots.values()
            ),
        "all_raw_rows_checkpoints_failures_and_evolution_stop_recorded":
            len(raw)==220
            and all("support8_accuracy" in r["arms"]["before_new_families"]["scores"]
                    for r in raw)
            and len(snapshots)==2,
    }
    intervals={
        group:{
            "replay_vs_before":_paired_ci(raw,group,"replay_half_old_half_new",
                                          "before_new_families"),
            "newonly_vs_before":_paired_ci(raw,group,"new_only_all_new",
                                           "before_new_families"),
            "replay_vs_newonly":_paired_ci(raw,group,"replay_half_old_half_new",
                                          "new_only_all_new"),
            "replay_vs_halfunique":_paired_ci(raw,group,"replay_half_old_half_new",
                                              "half_novel_unique_no_replay"),
        } for group in ("old","novel","conditional_mux","signed_exact_k")
    }
    raw_sha=hashlib.sha256(json.dumps(
        raw,sort_keys=True,separators=(",",":")
    ).encode()).hexdigest()
    return {
        "status":EXPERIMENT_TITLE,
        "metadata":{
            "old_train_count":len(oldtrain),
            "old_heldout_count":len(oldheld),
            "novel_train_count":len(newtrain),
            "novel_heldout_count":len(newheld),
            "novel_completely_unused":len(unused),
            "novel_family_counts":NOVEL_COUNTS,
            "optimizer_pretrain_seeds":OPTIMIZER_SEEDS,
            "pretraining_old_steps":PRETRAIN_STEPS,
            "new_stage_steps":ADAPT_STEPS,
            "batch_size":BATCH,
            "old_training_episodes_per_seed": PRETRAIN_STEPS*32,
            "novel_newonly_episode_draws_per_seed":ADAPT_STEPS*32,
            "novel_halfunique_episode_draws_per_seed":ADAPT_STEPS*16,
            "novel_replay_episode_draws_per_seed":ADAPT_STEPS*16,
            "old_replay_episode_draws_per_seed":ADAPT_STEPS*16,
            "entirely_new_synthetic_function_families_for_zero_shot":True,
            "all_new_family_generators_human_programmed":True,
            "sequential_adaptation_supervised_by_fully_labeled_train_functions":True,
            "old_replay_memory_externally_provided":True,
            "same_neural_update_budget_N_H_R":True,
            "no_new_family_test_truth_in_train":True,
            "all_test_support_actions_random_harness_choices":True,
            "real_eight_support_rewards":True,
            "heldout_signatures_old": sorted(oldheld_signatures),
            "heldout_signatures_new": sorted(novelheld_signatures),
            "new_train_signatures":sorted(noveltrain_signatures),
            "unused_novel_signatures": sorted(unused_signatures),
        },
        "summary":summary,
        "deltas":deltas,
        "exploratory_paired_function_bootstrap":intervals,
        "screening_gates":checks,
        "local_family_transfer_and_retention_screen_pass":all(checks.values()),
        "learned_general_curiosity_policy":False,
        "autonomously_invented_neural_operator":False,
        "shared_meta_plastic_controller_learned":False,
        "independent_realworld_replication_complete":False,
        "autonomous_continual_learning_verified":False,
        "evolution_authorized":False,
        "advance_to_evolution":False,
        "sealed_M5_6_ecology_accessed":False,
        "raw_rows_sha256":raw_sha,
        "raw_rows":raw,
        "shared_neural_float32_checkpoints":{
            str(seed):{arm:model.checkpoint()
                       for arm,model in snapshots[seed].items()}
            for seed in OPTIMIZER_SEEDS
        },
        "posttraining_supervised_exposure":{
            str(seed):{arm:snapshots[seed][arm].training_phase2
                       for arm in ("new_only_all_new",
                                   "half_novel_unique_no_replay",
                                   "replay_half_old_half_new")}
            for seed in OPTIMIZER_SEEDS
        },
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,
                   default=Path("results/m5-35/new-family-retention-dev.json"))
    args=p.parse_args()
    result=run()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n",encoding="utf8")
    print("M5_35_FAMILY_RETENTION "+json.dumps({
        k:result[k] for k in (
            "status","summary","deltas",
            "exploratory_paired_function_bootstrap","screening_gates",
            "local_family_transfer_and_retention_screen_pass","raw_rows_sha256")
    },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
