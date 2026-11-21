"""M5.4B0-R: bounded EIGHT inheritable recurrent numeric genes (18 total).

Separate EXTERNALLY optimised representability test, NOT in-world evolution.
Native 15 input/5 logit/401 weight M5.0 actor, births, and energy unchanged.
The same selected 18-gene vector with recurrence off equals the old 10-gene
numerical scaffold bit-for-bit. Protocol docs/M5_4B0R_PROTOCOL.md was
preregistered at 7ffd8e06510a4c08a80c7610d7e0ff2617aec1a7 BEFORE code.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from dataclasses import dataclass, asdict, fields
from pathlib import Path

from . import m5_0 as base
from . import m5_3b as old
from . import m5_4b0 as b0

N_GENES = 18
RECURRENT_SELF_SCALE = 0.65
RECURRENT_LATERAL_SCALE = 0.35
BOOT_REPS = 20000
ARMS = (
    "recurrent_witness",
    "recurrence_clamped",
    "feedforward_witness",
    "recurrent_scrambled",
    "random_recurrent_0",
    "random_recurrent_1",
    "random_feedforward_0",
    "scripted_greedy_novelty",
)
CONTRASTS = ARMS[1:7]
PROTOCOL_COMMIT = "7ffd8e06510a4c08a80c7610d7e0ff2617aec1a7"


@dataclass(frozen=True)
class Settings(old.Settings):
    development_start: int = 89000
    development_seeds: int = 4
    validation_start: int = 89100
    validation_seeds: int = 8
    seed_start: int = 89200
    seeds: int = 24
    shifted_start: int = 89300
    shifted_seeds: int = 12
    odor_radius: int = 2
    shifted_odor_radius: int = 2
    no_food_controls: int = 0
    external_pool: int = 48
    external_generations: int = 3
    external_offspring: int = 16
    external_elites: int = 6
    external_mutation_sd: float = 0.22

    def validate(self) -> None:
        super().validate()
        if not (
            self.odor_radius == 2 and self.shifted_odor_radius == 2 and
            self.external_pool >= 1 and self.external_generations >= 1 and
            self.external_offspring >= 1 and
            1 <= self.external_elites <= self.external_pool and
            self.external_mutation_sd > 0
        ):
            raise ValueError("18-gene recurrent test controls must remain valid")
        blocks = [
            set(range(self.development_start,
                      self.development_start + self.development_seeds)),
            set(range(self.validation_start,
                      self.validation_start + self.validation_seeds)),
            set(range(self.seed_start, self.seed_start + self.seeds)),
            set(range(self.shifted_start,
                      self.shifted_start + self.shifted_seeds)),
        ]
        historic = set()
        for start,n in (
            (79000,6),(79100,2),(80000,24),(81000,12),(82000,24),
            (83000,12),(84000,24),(85000,12),(86000,16),(87000,8),
            (88000,4),(88100,8),(88200,24),(88300,12),
        ):
            historic.update(range(start,start+n))
        if any(x&historic for x in blocks):
            raise ValueError("Previously sealed ecology worlds cannot be reused")
        if any(blocks[i]&blocks[j]
               for i in range(4) for j in range(i+1,4)):
            raise ValueError("Development/validation/final worlds must be disjoint")
        if self.ticks>=400:
            standard=Settings()
            for f in fields(Settings):
                if getattr(self,f.name)!=getattr(standard,f.name):
                    raise ValueError("Frozen full-study parameter changed: "+f.name)


def to_weights(genes: tuple[float,...], *,
               recurrence_enabled: bool=True) -> tuple[float,...]:
    """Exactly old scaffold + 4 tanh-bounded self + 4 lateral recurrence.

    In each of four neuron states: h_i(t) = tanh(original sensors/biases
        + 0.65*tanh(g_10+i)*h_i(t-1)
        + 0.35*tanh(g_14+i)*h_((i+1)%4)(t-1)).
    This is real numerical recurrence inside the UNCHANGED original
    M5.0.network_logits; no Python-coded movement or novelty behavior.
    """
    if len(genes) not in (10,N_GENES):
        raise ValueError("Only original 10 genes or extended 18 genes accepted")
    if any(not math.isfinite(x) for x in genes):
        raise ValueError("All numerical genes must be finite")
    weights=old.to_weights(tuple(genes[:10]))
    if len(genes)==10 or not recurrence_enabled:
        return weights
    weights=list(weights)
    stride=base.INPUTS+old.HIDDEN+1
    recurrent_start=base.INPUTS
    for i in range(4):
        self_ix=i*stride+recurrent_start+i
        lateral_ix=i*stride+recurrent_start+(i+1)%4
        if self_ix==lateral_ix:
            raise AssertionError("Self/lateral recurrence slots overlap")
        weights[self_ix]=RECURRENT_SELF_SCALE*math.tanh(genes[10+i])
        weights[lateral_ix]=RECURRENT_LATERAL_SCALE*math.tanh(genes[14+i])
    return tuple(weights)


def numeric_genes_digest(genes:tuple[float,...]) -> str:
    import hashlib
    return hashlib.sha256(",".join(
        format(x,".17g") for x in genes
    ).encode("ascii")).hexdigest()[:24]


def simulate_numeric(seed:int, settings: Settings,
                     genes:tuple[float,...], *, shifted:bool=False,
                     recurrence_enabled:bool=True, detailed:bool=False
                     ) -> tuple[dict,list[dict],dict|None]:
    settings.validate()
    weights=to_weights(genes,recurrence_enabled=recurrence_enabled)
    digest=base.genome_digest(weights)
    eco=base.create_ecology(
        seed,settings,"heritable_no_mutation",shifted=shifted
    )
    initial_positions=tuple(a.position for a in eco.agents.values())
    for agent in eco.agents.values():
        agent.genome=weights
        agent.hidden=[0.]*settings.hidden
    eco.initial_genome_digests=tuple(digest for _ in eco.agents)
    food_schedule=base._food_schedule(seed,settings,eco,False)
    checks=[]
    for tick in range(settings.ticks+1):
        if tick:
            ids=list(eco.agents)
            base._local_rng(seed,tick,0,0).shuffle(ids)
            for aid in ids:
                base.act(eco,aid,tick)
            base._food_regeneration(eco,food_schedule[tick-1])
            eco.ticks_elapsed=tick
            if not eco.agents and eco.extinct_at is None:
                eco.extinct_at=tick
        if (tick==0 or tick%settings.checkpoint_every==0
                or tick==settings.ticks):
            ck=base.snapshot(eco,tick)
            ck["seed"]=seed
            ck["shifted"]=int(shifted)
            checks.append(ck)
            if abs(base.energy_balance(eco))>1e-6:
                raise AssertionError("Physical energy ledger mismatch")
    if any(e["genome"]!=digest for e in eco.events if e["type"]=="birth"):
        raise AssertionError("Numerical births must be exact genetic CLONES")
    if any(a.genome!=weights for a in eco.agents.values()):
        raise AssertionError("Living numerical genotype was mutated")
    row={
        "seed":seed,"shifted":int(shifted),
        "survived":int(bool(eco.agents)),
        "alive_final":len(eco.agents),
        "births":eco.births,
        "deaths":eco.deaths,
        "food_eaten":eco.food_eaten,
        "food_spawned":eco.food_spawned,
        "food_remaining":len(eco.food),
        "max_generation":eco.max_generation,
        "extinct_at":eco.extinct_at,
        "motor_mask_corrections":eco.corrected_actions,
        "genotype_digest":digest,
        "numeric_gene_digest":numeric_genes_digest(genes),
        "energy_ledger_residual":base.energy_balance(eco),
        "native_neural_actions":1,
        "scripted_behavior":0,
        "in_world_birth_mutations":0,
        "recurrent_genes_enabled":int(len(genes)==18 and recurrence_enabled),
    }
    trace=None
    if detailed:
        trace={
            "seed":seed,"shifted":shifted,
            "numeric_genes":list(genes),
            "recurrent_genes_enabled":int(len(genes)==18 and recurrence_enabled),
            "initial_positions":[list(x) for x in initial_positions],
            "world_walls":[list(p) for p in sorted(eco.world.walls)],
            "initial_food":[list(p) for p in eco.world.initial_food],
            "checkpoints":checks,
            "events":eco.events,
        }
    return row,checks,trace


def scripted_control(seed:int, settings:Settings, *, shifted:bool=False,
                     detailed:bool=False) -> tuple[dict,list[dict],dict|None]:
    row,checks,trace=b0.scripted_ecological_control(
        seed,settings,shifted=shifted,detailed=detailed
    )
    row=dict(row)
    row["native_neural_actions"]=0
    row["scripted_behavior"]=1
    row["in_world_birth_mutations"]=None
    row["recurrent_genes_enabled"]=0
    return row,checks,trace


def outcome_score(rows:list[dict]) -> list[int]:
    return [
        sum(r["survived"] for r in rows),
        sum(r["alive_final"] for r in rows),
        sum(r["food_eaten"] for r in rows),
        sum(r["births"] for r in rows),
    ]


def _ranking(candidate:dict) -> tuple:
    return tuple(-n for n in candidate["development_score"])+(
        candidate["candidate_id"],
    )


def external_search(settings:Settings, kind:str, *,
                    report:bool=False
                    ) -> tuple[list[dict],list[dict],list[dict],dict,dict]:
    """Two independent equal-budget EXTERNAL researchers' searches.

    Only development and validation worlds are read. All fitness is from
    physically costed original M5.0 real numeric acts and exact clonal births.
    """
    settings.validate()
    if kind not in ("recurrent","feedforward"):
        raise ValueError("Only independently searched recurrent/feedforward")
    rng=random.Random(89411 if kind=="recurrent" else 89412)
    size=N_GENES if kind=="recurrent" else old.N_GENES
    records,develop,validate=[],[],[]
    def evaluate(genes:tuple[float,...],generation:int,
                 parent:int|None,changed:list[int]):
        cid=len(records)
        outcomes=[]
        for seed in range(settings.development_start,
                          settings.development_start+settings.development_seeds):
            row,_,_=simulate_numeric(seed,settings,genes)
            row.update({
                "stage":"development","architecture":kind,
                "candidate_id":cid
            })
            outcomes.append(row)
            develop.append(row)
        record={
            "architecture":kind,
            "candidate_id":cid,
            "external_round":generation,
            "external_parent_candidate_id":parent,
            "external_mutated_gene_indices":changed,
            "genes":list(genes),
            "numeric_gene_digest":numeric_genes_digest(genes),
            "native_401_genome_digest":base.genome_digest(to_weights(genes)),
            "development_score":outcome_score(outcomes),
        }
        records.append(record)
        if report:
            print("M5.4B0R DEVELOPMENT CANDIDATE "+
                  json.dumps(record,sort_keys=True),flush=True)
    for _ in range(settings.external_pool):
        g=tuple(rng.gauss(0.,settings.initial_gene_sd)
                for _ in range(size))
        evaluate(g,0,None,[])
    for generation in range(1,settings.external_generations+1):
        top=sorted(records,key=_ranking)[:settings.external_elites]
        for k in range(settings.external_offspring):
            parent=top[k%len(top)]
            indices=sorted(rng.sample(range(size),2))
            genes=list(parent["genes"])
            for ix in indices:
                genes[ix]+=rng.gauss(0.,settings.external_mutation_sd)
            evaluate(tuple(genes),generation,parent["candidate_id"],indices)
    finalists=sorted(records,key=_ranking)[:settings.external_elites]
    for rank,record in enumerate(finalists):
        results=[]
        for seed in range(settings.validation_start,
                          settings.validation_start+settings.validation_seeds):
            row,_,_=simulate_numeric(seed,settings,tuple(record["genes"]))
            row.update({
                "stage":"validation","architecture":kind,
                "candidate_id":record["candidate_id"],
                "development_rank":rank
            })
            validate.append(row)
            results.append(row)
        record["development_rank"]=rank
        record["validation_score"]=outcome_score(results)
    winner=min(finalists,key=lambda item:(
        *(-x for x in item["validation_score"]),
        item["development_rank"],item["candidate_id"]
    ))
    if len(records)!=(settings.external_pool+
                      settings.external_generations*settings.external_offspring):
        raise AssertionError("Not the frozen external search budget")
    summary={
        "architecture":kind,
        "external_rng_seed":89411 if kind=="recurrent" else 89412,
        "external_optimizer_not_within_world_evolution":True,
        "development_seeds":list(range(
            settings.development_start,
            settings.development_start+settings.development_seeds
        )),
        "validation_seeds":list(range(
            settings.validation_start,
            settings.validation_start+settings.validation_seeds
        )),
        "candidate_count":len(records),
        "source_trial_count":len(develop),
        "validation_trial_count":len(validate),
        "finalist_candidate_ids":[x["candidate_id"] for x in finalists],
        "chosen_candidate_id":winner["candidate_id"],
        "chosen_genes":winner["genes"],
        "chosen_expanded_genome_digest":winner["native_401_genome_digest"],
        "chosen_development_score":winner["development_score"],
        "chosen_validation_score":winner["validation_score"],
    }
    return records,develop,validate,winner,summary


def frozen_controls(rec:dict,ff:dict,settings:Settings
                    ) -> dict[str,tuple[tuple[float,...],bool]|None]:
    rg=tuple(rec["genes"])
    fg=tuple(ff["genes"])
    perm=list(rg)
    random.Random(89470).shuffle(perm)
    if tuple(perm)==rg:
        raise AssertionError("Scrambled same-gene control accidentally identical")
    controls={
        "recurrent_witness":(rg,True),
        "recurrence_clamped":(rg,False),
        "feedforward_witness":(fg,False),
        "recurrent_scrambled":(tuple(perm),True)
    }
    for k in range(2):
        rng=random.Random(89500+k)
        controls["random_recurrent_"+str(k)]=(
            tuple(rng.gauss(0.,settings.initial_gene_sd)
                  for _ in range(N_GENES)),True
        )
    rng=random.Random(89502)
    controls["random_feedforward_0"]=(
        tuple(rng.gauss(0.,settings.initial_gene_sd)
              for _ in range(old.N_GENES)),False
    )
    controls["scripted_greedy_novelty"]=None
    if tuple(controls)!=ARMS:
        raise AssertionError("Prespecified frozen controls altered")
    return controls


def paired_bootstrap(values:list[float],seed:int,reps:int) -> list[float]:
    return list(b0.paired_bootstrap(values,seed,reps))


def analyze(rows:list[dict],settings:Settings, *,
            bootstrap_reps:int=BOOT_REPS)->dict:
    out={}
    for split,start,count,bs in (
        ("main",settings.seed_start,settings.seeds,89601),
        ("shifted",settings.shifted_start,settings.shifted_seeds,89602),
    ):
        seeds=tuple(range(start,start+count))
        sub=[r for r in rows if r["split"]==split]
        if len(sub)!=count*len(ARMS):
            raise AssertionError("Incomplete heldout ecological factorial")
        by={arm:{
            row["seed"]:row for row in sub if row["arm"]==arm
        } for arm in ARMS}
        if any(tuple(sorted(by[arm]))!=seeds for arm in ARMS):
            raise AssertionError("Missing or duplicated same-seed arm")
        armstats={
            arm:{
                "survived_worlds":sum(r["survived"] for r in by[arm].values()),
                "total_worlds":count,
                "worlds_with_births":sum(r["births"]>0 for r in by[arm].values()),
                "mean_births":statistics.mean(r["births"] for r in by[arm].values()),
                "mean_food_eaten":statistics.mean(
                    r["food_eaten"] for r in by[arm].values()
                ),
                "max_generation":max(r["max_generation"]
                                     for r in by[arm].values()),
            } for arm in ARMS
        }
        contrasts={}
        for index,other in enumerate(CONTRASTS):
            deltas=[
                by["recurrent_witness"][seed]["survived"]-
                by[other][seed]["survived"]
                for seed in seeds
            ]
            ci=paired_bootstrap(deltas,bs+index*1000,bootstrap_reps)
            contrasts["recurrent_minus_"+other]={
                "mean_paired_survival_difference":statistics.mean(deltas),
                "world_paired_bootstrap95":ci,
                "positive_lower_bound":bool(ci[0]>0),
                "per_world":dict(zip(map(str,seeds),deltas)),
            }
        out[split]={
            "arm_stats":armstats,
            "contrasts":contrasts,
            "world_count":count,
        }
    main=out["main"]
    hard=out["shifted"]
    mainw=main["arm_stats"]["recurrent_witness"]
    hardw=hard["arm_stats"]["recurrent_witness"]
    g={
        "main_rec_survive_ge12":mainw["survived_worlds"]>=12,
        "main_rec_native_births_ge12":mainw["worlds_with_births"]>=12,
        "main_scripted_viability_ge18":(
            main["arm_stats"]["scripted_greedy_novelty"]["survived_worlds"]>=18
        ),
        "main_rec_beats_own_clamp":(
            main["contrasts"]["recurrent_minus_recurrence_clamped"]
            ["positive_lower_bound"]
        ),
        "main_rec_beats_independently_optimized_feedforward":(
            main["contrasts"]["recurrent_minus_feedforward_witness"]
            ["positive_lower_bound"]
        ),
        "shifted_rec_survive_ge6":hardw["survived_worlds"]>=6,
        "shifted_rec_beats_clamp":(
            hard["contrasts"]["recurrent_minus_recurrence_clamped"]
            ["positive_lower_bound"]
        ),
        "shifted_rec_beats_feedforward":(
            hard["contrasts"]["recurrent_minus_feedforward_witness"]
            ["positive_lower_bound"]
        ),
    }
    g["main_representability_and_recurrence_gain"]=all((
        g["main_rec_survive_ge12"],
        g["main_rec_native_births_ge12"],
        g["main_scripted_viability_ge18"],
        g["main_rec_beats_own_clamp"],
        g["main_rec_beats_independently_optimized_feedforward"],
    ))
    g["independent_hard_transfer"]=all((
        g["shifted_rec_survive_ge6"],
        g["shifted_rec_beats_clamp"],
        g["shifted_rec_beats_feedforward"],
    ))
    return {"main":main,"shifted":hard,"gates":g,
            "method":(
                "Two independent equal-budget EXTERNAL genotype optimizers; "
                "original M5.0 recurrent numeric processor and costed ecology; "
                "neural organism reproduction always yields genetic CLONES. "
                "No within-world evolution, teacher, programmed behavior or "
                "source code manipulation. Comparisons condition on an "
                "externally selected 18-gene genome; only world seeds are "
                "bootstrap units, not independent search repetitions."
            )}


def write_csv(path:Path,records:list[dict])->None:
    if not records:
        raise ValueError("Empty scientific report")
    keys=list(dict.fromkeys(k for row in records for k in row))
    with path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)


def run(settings:Settings,out:Path, *,
        bootstrap_reps:int=BOOT_REPS)->dict:
    settings.validate()
    out.mkdir(exist_ok=True,parents=True)
    archives=[]
    source=[]
    validates=[]
    selected={}
    for kind in ("recurrent","feedforward"):
        full,dev,val,winner,summary=external_search(
            settings,kind,report=True
        )
        archives.extend(full)
        source.extend(dev)
        validates.extend(val)
        selected[kind]=winner
        print("M5.4B0R FROZEN "+json.dumps(
            summary,sort_keys=True),flush=True)
    controls=frozen_controls(
        selected["recurrent"],selected["feedforward"],settings
    )
    bank={
        arm:None if config is None else {
            "genes":list(config[0]),
            "recurrence_enabled":config[1],
            "gene_digest":numeric_genes_digest(config[0]),
            "native_401_genome_digest":base.genome_digest(to_weights(
                config[0],recurrence_enabled=config[1]
            )),
        } for arm,config in controls.items()
    }
    (out/"frozen_full_precision_gene_bank.json").write_text(
        json.dumps({
            "preregistration_commit":PROTOCOL_COMMIT,
            "selected_external_candidate_records":selected,
            "by_arm":bank,
            "not_in_world_evolution":True,
        },sort_keys=True,indent=2)+"\n",encoding="utf-8"
    )
    write_csv(out/"external_768_development_worlds.csv",source)
    write_csv(out/"external_96_validation_worlds.csv",validates)
    (out/"external_192_all_candidate_genes.json").write_text(
        json.dumps(archives,sort_keys=True,indent=2)+"\n",encoding="utf-8"
    )
    allrows=[]
    checkpoints=[]
    traces=[]
    for split,start,n,hard in (
        ("main",settings.seed_start,settings.seeds,False),
        ("shifted",settings.shifted_start,settings.shifted_seeds,True)
    ):
        for seed in range(start,start+n):
            for arm,config in controls.items():
                if config is None:
                    row,ck,trace=scripted_control(
                        seed,settings,shifted=hard,detailed=(seed==start)
                    )
                else:
                    row,ck,trace=simulate_numeric(
                        seed,settings,config[0],shifted=hard,
                        recurrence_enabled=config[1],detailed=(seed==start)
                    )
                row["split"]=split
                row["arm"]=arm
                allrows.append(row)
                for item in ck:
                    item["split"]=split
                    item["arm"]=arm
                checkpoints.extend(ck)
                if trace is not None:
                    trace["split"]=split
                    trace["arm"]=arm
                    traces.append(trace)
                print("M5.4B0R SEALED "+json.dumps({
                    "split":split,"seed":seed,"arm":arm,
                    "survived":row["survived"],
                    "alive_final":row["alive_final"],
                    "births":row["births"],
                    "food_eaten":row["food_eaten"],
                    "genotype_digest":row["genotype_digest"],
                    "energy_residual":row["energy_ledger_residual"],
                },sort_keys=True),flush=True)
    result=analyze(allrows,settings,bootstrap_reps=bootstrap_reps)
    summary={
        "preregistration_commit":PROTOCOL_COMMIT,
        "settings":asdict(settings),
        "external_genome_candidate_count":len(archives),
        "external_development_trial_count":len(source),
        "external_validation_trial_count":len(validates),
        "sealed_trial_count":len(allrows),
        "total_physical_world_trials":len(source)+len(validates)+len(allrows),
        "selected_genotype_control_bank":bank,
        "external_source_selection":{
            k:{
                "candidate_id":v["candidate_id"],
                "genes":v["genes"],
                "development_score":v["development_score"],
                "validation_score":v["validation_score"],
            } for k,v in selected.items()
        },
        **result,
    }
    if summary["total_physical_world_trials"]!=1152 and settings.ticks>=400:
        raise AssertionError("Expected exact 1152 full preregistered worlds")
    write_csv(out/"sealed_288_world_outcomes.csv",allrows)
    write_csv(out/"sealed_population_checkpoints.csv",checkpoints)
    for t in traces:
        (out/("events_%s_%s_%s.json"%(t["split"],t["seed"],t["arm"]))).write_text(
            json.dumps(t,sort_keys=True,indent=2)+"\n",encoding="utf-8"
        )
    (out/"summary.json").write_text(
        json.dumps(summary,sort_keys=True,indent=2)+"\n",encoding="utf-8"
    )
    print("M5.4B0R FINAL RESULT "+json.dumps({
        "selected":summary["external_source_selection"],
        "main":result["main"],"shifted":result["shifted"],
        "gates":result["gates"],
        "worlds":summary["total_physical_world_trials"],
    },sort_keys=True),flush=True)
    return summary


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__)
    for f in fields(Settings):
        parser.add_argument("--"+f.name.replace("_","-"),
                            default=f.default,type=type(f.default))
    parser.add_argument("--out",type=Path,default=Path("results/m5-4b0r"))
    parser.add_argument("--bootstrap-reps",type=int,default=BOOT_REPS)
    d=vars(parser.parse_args())
    dest=d.pop("out")
    reps=d.pop("bootstrap_reps")
    if reps<100:
        raise ValueError("At least 100 deterministic paired bootstrap draws")
    if d["ticks"]>=400 and reps!=BOOT_REPS:
        raise ValueError("Full-study paired bootstrap count is frozen")
    run(Settings(**d),dest,bootstrap_reps=reps)


if __name__=="__main__":
    main()
