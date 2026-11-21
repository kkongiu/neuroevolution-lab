"""M5.4B0: externally search the ORIGINAL 10-gene neural scaffold for odor-2 viability.

NO natural evolution takes place in a simulated ecology in this experiment.
Real agents use untouched m5_0.act and physical laws; the researcher applies
a fixed-budget external genome ranking ONLY on development/validation worlds.
All organisms born in a test world inherit exact numeric clones. Scripted
M5.1 is separately labeled as a PROGRAMMED ecological positive reference.

Protocol: docs/M5_4B0_PROTOCOL.md, frozen commit
fa45399a24cb52f0f8d95436cad87a273e991a54.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from . import m5_0 as physics
from . import m5_1 as scripted
from . import m5_3b as genome

EXTERNAL_OPTIMIZER = "researcher_external_numeric_search"
NEURAL_ARMS = ("witness", "scrambled", "random_0", "random_1",
               "random_2", "random_3")
SCRIPT_ARM = "scripted_greedy_novelty"
ALL_ARMS = NEURAL_ARMS + (SCRIPT_ARM,)
SCRAMBLE_SEED = 88470
BOOT_MAIN_SEED = 88701
BOOT_SHIFTED_SEED = 88702
BOOT_REPS = 20000


@dataclass(frozen=True)
class Settings(genome.Settings):
    development_start: int = 88000
    development_seeds: int = 4
    validation_start: int = 88100
    validation_seeds: int = 8
    search_seed: int = 88191
    initial_pool: int = 48
    generations: int = 3
    offspring_per_generation: int = 16
    keep_elites: int = 6
    external_mutation_sd: float = .22
    random_controls: int = 4
    seed_start: int = 88200
    seeds: int = 24
    shifted_start: int = 88300
    shifted_seeds: int = 12
    odor_radius: int = 2
    shifted_odor_radius: int = 2
    no_food_controls: int = 0

    def validate(self) -> None:
        super().validate()
        if (min(self.development_seeds, self.validation_seeds,
                self.initial_pool, self.generations,
                self.offspring_per_generation, self.keep_elites,
                self.random_controls) < 1 or
                self.keep_elites > self.initial_pool or
                self.external_mutation_sd <= 0 or
                self.odor_radius != 2 or self.shifted_odor_radius != 2 or
                self.random_controls != 4):
            raise ValueError("External numerical search configuration invalid")
        blocks=[
            set(range(self.development_start,
                      self.development_start+self.development_seeds)),
            set(range(self.validation_start,
                      self.validation_start+self.validation_seeds)),
            set(range(self.seed_start,self.seed_start+self.seeds)),
            set(range(self.shifted_start,self.shifted_start+self.shifted_seeds))
        ]
        historic=set()
        for first,count in (
            (79000,6),(79100,2),(80000,24),(81000,12),(82000,24),
            (83000,12),(84000,24),(85000,12),(86000,16),(87000,8),
        ):
            historic.update(range(first,first+count))
        if any(block & historic for block in blocks):
            raise ValueError("M5.4B0 reused an earlier sealed world")
        if any(blocks[i]&blocks[j] for i in range(4) for j in range(i+1,4)):
            raise ValueError("Development, validation and sealed worlds overlap")
        if self.ticks >= 400:
            canonical=Settings()
            for f in fields(Settings):
                if getattr(self,f.name)!=getattr(canonical,f.name):
                    raise ValueError(
                        "M5.4B0 full campaign frozen protocol: "+f.name
                    )


def _metric_signature(row: dict) -> tuple[int,int,int,int]:
    return (
        int(row["survived"]), int(row["alive_final"]),
        int(row["food_eaten"]), int(row["births"])
    )


def simulate_neural(seed: int, settings: Settings,
                    genes: tuple[float,...], *,
                    shifted: bool=False, detailed: bool=False
                    ) -> tuple[dict,list[dict],dict | None]:
    """Exact unmodified numerical M5.0 actor, original full physical world.

    All 20 initial founders share this genotype and every newborn is an
    unmutated clone. This is NOT a within-world genetic selection experiment.
    """
    settings.validate()
    weights=genome.to_weights(genes)
    digest=physics.genome_digest(weights)
    eco=physics.create_ecology(
        seed,settings,"heritable_no_mutation",shifted=shifted
    )
    for agent in eco.agents.values():
        agent.genome=weights
        agent.hidden=[0.]*settings.hidden
    eco.initial_genome_digests=tuple(digest for _ in eco.agents)
    schedule=physics._food_schedule(seed,settings,eco,False)
    checks=[]
    for tick in range(settings.ticks+1):
        if tick:
            ids=list(eco.agents)
            physics._local_rng(seed,tick,0,0).shuffle(ids)
            for aid in ids:
                physics.act(eco,aid,tick)
            physics._food_regeneration(eco,schedule[tick-1])
            eco.ticks_elapsed=tick
            if not eco.agents and eco.extinct_at is None:
                eco.extinct_at=tick
        if (tick==0 or tick%settings.checkpoint_every==0
                or tick==settings.ticks):
            ck=physics.snapshot(eco,tick)
            ck.update({
                "seed":seed,"shifted":int(shifted),
                "numeric_genome_digest":digest
            })
            checks.append(ck)
            if abs(physics.energy_balance(eco))>1e-6:
                raise AssertionError("Numerical agent violated original energy laws")
    if any(e["genome"]!=digest for e in eco.events
           if e["type"]=="birth"):
        raise AssertionError("A clonal numerical birth altered its genotype")
    if any(agent.genome!=weights for agent in eco.agents.values()):
        raise AssertionError("Living numeric genome changed inside-world")
    row={
        "seed":seed, "shifted":int(shifted),
        "survived":int(bool(eco.agents)), "alive_final":len(eco.agents),
        "births":eco.births, "deaths":eco.deaths,
        "food_eaten":eco.food_eaten,"food_spawned":eco.food_spawned,
        "food_remaining":len(eco.food),
        "max_generation":eco.max_generation,
        "extinct_at":eco.extinct_at,
        "motor_mask_corrections":eco.corrected_actions,
        "genotype_digest":digest,
        "numeric_gene_digest":genome.gene_digest(genes),
        "energy_ledger_residual":physics.energy_balance(eco),
        "all_actions_native_m50_neural":1,
        "programmed_policy":0,
        "internal_birth_mutations":0
    }
    trace=None
    if detailed:
        trace={
            "seed":seed,"shifted":shifted,
            "genome":list(genes),"genotype_digest":digest,
            "walls":[list(p) for p in sorted(eco.world.walls)],
            "initial_food":[list(p) for p in eco.world.initial_food],
            "initial_founders":[
                list(a.position) for a in sorted((
                    physics.create_ecology(
                        seed,settings,"heritable_no_mutation",
                        shifted=shifted
                    ).agents.values()),key=lambda a:a.id)
            ],
            "checkpoints":checks,"events":eco.events,
            "final_agents":[{
                "id":a.id,"founder":a.founder,"parent":a.parent,
                "generation":a.generation,"position":list(a.position),
                "energy":a.energy
            } for a in sorted(eco.agents.values(),key=lambda a:a.id)]
        }
    return row,checks,trace


def scripted_ecological_control(seed:int,settings: Settings, *,
                               shifted:bool=False,detailed:bool=False
                               ) -> tuple[dict,list[dict],dict|None]:
    native_script_settings=scripted.Settings(
        **{field.name:getattr(settings,field.name)
           for field in fields(scripted.Settings)
           if hasattr(settings,field.name)},
        negative_control_worlds=min(4,settings.seeds)
    )
    source,checks,trace=scripted.simulate(
        seed,native_script_settings,"greedy_novelty",
        shifted=shifted,detailed=detailed
    )
    row={
        "seed":seed,"shifted":int(shifted),
        "survived":source["survived"],
        "alive_final":source["alive_final"],
        "births":source["births"],"deaths":source["deaths"],
        "food_eaten":source["food_eaten"],
        "food_spawned":source["food_spawned"],
        "food_remaining":source["food_remaining"],
        "max_generation":source["max_generation"],
        "extinct_at":source["extinct_at"],
        "motor_mask_corrections":source["motor_mask_corrections"],
        "genotype_digest":None,"numeric_gene_digest":None,
        "energy_ledger_residual":source["energy_ledger_residual"],
        "all_actions_native_m50_neural":0,
        "programmed_policy":1,
        "internal_birth_mutations":0
    }
    return row,checks,trace


def _score(rows: list[dict]) -> tuple[int,int,int,int]:
    return (
        sum(r["survived"] for r in rows),
        sum(r["alive_final"] for r in rows),
        sum(r["food_eaten"] for r in rows),
        sum(r["births"] for r in rows)
    )


def _ranking(record: dict) -> tuple[int,int,int,int,int]:
    return tuple(-x for x in record["development_score"])+(
        record["candidate_id"],
    )


def external_search(settings: Settings, *,
                    trace:bool=False
                    ) -> tuple[list[dict],list[dict],list[dict],dict,dict]:
    """Fixed seed and budget. Absolutely no sealed-world accesses here."""
    settings.validate()
    rng=random.Random(settings.search_seed)
    records=[]
    development=[]
    def evaluate(g:tuple[float,...],round_number:int,
                 parent_id:int|None,
                 changed:list[int]) -> None:
        candidate_id=len(records)
        rows=[]
        for seed in range(
            settings.development_start,
            settings.development_start+settings.development_seeds
        ):
            row,_,_=simulate_neural(seed,settings,g)
            row["stage"]="development"
            row["candidate_id"]=candidate_id
            rows.append(row)
            development.append(row)
        item={
            "candidate_id":candidate_id,
            "external_optimization_round":round_number,
            "parent_candidate_id":parent_id,
            "externally_mutated_gene_indices":changed,
            "gene_values":list(g),
            "gene_digest":genome.gene_digest(g),
            "expanded_genome_digest":physics.genome_digest(
                genome.to_weights(g)
            ),
            "development_score":list(_score(rows)),
        }
        records.append(item)
        if trace:
            print(
                "M5.4B0 DEVELOPMENT "
                +json.dumps({
                    "candidate_id":candidate_id,
                    "round":round_number,
                    "score":item["development_score"]
                },sort_keys=True),flush=True
            )
    for _ in range(settings.initial_pool):
        genes=tuple(
            rng.gauss(0.,settings.initial_gene_sd)
            for _ in range(genome.N_GENES)
        )
        evaluate(genes,0,None,[])
    for gen in range(1,settings.generations+1):
        elite=sorted(records,key=_ranking)[:settings.keep_elites]
        if not elite:
            raise AssertionError("External optimization has no parents")
        for child_index in range(settings.offspring_per_generation):
            parent=elite[child_index % len(elite)]
            indices=sorted(rng.sample(range(genome.N_GENES),2))
            genes=list(parent["gene_values"])
            for ix in indices:
                genes[ix]+=rng.gauss(0.,settings.external_mutation_sd)
            evaluate(
                tuple(genes),gen,parent["candidate_id"],indices
            )
    finalists=sorted(records,key=_ranking)[:settings.keep_elites]
    selection=[]
    for development_rank,member in enumerate(finalists):
        rows=[]
        for seed in range(
            settings.validation_start,
            settings.validation_start+settings.validation_seeds
        ):
            row,_,_=simulate_neural(
                seed,settings,tuple(member["gene_values"])
            )
            row["stage"]="validation"
            row["candidate_id"]=member["candidate_id"]
            row["development_rank"]=development_rank
            selection.append(row)
            rows.append(row)
        member["development_rank"]=development_rank
        member["validation_score"]=list(_score(rows))
    selected=min(
        finalists,
        key=lambda c:(
            *(-x for x in c["validation_score"]),
            c["development_rank"],c["candidate_id"]
        )
    )
    if len(records)!=(
        settings.initial_pool+
        settings.generations*settings.offspring_per_generation
    ):
        raise AssertionError("External search budget deviated")
    if selected not in finalists:
        raise AssertionError("External selection not drawn from predefined top")
    return records,development,selection,selected,{
        "external_optimizer":EXTERNAL_OPTIMIZER,
        "search_seed":settings.search_seed,
        "development_seeds":list(range(
            settings.development_start,
            settings.development_start+settings.development_seeds
        )),
        "validation_seeds":list(range(
            settings.validation_start,
            settings.validation_start+settings.validation_seeds
        )),
        "n_development_candidates":len(records),
        "n_validation_candidates":len(finalists),
        "selected_candidate_id":selected["candidate_id"],
        "selected_genes":selected["gene_values"],
        "selected_genome_digest":selected["expanded_genome_digest"],
        "selected_development_score":selected["development_score"],
        "selected_validation_score":selected["validation_score"],
        "all_finalist_ids":[x["candidate_id"] for x in finalists],
        "not_within_world_evolution":True,
        "not_scripted_organism":True,
    }


def fixed_bank(selected: dict,settings:Settings) -> dict[str,tuple[float,...]]:
    original=tuple(selected["gene_values"])
    shuffled=list(original)
    random.Random(SCRAMBLE_SEED).shuffle(shuffled)
    if tuple(shuffled)==original:
        raise AssertionError("Numerical gene scramble had no effect")
    bank={"witness":original,"scrambled":tuple(shuffled)}
    for i in range(settings.random_controls):
        rng=random.Random(88600+i)
        bank[f"random_{i}"]=tuple(
            rng.gauss(0.,settings.initial_gene_sd)
            for _ in range(genome.N_GENES)
        )
    if tuple(bank)!=NEURAL_ARMS:
        raise AssertionError("Fixed control group changed")
    return bank


def paired_bootstrap(values: list[float],seed:int,reps:int) -> tuple[float,float]:
    if not values:
        raise ValueError("No held-out paired worlds")
    rng=random.Random(seed)
    n=len(values)
    draws=sorted(
        sum(values[rng.randrange(n)] for _ in range(n))/n
        for _ in range(reps)
    )
    return (
        draws[int(.025*(reps-1))],
        draws[int(.975*(reps-1))]
    )


def analyze(rows:list[dict],settings:Settings,
            *,boots:int=BOOT_REPS) -> dict:
    out={}
    for split,count in (("main",settings.seeds),
                        ("shifted",settings.shifted_seeds)):
        matched={arm:{
            row["seed"]:row
            for row in rows if row["split"]==split and row["arm"]==arm
        } for arm in ALL_ARMS}
        start=settings.seed_start if split=="main" else settings.shifted_start
        seeds=list(range(start,start+count))
        if any(sorted(matched[a])!=seeds for a in ALL_ARMS):
            raise AssertionError("One or more heldout worlds missing/duplicated")
        means={
            arm:{
                "surviving_worlds":sum(
                    matched[arm][seed]["survived"] for seed in seeds
                ),
                "worlds":count,
                "mean_food_eaten":statistics.mean(
                    matched[arm][seed]["food_eaten"] for seed in seeds
                ),
                "worlds_with_real_births":sum(
                    matched[arm][seed]["births"]>0 for seed in seeds
                ),
                "mean_births":statistics.mean(
                    matched[arm][seed]["births"] for seed in seeds
                ),
            } for arm in ALL_ARMS
        }
        random_deltas=[
            matched["witness"][seed]["survived"]-
            statistics.mean(
                matched[f"random_{i}"][seed]["survived"]
                for i in range(settings.random_controls)
            )
            for seed in seeds
        ]
        scramble_deltas=[
            matched["witness"][seed]["survived"]-
            matched["scrambled"][seed]["survived"]
            for seed in seeds
        ]
        ran=paired_bootstrap(
            random_deltas,
            BOOT_MAIN_SEED if split=="main" else BOOT_SHIFTED_SEED,
            boots
        )
        scramble=paired_bootstrap(
            scramble_deltas,
            BOOT_MAIN_SEED+11 if split=="main" else BOOT_SHIFTED_SEED+11,
            boots
        )
        out[split]={
            "by_arm":means,
            "witness_minus_random4_mean_survival":{
                "mean_delta":statistics.mean(random_deltas),
                "paired_world_bootstrap_95":list(ran),
                "by_world":{
                    str(s):v for s,v in zip(seeds,random_deltas)
                }
            },
            "witness_minus_scrambled_survival":{
                "mean_delta":statistics.mean(scramble_deltas),
                "paired_world_bootstrap_95":list(scramble),
                "by_world":{
                    str(s):v for s,v in zip(seeds,scramble_deltas)
                }
            }
        }
    main=out["main"];hard=out["shifted"]
    mainsurv=main["by_arm"]["witness"]["surviving_worlds"]
    mainbirths=main["by_arm"]["witness"]["worlds_with_real_births"]
    script_surv=main["by_arm"][SCRIPT_ARM]["surviving_worlds"]
    maincontrast=main["witness_minus_random4_mean_survival"]
    hardcontrast=hard["witness_minus_random4_mean_survival"]
    gates={
        "main_witness_survives_ge12of24":mainsurv>=12,
        "main_witness_native_births_ge12of24":mainbirths>=12,
        "scripted_ecological_viability_ge18of24":script_surv>=18,
        "main_witness_random_controls_gain_ge25pp":bool(
            maincontrast["mean_delta"]>=.25 and
            maincontrast["paired_world_bootstrap_95"][0]>0
        ),
        "shifted_witness_survives_ge6of12":(
            hard["by_arm"]["witness"]["surviving_worlds"]>=6
        ),
        "shifted_witness_random_controls_gain":bool(
            hardcontrast["mean_delta"]>0 and
            hardcontrast["paired_world_bootstrap_95"][0]>0
        ),
    }
    gates["numeric_architecture_short_odor_representation"]=all([
        gates["main_witness_survives_ge12of24"],
        gates["main_witness_native_births_ge12of24"],
        gates["scripted_ecological_viability_ge18of24"],
        gates["main_witness_random_controls_gain_ge25pp"]
    ])
    gates["short_odor_ood_transfer"]=all([
        gates["shifted_witness_survives_ge6of12"],
        gates["shifted_witness_random_controls_gain"]
    ])
    return {"main":main,"shifted":hard,"gates":gates,
            "caveat":(
                "Fixed-budget EXTERNAL researcher-led numerical genotype "
                "optimization, no within-world mutations or evolution. "
                "Four fixed random control genomes do not estimate the "
                "entire random genotype distribution. Short-odor worlds "
                "use the original M5.0 map generator, NOT M5.4A matched "
                "common-food-support maps."
            )}


def run(settings:Settings,out:Path,*,boots:int=BOOT_REPS) -> dict:
    settings.validate()
    out.mkdir(parents=True,exist_ok=True)
    records,development,validation,selected,selection=external_search(settings,trace=True)
    bank=fixed_bank(selected,settings)
    (out/"frozen_selected_genotype_and_controls.json").write_text(
        json.dumps({
            "selected":selection,
            "arms":{
                arm:{
                    "genes":list(genes),
                    "gene_digest":genome.gene_digest(genes),
                    "expanded_genome_digest":physics.genome_digest(
                        genome.to_weights(genes)
                    )
                }
                for arm,genes in bank.items()
            },
            "positive_benchmark":(
                "human-written M5.1 greedy_novelty, NOT a neural genotype"
            ),
            "protocol_frozen_commit":
                "fa45399a24cb52f0f8d95436cad87a273e991a54"
        },indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    # Every candidate and every original developmental world is archived,
    # preserving unsuccessful numerical candidates and non-survival outcomes.
    _write_csv(out/"external_search_development_metrics.csv",development)
    _write_csv(out/"external_search_validation_metrics.csv",validation)
    (out/"external_search_all_96_gene_records.json").write_text(
        json.dumps(records,sort_keys=True,indent=2)+"\n",encoding="utf-8"
    )
    print("M5.4B0 GENOTYPE FROZEN PRIOR TO SEALED WORLDS "+
          json.dumps(selection,sort_keys=True),flush=True)
    heldouts,checks,traces=[],[],[]
    for split,start,count,shift in (
        ("main",settings.seed_start,settings.seeds,False),
        ("shifted",settings.shifted_start,settings.shifted_seeds,True)
    ):
        for seed in range(start,start+count):
            pair=[]
            for arm in ALL_ARMS:
                if arm==SCRIPT_ARM:
                    row,ck,trace=scripted_ecological_control(
                        seed,settings,shifted=shift,
                        detailed=(seed==start)
                    )
                else:
                    row,ck,trace=simulate_neural(
                        seed,settings,bank[arm],shifted=shift,
                        detailed=(seed==start)
                    )
                row["arm"]=arm
                row["split"]=split
                heldouts.append(row)
                pair.append(row)
                for ch in ck:
                    ch["arm"]=arm
                    ch["split"]=split
                checks.extend(ck)
                if trace is not None:
                    trace["arm"]=arm
                    trace["split"]=split
                    traces.append(trace)
                print("M5.4B0 SEALED "+ " ".join((
                    split,str(seed),arm,
                    str(row["survived"]),str(row["alive_final"]),
                    str(row["births"]),str(row["food_eaten"])
                )),flush=True)
            if len(pair)!=len(ALL_ARMS):
                raise AssertionError("All paired arms required")
    results=analyze(heldouts,settings,boots=boots)
    report={
        "protocol":"docs/M5_4B0_PROTOCOL.md",
        "original_preregistration_commit":
            "fa45399a24cb52f0f8d95436cad87a273e991a54",
        "settings":asdict(settings),
        "external_optimization_researcher_supervision":True,
        "within_world_mutations":0,
        "search_candidates":len(records),
        "development_trial_count":len(development),
        "validation_trial_count":len(validation),
        "sealed_trial_count":len(heldouts),
        "selected_candidate":selection,
        **results
    }
    _write_csv(out/"heldout_world_metrics.csv",heldouts)
    _write_csv(out/"heldout_population_checkpoints.csv",checks)
    (out/"summary.json").write_text(
        json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    for trace in traces:
        (out/("events_%s_%s_%s.json" %(
            trace["split"],trace["seed"],trace["arm"]
        ))).write_text(
            json.dumps(trace,indent=2,sort_keys=True)+"\n",encoding="utf-8"
        )
    print("M5.4B0 FULL RESULT "+json.dumps({
        "chosen_genes":selection,
        "main":report["main"],
        "shifted":report["shifted"],
        "gates":report["gates"]
    },sort_keys=True),flush=True)
    return report


def _write_csv(path:Path,rows:list[dict]) -> None:
    if not rows:
        raise ValueError("Empty scientific data output")
    # Scripted M5.1 checkpoints include explicit local policy diagnostics.
    # Retain their extra columns instead of silently dropping audit evidence.
    keys=list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    for f in fields(Settings):
        parser.add_argument("--"+f.name.replace("_","-"),
                            default=f.default,type=type(f.default))
    parser.add_argument("--out",type=Path,default=Path("results/m5-4b0"))
    parser.add_argument("--bootstrap-reps",type=int,default=BOOT_REPS)
    d=vars(parser.parse_args())
    path=d.pop("out")
    reps=d.pop("bootstrap_reps")
    if reps<100:
        raise ValueError("At least 100 descriptive bootstrap replicates")
    if d["ticks"]>=400 and reps!=BOOT_REPS:
        raise ValueError("Full campaign must use preregistered 20,000 bootstraps")
    run(Settings(**d),path,boots=reps)


if __name__=="__main__":
    main()
