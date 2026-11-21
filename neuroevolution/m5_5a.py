"""M5.5A — human-designed innate drives with eight truly heritable traits.

Distinct from neural M5.0–M5.4C and from external genotype optimization.
Here the programmer supplies innate legal-movement, food/odor, hunger,
novelty, inertia, stochastic escape and energy-gated reproduction primitives.
The 8 real numbers vary their salience and CAN mutate on genuine original
energy-costed physical births. No original 401-weight neural model is acting.
Prospective frozen protocol SHA a3aa39e4f68acc38d17a4292b4ebefbd4b72b10b.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from . import m5_0 as base
from . import m5_1 as old_script
from . import m5_4b0 as stats_legacy

PROTOCOL_SHA = "a3aa39e4f68acc38d17a4292b4ebefbd4b72b10b"
N_GENES = 8
ARMS = (
    "innate_clone",
    "innate_sparse_mutation",
    "innate_dense_mutation",
    "innate_random_newborn",
    "random_legal_movement",
)
CLONE,SPARSE,DENSE,RANDOM_CHILD,RANDOM_MOTOR = ARMS
CONTRASTS = (
    ("sparse_minus_clone",SPARSE,CLONE),
    ("dense_minus_clone",DENSE,CLONE),
    ("random_newborn_minus_clone",RANDOM_CHILD,CLONE),
    ("sparse_minus_random_newborn",SPARSE,RANDOM_CHILD),
    ("sparse_minus_dense",SPARSE,DENSE),
    ("clone_minus_random_motor",CLONE,RANDOM_MOTOR),
)
BOOT_REPS=20000


@dataclass(frozen=True)
class Settings(base.Settings):
    seed_start:int=93000
    seeds:int=24
    shifted_start:int=93100
    shifted_seeds:int=12
    odor_radius:int=2
    shifted_odor_radius:int=2
    no_food_controls:int=0
    innate_initial_gene_sd:float=.65
    innate_mutation_sd:float=.22
    innate_gene_limit:float=2.5

    def validate(self)->None:
        super().validate()
        if not(
            self.odor_radius==2 and self.shifted_odor_radius==2 and
            self.innate_initial_gene_sd==.65 and
            self.innate_mutation_sd==.22 and
            self.innate_gene_limit==2.5
        ):
            raise ValueError("Original preregistered innate trait scales")
        left=set(range(self.seed_start,self.seed_start+self.seeds))
        right=set(range(self.shifted_start,
                        self.shifted_start+self.shifted_seeds))
        historic=set()
        for start,n in (
            (79000,6),(79100,2),(80000,24),(81000,12),
            (82000,24),(83000,12),(84000,24),(85000,12),
            (86000,16),(87000,8),(88000,4),(88100,8),
            (88200,24),(88300,12),(89000,4),(89100,8),
            (89200,24),(89300,12),(91000,24),(91100,12),
            (92000,24),(92100,12),
        ):
            historic.update(range(start,start+n))
        if left&right or left&historic or right&historic:
            raise ValueError("Original/new ecology worlds must be independent")
        if self.ticks>=400:
            canonical=Settings()
            for f in fields(Settings):
                if getattr(self,f.name)!=getattr(canonical,f.name):
                    raise ValueError(
                        "Full preregistered M5.5A study cannot change "+f.name
                    )


def check_genes(genes:tuple[float,...]) -> None:
    if len(genes)!=N_GENES or any(not math.isfinite(x) for x in genes):
        raise ValueError("Exactly eight finite real inherited temperament genes")


def founder_genes(seed:int,settings:Settings
                  )->tuple[tuple[float,...],...]:
    rng=random.Random(seed*2777+5501)
    bank=tuple(tuple(
        rng.gauss(0.,settings.innate_initial_gene_sd)
        for _ in range(N_GENES)
    ) for _ in range(settings.initial_population))
    for genes in bank:
        check_genes(genes)
    return bank


def gene_sha(genes:tuple[float,...])->str:
    check_genes(genes)
    return hashlib.sha256(",".join(
        format(x,".17g") for x in genes
    ).encode("ascii")).hexdigest()[:24]


def instinct_parameters(genes:tuple[float,...],energy:float,
                        settings:Settings)->dict[str,float]:
    check_genes(genes)
    h=max(0.,min(1.,1.-energy/settings.max_energy))
    return {
        "hunger":h,
        "gain":1.+.4*math.tanh(genes[5])*h,
        "adjacent_food_weight":3.2+1.2*math.tanh(genes[0]),
        "odor_weight":.8+.6*math.tanh(genes[1]),
        "novelty_weight":.20+.20*math.tanh(genes[2]),
        "persistence_weight":.12*math.tanh(genes[3]),
        "random_motor_amplitude":.50/(1.+math.exp(-genes[4])),
        "birth_energy_threshold":46.+6.*math.tanh(genes[6]),
        "birth_probability":.65+.25*math.tanh(genes[7]),
    }


def choose_action(eco:base.Ecology,agent:base.Agent,tick:int,
                  visits:dict[base.Point,int],
                  last_heading:int|None,arm:str
                  )->tuple[int|None,dict]:
    """Only local native 15 sensors plus explicitly programmed personal state.

    Original neural logits are NOT consulted. Never invent unseen threats,
    global food positions or an optimal spatial route.
    """
    if arm not in ARMS:
        raise ValueError(arm)
    blocked,free=base.blocked_and_free(eco,agent)
    sensors=base.observe(eco,agent,blocked)
    legal=[d for d,b in enumerate(blocked) if not b]
    physically_eligible=base.can_reproduce(eco,agent,free)
    g=instinct_parameters(agent.genome,agent.energy,eco.settings)
    proposed_birth=bool(
        physically_eligible and agent.energy>=g["birth_energy_threshold"]
        and base._local_rng(
            eco.seed,tick,agent.id,84
        ).random()<g["birth_probability"]
    )
    obs={
        "hunger":g["hunger"],
        "food_sensed":int(any(sensors[8+d]>0 for d in legal)),
        "odor_sensed":int(any(sensors[4+d]>0 for d in legal)),
        "birth_eligible":int(physically_eligible),
        "birth_requested":int(proposed_birth),
        "used_programmed_instinct":int(arm!=RANDOM_MOTOR),
        "random_legal_motor_control":int(arm==RANDOM_MOTOR),
    }
    if proposed_birth:
        return base.REPRODUCE,obs
    if not legal:
        return None,obs
    if arm==RANDOM_MOTOR:
        return base._local_rng(
            eco.seed,tick,agent.id,37
        ).choice(legal),obs
    rng=base._local_rng(eco.seed,tick,agent.id,83)
    noise=[rng.uniform(-1.,1.) for _ in range(4)]
    x,y=agent.position
    scores={}
    for d in legal:
        dx,dy=base.DIRECTIONS[d]
        target=(x+dx,y+dy)
        scores[d]=(
            g["gain"]*(
                g["adjacent_food_weight"]*sensors[8+d]+
                g["odor_weight"]*sensors[4+d]
            )+
            g["novelty_weight"]/(1.+visits.get(target,0))+
            g["persistence_weight"]*int(d==last_heading)+
            g["random_motor_amplitude"]*noise[d]
        )
    top=max(scores.values())
    tied=[d for d in legal if abs(scores[d]-top)<1e-12]
    return base._local_rng(
        eco.seed,tick,agent.id,37
    ).choice(tied),obs


def child_genes(parent:tuple[float,...],seed:int,tick:int,aid:int,
                arm:str,settings:Settings
                )->tuple[tuple[float,...],list[int]]:
    check_genes(parent)
    if arm in (CLONE,RANDOM_MOTOR):
        return parent,[]
    rng=base._local_rng(seed,tick,aid,85)
    if arm==RANDOM_CHILD:
        new=tuple(
            rng.gauss(0.,settings.innate_initial_gene_sd)
            for _ in range(N_GENES)
        )
        check_genes(new)
        return new,list(range(N_GENES))
    if arm==SPARSE:
        indices=sorted(rng.sample(range(N_GENES),2))
    elif arm==DENSE:
        indices=list(range(N_GENES))
    else:
        raise ValueError("Unknown in-world genetic mutation arm")
    genes=list(parent)
    for i in indices:
        genes[i]=max(-settings.innate_gene_limit,
                     min(settings.innate_gene_limit,
                         genes[i]+rng.gauss(0.,settings.innate_mutation_sd)))
    result=tuple(genes)
    check_genes(result)
    return result,indices


def _geometry_sha(eco:base.Ecology,schedule,founder_positions)->str:
    obj={
        "seed":eco.seed,"shifted":eco.shifted,
        "walls":sorted(eco.world.walls),
        "initial_food":sorted(eco.world.initial_food),
        "founder_positions":founder_positions,
        "reachable_cells":eco.accessible,
        "full_initial_food_rng_schedule":schedule,
    }
    return hashlib.sha256(json.dumps(
        obj,sort_keys=True,separators=(",",":")
    ).encode("utf-8")).hexdigest()


def simulate(seed:int,settings:Settings,arm:str, *,
             shifted:bool=False,detailed:bool=False
             )->tuple[dict,list[dict],dict|None]:
    settings.validate()
    if arm not in ARMS:
        raise ValueError(arm)
    original=base.create_ecology(
        seed,settings,"heritable_no_mutation",shifted=shifted
    )
    source=founder_genes(seed,settings)
    original_positions=tuple(
        a.position for a in sorted(original.agents.values(),key=lambda a:a.id)
    )
    for agent,g in zip(
        sorted(original.agents.values(),key=lambda a:a.id),source
    ):
        agent.genome=g
        agent.hidden=[0.]*settings.hidden
    original.initial_genome_digests=tuple(
        base.genome_digest(g) for g in source
    )
    schedule=base._food_schedule(seed,settings,original,False)
    geometry_sha=_geometry_sha(original,schedule,original_positions)
    visit={a.id:{a.position:1} for a in original.agents.values()}
    heading={a.id:None for a in original.agents.values()}
    initial_traits_mean=[
        statistics.mean(g[i] for g in source) for i in range(N_GENES)
    ]
    moves=0
    repeats=0
    distinct={a.position for a in original.agents.values()}
    mutations=0
    actual_mutated_births=0
    random_child_births=0
    birth_gene_records=[]
    checkpoints=[]
    for tick in range(settings.ticks+1):
        if tick:
            agent_ids=list(original.agents)
            base._local_rng(seed,tick,0,0).shuffle(agent_ids)
            for aid in agent_ids:
                a=original.agents.get(aid)
                if a is None:
                    continue
                before_pos=a.position
                before_genes=a.genome
                before_move=original.move_actions
                before_birth=original.births
                child_id=original.next_id
                chosen,obs=choose_action(
                    original,a,tick,visit[aid],heading[aid],arm
                )
                old_script._act_scripted(original,aid,tick,chosen)
                newborn=int(original.births!=before_birth)
                moved=int(original.move_actions!=before_move)
                if newborn!=int(chosen==base.REPRODUCE):
                    raise AssertionError("An instinct did not select a real birth")
                if moved!=int(chosen in range(4)):
                    raise AssertionError("An instinct did not select a real legal move")
                if newborn:
                    child=original.agents[child_id]
                    if child.genome!=before_genes:
                        raise AssertionError("Original native birth was not a clone")
                    genes,changed=child_genes(
                        before_genes,seed,tick,aid,arm,settings
                    )
                    child.genome=genes
                    original.cosine_births[-1]=base.genome_cosine(
                        before_genes,genes
                    )
                    event=original.events[-1]
                    if event["type"]!="birth" or event["child"]!=child_id:
                        raise AssertionError("Native physical genealogical event lost")
                    event["genome"]=base.genome_digest(genes)
                    event["gene_sha24"]=gene_sha(genes)
                    event["parent_gene_sha24"]=gene_sha(before_genes)
                    event["changed_gene_indices"]=changed
                    event["genetic_control_arm"]=arm
                    mutations+=len(changed) if arm in (SPARSE,DENSE) else 0
                    actual_mutated_births+=int(
                        genes!=before_genes and arm in (SPARSE,DENSE)
                    )
                    random_child_births+=int(
                        genes!=before_genes and arm==RANDOM_CHILD
                    )
                    birth_gene_records.append({
                        "seed":seed,"shifted":shifted,"arm":arm,
                        "tick":tick,"parent":aid,"child":child_id,
                        "founder":child.founder,
                        "generation":child.generation,
                        "changed_indices":changed,
                        "parent_genes":list(before_genes),
                        "newborn_genes":list(genes),
                        "parent_gene_digest":gene_sha(before_genes),
                        "child_gene_digest":gene_sha(genes),
                    })
                    visit[child_id]={child.position:1}
                    heading[child_id]=None
                if moved:
                    moves+=1
                    dx,dy=base.DIRECTIONS[chosen]
                    newpos=(before_pos[0]+dx,before_pos[1]+dy)
                    if aid in original.agents:
                        if original.agents[aid].position!=newpos:
                            raise AssertionError("Movement did not reach legal target")
                    repeats+=int(newpos in visit[aid])
                    visit[aid][newpos]=visit[aid].get(newpos,0)+1
                    heading[aid]=chosen
                    distinct.add(newpos)
            base._food_regeneration(original,schedule[tick-1])
            original.ticks_elapsed=tick
            if not original.agents and original.extinct_at is None:
                original.extinct_at=tick
        if tick==0 or tick%settings.checkpoint_every==0 or tick==settings.ticks:
            pop=list(original.agents.values())
            ck=base.snapshot(original,tick)
            ck.update({
                "seed":seed,
                "split":"shifted" if shifted else "main",
                "arm":arm,
                "world_sha256":geometry_sha,
                "initial_founder_gene_bank_sha":hashlib.sha256(
                    json.dumps(source,separators=(",",":")).encode()
                ).hexdigest(),
                "distinct_visited_cells":len(distinct),
                "individual_revisits":repeats,
                "individual_movement_steps":moves,
                "genuinely_mutated_births":actual_mutated_births,
                "random_genotype_newborns":random_child_births,
                "mean_trait_vector":[
                    statistics.mean(a.genome[i] for a in pop)
                    if pop else None
                    for i in range(N_GENES)
                ],
                "source_naive_founders_count":settings.initial_population,
            })
            if abs(base.energy_balance(original))>1e-6:
                raise AssertionError("Original original-physics energy leaked")
            checkpoints.append(ck)

    if moves!=original.move_actions or len(visit)!=original.next_id:
        raise AssertionError("Loss of action/genetic newborn records")
    if len(birth_gene_records)!=original.births:
        raise AssertionError("Some original physically costed births have no ancestry")
    if arm in (CLONE,RANDOM_MOTOR) and actual_mutated_births:
        raise AssertionError("Exact inherited clone control had hidden mutation")
    if arm==SPARSE and mutations!=2*original.births:
        raise AssertionError("Not exactly two inherited mutation coordinates")
    if arm==DENSE and mutations!=N_GENES*original.births:
        raise AssertionError("Not exactly all eight inherited mutant coordinates")
    if arm==RANDOM_CHILD and random_child_births!=original.births:
        raise AssertionError("Independent newborn genotype did not change")
    snapshot=base.snapshot(original,settings.ticks)
    survivors=list(original.agents.values())
    surviving_nonfounder_alleles=sum(
        a.genome!=source[a.founder] for a in survivors
    )
    row={
        "seed":seed,
        "split":"shifted" if shifted else "main",
        "arm":arm,
        "world_sha256":geometry_sha,
        "genetic_policies_encoded_as_innate_scaffold_not_neural":1,
        "native_neural_401_weight_processor_used":0,
        "researcher_programmed_instinct_primitives":1,
        "true_within_world_genetic_birth_mutation":int(arm in (SPARSE,DENSE)),
        "random_genotype_newborn_control":int(arm==RANDOM_CHILD),
        "initial_20_naive_founders":settings.initial_population,
        "initial_gene_bank_sha":ck["initial_founder_gene_bank_sha"],
        "survived":int(bool(survivors)),
        "alive_final":len(survivors),
        "births":original.births,
        "deaths":original.deaths,
        "food_eaten":original.food_eaten,
        "max_generation":original.max_generation,
        "extinct_at":original.extinct_at,
        "source_founder_lineages_surviving":
            len({a.founder for a in survivors}),
        "nonfounder_allele_survivors":surviving_nonfounder_alleles,
        "fraction_survivors_nonfounder_alleles":(
            surviving_nonfounder_alleles/len(survivors)
            if survivors else 0.
        ),
        "genetically_mutated_births":actual_mutated_births,
        "mutated_gene_coordinates_total":mutations,
        "independent_random_gene_newborns":random_child_births,
        "mutated_descendant_ever_created":int(actual_mutated_births>0),
        "repeated_cell_move_fraction":repeats/moves if moves else 0.,
        "distinct_cells_visited_by_any_agent":len(distinct),
        "mean_unique_cells_per_organism_life":statistics.mean(
            len(path) for path in visit.values()
        ) if visit else 0.,
        "food_per_100_moves":(
            100.*original.food_eaten/moves if moves else 0.
        ),
        "population_auc_mean_checkpoint":statistics.mean(
            x["population"] for x in checkpoints
        ),
        "mean_organism_generation_alive":statistics.mean(
            a.generation for a in survivors
        ) if survivors else 0.,
        "energy_ledger_residual":base.energy_balance(original),
        "natively_costed_birth_actions":original.birth_actions,
        "natively_costed_move_actions":original.move_actions,
    }
    if original.birth_actions!=original.births:
        raise AssertionError("Original birth bookkeeping incomplete")
    trace=None
    if detailed:
        trace={
            "seed":seed,"shifted":shifted,"arm":arm,
            "world_sha256":geometry_sha,
            "initial_20_founder_genes":[list(x) for x in source],
            "initial_positions":[list(x) for x in original_positions],
            "initial_food":[list(x) for x in original.world.initial_food],
            "walls":[list(x) for x in sorted(original.world.walls)],
            "checks":checkpoints,
            "all_physically_costed_birth_genealogy":birth_gene_records,
            "all_original_physical_food_birth_events":original.events,
            "survivors":[{
                "founder":a.founder,"id":a.id,"parent":a.parent,
                "generation":a.generation,"genes":list(a.genome),
                "gene_digest":gene_sha(a.genome),
                "energy":a.energy,
            } for a in survivors],
        }
    return row,checkpoints,trace


def analyze(rows:list[dict],checkpoints:list[dict],
            settings:Settings,bootstrap_reps:int=BOOT_REPS)->dict:
    out={}
    for split,start,n,seed_bs in (
        ("main",settings.seed_start,settings.seeds,93501),
        ("shifted",settings.shifted_start,settings.shifted_seeds,93502),
    ):
        seeds=list(range(start,start+n))
        selected=[r for r in rows if r["split"]==split]
        if len(selected)!=n*len(ARMS):
            raise AssertionError("Missing original preregistered full trial arms")
        by={arm:{
            r["seed"]:r for r in selected if r["arm"]==arm
        } for arm in ARMS}
        if any(sorted(by[arm])!=seeds for arm in ARMS):
            raise AssertionError("Duplicate/missing paired original source seed")
        for seed in seeds:
            if len({by[a][seed]["world_sha256"] for a in ARMS})!=1:
                raise AssertionError("Original physical world not matched")
            if len({by[a][seed]["initial_gene_bank_sha"] for a in ARMS})!=1:
                raise AssertionError("Naive 20 founder genotype banks differ")
        totals={
            arm:{
                "survivor_worlds":sum(by[arm][seed]["survived"] for seed in seeds),
                "worlds":n,
                "mean_alive_final":statistics.mean(
                    by[arm][seed]["alive_final"] for seed in seeds),
                "mean_food_eaten":statistics.mean(
                    by[arm][seed]["food_eaten"] for seed in seeds),
                "mean_native_births":statistics.mean(
                    by[arm][seed]["births"] for seed in seeds),
                "mean_population_auc":statistics.mean(
                    by[arm][seed]["population_auc_mean_checkpoint"]
                    for seed in seeds),
                "worlds_with_genuinely_mutated_offspring":sum(
                    by[arm][seed]["mutated_descendant_ever_created"]
                    for seed in seeds),
                "mean_fraction_final_nonfounder_alleles":statistics.mean(
                    by[arm][seed]["fraction_survivors_nonfounder_alleles"]
                    for seed in seeds),
                "mean_revisited_move_fraction":statistics.mean(
                    by[arm][seed]["repeated_cell_move_fraction"]
                    for seed in seeds),
                "mean_unique_cells_per_organism_life":statistics.mean(
                    by[arm][seed]["mean_unique_cells_per_organism_life"]
                    for seed in seeds),
            } for arm in ARMS
        }
        comparisons={}
        for index,(name,one,two) in enumerate(CONTRASTS):
            for measure,offset in (
                ("survived",0),("population_auc_mean_checkpoint",10000)
            ):
                delta=[
                    by[one][seed][measure]-by[two][seed][measure]
                    for seed in seeds
                ]
                lo,hi=stats_legacy.paired_bootstrap(
                    delta,seed_bs+1000*index+offset,bootstrap_reps
                )
                comparisons[name+"_"+("survival" if offset==0 else "auc")]={
                    "mean_paired_effect":statistics.mean(delta),
                    "bootstrap_95":[lo,hi],
                    "per_world":dict(zip(map(str,seeds),delta)),
                    "strictly_positive_lower95":bool(lo>0),
                }
        out[split]={
            "arms":totals,"comparisons":comparisons,
            "independent_matched_worlds":n
        }
    g=out["main"]
    more=g["comparisons"]
    gates={
        "programmed_innate_clone_positive_ecology_ge18of24":
            g["arms"][CLONE]["survivor_worlds"]>=18,
        "observed_sparse_mutation_worlds_ge12of24":
            g["arms"][SPARSE]["worlds_with_genuinely_mutated_offspring"]>=12,
        "sparse_vs_clones_both_survival_and_auc_lower95_positive":bool(
            more["sparse_minus_clone_survival"]["strictly_positive_lower95"]
            and more["sparse_minus_clone_auc"]["strictly_positive_lower95"]
        ),
        "source_founders_naive_and_independent":True,
        "true_within_world_genetic_perturbations_implemented":True,
        "causal_beneficial_mutation_over_own_founder_transfer_proven":False,
        "spontaneously_learned_neural_instincts":False,
    }
    gates["observational_mutation_advantage_gate"]=bool(
        gates["observed_sparse_mutation_worlds_ge12of24"] and
        gates["sparse_vs_clones_both_survival_and_auc_lower95_positive"]
    )
    return {
        **out,"gates":gates,
        "critical_inference_limits":(
            "Human-programmed innate foraging/hunger/reflex/novelty/"
            "reproduction, variable inherited numeric GAIN/THRESHOLD traits. "
            "Original numerical M5.0 RNN policy is not used by this new "
            "architecture. True birth mutations can occur, but a mutation "
            "advantage over each lineage's OWN previously random founder "
            "has NOT been tested by an independent transfer test, and "
            "selection adaptation must not be inferred from survival "
            "alone. Paired world-level bootstrap does not resample "
            "independently chosen genetic architectures."
        ),
    }


def write_csv(path:Path,rows:list[dict])->None:
    keys=list(dict.fromkeys(k for row in rows for k in row.keys()))
    with path.open("w",encoding="utf-8",newline="") as out:
        writer=csv.DictWriter(out,fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def run(settings:Settings,out:Path, *, bootstrap_reps:int=BOOT_REPS
        )->dict:
    settings.validate()
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    checks=[]
    trajectories=[]
    founder_banks=[]
    for split,start,count,hard in (
        ("main",settings.seed_start,settings.seeds,False),
        ("shifted",settings.shifted_start,settings.shifted_seeds,True),
    ):
        for seed in range(start,start+count):
            # First arm gives the original exact physical source world,
            # naive random source founder genetics and future food RNG hash.
            for arm in ARMS:
                row,ck,trace=simulate(
                    seed,settings,arm,shifted=hard,detailed=True
                )
                rows.append(row)
                checks.extend(ck)
                trajectories.append(trace)
                if arm==CLONE:
                    founder_banks.append({
                        "seed":seed,"split":split,
                        "initial_naive_20_genes":
                            trace["initial_20_founder_genes"],
                        "initial_founder_positions":
                            trace["initial_positions"],
                        "original_world_sha256":trace["world_sha256"],
                    })
                print("M5.5A SEALED "+json.dumps(
                    row,sort_keys=True),flush=True)
                for checkpoint in ck:
                    if settings.ticks>=400:
                        print("M5.5A CHECKPOINT "+json.dumps(
                            checkpoint,sort_keys=True),flush=True)
    stats=analyze(rows,checks,settings,bootstrap_reps)
    result={
        "preregistration_sha":PROTOCOL_SHA,
        "worlds":len(founder_banks),
        "original_full_world_policy_trials":len(rows),
        "original_population_checkpoints":len(checks),
        "settings":asdict(settings),
        **stats,
    }
    if settings.ticks>=400 and (
        result["worlds"]!=36 or len(rows)!=180 or
        len(checks)!=1620
    ):
        raise AssertionError("Full preregistered world budget altered")
    write_csv(out/"all_180_innate_genetic_worlds.csv",rows)
    # Checkpoints contain 8 element gene-vector means: JSON rather than
    # invalid Python object repr in CSV for exact replay.
    for ck in checks:
        ck["mean_trait_vector"]=json.dumps(
            ck["mean_trait_vector"],separators=(",",":")
        )
    write_csv(out/"all_1620_native_population_checkpoints.csv",checks)
    (out/"all_36_exact_naive_source_founder_genotype_banks.json").write_text(
        json.dumps(founder_banks,sort_keys=True,indent=2)+"\n",
        encoding="utf-8"
    )
    (out/"all_true_birth_genotype_genealogies.json").write_text(
        json.dumps([
            {
                "seed":x["seed"],"arm":x["arm"],
                "split":"shifted" if x["shifted"] else "main",
                "world_sha256":x["world_sha256"],
                "all_birth_genealogy":x["all_physically_costed_birth_genealogy"]
            } for x in trajectories
        ],sort_keys=True)+"\n",encoding="utf-8"
    )
    (out/"all_source_detailed_first_world_event_ledgers.json").write_text(
        json.dumps([
            x for x in trajectories if x["seed"] in (
                settings.seed_start,settings.shifted_start
            )
        ],sort_keys=True)+"\n",encoding="utf-8"
    )
    (out/"summary.json").write_text(
        json.dumps(result,sort_keys=True,indent=2)+"\n",encoding="utf-8"
    )
    print("M5.5A FULL RESULT "+json.dumps({
        "worlds":result["worlds"],
        "trials":result["original_full_world_policy_trials"],
        "main":result["main"],"shifted":result["shifted"],
        "gates":result["gates"],
    },sort_keys=True),flush=True)
    return result


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__)
    for f in fields(Settings):
        parser.add_argument(
            "--"+f.name.replace("_","-"),
            default=f.default,type=type(f.default)
        )
    parser.add_argument("--out",type=Path,default=Path("results/m5-5a"))
    parser.add_argument("--bootstrap-reps",type=int,default=BOOT_REPS)
    args=vars(parser.parse_args())
    out=args.pop("out")
    bs=args.pop("bootstrap_reps")
    if bs<100:
        raise ValueError("At least 100 paired independent world bootstraps")
    if args["ticks"]>=400 and bs!=BOOT_REPS:
        raise ValueError("Frozen full research requires 20k paired bootstraps")
    run(Settings(**args),out,bootstrap_reps=bs)


if __name__=="__main__":
    main()
