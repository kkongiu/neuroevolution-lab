"""M5.4A: paired 2^3 environmental factorial on fixed numerical neural genotypes.

This is a diagnostic of ecological vulnerabilities, NOT evolution.
All ten neural genotypes come from the frozen M5.3C origin archive.
Human-scripted M5.1 greedy+novelty is a separate positive benchmark.
Original M5.0 native physics/network action and M5.1 scripted physics stay
untouched; the new wrapper samples nested paired maps/resources to remove
unwanted initial food/founder/renewal map confounds.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

from . import m4_0 as grid
from . import m5_0 as base
from . import m5_1 as script
from . import m5_3b as gene
from . import m5_3c as source

FAMILIES = source.SOURCES
NEURAL_GENOTYPES = ("descendant", "ancestor")
POLICY_KINDS = ("neural", "scripted_greedy_novelty")
AXES = ("walls", "odor", "regrowth")
AXIS_N = {"walls": 0, "odor": 1, "regrowth": 2}
BOOT_MAIN = 20000
BOOT_VALIDATION = 5000


@dataclass(frozen=True)
class Settings(script.Settings):
    seed_start: int = 86000
    seeds: int = 16
    shifted_start: int = 87000
    shifted_seeds: int = 8
    hidden: int = 12
    no_food_controls: int = 0
    negative_control_worlds: int = 1

    def validate(self) -> None:
        super().validate()
        historic = set(range(79000,79006)) | set(range(79100,79102))
        for start,n in (
            (80000,24),(81000,12),(82000,24),(83000,12),
            (84000,24),(85000,12)
        ):
            historic.update(range(start,start+n))
        main = set(range(self.seed_start,self.seed_start+self.seeds))
        validation = set(range(
            self.shifted_start,self.shifted_start+self.shifted_seeds
        ))
        if (main & validation or main & historic or validation & historic):
            raise ValueError("M5.4A world seeds overlap prior sealed worlds")
        if self.hidden != 12:
            raise ValueError("M5.4A original numeric neural genotypes require hidden=12")
        if self.ticks >= 400:
            fixed=Settings()
            frozen=(
                "seed_start","seeds","shifted_start","shifted_seeds",
                "ticks","size","initial_food","food_cap",
                "initial_population","max_population",
                "regeneration_attempts","initial_energy","max_energy",
                "food_energy","movement_cost","newborn_energy",
                "reproduction_energy_cost","min_reproduction_energy",
                "min_reproduction_age","max_age","checkpoint_every",
                "scripted_reproduction_threshold","hidden","no_food_controls",
                "obstacle_density","shifted_obstacle_density",
                "odor_radius","shifted_odor_radius",
                "regeneration_probability","shifted_regeneration_probability",
            )
            for name in frozen:
                if getattr(self,name) != getattr(fixed,name):
                    raise ValueError("Frozen M5.4A protocol changed: "+name)


def factorial_conditions() -> list[tuple[int,int,int]]:
    return [(w,o,r) for w in (0,1) for o in (0,1) for r in (0,1)]


def scenario_label(flags: tuple[int,int,int]) -> str:
    return "w%do%dr%d" % flags


def _digest(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj,separators=(",",":"),sort_keys=True).encode()
    ).hexdigest()[:20]


def coupled_world(seed: int, settings: Settings) -> dict:
    """Generate NESTED walls and identical food/founder positions.

    Common resource candidate pool is the HIGH density reachable component.
    All 8 conditions share this pool. Food never regrows exclusively in the
    low-wall-only cells; this deliberately identifies a matched-food
    conditional wall effect, not a total world-regeneration effect.
    """
    settings.validate()
    size=settings.size
    start=(size//2,size//2)
    protected={start}|{
        (start[0]+dx,start[1]+dy) for dx,dy in base.DIRECTIONS
    }
    border=frozenset(
        (x,y) for y in range(size) for x in range(size)
        if x in (0,size-1) or y in (0,size-1)
    )
    interior=[
        (x,y) for y in range(1,size-1) for x in range(1,size-1)
        if (x,y) not in protected
    ]
    chosen=None
    for attempt in range(100):
        rng=random.Random(seed*1009+421+attempt*10000019)
        uniform={cell:rng.random() for cell in interior}
        low=frozenset(set(border)|{
            pos for pos,draw in uniform.items()
            if draw < settings.obstacle_density
        })
        high=frozenset(set(border)|{
            pos for pos,draw in uniform.items()
            if draw < settings.shifted_obstacle_density
        })
        if not low <= high:
            raise AssertionError("Walls must be nested")
        high_map=grid.World(seed,size,start,high,(),settings.odor_radius,False)
        common=tuple(sorted(high_map.reachable()))
        if len(common) >= settings.initial_food+settings.initial_population:
            chosen=(low,high,common,attempt)
            break
    if chosen is None:
        raise RuntimeError("No coupled map with 20 food and 20 founder cells")
    low,high,common,attempt=chosen
    rng=random.Random(seed*7717+337)
    foods=tuple(sorted(rng.sample(common,settings.initial_food)))
    population_candidates=[pos for pos in common if pos not in foods]
    founders=tuple(rng.sample(population_candidates,settings.initial_population))
    if set(foods) & set(founders):
        raise AssertionError("Food and founders cannot overlap")
    lowmap=grid.World(seed,size,start,low,foods,settings.odor_radius,False)
    highmap=grid.World(seed,size,start,high,foods,settings.odor_radius,False)
    if not set(common) <= lowmap.reachable():
        raise AssertionError("High accessible cells not accessible in low density")
    sample={
        "seed":seed, "attempt":attempt,
        "world_by_wall_flag":(lowmap,highmap),
        "common_food_candidate_pool":common,
        "initial_food":foods,"initial_founders":founders,
        "low_wall_count":len(low),"high_wall_count":len(high),
        "low_reachable_count":len(lowmap.reachable()),
        "high_reachable_count":len(highmap.reachable()),
        "wall_hashes":(_digest(sorted(low)),_digest(sorted(high))),
        "initial_food_digest":_digest(foods),
        "founder_position_digest":_digest(founders),
        "food_candidate_pool_digest":_digest(common),
    }
    return sample


def common_food_schedule(seed: int, settings: Settings, paired: dict
                         ) -> tuple[tuple[tuple[float,tuple[int,int]],...],...]:
    rng=random.Random(seed*9191+499)
    pool=paired["common_food_candidate_pool"]
    return tuple(
        tuple((rng.random(),rng.choice(pool))
              for _ in range(settings.regeneration_attempts))
        for _ in range(settings.ticks)
    )


def new_ecology(seed: int, settings: Settings, paired: dict,
                flags: tuple[int,int,int], genes: tuple[float,...] | None
                ) -> base.Ecology:
    w,o,r=flags
    world=paired["world_by_wall_flag"][w]
    physical=replace(
        settings,
        odor_radius=(settings.shifted_odor_radius if o else settings.odor_radius),
        regeneration_probability=(
            settings.shifted_regeneration_probability if r
            else settings.regeneration_probability
        ),
    )
    if genes is not None:
        genome=gene.to_weights(genes)
    else:
        genome=tuple([0.] * base.genome_size(settings.hidden))
    agents={
        idx:base.Agent(
            id=idx,founder=idx,parent=None,generation=0,
            position=pos,energy=settings.initial_energy,
            genome=genome,hidden=[0.]*settings.hidden
        )
        for idx,pos in enumerate(paired["initial_founders"])
    }
    return base.Ecology(
        seed=seed,shifted=False,arm="heritable_no_mutation",
        world=world,settings=physical,agents=agents,
        food=set(paired["initial_food"]),
        accessible=tuple(sorted(world.reachable())),
        initial_genome_digests=tuple(
            base.genome_digest(genome) for _ in agents
        ),
        next_id=settings.initial_population,
        food_spawned=len(paired["initial_food"])
    )


def candidate_bank() -> tuple[dict[int,dict[str,tuple[float,...]]],list[dict]]:
    """Reconstruct exact previously frozen source genes; no heldout access."""
    bank=source.recover_candidates()
    if tuple(x["source_seed"] for x in bank)!=FAMILIES:
        raise AssertionError("Previously frozen families altered")
    selected={}
    for item in bank:
        src=item["source_seed"]
        selected[src]={}
        for kind in NEURAL_GENOTYPES:
            entry=item["genotypes"][kind]
            vals=tuple(entry["genes"])
            if (gene.gene_digest(vals)!=entry["gene_digest"] or
                    base.genome_digest(gene.to_weights(vals)) !=
                    entry["native_401_genome_digest"]):
                raise AssertionError("Genotype source digest mismatch")
            selected[src][kind]=vals
    return selected,bank


def simulate(seed: int, settings: Settings, paired: dict,
             schedule: tuple, flags: tuple[int,int,int],
             kind: str, source_seed: int | None = None,
             genes: tuple[float,...] | None = None, *,
             detailed: bool = False) -> tuple[dict,list[dict],dict | None]:
    settings.validate()
    if flags not in factorial_conditions():
        raise ValueError("Invalid factorial flags")
    if kind == "scripted_greedy_novelty":
        if genes is not None or source_seed is not None:
            raise ValueError("Scripted control must not use selected genotype")
    elif kind not in NEURAL_GENOTYPES or genes is None or source_seed not in FAMILIES:
        raise ValueError("Numeric neural transplant must reference frozen family")
    e=new_ecology(seed,settings,paired,flags,genes)
    r=flags[2]
    probability=(settings.shifted_regeneration_probability if r else
                 settings.regeneration_probability)
    allowed_schedule=tuple([
        (chance<probability,pos) for chance,pos in events
    ] for events in schedule)
    genotype_digest=(
        base.genome_digest(gene.to_weights(genes)) if genes else None
    )
    initial_food=tuple(sorted(e.food))
    actions_observed=0
    odor_actions=0
    food_adjacent_actions=0
    visits={}
    if kind=="scripted_greedy_novelty":
        visits={a.id:{a.position:1} for a in e.agents.values()}
    checks=[]
    for tick in range(settings.ticks+1):
        if tick:
            ids=list(e.agents)
            base._local_rng(seed,tick,0,0).shuffle(ids)
            for aid in ids:
                agent=e.agents.get(aid)
                if agent is None:
                    continue
                blocked,free=base.blocked_and_free(e,agent)
                sensors=base.observe(e,agent,blocked)
                legal=[i for i,b in enumerate(blocked) if not b]
                actions_observed+=1
                odor_actions+=int(any(sensors[4+i]>0 for i in legal))
                food_adjacent_actions+=int(any(
                    sensors[8+i]>0 for i in legal
                ))
                if kind=="scripted_greedy_novelty":
                    choice=script.scripted_choice(
                        e,agent,"greedy_novelty",tick,visits[aid]
                    )
                    script._act_scripted(e,aid,tick,choice)
                    agent_after=e.agents.get(aid)
                    if agent_after is not None:
                        place=agent_after.position
                        vis=visits[aid]
                        vis[place]=vis.get(place,0)+1
                    # Births are *researcher-scripted* here and never evidence
                    # of evolved neural competence.
                    for new_id,child in e.agents.items():
                        if new_id not in visits:
                            visits[new_id]={child.position:1}
                else:
                    # NATIVE numeric neural action logits and motor/birth physics
                    base.act(e,aid,tick)
            base._food_regeneration(e,allowed_schedule[tick-1])
            e.ticks_elapsed=tick
            if not e.agents and e.extinct_at is None:
                e.extinct_at=tick
        if (tick==0 or tick%settings.checkpoint_every==0
                or tick==settings.ticks):
            ck=base.snapshot(e,tick)
            ck.update({
                "split":None,"seed":seed,
                "flags":scenario_label(flags),
                "kind":kind,"source_seed":source_seed,
            })
            checks.append(ck)
            if abs(base.energy_balance(e))>1e-6:
                raise AssertionError("Original M5.0 physical energy ledger failed")
    if genes is not None:
        if any(a.genome!=gene.to_weights(genes) for a in e.agents.values()):
            raise AssertionError("Neural genotype changed during diagnostics")
        if any(
            evt["genome"]!=genotype_digest
            for evt in e.events if evt["type"]=="birth"
        ):
            raise AssertionError("All numeric offspring must be exact clones")
    row={
        "seed":seed,"source_seed":source_seed,
        "kind":kind,"flags":scenario_label(flags),
        "walls_stress":flags[0],"odor_stress":flags[1],
        "regrowth_stress":flags[2],
        "wall_count":len(e.world.walls),
        "accessible_cells":len(e.accessible),
        "map_attempt":paired["attempt"],
        "initial_food_digest":paired["initial_food_digest"],
        "initial_founders_digest":paired["founder_position_digest"],
        "candidate_pool_digest":paired["food_candidate_pool_digest"],
        "map_hash":paired["wall_hashes"][flags[0]],
        "genome_digest":genotype_digest,
        "survived":int(bool(e.agents)),
        "alive_final":len(e.agents),"births":e.births,
        "deaths":e.deaths,"food_eaten":e.food_eaten,
        "food_spawned":e.food_spawned,
        "food_remaining":len(e.food),"max_generation":e.max_generation,
        "extinct_at":e.extinct_at,
        "actions_observed":actions_observed,
        "odor_available_fraction":(
            odor_actions/actions_observed if actions_observed else 0
        ),
        "food_adjacent_fraction":(
            food_adjacent_actions/actions_observed if actions_observed else 0
        ),
        "neural_motor_mask_corrections":e.corrected_actions,
        "scripted_positive_benchmark":int(kind=="scripted_greedy_novelty"),
        "native_neural_action":int(kind!="scripted_greedy_novelty"),
        "fixed_clonal_genome":int(genes is not None),
        "energy_ledger_residual":base.energy_balance(e),
    }
    trace=None
    if detailed:
        trace={
            "seed":seed,"flags":scenario_label(flags),
            "kind":kind,"source_seed":source_seed,
            "world_walls":[list(x) for x in sorted(e.world.walls)],
            "initial_food":[list(x) for x in initial_food],
            "initial_founders":[list(x) for x in paired["initial_founders"]],
            "genotype_digest":genotype_digest,
            "checkpoints":checks,"events":e.events,
            "final_agents":[
                {
                    "id":a.id,"parent":a.parent,"founder":a.founder,
                    "generation":a.generation,"position":list(a.position),
                    "energy":a.energy
                }
                for a in sorted(e.agents.values(),key=lambda a:a.id)
            ]
        }
    return row,checks,trace


def _cross_boot(grid: list[list[float]], reps: int, seed: int
                ) -> tuple[float,float]:
    n=len(grid)
    m=len(grid[0])
    if n==0 or m==0 or any(len(row)!=m for row in grid):
        raise AssertionError("Invalid paired family/world grid")
    rng=random.Random(seed)
    vals=[]
    for _ in range(reps):
        fs=[rng.randrange(n) for _ in range(n)]
        ws=[rng.randrange(m) for _ in range(m)]
        vals.append(sum(
            grid[i][j] for i in fs for j in ws
        )/(n*m))
    vals.sort()
    return vals[int(.025*(reps-1))],vals[int(.975*(reps-1))]


def _mean_grid(grid: list[list[float]]) -> float:
    return sum(sum(row) for row in grid)/(len(grid)*len(grid[0]))


def factor_analyze(rows: list[dict], split: str, settings: Settings,
                   *, reps: int) -> dict:
    selected=[
        r for r in rows
        if r["split"]==split and r["kind"]=="descendant"
    ]
    families=list(FAMILIES)
    world_ids=sorted({r["seed"] for r in selected})
    expectation=settings.seeds if split=="main" else settings.shifted_seeds
    if len(world_ids)!=expectation:
        raise AssertionError("Wrong heldout world count")
    tab={(r["source_seed"],r["seed"],
          (r["walls_stress"],r["odor_stress"],r["regrowth_stress"])):
         r["survived"] for r in selected}
    if len(tab)!=len(families)*len(world_ids)*8:
        raise AssertionError("Incomplete paired factorial treatment grid")
    condition=sorted(factorial_conditions())
    def grid_by_effect(weights: dict[tuple[int,int,int],float]) -> list[list[float]]:
        return [[
            sum(weights[f]*tab[(origin,world,f)] for f in condition)
            for world in world_ids
        ] for origin in families]
    effects={}
    effect_weights={}
    for axis in AXES:
        idx=AXIS_N[axis]
        weights={f:(1./4 if f[idx] else -1./4) for f in condition}
        effect_weights[axis]=weights
        outcome=grid_by_effect(weights)
        avg=_mean_grid(outcome)
        lo,hi=_cross_boot(outcome,reps,88001+101*idx+(1000 if split=="validation" else 0))
        effects[axis]={
            "mean_stress_minus_benign":avg,
            "crossed_bootstrap_descriptive_95":[lo,hi],
            "harmful_supported":bool(avg<0 and hi<0),
            "source_family_effects":{
                str(f):statistics.mean(outcome[i]) for i,f in enumerate(families)
            }
        }
    ranked=sorted(AXES,key=lambda axis:effects[axis]["mean_stress_minus_benign"])
    worst=ranked[0]
    superiority={}
    for i,other in enumerate(AXES):
        if other==worst:continue
        weights={
            f:effect_weights[worst][f]-effect_weights[other][f]
            for f in condition
        }
        grid_diff=grid_by_effect(weights)
        gap=_mean_grid(grid_diff)
        lo,hi=_cross_boot(grid_diff,reps,89101+137*i+
                          (1000 if split=="validation" else 0))
        superiority[worst+"_minus_"+other]={
            "mean_difference_of_marginal_effects":gap,
            "crossed_bootstrap_descriptive_95":[lo,hi],
            "worse_with_upper_below_zero":bool(gap<0 and hi<0),
        }
    primary_dominant=bool(
        effects[worst]["harmful_supported"] and
        len(superiority)==2 and
        all(x["worse_with_upper_below_zero"] for x in superiority.values())
    )
    # Two-way interactions are paired diff-in-diff, averaged over the third
    # environmental axis, full 2^3 balance.
    interactions={}
    for a,b in (("walls","odor"),("walls","regrowth"),("odor","regrowth")):
        ai,bi=AXIS_N[a],AXIS_N[b]
        weights={f:(
            (1 if f[ai] else -1)*(1 if f[bi] else -1)/2
        ) for f in condition}
        interactions[a+"_x_"+b]=_mean_grid(grid_by_effect(weights))
    three={f:(
        (1 if f[0] else -1)*(1 if f[1] else -1)*
        (1 if f[2] else -1)
    ) for f in condition}
    interactions["walls_x_odor_x_regrowth"]=_mean_grid(grid_by_effect(three))
    by_condition={
        scenario_label(f):{
            "descendant_survival_rate":statistics.mean(
                tab[(origin,seed,f)] for origin in families for seed in world_ids
            ),
            "mean_descendant_food_eaten":statistics.mean(
                r["food_eaten"] for r in selected
                if (r["walls_stress"],r["odor_stress"],r["regrowth_stress"])==f
            ),
            "mean_descendant_births":statistics.mean(
                r["births"] for r in selected
                if (r["walls_stress"],r["odor_stress"],r["regrowth_stress"])==f
            )
        }
        for f in condition
    }
    return {
        "worlds":len(world_ids),"source_families":len(families),
        "effects":effects,"effect_order_most_negative_first":ranked,
        "most_negative_factor":worst,
        "factor_bottleneck_supported":primary_dominant,
        "dominance_pairwise":superiority,
        "interaction_point_estimates":interactions,
        "by_condition":by_condition
    }


def analyze(rows: list[dict],settings: Settings,
            *, boot_main: int = BOOT_MAIN,
            boot_validation: int = BOOT_VALIDATION) -> dict:
    by={}
    for split in ("main","validation"):
        subset=[r for r in rows if r["split"]==split]
        expected=(settings.seeds if split=="main" else
                  settings.shifted_seeds)*8*(len(FAMILIES)*2+1)
        if len(subset)!=expected:
            raise AssertionError("Missing factorial world trials")
        by[split]=factor_analyze(
            rows,split,settings,
            reps=boot_main if split=="main" else boot_validation
        )
    selected={
        split:{
            kind:{
                scenario_label(flags):{
                    "survived_worlds":sum(
                        r["survived"] for r in rows
                        if r["split"]==split and
                        r["kind"]==kind and
                        (r["walls_stress"],r["odor_stress"],
                         r["regrowth_stress"])==flags
                    ),
                    "trials":sum(
                        1 for r in rows if r["split"]==split and
                        r["kind"]==kind and
                        (r["walls_stress"],r["odor_stress"],
                         r["regrowth_stress"])==flags
                    ),
                    "mean_food_eaten":statistics.mean(
                        r["food_eaten"] for r in rows
                        if r["split"]==split and
                        r["kind"]==kind and
                        (r["walls_stress"],r["odor_stress"],
                         r["regrowth_stress"])==flags
                    ),
                }
                for flags in factorial_conditions()
            }
            for kind in ("descendant","ancestor","scripted_greedy_novelty")
        }
        for split in ("main","validation")
    }
    validation_signs={
        axis:(
            by["main"]["effects"][axis]["mean_stress_minus_benign"]*
            by["validation"]["effects"][axis]["mean_stress_minus_benign"]>0
        ) for axis in AXES
    }
    scripted_all_stress=selected["validation"]["scripted_greedy_novelty"][
        "w1o1r1"
    ]["survived_worlds"]
    return {
        "protocol":"docs/M5_4A_PROTOCOL.md",
        "interpretation":(
            "Crossed-world fixed-genome factorial, NOT evolution; five "
            "historically success-selected source families and nested "
            "matched-wall common restricted food candidate pool. "
            "Scripted benchmark uses researcher-written local greedy+novelty "
            "motor and birth, NOT neural weights or evolved competence."
        ),
        "main":by["main"],"validation":by["validation"],
        "by_policy":selected,
        "gates":{
            "single_dominant_main_stressor":
                by["main"]["factor_bottleneck_supported"],
            "validation_sign_concordance":validation_signs,
            "scripted_all_hard_validation_survivors":
                scripted_all_stress,
            "scripted_all_hard_physical_viability":
                scripted_all_stress>=4
        }
    }


def summary_svg(results: dict) -> str:
    by=results["by_policy"]
    scenarios=factorial_conditions()
    series=("descendant","ancestor","scripted_greedy_novelty")
    palette=("#2563eb","#d97706","#059669")
    width,height=990,460
    blocks=[]
    for idx,scenario in enumerate(scenarios):
        x=70+idx*112
        for k,kind in enumerate(series):
            stats=by["main"][kind][scenario_label(scenario)]
            height_bar=280*stats["survived_worlds"]/stats["trials"]
            blocks.append(
                f'<rect x="{x+k*24}" y="{332-height_bar:.2f}" '
                f'width="21" height="{height_bar:.2f}" fill="{palette[k]}"/>'
            )
        blocks.append(
            f'<text x="{x}" y="353" font-family="sans-serif" '
            f'font-size="13">{scenario_label(scenario)}</text>'
        )
    labels="".join(
        f'<rect x="{80+i*300}" y="402" width="13" height="13" '
        f'fill="{palette[i]}"/><text x="{98+i*300}" y="414" '
        f'font-size="13" font-family="sans-serif">{s}</text>'
        for i,s in enumerate(series)
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="white"/>'
        '<text x="70" y="26" font-size="19" font-family="sans-serif">'
        'M5.4A: MAIN survival, eight matched environmental conditions</text>'
        '<text x="20" y="58" font-size="13" font-family="sans-serif">'
        '100%</text><text x="34" y="333" font-size="13">0%</text>'
        '<line x1="65" y1="52" x2="65" y2="332" stroke="black"/>'
        '<line x1="65" y1="332" x2="981" y2="332" stroke="black"/>'
        + "".join(blocks)+labels+'</svg>\n'
    )


def run(settings: Settings,out: Path,*,boot_main:int=BOOT_MAIN,
        boot_validation:int=BOOT_VALIDATION) -> dict:
    settings.validate()
    out.mkdir(parents=True,exist_ok=True)
    selected,original= candidate_bank()
    bank={
        str(origin):{
            label:{
                "genes":list(selected[origin][label]),
                "gene_digest":gene.gene_digest(selected[origin][label]),
                "expanded_genome_digest":base.genome_digest(
                    gene.to_weights(selected[origin][label])
                )
            } for label in NEURAL_GENOTYPES
        } for origin in FAMILIES
    }
    (out/"frozen_numerical_genotype_bank.json").write_text(
        json.dumps(bank,sort_keys=True,indent=2)+"\n",encoding="utf-8"
    )
    print("M5.4A GENOTYPE BANK FROZEN",flush=True)
    rows,checkpoints,traces,maps=[],[],[],[]
    for split,start,n in (
        ("main",settings.seed_start,settings.seeds),
        ("validation",settings.shifted_start,settings.shifted_seeds),
    ):
        for seed in range(start,start+n):
            paired=coupled_world(seed,settings)
            schedule=common_food_schedule(seed,settings,paired)
            maps.append({
                key:value for key,value in paired.items() if key in (
                    "seed","attempt","low_wall_count","high_wall_count",
                    "low_reachable_count","high_reachable_count",
                    "wall_hashes","initial_food_digest",
                    "founder_position_digest","food_candidate_pool_digest"
                )
            })
            for f in factorial_conditions():
                for family in FAMILIES:
                    for kind in NEURAL_GENOTYPES:
                        row,ck,trace=simulate(
                            seed,settings,paired,schedule,f,kind,
                            source_seed=family,genes=selected[family][kind],
                            detailed=(seed==start)
                        )
                        row["split"]=split
                        rows.append(row)
                        for x in ck:
                            x["split"]=split
                        checkpoints.extend(ck)
                        if trace:
                            trace["split"]=split
                            traces.append(trace)
                        print(
                            "M5.4A ROW "+ " ".join(
                                [split,str(seed),scenario_label(f),
                                 kind,str(family),
                                 str(row["survived"]),str(row["alive_final"]),
                                 str(row["births"]),str(row["food_eaten"])]
                            ),flush=True
                        )
                row,ck,trace=simulate(
                    seed,settings,paired,schedule,f,
                    "scripted_greedy_novelty",detailed=(seed==start)
                )
                row["split"]=split
                rows.append(row)
                for x in ck:
                    x["split"]=split
                checkpoints.extend(ck)
                if trace:
                    trace["split"]=split
                    traces.append(trace)
                print("M5.4A ROW "+ " ".join([
                    split,str(seed),scenario_label(f),
                    "scripted_greedy_novelty","none",
                    str(row["survived"]),str(row["alive_final"]),
                    str(row["births"]),str(row["food_eaten"])
                ]),flush=True)
            print("M5.4A COMPLETE WORLD "+split+" "+str(seed),flush=True)
    if len(rows)!=(settings.seeds+settings.shifted_seeds)*8*11:
        raise AssertionError("Not all factorial trials ran")
    scientific=analyze(rows,settings,
                       boot_main=boot_main,boot_validation=boot_validation)
    report={
        "settings":asdict(settings),
        "simulation_count":len(rows),
        "genome_sources":list(FAMILIES),
        "map_constructor":"nested walls, common food/founder and candidate pool",
        **scientific
    }
    (out/"summary.json").write_text(
        json.dumps(report,sort_keys=True,indent=2)+"\n",encoding="utf-8"
    )
    for filename,data in (
        ("world_metrics.csv",rows),("population_checkpoints.csv",checkpoints)
    ):
        with (out/filename).open("w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]))
            writer.writeheader();writer.writerows(data)
    (out/"coupled_world_geometry.json").write_text(
        json.dumps(maps,sort_keys=True,indent=2)+"\n",encoding="utf-8"
    )
    for trace in traces:
        (out/("events_%s_%s_%s_%s.json" %(
            trace["split"],trace["seed"],trace["flags"],
            trace["kind"]+(
                "_"+str(trace["source_seed"])
                if trace["source_seed"] is not None else ""
            )
        ))).write_text(
            json.dumps(trace,sort_keys=True,indent=2)+"\n",encoding="utf-8"
        )
    (out/"factorial_survival.svg").write_text(
        summary_svg(report),encoding="utf-8"
    )
    print("M5.4A FACTORIAL "+json.dumps({
        "main":report["main"],"validation":report["validation"],
        "gates":report["gates"],"by_policy":report["by_policy"]
    },sort_keys=True),flush=True)
    return report


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    for f in fields(Settings):
        parser.add_argument("--"+f.name.replace("_","-"),
                            type=type(f.default),default=f.default)
    parser.add_argument("--out",type=Path,default=Path("results/m5-4a"))
    parser.add_argument("--bootstrap-main",type=int,default=BOOT_MAIN)
    parser.add_argument("--bootstrap-validation",type=int,default=BOOT_VALIDATION)
    args=vars(parser.parse_args())
    out=args.pop("out");bm=args.pop("bootstrap_main");bv=args.pop("bootstrap_validation")
    if bm<100 or bv<100:
        raise ValueError("Minimum 100 descriptive bootstrap draws")
    if args["ticks"]>=400 and (bm!=BOOT_MAIN or bv!=BOOT_VALIDATION):
        raise ValueError("Sealed full-campaign bootstrap counts cannot change")
    run(Settings(**args),out,boot_main=bm,boot_validation=bv)


if __name__=="__main__":
    main()
