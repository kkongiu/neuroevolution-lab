"""M5.4C preregistered 2×2 foraging-movement × birth-policy intervention.

Uses EXACT preselected externally optimised 18-gene recurrent numeric RNN.
NN motor and birth=unchanged M5.0; scripted motor and/or birth=RESEARCHER
PROGRAMMED diagnostic policies only. Original energy, food, movement, legal
birth, deterministic worlds and 400 ticks unchanged. Every real offspring
is an exact 401-weight numeric CLONE. NO TRUE WITHIN-WORLD EVOLUTION.

Protocol was frozen in docs/M5_4C_PROTOCOL.md in commit
c351085a15ebb9a56af5f3d198c2f3939905ab80 BEFORE this module.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from . import m5_0 as base
from . import m5_1 as script
from . import m5_3b as old
from . import m5_4b0 as b0
from . import m5_4b0r as recurrent

PROTOCOL_SHA = "c351085a15ebb9a56af5f3d198c2f3939905ab80"
SOURCE_MAIN_SHA = "387b7abe0bbe97526cfcb82a7d8050ed2c16f52f"
PINNED_401_DIGEST = "c5e5795332725de4"
SCRIPT_BIRTH_ENERGY = 46.0
ARMS = (
    "neural_move_neural_birth",
    "neural_move_scripted_birth",
    "scripted_move_neural_birth",
    "scripted_move_scripted_birth",
)
NN, NS, SN, SS = ARMS
SURVIVAL_CONTRASTS = (
    "movement_marginal",
    "reproduction_marginal",
    "factorial_interaction",
    "movement_minus_reproduction",
    "movement_only_rescue",
    "reproduction_only_rescue",
    "double_script_rescue",
)
SECONDARY_CONTRASTS = (
    "food_movement_marginal",
    "food_reproduction_marginal",
    "births_movement_marginal",
    "births_reproduction_marginal",
)
ALL_CONTRASTS = SURVIVAL_CONTRASTS + SECONDARY_CONTRASTS
BOOT_REPS = 20000


@dataclass(frozen=True)
class Settings(old.Settings):
    seed_start: int = 91000
    seeds: int = 24
    shifted_start: int = 91100
    shifted_seeds: int = 12
    odor_radius: int = 2
    shifted_odor_radius: int = 2
    no_food_controls: int = 0
    scripted_reproduction_threshold: float = SCRIPT_BIRTH_ENERGY

    def validate(self) -> None:
        super().validate()
        if not (
            self.odor_radius == 2 and self.shifted_odor_radius == 2
            and self.min_reproduction_energy <=
                self.scripted_reproduction_threshold <= self.max_energy
        ):
            raise ValueError("Short odor and safe physical scripted birth required")
        blocks=[
            set(range(self.seed_start,self.seed_start+self.seeds)),
            set(range(self.shifted_start,self.shifted_start+self.shifted_seeds)),
        ]
        historic=set()
        for start,n in (
            (79000,6),(79100,2),(80000,24),(81000,12),(82000,24),
            (83000,12),(84000,24),(85000,12),(86000,16),(87000,8),
            (88000,4),(88100,8),(88200,24),(88300,12),(89000,4),
            (89100,8),(89200,24),(89300,12),
        ):
            historic.update(range(start,start+n))
        if blocks[0]&blocks[1] or any(x&historic for x in blocks):
            raise ValueError("M5.4C new world ranges overlap prior research")
        if self.ticks>=400:
            canonical=Settings()
            for f in fields(Settings):
                if getattr(self,f.name)!=getattr(canonical,f.name):
                    raise ValueError(
                        "Precommitted full 400-tick factorial changed: "+f.name
                    )


def source_genes() -> tuple[float,...]:
    """Load EXACT ORIGINAL published source genotype, NEVER select anew.

    On every experiment call verify prior established pinned source digest,
    number of genes and original recurrent to_weights, independently of user
    supplied arguments or current branch configuration.
    """
    path=(
        Path(__file__).resolve().parents[1]/"reports"/
        "M5_4B0R_frozen_recurrent_genes_and_preregistered_statistics_2026-11-21.json"
    )
    source=json.loads(path.read_text(encoding="utf-8"))
    if source["frozen_preregistered_protocol_commit"] != (
        "7ffd8e06510a4c08a80c7610d7e0ff2617aec1a7"
    ):
        raise AssertionError("Invalid earlier prospectively frozen genotype")
    item=source["selected"]["recurrent"]
    genes=tuple(item["genes"])
    if (len(genes)!=18 or item["candidate_id"]!=83 or
            base.genome_digest(recurrent.to_weights(genes))!=PINNED_401_DIGEST):
        raise AssertionError("M5.4C cannot use newly tuned genotype")
    return genes


def scripted_movement(eco:base.Ecology, agent:base.Agent,
                      tick:int, visits:dict[base.Point,int],
                      sensors:tuple[float,...],
                      legal:list[int]) -> int|None:
    """Exact ORIGINAL human M5.1 greedy_novelty movement-only subpolicy.

    Never decides birth, and never sees global food, walls or a map beyond
    the native local sensors and this agent's own programmer-supplied visits.
    Tests compare to M5.1 scripted_choice when scripted birth ineligible.
    """
    if not legal:
        return None
    x,y=agent.position
    scores={}
    for direction in legal:
        dx,dy=base.DIRECTIONS[direction]
        dest=(x+dx,y+dy)
        scores[direction]=(
            4.*sensors[8+direction]+sensors[4+direction]+
            .35/(1.+visits.get(dest,0))
        )
    maximum=max(scores.values())
    options=[
        action for action in legal
        if abs(scores[action]-maximum)<1e-12
    ]
    return base._local_rng(
        eco.seed,tick,agent.id,37
    ).choice(options)


def _neural_best(logits:list[float],legal_moves:list[int],
                 birth_legal:bool) -> int|None:
    possible=legal_moves+([base.REPRODUCE] if birth_legal else [])
    return max(possible,key=lambda a:(logits[a],-a)) if possible else None


@dataclass
class Diagnostics:
    actions_observed: int=0
    movement_actions: int=0
    birth_actions: int=0
    scripted_movement_actions: int=0
    neural_movement_actions: int=0
    scripted_birth_actions: int=0
    neural_birth_actions: int=0
    birth_eligible_opportunities: int=0
    opportunities_script_birth_threshold: int=0
    neural_birth_requested_while_eligible: int=0
    neural_birth_preferred_unmasked: int=0
    odorless_actions: int=0
    adjacent_food_actions: int=0
    food_collected_in_actions: int=0
    odor_aligned_moves: int=0
    odor_alignable_moves: int=0
    revisited_moves: int=0
    energy_actions_total: float=0.
    energy_birth_total: float=0.
    age_birth_total: int=0
    max_observed_odorless_streak: int=0

    def as_metrics(self,visits:dict[int,dict[base.Point,int]],
                   global_cells:set[base.Point],
                   odorless_lifetime:dict[int,int],
                   eco:base.Ecology) -> dict:
        count=self.actions_observed
        return {
            "actions_observed":count,
            "movement_actions":self.movement_actions,
            "birth_actions":self.birth_actions,
            "scripted_movement_actions":self.scripted_movement_actions,
            "neural_movement_actions":self.neural_movement_actions,
            "scripted_birth_actions":self.scripted_birth_actions,
            "neural_birth_actions":self.neural_birth_actions,
            "birth_eligible_opportunities":self.birth_eligible_opportunities,
            "opportunities_script_birth_threshold":
                self.opportunities_script_birth_threshold,
            "neural_birth_requested_while_eligible":
                self.neural_birth_requested_while_eligible,
            "neural_birth_preferred_unmasked":
                self.neural_birth_preferred_unmasked,
            "birth_opportunity_fraction":(
                self.birth_eligible_opportunities/count if count else 0.
            ),
            "odorless_actions":self.odorless_actions,
            "odorless_action_fraction":(
                self.odorless_actions/count if count else 0.
            ),
            "adjacent_food_opportunities":self.adjacent_food_actions,
            "food_collected_in_actions":self.food_collected_in_actions,
            "food_per_100_movement_actions":(
                100.*self.food_collected_in_actions/self.movement_actions
                if self.movement_actions else 0.
            ),
            "odor_aligned_moves":self.odor_aligned_moves,
            "odor_alignable_moves":self.odor_alignable_moves,
            "odor_alignment_rate":(
                self.odor_aligned_moves/self.odor_alignable_moves
                if self.odor_alignable_moves else 0.
            ),
            "revisited_moves":self.revisited_moves,
            "revisit_fraction_of_moves":(
                self.revisited_moves/self.movement_actions
                if self.movement_actions else 0.
            ),
            "distinct_cells_visited_by_any_agent":len(global_cells),
            "total_organisms_ever_born":len(visits),
            "mean_unique_cells_visited_per_agent_life":(
                statistics.mean(len(path) for path in visits.values())
                if visits else 0.
            ),
            "mean_max_odorless_streak_per_agent_life":(
                statistics.mean(odorless_lifetime.values())
                if odorless_lifetime else 0.
            ),
            "max_observed_odorless_streak":
                self.max_observed_odorless_streak,
            "mean_energy_at_action":(
                self.energy_actions_total/count if count else 0.
            ),
            "mean_energy_at_real_birth":(
                self.energy_birth_total/self.birth_actions
                if self.birth_actions else 0.
            ),
            "mean_parent_age_at_real_birth":(
                self.age_birth_total/self.birth_actions
                if self.birth_actions else 0.
            ),
            "surviving_founder_lineages":len({
                a.founder for a in eco.agents.values()
            }),
        }


def _step(eco:base.Ecology,agent_id:int,tick:int,arm:str,
          diagnostic:Diagnostics,
          visits:dict[int,dict[base.Point,int]],
          global_cells:set[base.Point],
          odorless_run:dict[int,int],
          odorless_lifetime:dict[int,int]) -> None:
    agent=eco.agents.get(agent_id)
    if agent is None:
        return
    old_position=agent.position
    old_energy=agent.energy
    old_age=agent.age
    old_new_id=eco.next_id
    old_moves=eco.move_actions
    old_births=eco.births
    old_food=eco.food_eaten
    blocked,free=base.blocked_and_free(eco,agent)
    sensors=base.observe(eco,agent,blocked)
    legal=[action for action,blocked_dir in enumerate(blocked)
           if not blocked_dir]
    birth_legal=base.can_reproduce(eco,agent,free)

    # For ORIGINAL NN control the forward-pass is purely observational:
    # restore original hidden state before calling unchanged M5.0.act.
    # For the other three policy arms original RNN is evaluated ONCE and
    # retains its authentic recurrent hidden state.
    old_hidden=list(agent.hidden)
    logits=base.network_logits(agent,sensors,eco.settings)
    if arm==NN:
        agent.hidden=old_hidden

    neural_choice=_neural_best(logits,legal,birth_legal)
    energy_script_birth=bool(
        birth_legal and old_energy>=eco.settings.scripted_reproduction_threshold
    )
    if arm==NN:
        chosen=neural_choice
    elif arm==NS:
        chosen=(base.REPRODUCE if energy_script_birth else
                (max(legal,key=lambda a:(logits[a],-a))
                 if legal else None))
    elif arm==SN:
        chosen=(base.REPRODUCE if neural_choice==base.REPRODUCE
                else scripted_movement(
                    eco,agent,tick,visits[agent_id],sensors,legal
                ))
    elif arm==SS:
        chosen=script.scripted_choice(
            eco,agent,"greedy_novelty",tick,visits[agent_id]
        )
    else:
        raise ValueError("Unregistered intervention arm "+arm)

    if chosen is not None and (
        chosen not in legal and not (
            chosen==base.REPRODUCE and birth_legal
        )
    ):
        raise AssertionError("Hybrid attempted physically illegal movement/birth")

    diagnostic.actions_observed+=1
    diagnostic.energy_actions_total+=old_energy
    diagnostic.birth_eligible_opportunities+=int(birth_legal)
    diagnostic.opportunities_script_birth_threshold+=int(energy_script_birth)
    diagnostic.neural_birth_requested_while_eligible+=int(
        neural_choice==base.REPRODUCE
    )
    diagnostic.neural_birth_preferred_unmasked+=int(
        max(range(base.ACTIONS),key=lambda a:(logits[a],-a))==base.REPRODUCE
    )
    has_odor=any(sensors[4:8])
    if not has_odor:
        diagnostic.odorless_actions+=1
        odorless_run[agent_id]+=1
        odorless_lifetime[agent_id]=max(
            odorless_lifetime[agent_id],odorless_run[agent_id]
        )
        diagnostic.max_observed_odorless_streak=max(
            diagnostic.max_observed_odorless_streak,odorless_run[agent_id]
        )
    else:
        odorless_run[agent_id]=0
    diagnostic.adjacent_food_actions+=int(any(
        sensors[8+a]>0 for a in legal
    ))

    if arm==NN:
        base.act(eco,agent_id,tick)
    else:
        script._act_scripted(eco,agent_id,tick,chosen)

    actual_moves=eco.move_actions-old_moves
    actual_births=eco.births-old_births
    actual_food=eco.food_eaten-old_food
    if actual_moves not in (0,1) or actual_births not in (0,1):
        raise AssertionError("One agent action cannot move/birth multiple times")
    if actual_births!=int(chosen==base.REPRODUCE):
        raise AssertionError("Physical birth contradicts selected action")
    if actual_moves!=int(chosen in range(4)):
        raise AssertionError("Physical move contradicts selected action")
    diagnostic.movement_actions+=actual_moves
    diagnostic.birth_actions+=actual_births
    diagnostic.food_collected_in_actions+=actual_food

    if actual_moves:
        dx,dy=base.DIRECTIONS[chosen]
        destination=(old_position[0]+dx,old_position[1]+dy)
        if agent_id in eco.agents:
            if eco.agents[agent_id].position!=destination:
                raise AssertionError("Native motion did not reach selected cell")
        # Even a mover who STARVES on this action visited this final cell:
        # count its move, revisit, and footprint BEFORE losing the agent.
        path=visits[agent_id]
        diagnostic.revisited_moves+=int(destination in path)
        path[destination]=path.get(destination,0)+1
        global_cells.add(destination)
        if any(sensors[4+a]>0 for a in legal):
            diagnostic.odor_alignable_moves+=1
            max_odor=max(sensors[4+a] for a in legal)
            diagnostic.odor_aligned_moves+=int(
                abs(sensors[4+chosen]-max_odor)<1e-12
            )
        if arm in (SN,SS):
            diagnostic.scripted_movement_actions+=1
        else:
            diagnostic.neural_movement_actions+=1

    if actual_births:
        diagnostic.energy_birth_total+=old_energy
        diagnostic.age_birth_total+=old_age
        if arm in (NS,SS):
            diagnostic.scripted_birth_actions+=1
        else:
            diagnostic.neural_birth_actions+=1
        if eco.next_id!=old_new_id+1:
            raise AssertionError("A birth must create exactly one newborn ID")
        offspring=eco.agents[old_new_id]
        visits[old_new_id]={offspring.position:1}
        global_cells.add(offspring.position)
        odorless_run[old_new_id]=0
        odorless_lifetime[old_new_id]=0
        if offspring.genome!=agent.genome:
            raise AssertionError("All within-world offspring are genetic clones")


def _geometry_digest(eco:base.Ecology,schedule:tuple,
                     founders:tuple[base.Point,...]) -> str:
    x={
        "seed":eco.seed,
        "shifted":eco.shifted,
        "walls":sorted(eco.world.walls),
        "initial_food":sorted(eco.world.initial_food),
        "founders":founders,
        "accessible":eco.accessible,
        "food_rng_schedule":schedule,
    }
    return hashlib.sha256(json.dumps(
        x,sort_keys=True,separators=(",",":")
    ).encode("utf-8")).hexdigest()


def simulate(seed:int,settings:Settings,arm:str, *,
             shifted:bool=False,detailed:bool=False
             ) -> tuple[dict,list[dict],dict|None]:
    settings.validate()
    if arm not in ARMS:
        raise ValueError("Not preregistered "+arm)
    genes=source_genes()
    weights=recurrent.to_weights(genes)
    digest=base.genome_digest(weights)
    eco=base.create_ecology(
        seed,settings,"heritable_no_mutation",shifted=shifted
    )
    founders=tuple(agent.position for agent in eco.agents.values())
    for agent in eco.agents.values():
        agent.genome=weights
        agent.hidden=[0.]*settings.hidden
    eco.initial_genome_digests=tuple(digest for _ in eco.agents)
    schedule=base._food_schedule(seed,settings,eco,False)
    world_sha=_geometry_digest(eco,schedule,founders)
    visits={a.id:{a.position:1} for a in eco.agents.values()}
    global_cells={a.position for a in eco.agents.values()}
    odorless_run={a.id:0 for a in eco.agents.values()}
    odorless_lifetime={a.id:0 for a in eco.agents.values()}
    diagnostic=Diagnostics()
    checkpoints=[]
    for tick in range(settings.ticks+1):
        if tick:
            ids=list(eco.agents)
            base._local_rng(seed,tick,0,0).shuffle(ids)
            for aid in ids:
                _step(
                    eco,aid,tick,arm,diagnostic,visits,global_cells,
                    odorless_run,odorless_lifetime
                )
            base._food_regeneration(eco,schedule[tick-1])
            eco.ticks_elapsed=tick
            if not eco.agents and eco.extinct_at is None:
                eco.extinct_at=tick
        if (tick==0 or tick%settings.checkpoint_every==0
                or tick==settings.ticks):
            metrics=diagnostic.as_metrics(
                visits,global_cells,odorless_lifetime,eco
            )
            ck=base.snapshot(eco,tick)
            ck.update({
                "seed":seed,
                "split":"shifted" if shifted else "main",
                "arm":arm,
                "world_sha256":world_sha,
                "numeric_genome_digest":digest,
                "movement_actions_diagnosed":metrics["movement_actions"],
                "birth_actions_diagnosed":metrics["birth_actions"],
                "revisited_moves":metrics["revisited_moves"],
                "distinct_cells_visited_by_any_agent":
                    metrics["distinct_cells_visited_by_any_agent"],
                "odorless_action_fraction":
                    metrics["odorless_action_fraction"],
                "revisit_fraction_of_moves":
                    metrics["revisit_fraction_of_moves"],
                "food_per_100_movement_actions":
                    metrics["food_per_100_movement_actions"],
                "mean_energy_at_real_birth":
                    metrics["mean_energy_at_real_birth"],
                "surviving_founder_lineages":
                    metrics["surviving_founder_lineages"],
            })
            checkpoints.append(ck)
            if abs(base.energy_balance(eco))>1e-6:
                raise AssertionError("Unchanged M5.0 energy physics violated")

    if (diagnostic.movement_actions!=eco.move_actions or
            diagnostic.birth_actions!=eco.birth_actions or
            diagnostic.food_collected_in_actions!=eco.food_eaten):
        raise AssertionError("Action diagnostic counters disagree with physics")
    if (diagnostic.scripted_birth_actions+diagnostic.neural_birth_actions !=
            eco.births):
        raise AssertionError("Inconsistent physical birth intervention source")
    if (diagnostic.scripted_movement_actions+
            diagnostic.neural_movement_actions!=eco.move_actions):
        raise AssertionError("Inconsistent physical movement intervention source")
    if any(event["genome"]!=digest for event in eco.events
           if event["type"]=="birth"):
        raise AssertionError("True M5.0 birth mutated its numerical genotype")
    if any(a.genome!=weights for a in eco.agents.values()):
        raise AssertionError("Numerical population mutated inside-world")
    if len(visits)!=eco.next_id:
        raise AssertionError("Every newborn must receive a fresh personal visit map")
    metrics=diagnostic.as_metrics(visits,global_cells,odorless_lifetime,eco)
    row={
        "seed":seed,
        "split":"shifted" if shifted else "main",
        "arm":arm,
        "neural_movement":int(arm in (NN,NS)),
        "neural_reproduction":int(arm in (NN,SN)),
        "scripted_movement":int(arm in (SN,SS)),
        "scripted_reproduction":int(arm in (NS,SS)),
        "scripted_policy_is_human_written":int(arm!=NN),
        "numeric_genome_inherited":1,
        "within_world_genetic_mutations":0,
        "source_externally_trained":1,
        "original_401_genome_digest":digest,
        "original_18_numeric_gene_digest":recurrent.numeric_genes_digest(genes),
        "world_sha256":world_sha,
        "survived":int(bool(eco.agents)),
        "alive_final":len(eco.agents),
        "extinct_at":eco.extinct_at,
        "births":eco.births,
        "deaths":eco.deaths,
        "food_eaten":eco.food_eaten,
        "food_spawned":eco.food_spawned,
        "food_remaining":len(eco.food),
        "max_generation":eco.max_generation,
        "initial_population":settings.initial_population,
        "final_energy_ledger_residual":base.energy_balance(eco),
        **metrics,
    }
    if row["births"]!=metrics["birth_actions"]:
        raise AssertionError("Real birth count mismatch")
    trace=None
    if detailed:
        trace={
            "seed":seed,"split":row["split"],"arm":arm,
            "source_numeric_genes":list(genes),
            "native_401_genome_digest":digest,
            "world_sha256":world_sha,
            "world_walls":[list(p) for p in sorted(eco.world.walls)],
            "initial_food":[list(p) for p in eco.world.initial_food],
            "initial_founders":[list(p) for p in founders],
            "checkpoints":checkpoints,
            "physical_birth_and_food_events":eco.events,
            "diagnostic_visited_cells_by_agent":{
                str(k):{
                    str(point):count for point,count in sorted(v.items())
                }
                for k,v in visits.items()
            },
        }
    return row,checkpoints,trace


def _factorial(world_values:dict[str,float])->dict[str,float]:
    nn,ns,sn,ss=(world_values[a] for a in ARMS)
    return {
        "movement_marginal":((sn-nn)+(ss-ns))/2.,
        "reproduction_marginal":((ns-nn)+(ss-sn))/2.,
        "factorial_interaction":ss-sn-ns+nn,
        "movement_minus_reproduction":sn-ns,
        "movement_only_rescue":sn-nn,
        "reproduction_only_rescue":ns-nn,
        "double_script_rescue":ss-nn,
    }


def analyze(rows:list[dict],settings:Settings, *,
            bootstrap_reps:int=BOOT_REPS)->dict:
    settings.validate()
    outcome={}
    for split,start,n,bs in (
        ("main",settings.seed_start,settings.seeds,91301),
        ("shifted",settings.shifted_start,settings.shifted_seeds,91302),
    ):
        selected=[r for r in rows if r["split"]==split]
        if len(selected)!=n*4:
            raise AssertionError("Incomplete preregistered split factorial")
        by={arm:{
            r["seed"]:r for r in selected if r["arm"]==arm
        } for arm in ARMS}
        seeds=list(range(start,start+n))
        if any(sorted(by[arm])!=seeds for arm in ARMS):
            raise AssertionError("Non-unique/missing matched world seed")
        if any(len({by[arm][seed]["world_sha256"] for arm in ARMS})!=1
               for seed in seeds):
            raise AssertionError("World geometry, food or founders not matched")
        if any(len({by[arm][seed]["original_401_genome_digest"]
                    for arm in ARMS})!=1
               for seed in seeds):
            raise AssertionError("World arms did not share the SAME neural genome")

        arm_stats={
            arm:{
                "worlds":n,
                "survived_worlds":sum(by[arm][s]["survived"] for s in seeds),
                "mean_alive_at_tick400":statistics.mean(
                    by[arm][s]["alive_final"] for s in seeds
                ),
                "mean_real_births":statistics.mean(
                    by[arm][s]["births"] for s in seeds
                ),
                "mean_food_eaten":statistics.mean(
                    by[arm][s]["food_eaten"] for s in seeds
                ),
                "mean_unique_cells_per_agent_life":statistics.mean(
                    by[arm][s]["mean_unique_cells_visited_per_agent_life"]
                    for s in seeds
                ),
                "mean_revisit_fraction":statistics.mean(
                    by[arm][s]["revisit_fraction_of_moves"]
                    for s in seeds
                ),
                "mean_odorless_action_fraction":statistics.mean(
                    by[arm][s]["odorless_action_fraction"]
                    for s in seeds
                ),
                "mean_birth_energy":statistics.mean(
                    by[arm][s]["mean_energy_at_real_birth"]
                    for s in seeds
                ),
                "mean_surviving_founder_lineages":statistics.mean(
                    by[arm][s]["surviving_founder_lineages"]
                    for s in seeds
                ),
            }
            for arm in ARMS
        }
        metrics_by_seed={}
        for seed in seeds:
            surv={arm:float(by[arm][seed]["survived"]) for arm in ARMS}
            food={arm:float(by[arm][seed]["food_eaten"]) for arm in ARMS}
            births={arm:float(by[arm][seed]["births"]) for arm in ARMS}
            metrics_by_seed[seed]={
                **_factorial(surv),
                "food_movement_marginal":_factorial(food)["movement_marginal"],
                "food_reproduction_marginal":
                    _factorial(food)["reproduction_marginal"],
                "births_movement_marginal":
                    _factorial(births)["movement_marginal"],
                "births_reproduction_marginal":
                    _factorial(births)["reproduction_marginal"],
            }
        contrasts={}
        for index,name in enumerate(ALL_CONTRASTS):
            vals=[metrics_by_seed[seed][name] for seed in seeds]
            lo,hi=b0.paired_bootstrap(vals,bs+1000*index,bootstrap_reps)
            contrasts[name]={
                "mean_paired_effect":statistics.mean(vals),
                "world_paired_bootstrap_95":[lo,hi],
                "per_world":{str(s):v for s,v in zip(seeds,vals)},
                "bootstrap_excludes_zero":bool(lo>0 or hi<0),
            }
        outcome[split]={
            "arm_stats":arm_stats,
            "contrasts":contrasts,
            "worlds":n,
        }
    main=outcome["main"]
    em=main["contrasts"]["movement_marginal"]
    er=main["contrasts"]["reproduction_marginal"]
    dif=main["contrasts"]["movement_minus_reproduction"]
    main_arms=main["arm_stats"]
    movement=bool(
        em["mean_paired_effect"]>=.25 and
        em["world_paired_bootstrap_95"][0]>0 and
        dif["world_paired_bootstrap_95"][0]>0
    )
    reproduction=bool(
        er["mean_paired_effect"]>=.25 and
        er["world_paired_bootstrap_95"][0]>0 and
        dif["world_paired_bootstrap_95"][1]<0
    )
    interaction=main["contrasts"]["factorial_interaction"]
    joint=bool(
        main_arms[SS]["survived_worlds"]>=18 and
        main_arms[NS]["survived_worlds"]<12 and
        main_arms[SN]["survived_worlds"]<12
    )
    return {
        **outcome,
        "gates":{
            "positive_scripted_physical_control_ge18of24":bool(
                main_arms[SS]["survived_worlds"]>=18
            ),
            "causal_movement_bottleneck_dominant":movement,
            "causal_reproduction_bottleneck_dominant":reproduction,
            "dual_script_required_pattern":joint,
            "factorial_interaction_nonzero_ci":bool(
                interaction["bootstrap_excludes_zero"]
            ),
            "both_script_arms_genuinely_programmed":True,
            "within_world_genetic_evolution":False,
        },
        "diagnostic_conclusion":(
            "movement-dominant-programmed-rescue" if movement
            else "reproduction-dominant-programmed-rescue" if reproduction
            else "dual-script-required-pattern" if joint
            else "mixed-interacting-or-inconclusive"
        ),
        "inference_boundary":(
            "ONE fixed previously researcher-optimized numeric 18-gene "
            "RNN genotype. The intervention is a programmer-supplied local "
            "greedy+novelty motor choice and/or a programmer-supplied "
            "energy-threshold birth choice. The 20k paired bootstrap "
            "resamples only independent WORLD seeds, never organism "
            "actions or descendants. No in-world genotype mutation and "
            "no genuine evolutionary learning. Script-based rescue can "
            "localize a failure mode but is NOT evolved competence."
        ),
    }


def write_csv(path:Path,rows:list[dict])->None:
    if not rows:
        raise ValueError("No scientific data rows")
    keys=list(dict.fromkeys(k for row in rows for k in row.keys()))
    with path.open("w",encoding="utf-8",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def run(settings:Settings,out:Path, *,
        bootstrap_reps:int=BOOT_REPS)->dict:
    settings.validate()
    out.mkdir(parents=True,exist_ok=True)
    genes=source_genes()
    weights=recurrent.to_weights(genes)
    src={
        "preregistration_sha":PROTOCOL_SHA,
        "prior_main_source_sha":SOURCE_MAIN_SHA,
        "prior_selected_18_gene_candidate_id":83,
        "source_genes":list(genes),
        "source_full_401_weight_digest":base.genome_digest(weights),
        "source_18_gene_digest":recurrent.numeric_genes_digest(genes),
        "human_scripted_birth_threshold":SCRIPT_BIRTH_ENERGY,
        "human_scripted_movement_code":"m5_1.scripted_choice greedy_novelty",
        "no_within_world_genetic_mutations":True,
        "hybrid_programming_not_learned":True,
    }
    (out/"frozen_genotype_and_2x2_intervention.json").write_text(
        json.dumps(src,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    rows=[]
    checks=[]
    traces=[]
    worlds=[]
    for split,start,n,shifted in (
        ("main",settings.seed_start,settings.seeds,False),
        ("shifted",settings.shifted_start,settings.shifted_seeds,True),
    ):
        for seed in range(start,start+n):
            first_sha=None
            for arm in ARMS:
                row,ck,trace=simulate(
                    seed,settings,arm,shifted=shifted,detailed=(seed==start)
                )
                rows.append(row)
                checks.extend(ck)
                if first_sha is None:
                    first_sha=row["world_sha256"]
                    worlds.append({
                        "split":split,
                        "seed":seed,
                        "world_sha256":first_sha,
                        "source_genome_digest":row["original_401_genome_digest"],
                    })
                elif first_sha!=row["world_sha256"]:
                    raise AssertionError("Mismatched world across policy arms")
                if trace is not None:
                    traces.append(trace)
                # Emit original FULL rows on the sealed run, not just a
                # reduced summary. Permanent GitHub reports can therefore
                # retain every single preregistered causal diagnostic even
                # after GitHub Actions' 30-day binary artifact expires.
                print("M5.4C SEALED "+json.dumps(
                    row,sort_keys=True),flush=True)
                if settings.ticks>=400:
                    for checkpoint in ck:
                        print("M5.4C CHECKPOINT "+json.dumps(
                            checkpoint,sort_keys=True),flush=True)
    if settings.ticks>=400 and (len(rows)!=144 or len(worlds)!=36):
        raise AssertionError("Original frozen 144-world campaign incomplete")
    stats=analyze(rows,settings,bootstrap_reps=bootstrap_reps)
    final={
        "preregistration_sha":PROTOCOL_SHA,
        "source_genotype":src,
        "settings":asdict(settings),
        "worlds":len(worlds),
        "world_policy_trials":len(rows),
        "original_physical_ticks":settings.ticks,
        **stats,
    }
    write_csv(out/"all_144_matched_factorial_world_outcomes.csv",rows)
    write_csv(out/"all_population_checkpoints.csv",checks)
    (out/"world_geometry_and_rng_sha256.json").write_text(
        json.dumps(worlds,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    for t in traces:
        filename=f"events_{t['split']}_{t['seed']}_{t['arm']}.json"
        (out/filename).write_text(
            json.dumps(t,indent=2,sort_keys=True)+"\n",encoding="utf-8"
        )
    (out/"summary.json").write_text(
        json.dumps(final,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    print("M5.4C FULL RESULT "+json.dumps({
        "main":final["main"],
        "shifted":final["shifted"],
        "gates":final["gates"],
        "conclusion":final["diagnostic_conclusion"],
        "worlds":final["worlds"],
        "trials":final["world_policy_trials"],
    },sort_keys=True),flush=True)
    return final


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    for f in fields(Settings):
        parser.add_argument(
            "--"+f.name.replace("_","-"),
            default=f.default,type=type(f.default)
        )
    parser.add_argument("--out",type=Path,default=Path("results/m5-4c"))
    parser.add_argument("--bootstrap-reps",type=int,default=BOOT_REPS)
    args=vars(parser.parse_args())
    out=args.pop("out")
    bootstrap=args.pop("bootstrap_reps")
    if bootstrap<100:
        raise ValueError("Scientific bootstrap requires ≥100 paired draws")
    if args["ticks"]>=400 and bootstrap!=BOOT_REPS:
        raise ValueError("The original full campaign requires exactly 20000")
    run(Settings(**args),out,bootstrap_reps=bootstrap)


if __name__=="__main__":
    main()
