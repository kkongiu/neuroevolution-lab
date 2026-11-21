"""M5.3B: de novo neural genetic search with random 10-gene founders.

The 10 genes expand into a *numerical* 401-weight RNN with an explicitly
researcher-designed directional weight-sharing scaffold. All organism actions
are calculated by unchanged M5.0 network_logits/act; births and resources obey
unchanged M5.0 physics. Only actual in-world births alter inherited genes.
NO teacher, engineered genotype, external elite ranking or policy scripting.
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

from . import m5_0 as base

N_GENES = 10
HIDDEN = 12
ARMS = ("sparse_2", "dense_10", "clone", "random_child")
CONTRASTS = (
    ("sparse_2", "random_child"),
    ("sparse_2", "clone"),
    ("sparse_2", "dense_10"),
    ("dense_10", "random_child"),
)


@dataclass(frozen=True)
class Settings(base.Settings):
    seed_start: int = 82000
    seeds: int = 24
    shifted_start: int = 83000
    shifted_seeds: int = 12
    hidden: int = HIDDEN
    no_food_controls: int = 0
    initial_gene_sd: float = .65
    gene_mutation_sd: float = .16
    sparse_gene_count: int = 2

    def validate(self) -> None:
        super().validate()
        if (self.initial_gene_sd <= 0 or self.gene_mutation_sd <= 0
                or not 1 <= self.sparse_gene_count <= N_GENES):
            raise ValueError("Gene mutation parameters invalid")
        if self.hidden != HIDDEN:
            raise ValueError("Genes require 12-hidden numerical RNN")
        if self.ticks >= 400 and (
            self.initial_gene_sd != .65 or
            self.gene_mutation_sd != .16 or
            self.sparse_gene_count != 2
        ):
            raise ValueError("M5.3B sealed genomic parameters frozen")


def to_weights(genes: tuple[float, ...]) -> tuple[float, ...]:
    """Fixed symmetrical wiring; freely signed genes control strengths.

    This is a biased architecture, not an evolved topology or a scripted
    movement/reproduction policy. The UNMODIFIED m5_0.act chooses actions.
    """
    if len(genes) != N_GENES or any(not math.isfinite(x) for x in genes):
        raise ValueError("Exactly ten finite genes required")
    n = HIDDEN
    stride = base.INPUTS + n + 1
    head = n * stride
    weights = [0.] * base.genome_size(n)
    for action in range(4):
        off = action * stride
        weights[off + action] = -2. * genes[2]      # blocked
        weights[off + 4 + action] = 3. * genes[1]  # odor
        weights[off + 8 + action] = 6. * genes[0]  # food adjacent
        weights[off + base.INPUTS + n] = genes[3 + action]
        weights[head + action*(n + 1) + action] = 1.4
    # Birth energy response and its independently inherited threshold/scale.
    off = 4 * stride
    weights[off + 12] = 8. * genes[7]
    weights[off + base.INPUTS + n] = -6.24 * genes[7] + genes[8]
    weights[head + 4*(n + 1) + 4] = 2. * genes[9]
    weights[head + 4*(n + 1) + n] = -.2
    return tuple(weights)


def founder_genes(seed: int, settings: Settings
                  ) -> tuple[tuple[float, ...], ...]:
    rng = random.Random(seed * 8929 + 415)
    return tuple(tuple(rng.gauss(0., settings.initial_gene_sd)
                       for _ in range(N_GENES))
                 for _ in range(settings.initial_population))


def child_genes(parent: tuple[float, ...], settings: Settings,
                seed: int, tick: int, parent_id: int, arm: str
                ) -> tuple[tuple[float, ...], tuple[int, ...]]:
    if arm not in ARMS or len(parent) != N_GENES:
        raise ValueError("Invalid numeric genome or genetic arm")
    rng = base._local_rng(seed, tick, parent_id, 79)
    if arm == "clone":
        return parent, ()
    if arm == "random_child":
        return tuple(rng.gauss(0., settings.initial_gene_sd)
                     for _ in range(N_GENES)), tuple(range(N_GENES))
    indices = (tuple(sorted(rng.sample(range(N_GENES),
                                       settings.sparse_gene_count)))
               if arm == "sparse_2" else tuple(range(N_GENES)))
    child = list(parent)
    for i in indices:
        child[i] += rng.gauss(0., settings.gene_mutation_sd)
    return tuple(child), indices


def simulate(seed: int, settings: Settings, arm: str, *,
             shifted: bool = False, detailed: bool = False
             ) -> tuple[dict, list[dict], dict | None]:
    settings.validate()
    if arm not in ARMS:
        raise ValueError(arm)
    eco = base.create_ecology(seed, settings, "heritable_no_mutation",
                              shifted=shifted)
    genepool = founder_genes(seed, settings)
    genes_by_id: dict[int, tuple[float, ...]] = {}
    for agent, genes in zip(
        sorted(eco.agents.values(), key=lambda a: a.id), genepool
    ):
        genes_by_id[agent.id] = genes
        agent.genome = to_weights(genes)
    eco.initial_genome_digests = tuple(
        base.genome_digest(to_weights(g)) for g in genepool
    )
    schedule = base._food_schedule(seed, settings, eco, False)
    checks = []
    genetic_births = 0
    gene_changes_total = 0
    gene_distances: list[float] = []
    for tick in range(settings.ticks + 1):
        if tick:
            ids = list(eco.agents)
            base._local_rng(seed, tick, 0, 0).shuffle(ids)
            for aid in ids:
                parent = eco.agents.get(aid)
                if parent is None:
                    continue
                old_next = eco.next_id
                parent_weights = parent.genome
                # This handler decides ALL FIVE actions, with original
                # M5.0 masking and original birth-energy/food physics.
                base.act(eco, aid, tick)
                if eco.next_id != old_next:
                    if eco.next_id != old_next + 1:
                        raise AssertionError("One action produced >1 child")
                    child = eco.agents[old_next]
                    if child.genome != parent_weights:
                        raise AssertionError("Expected original cloned birth")
                    parent_genes = genes_by_id[aid]
                    new_genes, changed = child_genes(
                        parent_genes, settings, seed, tick, aid, arm
                    )
                    child_weights = to_weights(new_genes)
                    genes_by_id[child.id] = new_genes
                    child.genome = child_weights
                    eco.cosine_births[-1] = base.genome_cosine(
                        parent_weights, child_weights
                    )
                    event = eco.events[-1]
                    if event["type"] != "birth" or event["child"] != child.id:
                        raise AssertionError("Birth and genotype event mismatch")
                    event["genome"] = base.genome_digest(child_weights)
                    event["gene_digest"] = gene_digest(new_genes)
                    event["changed_gene_indices"] = list(changed)
                    event["gene_values"] = list(new_genes)
                    genetic_births += 1
                    gene_changes_total += len(changed)
                    gene_distances.append(math.sqrt(sum(
                        (x-y)**2 for x,y in zip(parent_genes,new_genes)
                    )))
            base._food_regeneration(eco, schedule[tick - 1])
            eco.ticks_elapsed = tick
            if not eco.agents and eco.extinct_at is None:
                eco.extinct_at = tick
        if (tick == 0 or tick % settings.checkpoint_every == 0
                or tick == settings.ticks):
            ck = base.snapshot(eco, tick)
            ck.update({"seed": seed, "arm": arm,
                       "split": "shifted" if shifted else "main"})
            checks.append(ck)
            if abs(base.energy_balance(eco)) > 1e-6:
                raise AssertionError("Ecology energy not conserved")
    if genetic_births != eco.births:
        raise AssertionError("Every new individual must have genetic provenance")
    if arm == "clone" and gene_changes_total != 0:
        raise AssertionError("Clones cannot mutate")
    if arm == "sparse_2" and gene_changes_total != eco.births*2:
        raise AssertionError("Sparse births must alter exactly two genes")
    if arm in ("dense_10","random_child") and gene_changes_total != eco.births*N_GENES:
        raise AssertionError("Dense/random treatments must replace ten genes")
    by_founder = {}
    for e in eco.events:
        if e["type"] == "birth":
            f = e["founder"]
            by_founder[f] = by_founder.get(f, 0) + 1
    alive = list(eco.agents.values())
    row = {
        "seed": seed, "split": "shifted" if shifted else "main",
        "arm": arm,
        "survived": int(bool(alive)), "alive_final": len(alive),
        "births": eco.births, "deaths": eco.deaths,
        "food_eaten": eco.food_eaten, "food_remaining": len(eco.food),
        "max_generation": eco.max_generation,
        "has_generation3": int(eco.max_generation >= 3),
        "extinct_at": eco.extinct_at,
        "top_lineage_birth_share": (
            max(by_founder.values()) / eco.births if eco.births else 0.
        ),
        "surviving_founder_lineages": len({a.founder for a in alive}),
        "mean_parent_child_cosine": (
            statistics.mean(eco.cosine_births) if eco.cosine_births else None
        ),
        "mean_parent_child_gene_distance": (
            statistics.mean(gene_distances) if gene_distances else 0.
        ),
        "genetic_births": genetic_births,
        "gene_changes_total": gene_changes_total,
        "motor_mask_corrections": eco.corrected_actions,
        "ineligible_reproduction_preferences":
            eco.reproduction_ineligible_wishes,
        "total_food_energy_gained": eco.cumulative_food_energy,
        "energy_ledger_residual": base.energy_balance(eco),
        "n_genes": N_GENES, "rnn_numerical_weights": base.genome_size(HIDDEN),
        "all_actions_native_neural": 1,
        "externally_selected_founders": 0,
    }
    trace = None
    if detailed:
        trace = {
            "seed": seed, "arm": arm, "split": row["split"],
            "initial_founder_genes": [list(g) for g in genepool],
            "initial_founder_digests": list(eco.initial_genome_digests),
            "walls": [list(p) for p in sorted(eco.world.walls)],
            "initial_food": [list(p) for p in eco.world.initial_food],
            "events": eco.events, "checkpoints": checks,
            "final_agents": [
                {"id": a.id, "founder": a.founder,
                 "parent": a.parent, "generation": a.generation,
                 "position": list(a.position), "energy": a.energy,
                 "gene_values": list(genes_by_id[a.id]),
                 "genome_digest": base.genome_digest(a.genome)}
                for a in sorted(alive, key=lambda a: a.id)
            ],
        }
    return row, checks, trace


def gene_digest(genes: tuple[float, ...]) -> str:
    # Exact decimal repr means this tag distinguishes actual inherited vectors.
    import hashlib
    return hashlib.sha256(
        "|".join(float(x).hex() for x in genes).encode("utf-8")
    ).hexdigest()[:20]


def contrasts(rows: list[dict], split: str, measure: str, seed: int) -> dict:
    group = {
        arm: {r["seed"]: r[measure] for r in rows
              if r["split"] == split and r["arm"] == arm}
        for arm in ARMS
    }
    keys = sorted(group["sparse_2"])
    if not keys or any(sorted(group[a]) != keys for a in ARMS):
        raise AssertionError("Genetic interventions must use paired worlds")
    result = {}
    for i, (a,b) in enumerate(CONTRASTS):
        delta = [group[a][k]-group[b][k] for k in keys]
        lo, hi = base.paired_bootstrap(delta, seed+i*103)
        mean = statistics.mean(delta)
        plus = sum(v>0 for v in delta)
        minus = sum(v<0 for v in delta)
        result[a+"_minus_"+b] = {
            "mean_delta": mean,
            "bootstrap_95_descriptive": [lo,hi],
            "positive_worlds": plus, "negative_worlds": minus,
            "tied_worlds": len(delta)-plus-minus,
            "supported": bool(mean>0 and lo>0 and plus>minus),
            "by_world": {str(k): v for k,v in zip(keys,delta)},
        }
    return result


def population_svg(checkpoints: list[dict], settings: Settings) -> str:
    width, height, left, top, bottom = 720, 405, 65, 45, 325
    ticks = sorted({x["tick"] for x in checkpoints if x["split"]=="main"})
    highest = max(settings.initial_population,max(
        x["population"] for x in checkpoints if x["split"]=="main"
    ))
    colors = ("#0d9488","#9333ea","#2563eb","#d97706")
    paths = []
    for arm,color in zip(ARMS,colors):
        values = [statistics.mean(
            x["population"] for x in checkpoints if
            x["arm"]==arm and x["split"]=="main" and x["tick"]==tick
        ) for tick in ticks]
        pts = " ".join(
            f"{left+(width-left-15)*tick/settings.ticks:.1f},"
            f"{bottom-(bottom-top)*value/highest:.1f}"
            for tick,value in zip(ticks,values)
        )
        paths.append(
            f'<polyline points="{pts}" fill="none" stroke="{color}" '
            f'stroke-width="2.6"/>'
        )
    keys = "".join(
        f'<text x="{left+i*153}" y="372" fill="{colors[i]}" '
        f'font-family="sans-serif" font-size="12">{a}</text>'
        for i,a in enumerate(ARMS)
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="white"/>'
        '<text x="65" y="25" font-family="sans-serif" font-size="16">'
        'M5.3B de novo neural genetic world populations</text>'
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" '
        'stroke="black"/>'
        f'<line x1="{left}" y1="{bottom}" x2="{width-15}" y2="{bottom}" '
        'stroke="black"/>'
        + "".join(paths) + keys + '</svg>\n'
    )


def run(settings: Settings, out: Path) -> dict:
    settings.validate()
    out.mkdir(parents=True, exist_ok=True)
    rows, checks, traces = [], [], []
    for split,start,count,shift in (
        ("main",settings.seed_start,settings.seeds,False),
        ("shifted",settings.shifted_start,settings.shifted_seeds,True),
    ):
        for seed in range(start,start+count):
            founder_ids = None
            for arm in ARMS:
                row,ck,trace = simulate(
                    seed,settings,arm,shifted=shift,
                    detailed=(seed==start)
                )
                rows.append(row)
                checks.extend(ck)
                if trace is not None:
                    if founder_ids is None:
                        founder_ids = trace["initial_founder_digests"]
                    elif founder_ids != trace["initial_founder_digests"]:
                        raise AssertionError("Founder genomes differ across arms")
                    traces.append(trace)
            print(f"M5.3B {split} {seed}: " + " ".join(
                f"{r['arm']}={r['alive_final']}/{r['births']}"
                for r in rows[-len(ARMS):]
            ),flush=True)
    means = {
        split: {
            arm: {
                metric: statistics.mean(
                    r[metric] for r in rows
                    if r["split"]==split and r["arm"]==arm
                )
                for metric in (
                    "survived","alive_final","births","food_eaten",
                    "max_generation","has_generation3","food_remaining",
                    "mean_parent_child_gene_distance",
                    "motor_mask_corrections"
                )
            } for arm in ARMS
        } for split in ("main","shifted")
    }
    main = contrasts(rows,"main","survived",settings.seed_start+17)
    shifted = contrasts(rows,"shifted","survived",settings.shifted_start+17)
    s=means["main"]["sparse_2"]
    shift_s=means["shifted"]["sparse_2"]
    gates = {
        "sparse_main_surviving_worlds":
            int(round(s["survived"]*settings.seeds)),
        "sparse_main_generation3_worlds":
            int(round(s["has_generation3"]*settings.seeds)),
        "main_viable": s["survived"]>=.5,
        "main_generational_depth": s["has_generation3"]>=.5,
        "inherited_vs_random":
            main["sparse_2_minus_random_child"]["supported"],
        "main_all_primary_pass": bool(
            s["survived"]>=.5 and s["has_generation3"]>=.5
            and main["sparse_2_minus_random_child"]["supported"]
        ),
        "novel_mutation_vs_clone":
            main["sparse_2_minus_clone"]["supported"],
        "shifted_surviving_worlds":
            int(round(shift_s["survived"]*settings.shifted_seeds)),
        "shifted_transfer": bool(
            shift_s["survived"]>=.5 and
            shifted["sparse_2_minus_random_child"]["supported"]
        ),
    }
    summary={
        "protocol": "M5.3B; docs/M5_3B_PROTOCOL.md",
        "settings": asdict(settings),
        "architecture": (
            "Ten freely signed numeric genes expanded to fixed, researcher-biased "
            "direction-shared 401-weight neural circuit; original m5_0.act"
        ),
        "means": means,
        "main_survival": main,
        "shifted_survival": shifted,
        "exploratory_main_food": contrasts(
            rows,"main","food_eaten",settings.seed_start+309
        ),
        "preregistered_gates": gates,
        "interpretation_limits":[
            "Spatially shared architectural features/scales fixed by researcher",
            "Every founding ten-gene vector RANDOM, no external training or elite",
            "All movements and births under original numeric neural action function",
            "Only natural within-world birth/death generates selection pressure",
            "Inherited advantage without clone advantage may be founder selection",
            "Genotype transfer to new worlds required for causal phenotype claims",
            "New organisms do not edit code, file systems or networks",
            "Whole worlds, not organisms, are independent units",
        ],
    }
    (out/"summary.json").write_text(
        json.dumps(summary,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    for filename,data in (
        ("world_metrics.csv",rows),
        ("population_checkpoints.csv",checks),
    ):
        with (out/filename).open("w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]))
            writer.writeheader();writer.writerows(data)
    for trace in traces:
        (out/f"events_{trace['split']}_{trace['arm']}.json").write_text(
            json.dumps(trace,indent=2,sort_keys=True)+"\n",encoding="utf-8"
        )
    (out/"population_curves.svg").write_text(
        population_svg(checks,settings),encoding="utf-8"
    )
    print("M5.3B GATES",gates,flush=True)
    print("M5.3B MAIN",main,flush=True)
    print("M5.3B SHIFTED",shifted,flush=True)
    print("M5.3B MEANS",means,flush=True)
    return summary


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    for f in fields(Settings):
        parser.add_argument("--"+f.name.replace("_","-"),
                            type=type(f.default),default=f.default)
    parser.add_argument("--out",type=Path,default=Path("results/m5-3b"))
    args=vars(parser.parse_args())
    out=args.pop("out")
    run(Settings(**args),out)


if __name__=="__main__":
    main()
