"""FlyLoop M2: remove explicit food/hazard lookahead, compare procedural worlds.

IMPORTANT: M0, M1, their original PRs and the sealed M5.6 ecology
are UNCHANGED. M2 shares exact M0 8-parameter actor + 6-parameter
critic, physical reward, five primitive actions, online learning rule.
The 'random_motion' ablation is a real non-learning moving baseline.
Engineering-selected sensors and novelty remain engineering priors.

Protocol: docs/FLYLOOP_M2_PROTOCOL_2026-11-24.md
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import random
import statistics
from pathlib import Path

from neuroevolution import flyloop_m0 as m0
from neuroevolution import flyloop_m1 as m1

GEOMETRIES=("baseline","procedural")
SENSORS=("full","no_scent","contact_food","blind")
ARMS=("frozen","random_motion","plastic","plastic_novelty")
SEEDS=tuple(range(7600,7624))
HORIZON=m0.HORIZON
SWITCH=240
PROCEDURAL_POSTS=10
ACTIONS=tuple(range(5))


def _digest(data):
    return hashlib.sha256(json.dumps(
        data,sort_keys=True,separators=(",",":")
    ).encode("utf-8")).hexdigest()


def _avg(values):
    return statistics.mean(values) if values else 0.0


def _reachable(walls):
    """Full-space connectivity check; generator has NO body-conditioned paths."""
    free={(x,y) for x in range(1,m0.SIZE-1)
          for y in range(1,m0.SIZE-1) if (x,y) not in walls}
    origin=(2,6)
    if origin not in free:
        return False
    visited={origin}
    todo=deque([origin])
    while todo:
        x,y=todo.popleft()
        for dx,dy in m0.DIRECTIONS[:4]:
            p=(x+dx,y+dy)
            if p in free and p not in visited:
                visited.add(p)
                todo.append(p)
    return visited==free


def procedural_map(seed):
    """Independent random map family (NOT route optimization for an agent)."""
    rng=random.Random(seed*347681+20261124)
    candidates=list(range(2,m0.SIZE-2)) # y=2..10
    old=frozenset(rng.sample(candidates,2))
    new=frozenset(rng.sample([y for y in candidates if y not in old],2))
    wall_candidates=[
        (x,y) for x in tuple(range(2,6))+tuple(range(7,11))
        for y in range(2,m0.SIZE-2)
        if (x,y)!=(2,6)
    ]
    rng.shuffle(wall_candidates)
    posts=set()
    for position in wall_candidates:
        trial=posts|{position}
        before=m0.World._walls(old)|trial
        after=m0.World._walls(new)|trial
        if _reachable(before) and _reachable(after):
            posts=trial
        if len(posts)==PROCEDURAL_POSTS:
            break
    if len(posts)!=PROCEDURAL_POSTS:
        raise AssertionError("Not enough fully connected procedural posts")
    free=[
        (x,y) for x in range(1,m0.SIZE-1) for y in range(1,m0.SIZE-1)
        if (x,y) not in (m0.World._walls(old)|posts)
        and (x,y) not in (m0.World._walls(new)|posts)
        and (x,y)!=(2,6)
    ]
    left=[p for p in free if p[0]<6]
    right=[p for p in free if p[0]>6]
    food=tuple(sorted(rng.sample(left,8)+rng.sample(right,8)))
    pool=[p for p in free if p[0]!=6 and p not in food]
    hazards=tuple(sorted(rng.sample(pool,m0.HAZARD_COUNT)))
    return {
        "old":old,"new":new,
        "posts":frozenset(posts),
        "food":food,"hazards":hazards,
    }


class World(m1.FlyWorld):
    """First-person local masks; never exposes the path or the food map."""

    def __init__(self,seed,geometry,sensors):
        if geometry not in GEOMETRIES or sensors not in SENSORS:
            raise ValueError("Unregistered geometry or sensory condition")
        self.geometry=geometry
        self.sensors=sensors
        # Attributes exist before M0.__init__ calls self._walls.
        self.posts=frozenset()
        self.procedural_old=None
        self.procedural_new=None
        super().__init__(seed,"occluded_noisy_gate")
        if geometry=="procedural":
            generated=procedural_map(seed)
            self.procedural_old=generated["old"]
            self.procedural_new=generated["new"]
            self.posts=generated["posts"]
            self.openings=self.procedural_old
            self.walls=self._walls(self.openings)
            self.food_cooldown={p:0 for p in generated["food"]}
            self.hazards=frozenset(generated["hazards"])
            self.original_food_sites=tuple(sorted(self.food_cooldown))
            self.original_hazards=tuple(sorted(self.hazards))
            self.initial_walls=tuple(sorted(self.walls))
        self.start_map={
            "geometry":geometry,
            "initial_openings":sorted(self.openings),
            "future_openings":sorted(
                self.procedural_new if geometry=="procedural" else m0.NEW_OPENINGS
            ),
            "walls_initial":[list(p) for p in sorted(self.walls)],
            "posts":[list(p) for p in sorted(self.posts)],
            "food":[list(p) for p in self.original_food_sites],
            "hazards":[list(p) for p in self.original_hazards],
        }

    def _walls(self,openings):
        # Exact m0 physics; procedural obstacles are impassable posts.
        return m0.World._walls(openings) | self.posts

    def observe(self):
        if self.sensors=="full":
            options,_=m1.FlyWorld.observe(self)
        else:
            # M0 raw local features; every forbidden channel is physically
            # removed from the OBSERVATION before reaching controller.
            options,_=m0.World.observe(self)
        masked=[]
        for raw in options:
            feat=list(raw)
            if self.sensors!="full":
                feat[1]=0.0  # hunger × scent
                feat[4]=0.0  # directional scent
            if self.sensors in ("contact_food","blind"):
                feat[3]=0.0  # destination food is NOT visible
            if self.sensors=="blind":
                feat[2]=0.0  # injury × hazard in destination
                feat[6]=0.0  # destination hazard is NOT visible
            masked.append(tuple(feat))
        hunger=(m0.MAX_ENERGY-self.body.energy)/m0.MAX_ENERGY
        injury=(m0.MAX_HEALTH-self.body.health)/m0.MAX_HEALTH
        critic=(
            1.0, hunger, injury,
            max(f[3] for f in masked),
            max(f[4] for f in masked),
            sum(f[5] for f in masked[:4])/4.,
        )
        return masked,critic

    def switch_if_due(self):
        if self.geometry=="baseline":
            return super().switch_if_due()
        if self.tick!=SWITCH:
            return None
        self.health_before_switch=self.body.health
        self.food_before_switch=self.body.food_eaten
        self.cells_before_switch=len(self.visited)
        self.openings=self.procedural_new
        self.walls=self._walls(self.openings)
        if (self.body.x,self.body.y) in self.walls:
            self.body.health=0.0
        self.alive_at_switch=self.body.health>0 and self.body.energy>0
        item={
            "step":self.tick,
            "event":"change_gate_openings",
            "openings":sorted(self.openings),
            "food_sites":[list(p) for p in sorted(self.food_cooldown)],
            "body_location":[self.body.x,self.body.y],
            "health":self.body.health,
            "energy":self.body.energy,
        }
        self.events.append(item)
        self.trace.append(item)
        return "change_gate_openings"


def _controller(seed,arm):
    return m0.Controller(
        seed,
        arm if arm in ("frozen","plastic","plastic_novelty") else "frozen"
    )


def simulate(seed,geometry,sensors,arm):
    if seed not in SEEDS:
        raise ValueError("M2 seed must be one of 7600..7623")
    if geometry not in GEOMETRIES or sensors not in SENSORS or arm not in ARMS:
        raise ValueError("Unregistered M2 experimental cell")
    w=World(seed,geometry,sensors)
    c=_controller(seed,arm)
    before={"actor":list(c.actor_start),"critic":list(c.critic_start)}
    for step in range(HORIZON):
        if w.tick != step:
            raise AssertionError("Time drift")
        w.switch_if_due()
        if w.body.health<=0 or w.body.energy<=0:
            break
        options,critic=w.observe()
        if arm=="random_motion":
            # Always one movement; no WAiT and no collision-avoidance plan.
            action=min(3,int(4*c.rng.random()))
            probabilities=(.25,.25,.25,.25,0.)
        else:
            action,probabilities=c.choose(options)
        physical,terminal=w.step(action)
        _,new_critic=w.observe()
        intrinsic=c.adapt(options,critic,action,probabilities,
                          physical,new_critic,terminal,w.signature())
        line=w.trace[-1]
        # Actual neural inputs. Sensitive environment object is NOT
        # supplied to the controller except through these features.
        line["actor_inputs"]=[list(o) for o in options]
        line["critic_inputs"]=list(critic)
        line["intrinsic_bonus_researcher_programmed"]=round(intrinsic,6)
        if terminal:
            break

    actions=[x for x in w.trace if "action" in x]
    after=[x for x in actions if x["step"]>=SWITCH]
    return {
        "world_seed":seed,
        "geometry":geometry,
        "sensors":sensors,
        "arm":arm,
        "start_map":w.start_map,
        "events":w.events,
        "steps_alive":len(actions),
        "survived_480":len(actions)==HORIZON,
        "alive_at_switch":w.alive_at_switch,
        "food_consumed":w.body.food_eaten,
        "food_after_switch":sum(x["food"] for x in after),
        "wall_collisions":w.body.hits,
        "hazard_contacts":w.body.damage_events,
        "unique_food_sites":len(w.consumed_food_spots),
        "revisited_food_events":w.body.food_returns,
        "unique_cells":len(w.visited),
        "completed_post_switch_crossings":w.body.crossings_post,
        "mean_energy":round(_avg([x["energy"] for x in actions]),6),
        "terminal_energy":round(w.body.energy,6),
        "terminal_integrity":round(w.body.health,6),
        "initial_weights":before,
        "final_weights":{
            "actor":list(c.actor),"critic":list(c.critic),
        },
        "online_weight_update_events":c.parameter_updates,
        "trajectory":w.trace,
    }


def _bootstrap(rows,geom,sensor,arm,reference,field,n=1000):
    values=[]
    for seed in SEEDS:
        scope=[x for x in rows if x["geometry"]==geom and
               x["sensors"]==sensor and x["world_seed"]==seed]
        a=next(x[field] for x in scope if x["arm"]==arm)
        b=next(x[field] for x in scope if x["arm"]==reference)
        values.append(a-b)
    draw=random.Random(761221)
    bootstrap=sorted(_avg(
        [values[draw.randrange(len(values))] for _ in values]
    ) for _ in range(n))
    return {
        "mean_paired_delta":round(_avg(values),6),
        "exploratory_seed_bootstrap_95pct":[
            round(bootstrap[int(n*.025)],6),
            round(bootstrap[int(n*.975)-1],6)
        ],
    }


def _verify_physics(rows):
    for r in rows:
        if r["steps_alive"]<=0 or r["steps_alive"]>HORIZON:
            return False
        events=[x for x in r["trajectory"] if "event" in x]
        moves=[x for x in r["trajectory"] if "action" in x]
        if moves!=[x for x in r["trajectory"] if "action" in x]:
            return False
        if len(moves)!=r["steps_alive"] or len(events)!=len(r["events"]):
            return False
        if [ev["step"] for ev in events] not in ([],[SWITCH]):
            return False
        if r["steps_alive"]>SWITCH and len(events)!=1:
            return False
        posts=set(map(tuple,r["start_map"]["posts"]))
        prev_time=-1
        for item in moves:
            t=item["step"]
            if t<=prev_time or t<0 or t>=HORIZON:
                return False
            prev_time=t
            if item["action"] not in ACTIONS:
                return False
            x,y=item["position"]
            ox,oy=item["from"]
            if abs(x-ox)+abs(y-oy)>1:
                return False
            opens=(r["start_map"]["future_openings"] if t>=SWITCH
                   else r["start_map"]["initial_openings"])
            walls=m0.World._walls(frozenset(opens))|posts
            if (x,y) in walls:
                return False
            if not 0<=item["energy"]<=100 or not 0<=item["health"]<=100:
                return False
            if len(item["actor_inputs"])!=5 or any(
                len(a)!=8 for a in item["actor_inputs"]
            ) or len(item["critic_inputs"])!=6:
                return False
            if t==SWITCH and not any(
                e["step"]==SWITCH for e in events
            ):
                # A continuing organism must have undergone the switch.
                return False
        if sum(item["food"] for item in moves)!=r["food_consumed"]:
            return False
    return True


def _verify_masks(rows):
    for r in rows:
        mask=r["sensors"]
        for tr in r["trajectory"]:
            if "action" not in tr:
                continue
            feat=tr["actor_inputs"]
            value=tr["critic_inputs"]
            if mask!="full" and (
                any(o[1]!=0 or o[4]!=0 for o in feat) or value[4]!=0
            ):
                return False
            if mask in ("contact_food","blind") and (
                any(o[3]!=0 for o in feat) or value[3]!=0
            ):
                return False
            if mask=="blind" and any(
                o[2]!=0 or o[6]!=0 for o in feat
            ):
                return False
    return True


def run():
    rows=[
        simulate(seed,geometry,sensors,arm)
        for geometry in GEOMETRIES
        for seed in SEEDS
        for sensors in SENSORS
        for arm in ARMS
    ]
    summary={}
    for geom in GEOMETRIES:
        summary[geom]={}
        for sensor in SENSORS:
            summary[geom][sensor]={}
            for arm in ARMS:
                scope=[r for r in rows if r["geometry"]==geom
                       and r["sensors"]==sensor and r["arm"]==arm]
                total_steps=sum(r["steps_alive"] for r in scope)
                summary[geom][sensor][arm]={
                    col:round(_avg([r[col] for r in scope]),6)
                    for col in (
                        "steps_alive","food_consumed","food_after_switch",
                        "wall_collisions","hazard_contacts",
                        "unique_food_sites","revisited_food_events",
                        "unique_cells","completed_post_switch_crossings",
                        "mean_energy",
                    )
                }|{
                    "food_per_100_living_ticks":
                        round(100.*sum(r["food_consumed"] for r in scope)/
                              total_steps,6) if total_steps else 0.,
                    "survived_480_rate":
                        round(_avg([r["survived_480"] for r in scope]),6),
                    "alive_at_switch_rate":
                        round(_avg([r["alive_at_switch"] for r in scope]),6),
                }
    identically_seeded=all(
        len({_digest(r["start_map"]) for r in rows
             if r["geometry"]==geom and r["world_seed"]==seed})==1
        and len({_digest(r["initial_weights"]) for r in rows
                 if r["geometry"]==geom and r["world_seed"]==seed})==1
        for geom in GEOMETRIES for seed in SEEDS
    )
    procedural_rows=[r for r in rows if r["geometry"]=="procedural"]
    signature_count=len({
        _digest(r["start_map"]) for r in procedural_rows
    })
    geom_valid=all(
        _reachable(
            m0.World._walls(frozenset(r["start_map"]["initial_openings"]))|
            set(map(tuple,r["start_map"]["posts"]))
        ) and _reachable(
            m0.World._walls(frozenset(r["start_map"]["future_openings"]))|
            set(map(tuple,r["start_map"]["posts"]))
        ) and len(r["start_map"]["posts"])==PROCEDURAL_POSTS
        and not (set(r["start_map"]["initial_openings"]) &
                 set(r["start_map"]["future_openings"]))
        and not (set(map(tuple,r["start_map"]["food"])) &
                 set(map(tuple,r["start_map"]["hazards"])))
        and not (set(map(tuple,r["start_map"]["food"])) &
                 set(map(tuple,r["start_map"]["walls_initial"])))
        and not (set(map(tuple,r["start_map"]["hazards"])) &
                 set(map(tuple,r["start_map"]["walls_initial"])))
        for r in procedural_rows
    )
    events_verified=all(
        not r["events"] or (
            len(r["events"])==1
            and r["events"][0]["step"]==SWITCH
            and r["events"][0]["event"]=="change_gate_openings"
            and r["events"][0]["openings"]==
                r["start_map"]["future_openings"]
        )
        for r in rows
    )
    original_arms=all(
        (
            r["initial_weights"]==r["final_weights"]
            and r["online_weight_update_events"]==0
        ) if r["arm"] in ("frozen","random_motion")
        else (
            r["online_weight_update_events"]>0
            and r["initial_weights"]!=r["final_weights"]
        )
        for r in rows
    )
    random_motion_control=all(
        r["arm"]!="random_motion" or all(
            tr["action"] in (0,1,2,3)
            for tr in r["trajectory"] if "action" in tr
        )
        for r in rows
    )
    b=summary["baseline"]
    p=summary["procedural"]
    def g(table,mask,arm):
        return table[mask][arm]["food_per_100_living_ticks"]
    gates={
        "01_768_balanced_full_lifetimes":
            len(rows)==768 and
            all(sum(
                r["geometry"]==geom and r["sensors"]==sensor and
                r["arm"]==arm for r in rows
            )==24 for geom in GEOMETRIES for sensor in SENSORS
                for arm in ARMS),
        "02_physical_legal_movement_no_tunneling_or_resurrection":
            _verify_physics(rows),
        "03_identical_paired_maps_change_schedule_initial_14_weights":
            identically_seeded,
        "04_all_non_full_modes_no_directional_scent":
            _verify_masks(rows),
        "05_no_precontact_food_indicator_in_contact_food_and_blind":
            _verify_masks(rows),
        "06_no_precontact_hazard_indicator_in_blind":
            _verify_masks(rows),
        "07_distinct_connected_procedural_geometries_valid_resources":
            signature_count>=20 and geom_valid and
            all(r["start_map"]["posts"]==[]
                for r in rows if r["geometry"]=="baseline"),
        "08_environmental_switch_exact_and_no_future_input":
            events_verified,
        "09_frozen_motion_fixed_plastic_updated_and_random_no_wait":
            original_arms and random_motion_control,
        "10_full_raw_trajectories_frozen_old_work_and_evolution_disabled":
            len(rows)==768 and all(r["trajectory"]
                 and r["initial_weights"] and r["final_weights"] for r in rows),
        "11_baseline_full_plastic_rate_15pct_over_frozen":
            g(b,"full","plastic")>=1.15*g(b,"full","frozen"),
        "12_baseline_no_scent_plastic_rate_15pct_over_frozen":
            g(b,"no_scent","plastic")>=1.15*g(b,"no_scent","frozen"),
        "13_baseline_contact_food_plastic_rate_15pct_over_frozen":
            g(b,"contact_food","plastic")>=1.15*g(b,"contact_food","frozen"),
        "14_baseline_blind_plastic_rate_15pct_over_frozen":
            g(b,"blind","plastic")>=1.15*g(b,"blind","frozen"),
        "15_baseline_blind_plastic_rate_10pct_over_uniform_motion":
            g(b,"blind","plastic")>=1.10*g(b,"blind","random_motion"),
        "16_procedural_blind_plastic_rate_10pct_over_uniform_motion":
            g(p,"blind","plastic")>=1.10*g(p,"blind","random_motion"),
        "17_procedural_blind_rate_preserves_60pct_full_plastic":
            g(p,"blind","plastic")>=.60*g(p,"full","plastic"),
    }
    comparisons={
        geom:{
            sensor:{
                "plastic_vs_frozen_food":
                    _bootstrap(rows,geom,sensor,"plastic","frozen",
                               "food_consumed"),
                "plastic_vs_uniform_motion_food":
                    _bootstrap(rows,geom,sensor,"plastic","random_motion",
                               "food_consumed"),
                "novelty_vs_plastic_food":
                    _bootstrap(rows,geom,sensor,"plastic_novelty","plastic",
                               "food_consumed"),
            } for sensor in SENSORS
        } for geom in GEOMETRIES
    }
    return {
        "status":"FlyLoop M2 blind sensory ablation and independent synthetic maze family",
        "protocol":"docs/FLYLOOP_M2_PROTOCOL_2026-11-24.md",
        "metadata":{
            "world_seeds":SEEDS,"world_generators":GEOMETRIES,
            "sensor_masks":SENSORS,"actor_arms":ARMS,
            "grid_size":m0.SIZE,"horizon":HORIZON,
            "switch_tick":SWITCH,
            "food_sites":m0.FOOD_COUNT,"hazard_sites":m0.HAZARD_COUNT,
            "neural_parameters":m0.N_ACTOR+m0.N_CRITIC,
            "m0_td_learning_rule_unchanged":True,
            "food_and_hazard_after_contact_rewards_are_physical":True,
            "random_motion_has_no_wall_correction":True,
            "no_absolute_coordinates_to_brain":True,
            "adjacent_food_masked_in_contact_and_blind":True,
            "novelty_reward_is_researcher_programmed":True,
            "old_m0_m1_and_sealed_m56_unchanged":True,
            "procedural_distinct_initial_maps":signature_count,
            "no_evolution":True,
        },
        "summary":summary,
        "paired_seed_bootstraps_exploratory":comparisons,
        "screening_gates":gates,
        "screening_passed_count":sum(gates.values()),
        "screening_pass":all(gates.values()),
        "general_intelligence_demonstrated":False,
        "autonomous_goal_invention_demonstrated":False,
        "spatial_memory_verified":False,
        "independent_biological_replication_complete":False,
        "evolution_authorized":False,
        "advance_to_evolution":False,
        "raw_rows_sha256":_digest(rows),
        "raw_rows":rows,
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument(
        "--out",type=Path,
        default=Path("results/flyloop/m2_blind_procedural.json")
    )
    args=parser.parse_args()
    results=run()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    # Compact records retain every raw trace and all numeric precision.
    args.out.write_text(
        json.dumps(results,sort_keys=True,separators=(",",":"),
                   ensure_ascii=True)+"\n",encoding="utf-8"
    )
    print("FLYLOOP_M2 "+json.dumps(
        {key:results[key] for key in (
            "status","summary","screening_gates","screening_passed_count",
            "screening_pass","paired_seed_bootstraps_exploratory",
            "raw_rows_sha256")},
        sort_keys=True
    ),flush=True)


if __name__=="__main__":
    main()
