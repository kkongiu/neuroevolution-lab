"""FlyLoop M0: first continuous artificial organism, not a fly brain.

World is independent from policy. Same local sensor map, five atomic
actions and 14-weight actor/critic topology for three paired arms.
No learned symbolic hypothesis or path-planning script is injected.
The third arm has an EXPLICIT, researcher-coded novelty bonus,
not a spontaneously learned curiosity policy.

Do not modify M5.6, main, or authorize evolution on these results.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path

SIZE = 13
HORIZON = 480
SWITCH_TICK = 240
WORLD_SEEDS = tuple(range(7100, 7132))
ARMS = ("frozen", "plastic", "plastic_novelty")
DIRECTIONS = ((0, -1), (1, 0), (0, 1), (-1, 0), (0, 0))
OLD_OPENINGS = frozenset((2, 10))
NEW_OPENINGS = frozenset((5, 8))
FOOD_COUNT = 16
HAZARD_COUNT = 9
REGROW_TICKS = 17
GAMMA = .92
LR_ACTOR = .03
LR_CRITIC = .012
TEMPERATURE = .8
NOVELTY_COEFFICIENT = 1.0
N_ACTOR = 8
N_CRITIC = 6
MAX_ENERGY = 100.0
MAX_HEALTH = 100.0

def _sha(data):
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

def _mean(items):
    return statistics.mean(items) if items else 0.0

@dataclass
class Body:
    x: int = 2
    y: int = 6
    energy: float = 70.0
    health: float = 100.0
    food_eaten: int = 0
    hits: int = 0
    damage_events: int = 0
    crossings_pre: int = 0
    crossings_post: int = 0
    food_returns: int = 0
    # Detect a completed crossing, not an attempted hop into gate cell.
    gate_entry_side: int | None = None

class World:
    """Only the environment knows resource positions and future barrier."""
    def __init__(self, seed: int):
        self.seed = seed
        self.rng = random.Random(seed * 7907 + 2003)
        self.tick = 0
        self.body = Body()
        self.visited = {(self.body.x, self.body.y)}
        self.consumed_food_spots = set()
        self.openings = OLD_OPENINGS
        self.walls = self._walls(self.openings)
        accessible = [
            (x, y) for x in range(1, SIZE-1) for y in range(1, SIZE-1)
            if (x,y) not in self.walls and (x,y) != (2,6)
        ]
        left = [(x,y) for x,y in accessible if x < 6]
        right = [(x,y) for x,y in accessible if x > 6]
        food_positions = self.rng.sample(left, 8)+self.rng.sample(right, 8)
        self.food_cooldown = {p:0 for p in food_positions}
        hazards_pool = [p for p in accessible
                        if p not in self.food_cooldown and p[0] != 6]
        self.hazards = frozenset(self.rng.sample(hazards_pool, HAZARD_COUNT))
        self.trace = []
        self.alive_at_switch = False
        self.health_before_switch = None
        self.food_before_switch = None
        self.cells_before_switch = None

    @staticmethod
    def _walls(openings):
        boundary = {
            (x,y) for x in range(SIZE) for y in range(SIZE)
            if x in (0,SIZE-1) or y in (0,SIZE-1)
        }
        return boundary | {(6,y) for y in range(1,SIZE-1) if y not in openings}

    def observe(self):
        b=self.body
        hunger=(MAX_ENERGY-b.energy)/MAX_ENERGY
        injury=(MAX_HEALTH-b.health)/MAX_HEALTH
        options=[]
        for dx,dy in DIRECTIONS:
            new=(b.x+dx,b.y+dy)
            is_wall=int(new in self.walls)
            is_hazard=int(new in self.hazards and not is_wall)
            is_food=int(
                new in self.food_cooldown
                and self.food_cooldown[new] <= self.tick
                and not is_wall
            )
            if dx==0 and dy==0:
                scent=0.0
            else:
                forward_candidates=[]
                for (fx,fy),ready in self.food_cooldown.items():
                    if ready > self.tick:
                        continue
                    vx,vy=fx-b.x,fy-b.y
                    distance=abs(vx)+abs(vy)
                    if 0 < distance <= 4 and vx*dx+vy*dy>0:
                        forward_candidates.append((5.0-distance)/4.0)
                scent=max(forward_candidates,default=0.0)
            # No absolute position or goal/map identity supplied here.
            options.append((
                1.0, hunger*scent, injury*is_hazard,
                float(is_food), float(scent), float(is_wall),
                float(is_hazard), float(dx==0 and dy==0),
            ))
        critic=(
            1.0, hunger, injury,
            max(v[3] for v in options),
            max(v[4] for v in options),
            sum(v[5] for v in options[:4])/4.0,
        )
        return options,critic

    def signature(self):
        options,_=self.observe()
        # Explicit coarse sensory novelty; NEVER use x/y or world seed.
        return tuple(
            (int(f[3]>.5),int(f[4]>.5),int(f[5]>.5),int(f[6]>.5))
            for f in options[:4]
        )+(round((MAX_ENERGY-self.body.energy)/25),)

    def switch_if_due(self):
        if self.tick != SWITCH_TICK:
            return False
        self.health_before_switch = self.body.health
        self.food_before_switch = self.body.food_eaten
        self.cells_before_switch = len(self.visited)
        self.openings = NEW_OPENINGS
        self.walls = self._walls(self.openings)
        # Environment can crush an organism standing in a closing
        # aperture; never teleport or grant nonphysical wall passage.
        if (self.body.x,self.body.y) in self.walls:
            self.body.health=0.0
            self.trace.append({
                "step":self.tick,"event":"gate_closure_crush",
                "position":[self.body.x,self.body.y],
                "energy":self.body.energy,"health":0.0,
            })
        self.alive_at_switch=self.body.health>0.0 and self.body.energy>0.0
        return True

    def step(self, action: int):
        if action not in range(5):
            raise ValueError("Only five allowed primitive actions")
        b=self.body
        if b.energy<=0 or b.health<=0:
            raise ValueError("Dead organism cannot act")
        old_x,old_y=b.x,b.y
        energy_before, health_before=b.energy,b.health
        dx,dy=DIRECTIONS[action]
        new=(old_x+dx,old_y+dy)
        hit= int(new in self.walls)
        if hit:
            b.hits+=1
        else:
            b.x,b.y=new
        b.energy-=.25+(.5 if hit else 0.0)
        ate=0
        hazard=0
        now=(b.x,b.y)
        if not hit and now in self.food_cooldown and self.food_cooldown[now]<=self.tick:
            b.energy=min(MAX_ENERGY,b.energy+17.0)
            self.food_cooldown[now]=self.tick+REGROW_TICKS
            ate=1
            if now in self.consumed_food_spots:
                b.food_returns+=1
            self.consumed_food_spots.add(now)
            b.food_eaten+=1
        if not hit and now in self.hazards:
            b.energy-=1.0
            b.health-=4.0
            hazard=1
            b.damage_events+=1
        b.energy=max(0.0,min(MAX_ENERGY,b.energy))
        b.health=max(0.0,min(MAX_HEALTH,b.health))
        self.visited.add(now)
        gate_completed=0
        if old_x!=6 and b.x==6:
            b.gate_entry_side = -1 if old_x<6 else 1
        elif old_x==6 and b.x!=6:
            if b.gate_entry_side is not None:
                exit_side = -1 if b.x<6 else 1
                if exit_side != b.gate_entry_side:
                    gate_completed=1
                    if self.tick>=SWITCH_TICK:
                        b.crossings_post+=1
                    else:
                        b.crossings_pre+=1
            b.gate_entry_side=None
        physical_reward=(b.energy-energy_before)+1.25*(b.health-health_before)
        self.trace.append({
            "step":self.tick,
            "from":[old_x,old_y],
            "action":action,
            "position":[b.x,b.y],
            "collision":hit,
            "food":ate,
            "hazard":hazard,
            "completed_gate_crossing":gate_completed,
            "physical_reward":round(physical_reward,6),
            "energy":round(b.energy,6),
            "health":round(b.health,6),
        })
        self.tick+=1
        return physical_reward, b.energy<=0.0 or b.health<=0.0


class Controller:
    def __init__(self, seed: int, arm: str, novelty_coefficient=None):
        if arm not in ARMS:
            raise ValueError("Unknown control arm")
        init_rng=random.Random(seed*13007+41)
        self.actor=[init_rng.uniform(-.15,.15) for _ in range(N_ACTOR)]
        self.critic=[init_rng.uniform(-.15,.15) for _ in range(N_CRITIC)]
        self.actor_start=tuple(self.actor)
        self.critic_start=tuple(self.critic)
        self.rng=random.Random(seed*16001+71)
        self.arm=arm
        self.novelty_coefficient=(
            0.0 if arm!="plastic_novelty" else (
                NOVELTY_COEFFICIENT if novelty_coefficient is None
                else novelty_coefficient
            )
        )
        self.observation_counts={}
        self.parameter_updates=0

    def _val(self, feat):
        return sum(w*f for w,f in zip(self.critic,feat))

    def choose(self, opts):
        logits=[sum(w*f for w,f in zip(self.actor,opt)) / TEMPERATURE for opt in opts]
        peak=max(logits)
        weights=[math.exp(z-peak) for z in logits]
        total=sum(weights)
        probabilities=[w/total for w in weights]
        draw=self.rng.random()
        cumulative=0.0
        chosen=4
        for i,p in enumerate(probabilities):
            cumulative+=p
            if draw<cumulative:
                chosen=i
                break
        return chosen,probabilities

    def adapt(self, opts,critic_feat,action,probs,physical_reward,
              new_critic_feat,terminal,new_signature):
        if self.arm=="frozen":
            return 0.0
        count=self.observation_counts.get(new_signature,0)+1
        self.observation_counts[new_signature]=count
        intrinsic=self.novelty_coefficient / math.sqrt(count)
        value=self._val(critic_feat)
        next_value=0.0 if terminal else self._val(new_critic_feat)
        delta=(physical_reward+intrinsic+GAMMA*next_value-value)
        clipped=max(-12.,min(12.,delta))
        expected=[sum(p*o[j] for p,o in zip(probs,opts)) for j in range(N_ACTOR)]
        for j in range(N_ACTOR):
            self.actor[j]+=LR_ACTOR*clipped*(opts[action][j]-expected[j])
        for j in range(N_CRITIC):
            self.critic[j]+=LR_CRITIC*clipped*critic_feat[j]
        self.parameter_updates+=1
        return intrinsic


def simulate(seed: int, arm: str, novelty_coefficient=None):
    w=World(seed)
    c=Controller(seed,arm,novelty_coefficient=novelty_coefficient)
    start_map={
        "food":[list(p) for p in sorted(w.food_cooldown)],
        "hazards":[list(p) for p in sorted(w.hazards)],
        "old_openings":sorted(OLD_OPENINGS),
        "new_openings":sorted(NEW_OPENINGS),
    }
    for tick in range(HORIZON):
        if w.tick != tick:
            raise AssertionError("World time drift")
        w.switch_if_due()
        if w.body.health<=0.0 or w.body.energy<=0.0:
            break
        options,critic=w.observe()
        action,probabilities=c.choose(options)
        physical,done=w.step(action)
        _,nextcritic=w.observe()
        bonus=c.adapt(options,critic,action,probabilities,physical,
                      nextcritic,done,w.signature())
        w.trace[-1]["intrinsic_novelty_bonus"]=round(bonus,6)
        w.trace[-1]["prediction_delta_is_neural_not_rule"]=True
        if done:
            break
    b=w.body
    action_rows=[r for r in w.trace if "action" in r]
    after=[r for r in action_rows if r["step"]>=SWITCH_TICK]
    row={
        "world_seed":seed,
        "arm":arm,
        "steps_alive":len(action_rows),
        "death_tick":w.tick if b.energy<=0 or b.health<=0 else None,
        "survived_480":len(action_rows)==HORIZON,
        "alive_at_switch":w.alive_at_switch,
        "food_consumed":b.food_eaten,
        "food_after_change":sum(r["food"] for r in after),
        "total_collisions":b.hits,
        "hazard_contacts":b.damage_events,
        "unique_cells_visited":len(w.visited),
        "unique_food_sites_visited":len(w.consumed_food_spots),
        "revisited_food_events":b.food_returns,
        "gate_crossings_before_change":b.crossings_pre,
        "gate_crossings_after_change":b.crossings_post,
        "terminal_energy":round(b.energy,6),
        "terminal_health":round(b.health,6),
        "mean_energy":round(_mean([r["energy"] for r in action_rows]),6),
        "mean_health":round(_mean([r["health"] for r in action_rows]),6),
        "start_map":start_map,
        "initial_weights":{
            "actor":list(c.actor_start),
            "critic":list(c.critic_start),
        },
        "final_weights":{
            "actor":list(c.actor),
            "critic":list(c.critic),
        },
        "online_parameter_updates":c.parameter_updates,
        "sensor_count_patterns":len(c.observation_counts),
        "novelty_coefficient":c.novelty_coefficient,
        "switch_began_at_tick":SWITCH_TICK,
        "trace":w.trace,
    }
    return row

def paired_seed_bootstrap(rows,arm,field,against="frozen",draws=1000):
    per_seed={
        seed: next(r[field] for r in rows if r["world_seed"]==seed and r["arm"]==arm)
              - next(r[field] for r in rows if r["world_seed"]==seed and r["arm"]==against)
        for seed in WORLD_SEEDS
    }
    rng=random.Random(711107)
    effects=sorted(_mean([per_seed[rng.choice(WORLD_SEEDS)]
                         for _ in WORLD_SEEDS]) for _ in range(draws))
    return {
        "mean_paired_difference":round(_mean(per_seed.values()),6),
        "exploratory_paired_world_seed_bootstrap95":[
            round(effects[int(draws*.025)],6),
            round(effects[int(draws*.975)-1],6),
        ],
    }

def run():
    rows=[simulate(seed,arm) for seed in WORLD_SEEDS for arm in ARMS]
    summary={
        arm:{
            field:round(_mean(r[field] for r in rows if r["arm"]==arm),6)
            for field in (
                "steps_alive","food_consumed","food_after_change",
                "total_collisions","hazard_contacts",
                "unique_cells_visited","revisited_food_events",
                "gate_crossings_before_change","gate_crossings_after_change",
                "terminal_energy","mean_energy","mean_health",
                "online_parameter_updates"
            )
        } | {
            "survived_480_fraction":round(_mean(
                r["survived_480"] for r in rows if r["arm"]==arm),6),
            "alive_at_switch_fraction":round(_mean(
                r["alive_at_switch"] for r in rows if r["arm"]==arm),6),
        }
        for arm in ARMS
    }
    f=summary["frozen"];p=summary["plastic"];n=summary["plastic_novelty"]
    # Identical system except beta = 0 yields bit-identical learning.
    zero_novelty_ablation=simulate(WORLD_SEEDS[0],"plastic_novelty",
                                   novelty_coefficient=0)
    same_novelty=simulate(WORLD_SEEDS[0],"plastic")
    def matching_except_arm(a,b):
        return all(a[k]==b[k] for k in a if k not in
                   ("arm","novelty_coefficient"))
    ablation_exact=matching_except_arm(zero_novelty_ablation,same_novelty)
    no_wall_violations=all(
        all(
            (rec["position"][0],rec["position"][1]) not in World._walls(
                NEW_OPENINGS if rec["step"]>=SWITCH_TICK else OLD_OPENINGS
            )
            and all(0.0 <= rec[key] <= 100.0 for key in ("energy","health"))
            and rec["action"] in range(5)
            for rec in row["trace"] if "action" in rec
        )
        for row in rows
    )
    paired_maps_weights=all(
        all(r["start_map"]==next(x["start_map"] for x in rows
                                     if x["world_seed"]==seed)
            and r["initial_weights"]==next(x["initial_weights"] for x in rows
                                     if x["world_seed"]==seed)
            for r in rows if r["world_seed"]==seed)
        for seed in WORLD_SEEDS
    )
    gates={
        "exact_32_seeds_by_3_arms_all_rows_present":len(rows)==96
            and {r["world_seed"] for r in rows}==set(WORLD_SEEDS)
            and all(sum(r["arm"]==arm and r["world_seed"]==seed
                        for r in rows)==1
                    for seed in WORLD_SEEDS for arm in ARMS),
        "legal_actions_physical_states_and_no_wall_passage":no_wall_violations,
        "identical_map_and_initial_14_weights_per_seed":paired_maps_weights,
        "frozen_weights_unchanged_plastic_both_update":
            all(r["initial_weights"]==r["final_weights"]
                and r["online_parameter_updates"]==0
                for r in rows if r["arm"]=="frozen")
            and all(r["online_parameter_updates"]>0
                    and r["initial_weights"]!=r["final_weights"]
                    for r in rows if r["arm"]!="frozen"),
        "zero_novelty_full_trace_and_weight_ablation_identical":ablation_exact,
        "gate_reconfiguration_verified_two_new_openings_distinct":
            OLD_OPENINGS.isdisjoint(NEW_OPENINGS)
            and all(r["start_map"]["old_openings"]==[2,10]
                    and r["start_map"]["new_openings"]==[5,8]
                    for r in rows),
        "plastic_food_ge_frozen_plus10pct":
            p["food_consumed"]>=1.10*f["food_consumed"],
        "plastic_lifetime_ge_frozen_plus5pct":
            p["steps_alive"]>=1.05*f["steps_alive"],
        "novelty_post_change_food_ge_plastic_plus10pct":
            n["food_after_change"]>=1.10*p["food_after_change"],
        "novelty_post_change_crossings_ge_plastic_plus0_5":
            n["gate_crossings_after_change"]>=
                p["gate_crossings_after_change"]+0.5,
    }
    paired={
        arm:{
            field:paired_seed_bootstrap(rows,arm,field)
            for field in ("steps_alive","food_consumed",
                          "food_after_change","gate_crossings_after_change")
        }
        for arm in ("plastic","plastic_novelty")
    }
    raw_sha=_sha(rows)
    return {
        "status":"FlyLoop M0 persistent ecology development (not biology)",
        "protocol":"docs/FLYLOOP_M0_PROTOCOL_2026-11-23.md",
        "metadata":{
            "world_seeds":WORLD_SEEDS,"world_count":len(WORLD_SEEDS),
            "horizon":HORIZON,"change_tick":SWITCH_TICK,
            "grid":[SIZE,SIZE],"food_patches":FOOD_COUNT,
            "hazard_patches":HAZARD_COUNT,
            "regrowth_steps":REGROW_TICKS,
            "actor_parameters":N_ACTOR,
            "critic_parameters":N_CRITIC,
            "actions":["north","east","south","west","wait"],
            "world_reward_based_on_actual_energy_and_health":True,
            "novelty_intrinsic_reward_programmed_by_researchers":True,
            "no_explicit_pathfinding_foraging_or_flee_script":True,
            "old_m56_sealed_ecology_untouched":True,
            "population_learning_between_lifetimes":False,
        },
        "summary":summary,
        "paired_exploratory_world_bootstrap":paired,
        "screening_gates":gates,
        "passed_count":sum(gates.values()),
        "narrow_dev_screen_pass":all(gates.values()),
        "general_intelligence_demonstrated":False,
        "autonomous_operator_invention_verified":False,
        "independent_ecology_replication_complete":False,
        "evolution_authorized":False,
        "advance_to_evolution":False,
        "raw_rows_sha256":raw_sha,
        "raw_rows":rows,
    }

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,
                        default=Path("results/flyloop/m0_persistent_ecology.json"))
    args=parser.parse_args()
    result=run()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n",
                        encoding="utf8")
    print("FLYLOOP_M0 "+json.dumps({
        k:result[k] for k in (
            "status","summary","paired_exploratory_world_bootstrap",
            "screening_gates","passed_count","narrow_dev_screen_pass",
            "raw_rows_sha256")
    },sort_keys=True),flush=True)

if __name__=="__main__":
    main()
