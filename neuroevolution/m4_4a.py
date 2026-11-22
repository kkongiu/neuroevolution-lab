"""M4.4A: online reward-trained Elman RNN, real physical-corridor food, causal memory.

Strict two-stage preregistered workflow: source bank with 288 actual rewarded
physical episodes is frozen in Git BEFORE any final world is evaluated.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from dataclasses import asdict
from pathlib import Path

from . import m4_2 as old

PROTOCOL_SHA = "af9e64e9e2ec3a5f47d27f963c27467661ee1681"
INITIALS = tuple(range(124000, 124008))
TRAIN_WORLDS = tuple(range(121000, 121024))
MAIN_WORLDS = tuple(range(122000, 122032))
OOD_WORLDS = tuple(range(123000, 123016))
AUTONOMOUS_WORLDS = tuple(range(123100, 123108))
DEVELOPMENT_WORLDS = tuple(range(120000, 120012))
EPOCHS = 6
UPDATES = len(TRAIN_WORLDS) * 2 * EPOCHS
LR = 0.045
MAX_NORM = 2.
L2 = 1e-5
BOOTSTRAP = 20000
ARMS = (
    "online_rnn", "initial_rnn", "online_rnn_reset",
    "online_rnn_flipcue", "shuffled_reward_rnn", "online_ff",
    "scripted_one_bit", "random_binary",
)
COMPARE = (
    ("online_minus_initial", "online_rnn", "initial_rnn"),
    ("online_minus_shuffled_feedback", "online_rnn", "shuffled_reward_rnn"),
    ("online_minus_memory_reset", "online_rnn", "online_rnn_reset"),
    ("online_minus_cue_flip", "online_rnn", "online_rnn_flipcue"),
    ("online_minus_feedforward", "online_rnn", "online_ff"),
)


def digest(x: object) -> str:
    return hashlib.sha256(json.dumps(
        x, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def _norm(x: list[float]) -> float:
    return math.sqrt(sum(v*v for v in x))


def _initial_genome(seed: int, kind: str, settings: old.Settings) -> list[float]:
    if kind not in ("rnn", "ff"):
        raise ValueError(kind)
    rng = random.Random(seed * 14327 + (11 if kind == "rnn" else 29))
    genome = list(old.fresh_genome(rng, settings, kind))
    genome[:] = [min(4., max(-4., w * (.35/settings.initial_weight_sd)))
                 for w in genome]
    n = settings.hidden if kind == "rnn" else settings.ff_hidden
    stride = old.INPUTS + (n if kind == "rnn" else 0) + 1
    out_at = n * stride
    for a in range(4):
        genome[out_at + a * (n+1) + n] = 0.
    if len(genome) != old.genome_size(kind, settings):
        raise AssertionError("Original M4.2 architecture changed")
    return genome


def physical_world_hash(world: old.TrialWorld) -> str:
    """Exclude hidden target side: both goals share physical corridor geometry."""
    return digest({
        "start": world.start, "fork": world.fork, "gates": world.gates,
        "forward": world.forward, "walls": sorted(world.walls),
        "stem_length": world.stem_length, "quarter": world.quarter_turns,
        "reflected": world.reflected, "shifted": world.shifted
    })


def _forward(genome: list[float] | tuple[float, ...],
             architecture: str, inputs: list[tuple[float, ...]],
             settings: old.Settings, *, reset_at_fork: bool = False
             ) -> tuple[list[float], list[tuple]]:
    """EXACT original M4.2 flat 260-RNN/268-FF tanh and linear logits."""
    n = settings.hidden if architecture == "rnn" else settings.ff_hidden
    if len(genome) != old.genome_size(architecture, settings):
        raise AssertionError("Architecture weights have wrong size")
    stride = old.INPUTS + (n if architecture == "rnn" else 0) + 1
    hidden = [0.] * n
    caches = []
    for j, x in enumerate(inputs):
        if len(x) != old.INPUTS:
            raise AssertionError("No extra hidden target/seed signal allowed")
        if reset_at_fork and j == len(inputs)-1:
            hidden = [0.] * n
        previous = hidden[:]
        hh = []
        for i in range(n):
            k = i * stride
            a = sum(genome[k+t] * x[t] for t in range(old.INPUTS))
            if architecture == "rnn":
                a += sum(genome[k+old.INPUTS+t] * previous[t]
                         for t in range(n))
            a += genome[k+stride-1]
            hh.append(math.tanh(a))
        caches.append((x, previous, hh))
        hidden = hh
    head = n * stride
    logits = [
        sum(genome[head+a*(n+1)+k]*hidden[k] for k in range(n))
        + genome[head+a*(n+1)+n]
        for a in range(old.ACTIONS)
    ]
    return logits, caches


def _softmax_two(logits: list[float],
                 legal: tuple[int, int]) -> tuple[float, float]:
    m = max(logits[a] for a in legal)
    v = [math.exp(logits[a]-m) for a in legal]
    return v[0]/sum(v), v[1]/sum(v)


def _reward_gradient(genome: list[float], architecture: str,
                     caches: list[tuple], legal: tuple[int, int],
                     chosen: int, feedback: int,
                     settings: old.Settings) -> tuple[list[float], float]:
    """Exact REINFORCE backprop THROUGH sensory history and recurrence.

    Reward is ONLY observed physical terminal +1/−1, NOT target direction.
    This is policy gradient on stochastic chosen action log-probability.
    """
    if feedback not in (-1, 1):
        raise AssertionError("Only scalar local physical feedback used")
    n = settings.hidden if architecture == "rnn" else settings.ff_hidden
    stride = old.INPUTS + (n if architecture == "rnn" else 0) + 1
    head = n*stride
    outputs = [
        sum(genome[head+a*(n+1)+i]*caches[-1][2][i]
            for i in range(n)) + genome[head+a*(n+1)+n]
        for a in range(4)
    ]
    ps = _softmax_two(outputs, legal)
    gradient = [0.] * len(genome)
    last_h = caches[-1][2]
    d_h = [0.] * n
    for a, prob in zip(legal, ps):
        coeff = feedback * (float(a == chosen)-prob)
        for i in range(n):
            gradient[head+a*(n+1)+i] += coeff * last_h[i]
            d_h[i] += coeff * genome[head+a*(n+1)+i]
        gradient[head+a*(n+1)+n] += coeff
    # Fully propagated exact BPTT gradient through tanh recurrent states.
    # A feedforward policy uses the final input only (earlier actions discarded).
    for x, previous, h in reversed(caches):
        new_d_h = [0.] * n
        for i in range(n):
            delta = d_h[i] * (1. - h[i] * h[i])
            off = i*stride
            for t in range(old.INPUTS):
                gradient[off+t] += delta*x[t]
            if architecture == "rnn":
                for t in range(n):
                    gradient[off+old.INPUTS+t] += delta*previous[t]
                    new_d_h[t] += delta*genome[off+old.INPUTS+t]
            gradient[off+stride-1] += delta
        if architecture != "rnn":
            break
        d_h = new_d_h
    return gradient, _norm(gradient)


def _apply_gradient(genome: list[float], gradient: list[float],
                    *, rate: float = LR) -> float:
    norm = _norm(gradient)
    scale = min(1., MAX_NORM/max(norm, 1e-12))
    old_weights = genome[:]
    for i in range(len(genome)):
        genome[i] = min(4., max(-4., genome[i]*(1.-rate*L2)
                                 + rate*scale*gradient[i]))
        if not math.isfinite(genome[i]):
            raise AssertionError("Non-finite online synapse")
    return _norm([a-b for a,b in zip(genome, old_weights)])


def physical_episode(world: old.TrialWorld, settings: old.Settings,
                     *, genome: list[float] | tuple[float, ...] | None,
                     architecture: str = "rnn", flip_cue: bool = False,
                     reset_at_fork: bool = False,
                     random_action_seed: int | None = None,
                     symbolic: bool = False,
                     fixed_binary_choice: int | None = None
                     ) -> tuple[dict, list[tuple] | None, tuple[int,int]]:
    """M4.2 full physical walls/position/energy/cost/food on scripted stem.

    The neural motor makes ONE physical legal fork action. This is explicitly
    GUIDED approach, not autonomous navigation; all reward only at the gate.
    """
    position = world.start
    energy = settings.initial_energy
    steps = 0
    sensory = []
    while position != world.fork:
        if steps >= settings.horizon:
            raise AssertionError("Physical gait failed to reach fork")
        x = old.observation(world, position, steps, energy, 0, 0, settings,
                            flip_cue=flip_cue)
        sensory.append(x)
        assert x[world.forward] == 0.
        dx, dy = old.DIRECTIONS[world.forward]
        target = position[0]+dx,position[1]+dy
        if target in world.walls:
            raise AssertionError("Physical path crossed wall")
        energy -= settings.movement_cost
        position = target
        steps += 1
    if steps != world.stem_length:
        raise AssertionError("Physical corridor length changed")
    obs = old.observation(world, position, steps, energy, 0, 0, settings,
                          flip_cue=flip_cue)
    sensory.append(obs)
    if any(obs[15:]):
        raise AssertionError("Goal cue leaks into fork decision")
    legal = tuple(sorted(old.direction(world.fork, g) for g in world.gates))
    if len(legal) != 2 or not all(obs[a] == 0. for a in legal):
        raise AssertionError("Fork branch physical mask invalid")
    if symbolic:
        initial_cue = sensory[0][15:]
        if initial_cue.count(1.) != 1:
            raise AssertionError("Scripted reference lacks one early cue")
        action = initial_cue.index(1.)
        if action not in legal:
            raise AssertionError("Illegal symbolic branch")
        caches = None
        probs = None
    elif fixed_binary_choice is not None:
        if fixed_binary_choice not in legal:
            raise AssertionError("Chosen action not in physical gates")
        action = fixed_binary_choice
        caches = None
        probs = None
    else:
        if genome is None:
            raise ValueError("Neural model needs genuine numerical weights")
        logits, caches = _forward(genome, architecture, sensory, settings,
                                  reset_at_fork=reset_at_fork)
        probs = _softmax_two(logits, legal)
        if random_action_seed is None:
            action = max(legal,key=lambda a:(logits[a], -a))
        else:
            draw = random.Random(random_action_seed).random()
            action = legal[0] if draw < probs[0] else legal[1]
    dx,dy = old.DIRECTIONS[action]
    position = position[0]+dx,position[1]+dy
    if position not in world.gates or position in world.walls:
        raise AssertionError("Final fork action not a real gate")
    energy -= settings.movement_cost
    steps += 1
    physically_ate_food = int(position == world.target)
    if physically_ate_food:
        energy = min(settings.max_energy, energy + settings.food_energy)
    # Reward comes EXCLUSIVELY from the physical consequence of own action.
    reward = 1 if physically_ate_food else -1
    if not 0 <= energy <= settings.max_energy:
        raise AssertionError("Physical energy balance invalid")
    trace = {
        "seed": world.seed,
        "side": world.side,
        "shifted": int(world.shifted),
        "geometry_sha": physical_world_hash(world),
        "fork_sensors_sha": digest(obs),
        "cue_observed_at_start": int(bool(any(sensory[0][15:]))),
        "cue_absent_at_fork": int(not any(obs[15:])),
        "guided_forward_moves": world.stem_length,
        "fork_reached": 1, "real_cardinal_moves": steps,
        "physical_food_eaten": physically_ate_food,
        "reward_from_food": reward,
        "chosen_action": action, "actual_final_energy": energy,
        "physical_collisions": 0, "food_energy_added":
            settings.food_energy if physically_ate_food else 0.,
        "motor_scaffold": 1,
        "branch_prob_chosen": None if probs is None else (
            probs[0] if action == legal[0] else probs[1]),
    }
    if (settings.initial_energy - steps*settings.movement_cost +
            settings.food_energy*physically_ate_food) != energy:
        raise AssertionError("Physical energy cost and reward mismatch")
    return trace, caches, legal


def _train_order(init_seed: int, epoch: int,
                 settings: old.Settings) -> list[old.TrialWorld]:
    worlds = [v for seed in TRAIN_WORLDS for v in old.make_pair(seed, settings)]
    random.Random(init_seed*1301 + epoch*23 + 71).shuffle(worlds)
    return worlds


def _modules_drift(before: list[float], after: list[float],
                   arch: str, settings: old.Settings) -> dict:
    n = settings.hidden if arch == "rnn" else settings.ff_hidden
    stride = old.INPUTS + (n if arch == "rnn" else 0)+1
    v={"input":0.,"recurrent":0.,"readout":0.,"hidden_bias":0.}
    for i,(a,b) in enumerate(zip(before,after)):
        if i>=n*stride:
            key="readout"
        else:
            o=i%stride
            key=("input" if o<old.INPUTS else
                 "recurrent" if arch=="rnn" and o<old.INPUTS+n else
                 "hidden_bias")
        v[key]+=(a-b)**2
    return {k:math.sqrt(value) for k,value in v.items()}


def train_source(*, settings: old.Settings | None = None,
                 initials: tuple[int,...] = INITIALS,
                 train_worlds: tuple[int,...] = TRAIN_WORLDS,
                 epochs: int = EPOCHS) -> tuple[dict,list[dict]]:
    settings = settings or old.Settings()
    settings.validate()
    if train_worlds != TRAIN_WORLDS:
        # Only isolate tiny development tests by explicitly opting in; full
        # frozen experiment always gets 24 original real SOURCE world seeds.
        if any(seed not in DEVELOPMENT_WORLDS for seed in train_worlds):
            raise ValueError("Unregistered training seeds")
    if set(train_worlds) & (set(MAIN_WORLDS)|set(OOD_WORLDS)|set(AUTONOMOUS_WORLDS)):
        raise AssertionError("Heldout world touched during synaptic training")
    if epochs != EPOCHS and train_worlds == TRAIN_WORLDS:
        raise ValueError("Cannot alter registered full training budget")
    families = []
    curves = []
    # old.make_pair NEVER called with a heldout range in this function.
    for init_seed in initials:
        if init_seed not in INITIALS and train_worlds == TRAIN_WORLDS:
            raise ValueError("Unknown full-campaign initial network")
        initial = {
            "rnn":_initial_genome(init_seed,"rnn",settings),
            "ff":_initial_genome(init_seed,"ff",settings),
        }
        learned={}
        for name,arch in (("online_rnn","rnn"),
                          ("shuffled_reward_rnn","rnn"),
                          ("online_ff","ff")):
            g=initial[arch][:]
            count=0
            for epoch in range(epochs):
                worlds=[w for seed in train_worlds
                        for w in old.make_pair(seed,settings)]
                random.Random(init_seed*1301 + epoch*23 + 71).shuffle(worlds)
                for index, world in enumerate(worlds):
                    episode = epoch*len(worlds)+index
                    trace, caches, legal=physical_episode(
                        world,settings,genome=g,architecture=arch,
                        random_action_seed=init_seed*10000019+episode*37+9)
                    reward=trace["reward_from_food"]
                    feedback=(
                        random.Random(init_seed*1000003+episode*43+317).choice((-1,1))
                        if name=="shuffled_reward_rnn" else reward)
                    gradient, grad_norm=_reward_gradient(
                        g,arch,caches,legal,trace["chosen_action"],
                        feedback,settings)
                    change=_apply_gradient(g,gradient)
                    count+=1
                    curves.append({
                        "init_seed":init_seed,"arm":name,"epoch":epoch,
                        "training_index":episode,"world_seed":world.seed,
                        "goal_side":world.side,
                        "physical_food_eaten":trace["physical_food_eaten"],
                        "physical_reward":reward,"reinforcement_feedback_used":feedback,
                        "gradient_norm":grad_norm,"weight_change_l2":change,
                        "physical_moves":trace["real_cardinal_moves"],
                        "physical_energy_after":trace["actual_final_energy"],
                        "source_world_geometry_sha":trace["geometry_sha"],
                    })
            if count != len(train_worlds)*2*epochs:
                raise AssertionError("Unmatched within-life synaptic update dose")
            learned[name]=g
        recurrent_drift=_modules_drift(
            initial["rnn"],learned["online_rnn"],"rnn",settings)
        if any(v<=1e-10 for v in recurrent_drift.values()):
            raise AssertionError("All RNN synaptic parameter modules MUST update")
        families.append({
            "init_seed":init_seed,"initial_rnn":initial["rnn"],
            "initial_ff":initial["ff"],
            "online_rnn":learned["online_rnn"],
            "shuffled_reward_rnn":learned["shuffled_reward_rnn"],
            "online_ff":learned["online_ff"],
            "initial_rnn_sha256":digest(initial["rnn"]),
            "initial_ff_sha256":digest(initial["ff"]),
            "online_rnn_sha256":digest(learned["online_rnn"]),
            "shuffled_reward_rnn_sha256":digest(learned["shuffled_reward_rnn"]),
            "online_ff_sha256":digest(learned["online_ff"]),
            "online_rnn_l2_drift_per_module":recurrent_drift,
            "training_physical_episodes_per_learned_variant":count,
            "source_training_world_geometry_hashes": [
                physical_world_hash(old.make_pair(s,settings)[0])
                for s in train_worlds],
        })
        print("M4.4A SOURCE "+json.dumps({
            "init_seed":init_seed,
            "rnn_synaptic_drift":recurrent_drift,
            "episodes_per_trained_arm":count
        }, sort_keys=True),flush=True)
    bank={
        "protocol_commit": PROTOCOL_SHA,
        "source_worlds":list(train_worlds),
        "heldout_worlds_examined":0,
        "genetic_initialization_seeds":list(initials),
        "episodes_per_training_arm":len(train_worlds)*2*epochs,
        "learning_rate":LR,"gradient_norm_clip":MAX_NORM,"l2_penalty":L2,
        "networks":families,
        "source_training_rows_sha256":digest(curves),
    }
    bank["bank_sha256"]=digest(bank)
    return bank, curves


def load_bank(path:Path, *, allow_development: bool=False)->dict:
    bank=json.loads(path.read_text(encoding="utf-8"))
    ref=bank.copy()
    original_sha=ref.pop("bank_sha256",None)
    if (ref.get("protocol_commit")!=PROTOCOL_SHA
        or original_sha!=digest(ref)
        or bank.get("heldout_worlds_examined")!=0):
        raise AssertionError("Source bank mutated after freeze")
    if not allow_development and (
        bank.get("source_worlds")!=list(TRAIN_WORLDS)
        or bank.get("genetic_initialization_seeds")!=list(INITIALS)
        or bank.get("episodes_per_training_arm")!=UPDATES):
        raise AssertionError("Exact predeclared 288×8 source bank required")
    for family in bank["networks"]:
        seed=family["init_seed"]
        for name in ("initial_rnn","initial_ff","online_rnn",
                     "shuffled_reward_rnn","online_ff"):
            if family[name+"_sha256"]!=digest(family[name]):
                raise AssertionError("Frozen full numerical weights altered")
        if family["initial_rnn"] != _initial_genome(seed,"rnn",old.Settings()):
            raise AssertionError("Original full source RNN initial state changed")
        if family["initial_ff"] != _initial_genome(seed,"ff",old.Settings()):
            raise AssertionError("Original full source FF initial state changed")
        if family["online_rnn_l2_drift_per_module"]["recurrent"] <= 0:
            raise AssertionError("No genuine synaptic recurrent plasticity")
    return bank


def _evaluate_arm(world:old.TrialWorld, arm:str,
                  family:dict, settings:old.Settings)->dict:
    arch="ff" if arm=="online_ff" else "rnn"
    if arm=="random_binary":
        legal=sorted(old.direction(world.fork,g) for g in world.gates)
        forced=random.Random(world.seed*1117+23).choice(legal)
        trace,_,_=physical_episode(
            world,settings,genome=None,fixed_binary_choice=forced)
        genom="random_baseline"
    elif arm=="scripted_one_bit":
        trace,_,_=physical_episode(
            world,settings,genome=None,symbolic=True)
        genom="symbolic_upper_bound"
    else:
        variant={
            "online_rnn":"online_rnn","initial_rnn":"initial_rnn",
            "online_rnn_reset":"online_rnn","online_rnn_flipcue":"online_rnn",
            "shuffled_reward_rnn":"shuffled_reward_rnn","online_ff":"online_ff",
        }[arm]
        g=family[variant]
        trace,_,_=physical_episode(
            world,settings,genome=g,architecture=arch,
            reset_at_fork=(arm=="online_rnn_reset"),
            flip_cue=(arm=="online_rnn_flipcue"))
        genom=family[variant+"_sha256"]
    return {
        "init_seed":family["init_seed"],"arm":arm,
        "architecture":arch if "one_bit" not in arm and
        "random_binary" not in arm else "programmed_reference",
        "genome_sha256":genom,
        **trace,
    }


def _bootstrap(diffs:list[float],seed:int,reps:int)->tuple[float,float]:
    rng=random.Random(seed)
    vals=sorted(statistics.mean(rng.choices(diffs,k=len(diffs)))
                for _ in range(reps))
    return vals[int(.025*(reps-1))],vals[int(.975*(reps-1))]


def evaluate(rows:list[dict],bank:dict,
             *, main:tuple[int,...]=MAIN_WORLDS,
             ood:tuple[int,...]=OOD_WORLDS,
             reps:int=BOOTSTRAP)->dict:
    result={}
    ids=[z["init_seed"] for z in bank["networks"]]
    for split, seeds, bootstrap_seed in (
        ("main",main,125001),("ood",ood,125002)):
        selected=[r for r in rows if r["split"]==split]
        if len(selected)!=len(ids)*len(seeds)*2*len(ARMS):
            raise AssertionError("Incomplete preregistered final factorial")
        mapping={(r["init_seed"],r["seed"],r["side"],r["arm"]):r
                 for r in selected}
        if len(mapping)!=len(selected):
            raise AssertionError("Duplicated final world-goal trials")
        for seed in seeds:
            for ini in ids:
                for side in (0,1):
                    matched=[mapping[(ini,seed,side,arm)] for arm in ARMS]
                    if len({x["geometry_sha"] for x in matched})!=1:
                        raise AssertionError("Different geometry per intervention")
                    if len({x["fork_sensors_sha"] for x in matched})!=1:
                        raise AssertionError("Goal signal leaks at fork")
                    if any(x["cue_absent_at_fork"]!=1 or
                           x["fork_reached"]!=1 or
                           x["physical_collisions"]!=0 for x in matched):
                        raise AssertionError("Scaffold violated original physical maze")
                for arm in ARMS:
                    x=mapping[(ini,seed,0,arm)]
                    y=mapping[(ini,seed,1,arm)]
                    if (x["geometry_sha"]!=y["geometry_sha"] or
                        x["fork_sensors_sha"]!=y["fork_sensors_sha"]):
                        raise AssertionError("Paired goals reveal target at fork")
        averages={}
        for arm in ARMS:
            vals=[z["physical_food_eaten"] for z in selected if z["arm"]==arm]
            averages[arm]={
                "collected_food_rate":statistics.mean(vals),
                "actual_food_collected":sum(vals),
                "nested_family_goal_world_trials":len(vals),
                "mean_physical_energy_after":statistics.mean(
                    z["actual_final_energy"] for z in selected if z["arm"]==arm),
            }
        comps={}
        for ix,(name,left,right) in enumerate(COMPARE):
            diffs=[
                statistics.mean(
                    mapping[(ini,seed,side,left)]["physical_food_eaten"]-
                    mapping[(ini,seed,side,right)]["physical_food_eaten"]
                    for seed in seeds for side in (0,1)
                ) for ini in ids
            ]
            lo,hi=_bootstrap(diffs,bootstrap_seed+ix*1000,reps)
            comps[name]={
                "mean_paired_effect":statistics.mean(diffs),
                "source_init_cluster_bootstrap95":[lo,hi],
                "per_independent_neural_initialization":dict(zip(map(str,ids),diffs)),
                "positive_independent_initializations":sum(d>0 for d in diffs),
                "lower95_greater_than_zero":lo>0,
            }
        result[split]={
            "world_geometries":len(seeds),
            "independent_network_initializations":len(ids),
            "arms":averages,"comparisons":comps,
        }
    m=result["main"];o=result["ood"]
    g={
        "main_recurrent_online_lifetime_learning_gate":(
            m["arms"]["online_rnn"]["collected_food_rate"]>=.65
            and m["comparisons"]["online_minus_initial"]["lower95_greater_than_zero"]
            and m["comparisons"]["online_minus_shuffled_feedback"]["lower95_greater_than_zero"]
            and m["comparisons"]["online_minus_memory_reset"]["lower95_greater_than_zero"]
            and m["comparisons"]["online_minus_initial"]["positive_independent_initializations"]>=6
        ),
        "ood_long_delay_transfer_gate":(
            o["arms"]["online_rnn"]["collected_food_rate"]>=.60
            and o["comparisons"]["online_minus_memory_reset"]["lower95_greater_than_zero"]
        ),
        "physical_scaffold_required":True,
        "unassisted_autonomous_lifelong_learning_proven":False,
        "genetic_inheritance_of_acquired_weight_changes":False,
    }
    return {
        **result,"gates":g,
        "inference_warning":(
            "Eight independent network initializations, NOT hundreds of "
            "independent lifetimes. Matched target pairs and memory reset "
            "controls are not statistically independent confirmations. "
            "All motor approaches are physically executed but GUIDE-scripted; "
            "terminal gate action alone is neural and reward-trained."
        ),
    }


def write_csv(p:Path,records:list[dict])->None:
    if not records:
        raise ValueError("Missing physical results")
    with p.open("w",encoding="utf-8",newline="") as f:
        wr=csv.DictWriter(f,fieldnames=list(records[0]))
        wr.writeheader()
        wr.writerows(records)


def save_source(bank_path:Path, curve_path:Path)->dict:
    bank,curves=train_source()
    bank_path.parent.mkdir(parents=True,exist_ok=True)
    curve_path.parent.mkdir(parents=True,exist_ok=True)
    bank_path.write_text(json.dumps(bank,indent=2,sort_keys=True)+"\n",
                         encoding="utf-8")
    write_csv(curve_path,curves)
    if load_bank(bank_path)["bank_sha256"]!=bank["bank_sha256"]:
        raise AssertionError("Frozen source write failed")
    if len(curves)!=len(INITIALS)*3*UPDATES:
        raise AssertionError("All original physical source episodes must survive")
    print("M4.4A FROZEN SOURCE "+
          json.dumps({"bank_sha256":bank["bank_sha256"],
                      "training_original_physical_episodes":len(curves),
                      "independent_initializations":len(INITIALS),
                      "test_worlds_used":0},sort_keys=True),flush=True)
    return bank


def save_final(bank_path:Path,out:Path, *, bootstrap_reps:int=BOOTSTRAP)->dict:
    if bootstrap_reps!=BOOTSTRAP:
        raise ValueError("No altered preregistered full final bootstrap dose")
    bank=load_bank(bank_path)
    settings=old.Settings()
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for split,seeds,shifted in (
        ("main",MAIN_WORLDS,False),("ood",OOD_WORLDS,True)):
        for source in bank["networks"]:
            for seed in seeds:
                for world in old.make_pair(seed,settings,shifted=shifted):
                    for arm in ARMS:
                        row=_evaluate_arm(world,arm,source,settings)
                        row["split"]=split
                        rows.append(row)
            print("M4.4A "+split.upper()+" FAMILY "+str(source["init_seed"]),
                  flush=True)
    final=evaluate(rows,bank)
    autonomous=[]
    for family in bank["networks"]:
        for seed in AUTONOMOUS_WORLDS:
            for world in old.make_pair(seed,settings):
                for name in ("online_rnn","initial_rnn"):
                    frozen=tuple(family[name])
                    r=old.simulate(world,settings,"evolved_rnn",frozen)
                    autonomous.append({
                        "init_seed":family["init_seed"],"world_seed":seed,
                        "target_side":world.side,"arm":name,
                        "fork_reached":r["fork_reached"],
                        "physical_food_eaten":r["success"],
                        "physically_autonomous_moves":r["steps"],
                        "end_energy":r["end_energy"],
                        "genome_sha256":family[name+"_sha256"],
                    })
    if len(rows)!=6144 or len(autonomous)!=256:
        raise AssertionError("Missing registered full original world evaluations")
    if any(r["fork_reached"] not in (0,1) for r in autonomous):
        raise AssertionError("Corrupt independent unguided navigation test")
    final.update({
        "protocol_commit":PROTOCOL_SHA,
        "frozen_original_training_bank_sha256":bank["bank_sha256"],
        "training_original_physical_episodes":len(INITIALS)*3*UPDATES,
        "final_original_guided_physical_world_trials":len(rows),
        "autonomous_original_physical_world_trials":len(autonomous),
        "bootstrap_replicates":BOOTSTRAP,
        "autonomous_diagnostic":{
            name:{
                "food_success_rate":statistics.mean(
                    r["physical_food_eaten"] for r in autonomous if r["arm"]==name),
                "reach_fork_rate":statistics.mean(
                    r["fork_reached"] for r in autonomous if r["arm"]==name),
            } for name in ("online_rnn","initial_rnn")
        },
    })
    write_csv(out/"M4_4A_all_6144_original_guided_final_world_trials.csv",rows)
    write_csv(out/"M4_4A_all_256_original_autonomous_transfer_trials.csv",
              autonomous)
    (out/"M4_4A_preregistered_results.json").write_text(
        json.dumps(final,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("M4.4A FINAL "+json.dumps({
        "gates":final["gates"],
        "main_arm_success_rates":{
            k:round(v["collected_food_rate"],6)
            for k,v in final["main"]["arms"].items()},
        "ood_arm_success_rates":{
            k:round(v["collected_food_rate"],6)
            for k,v in final["ood"]["arms"].items()},
        "autonomous":final["autonomous_diagnostic"],
    },sort_keys=True),flush=True)
    return final


def main()->None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase",choices=("source","final"),required=True)
    p.add_argument("--bank",type=Path,default=Path(
        "reports/M4_4A_FROZEN_ONLINE_SOURCE_BANK_2026-11-22.json"))
    p.add_argument("--curves",type=Path,default=Path(
        "reports/M4_4A_all_6912_original_rewarded_source_episodes_2026-11-22.csv"))
    p.add_argument("--out",type=Path,default=Path("results/m4-4a"))
    args=p.parse_args()
    if args.phase=="source":
        save_source(args.bank,args.curves)
    else:
        save_final(args.bank,args.out)


if __name__=="__main__":
    main()
