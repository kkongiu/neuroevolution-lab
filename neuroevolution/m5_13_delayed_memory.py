"""M5.13: delay-dependent retention after synapse pruning.

New development seeds 3600..3619; unlike M5.12 no test seeds reused.
Learn A (160 trials), then B/C interference for 0/160/480/960 trials;
finally score A for 40 trials WITHOUT reward-based weight updates or
structural updates. Success feedback within probe still appears as sensory
input (identical for all arms); network never gets correct action labels.

Compare shared 8-hidden-unit/260-slot native RNN, initially 200 active
edges, four arms: no prune, uniform random pruning, fast local usage and
50:50 fast+slow coactivation. All pruning arms remove exactly four edges
every 40 lifetime actions (until 120 remain). Fixed context sensors but no
context-specific neural routing. The initial weights, mask and task/cue
sequences are matched within each seed/delay. This is NOT blinded context
recognition, a biological brain or an evolutionary experiment.

Potential limitation by design: shorter delays have undergone fewer pruning
events (a realistic exposure effect), and the return probe is on a previously
known *visible context* rather than completely novel task transfer.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import statistics
from pathlib import Path
from .m5_6_online import Learner
from .m5_12_usage_pruning import sample_prunes, observation, TOTAL_EDGES, INITIAL_EDGES, MIN_EDGES, PRUNE_PERIOD, REMOVALS_PER_EVENT, FAST_DECAY, SLOW_DECAY, PROTECTION_FRACTION, LEARNING_RATE

SEEDS=tuple(range(3600,3620))
DELAYS=(0,160,480,960)
ARMS=("static","random","recent_usage","historical_usage")
ACQUISITION=160
PROBE=40

def genome(seed):
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError("Only separate M5.13 development seeds")
    rng=random.Random(seed*107_113+17)
    return tuple(rng.gauss(0.,.26) for _ in range(TOTAL_EDGES))

def mask(seed):
    return set(random.Random(seed*17_033+227).sample(
        range(TOTAL_EDGES),INITIAL_EDGES))

def rule_bits(seed):
    if seed not in SEEDS: raise ValueError("Unregistered seed")
    return tuple(random.Random(seed*10_007+51+c*113).randrange(2)
                 for c in range(3))

def curriculum(seed,delay):
    if seed not in SEEDS or delay not in DELAYS:
        raise ValueError("Unregistered seed/delay")
    rng=random.Random(seed*211_001+271)
    def segment(context,length):
        result=[]
        for _ in range(length//40):
            v=[0]*20+[1]*20
            rng.shuffle(v)
            result.extend((context,cue) for cue in v)
        return result
    # Fixed cyclic B,C block order as exposure grows.
    experience=segment(0,ACQUISITION)
    for chunk in range(delay//160):
        experience.extend(segment(1+chunk%2,160))
    test=segment(0,PROBE)
    return experience,test

class SynapticOrganism:
    def __init__(self,seed,arm):
        self.active=mask(seed)
        self.agent=Learner.fresh(genome(seed),(5,5))
        self.fast=[0.]*TOTAL_EDGES
        self.slow=[0.]*TOTAL_EDGES
        self.rng=random.Random(seed*97_337+783)
        self.events=[]
        self._mask()

    def _mask(self):
        for i in range(TOTAL_EDGES):
            if i not in self.active:
                self.agent.weights[i]=0.

    def learn(self,cache,success):
        obs,prev,hidden,probs,selected=cache
        pre=list(obs)+list(prev)+[1.]
        for i in self.active:
            if i<224:
                usage=abs(pre[i%28])*abs(hidden[i//28])
            else:
                a=(i-224)//9
                j=(i-224)%9
                usage=(abs(hidden[j]) if j<8 else 1.)*probs.get(a,0.)
            self.fast[i]=FAST_DECAY*self.fast[i]+(1-FAST_DECAY)*usage
            self.slow[i]=SLOW_DECAY*self.slow[i]+(1-SLOW_DECAY)*usage
        self.agent.learn(cache,1. if success else -.15,
                         learning_rate=LEARNING_RATE)
        self._mask()

    def maybe_prune(self,tick,arm):
        if arm=="static" or (tick+1)%PRUNE_PERIOD: return
        n=min(REMOVALS_PER_EVENT,len(self.active)-MIN_EDGES)
        if not n:return
        if arm=="random":
            indices=self.rng.sample(sorted(self.active),n)
        elif arm=="recent_usage":
            indices=sample_prunes(self.active,self.fast,n,self.rng)
        elif arm=="historical_usage":
            importance=[(1-PROTECTION_FRACTION)*a+PROTECTION_FRACTION*b
                        for a,b in zip(self.fast,self.slow)]
            indices=sample_prunes(self.active,importance,n,self.rng)
        else: raise AssertionError("Invalid arm")
        for i in indices:
            self.active.remove(i)
            self.agent.weights[i]=0.
            self.fast[i]=0.
            self.slow[i]=0.
        self.events.append((tick+1,n,len(self.active)))

def run(seed,delay,arm):
    if seed not in SEEDS or delay not in DELAYS or arm not in ARMS:
        raise ValueError("Registered M5.13 seed/delay/arm required")
    env_rules=rule_bits(seed)
    train,probe=curriculum(seed,delay)
    org=SynapticOrganism(seed,arm)
    previous_context=None
    last_feedback=0
    acquisition=[]
    allocated=0
    for tick,(context,cue) in enumerate(train):
        if context!=previous_context:
            last_feedback=0
            previous_context=context
        decision,cache=org.agent.action(
            observation(context,cue,last_feedback),(0,1),
            random.Random(seed*997_007+tick*1069+21))
        success=int(decision==(cue^env_rules[context]))
        if tick<ACQUISITION: acquisition.append(success)
        org.learn(cache,success)
        allocated+=len(org.active)
        org.maybe_prune(tick,arm)
        last_feedback=success
    # Measure *immediate* retained competence rather than learning within
    # the return phase: no synaptic change or pruning during the probe.
    last_feedback=0
    outcomes=[]
    before_weights=tuple(org.agent.weights)
    before_active=frozenset(org.active)
    for j,(context,cue) in enumerate(probe):
        decision,_=org.agent.action(
            observation(context,cue,last_feedback),(0,1),
            random.Random(seed*997_007+(len(train)+j)*1069+21))
        success=int(decision==(cue^env_rules[context]))
        outcomes.append(success)
        last_feedback=success
    if before_weights!=tuple(org.agent.weights) or before_active!=frozenset(org.active):
        raise AssertionError("Frozen probe mutated synaptic state")
    return {
        "seed":seed,"delay":delay,"arm":arm,
        "acquisition_late":sum(acquisition[-40:])/40,
        "frozen_probe_accuracy":sum(outcomes)/len(outcomes),
        "retention_delta":sum(outcomes)/len(outcomes)-sum(acquisition[-40:])/40,
        "train_steps":len(train),"frozen_probe_steps":len(probe),
        "mean_active_during_training":allocated/len(train),
        "final_active":len(org.active),
        "prune_events":org.events,
        "final_edges_digest":hashlib.sha256(
            json.dumps(sorted(org.active)).encode()).hexdigest(),
        "frozen_probe":True,"explicit_context_router":False,
        "fitness_selection":False,
    }

def experiment(seeds=SEEDS):
    if not seeds or len(set(seeds))!=len(seeds) or any(
        type(s) is not int or s not in SEEDS for s in seeds):
        raise ValueError("Unique M5.13 development seeds only")
    rows=[run(s,d,a) for d in DELAYS for s in seeds for a in ARMS]
    index={(r["seed"],r["delay"],r["arm"]):r for r in rows}
    for d in DELAYS:
        for s in seeds:
            timing={tuple(index[(s,d,a)]["prune_events"])
                    for a in ARMS if a!="static"}
            if len(timing)!=1: raise AssertionError(
                "Unequal number/timing of pruned connections")
    summary={}
    for arm in ARMS:
        summary[arm]={}
        for delay in DELAYS:
            group=[index[(s,delay,arm)] for s in seeds]
            summary[arm][str(delay)]={
                k:statistics.mean(x[k] for x in group)
                for k in ("acquisition_late","frozen_probe_accuracy",
                          "retention_delta","mean_active_during_training",
                          "final_active")}
    longest=str(DELAYS[-1])
    historical=summary["historical_usage"][longest]
    recent=summary["recent_usage"][longest]
    random_control=summary["random"][longest]
    static=summary["static"][longest]
    gate={
        "long_delay_history_exceeds_recent_by_5pp":
            historical["frozen_probe_accuracy"]-
            recent["frozen_probe_accuracy"]>=.05,
        "long_delay_history_exceeds_random_by_5pp":
            historical["frozen_probe_accuracy"]-
            random_control["frozen_probe_accuracy"]>=.05,
        "long_delay_history_reaches_65pct":
            historical["frozen_probe_accuracy"]>=.65,
        "long_delay_history_95pct_of_static":
            historical["frozen_probe_accuracy"]>=
            .95*static["frozen_probe_accuracy"],
        "fewer_synapses_than_static":
            historical["mean_active_during_training"]<
            static["mean_active_during_training"],
    }
    return {
        "status":"M5.13_development_long_delayed_recall_not_final",
        "seeds":list(seeds),"delays":list(DELAYS),"arms":list(ARMS),
        "acquisition_trials":ACQUISITION,"frozen_probe_trials":PROBE,
        "pruning_dose_timing_matched":True,
        "no_new_synapse_or_neurons":True,
        "probe_synapses_frozen":True,"task_context_observable":True,
        "summary":summary,"screening_gate":gate,
        "advance_to_evolution":all(gate.values()),"raw_rows":rows,
        "rows_sha256":hashlib.sha256(json.dumps(
            rows,sort_keys=True,separators=(",",":")).encode()).hexdigest(),
    }

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,default=Path(
        "results/m5-13/delayed-retention-development.json"))
    args=parser.parse_args()
    r=experiment()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(r,indent=2,sort_keys=True)+"\n",
                        encoding="utf-8")
    print("M5_13_DELAYED_RETENTION "+json.dumps({
        k:r[k] for k in ("status","summary","screening_gate",
                        "advance_to_evolution","rows_sha256")},
        sort_keys=True),flush=True)

if __name__=="__main__":
    main()
