"""R16B: change training experience, not controller."""
import argparse
import json
import random
from pathlib import Path
from neuroevolution import flyloop_r16 as r, flyloop_r14 as m

SEEDS=range(19100,19124)
CURRICULA=("free","mixed","obstacle")
MODES=("gated","sham")
PHASES=("near","far","obstacle","masked")

def obstacle_training(curriculum,index):
    if curriculum not in CURRICULA:
        raise ValueError(curriculum)
    return curriculum=="obstacle" or (curriculum=="mixed" and index%2==1)

def simulate(seed,curriculum,train_n=80,test_n=20,horizon=80):
    critic=m.Critic()
    evidence=r.Evidence()
    training=[]
    for i in range(train_n):
        obstacle=obstacle_training(curriculum,i)
        e=r.episode(seed,i,"gated",critic,evidence,True,True,False,obstacle,horizon)
        e["obstacle_training"]=obstacle
        training.append(e)
    weights=[w[:] for w in critic.weights]
    counts=dict(evidence.data)
    evaluations={}
    for mode in MODES:
        evaluations[mode]={}
        for phase,offset,far,masked,obstacle in (
            ("near",100,False,False,False),
            ("far",200,True,False,False),
            ("obstacle",300,True,False,True),
            ("masked",300,True,True,True)):
            evaluations[mode][phase]=[r.episode(seed,offset+i,mode,critic,evidence,False,far,masked,obstacle,horizon) for i in range(test_n)]
        assert weights==critic.weights and counts==evidence.data
    return dict(seed=seed,curriculum=curriculum,training=training,
                weights=weights,evidence=[(k,v) for k,v in sorted(counts.items())],
                evaluations=evaluations)

def aggregate(rows,mode,phase):
    return r.aggregate([{phase:row["evaluations"][mode][phase]} for row in rows],phase)

def summarize(rows):
    summary={}
    for curriculum in CURRICULA:
        subset=[x for x in rows if x["curriculum"]==curriculum]
        summary[curriculum]={
            "training":r.aggregate(subset,"training"),
            "obstacle_training_episodes":sum(e["obstacle_training"] for row in subset for e in row["training"]),
            "evidence_bins":sum(len(row["evidence"]) for row in subset),
            **{mode:{p:aggregate(subset,mode,p) for p in PHASES} for mode in MODES}}
    return summary

def run():
    rows=[simulate(seed,c) for seed in SEEDS for c in CURRICULA]
    summary=summarize(rows)
    comparisons=(
        ("mixed_vs_free_obstacle","mixed","gated","obstacle","free","gated","obstacle",.05),
        ("mixed_vs_sham_obstacle","mixed","gated","obstacle","mixed","sham","obstacle",.05),
        ("retain_near","mixed","gated","near","mixed","sham","near",-.05),
        ("retain_far","mixed","gated","far","mixed","sham","far",-.05))
    gates={}
    intervals={}
    by={(x["seed"],x["curriculum"]):x for x in rows}
    for name,c,m1,p,d,m2,q,threshold in comparisons:
        gates[name]=summary[c][m1][p]["encounter"]-summary[d][m2][q]["encounter"]>=threshold
        delta=[aggregate([by[s,c]],m1,p)["encounter"]-aggregate([by[s,d]],m2,q)["encounter"] for s in SEEDS]
        rng=random.Random(191919)
        values=sorted(sum(rng.choice(delta) for _ in delta)/len(delta) for _ in range(2000))
        intervals[name]=[values[50],values[1949]]
    gates["nontrivial_interventions"]=summary["mixed"]["gated"]["obstacle"]["intervention_fraction"]>=.01
    return dict(summary=summary,gates=gates,passed=all(gates.values()),intervals=intervals,rows=rows)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",default="results/r16b.json")
    args=p.parse_args()
    result=run()
    path=Path(args.out)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,sort_keys=True,separators=(",",":")))
    print(json.dumps({k:v for k,v in result.items() if k!="rows"},indent=2))
