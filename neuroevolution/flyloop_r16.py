"""R16: isolated R15 input-timestamp intervention; no direct fast reward."""
import argparse
import gzip
import hashlib
import importlib.util
import json
import random
from pathlib import Path
from neuroevolution import flyloop_r15 as r15, flyloop_r11 as r11, flyloop_r13 as r13

SEEDS=tuple(range(13600,13624))
MODES=("early","aligned")
def source_seed(identity,geo,slot):
    return 150000+16*(identity-SEEDS[0])+8*r11.GEOMETRIES.index(geo)+slot

def make_model(mode):
    if mode not in MODES:raise ValueError(mode)
    spec=importlib.util.spec_from_file_location("r16_"+mode,r15.__file__)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    m.SEEDS=SEEDS;m.source_seed=source_seed
    original_source=m.source_model
    original_dataset=m.dataset
    states={};lags={}
    def source(ws):
        model=original_source(ws);simulate=model.simulate
        def capture(*args,**kwargs):
            row=simulate(*args,**kwargs)
            for e in row["real_successful_recovery_episodes"]:
                first=e["first_post_contact_micro"];later=e["first_recovery_motor_micro"]
                frame=row["raw_micro_trace"][later]
                assert frame["micro_tick"]==later and later>=first
                assert frame["actually_executed_analog_motor"]==e["actual_first_recovery_motor"]
                k=(ws,e["collision_micro"],later)
                states[k]=frame["four_fast_recurrent_states"]
                lags[k]=later-first
            return row
        model.simulate=capture
        return model
    def dataset(rec,safe,variant):
        if mode=="aligned":
            rec=[{**x,"hidden":states[tuple(x["provenance"])]} for x in rec]
        data=original_dataset(rec,safe,variant)
        for x in data:
            k=tuple(x["input_provenance"])
            x["input_timestamp_mode"]=mode
            x["input_micro_tick"]=k[2] if mode=="aligned" else k[1]+1
        return data
    m.source_model=source;m.dataset=dataset
    return m,lags

def digest(obj):return r11.sha(obj)
def audit_pair(early,aligned):
    def scan(path):
        sources={};targets={};datasets={};control={};enc={}
        with gzip.open(path,"rt") as f:
            for line in f:
                r=json.loads(line)
                if r["phase"]=="source":
                    sources[r["world_seed"]]=digest(r)
                elif r["phase"]=="target":
                    k=(r["seed"],r["geometry"],r["variant"],r["authority"])
                    targets[k]=r13.trace_signature(r)
                    enc[k]=digest([r["initial_10_rectangles"],r["initial_body_pose"],r15.encoder(r)])
                    if r["variant"] in ("slow_only","fixed"):control[k]=digest(r)
                else:
                    datasets[r["identity"]]={v:[(d["label"],d["input_provenance"],d["label_provenance"],d["input_micro_tick"],d["hidden"]) for d in ds] for v,ds in r["datasets"].items()}
        return sources,targets,enc,control,datasets
    a,b=scan(early),scan(aligned)
    labels=True;changed=0;distinct=0;hidden_changed=0
    if a[4].keys()!=b[4].keys():labels=False
    for seed in a[4].keys() & b[4].keys():
        for v in r15.VARIANTS:
            x,y=a[4][seed][v],b[4][seed][v]
            labels &= len(x)==len(y) and all(i[:3]==j[:3] for i,j in zip(x,y))
            if v=="true":
                distinct+=len(x);changed+=sum(i[3]!=j[3] for i,j in zip(x,y))
                hidden_changed+=sum(i[4]!=j[4] for i,j in zip(x,y))
    return {"source_exact":a[0]==b[0],"target_training_exact":a[1]==b[1],"target_maps_and_encoders_exact":a[2]==b[2],"fixed_and_slow_exact":a[3]==b[3],"labels_and_counts_exact":labels}, {"usable_distinct_examples":distinct,"timestamps_changed":changed,"hidden_vectors_changed":hidden_changed}

def paired_deltas(a,b):
    def scan(path):
        rows={}
        with gzip.open(path,"rt") as f:
            for line in f:
                r=json.loads(line)
                if r["phase"]=="target" and (r["geometry"],r["variant"],r["authority"])==("nearfield","true",.5):
                    if r["postchange_actual_meters"]>0:rows[r["seed"]]=r["postchange_collision_microframes"]/r["postchange_actual_meters"]
        return rows
    x,y=scan(a),scan(b)
    d=[y[s]-x[s] for s in SEEDS if s in x and s in y]
    rng=random.Random(136001)
    draws=sorted(sum(rng.choice(d) for _ in d)/len(d) for _ in range(5000)) if d else []
    return {"paired_seeds":len(d),"mean_delta":sum(d)/len(d) if d else None,"bootstrap95":[draws[125],draws[4874]] if draws else None}

def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    results={};lagsets={}
    for mode in MODES:
        m,lags=make_model(mode)
        results[mode]=m.run(out/mode)
        lagsets[mode]=dict(lags)
    early=out/"early"/"lives.jsonl.gz";aligned=out/"aligned"/"lives.jsonl.gz"
    integrity,changed=audit_pair(early,aligned)
    integrity["lag_provenance_exact"]=lagsets["early"]==lagsets["aligned"]
    integrity["both_r15_integrity"]=all(all(x["integrity"].values()) for x in results.values())
    counts={}
    for lag in lagsets["aligned"].values():counts[str(lag)]=counts.get(str(lag),0)+1
    a=results["early"]["summary"]["nearfield"]["true@0.5"]
    b=results["aligned"]["summary"]["nearfield"]["true@0.5"]
    den="collision_microframes_per_real_meter";dist="mean_postchange_meters"
    gates=dict(results["aligned"]["gates"])
    gates["beats_early"]=b[dist]>0 and a[dist]>0 and a[den]>0 and b[den]<=.9*a[den]
    report={"modes":results,"integrity":integrity,"input_alignment":changed,"source_lag_microstep_histogram":counts,"primary_contrast":paired_deltas(early,aligned),"gates":gates,"science_screen_pass":all(integrity.values()) and all(gates.values()),"autonomous_escape_demonstrated":False}
    (out/"summary.json").write_text(json.dumps(report,sort_keys=True,separators=(",",":"))+"\n")
    print("FLYLOOP_R16 "+json.dumps({k:v for k,v in report.items() if k!="modes"},sort_keys=True),flush=True)
    if not all(integrity.values()):raise AssertionError("R16 integrity failure")
    return report
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",default="results/flyloop_r16");run(p.parse_args().out)
