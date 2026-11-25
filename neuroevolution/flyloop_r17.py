"""R17: general safe movement before recovery specialization."""
import argparse
import gzip
import json
import math
import random
from pathlib import Path
from neuroevolution import flyloop_r11 as r11, flyloop_r15 as r15, flyloop_r13 as r13

SEEDS=tuple(range(13900,13924))
VARIANTS=("fixed","general","recovery","curriculum","curriculum_replay","general_matched")
CONDITIONS=(("slow_only",0.),)+tuple((v,h) for v in VARIANTS for h in (.5,1.))
EPOCHS=10

def source_seed(s,g,i):
    return 160000+16*(s-SEEDS[0])+8*r11.GEOMETRIES.index(g)+i

def source_model(ws):
    m=r15.source_model(ws)
    m.SEEDS=SEEDS
    return m

def target_model(s,w,h):
    m=r15.target_model(s,w,h)
    m.SEEDS=SEEDS
    return m

def select_general(safe):
    held=[x for i,x in enumerate(safe) if i%5==0]
    pool=[x for i,x in enumerate(safe) if i%5!=0]
    if len(pool)>128:
        pool=[pool[i*len(pool)//128] for i in range(128)]
    return pool,held

def mse(f,examples):
    if not examples:return None
    return sum((math.tanh(sum(a*b for a,b in zip(f.output[k],tuple(e["hidden"])+(1.,))))-e["motor"][k])**2 for e in examples for k in (0,1))/(2*len(examples))

def train(seed,general,recovery,held,variant):
    f=r11.FastStudent(seed)
    def step(e):f.learn_slow_command(e["hidden"],e["motor"])
    initial=mse(f,held)
    if variant in ("general","curriculum","curriculum_replay","general_matched"):
        for _ in range(EPOCHS):
            for e in general:step(e)
    prefix=r11.sha(f.output)
    before=mse(f,held)
    if variant in ("recovery","curriculum","curriculum_replay"):
        i=0
        for _ in range(EPOCHS):
            for e in recovery:
                step(e)
                if variant=="curriculum_replay" and general:step(general[i%len(general)])
                i+=1
    elif variant=="general_matched" and general:
        for i in range(EPOCHS*len(recovery)):step(general[i%len(general)])
    return {"weights":f.output,"updates":f.supervised_updates,"general_prefix":prefix,
            "initial_general_mse":initial,"before_specialization_mse":before,"final_general_mse":mse(f,held)}

def onset_count(row):
    prev=None
    n=0
    for f in row["raw_micro_trace"]:
        if f["macro_tick"]<r11.TRAIN_MACROS:continue
        hit=f["actual_physical_body_after_micro"]["collision"]
        if prev==0 and hit:n+=1
        prev=hit
    return n

def bootstrap(d):
    if not d:return None
    rng=random.Random(139001)
    x=sorted(sum(rng.choice(d) for _ in d)/len(d) for _ in range(5000))
    return [x[125],x[4874]]

def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    checks={k:True for k in ("source_audit","target_audit","same_encoder","disjoint_maps","paired_target_training","frozen_fast","matched_updates","identical_general_prefix","source_provenance")}
    sources=set();targets=set();rows=[];records=[]
    with open(out/"lives.jsonl.gz","wb") as raw:
        with gzip.GzipFile(filename="",mode="wb",fileobj=raw,mtime=0) as gz:
            for seed in SEEDS:
                safe=[];recovery=[];enc=set();worlds=[]
                for geo in r11.GEOMETRIES:
                    for i in range(8):
                        ws=source_seed(seed,geo,i);worlds.append(ws)
                        m=source_model(ws);r=m.simulate(seed,geo,"static","slow_only")
                        checks["source_audit"] &= all(m.audit([r]).values())
                        sources.add(r11.sha(r["initial_10_rectangles"]));enc.add(r15.encoder(r))
                        for e in r["same_life_real_safe_macro_motors"]:
                            tick=e["macro"]*r11.SUBSTEPS
                            f=r["raw_micro_trace"][tick]
                            checks["source_provenance"] &= f["micro_tick"]==tick and f["actually_executed_analog_motor"]==e["motor"]
                            safe.append({"hidden":f["four_fast_recurrent_states"],"motor":e["motor"],"provenance":[ws,tick]})
                        for e in r["real_successful_recovery_episodes"]:
                            recovery.append({"hidden":e["first_post_contact_fast_hidden"],"motor":e["actual_first_recovery_motor"],"provenance":[ws,e["first_post_contact_micro"],e["first_recovery_motor_micro"]]})
                        r15.write(gz,{**r,"phase":"source","world_seed":ws})
                general,held=select_general(safe)
                fit={v:train(seed,general,recovery,held,v) for v in VARIANTS}
                checks["matched_updates"] &= fit["curriculum"]["updates"]==fit["general_matched"]["updates"]
                checks["identical_general_prefix"] &= len({fit[v]["general_prefix"] for v in ("general","curriculum","curriculum_replay","general_matched")})==1
                record={"identity":seed,"source_worlds":worlds,"general_examples":len(general),"heldout_general":len(held),"recovery_examples":len(recovery),"fit":fit}
                records.append(record)
                r15.write(gz,{**record,"phase":"pretraining","general":general,"heldout":held,"recovery":recovery})
                for geo in r11.GEOMETRIES:
                    sig=set()
                    for v,h in CONDITIONS:
                        w=fit["fixed" if v=="slow_only" else v]["weights"]
                        m=target_model(seed,w,h)
                        r=m.simulate(seed,geo,"static","slow_only" if v=="slow_only" else "fast_fixed")
                        checks["target_audit"] &= all(m.audit([r]).values())
                        checks["frozen_fast"] &= r["fast_10_output_coefficients_initial"]==w and r["fast_10_output_coefficients_after_training"]==w and r["supervised_fast_student_updates"]==0
                        enc.add(r15.encoder(r));sig.add(r13.trace_signature(r));targets.add(r11.sha(r["initial_10_rectangles"]))
                        r.update(phase="target",variant=v,authority=h,contact_onsets=onset_count(r))
                        rows.append(r13.compact(r));r15.write(gz,r)
                    checks["paired_target_training"] &= len(sig)==1
                checks["same_encoder"] &= len(enc)==1
    checks["disjoint_maps"]=not bool(sources&targets)
    def key(v,h):return v+"@"+str(h)
    def summarize(group):
        x=r11.summarize(group)
        meters=sum(r["postchange_actual_meters"] for r in group)
        x["contact_onsets"]=sum(r["contact_onsets"] for r in group)
        x["contact_onsets_per_meter"]=x["contact_onsets"]/meters if meters>0 else None
        x["total_meters"]=meters
        return x
    summary={g:{key(v,h):summarize([r for r in rows if (r["geometry"],r["variant"],r["authority"])==(g,v,h)]) for v,h in CONDITIONS} for g in r11.GEOMETRIES}
    n=summary["nearfield"];s=n[key("slow_only",0.)];a=n[key("general",.5)];c=n[key("curriculum",.5)]
    den="collision_microframes_per_real_meter";dist="mean_postchange_meters"
    def beats(x,y,ratio):
        return x[dist]>0 and y[dist]>0 and y[den]>0 and x[den]<=ratio*y[den]
    def retains(x,y):
        return y[dist]>0 and x[dist]>=.9*y[dist] and x["mean_postchange_cells"]>=.9*y["mean_postchange_cells"]
    gates={"general_beats_slow":beats(a,s,.9),"general_beats_fixed":beats(a,n[key("fixed",.5)],.85),
           "curriculum_beats_general":beats(c,a,.9),"curriculum_beats_recovery":beats(c,n[key("recovery",.5)],.85),
           "curriculum_beats_general_matched":beats(c,n[key("general_matched",.5)],.9),
           "retains_exploration":retains(c,a) and retains(c,s),
           "prevents_contact_onsets":a["contact_onsets_per_meter"] is not None and a["contact_onsets_per_meter"]>0 and c["contact_onsets_per_meter"] is not None and c["contact_onsets_per_meter"]<=.9*a["contact_onsets_per_meter"],
           "source_coverage":sum(r["recovery_examples"]>=10 and r["general_examples"]>=64 for r in records)>=12,
           "fast_response":c["physical_fast_override_fraction"]>=.1 and c["real_same_macro_contact_followup_opportunities"]>0 and c["same_macro_next_micro_fast_response_fraction"]>=.5}
    lookup={(r["geometry"],r["seed"],r["variant"],r["authority"]):r for r in rows}
    contrasts={}
    for v,w,h in (("general","slow_only",0.),("general","fixed",.5),("curriculum","general",.5),("curriculum","recovery",.5),("curriculum","general_matched",.5)):
        d=[]
        for seed in SEEDS:
            x=lookup["nearfield",seed,v,.5];y=lookup["nearfield",seed,w,h]
            if x["postchange_actual_meters"]>0 and y["postchange_actual_meters"]>0:
                d.append(x["postchange_collision_microframes"]/x["postchange_actual_meters"]-y["postchange_collision_microframes"]/y["postchange_actual_meters"])
        contrasts[v+"_vs_"+w]={"paired_seeds":len(d),"mean_delta":sum(d)/len(d) if d else None,"bootstrap95":bootstrap(d)}
    report={"source_lives":sum(len(r["source_worlds"]) for r in records),"target_lives":len(rows),"records":records,"summary":summary,"integrity":checks,"gates":gates,"contrasts":contrasts,"science_screen_pass":all(checks.values()) and all(gates.values()),"autonomous_escape_demonstrated":False}
    (out/"summary.json").write_text(json.dumps(report,sort_keys=True,separators=(",",":"))+"\n")
    print("FLYLOOP_R17 "+json.dumps(report,sort_keys=True),flush=True)
    if not all(checks.values()):raise AssertionError("R17 integrity failure")
    return report
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",default="results/flyloop_r17");run(p.parse_args().out)
