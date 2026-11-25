"""R15: external source-only slow teacher pretraining of the fast readout."""
import argparse
import gzip
import json
import random
from pathlib import Path
from neuroevolution import flyloop_r11 as r11, flyloop_r13 as r13, flyloop_r14 as r14

SEEDS=tuple(range(13300,13324))
VARIANTS=("fixed","true","shuffled","generic")
AUTH=(.5,1.)
CONDITIONS=(("slow_only",0.),)+tuple((v,h) for v in VARIANTS for h in AUTH)
EPOCHS=10

def source_seed(identity,geo,slot):
    return 140000+16*(identity-SEEDS[0])+8*r11.GEOMETRIES.index(geo)+slot

def source_model(world_seed):
    m=r13.isolated("original")
    m.SEEDS=SEEDS
    m.MACROS=m.TRAIN_MACROS
    parent=m.World
    class SourceWorld(parent):
        def __init__(self,seed,geometry,regime):
            super().__init__(seed,geometry,regime)
            import math
            self.initial_rectangles=(m.r1.procedural_rectangles(world_seed) if geometry=="obstacles" else m.r2.nearfield_rectangles(world_seed))
            self.rectangles=list(self.initial_rectangles)
            self.body=m.r1.Body(heading=random.Random(world_seed*11113+51).uniform(-math.pi,math.pi))
            self.visited={(int(self.body.x),int(self.body.y))}
    m.World=SourceWorld
    return m

def dataset(recoveries,safe,variant):
    if variant=="fixed" or not safe:return []
    labels=[r["motor"] for r in recoveries]
    result=[]
    for i,r in enumerate(recoveries):
        if variant=="true": j=i; label=labels[j]; origin=recoveries[j]["provenance"]
        elif variant=="shuffled":j=(i+1)%len(labels);label=labels[j];origin=recoveries[j]["provenance"]
        elif variant=="generic":j=i%len(safe);label=safe[j]["motor"];origin=safe[j]["provenance"]
        else:raise ValueError(variant)
        result.append({"hidden":r["hidden"],"label":label,"input_provenance":r["provenance"],"label_provenance":origin})
    return result

def pretrain(seed,data):
    f=r11.FastStudent(seed)
    for _ in range(EPOCHS):
        for item in data:f.learn_slow_command(item["hidden"],item["label"])
    return f.output,f.supervised_updates

def target_model(seed,weights,h):
    m=r14.model_for(h)
    m.SEEDS=SEEDS
    parent=m.FastStudent
    class Pretrained(parent):
        def __init__(self,s):
            super().__init__(s)
            if s!=seed:raise ValueError("identity mismatch")
            import copy
            self.output=copy.deepcopy(weights)
    m.FastStudent=Pretrained
    return m

def write(gz,row):
    gz.write((json.dumps(row,sort_keys=True,separators=(",",":"))+"\n").encode())

def bootstrap(d):
    if not d:return None
    rng=random.Random(133001)
    vals=sorted(sum(rng.choice(d) for _ in d)/len(d) for _ in range(5000))
    return [vals[125],vals[4874]]

def encoder(row):
    return r11.sha([row[k] for k in ("original_434_slow_coefficients_initial","fast_four_initial_input","fast_four_initial_recurrent","fast_four_initial_bias")])

def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    integrity={k:True for k in ("source_audit","target_audit","encoder_identity","disjoint_maps","paired_target_training","matched_pretraining","frozen_target_weights")}
    targets=[];records=[];source_hashes=set();target_hashes=set()
    with open(out/"lives.jsonl.gz","wb") as raw:
        with gzip.GzipFile(filename="",mode="wb",fileobj=raw,mtime=0) as gz:
            for seed in SEEDS:
                rec=[];safe=[];enc=set();worlds=[]
                for geo in r11.GEOMETRIES:
                    for slot in range(8):
                        ws=source_seed(seed,geo,slot)
                        m=source_model(ws)
                        row=m.simulate(seed,geo,"static","slow_only")
                        integrity["source_audit"] &= all(m.audit([row]).values())
                        enc.add(encoder(row))
                        source_hashes.add(r11.sha(row["initial_10_rectangles"]))
                        worlds.append(ws)
                        for e in row["real_successful_recovery_episodes"]:
                            rec.append({"hidden":e["first_post_contact_fast_hidden"],"motor":e["actual_first_recovery_motor"],"provenance":[ws,e["collision_micro"],e["first_recovery_motor_micro"]]})
                        for e in row["same_life_real_safe_macro_motors"]:
                            safe.append({"motor":e["motor"],"provenance":[ws,e["macro"]]})
                        row.update(phase="source",world_seed=ws);write(gz,row)
                data={v:dataset(rec,safe,v) for v in VARIANTS}
                weights={};updates={}
                for v in VARIANTS:weights[v],updates[v]=pretrain(seed,data[v])
                integrity["matched_pretraining"] &= len({updates[v] for v in VARIANTS[1:]})==1 and updates["fixed"]==0 and all(updates[v]==EPOCHS*len(data[v]) for v in VARIANTS)
                integrity["matched_pretraining"] &= all([x["hidden"] for x in data[v]]==[x["hidden"] for x in data["true"]] for v in VARIANTS[1:])
                record={"identity":seed,"source_worlds":worlds,"distinct_recoveries":len(rec),"usable_recoveries":len(data["true"]),"safe_examples":len(safe),"datasets":data,"weights":weights,"updates":updates}
                records.append({k:v for k,v in record.items() if k not in ("datasets","weights")})
                write(gz,{"phase":"pretraining",**record})
                for geo in r11.GEOMETRIES:
                    signatures=set()
                    for v,h in CONDITIONS:
                        w=weights["fixed" if v=="slow_only" else v]
                        m=target_model(seed,w,h)
                        arm="slow_only" if v=="slow_only" else "fast_fixed"
                        row=m.simulate(seed,geo,"static",arm)
                        integrity["target_audit"] &= all(m.audit([row]).values())
                        integrity["frozen_target_weights"] &= row["fast_10_output_coefficients_initial"]==w and row["fast_10_output_coefficients_after_training"]==w and row["supervised_fast_student_updates"]==0
                        enc.add(encoder(row));signatures.add(r13.trace_signature(row))
                        target_hashes.add(r11.sha(row["initial_10_rectangles"]))
                        row.update(phase="target",variant=v,authority=h,world_seed=seed)
                        targets.append(r13.compact(row));write(gz,row)
                    integrity["paired_target_training"] &= len(signatures)==1
                integrity["encoder_identity"] &= len(enc)==1
    integrity["disjoint_maps"] = not bool(source_hashes & target_hashes)
    def key(v,h):return v+"@"+str(h)
    summary={g:{key(v,h):r11.summarize([r for r in targets if (r["geometry"],r["variant"],r["authority"])==(g,v,h)]) for v,h in CONDITIONS} for g in r11.GEOMETRIES}
    n=summary["nearfield"];t=n[key("true",.5)];s=n[key("slow_only",0.)]
    den="collision_microframes_per_real_meter";dist="mean_postchange_meters"
    def beats(ref,ratio):return t[dist]>0 and ref[dist]>0 and ref[den]>0 and t[den]<=ratio*ref[den]
    gates={"source_coverage":sum(r["usable_recoveries"]>=10 for r in records)>=12,
           "beats_slow":beats(s,.9),
           **{"beats_"+v:beats(n[key(v,.5)],.85) for v in ("fixed","shuffled","generic")},
           "preserves_exploration":s[dist]>0 and t[dist]>=.9*s[dist] and t["mean_postchange_cells"]>=.9*s["mean_postchange_cells"],
           "repeat_reduction":s["actual_postchange_prior_contact_followups"]>=50 and t["actual_postchange_prior_contact_followups"]>0 and t["next_microstep_repeat_contact_fraction"]<=.9*s["next_microstep_repeat_contact_fraction"],
           "fast_response":t["physical_fast_override_fraction"]>=.1 and t["real_same_macro_contact_followup_opportunities"]>0 and t["same_macro_next_micro_fast_response_fraction"]>=.5}
    lookup={(r["geometry"],r["seed"],r["variant"],r["authority"]):r for r in targets}
    contrasts={}
    for v,h in (("slow_only",0.),("fixed",.5),("shuffled",.5),("generic",.5),("true",1.)):
        d=[]
        for seed in SEEDS:
            a=lookup["nearfield",seed,"true",.5];b=lookup["nearfield",seed,v,h]
            if a["postchange_actual_meters"]>0 and b["postchange_actual_meters"]>0:
                d.append(a["postchange_collision_microframes"]/a["postchange_actual_meters"]-b["postchange_collision_microframes"]/b["postchange_actual_meters"])
        contrasts[key(v,h)]={"paired_seeds":len(d),"mean_delta":sum(d)/len(d) if d else None,"bootstrap95":bootstrap(d)}
    report={"source_lives":sum(len(r["source_worlds"]) for r in records),"target_lives":len(targets),"source_records":records,"summary":summary,"integrity":integrity,"gates":gates,"contrasts":contrasts,"science_screen_pass":all(integrity.values()) and all(gates.values()),"autonomous_escape_demonstrated":False}
    (out/"summary.json").write_text(json.dumps(report,sort_keys=True,separators=(",",":"))+"\n")
    print("FLYLOOP_R15 "+json.dumps(report,sort_keys=True),flush=True)
    if not all(integrity.values()):raise AssertionError("R15 integrity failure")
    return report

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",default="results/flyloop_r15");run(p.parse_args().out)
