"""R18: independently replicate recovery-only pretraining at full authority."""
import argparse
import gzip
import json
import random
from pathlib import Path
from neuroevolution import flyloop_r11 as r11, flyloop_r15 as r15, flyloop_r13 as r13, flyloop_r17 as r17

SEEDS=tuple(range(14200,14232))
DOSES=(16,32)
VARIANTS=("true","shuffled","generic")
ARMS=("slow_only","fixed")+tuple(str(d)+"_"+v for d in DOSES for v in VARIANTS)

def source_seed(s,g,i):
    return 170000+32*(s-SEEDS[0])+16*r11.GEOMETRIES.index(g)+i

def source_model(ws):
    m=r15.source_model(ws);m.SEEDS=SEEDS;return m

def target_model(s,w):
    m=r15.target_model(s,w,1.);m.SEEDS=SEEDS;return m

def pool(sources,dose):
    selected=[x for x in sources if x["slot"]<dose//2]
    return ([e for x in selected for e in x["recovery"]],
            [e for x in selected for e in x["safe"]],
            [x["world_seed"] for x in selected])

def ratio(rows):
    d=sum(r["postchange_actual_meters"] for r in rows)
    return sum(r["postchange_collision_microframes"] for r in rows)/d if d>0 else None

def paired_bootstrap(a,b):
    assert len(a)==len(b)==len(SEEDS)
    rng=random.Random(142001);draws=[];invalid=0
    for _ in range(5000):
        ix=[rng.randrange(len(a)) for _ in a]
        x=ratio([a[i] for i in ix]);y=ratio([b[i] for i in ix])
        if x is None or y is None:invalid+=1
        else:draws.append(x-y)
    draws.sort()
    x,y=ratio(a),ratio(b)
    return {"paired_seeds":len(a),"invalid_draws":invalid,
            "observed_delta":x-y if x is not None and y is not None else None,
            "bootstrap95":[draws[int(.025*len(draws))],draws[min(len(draws)-1,int(.975*len(draws)))]] if draws else None}

def summarize(rows):
    x=r11.summarize(rows)
    d=sum(r["postchange_actual_meters"] for r in rows)
    n=sum(r["contact_onsets"] for r in rows)
    x.update(total_meters=d,contact_onsets=n,contact_onsets_per_meter=n/d if d>0 else None)
    return x

def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    checks={k:True for k in ("source_audit","target_audit","encoder_identity","disjoint_maps","paired_training","frozen_fast","matched_datasets","nested_sources","source_provenance")}
    source_maps=set();target_maps=set();rows=[];records=[]
    with open(out/"lives.jsonl.gz","wb") as raw:
        with gzip.GzipFile(filename="",mode="wb",fileobj=raw,mtime=0) as gz:
            for s in SEEDS:
                sources=[];enc=set()
                for g in r11.GEOMETRIES:
                    for i in range(16):
                        ws=source_seed(s,g,i);m=source_model(ws)
                        r=m.simulate(s,g,"static","slow_only")
                        checks["source_audit"] &= all(m.audit([r]).values())
                        enc.add(r15.encoder(r));source_maps.add(r11.sha(r["initial_10_rectangles"]))
                        rec=[];safe=[]
                        for e in r["real_successful_recovery_episodes"]:
                            f=r["raw_micro_trace"][e["first_post_contact_micro"]]
                            q=r["raw_micro_trace"][e["first_recovery_motor_micro"]]
                            checks["source_provenance"] &= f["four_fast_recurrent_states"]==[round(v,9) for v in e["first_post_contact_fast_hidden"]] and q["actually_executed_analog_motor"]==e["actual_first_recovery_motor"]
                            rec.append({"hidden":e["first_post_contact_fast_hidden"],"motor":e["actual_first_recovery_motor"],"provenance":[ws,e["collision_micro"],e["first_recovery_motor_micro"]]})
                        for e in r["same_life_real_safe_macro_motors"]:
                            safe.append({"motor":e["motor"],"provenance":[ws,e["macro"]]})
                        sources.append({"slot":i,"world_seed":ws,"recovery":rec,"safe":safe})
                        r15.write(gz,{**r,"phase":"source","world_seed":ws})
                weights={"fixed":r15.pretrain(s,[])[0]};record={"identity":s,"doses":{}}
                datasets={};worldsets={};recs={}
                for d in DOSES:
                    rec,safe,worlds=pool(sources,d);worldsets[d]=set(worlds);recs[d]=set(tuple(e["provenance"]) for e in rec)
                    data={v:r15.dataset(rec,safe,v) for v in VARIANTS}
                    updates={}
                    for v in VARIANTS:
                        k=str(d)+"_"+v;datasets[k]=data[v]
                        weights[k],updates[v]=r15.pretrain(s,data[v])
                    checks["matched_datasets"] &= len({updates[v] for v in VARIANTS})==1 and all(updates[v]==10*len(data[v]) for v in VARIANTS) and all([x["hidden"] for x in data[v]]==[x["hidden"] for x in data["true"]] for v in VARIANTS)
                    record["doses"][str(d)]={"world_count":len(worlds),"recoveries":len(rec),"usable":len(data["true"]),"distinct_labels":len({tuple(e["label"]) for e in data["true"]}),"safe":len(safe),"updates":updates}
                checks["nested_sources"] &= worldsets[16]<worldsets[32] and len(worldsets[16])==16 and len(worldsets[32])==32 and recs[16]<=recs[32]
                records.append(record);r15.write(gz,{**record,"phase":"pretraining","datasets":datasets,"weights":weights,"source_worlds":{str(d):sorted(worldsets[d]) for d in DOSES}})
                for g in r11.GEOMETRIES:
                    sig=set()
                    for arm in ARMS:
                        w=weights["fixed" if arm=="slow_only" else arm]
                        m=target_model(s,w);r=m.simulate(s,g,"static","slow_only" if arm=="slow_only" else "fast_fixed")
                        checks["target_audit"] &= all(m.audit([r]).values())
                        checks["frozen_fast"] &= r["fast_10_output_coefficients_initial"]==w and r["fast_10_output_coefficients_after_training"]==w and r["supervised_fast_student_updates"]==0
                        enc.add(r15.encoder(r));sig.add(r13.trace_signature(r));target_maps.add(r11.sha(r["initial_10_rectangles"]))
                        r.update(phase="target",variant=arm,contact_onsets=r17.onset_count(r));rows.append(r13.compact(r));r15.write(gz,r)
                    checks["paired_training"] &= len(sig)==1
                checks["encoder_identity"] &= len(enc)==1
    checks["disjoint_maps"]=not bool(source_maps&target_maps)
    lookup={(r["geometry"],r["variant"],r["seed"]):r for r in rows}
    def group(g,a):return [lookup[g,a,s] for s in SEEDS]
    summary={g:{a:summarize(group(g,a)) for a in ARMS} for g in r11.GEOMETRIES}
    contrasts={g:{a:paired_bootstrap(group(g,"32_true"),group(g,a)) for a in ARMS if a!="32_true"} for g in r11.GEOMETRIES}
    n=summary["nearfield"];o=summary["obstacles"];t=n["32_true"];s=n["slow_only"]
    den="collision_microframes_per_real_meter";dist="mean_postchange_meters"
    def beats(x,y,k):return x[dist]>0 and y[dist]>0 and y[den]>0 and x[den]<=k*y[den]
    def retains(g):
        a,b=g["32_true"],g["slow_only"]
        return b[dist]>0 and a[dist]>=.9*b[dist] and a["mean_postchange_cells"]>=.9*b["mean_postchange_cells"]
    ci=contrasts["nearfield"]["slow_only"]
    gates={"large_beats_slow":beats(t,s,.75),
           **{"large_beats_"+a:beats(t,n[a],.85) for a in ("fixed","32_shuffled","32_generic")},
           "small_replication_nearfield":beats(n["16_true"],s,.8),
           "small_replication_obstacles":beats(o["16_true"],o["slow_only"],.8),
           "large_replication_obstacles":beats(o["32_true"],o["slow_only"],.8),
           "dose_benefit":beats(t,n["16_true"],.9),
           "retains_exploration_both":retains(n) and retains(o),
           "repeat_reduction":s["actual_postchange_prior_contact_followups"]>=50 and t["actual_postchange_prior_contact_followups"]>0 and t["next_microstep_repeat_contact_fraction"]<=.95*s["next_microstep_repeat_contact_fraction"],
           "onset_rate":t["contact_onsets_per_meter"] is not None and s["contact_onsets_per_meter"] is not None and s["contact_onsets_per_meter"]>0 and t["contact_onsets_per_meter"]<=s["contact_onsets_per_meter"],
           "source_coverage":sum(r["doses"]["32"]["usable"]>=10 for r in records)>=16,
           "fast_response":t["physical_fast_override_fraction"]>=.1 and t["real_same_macro_contact_followup_opportunities"]>0 and t["same_macro_next_micro_fast_response_fraction"]>=.5,
           "primary_paired_ci":ci["bootstrap95"] is not None and ci["invalid_draws"]==0 and ci["bootstrap95"][1]<0}
    report={"source_lives":len(SEEDS)*32,"target_lives":len(rows),"records":records,"summary":summary,"contrasts":contrasts,"integrity":checks,"gates":gates,"science_screen_pass":all(checks.values()) and all(gates.values()),"autonomous_escape_demonstrated":False}
    (out/"summary.json").write_text(json.dumps(report,sort_keys=True,separators=(",",":"))+"\n")
    print("FLYLOOP_R18 "+json.dumps(report,sort_keys=True),flush=True)
    if not all(checks.values()):raise AssertionError("R18 integrity failure")
    return report

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",default="results/flyloop_r18");run(p.parse_args().out)
