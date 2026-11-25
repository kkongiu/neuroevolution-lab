"""R14: isolate fast motor authority, no additional learning."""
import argparse
import gzip
import json
import random
from pathlib import Path
from neuroevolution import flyloop_r11 as r11, flyloop_r13 as r13

SEEDS=tuple(range(13000,13024))
ARMS=("fast_fixed","fast_recovery_true","fast_recovery_shuffled")
AUTH=(.25,.5,1.)
CONDITIONS=(("slow_only",0.),)+tuple((a,h) for a in ARMS for h in AUTH)
def blend(slow,fast,h):
    if not 0<=h<=1: raise ValueError(h)
    if h==1: return tuple(fast)
    if h==0: return tuple(slow)
    return tuple((1-h)*s+h*f for s,f in zip(slow,fast))

def model_for(h):
    m=r13.isolated("original")
    m.SEEDS=SEEDS
    parent=m.FastStudent
    class Student(parent):
        def fast_motor(self,enabled,slow_command):
            fast,noise,active=super().fast_motor(enabled,slow_command)
            return (blend(slow_command,fast,h) if active else fast),noise,active
    m.FastStudent=Student
    return m

def boot(d):
    if not d:return None
    rng=random.Random(130001)
    v=sorted(sum(rng.choice(d) for _ in d)/len(d) for _ in range(5000))
    return [v[125],v[4874]]

def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    models={h:model_for(h) for h in (0.,)+AUTH}
    rows=[]
    integrity={"audit":True,"training_pairing":True,"initial_pairing":True,"labels_pairing":True}
    with open(out/"lives.jsonl.gz","wb") as raw:
        with gzip.GzipFile(filename="",mode="wb",fileobj=raw,mtime=0) as gz:
            for geo in r11.GEOMETRIES:
                for seed in SEEDS:
                    group=[]
                    for arm,h in CONDITIONS:
                        m=models[h]
                        row=m.simulate(seed,geo,"static",arm)
                        integrity["audit"] &= all(m.audit([row]).values())
                        row["authority"]=h
                        group.append(row)
                    integrity["training_pairing"] &= len({r13.trace_signature(r) for r in group})==1
                    keys=("initial_10_rectangles","initial_body_pose","original_434_slow_coefficients_initial","fast_four_initial_input","fast_four_initial_recurrent","fast_four_initial_bias","fast_10_output_coefficients_initial")
                    integrity["initial_pairing"] &= len({r11.sha([r[k] for k in keys]) for r in group})==1
                    for arm in ARMS:
                        integrity["labels_pairing"] &= len({r11.sha(r["actual_recovery_training_lessons"]) for r in group if r["arm"]==arm})==1
                    for r in group:
                        rows.append(r13.compact(r))
                        gz.write((json.dumps(r,sort_keys=True,separators=(",",":"))+"\n").encode())
    def key(a,h):return a+"@"+str(h)
    summary={g:{key(a,h):r11.summarize([r for r in rows if (r["geometry"],r["arm"],r["authority"])==(g,a,h)]) for a,h in CONDITIONS} for g in r11.GEOMETRIES}
    n=summary["nearfield"]
    true=n[key("fast_recovery_true",.5)]
    slow=n[key("slow_only",0.)]
    den="collision_microframes_per_real_meter";dist="mean_postchange_meters"
    def beats(ref,ratio):
        return true[dist]>0 and ref[dist]>0 and ref[den]>0 and true[den]<=ratio*ref[den]
    teachers=r13.teacher_stats([r for r in rows if r["geometry"]=="nearfield" and r["arm"]=="slow_only"])
    gates={
        "beats_full_authority":beats(n[key("fast_recovery_true",1.)],.8),
        "beats_slow":beats(slow,.9),
        "beats_fixed":beats(n[key("fast_fixed",.5)],.85),
        "beats_shuffled":beats(n[key("fast_recovery_shuffled",.5)],.85),
        "preserves_exploration":slow[dist]>0 and true[dist]>=.9*slow[dist] and true["mean_postchange_cells"]>=.9*slow["mean_postchange_cells"],
        "repeat_contact_reduction":slow["actual_postchange_prior_contact_followups"]>=50 and true["actual_postchange_prior_contact_followups"]>0 and true["next_microstep_repeat_contact_fraction"]<=.9*slow["next_microstep_repeat_contact_fraction"],
        "teacher_coverage":teachers["recoveries"]>=12 and teachers["seeds_ge2"]>=6,
        "fast_response":true["physical_fast_override_fraction"]>=.1 and true["real_same_macro_contact_followup_opportunities"]>0 and true["same_macro_next_micro_fast_response_fraction"]>=.5,
    }
    lookup={(r["geometry"],r["seed"],r["arm"],r["authority"]):r for r in rows}
    contrasts={}
    for a,h in (("slow_only",0.),("fast_fixed",.5),("fast_recovery_shuffled",.5),("fast_recovery_true",1.)):
        d=[]
        for seed in SEEDS:
            x=lookup["nearfield",seed,"fast_recovery_true",.5];y=lookup["nearfield",seed,a,h]
            if x["postchange_actual_meters"]>0 and y["postchange_actual_meters"]>0:
                d.append(x["postchange_collision_microframes"]/x["postchange_actual_meters"]-y["postchange_collision_microframes"]/y["postchange_actual_meters"])
        contrasts[key(a,h)]={"paired_seeds":len(d),"mean_delta":sum(d)/len(d) if d else None,"bootstrap95":boot(d)}
    report={"lives":len(rows),"microsteps":sum(r["actual_microsteps"] for r in rows),"teachers":teachers,"summary":summary,"integrity":integrity,"gates":gates,"contrasts":contrasts,"science_screen_pass":all(integrity.values()) and all(gates.values()),"autonomous_escape_demonstrated":False}
    (out/"summary.json").write_text(json.dumps(report,sort_keys=True,separators=(",",":"))+"\n")
    print("FLYLOOP_R14 "+json.dumps(report,sort_keys=True),flush=True)
    if not all(integrity.values()):raise AssertionError("R14 integrity failure")
    return report
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",default="results/flyloop_r14");run(p.parse_args().out)
