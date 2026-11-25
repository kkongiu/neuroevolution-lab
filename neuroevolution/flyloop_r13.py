"""R13: paired R11 recovery learning under two physical motor plants."""
import argparse
import gzip
import importlib.util
import json
import random
from pathlib import Path
from neuroevolution import flyloop_r11 as r11, flyloop_r12 as r12, flyloop_r9 as r9

SEEDS=tuple(range(12700,12724))
PLANTS=("original","signed")
GEOMETRIES=r11.GEOMETRIES
ARMS=r11.ARMS

def isolated(plant):
    if plant not in PLANTS:
        raise ValueError(plant)
    spec=importlib.util.spec_from_file_location("r13_"+plant,r11.__file__)
    model=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)
    model.SEEDS=SEEDS
    original_world=model.World
    if plant=="signed":
        class SignedWorld(original_world):
            def micro_integrate(self,command):
                b=self.body
                before=(b.x,b.y,b.heading)
                out=r12.integrate(self,command,"signed")
                out.update(position=[round(b.x,9),round(b.y,9)],
                           heading=round(b.heading,9),speed=round(b.speed,9),
                           turn=round(b.turn,9),energy=round(b.energy,9),
                           integrity=round(b.integrity,9),
                           before_position=[round(before[0],9),round(before[1],9)],
                           before_heading=round(before[2],9))
                return out
        model.World=SignedWorld
    return model

def mean(v):
    return sum(v)/len(v) if v else None

def bootstrap(d):
    if not d:return None
    rng=random.Random(127001)
    x=sorted(mean([rng.choice(d) for _ in d]) for _ in range(5000))
    return [x[125],x[4874]]

def compact(row):
    return {k:v for k,v in row.items() if k not in ("raw_micro_trace","macro_trace")}

def trace_signature(row):
    return r11.sha([(f["actually_executed_analog_motor"],f["actual_physical_body_after_micro"],f["original_slow_sixteen_hidden"])
                    for f in row["raw_micro_trace"] if f["macro_tick"]<r11.TRAIN_MACROS])

def teacher_stats(rows):
    n=[len(r["real_successful_recovery_episodes"]) for r in rows]
    return {"recoveries":sum(n),"mean":mean(n),"windows":sum(r["real_contact_window_onsets_training"] for r in rows),
            "seeds_ge1":sum(x>=1 for x in n),"seeds_ge2":sum(x>=2 for x in n)}

def run(out):
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    compact_rows=[]
    integrity={"physical_and_lesson_audit":True,"identical_training":True,"paired_initial_state":True,"matched_supervision":True}
    initial={}
    with open(out/"lives.jsonl.gz","wb") as raw:
        with gzip.GzipFile(filename="",mode="wb",fileobj=raw,mtime=0) as gz:
            for plant in PLANTS:
                m=isolated(plant)
                for geo in GEOMETRIES:
                    for seed in SEEDS:
                        group=[m.simulate(seed,geo,"static",arm) for arm in ARMS]
                        checks=m.audit(group)
                        integrity["physical_and_lesson_audit"] &= all(checks.values())
                        integrity["identical_training"] &= len({trace_signature(r) for r in group})==1
                        lesson_sets=[[(l["actual_contact_fast_state_micro"],l["actual_contact_fast_four_hidden"]) for l in r["actual_recovery_training_lessons"]] for r in group if r["arm"] in m.SUPERVISED_ARMS]
                        integrity["matched_supervision"] &= all(v==lesson_sets[0] for v in lesson_sets)
                        for row in group:
                            key=(seed,geo, row["arm"])
                            signature=r11.sha([row[k] for k in ("initial_10_rectangles","initial_body_pose","original_434_slow_coefficients_initial","fast_four_initial_input","fast_four_initial_recurrent","fast_four_initial_bias","fast_10_output_coefficients_initial")])
                            if plant=="original": initial[key]=signature
                            else: integrity["paired_initial_state"] &= initial[key]==signature
                            row["plant"]=plant
                            compact_rows.append(compact(row))
                            gz.write((json.dumps(row,sort_keys=True,separators=(",",":"))+"\n").encode())
    summary={p:{g:{a:r11.summarize([r for r in compact_rows if (r["plant"],r["geometry"],r["arm"])==(p,g,a)]) for a in ARMS} for g in GEOMETRIES} for p in PLANTS}
    teachers={}
    for geo in GEOMETRIES:
        groups={p:[r for r in compact_rows if (r["plant"],r["geometry"],r["arm"])==(p,geo,"slow_only")] for p in PLANTS}
        counts={p:{r["seed"]:len(r["real_successful_recovery_episodes"]) for r in groups[p]} for p in PLANTS}
        d=[counts["signed"][s]-counts["original"][s] for s in SEEDS]
        teachers[geo]={p:teacher_stats(groups[p]) for p in PLANTS}
        teachers[geo].update(delta=mean(d),bootstrap95=bootstrap(d))
    t=teachers["nearfield"]
    n=summary["signed"]["nearfield"]
    true=n["fast_recovery_true"]
    slow=n["slow_only"]
    density="collision_microframes_per_real_meter"
    meters="mean_postchange_meters"
    def beats(ref,ratio):
        return true[meters]>0 and ref[meters]>0 and ref[density]>0 and true[density]<=ratio*ref[density]
    gates={
        "teacher_count_ratio":t["signed"]["mean"]>=1.5*t["original"]["mean"] and t["signed"]["mean"]>0,
        "teacher_paired_ci_positive":t["bootstrap95"][0]>0,
        "teacher_ge2_in_12_seeds":t["signed"]["seeds_ge2"]>=12,
        "student_beats_slow":beats(slow,.80),
        "student_beats_fixed":beats(n["fast_fixed"],.85),
        "student_beats_shuffled":beats(n["fast_recovery_shuffled"],.85),
        "student_repeat_contacts":slow["actual_postchange_prior_contact_followups"]>=50 and true["actual_postchange_prior_contact_followups"]>0 and true["next_microstep_repeat_contact_fraction"]<=.85*slow["next_microstep_repeat_contact_fraction"],
        "student_retains_exploration":true[meters]>0 and slow[meters]>0 and true[meters]>=.90*slow[meters] and true["mean_postchange_cells"]>=.90*slow["mean_postchange_cells"] and true["mean_postchange_RMS_radius"]>=.85*slow["mean_postchange_RMS_radius"],
        "student_fast_response":true["physical_fast_override_fraction"]>=.10 and true["real_same_macro_contact_followup_opportunities"]>0 and true["same_macro_next_micro_fast_response_fraction"]>=.50,
    }
    contrasts={}
    lookup={(r["plant"],r["geometry"],r["arm"],r["seed"]):r for r in compact_rows}
    for p in PLANTS:
        contrasts[p]={}
        for g in GEOMETRIES:
            contrasts[p][g]={}
            for a in ("slow_only","fast_fixed","fast_recovery_shuffled"):
                d=[]
                for s in SEEDS:
                    x=lookup[p,g,"fast_recovery_true",s]; y=lookup[p,g,a,s]
                    if x["postchange_actual_meters"]>0 and y["postchange_actual_meters"]>0:
                        d.append(x["postchange_collision_microframes"]/x["postchange_actual_meters"]-y["postchange_collision_microframes"]/y["postchange_actual_meters"])
                contrasts[p][g][a]={"paired_seeds":len(d),"mean_delta":mean(d),"bootstrap95":bootstrap(d)}
    report={"lives":len(compact_rows),"microsteps":sum(r["actual_microsteps"] for r in compact_rows),
            "teachers":teachers,"summary":summary,"integrity":integrity,"gates":gates,
            "density_contrasts":contrasts,"science_screen_pass":all(integrity.values()) and all(gates.values()),
            "autonomous_escape_demonstrated":False,
            "limitations":["Signed speed sensor now spans negative values; unchanged forward-only ray layout provides no rear ray.","Teacher labels are retrospective associations, not proof of action causation."]}
    (out/"summary.json").write_text(json.dumps(report,sort_keys=True,separators=(",",":"))+"\n")
    print("FLYLOOP_R13 "+json.dumps(report,sort_keys=True),flush=True)
    if not all(integrity.values()):
        raise AssertionError("R13 integrity failure")
    return report

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",default="results/flyloop_r13")
    run(p.parse_args().out)
