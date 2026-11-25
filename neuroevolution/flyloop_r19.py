"""R19: slow-side learned delegation, never direct fast reinforcement."""
import argparse
import copy
import gzip
import json
import random
from pathlib import Path
from neuroevolution import flyloop_r11 as r11, flyloop_r15 as r15, flyloop_r18 as r18, flyloop_r13 as r13, flyloop_r9 as r9, flyloop_r1 as r1

SEEDS=tuple(range(14500,14524))
ARMS=("slow_only","fixed_half","fixed_full","random","learned","delayed","bias")
AUTH=(0.,.5,1.)
TRAIN=160
CALIBRATE=120
EVAL=100
LR=.03
EPS=.25
DELAY=8

def source_seed(s,g,i):
    return 180000+32*(s-SEEDS[0])+16*r11.GEOMETRIES.index(g)+i

def source_model(ws):
    m=r15.source_model(ws);m.SEEDS=SEEDS;return m

def target_world(seed,geo):
    m=r13.isolated("original");m.SEEDS=SEEDS
    return m.World(seed,geo,"static")

def features(sense,slow,fast,previous):
    x=tuple(sense)+tuple(slow)+tuple(fast)+(previous,)
    assert len(x)==13
    return x

class Delegator:
    def __init__(self,seed,arm):
        self.arm=arm
        self.weights=[[0.]*13 for _ in AUTH]
        self.rng=random.Random(seed*93719+1901)
        self.previous=0.
        self.outcomes=[]
        self.updates=0
    def phi(self,x):
        return (1.,)+(0.,)*12 if self.arm=="bias" else tuple(x)
    def choose(self,x,training):
        phi=self.phi(x)
        scores=[r1.dot(w,phi) for w in self.weights]
        if self.arm=="slow_only":i=0
        elif self.arm=="fixed_half":i=1
        elif self.arm=="fixed_full":i=2
        elif self.arm=="random":i=self.rng.randrange(3)
        elif training and self.rng.random()<EPS:i=self.rng.randrange(3)
        else:i=max(range(3),key=lambda j:scores[j])
        self.previous=AUTH[i]
        return i,phi,scores
    def observe(self,i,phi,reward,training):
        source=None;target=None
        if training and self.arm in ("learned","delayed","bias"):
            if self.arm!="delayed":
                source=len(self.outcomes);target=reward
            elif len(self.outcomes)>=DELAY:
                source=len(self.outcomes)-DELAY;target=self.outcomes[source]
            if target is not None:
                err=target-r1.dot(self.weights[i],phi)
                self.weights[i]=[r1.clamp(w+LR*err*x,-2.,2.) for w,x in zip(self.weights[i],phi)]
                self.updates+=1
        if training:self.outcomes.append(reward)
        return source,target

def intended(f,hidden):
    import math
    phi=tuple(hidden)+(1.,)
    return tuple(math.tanh(r1.dot(w,phi)) for w in f.output)

def blend(s,f,h):
    if h==0:return tuple(s)
    if h==1:return tuple(f)
    return tuple((1-h)*a+h*b for a,b in zip(s,f))

def alive(w):return w.body.energy>0 and w.body.integrity>0

def simulate(seed,geo,arm,weights):
    w=target_world(seed,geo)
    slow=r11.r2.MotorLearner(seed,"recurrent_frozen")
    teacher=r11.SlowRecoveryCritic(seed)
    fast=r11.FastStudent(seed);fast.output=copy.deepcopy(weights)
    episodes=r11.RecoveryEpisodes()
    arb=Delegator(seed,arm)
    trace=[];macros=[];contact_until=-1;frozen=None
    initial={"map":w.initial_rectangles,"slow":copy.deepcopy(slow.circuit.weights),"fast_input":fast.input,"fast_recurrent":fast.recurrent,"fast_bias":fast.bias,"fast_output":copy.deepcopy(fast.output)}
    for macro in range(TRAIN+CALIBRATE+EVAL):
        if not alive(w):break
        if macro==TRAIN+CALIBRATE:frozen=copy.deepcopy(arb.weights)
        sense=tuple(w.perceive())
        _,hidden,_,_,base=slow.command(sense)
        choice=teacher.choose(macro,hidden,sense[1:4],base,contact_until)
        cmd=choice["command"];physical=[];resolved=[]
        train_arb=TRAIN<=macro<TRAIN+CALIBRATE
        idx=0;phi=None;scores=None
        for sub in range(4):
            if not alive(w):break
            tick=w.frame;local=tuple(w.perceive());seven=r9.local_fast_sensors(local)
            fh=fast.observe(seven);proposal=intended(fast,fh)
            gate=r9.is_triggered(seven)
            if sub==0 and macro>=TRAIN:
                idx,phi,scores=arb.choose(features(local,cmd,proposal,arb.previous),train_arb)
            if macro<TRAIN:episodes.before_micro(tick,fh,seven)
            # Advance the same fast motor noise stream on every eligible post-training step, even at authority zero.
            fast_cmd,noise,_=fast.fast_motor(macro>=TRAIN and gate,cmd)
            h=AUTH[idx] if macro>=TRAIN and gate else 0.
            actual=blend(cmd,fast_cmd,h)
            p=w.micro_integrate(actual);physical.append(p)
            if macro<TRAIN:
                events,onset=episodes.after_real_micro(macro,tick,p,cmd,fh,seven,choice["pre_action_phi"])
                resolved.extend(events)
                if onset:contact_until=max(contact_until,macro+r11.CONTACT_EXPLORATION_MACROS+1)
            trace.append({"macro":macro,"tick":tick,"sub":sub,"sense":local,"hidden":fh,"slow":cmd,"fast_proposal":proposal,"fast":fast_cmd,"noise":noise,"eligible":gate,"authority":h,"motor":actual,"physics":p})
        if not physical:break
        teacher.complete_actual_macro(macro,choice,physical)
        teacher.train_delayed_recovery(macro,resolved)
        reward=sum(p["moved"] for p in physical)/(len(physical)*r1.V_MAX*r9.MICRO_DT)-1.2*sum(p["collision"] for p in physical)/len(physical)
        before=copy.deepcopy(arb.weights)
        source,target=arb.observe(idx,phi,reward,train_arb) if macro>=TRAIN else (None,None)
        macros.append({"macro":macro,"authority_index":idx,"phi":phi,"scores":scores,"reward":reward,"feedback_source":source,"feedback_target":target,"weights_before":before,"weights_after":copy.deepcopy(arb.weights)})
    post=[f for f in trace if f["macro"]>=TRAIN+CALIBRATE]
    meters=sum(f["physics"]["moved"] for f in post)
    hits=[f["physics"]["collision"] for f in post]
    eligible=sum(f["eligible"] for f in post)
    delegated=sum(f["authority"]>0 for f in post)
    priors=sum(hits[:-1]);repeat=sum(a and b for a,b in zip(hits,hits[1:]))
    immediate=[f for i,f in enumerate(post) if i and f["sub"]>0 and hits[i-1]]
    return {"seed":seed,"geometry":geo,"arm":arm,"initial":initial,"fast_final":fast.output,"slow_final":slow.circuit.weights,"arb_final":arb.weights,"arb_frozen":frozen,"updates":arb.updates,"trace":trace,"macros":macros,
            "postchange_actual_meters":meters,"postchange_collision_microframes":sum(hits),"cells":len({tuple(int(v) for v in f["physics"]["position"]) for f in post}),"onsets":sum(not a and b for a,b in zip(hits,hits[1:])),"repeat":repeat,"priors":priors,"eligible":eligible,"delegated":delegated,"post_steps":len(post),"immediate":len(immediate),"rapid_immediate":sum(f["authority"]>0 for f in immediate),"authority_counts":[sum(f["authority"]==h for f in post) for h in AUTH],"switches":sum(a["authority"]!=b["authority"] for a,b in zip(post,post[1:]))}

def audit(row):
    w=target_world(row["seed"],row["geometry"])
    arb=Delegator(row["seed"],row["arm"])
    trace=row["trace"];i=0
    for m in row["macros"]:
        macro=m["macro"];frames=[]
        while i<len(trace) and trace[i]["macro"]==macro:
            f=trace[i]
            if f["tick"]!=w.frame or tuple(w.perceive())!=tuple(f["sense"]):return False
            if f["authority"]!=(AUTH[m["authority_index"]] if macro>=TRAIN and f["eligible"] else 0.):return False
            if tuple(f["motor"])!=blend(f["slow"],f["fast"],f["authority"]):return False
            if w.micro_integrate(f["motor"])!=f["physics"]:return False
            frames.append(f);i+=1
        if not frames:return False
        training=TRAIN<=macro<TRAIN+CALIBRATE
        if macro>=TRAIN:
            f=frames[0];idx,phi,scores=arb.choose(features(f["sense"],f["slow"],f["fast_proposal"],arb.previous),training)
            if (idx,phi,scores)!=(m["authority_index"],m["phi"],m["scores"]):return False
        if arb.weights!=m["weights_before"]:return False
        reward=sum(f["physics"]["moved"] for f in frames)/(len(frames)*r1.V_MAX*r9.MICRO_DT)-1.2*sum(f["physics"]["collision"] for f in frames)/len(frames)
        if reward!=m["reward"]:return False
        src,tgt=arb.observe(m["authority_index"],m["phi"],reward,training) if macro>=TRAIN else (None,None)
        if (src,tgt)!=(m["feedback_source"],m["feedback_target"]) or arb.weights!=m["weights_after"]:return False
    return i==len(trace) and arb.weights==row["arb_final"] and arb.updates==row["updates"] and row["fast_final"]==row["initial"]["fast_output"] and row["slow_final"]==row["initial"]["slow"] and (row["arb_frozen"] is None or row["arb_frozen"]==row["arb_final"])

def summarize(rows):
    def s(k):return sum(r[k] for r in rows)
    def q(a,b):return a/b if b>0 else None
    return {"lives":len(rows),"density":q(s("postchange_collision_microframes"),s("postchange_actual_meters")),"meters":s("postchange_actual_meters")/len(rows),"cells":s("cells")/len(rows),"onset_rate":q(s("onsets"),s("postchange_actual_meters")),"repeat_fraction":q(s("repeat"),s("priors")),"eligible":s("eligible"),"delegated_fraction":q(s("delegated"),s("eligible")),"rapid_response":q(s("rapid_immediate"),s("immediate")),"authority_counts":[sum(r["authority_counts"][i] for r in rows) for i in range(3)],"switches":s("switches"),"updates":s("updates")}

def bootstrap(a,b):
    rng=random.Random(145001);d=[];invalid=0
    for _ in range(5000):
        ix=[rng.randrange(len(a)) for _ in a]
        x=r18.ratio([a[i] for i in ix]);y=r18.ratio([b[i] for i in ix])
        if x is None or y is None:invalid+=1
        else:d.append(x-y)
    d.sort()
    return {"invalid_draws":invalid,"bootstrap95":[d[int(.025*len(d))],d[min(len(d)-1,int(.975*len(d)))]] if d else None}

def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    checks={k:True for k in ("source_audit","source_provenance","target_replay","initial_pairing","source_target_disjoint","encoder_identity","initial_training_pairing")}
    sources=set();targets=set();rows=[];records=[]
    with open(out/"lives.jsonl.gz","wb") as raw:
        with gzip.GzipFile(filename="",mode="wb",fileobj=raw,mtime=0) as gz:
            for seed in SEEDS:
                rec=[];safe=[];enc=set()
                for geo in r11.GEOMETRIES:
                    for i in range(16):
                        ws=source_seed(seed,geo,i);m=source_model(ws);r=m.simulate(seed,geo,"static","slow_only")
                        checks["source_audit"] &= all(m.audit([r]).values())
                        enc.add(r15.encoder(r));sources.add(r11.sha(r["initial_10_rectangles"]))
                        for e in r["real_successful_recovery_episodes"]:
                            f=r["raw_micro_trace"][e["first_post_contact_micro"]];q=r["raw_micro_trace"][e["first_recovery_motor_micro"]]
                            checks["source_provenance"] &= f["four_fast_recurrent_states"]==[round(v,9) for v in e["first_post_contact_fast_hidden"]] and q["actually_executed_analog_motor"]==e["actual_first_recovery_motor"]
                            rec.append({"hidden":e["first_post_contact_fast_hidden"],"motor":e["actual_first_recovery_motor"],"provenance":[ws,e["collision_micro"],e["first_recovery_motor_micro"]]})
                        for e in r["same_life_real_safe_macro_motors"]:safe.append({"motor":e["motor"],"provenance":[ws,e["macro"]]})
                        r15.write(gz,{**r,"phase":"source","world_seed":ws})
                data=r15.dataset(rec,safe,"true");weights,updates=r15.pretrain(seed,data)
                records.append({"seed":seed,"examples":len(data),"updates":updates})
                r15.write(gz,{"phase":"pretraining","seed":seed,"data":data,"weights":weights})
                for geo in r11.GEOMETRIES:
                    sig=set();initial=set()
                    for arm in ARMS:
                        r=simulate(seed,geo,arm,weights)
                        checks["target_replay"] &= audit(r)
                        z=r["initial"];enc.add(r11.sha([z["slow"],z["fast_input"],z["fast_recurrent"],z["fast_bias"]]))
                        targets.add(r11.sha(z["map"]));initial.add(r11.sha(z));sig.add(r11.sha([f for f in r["trace"] if f["macro"]<TRAIN]))
                        r15.write(gz,{**r,"phase":"target"});rows.append({k:v for k,v in r.items() if k not in ("trace","macros","initial")})
                    checks["initial_pairing"] &= len(initial)==1
                    checks["initial_training_pairing"] &= len(sig)==1
                checks["encoder_identity"] &= len(enc)==1
    checks["source_target_disjoint"]=not bool(sources&targets)
    lookup={(r["geometry"],r["arm"],r["seed"]):r for r in rows}
    def group(g,a):return [lookup[g,a,s] for s in SEEDS]
    summary={g:{a:summarize(group(g,a)) for a in ARMS} for g in r11.GEOMETRIES}
    contrasts={g:{a:bootstrap(group(g,"learned"),group(g,a)) for a in ARMS if a!="learned"} for g in r11.GEOMETRIES}
    n=summary["nearfield"];t=n["learned"];s=n["slow_only"]
    def beats(a,k):
        b=n[a];return t["density"] is not None and b["density"] is not None and b["density"]>0 and t["density"]<=k*b["density"]
    ci=contrasts["nearfield"]["slow_only"]
    gates={"beats_slow":beats("slow_only",.85),"beats_fixed_full":beats("fixed_full",.9),"beats_random":beats("random",.85),"beats_delayed":beats("delayed",.85),"beats_bias":beats("bias",.9),"retains_exploration":s["meters"]>0 and t["meters"]>=.9*s["meters"] and t["cells"]>=.9*s["cells"],"nontrivial_delegation":t["delegated_fraction"] is not None and .1<=t["delegated_fraction"]<=.9,"paired_ci":ci["invalid_draws"]==0 and ci["bootstrap95"] is not None and ci["bootstrap95"][1]<0}
    report={"source_lives":len(SEEDS)*32,"target_lives":len(rows),"records":records,"summary":summary,"contrasts":contrasts,"integrity":checks,"gates":gates,"science_screen_pass":all(checks.values()) and all(gates.values()),"autonomous_escape_demonstrated":False}
    (out/"summary.json").write_text(json.dumps(report,sort_keys=True,separators=(",",":"))+"\n")
    print("FLYLOOP_R19 "+json.dumps(report,sort_keys=True),flush=True)
    if not all(checks.values()):raise AssertionError("R19 integrity failure")
    return report

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",default="results/flyloop_r19");run(p.parse_args().out)
