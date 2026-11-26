"""Infrastructure only: original R18B simulations and aggregation."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
from neuroevolution import flyloop_r18b as r

def dump(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"))

def strip(row):
    # Keep all episode-level observations; raw traces live in separate archives.
    return {**row,"training":[{k:v for k,v in ep.items() if k!="trace"} for ep in row["training"]],
            "tests":{mode:{p:[{k:v for k,v in ep.items() if k!="trace"} for ep in eps]
                           for p,eps in phases.items()} for mode,phases in row["tests"].items()}}

def shard(seed,out):
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    summaries=[]
    hashes={}
    for arm in r.ARMS:
        row=r.simulate(seed,arm)
        raw=dump(row).encode()
        hashes[arm]=hashlib.sha256(raw).hexdigest()
        with (out/f"{seed}-{arm}.json.gz").open("wb") as dest:
            with gzip.GzipFile(fileobj=dest,mode="wb",mtime=0,filename="") as z:
                z.write(raw)
        summaries.append(strip(row))
        del row,raw
        print(seed,arm,hashes[arm],flush=True)
    (out/f"{seed}-summary.json").write_text(dump({"seed":seed,"hashes":hashes,"rows":summaries}))

def combine(directory):
    directory=Path(directory)
    rows=[]
    for seed in r.SEEDS:
        paths=list(directory.rglob(f"{seed}-summary.json"))
        if len(paths)!=1:
            raise ValueError(f"Expected exactly one summary for seed {seed}, got {len(paths)}")
        data=json.loads(paths[0].read_text())
        assert data["seed"]==seed
        assert [row["arm"] for row in data["rows"]]==list(r.ARMS)
        assert set(data["hashes"])==set(r.ARMS)
        for row in data["rows"]:
            assert row["seed"]==seed and len(row["training"])==60
            for phases in row["tests"].values():
                assert set(phases)==set(r.SCENARIOS)
                assert all(len(eps)==20 for eps in phases.values())
        rows.extend(data["rows"])
    result=r.summarize(rows)
    return {k:v for k,v in result.items() if k!="rows"}

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int)
    p.add_argument("--out",default="results/shard")
    p.add_argument("--combine")
    args=p.parse_args()
    if args.combine:
        print(json.dumps(combine(args.combine),indent=2))
    else:
        assert args.seed in r.SEEDS
        shard(args.seed,args.out)
