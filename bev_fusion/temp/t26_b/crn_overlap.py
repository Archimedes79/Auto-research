import json, glob, os
home = os.path.expanduser("~")
tok = set(json.load(open(f"{home}/exp_gcar_b123/feats/val_tokens.json")))
for d in sorted(glob.glob("runs/t20_base_s0/eval/suite/quant/eval*_n1000")):
    fr = [json.loads(l) for l in open(os.path.join(d, "frames.jsonl")) if l.strip()]
    keys = [f["key"] for f in fr]
    print(os.path.basename(d), len(keys), "in the 1706 cached frames:", sum(k in tok for k in keys), "scenes:", len({f["scene"] for f in fr}), "of which in the 43:", len({f["scene"] for f in fr if f["key"] in tok}))
