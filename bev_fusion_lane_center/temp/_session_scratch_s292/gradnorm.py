import json, os, statistics as st
R = os.environ["BFLC_ROOT"] + "/runs/"
for arm in ("c2", "a"):
    for s in (0, 1):
        out = []
        for p in (3, 9):
            xs = [json.loads(l)["grad_norm"] for l in open(f"{R}e292_{arm}_s{s}_p{p}/log.jsonl") if "grad_norm" in l]
            out.append(f"p{p} n {len(xs)} median {st.median(xs):.1f} max {max(xs):.1f}")
        print(arm, s, " | ".join(out))
