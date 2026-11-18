import json, os
R = os.environ["BFLC_ROOT"]
for run in ("e379_d_s0", "e373_cb_s0"):
    c = json.load(open(os.path.join(R, "runs", run, "token_anatomy_long_ltl0.2_kt0.6.json")))
    flat = {}
    def walk(d, pre=""):
        for k, v in d.items():
            if isinstance(v, dict):
                walk(v, pre + k + ".")
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                flat[pre + k] = v
    walk(c)
    keys = [k for k in flat if any(t in k for t in ("delivered", "tp", "dup", "wrong", "false_end", "ap", "recall", "share")) ][:30]
    print(run, {k: (round(flat[k], 4) if isinstance(flat[k], float) else flat[k]) for k in keys})
