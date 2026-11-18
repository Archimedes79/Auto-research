import json, os, sys
R = os.environ["BFLC_ROOT"]
p = os.path.join(R, "runs", sys.argv[1], sys.argv[2])
c = json.load(open(p))
def walk(d, pre=""):
    for k, v in d.items():
        if isinstance(v, dict):
            walk(v, pre + k + ".")
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            if any(t in (pre + k) for t in ("topo_F1", "F1@1.0", "lanes_found", "junction_F1", "dup_frac", "junction_P", "junction_R", "diff", "lo", "hi", "Q")):
                print("%-60s %.4f" % (pre + k, v))
walk(c)
