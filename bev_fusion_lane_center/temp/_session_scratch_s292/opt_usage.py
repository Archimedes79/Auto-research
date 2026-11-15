"""S2 inventory (read-only): for each option of jends_train.py / jmaps_train.py how many run directories used a non-default value (config.json args / model.pt args are not read, only config.json)"""
import glob, json, os, re, sys
R = os.environ["BFLC_ROOT"]
def options(path):
    src = open(path, encoding="utf-8").read()
    out = {}
    for m in re.finditer(r'ap\.add_argument\("--([a-z0-9-]+)"([^\n]*)', src):
        d = re.search(r'default=([^,)\s]+)', m.group(2))
        out[m.group(1).replace("-", "_")] = (d.group(1) if d else ("False" if "store_true" in m.group(2) else None))
    return out
for trainer, pat in (("training/train/jends_train.py", "runs/*/config.json"), ("training/train/jmaps_train.py", "runs/*/config.json")):
    opts = options(os.path.join(R, trainer))
    used = {k: [] for k in opts}
    n = 0
    for c in glob.glob(os.path.join(R, pat)):
        try:
            a = json.load(open(c)).get("args") or {}
        except Exception:
            continue
        if "maps" in a and "jends" not in trainer and "e2e" in a:
            pass
        keys = set(a)
        if not (set(opts) & keys) or not ({"name", "seed"} <= keys):
            continue
        # a run of THIS trainer has most of its option names
        if len(set(opts) & keys) < 0.7 * len(opts):
            continue
        n += 1
        for k, dv in opts.items():
            if k in a and str(a[k]) not in (dv, str(dv).strip("'\"")) and a[k] not in (None, False, 0, 0.0) or (k in a and dv not in (None,) and str(a[k]) != str(dv).strip("'\"") and a[k] is not False and a[k] is not None):
                used[k].append(os.path.basename(os.path.dirname(c)))
    print(f"== {trainer}: {n} run configs")
    for k, v in used.items():
        print(f"  --{k.replace('_','-'):14s} default {str(opts[k]):10s} non-default in {len(v):3d} runs  {v[:3]}")
