p = "training/startable.py"
s = open(p, encoding="utf-8").read()
a = '''                stored = json.load(open(sc, encoding="utf-8"))
                diff = sorted(k for k in set(cfg) & set(stored.get("config", stored)) if k in ("steps", "lr", "bs", "width", "seed", "per100") and cfg[k] != stored.get("config", stored).get(k))
'''
b = '''                stored = json.load(open(sc, encoding="utf-8"))
                vals = dict(stored.get("args") or {})                      # the trainer writes the flat config and the command line of the run under "args"
                vals.update({k: v for k, v in stored.items() if k != "args"})
                diff = sorted(k for k in ("steps", "lr", "bs", "width", "seed", "per100", "budget") if k in cfg and k in vals and cfg[k] != vals[k])
'''
assert s.count(a) == 1
open(p, "w", encoding="utf-8").write(s.replace(a, b))
