import json, sys
ROOT = "/mnt/c/Auto-research/bev_fusion"
sys.path.insert(0, ROOT + "/training")
base = json.load(open(ROOT + "/training/configs/t20_base.json"))
def flat(d, p=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, p + k + "."))
        else:
            out[p + k] = v
    return out
fb = flat(base)
for n in ("pair", "pair4", "rec", "ctl"):
    c = json.load(open(f"{ROOT}/training/configs/t21_{n}.json"))
    fc = flat(c)
    diff = {k: (fb.get(k, "<absent>"), fc.get(k, "<absent>")) for k in sorted(set(fb) | set(fc)) if fb.get(k, "<absent>") != fc.get(k, "<absent>") and k not in ("_about", "name")}
    print(n, diff)
    from model import ModelConfig, check_config
    mc = ModelConfig(**c["model"]); check_config(mc)
print("model configs parse + check_config ok")
