import sys

import torch

a, b = torch.load(sys.argv[1], weights_only=False), torch.load(sys.argv[2], weights_only=False)
bad = 0
for name in a:
    ea, eb = a[name], b[name]
    msgs = []
    if ea["keys"] != eb["keys"]:
        ka, kb = set(ea["keys"]), set(eb["keys"])
        vd = [k for k in ka & kb if ea["keys"][k] != eb["keys"][k]][:5]
        msgs.append(f"state keys differ: only head {sorted(ka - kb)[:5]} only live {sorted(kb - ka)[:5]} value-diff {vd}")
    if ea["n_param"] != eb["n_param"]:
        msgs.append("n_param differs")
    for mode in ("eval", "train"):
        for i in (0, 1):
            fa, fb = ea[mode][i], eb[mode][i]
            if fa.keys() != fb.keys():
                msgs.append(f"{mode}[{i}] output keys differ {set(fa) ^ set(fb)}")
                continue
            d = [(k, float((fa[k].double() - fb[k].double()).abs().max())) for k in fa if not torch.equal(fa[k], fb[k])]
            if d:
                msgs.append(f"{mode}[{i}] not bit-equal: {d[:4]} (n={len(d)})")
    ga, gb = ea["grads"], eb["grads"]
    if ga.keys() != gb.keys():
        msgs.append(f"grad keys differ {set(ga) ^ set(gb)}")
    else:
        d = [(k, float((ga[k] - gb[k]).abs().max())) for k in ga if not torch.equal(ga[k], gb[k])]
        if d:
            msgs.append(f"grads not bit-equal n={len(d)} e.g. {d[:3]}")
    if "advance" in ea:
        d = [k for k in ea["advance"] if not torch.equal(ea["advance"][k], eb["advance"][k])]
        if d:
            msgs.append(f"advance state differs {d[:4]}")
    print(("DIFF " if msgs else "same ") + name, msgs)
    bad += bool(msgs)
print("configs different:", bad)
