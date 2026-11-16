"""S1 scratch: remove the never-run E302 `capacity` option (degree heads, Sinkhorn plan) from jends.py, jends_train.py, jends_eval.py, jmodel_cost.py."""
import re
def edit(path, fn):
    s = open(path, encoding="utf-8").read(); t = fn(s); assert t != s, path; open(path, "w", encoding="utf-8").write(t); print("edited", path, len(s) - len(t), "chars removed")

def jends(s):
    # docstring paragraph E302
    i = s.index("E302 `capacity`:"); j = s.index("E308 `ntok`:"); s = s[:i] + s[j:]
    # constants
    s = s.replace("SINK_ITERS = 10\n", "", 1)
    i = s.index("DEG_VALUES = "); j = s.index("\n", i) + 1; s = s[:i] + s[j:]
    # sinkhorn_plan
    i = s.index("def sinkhorn_plan("); j = s.index("ATT_M, ATT_REACH"); s = s[:i] + s[j:]
    # ctor
    s = s.replace("scales=False, capacity=False, ntok=0,", "scales=False, ntok=0,", 1)
    s = s.replace("self.k, self.order, self.evidence, self.scales, self.capacity, self.ntok, self.unc = k, order, evidence, scales, capacity, ntok, unc", "self.k, self.order, self.evidence, self.scales, self.ntok, self.unc = k, order, evidence, scales, ntok, unc", 1)
    i = s.index("        if capacity:"); j = s.index("        if scales:", i); s = s[:i] + s[j:]
    # forward
    i = s.index("        extra = {}\n        if self.capacity:"); j = s.index("        out = dict(heat=heat,", i); s = s[:i] + s[j:]
    s = s.replace("logit=logit, allowed=allowed, **extra)", "logit=logit, allowed=allowed)", 1)
    # degree_classes
    i = s.index("def degree_classes("); j = s.index("def geom_loss("); s = s[:i] + s[j:]
    # loss
    s = s.replace('    cap = "deg_e" in out\n    loss = bce + (0.0 if cap else 0.5 * dual)                                                               # the plan already makes rows and columns compete (E302)\n', "    loss = bce + 0.5 * dual\n", 1)
    s = s.replace('    if not cap:\n        info["dual"] = float(dual)\n', '    info["dual"] = float(dual)\n', 1)
    i = s.index("    if cap:  "); j = s.index('    if "scale" in out and "pscale" in batch', i); s = s[:i] + s[j:]
    # selftest
    i = s.index("    # E302: the capacitated plan."); j = s.index("    # critic 2026-11-11: with no real peak at all", i); s = s[:i] + s[j:]
    return s

def jtrain(s):
    s = s.replace("e2e=False, capacity=False, ntok=False,", "e2e=False, ntok=False,", 1)
    s = s.replace("scales=scales, capacity=capacity, ntok=64 if ntok else 0,", "scales=scales, ntok=64 if ntok else 0,", 1)
    i = s.index('    ap.add_argument("--capacity"'); j = s.index("\n", i) + 1; s = s[:i] + s[j:]
    s = s.replace("e2e=a.e2e, capacity=a.capacity, ntok=a.ntok,", "e2e=a.e2e, ntok=a.ntok,", 1)
    s = s.replace("e2e={a.e2e} capacity={a.capacity} ntok={a.ntok}", "e2e={a.e2e} ntok={a.ntok}", 1)
    s = s.replace("e2e=a.e2e, capacity=a.capacity", "e2e=a.e2e", 1)
    return s

def jeval(s):
    return s.replace('scales=bool(a.get("scales", False)), capacity=bool(a.get("capacity", False)), ntok=', 'scales=bool(a.get("scales", False)), ntok=', 1)

def jcost(s):
    return s.replace('scales=bool(ar.get("scales", False)), capacity=bool(ar.get("capacity", False)),', 'scales=bool(ar.get("scales", False)),', 1)

edit("training/grid/jends.py", jends); edit("training/train/jends_train.py", jtrain); edit("evaluation/jends_eval.py", jeval); edit("evaluation/jmodel_cost.py", jcost)
