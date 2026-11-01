"""throw-away: stored timing and the two labelled estimates of the prototype for a builder and a union entry"""
import evaluate as EV
import lane_prototype as LP

reg = EV.registry()
for n in ("recall_dp", "fast_recall_dp", "recall_ring", "ens_j1rc_23", "ens_j1rc_0123"):
    e = reg[n]
    base = reg[e["members"][0]] if e["kind"] == "union" else e
    t = LP.stored_timing(e, EV.builder_cfg(base).temporal)
    print(n, {k: v for k, v in t.items() if k not in ("files", "source")})
    if t.get("ms") is not None:
        s = dict(system=n, split="val", frames=1, scenes=[], fresh=False, date="-", commit="-", metrics={}, timing=t, out="x")
        print(LP.summary_md(s))
