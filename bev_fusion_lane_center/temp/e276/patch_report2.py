p = "temp/e276/e276_build_report.py"
L = open(p, encoding="utf-8").read().split("\n")
helper = '''

def long_stats():
    out = []
    for x in asg:
        lk = (x or {}).get("result", {}).get("links")
        if not lk:
            return None
        rc = next(v for v in lk["risk_coverage"] if v["threshold"] == 0.5)
        lg_ = lk["p>=0.5 long"]
        out.append(dict(prec=lg_["precision"], share=lg_["n"] * (1 - lg_["precision"]) / (rc["links"] * (1 - rc["precision"]))))
    return out


def rng(vals, d=0):
    vals = [v for v in vals if v is not None]
    return "pending" if not vals else f"{100 * min(vals):.{d}f} - {100 * max(vals):.{d}f} %"


LS = long_stats()
PERS = [((x or {}).get("result") or {}).get("delivered / long links", {}).get("persistent") for x in (tmpl0, tmpl1)]
CUT = None if not all(lthr) else mean([x["variants"]["0.5_0.5"]["topo_F1"] - x["variants"]["0.5_1.01"]["topo_F1"] for x in lthr])
AN = J("temp/e275/analysis_e273_p_s0.json") or {}
'''.split("\n")
i = next(k for k, l in enumerate(L) if l.startswith('css = """'))
L[i:i] = helper
s = "\n".join(L)
i0 = s.index('"<li><b>The weak spot is the links across junctions</b>')
i1 = s.index('</ul></div>")', i0) + len('</ul></div>")')
new = '''"<li><b>The weak spot is the links across junctions</b>, not the node-to-map assignment: long links are right in only " + (rng([x["prec"] for x in LS]) if LS else "pending") + " of the cases, only " + rng(PERS) + " of them survive half a second, they carry " + (rng([x["share"] for x in LS]) if LS else "pending") + " of the wrong links, and are still needed (without them the topology F1 falls by " + num(CUT, "{:.2f}") + ").</li></ul></div>")'''
s = s[:i0] + new + s[i1:]
s = s.replace('"<p>The step thins the dense node row (about 9 % of the nodes). The scorer counts 65 % of the dropped nodes as true, and the pictures show', '"<p>The step thins the dense node row (" + pct(AN.get("dropped_share"), d=0) + " of the nodes). The scorer counts " + pct(AN.get("dropped_counted_true_share"), d=0) + " of the dropped nodes as true, and the pictures show')
s = s.replace("the delivered row is as dense (31.5).", "the delivered row is as dense ({m2(Bl4, 'nodes_per_100m', 'nodes', '{:.1f}')}).")
open(p, "w", encoding="utf-8").write(s)
print("patched")
