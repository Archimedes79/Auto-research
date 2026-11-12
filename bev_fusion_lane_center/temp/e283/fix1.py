import re
p = "build_system_report.py"
s = open(p, encoding="utf-8").read()

# 1. assignment tile and timing tile
s = s.replace("""  f"<div class='tile'><span>Map assignment, lane accuracy with the map 1-2 m off</span><b>{num(g(W[0], 'lane_acc', 'assign_error'))}</b><span>E279 seed 0; geometric control {num((W[0] or {}).get('g', {}).get('assign_error', {}).get('lane_acc') if isinstance((W[0] or {}).get('g', {}).get('assign_error'), dict) else None)}</span></div>"
  f"<div class='tile'><span>Time per frame, head alone</span><b>{num(mean([t.get('total_ms') if isinstance(t, dict) else None for t in TIM]), '{:.1f}', 'see card') if any(TIM) else '24 ms'}</b><span>E273 head, idle RTX 3060, batch 1, cached features (card E273); the soft step is not timed yet</span></div></div>")""",
"""  f"<div class='tile'><span>Map assignment, lane accuracy with the map 1-2 m off</span><b>{m2(W, 'lane_acc', 'assign_error')}</b><span>E279 (mean of 2 seeds); geometric control {num(mean([(c or {}).get('g', {}).get('assign_error', {}).get('lane_acc') for c in W]))}</span></div>"
  f"<div class='tile'><span>Time per frame, head alone</span><b>{num(mean([(t or {}).get('total', {}).get('median') for t in TIM]), '{:.1f}')} ms</b><span>E273 head, idle RTX 3060, batch 1, cached features, median of 60 frames; the soft step and the CPU read-out are not timed</span></div></div>")""")

# 2. speed row
s = s.replace("""    ["Speed", "E273 head alone 24.2 / 23.2 ms per frame (seeds 0 / 1; pixel maps 4.3, nodes 2.9, links 5.0-5.5, sections 2.7-2.8, map encoder + assignment 7.9-8.3); the soft step and the CPU read-out are not timed", "fast enough"],""",
"""    ["Speed", "E273 head alone " + " / ".join(num((t or {}).get('total', {}).get('median'), '{:.1f}') for t in TIM) + " ms per frame (seeds 0 / 1; seed 0: pixel maps " + num((TIM[0] or {}).get('pyr', {}).get('median'), '{:.1f}') + ", nodes " + num((TIM[0] or {}).get('nodes', {}).get('median'), '{:.1f}') + ", links " + num((TIM[0] or {}).get('links', {}).get('median'), '{:.1f}') + ", sections " + num((TIM[0] or {}).get('sections', {}).get('median'), '{:.1f}') + ", map encoder + assignment " + num((TIM[0] or {}).get('assign', {}).get('median'), '{:.1f}') + " ms); the soft step and the CPU read-out are not timed", "fast enough"],""")

# 3. long stats
s = s.replace("""def long_stats():
    out = []
    for x in asg:
        lk = (x or {}).get("result", {}).get("links")
        if not lk:
            return None
        out.append(lk["p>=0.5 long"]["precision"])
    return out


LS = long_stats()""",
"""def long_stats():
    out = []
    for x in asg:
        lk = (x or {}).get("result", {}).get("links")
        if not lk:
            return None
        rc = next(v for v in lk["risk_coverage"] if v["threshold"] == 0.5)
        lg_ = lk["p>=0.5 long"]
        out.append(dict(prec=lg_["precision"], share=lg_["n"] / rc["links"], wrong=lg_["n"] * (1 - lg_["precision"]) / (rc["links"] * (1 - rc["precision"]))))
    return out


LSX = long_stats()
LS = None if LSX is None else [x["prec"] for x in LSX]
rng = lambda vals: "pending" if not vals or None in vals else f"{100 * min(vals):.0f}-{100 * max(vals):.0f} %"  # noqa: E731""")
s = s.replace("""lk = [["Delivered long links, share of all delivered links", "7-8 %", "E276 P9"],""", """lk = [["Delivered long links, share of all delivered links", rng([x["share"] for x in LSX]) if LSX else "pending", "E276 P9"],""")
s = s.replace("""      ["... share of the wrong links that are long links", "27-32 %", "E276 P9"],""", """      ["... share of the wrong links that are long links", rng([x["wrong"] for x in LSX]) if LSX else "pending", "E276 P9"],""")

# 4. E276 row and E274 row
s = s.replace("""f"arm t TIE: the own lane did not gain (t12 F1@1 {two(Tc, 'tier', 'tier')[:0] or ''}see card), far-lane recall and lanes found fell; arm s failed its gate (the per-cell loss cannot place a sparse row)\"""",
              """f"arm t TIE: the own lane did not gain (F1@1 on own lane + neighbours {two(Tc, 't12_F1_1', 'tier')} against {two(Bl4, 't12_F1_1', 'tier')} for B), far-lane recall and lanes found fell ({two(Tc, 'lanes_found')} against {two(Bl4, 'lanes_found')}); arm s failed its gate (the per-cell loss cannot place a sparse row)\"""")
s = s.replace("""CUT: neither arm beat the baseline B on its rule; the long-link loss lifted long-link precision (0.18 to 0.22) but broke the assignment guard""", """CUT: neither arm beat the baseline B on its written rule; the long-link loss lifted the long-link precision but broke the exact-map assignment guard""")

# 5. broken first th
s = re.sub(r"    th = \[\[\"0\.5 \(B today\)\".*?\n", "", s, count=1)
open(p, "w", encoding="utf-8").write(s)
print("fixed")
