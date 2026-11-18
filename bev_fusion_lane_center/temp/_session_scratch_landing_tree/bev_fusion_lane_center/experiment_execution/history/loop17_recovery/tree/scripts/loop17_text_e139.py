"""E206 section of the loop-17 report: the map-conditioned builder trained on REAL failure frames (arm B with the map, arm C control with the map always zero). Tables from results_master.json; the narrative sentences are written by hand after the results (NARR)."""
import html

esc = html.escape
NARR = ""  # filled after the results are in (see below)


def html_block(g):
    MJ, PERT3, PERT4, NAME, f3, table, S = g["MJ"], g["PERT3"], g["PERT4"], g["NAME"], g["f3"], g["table"], g["S"]
    out = ["<h3>4.7b E206: the same builder trained on real failure frames</h3>"]
    out.append("<p><b>Design.</b> Half of every batch is a real failure frame of the TRAIN rows (perturbed foundation features of all seven failure types, 842 rows each, dense maps from the cross-fitted dense head of the fold that did not train on the row, so they look like those of unseen scenes), half is clean; map situation as in E204; 20 % extra synthetic loss. 4 u (4800 iterations) from the E204 network. Arm <b>B</b> sees the SD map, arm <b>C</b> (control) sees a zero map in all frames, so any difference between B and C is the map. Evaluation on the validation failure series, whose scenes are not in the training.</p>")
    rows = []
    for p in PERT3:
        c = MJ["perts"][p]["cards"]
        for v, nm in (("single", "single frame"), ("hold", "spatial hold")):
            def cell(k, m="S"):
                return f3(c[k]["fail"][m]) if k in c else "-"

            base = "norec_single" if v == "single" else "norec_hold"
            rows.append([esc(NAME[p]), nm, cell(base), cell(f"map_{v}_m0"), cell(f"m139c_{v}_mnone"), cell(f"m139b_{v}_mnone"), cell(f"m139b_{v}_mwrong"), cell(f"m139b_{v}_m2"), cell(f"m139b_{v}_m0"), cell(f"m139b_{v}_m0", "lanes_found")])
    out.append(table(["failure", "input", "S champion", "S E204 (exact map)", "S control C (no map)", "B: no map", "B: wrong map", "B: map error 2 m", "B: exact map", "lanes found, B exact"], rows, nums=set(range(2, 10))))
    rows = []
    for p in PERT4:
        c = MJ["perts"][p]["cards"]
        for v, nm in (("single", "single frame"), ("hold", "spatial hold")):
            def cell(k, m="S"):
                return f3(c[k]["fail"][m]) if k in c else "-"

            rows.append([esc(NAME[p]), nm, cell(f"champ_{v}"), cell(f"m139c_{v}_mnone"), cell(f"m139b_{v}_m0"), cell(f"m139b_{v}_m0", "lanes_found")])
    out.append("<p>The four other failure types (not used in the choice of anything; the training saw all seven types on training rows):</p>" + table(["failure", "input", "S champion", "S control C (no map)", "S B (exact map)", "lanes found, B"], rows, nums=set(range(2, 6))))
    out.append(narr(g))
    out.append(gate_map_block(g))
    return "\n".join(out)


def html(g):
    return html_block(g)


def gate_map_block(g):
    """E207: the learned gate behind the map-conditioned builder B"""
    S, f3, table, MJ, PERT3, NAME, esc_ = g["S"], g["f3"], g["table"], g["MJ"], g["PERT3"], g["NAME"], esc
    has = lambda p, k: k in MJ["perts"][p]["cards"]
    out = ["<h3>4.7c E207: the learned gate behind the map-conditioned builder</h3>"]
    out.append("<p><b>Design.</b> The gate of section 4.4, trained again (out-of-fold, 1500 steps) with the map builder B frozen behind it: the fused 42 channels get the five SD channels appended, the loss is the builder loss, the map situation is drawn as in training (exact map with error 0 / 1 / 2 m, wrong 10 %, none 10 %). <b>Result.</b> It adds to B, and the map still decides how much:</p>")
    rows = []
    for p in PERT3:
        r = [esc_(NAME[p]), f3(S(p, "norec_hold")), f3(S(p, "gate"))]
        for k in ("m139b_hold_m0", "gateB_m0", "gateB_m1", "gateB_m2", "gateB_mnone", "gateB_mwrong"):
            r.append(f3(S(p, k)) if has(p, k) else "-")
        r.append(f3(S(p, "gateB_m0", m="lanes_found"), 2) if has(p, "gateB_m0") else "-")
        r.append(f3(S(p, "gateB_m0", "cl")) if has(p, "gateB_m0") else "-")
        rows.append(r)
    out.append(table(["failure", "S champion + hold", "S champion + gate", "S B + hold, exact map", "S B + gate: exact map", "map error 1 m", "map error 2 m", "no map", "wrong map", "lanes found (exact)", "S clean half (exact)"], rows, nums=set(range(1, 11))))
    rs = lambda tag, ph="fail", sig="0": (lambda c: (sum(v["route_success"] for k, v in c.items() if k.startswith(f"sigma{sig}_") and k.endswith(f"_{ph}")) / max(1, sum(1 for k in c if k.startswith(f"sigma{sig}_") and k.endswith(f"_{ph}")))) if c else float("nan"))(MJ["route"].get(tag, {}))
    rrows = []
    for p in PERT3:
        rrows.append([esc_(NAME[p]), f3(rs(f"e135_route/{p}_gate")), f3(rs(f"e139_route/{p}_maxdecay_0.95_B")), f3(rs(f"e140_route/{p}_gateB_B")), f3(rs(f"e140_route/{p}_gateB_B", sig="2")), f3(rs(f"e140_route/{p}_gateB_B", "cl"))])
    out.append("<p>Route success of the same chains (mean of two route-head seeds, SD map for the route head without error; the 2 m column gives the route head a map with a 2 m error, the builder keeps the exact one):</p>" + table(["failure", "champion + gate", "B + hold", "B + gate", "B + gate, route-head map error 2 m", "B + gate, clean half"], rrows, nums=set(range(1, 6))))
    return "\n".join(out)
