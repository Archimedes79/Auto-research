"""E206 / E207 sections of the loop-17 report (version 2 after the reviews): the map-conditioned builder trained on perturbed TRAIN rows (arm B with the map, arm C control with the map always zero), and the gate behind it.
The map effect is B with the exact map against B without a map at test time (the same network); C against the champion is the effect of the training alone; B without a map against C is the effect of the map-aware training."""
import html

esc = html.escape


def html_block(g):
    MJ, PERT3, PERT4, NAME, f3, table, S, bs, rb = g["MJ"], g["PERT3"], g["PERT4"], g["NAME"], g["f3"], g["table"], g["S"], g["bs"], g["rb"]
    out = ["<h3>4.7b E206: the same builder trained on perturbed training rows</h3>"]
    out.append("<p><b>Hypothesis.</b> Once the builder has seen perturbed frames of the training scenes, the map helps under real perturbation. <b>Design.</b> Half of every batch is a perturbed training row (perturbed foundation features of all seven failure types, 842 rows each; dense maps from the cross-fitted dense head of the fold that did not train on the row, so they look like those of unseen scenes), half is clean; map situation as in E204; 20 % extra synthetic loss; 4 u from the E204 network. Arm <b>B</b> sees the SD map, arm <b>C</b> sees a zero map in all frames. Evaluation on the validation failure series, whose scenes are not in the training. Three differences are separated below: <b>C against the champion</b> = what the training alone gives; <b>B with the map against B without a map at test time</b> = what the map input gives (the same network); <b>B without a map against C</b> = the effect of having trained with a map 90 % of the time.</p>")
    rows = []
    for p in PERT3:
        c = MJ["perts"][p]["cards"]
        for v, nm in (("single", "single frame"), ("hold", "spatial hold")):
            def cell(k, m="S"):
                return f3(c[k]["fail"][m]) if k in c else "-"

            base = "norec_single" if v == "single" else "norec_hold"
            rows.append([esc(NAME[p]), nm, cell(base), cell(f"m139c_{v}_mnone"), cell(f"m139b_{v}_mnone"), cell(f"m139b_{v}_mwrong"), cell(f"m139b_{v}_m2"), cell(f"m139b_{v}_m0"), cell(f"m139b_{v}_m0", "lanes_found")])
    out.append(table(["failure", "input", "S champion (no map)", "S C (trained, zero map)", "S B, no map at test", "B, wrong map", "B, map error 2 m", "B, exact map", "lanes found, B exact"], rows, nums=set(range(2, 9))))
    rows = []
    for p in PERT4:
        c = MJ["perts"][p]["cards"]
        for v, nm in (("single", "single frame"), ("hold", "spatial hold")):
            def cell(k, m="S"):
                return f3(c[k]["fail"][m]) if k in c else "-"

            rows.append([esc(NAME[p]), nm, cell(f"champ_{v}"), cell(f"m139c_{v}_mnone"), cell(f"m139b_{v}_m0"), cell(f"m139b_{v}_m0", "lanes_found")])
    out.append("<p>The four other failure types (the training saw all seven types on training rows; the reference column is the recursion-on champion, because no recursion-off champion exists for these types, so the comparison is not exactly like for like, by at most 0.01 S):</p>" + table(["failure", "input", "S champion", "S control C (no map)", "S B (exact map)", "lanes found, B"], rows, nums=set(range(2, 6))))
    out.append(narr(g))
    out.append(gate_map_block(g))
    return "\n".join(out)


def html(g):
    return html_block(g)


def narr(g):
    S, f3, MJ, PERT3, NAME, bs, rb = g["S"], g["f3"], g["MJ"], g["PERT3"], g["NAME"], g["bs"], g["rb"]
    rs = lambda tag, ph="fail", sig="0": (lambda c: (sum(v["route_success"] for k, v in c.items() if k.startswith(f"sigma{sig}_") and k.endswith(f"_{ph}")) / max(1, sum(1 for k in c if k.startswith(f"sigma{sig}_") and k.endswith(f"_{ph}")))) if c else float("nan"))(MJ["route"].get(tag, {}))
    d = lambda p, a, b_: S(p, a) - S(p, b_)
    t = []
    t.append(f"<p><b>Result, graph quality.</b> The training alone (C against the champion) gives {d('dark','m139c_single_mnone','norec_single'):+.3f} / {d('blur','m139c_single_mnone','norec_single'):+.3f} / {d('cam_front3_off','m139c_single_mnone','norec_single'):+.3f} S on single frames (night / blur / three cameras off; F1@1 {bs('dark','m139c_single_mnone - norec_single')} at night; lanes found {f3(S('dark','m139c_single_mnone',m='lanes_found'),2)} against {f3(S('dark','norec_single',m='lanes_found'),2)}), and {d('dark','m139c_hold_mnone','norec_hold'):+.3f} / {d('blur','m139c_hold_mnone','norec_hold'):+.3f} / {d('cam_front3_off','m139c_hold_mnone','norec_hold'):+.3f} with the hold: C is a weak control with the hold. The map input (B with the exact map against B without a map) adds {d('dark','m139b_single_m0','m139b_single_mnone'):+.3f} / {d('blur','m139b_single_m0','m139b_single_mnone'):+.3f} / {d('cam_front3_off','m139b_single_m0','m139b_single_mnone'):+.3f} on single frames and {d('dark','m139b_hold_m0','m139b_hold_mnone'):+.3f} / {d('blur','m139b_hold_m0','m139b_hold_mnone'):+.3f} / {d('cam_front3_off','m139b_hold_m0','m139b_hold_mnone'):+.3f} with the hold. B without a map is itself above C with the hold ({d('dark','m139b_hold_mnone','m139c_hold_mnone'):+.3f} / {d('blur','m139b_hold_mnone','m139c_hold_mnone'):+.3f} / {d('cam_front3_off','m139b_hold_mnone','m139c_hold_mnone'):+.3f}): training with the map in the input also helps when the map is taken away. The four milder failure types gain 0.000 to +0.015 S over the champion.</p>")
    t.append(f"<p><b>Result, a bad map.</b> With a map error of 2 m the network is back at the level it has without a map when it runs with the hold (night {f3(S('dark','m139b_hold_m2'))} against {f3(S('dark','m139b_hold_mnone'))}); on single frames 2 m is still {d('dark','m139b_single_m2','m139b_single_mnone'):+.3f} above no map. A wrong map is lower than no map ({f3(S('dark','m139b_hold_mwrong'))} with the hold at night; F1@1 behind the gate {bs('dark','gateB_mwrong - gateB_mnone')} at night, {bs('blur','gateB_mwrong - gateB_mnone')} in blur, {bs('cam_front3_off','gateB_mwrong - gateB_mnone')} with three cameras off). The network does not abstain: nothing tells it the map is wrong, and the label-free agreement of loop 8 is blind at night (AUROC 0.62).</p>")
    t.append(f"<p><b>Result, route success</b> (mean of two route-head seeds, which differ by 0.05 to 0.09; scene-bootstrap intervals of the mean). Champion + hold {rs('e135_route/dark_maxdecay_0.95'):.2f} / {rs('e135_route/blur_maxdecay_0.95'):.2f} / {rs('e135_route/cam_front3_off_maxdecay_0.95'):.2f}; C + hold {rs('e139_route/dark_maxdecay_0.95_C'):.2f} / {rs('e139_route/blur_maxdecay_0.95_C'):.2f} / {rs('e139_route/cam_front3_off_maxdecay_0.95_C'):.2f}; B + hold {rs('e139_route/dark_maxdecay_0.95_B'):.2f} / {rs('e139_route/blur_maxdecay_0.95_B'):.2f} / {rs('e139_route/cam_front3_off_maxdecay_0.95_B'):.2f}; the clean half {rs('e135_route/dark_single','cl'):.2f}. C against the champion with the hold: {rb('dark: C + hold - champion + hold')} / {rb('blur: C + hold - champion + hold')} / {rb('cam_front3_off: C + hold - champion + hold')}; B against C with the hold: {rb('dark: B + hold - C + hold')} / {rb('blur: B + hold - C + hold')} / {rb('cam_front3_off: B + hold - C + hold')}. So at night and in blur the larger part of the route-success gain comes from the training on perturbed rows (C), the map adds a second part. With three cameras off the hold changes the route success of B by {rb('cam_front3_off: B hold - B single')}: the sign depends on the route-head seed, the effect is not established. The 2 m error was applied to the route head only in these runs.</p>")
    t.append("<p><b>Conclusion.</b> Training the builder on perturbed rows is what moved the numbers; the map adds about 0.02 S with the hold at an exact registration and a larger share of the route success. Both statements are upper bounds for a map that is derived from the lane map (section 4.7a).</p>")
    return "\n".join(t)


def gate_map_block(g):
    S, f3, table, MJ, PERT3, NAME, bs = g["S"], g["f3"], g["table"], g["MJ"], g["PERT3"], g["NAME"], g["bs"]
    has = lambda p, k: k in MJ["perts"][p]["cards"]
    out = ["<h3>4.7c E207: the learned gate behind the map-conditioned builder</h3>"]
    out.append("<p><b>Design.</b> The gate of section 4.4, trained again (out-of-fold, 1500 steps) with the map builder B frozen behind it: the fused 42 channels get the five SD channels appended, the loss is the builder loss, the map situation is drawn as in training (exact map with error 0 / 1 / 2 m, wrong 10 %, none 10 %). <b>Result.</b> It adds to B; the map situation decides how much of the whole chain's S is left:</p>")
    rows = []
    for p in PERT3:
        r = [esc(NAME[p]), f3(S(p, "norec_hold")), f3(S(p, "gate"))]
        for k in ("m139b_hold_m0", "gateB_m0", "gateB_m1", "gateB_m2", "gateB_mnone", "gateB_mwrong"):
            r.append(f3(S(p, k)) if has(p, k) else "-")
        r.append(f3(S(p, "gateB_m0", m="lanes_found"), 2) if has(p, "gateB_m0") else "-")
        r.append(f3(S(p, "gateB_m0", "cl")) if has(p, "gateB_m0") else "-")
        rows.append(r)
    out.append(table(["failure", "S champion + hold", "S champion + gate", "S B + hold, exact map", "S B + gate: exact map", "map error 1 m", "map error 2 m", "no map", "wrong map", "lanes found (exact)", "S clean half (exact)"], rows, nums=set(range(1, 11))))
    irows = []
    for p in PERT3:
        irows.append([esc(NAME[p]), bs(p, "gateB_m0 - m139b_hold_m0"), bs(p, "gateB_m0 - gateB_mnone"), bs(p, "gateB_m2 - gateB_mnone"), bs(p, "gateB_mwrong - gateB_mnone"), bs(p, "gateB_m0 - gate")])
    out.append("<p>F1@1 differences with the scene-bootstrap interval (failure half):</p>" + table(["failure", "gate behind B minus B + hold", "exact map minus no map", "2 m minus no map", "wrong map minus no map", "B + gate (exact) minus champion + gate"], irows))
    rs = lambda tag, ph="fail", sig="0": (lambda c: (sum(v["route_success"] for k, v in c.items() if k.startswith(f"sigma{sig}_") and k.endswith(f"_{ph}")) / max(1, sum(1 for k in c if k.startswith(f"sigma{sig}_") and k.endswith(f"_{ph}")))) if c else float("nan"))(MJ["route"].get(tag, {}))
    rrows = []
    for p in PERT3:
        rrows.append([esc(NAME[p]), f3(rs(f"e135_route/{p}_gate")), f3(rs(f"e139_route/{p}_maxdecay_0.95_B")), f3(rs(f"e140_route/{p}_gateB_B")), f3(rs(f"e140_route/{p}_gateB_B", sig="2")), f3(rs(f"e140_route/{p}_gateB_B", "cl"))])
    out.append("<p>Route success of the same chains (mean of two route-head seeds; the 2 m column gives the route head a map with a 2 m error, the builder keeps the exact one; the clean half uses different frames):</p>" + table(["failure", "champion + gate", "B + hold", "B + gate", "B + gate, route-head map error 2 m", "B + gate, clean half"], rrows, nums=set(range(1, 6))))
    out.append("<p><b>Conclusion.</b> The gate adds a further 0.01 to 0.03 S behind B. The whole map chain is at its best with an exact map; with a 1 m error it keeps 36 to 50 % of the S benefit of the exact map (about half in F1@1), with 2 m none.</p>")
    return "\n".join(out)
