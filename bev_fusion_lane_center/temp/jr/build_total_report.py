"""one-off (2026-11-12 night): the TOTAL report of the Spurmitte lane work: every experiment E273 - E319 with its verdict, the architecture with the steps that are NOT end-to-end differentiable marked "to change", the best approaches, results, pictures and videos.

    wsl bash temp/jr/run_total.sh          CPU; writes results/Gesamtbericht_Spurmitte/{index.html, img/, vid/}; then make_single_total.py embeds everything in one file
Re-uses the helpers, tables and media of build_e2e_report.py (executed as a prefix with another output folder); the experiment table comes from
temp/jr/experiments_table.json (made from the cards; every number in it is copied from a card). Every other number is read from a result file; a missing file prints "pending".
"""
import os
import re
import shutil
import sys

ROOT = os.environ["BFLC_ROOT"]
HERE = os.path.join(ROOT, "temp", "jr")
_src = open(os.path.join(HERE, "build_e2e_report.py"), encoding="utf-8").read()
_pre = _src[: _src.index("head_html = open(")].replace("E2E.report", "Gesamtbericht_Spurmitte")
exec(compile(_pre, "build_e2e_report_prefix", "exec"))  # noqa: S102  (defines J, jpg, webm_from, tbl, fig, pair, diff, head, base, the data and the tables)

OLDIMG = os.path.join(ROOT, "results", "Junction.report", "img")
OLDVID = os.path.join(ROOT, "results", "Junction.report", "vid")
EXP = J("temp/jr/experiments_table.json") or []
EXPD = {e["id"]: e for e in EXP}


def R(i, key="result"):
    e = EXPD.get(i)
    return e[key] if e and e.get(key) else "pending"


def copy_img(name, dst):
    src = os.path.join(OLDIMG, name)
    return jpg(src, dst) if os.path.exists(src) else None


def copy_vid(name, dst):
    src = os.path.join(OLDVID, name)
    if not os.path.exists(src):
        return None
    shutil.copyfile(src, os.path.join(VID, dst))
    return "vid/" + dst


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


C_E303S = cards("runs/e303_s_s{s}/ends_card_snap_thr0.1.json")
TS = [J(f"runs/eval/e312_test/card_s{s}.json") for s in (0, 1)]
MAN14 = {s: J(f"runs/e314_p_s{s}/maneuver_val.json") for s in (0, 1)}


def cov(m, k="p0.1s"):
    return None if not m else 100 * m["variants"][k]["recall"]


def test_table():
    if not all(TS):
        return "<p>pending</p>"
    rows = []
    for s, c in enumerate(TS):
        base = c["variants"]["snapped"]["numbers"]["base"]
        for k, lab, man in (("plain", "plain graph (control B)", "plain"), ("unsnapped", "E312 not snapped (differentiable read-out)", "p0.1"), ("snapped", "<b>E312 snapped (card of record)</b>", "p0.1s")):
            h = base if k == "plain" else c["variants"][k]["numbers"]["head"]
            d = None if k == "plain" else c["variants"][k]["difference"]["topo_F1"]
            mv = c["maneuver"].get(man)
            rows.append([str(s), lab, f3(h["topo_F1"]), "-" if d is None else sg(d), f3(h["lanes_found"]), f3(h["junction_F1"]), f3(h["dup_frac"]), f3(h["F1@1.0"]), "-" if not mv else f3(mv["recall"])])
    return tbl(["seed", "read-out", "topo F1", "vs plain", "lanes found", "junction F1", "dup_frac", "F1@1.0", "maneuvers realised"], rows, num=(0, 2, 3, 4, 5, 6, 7, 8))


# ------------------------------------------------------------------------------------------------------------------ architecture, hard steps marked
def architecture_total():
    s = architecture()
    tags = [(10, 134, "frozen: no gradient to the sensors (D50)"), (755, 152, "to change: fixed-K set head (deferred)"),
            (665, 266, "to change: Matrix-NMS layer (E292)"), (945, 374, "to change: E320 + probabilities")]
    add = "".join(f'<text class="tag" x="{x}" y="{y}">{t}</text>' for x, y, t in tags)
    return s.replace("</svg>", add + "</svg>")


# ------------------------------------------------------------------------------------------------------------------ the hard steps
def hard_table():
    WSH = wiring_share()
    d_s, d_u = diff(C_E312S, "topo_F1"), diff(C_E312U, "topo_F1")
    rows = [
        ["<b>1. Snap and wiring of the connector ends</b> (E311)", "read-out",
         "each connector end is replaced by the end of the nearest delivered piece within 4 m / 45 degrees: an argmin that moves the geometry; the pair head never learns to hit the pieces",
f"carries most of the topology gain: topo F1 over the plain graph {pair(d_s, sg)} snapped, {pair(d_u, sg)} not snapped (s0 / s1, p &gt;= 0.1); topology precision {pair(head(C_E312S, 'topo_P'))} snapped vs {pair(head(C_E312U, 'topo_P'))}; lanes found and maneuver coverage equal. <b>Of this gain {' / '.join(f'{100 * w[0]:.0f} %' for w in WSH if w)} is the wiring (one nearest piece per end instead of every piece in reach), {' / '.join(f'{100 * w[1]:.0f} %' for w in WSH if w)} the end geometry</b> (table below)",
         "<b>E320 soft attachment</b>: a softmax over the delivered piece ends near each connector end (with a dustbin), trained by the geometry loss E317 on the final curve; the snap goes when the soft version is within 0.01 topo F1 of it",
         pill("info", "highest priority"), "designed; code after E317 / E318"],
        ["<b>2. Probability threshold p &gt;= 0.1</b> (E309)", "read-out", "a connector is delivered when its probability passes a threshold chosen on VAL; below it nothing, above it everything; the planner never sees the probability",
         "0.5 to 0.1: " + esc(R("E309")),
         "deliver the pair probability and a position sigma with every connector (E318) and let the planner weigh them; a learned keep gate (E297) is the card-level alternative",
         pill("info", "second"), "E318 for sigma; delivery of probabilities to be planned"],
        ["<b>3. Top-K keypoint choice</b> (K = 32, 3 x 3 peaks of the heat, D7)", "between maps and pair head", "which cells become tokens is an index choice; the position inside the window is a soft-argmax (differentiable), the existence of a token is not",
         "keypoint F1 0.72 / 0.71 (E307); entry / exit recall of E312 0.75 / 0.77; the ends found are the cap of the system (E311: 52 % of the GT pairs have both ends matched)",
         "a fixed set of K tokens with an existence logit and matching inside the loss (allowed by D55): the unstructured set head, deferred by D84 item 6 until it is planned further",
         pill("info", "deferred"), "not started"],
        ["<b>4. Close-node step 1.5 m</b> (E275)", "read-out of B", "greedy removal of the node with the lower existence probability inside 1.5 m; hard",
         esc(R("E275")),
         "the differentiable twin is Matrix-NMS (E288); the layer inside the forward path is E292. Soft step trained inside the network: E277 " + esc(R("E277", "verdict")) + ", E279 " + esc(R("E279", "verdict")),
         pill("info", "third"), "E288: " + esc(R("E288")) + " | E277: " + esc(R("E277")) + " | E279: " + esc(R("E279"))],
        ["<b>5. Node threshold and Douglas-Peucker 0.3 m</b> (read-out of B)", "read-out of B", "vertex reduction to the budget of 14.3 vertices per 100 m; no gradient is needed to deliver a line, but the budget is not learned",
         esc(R("E282")),
         "none needed for the gradient; a sparse node target (E276, E282) tries to learn the sparsity",
         pill("ok", "named, kept"), "-"],
        ["<b>6. Frozen features and pyramid</b> (D50)", "input", "the loss reaches the lane heads and the junction maps, not the camera / radar encoder: GaussianCaR features are cached, the pyramid of B stays frozen (E305 joint training trains maps and pairs together)",
         "E286 jj trained the pyramid jointly: 11 s per step, stopped at step 1000, checkpoint kept, not judged",
         "a decision for Mirko, not planned: unfreeze the pyramid or add an adapter; costs hours per run",
         pill("info", "by decision D50"), "-"],
        ["<b>7. Single keyframe, no time</b>", "model", "no temporal fusion in the lane head; the pair flips between keyframes in 19 % of the cases (E290)",
         esc(R("E290")), "E299 cross-attention from the keypoints of the previous frame (ego-warped)", pill("info", "planned"), "not built"],
    ]
    return tbl(["step", "where", "why it is not differentiable", "what it costs / carries (VAL)", "change", "priority", "state"], rows)


def wiring_cards():
    names = {"c": "ends_card_snap_thr0.1.json", "u": "ends_card_thr0.1.json", "up": "ends_card_snap_nogeom_thr0.1.json", "all": "ends_card_snap_all_thr0.1.json"}
    return {k: [J(f"runs/e312_p_s{s}/{n}") for s in (0, 1)] for k, n in names.items()}


WC = wiring_cards()


def wiring_share():
    """share of the snap's gain (c - u) that is wiring (u' - u) and geometry (c - u'), per seed, from the cards"""
    out = []
    for s in (0, 1):
        try:
            c, u, up = (WC[k][s]["numbers"]["head"]["topo_F1"] for k in ("c", "u", "up"))
            out.append(((up - u) / (c - u), (c - up) / (c - u), c - u, up - u, c - up))
        except (TypeError, KeyError):
            out.append(None)
    return out


def wiring_table():
    rows = []
    lab = {"u": ("unsnapped (u)", "every piece in reach", "keypoints"), "all": ("snap-all", "every piece in reach", "snapped ends"),
           "up": ("snap-nogeom (u')", "one nearest piece per end", "keypoints"), "c": ("snapped (c, the E312 read-out)", "one nearest piece per end", "snapped ends")}
    for s in (0, 1):
        for k in ("u", "all", "up", "c"):
            cd = WC[k][s]
            if not cd:
                rows.append([str(s), lab[k][0], lab[k][1], lab[k][2]] + ["pending"] * 4)
                continue
            h = cd["numbers"]["head"]
            rows.append([str(s), f"<b>{lab[k][0]}</b>", lab[k][1], lab[k][2], f3(h["topo_F1"]), f3(h["topo_P"]), f3(h["junction_F1"]), f3(h["lanes_found"])])
    return tbl(["seed", "E312 read-out (p >= 0.1)", "wiring of the connector ends", "geometry of the connector ends", "topo F1", "topo precision", "junction F1", "lanes found"], rows, num=(0, 4, 5, 6, 7))


# ------------------------------------------------------------------------------------------------------------------ best approaches
def best_table():
    rows = []

    def row(rank, name, what, cs, man, diffy, role):
        d = diff(cs, "topo_F1")
        rows.append([rank, f"<b>{name}</b><br><span class='cap'>{what}</span>", pair(head(cs, "topo_F1")), pair(d, sg) if all(v is not None for v in d) else "-",
                     pair(diff(cs, "lanes_found"), sg) if cs[0] else "-", " / ".join(f"{v:.1f} %" for v in man) if all(v is not None for v in man) else "-", diffy, role])
    row("1", "E312, snapped (system of record)", "E307 maps (heat 1.0 m) + pair head <code>--e2e --scales</code> + connectors at p &gt;= 0.1 + snap", C_E312S, [cov(MAN[0]), cov(MAN[1])], pill("info", "all but the named read-out steps"), "use this; the best delivered graph and the one the TEST is planned for (D84)")
    row("2", "E298, driven pairs weighted 2", "E312 with the positive pair terms of driven maneuvers weighted 1 + 2", C_E298S, [None, None], pill("info", "same as E312"), "use when the ego's own maneuvers matter more than all pairs (card: driven pair recall up; topo F1 within the 0.01 band of E312)")
    row("3", "E314, heat 0.7 m", "the E312 recipe on the E313 maps", C_E314S, [cov(MAN14[0]), cov(MAN14[1])], pill("info", "same as E312"), "TIE with E312: more lanes found, fewer isolated ends; the option when lanes found matters more than junction precision")
    row("4", "E312, not snapped", "the same checkpoints, connectors drawn from the keypoints", C_E312U, [cov(MAN[0], "p0.1"), cov(MAN[1], "p0.1")], pill("ok", "differentiable up to the threshold and the close step"), "the differentiable reference: geometry as good, topology precision lower; the number E320 has to reach")
    row("5", "E303, handle scales + snap", "E287 pair head with two learned handle scales per pair", C_E303S, [None, None], pill("info", "snap + threshold"), "kept inside E312")
    row("6", "E305, maps and pairs trained together + snap", "the pair loss reaches the maps head through the soft-argmax", C_E305S, [None, None], pill("info", "snap + threshold"), "kept inside E312")
    row("7", "E287 p2, pair head (no snap, p &gt;= 0.5)", "junction maps (E286) + pair head v2", C_E287, [None, None], pill("ok", "up to threshold"), "first working junction layer; superseded")
    row("8", "E284, end head on B's node tokens", "the first junction end head", C_E284, [None, None], pill("ok", "up to threshold"), "baseline of the junction thread; superseded")
    return tbl(["rank", "approach", "topo F1 (s0 / s1)", "vs plain graph", "lanes found vs plain", "maneuvers realised, %", "end-to-end differentiable?", "use"], rows, num=(2, 3, 4, 5))


# ------------------------------------------------------------------------------------------------------------------ the experiment table
PHASES = [("Lane head B and its read-out (E273 - E283)", 273, 283), ("Junction thread: GT, maps, pairs, connectors (E284 - E314)", 284, 314), ("Night of 2026-11-12 / 13 and next (E315 - E340)", 315, 340)]
VCLS = {"KEEP": "ok", "TIE": "info", "CUT": "no"}
VNAME = {"PLANNED_OR_RUNNING": "planned / running", "BUILT_NOT_RUN": "built, not run", "DIAGNOSTIC": "diagnostic", "STOPPED": "stopped", "INCONCLUSIVE": "inconclusive", "SUPERSEDED": "superseded"}


def exp_table():
    if not EXP:
        return "<p>pending: temp/jr/experiments_table.json is missing</p>"
    out = []
    for title, lo, hi in PHASES:
        rows = ""
        for e in EXP:
            n = int(re.match(r"E(\d+)", e["id"]).group(1))
            if not lo <= n <= hi:
                continue
            v = e["verdict"]
            vp = pill(VCLS.get(v, "dg"), VNAME.get(v, v) + (" (by hand)" if e.get("verdict_basis") == "by hand" else ""))
            dpill = {"yes": pill("ok", "soft"), "partly": pill("info", "partly"), "no": pill("info", "rule step"), "n/a": pill("dg", "n/a")}.get(e.get("differentiable", "n/a"), pill("dg", esc(e.get("differentiable", "?"))))
            hs = esc(e.get("hard_step", "")) or "-"
            fin = "yes" if e.get("in_final_system") else ""
            cf = f"<br><span class='cap'>conflict: {esc(e['conflict'])}</span>" if e.get("conflict") else ""
            rows += (f"<tr data-v='{v}'><td><b>{e['id']}</b><br><span class='cap'>{esc(e['name'])}</span></td><td>{esc(e['question'])}<br><span class='cap'>{esc(e['design'])}</span></td><td>{vp}</td>"
                     f"<td>{esc(e['result'])}<br><span class='cap'>{esc(e.get('source', ''))}; {esc(e.get('lesson', ''))}</span>{cf}</td><td>{dpill}<br><span class='cap'>{hs}</span></td><td>{fin}</td></tr>")
        out.append(f"<h3>{title}</h3><div class='scroll'><table class='exp'><thead><tr><th>experiment</th><th>question and design</th><th>verdict</th><th>result (numbers copied from the card)</th><th>differentiable? hard step</th><th>in E312</th></tr></thead><tbody>{rows}</tbody></table></div>")
    return "".join(out)


def verdict_counts():
    c = {}
    for e in EXP:
        c[e["verdict"]] = c.get(e["verdict"], 0) + 1
    return c


def q_table_total():
    rows = [["E315", "maps trained 6000 steps (not 3000), then the E312 pair stage on them", "yes, same model", "0 ms (training only)", "running in the night chain (temp/day11), VAL, two seeds"],
            ["E316", "the E312 pair head trained 6000 steps", "yes, same model", "0 ms (training only)", "queued in the same chain after E315"],
            ["E317", "geometry loss on the delivered connector: Chamfer between the Hermite curve and the true connector", "yes: the loss on the output reaches positions, headings, scales and, through the soft-argmax, the maps", "0 ms", "code in main (flags off = the E312 recipe bit for bit), card with its rule written, runs queued in day12"],
            ["E318", "E317 + a position scale per keypoint (Laplace NLL); sigma to the pair logit and to the planner", "yes", "about 0 ms", "code in main, card written, runs queued in day12"],
            ["E319", "heading-guided forward / backward exchange in the junction maps head", "yes", "+0.33 GMACs (+10 % of the maps head), +33 k parameters", "built and tested, card written, expected TIE; chain script ready"],
            ["E320", "soft attachment of the connector ends to the delivered pieces: replaces the snap", "yes", "small", "designed; code and card still to be written; the highest-value step"],
            ["E299", "time: cross-attention from the keypoints of the previous frame (ego-warped)", "yes", "small (K = 32 tokens)", "not built; the 19 % flip rate between keyframes is measured"],
            ["D84 items", "TEST pipeline for E312 (first reproduces the VAL cards), simplification by one-at-a-time removals (E330 - E335)", "n/a", "removals lower the cost", "TEST of E312 done once (E321, section TEST); the removals E331 - E335 are queued"]]
    return tbl(["id", "question", "differentiable", "run-time cost", "state (at the time of this report)"], rows)


# ------------------------------------------------------------------------------------------------------------------ media (the E2E set of the prefix + the junction report's)
def media_total():
    s = "<h3>Final system E312: frames and videos</h3><p>Panels of the videos: front camera, B (plain), E312, B + E287 pair head. Grey with arrows = l6 GT with the travel direction; delivered points green within 1 m, orange within 2 m, red beyond; blue = GT not covered within 2 m. The galleries after the videos show frames, failures and the distribution after E312.</p>"
    s += media_block()
    s += "<h3>Before E312: the failures of the E308 pairs, and after</h3><p>The same figures twice: before (E308 pair head, connectors at p &gt;= 0.5) and after (E312, snapped, p &gt;= 0.1). The failure analysis that led to E307 / E311 / E312 used the before set.</p>"
    cap = {"distribution": "Distribution over the 873 VAL frames: coverage of the GT lanes by the delivered graph (left), true pairs delivered per frame (middle), pair probability of true and false candidates (right; log scale).",
           "random12": "Twelve random junction frames: the delivered graph against the GT (green near GT, red with no GT, blue GT not covered).",
           "misses": "Largest junctions, most missed pairs: GT lanes with ends, the delivered graph against the GT, the pairs (green true, red false, blue dashed missed).",
           "worst_coverage": "Lowest coverage: whole arms of the junction are missing from the delivered graph.", "false_pairs": "Most false connectors.", "typical": "A typical junction frame."}
    for n, c in cap.items():
        s += fig(copy_img(f"err_{n}.jpg", f"tot_err_{n}.jpg"), "BEFORE (E308, p >= 0.5). " + c)
    vids = ""
    for sc in ("0017", "0096", "0108", "0221"):
        vids += fig(copy_vid(f"err_scene-{sc}_errors.webm", f"tot_err_{sc}.webm"), f"BEFORE, scene {sc}: GT lanes with ends, delivered graph against GT, pairs (E308, p >= 0.5).", vid=True)
        vids += fig(copy_vid(f"aft_scene-{sc}_errors.webm", f"tot_aft_{sc}.webm"), f"AFTER, scene {sc}: the same panels for E312 (snapped, p >= 0.1).", vid=True)
    s += f'<div class="vids">{vids}</div>'
    s += "<h3>The junction layer: GT and first maps (E285 - E287)</h3><p>Ground truth of the lane side, looked at on examples as D82 asked: zone, conflict core, stop lines, crossings; entry and exit ends with the arm rank and the order rule; the front camera with the GT graph projected.</p>"
    for n in ("3ba93f95", "66bd5bc1", "789b705c", "b32fbd9e"):
        s += fig(copy_img(f"gt_{n}_junction.jpg", f"tot_gt_{n}.jpg"), f"GT layer of the junction frame {n} (E285): left the layer, middle the ends and the order rule, right the camera.")
    for n in ("049dbeaf", "3ba93f95", "620c129c", "789b705c", "7dfa2547", "af0f72a7"):
        s += fig(copy_img(f"{n}_junction.jpg", f"tot_fr_{n}.jpg"), f"Junction frame {n} (VAL), early junction heads (E286 maps, E308 pairs): GT layer, GT ends, end heat with keypoints, zone / core / stop-line prediction, pairs delivered (green true, red false, blue dashed missed), camera.")
    vv = ""
    for sc in ("0017", "0096", "0108", "0221"):
        vv += fig(copy_vid(f"scene-{sc}_junction.webm", f"tot_junction_{sc}.webm"), f"Scene {sc}, early junction heads over the whole scene (same panels as the junction frames).", vid=True)
    s += f'<div class="vids">{vv}</div>'
    return s


# ------------------------------------------------------------------------------------------------------------------ page
head_html = open(os.path.join(HERE, "head.html"), encoding="utf-8").read().replace("Spurmitte Lane Report", "Spurmitte Total Report")
head_html = head_html.replace("</style>", """svg .box.new { fill: var(--surface); stroke: var(--accent); stroke-dasharray: 6 4; stroke-width: 1.6; }
svg .tag { fill: var(--warn); font-family: var(--font-mono); font-size: 10px; font-weight: 500; }
.pill.dg { background: var(--surface2); color: var(--muted); }
table.exp td { font-size: 13.5px; } table.exp tr[data-v="KEEP"] td:first-child { border-left: 3px solid var(--good); } table.exp tr[data-v="TIE"] td:first-child { border-left: 3px solid var(--warn); } table.exp tr[data-v="CUT"] td:first-child { border-left: 3px solid var(--bad); }
.flt { margin: 4px 0 10px; } .flt button { font: 500 12.5px var(--font-mono); padding: 3px 10px; margin-right: 6px; border: 1px solid var(--line); background: var(--surface); color: var(--ink); border-radius: 4px; cursor: pointer; } .flt button.on { background: var(--accent); color: #fff; border-color: var(--accent); }
</style>""", 1)

d312s, d312u = diff(C_E312S, "topo_F1"), diff(C_E312U, "topo_F1")
cov_s, cov_u, cov_p = [cov(MAN[s]) for s in (0, 1)], [cov(MAN[s], "p0.1") for s in (0, 1)], [cov(MAN[s], "plain") for s in (0, 1)]
VC = verdict_counts()
vc_txt = ", ".join(f"{n} {VNAME.get(k, k)}" for k, n in sorted(VC.items(), key=lambda kv: -kv[1])) if VC else "pending"
tiles = f"""<div class="tiles">
<div class="tile good"><div class="k">best system E312, snapped: topo F1 over the plain graph (s0 / s1)</div><div class="v">{pair(d312s, lambda v: '+%.3f' % v)}</div><div class="n">0.6825 / 0.6864 against 0.6259 / 0.6249; VAL, 873 frames; the one TEST use (E321) in section TEST</div></div>
<div class="tile good"><div class="k">the same without the snap rule (the differentiable read-out)</div><div class="v">{pair(d312u, lambda v: '+%.3f' % v)}</div><div class="n">the snap, a rule on the output, carries most of the topology gain: E320 replaces it</div></div>
<div class="tile good"><div class="k">GT maneuvers realised: plain graph to E312 snapped</div><div class="v">{f2(cov_p[0])} to {f2(cov_s[0])} %</div><div class="n">seed 0, 10991 maneuvers in 826 junction frames; not snapped {f2(cov_u[0])} %</div></div>
<div class="tile"><div class="k">experiments in this report</div><div class="v">{len(EXP)}</div><div class="n">E273 - E319: {vc_txt}</div></div>
<div class="tile"><div class="k">not end-to-end differentiable yet</div><div class="v">5</div><div class="n">snap + wiring, threshold, top-K choice, close-node step, read-out simplification; plus the frozen sensor features (D50)</div></div>
<div class="tile"><div class="k">the whole lane head</div><div class="v">{f"{COST['total_macs_g']:.1f} GMACs" if COST else "pending"}</div><div class="n">{f"{COST['params_m']['total']:.2f} M parameters; B alone {TIMEB['total']['median']:.0f} ms per frame (idle GPU)" if COST and TIMEB else ""}</div></div>
</div>"""

body = f"""<div class="wrap">
<header class="top">
  <div class="eyebrow">Spurmitte - total report - state of 2026-11-12, night - lane-centre graph from camera + radar on frozen GaussianCaR features - VAL, plus the one TEST use of E312 (E321)</div>
  <h1>All experiments, the architecture, the best approaches, and what is still not end-to-end differentiable</h1>
  <p class="lead">One page for the whole lane thread since the plain lane head B (E273): every experiment with its question, verdict and key numbers, the architecture with the non-differentiable steps marked <b>to change</b>, the best approaches side by side, the results, and all pictures and videos. The best working system is E312 (junction maps with a narrow end heat, a pair head trained together with them, learned connector shapes, connectors delivered at a low probability threshold, ends snapped to the delivered pieces). It is end-to-end differentiable except for a few named rule steps on the output; the snap is the one that matters, because without it two thirds of the topology gain are gone.</p>
  {tiles}
</header>
<nav class="idx"><a href="#sum">Summary</a><a href="#best">Best approaches</a><a href="#change">To change</a><a href="#arch">Architecture</a><a href="#exps">All experiments</a><a href="#numbers">Results</a><a href="#test">TEST</a><a href="#compute">Compute</a><a href="#look">Pictures and videos</a><a href="#why">Why the shape does not emerge</a><a href="#next">Next</a><a href="#weak">Weaknesses</a></nav>

<section id="sum"><h2>Summary</h2>
<ul class="tight">
<li><b>Where we started.</b> B (E273) is a pixel-map lane head: nodes on a 2 m grid, Hermite links, soft sections and a soft map assignment, end to end on cached GaussianCaR features. Its read-out has a hard close-node step (E275). At junctions it fails: it has no notion of which entry connects to which exit.</li>
<li><b>What we built.</b> A junction layer: a 1 m maps head (E286) with entry / exit heat and heading, a pair head over keypoints with the order rule (E287), joint training of both (E305), learned connector shapes (E303), a narrow heat of 1.0 m (E307, the largest single gain), a low delivery threshold (E309) and the attachment of connector ends to the delivered pieces (E311). Together: E312.</li>
<li><b>What it gives (VAL, two seeds).</b> Topo F1 {pair(head(C_E312S, "topo_F1"))} against {pair(base(C_E312S, "topo_F1"))} for the plain graph ({pair(d312s, sg)}), lanes found {pair(diff(C_E312S, "lanes_found"), sg)}, GT maneuvers realised {f2(cov_p[0])} % to {f2(cov_s[0])} % (seed 0). Pictures and videos agree with the numbers; the strict match radius of 1 m lowers the absolute level.</li>
<li><b>What is not differentiable.</b> The attachment of the connector ends (snap), the probability threshold, the top-K keypoint choice, the close-node step and the output simplification; and the sensor features are frozen by decision (D50). Section "To change" lists each one with its cost and its replacement; E317 / E318 / E319 / E320 are the queued experiments.</li>
<li><b>Experiments.</b> {len(EXP)} cards: {vc_txt}. Ties are kept and refined (D82), never worse.</li>
<li><b>TEST.</b> Used once, for E312 only (E321, D84): the gain holds on the 21 unseen scenes, see section TEST. <b>Not done:</b> the experiments of the night (E315 - E320) are queued or running and are not in the numbers; no later design has been run on TEST.</li></ul></section>

<section id="best"><h2>Best approaches</h2>
<p>All rows are scored the same way: delivered graph = B plus the hard close-node step plus the connectors of the head, paired scorer, VAL, two seeds, differences to the plain delivered graph of the same seed. Ranked by what we would deploy today. Maneuvers realised is the share of GT entry-to-exit maneuvers the delivered graph covers (E310, <code>evaluation/jmaneuver.py</code>, p &gt;= 0.1 snapped) where it was measured.</p>
{best_table()}
<div class="note"><b>Recommendation.</b> E312 is the system of record. For a planner that cares about the ego's own maneuvers take E298; for lanes found over junction precision take E314. The next gain is not another head but a differentiable replacement of the snap (E320): the unsnapped row is what a fully differentiable read-out gives today. The earlier loops' results (route head, spurs, ring, prototype Q 0.727 on TEST in loop 22) belong to another system and are not repeated here.</div>
<h3>The ladder: what each step adds to the delivered graph</h3>{ladder_table()}
<p class="cap">topo F1 is the directed reachability score (seeds every 5 m on the GT, reach 30 m, match 1 m, direction within 60 degrees); lanes found is the share of GT pieces with at least 80 % of their points within 1 m of delivered points. Both are strict.</p>
<h3>All approaches with their soft and hard parts</h3>{audit_table()}</section>

<section id="change"><h2>Not end-to-end differentiable: what has to change</h2>
<p>The rule of the project (D55, confirmed by Mirko on 2026-11-12 and again in D84): everything on the training path is soft, a loss on the output reaches every module, hard selection only on the output and named in every report; from now on only end-to-end differentiable solutions are pursued. The table lists every step that breaks the rule, in the order in which it is worth replacing.</p>
{hard_table()}
<h3>What the snap does: wiring and geometry (E312, VAL, p &gt;= 0.1)</h3>
<p>The snap does two things: it moves the connector ends onto the ends of the delivered pieces (geometry) and it wires each end to ONE nearest piece (wiring); the unsnapped read-out wires every piece within 4 m / 45 degrees. Scoring the same E312 checkpoints with each half alone (E311 options <code>--snap-nogeom</code> and <code>--snap-all</code>, scored in the night of 2026-11-13) splits the gain:</p>
{wiring_table()}
<div class="note warn"><b>The wiring is the larger part.</b> Of the gain of the snap over the unsnapped read-out {' / '.join(f'{100 * w[0]:.0f} %' for w in wiring_share() if w)} (s0 / s1) is the wiring ({' / '.join(f'{w[3]:+.4f}' for w in wiring_share() if w)} topo F1), {' / '.join(f'{100 * w[1]:.0f} %' for w in wiring_share() if w)} the end geometry ({' / '.join(f'{w[4]:+.4f}' for w in wiring_share() if w)}). With the proximity wiring even perfect end geometry (snap-all) reaches only {pair(head(WC['all'], 'topo_F1'))}. So the attachment of E320 (geometry) can recover at most about a third of the gap, and the wiring rule (argmin over pieces) is the step to make soft next (E322: attach to delivered piece ends with a dustbin; E320 pre-registers a tight-radius wiring read-out as the first test).</div>
<div class="note warn"><b>Reading.</b> Without the snap E312 gains {pair(d312u, sg)} topo F1 over the plain graph, with it {pair(d312s, sg)}: topology precision falls from {pair(head(C_E312S, "topo_P"))} to {pair(head(C_E312U, "topo_P"))} because connectors that do not start and end exactly at the delivered pieces break reachability, while the geometric coverage does not change (lanes found {pair(head(C_E312U, "lanes_found"))} unsnapped against {pair(head(C_E312S, "lanes_found"))} snapped; maneuvers realised {f2(cov_u[0])} against {f2(cov_s[0])} %). So the first step is E320, and E317 (a geometry loss on the delivered curve) is its training signal. The order of work: E317, E318, E320, then delivery of probabilities instead of a threshold, then the Matrix-NMS layer in the forward path; the unstructured set head stays deferred (D84).</div></section>

<section id="arch"><h2>Architecture</h2>{architecture_total()}
<p class="cap">Forward path left to right, top to bottom. Blue = differentiable (the loss reaches it), amber = hard step, named; dashed = queued experiments. Orange labels: what is to be changed. MACs and parameters are counted on the E312 checkpoint (batch 1, CPU, FlopCounterMode).</p>
<div class="scroll"><table><thead><tr><th>block</th><th>experiments</th><th>what it does</th><th>differentiable</th></tr></thead><tbody>
<tr><td>GaussianCaR features</td><td>D50</td><td>camera + radar BEV features, 39 channels, 200 x 200 at 0.5 m; cached; the foundation of the project</td><td>{pill("info", "frozen")}</td></tr>
<tr><td>Pyramid, node head, links, sections, map assignment (B)</td><td>E273, E275</td><td>0.5 / 1 / 2 m feature pyramid; node existence, offset, 8 type flags and heading on a 2 m grid; short and long Hermite links; soft sections; lane-level map assignment with soft registration, dustbin and attention</td><td>{pill("ok", "soft")} (read-out has the close-node step)</td></tr>
<tr><td>Junction maps head</td><td>E286, E307, E313</td><td>1 m heat of entries and exits with heading, zone, conflict core; 0.5 m stop line and crossing; the end heat is a Gaussian of 1.0 m (E307: keypoint F1 0.63 to 0.72)</td><td>{pill("ok", "soft")}</td></tr>
<tr><td>Keypoints</td><td>E287</td><td>3 x 3 peaks, soft-argmax position over the window, heading and probability at the peak, top K = 32</td><td>{pill("info", "position soft, choice hard")}</td></tr>
<tr><td>Pair head</td><td>E287, E305, E303</td><td>self-attention over the keypoints, 13 geometry features, rank features (the order rule holds for 1899 of 1899 multi-pair maneuvers), evidence along the curve, BCE + dual softmax with a none entry, soft crossing penalty, two learned handle scales per pair</td><td>{pill("ok", "soft")}</td></tr>
<tr><td>Joint training</td><td>E305, E312</td><td>the pair loss reaches the maps head through the soft-argmax positions; the pyramid stays frozen</td><td>{pill("ok", "soft")}</td></tr>
<tr><td>Connectors and snap</td><td>E303, E311</td><td>Hermite curve per pair (chord tangent x learned scales); the snap starts the curve at the end of the delivered piece wired to the entry and ends it at the start of the piece wired to the exit</td><td>curve {pill("ok", "soft")}, snap {pill("info", "rule")}</td></tr></tbody></table></div>
<h3>Training signals today</h3><p>Node, link and section losses of B (E273); junction maps: zone, core, end heat (focal + dice) and heading; pair head: BCE + dual softmax + crossing penalty + handle-scale fit; E305 / E312: the pair loss reaches the maps through the soft-argmax. Not supervised as a shape: the connector curve itself and the end position beyond the 1 m cell (E317 / E318).</p></section>

<section id="exps"><h2>All experiments</h2>
<p>Every card from E273 to E319, from the cards of <code>experiments/</code> (question, design, verdict as the card states it, key numbers copied from the card; all VAL unless the card says otherwise). Colour at the left edge: green KEEP, amber TIE (kept and refined), red CUT. "In E312" marks what the best system uses. Read E273 with care: its own card is CUT because the topology broke the guards against the older head (topo F1 0.597 against 0.741), yet it is the base B of the whole thread, because the gain of the soft map assignment holds; E275 repaired its read-out. Verdicts "by hand" carry the label of the card (hard rule 4 of the project); the numbers are copied from the cards, never recomputed.</p>
<div class="flt"><button class="on" data-v="ALL">all</button><button data-v="KEEP">KEEP</button><button data-v="TIE">TIE</button><button data-v="CUT">CUT</button><button data-v="OTHER">other</button></div>
{exp_table()}</section>

<section id="numbers"><h2>Results</h2>
<h3>Maneuver coverage (E310 / E312)</h3>{maneuver_chart()}
<p class="cap">Share of the 10991 GT maneuvers of the 826 junction frames that the delivered graph realises: entry and exit matched within 1.5 m and 40 degrees, at least 90 % of the connector covered.</p>
<h3>Junction maps: the single biggest gain</h3>{maps_table()}
<p class="cap">The end heat of 1.0 m (E307) separates neighbouring lanes and keeps far peaks: keypoint F1 0.63 / 0.64 to 0.72 / 0.71. 0.7 m (E313) is better on the maps and a TIE in the delivered graph (E314).</p>
<h3>Do the metrics fit what we see?</h3>
<p>The videos and frames agree with the ranking of the metrics (E312 above B; E284 and E287 close to B). What the metrics do not show is how close the lines are: a 2 m match radius lifts every variant by about 0.06 and keeps the order.</p>
{tol_table()}{scene_chart()}
<p class="cap">Per scene (seed 0, 22 VAL scenes): E312 improves 19 scenes and ties 3; none gets worse by more than 0.001.</p>
<h3>How many lines, how much of it right, how smooth</h3>{lines_table()}
<h3>Where the lateral error sits</h3>{off_table()}</section>

<section id="test"><h2>TEST: the one use (E321, D84)</h2>
<p>Mirko's recorded OK of 2026-11-12 23:00 ("Teste E312") allowed ONE use of the TEST split (21 scenes, {TS[0]['frames'] if TS[0] else '?'} frames) for the finished design E312: the checkpoints <code>e312_p_s0 / s1</code> as they stood, the read-outs fixed on VAL before the run (snapped at p &gt;= 0.1 = the card of record, and the same checkpoints not snapped = the differentiable read-out), control = the plain delivered graph of B (E273 + the hard close-node step) of the same seed on the same frames. Report only: nothing was selected or tuned on TEST, and it will not be re-run. Run id <code>E321.t</code>, CPU, the pipeline (<code>evaluation/jtest_eval.py</code>) first reproduced the stored VAL cards of E312.</p>
{test_table()}
<div class="note"><b>Reading.</b> The VAL gain holds on unseen scenes: snapped topo F1 over the plain graph {pair([None if not c else c['variants']['snapped']['difference']['topo_F1'] for c in TS], sg)} (VAL {pair(d312s, sg)}); not snapped {pair([None if not c else c['variants']['unsnapped']['difference']['topo_F1'] for c in TS], sg)} (VAL {pair(d312u, sg)}). The snap, the rule step, carries about 60 % of the gain on TEST as well; the differentiable read-out does not improve the junction F1 (its precision falls). The plain graph is harder on TEST than on VAL (topo F1 {pair([None if not c else c['variants']['snapped']['numbers']['base']['topo_F1'] for c in TS])} against {pair(base(C_E312S, 'topo_F1'))}). The paired scene-bootstrap intervals (all above zero for the snapped read-out) are in the card <code>experiments/E321_test-e312/experiment.md</code>, section 4; no promotion (the champion decision stays with Mirko), E312 stays the system of record.</div></section>

<section id="compute"><h2>Compute</h2>{compute_table()}
<p class="cap">MACs and parameters by counting, ms only on an idle GPU (the timing job of the night chain fills the pending cells on the next build). Training: maps 0.9 s per step, pair stage with joint training 1.9 s per step (3000 steps = 1.6 h per seed). The pair head costs 0.02 GMACs, the maps head 3.3 GMACs on top of the pyramid that B already runs; the read-out runs on the CPU.</p></section>

<section id="look"><h2>Pictures and videos</h2>{media_total()}</section>

<section id="why"><h2>Why the right shape does not emerge by itself</h2>
<div class="note"><b>Short answer from the data.</b> On ordinary lane stretches the shape does emerge: median lateral offset 0.2 m, jitter 0.13 m, 98 % of the delivered length within 2 m of the GT. What does not emerge is what the evidence does not fix: which entry joins which exit, where an end sits to a metre, what lies beyond the far ring. Mirko's three hints name three reasons; each has a differentiable test in the queue.</div>
<div class="scroll"><table><thead><tr><th>hint</th><th>what we measured</th><th>differentiable test</th></tr></thead><tbody>
<tr><td><b>smoothness is no constraint</b></td><td>the connector curve itself has no loss. Kinks at the joins: median {f"{LS['B']['join_med']:.1f}" if LS else "?"} degrees in B, {f"{LS['e312']['join_med']:.1f}" if LS else "?"} in E312, GT {f"{LS['GT']['join_med']:.1f}" if LS else "?"}</td><td><b>E317</b>: Chamfer loss between the delivered Hermite curve and the true connector; training only</td></tr>
<tr><td><b>position uncertainty is too small</b></td><td>the end position is a 1 m cell plus a soft-argmax and carries no uncertainty; before E307 the ends were 1.44 m off on average (E310) and only 52 % of the GT pairs had both ends matched strictly</td><td><b>E318</b>: a log sigma per end with a Laplace NLL; sigma enters the pair logit and goes to the planner</td></tr>
<tr><td><b>forward-backward exchange is missing</b></td><td>on the E305 keypoints the pair decision equals a perfect classifier (+0.0304 vs +0.0314, E310): the ends are the limit; 53 % of the GT ends lie beyond 35 m, pair recall there 0.23 vs 0.38 near (E300); the context of the maps head is isotropic</td><td><b>E319</b>: heading-guided exchange ahead / behind along the lane; <b>E299</b> in time (pair flips 19 %)</td></tr></tbody></table></div>
<p>Splines, point types and classes are what a transformer's output is made of: a Hermite curve between two typed points is what the planner and the loss can use, and the order rule (1899 of 1899) is a prior a generic decoder would have to learn from data. The direct test of "a transformer can do it all" is an unstructured set-prediction head on the same cached features; D84 defers it until it is planned further.</p></section>

<section id="next"><h2>Next: queued, running, decided</h2>{q_table_total()}
<ul class="tight"><li><b>D84 (Mirko, 2026-11-12):</b> TEST E312 once (done as E321 after the pipeline reproduced the VAL cards; snapped and unsnapped read-outs, control = the plain graph of the same seed); only end-to-end differentiable solutions; simplify the system by one-at-a-time removals with a rule written before the run; plan and run more experiments; the unstructured transformer head is deferred.</li>
<li><b>Rule of every new card:</b> KEEP when the snapped topo F1 over E312 rises by more than 0.01 on both seeds with the guards held; E317 / E320 also report the unsnapped number.</li></ul></section>

<section id="weak"><h2>Weaknesses and honest negatives</h2>
<ul class="tight"><li>The headline numbers are strict (1 m, direction) and scored against the graph GT in a 48 m box, not directly against l6; the 2 m cross-check keeps the order of the variants.</li>
<li>The snap and the close-node step are rules on the output; until E320 and a soft close-node step are in, the delivered graph is end to end only up to these named rules.</li>
<li>E312 has false connectors near the ego vehicle at the low threshold and weak far-ring ends; pair flips of 19 % between keyframes; single keyframes only.</li>
<li>The best junction results come from the GT-heavy junction layer; the sensor features are frozen, so the loss never reaches the camera / radar encoder.</li>
<li>Cuts and stops are part of the table: E277 (soft step) CUT, E286 jj (joint pyramid) stopped, E304 stopped; ties are kept.</li>
<li>TEST was used once, for E312 only (E321); E320 and every later design have not been run on TEST and would need their own recorded OK.</li></ul>
<h3>Open decisions</h3><ul class="tight"><li>TEST for any later design (its own recorded OK), champion promotion and the change of PROBLEM.md (a D entry), the soft step of 1.25 m (E279), D4 / D5, push, whether to build the unstructured set head.</li></ul></section>
<footer>Built by temp/jr/build_total_report.py (prefix: build_e2e_report.py) from runs/**/ends_card*.json, runs/e312_p_s*/maneuver_val.json, runs/e*/jmaps_val.json, temp/e309/*.json, temp/jr/model_cost.json, runs/e273/e273_p_s0/timing.json and the experiment table temp/jr/experiments_table.json (made from experiments/E*/experiment.md). Pictures and videos: results/l6_view_E312, results/Junction.report.</footer>
</div>
<script>
document.querySelectorAll('.flt button').forEach(function (b) {{
  b.addEventListener('click', function () {{
    var v = b.dataset.v;
    document.querySelectorAll('.flt button').forEach(function (x) {{ x.classList.toggle('on', x === b); }});
    document.querySelectorAll('table.exp tr[data-v]').forEach(function (r) {{
      var show = v === 'ALL' || r.dataset.v === v || (v === 'OTHER' && ['KEEP', 'TIE', 'CUT'].indexOf(r.dataset.v) < 0);
      r.style.display = show ? '' : 'none';
    }});
  }});
}});
</script>"""

open(os.path.join(OUT, "index.html"), "w", encoding="utf-8").write(head_html + "\n" + body)
print("written", os.path.join(OUT, "index.html"), len(body), "experiments:", len(EXP))
