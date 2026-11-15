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
            (665, 266, "to change: Matrix-NMS layer (E292)"), (945, 374, "to change: wiring head (E325 / E350) + probabilities")]
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
         "<b>E320 measured (VAL, 2026-11-13): the learned attachment replaces the geometry of the snap</b> (topo F1 of the attached read-out with the snap's one-piece wiring: a - c = -0.0021 / +0.0025, a tie with the snap; a - u' = +0.0097 / +0.0130) but the card is CUT by its duplicates guard (dup_frac +0.0184 on seed 0). E323: the excess is not the curve count. E322 gate G0: a learned selection of the wired piece has no accuracy headroom (the one-piece rule is a count effect and a named selection on the output, allowed by D55). <b>E325 (KEEP, 2026-11-14): a 545-parameter learned head replaces the nearest-piece rule at equal-or-better accuracy on E312 (+0.0023 / +0.0032 snapped topo F1); its final choice is still an argmax and it is trained on stored records, not jointly; E350 plans the transfer to E324 s</b>",
         pill("info", "highest priority"), "geometry: E320 / E324 measured; wiring: E325 KEEP on E312, E350 planned"],
        ["<b>2. Probability threshold p &gt;= 0.1</b> (E309)", "read-out", "a connector is delivered when its probability passes a threshold chosen on VAL; below it nothing, above it everything; the planner never sees the probability",
         "0.5 to 0.1: " + esc(R("E309")),
         "deliver the pair probability and a position sigma with every connector (E318) and let the planner weigh them; a learned keep gate (E297) is the card-level alternative",
         pill("info", "second"), "E318 (position scale) was CUT: the scale is informative but moves no graph number; delivery of probabilities to the planner still to be planned"],
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


def e324_table():
    spec = (("E312 snapped (c, the system of record)", "e312_p_s{s}", "ends_card_snap_thr0.1.json"), ("E316 snapped (E312 recipe, 6000 steps)", "e316_p_s{s}", "ends_card_snap_thr0.1.json"),
            ("E320 primary (attached ends, one-piece wiring)", "e320_a_s{s}", "ends_card_snap_nogeom_attach_thr0.1.json"),
            ("<b>E324 s primary (attached ends, one-piece wiring)</b>", "e324_s_s{s}", "ends_card_snap_nogeom_attach_thr0.1.json"),
            ("<b>E324 s own keypoints, snapped (the E312 read-out type)</b>", "e324_s_s{s}", "ends_card_snap_thr0.1.json"),
            ("E324 s own keypoints, one-piece wiring, no snap geometry", "e324_s_s{s}", "ends_card_snap_nogeom_thr0.1.json"))
    rows = []
    for s in (0, 1):
        for lab, run, f in spec:
            c = J(f"runs/{run.format(s=s)}/{f}")
            if not c:
                rows.append([str(s), lab] + ["pending"] * 6)
                continue
            h = c["numbers"]["head"]
            rows.append([str(s), lab, f3(h["topo_F1"]), f3(h["topo_P"]), f3(h["junction_F1"]), f3(h["lanes_found"]), f3(h["dup_frac"]), f3(h["F1@1.0"])])
    return tbl(["seed", "read-out (p >= 0.1, VAL)", "topo F1", "topo precision", "junction F1", "lanes found", "dup_frac", "F1@1.0"], rows, num=(0, 2, 3, 4, 5, 6, 7))


def e320_table():
    rows = []
    spec = (("E312 snapped (c, the card of record)", "e312_p_s{s}", "ends_card_snap_thr0.1.json"), ("E312 snap-nogeom (u')", "e312_p_s{s}", "ends_card_snap_nogeom_thr0.1.json"),
            ("<b>E320 primary: attached ends + one-piece wiring</b>", "e320_a_s{s}", "ends_card_snap_nogeom_attach_thr0.1.json"), ("E320 own keypoints snapped", "e320_a_s{s}", "ends_card_snap_thr0.1.json"),
            ("E320 attached, proximity wiring (secondary)", "e320_a_s{s}", "ends_card_attach_thr0.1.json"), ("E320 attached, tight wiring 1.5 m", "e320_a_s{s}", "ends_card_attach_r1.5_thr0.1.json"),
            ("E320 attached, tight wiring 1.0 m", "e320_a_s{s}", "ends_card_attach_r1.0_thr0.1.json"))
    for s in (0, 1):
        for lab, run, f in spec:
            c = J(f"runs/{run.format(s=s)}/{f}")
            if not c:
                rows.append([str(s), lab] + ["pending"] * 6)
                continue
            h = c["numbers"]["head"]
            rows.append([str(s), lab, f3(h["topo_F1"]), f3(h["topo_P"]), f3(h["junction_F1"]), f3(h["lanes_found"]), f3(h["dup_frac"]), f3(h["F1@1.0"])])
    return tbl(["seed", "read-out (p >= 0.1)", "topo F1", "topo precision", "junction F1", "lanes found", "dup_frac", "F1@1.0"], rows, num=(0, 2, 3, 4, 5, 6, 7))


G0TXT = "-0.0007 / +0.0013 topo F1 over the nearest-piece rule (E312 s0 / s1, card E322); the same piece is chosen in 84 % of the ends"


# ------------------------------------------------------------------------------------------------------------------ best approaches
def dif(a, b, k="topo_F1"):
    return [None if None in (x, y) else x - y for x, y in zip(head(a, k), head(b, k))]


def best_table():
    rows = []

    def row(rank, name, what, cs, man, diffy, role):
        d = diff(cs, "topo_F1")
        rows.append([rank, f"<b>{name}</b><br><span class='cap'>{what}</span>", pair(head(cs, "topo_F1")), pair(d, sg) if all(v is not None for v in d) else "-",
                     pair(diff(cs, "lanes_found"), sg) if cs[0] else "-", " / ".join(f"{v:.1f} %" for v in man) if all(v is not None for v in man) else "-", diffy, role])
    C_E325W = cards("runs/e312_p_s{s}/ends_card_e325w_snap_thr0.1.json")
    C_E341P = cards("runs/e341_p_s{s}/ends_card_snap_thr0.1.json")
    row("1", "E324 s, snapped own keypoints (system of record, D94)", "E312 recipe + geometry loss (<code>--geom-w 0.5</code>) + learned attachment + 6000 pair steps; read out like E312 (snapped, p &gt;= 0.1)", C_E324S, [None, None], pill("info", "as E312 + a learned attachment; read-out steps named"),
        f"use this: the VAL reference of every new arm; {pair(dif(C_E324S, C_E312S), sg)} topo F1 over E312, guards held; c2 (B fine-tuned) and the stack E338 are candidates next to it; adoption in the protected files and any TEST need Mirko's recorded OK (E343 on hold)")
    row("2", "E312, snapped (the base system; control of the one TEST use)", "E307 maps (heat 1.0 m) + pair head <code>--e2e --scales</code> + connectors at p &gt;= 0.1 + snap", C_E312S, [cov(MAN[0]), cov(MAN[1])], pill("info", "all but the named read-out steps"), "the older system of record and the control of E321 (section TEST); the recipe E324 s refines")
    row("3", "E325, E312 + the learned wiring head", "the nearest-piece rule of the snap replaced by a 545-parameter softmax head over the candidate places; trained on stored records, not jointly; the final choice is an argmax", C_E325W, [None, None], pill("info", "as E312; the wiring is a learned argmax"),
        f"KEEP: {pair(dif(C_E325W, C_E312S), sg)} snapped topo F1 over E312 at equal guards; the end-to-end replacement of a rule step; the transfer to E324 s is E350 (planned)")
    row("4", "E341, E312 recipe without the 0.5 m paint branch", "evidence along the curve with its four non-paint inputs; maps head -1.477 GMACs (-18.6 % of the system), -0.037 M parameters", C_E341P, [None, None], pill("info", "as E312"),
        f"ADOPT on the E312 recipe: {pair(dif(C_E341P, C_E312S), sg)} snapped topo F1 against E312, inside the band; the transfer to E324 s (E348) was NOT adopted by its written rule (one dup_frac guard)")
    row("5", "E298, driven pairs weighted 2", "E312 with the positive pair terms of driven maneuvers weighted 1 + 2", C_E298S, [None, None], pill("info", "same as E312"), "use when the ego's own maneuvers matter more than all pairs (card: driven pair recall up; topo F1 within the 0.01 band of E312)")
    row("6", "E314, heat 0.7 m", "the E312 recipe on the E313 maps", C_E314S, [cov(MAN14[0]), cov(MAN14[1])], pill("info", "same as E312"), "TIE with E312: more lanes found, fewer isolated ends; the option when lanes found matters more than junction precision")
    row("7", "E312, not snapped", "the same checkpoints, connectors drawn from the keypoints", C_E312U, [cov(MAN[0], "p0.1"), cov(MAN[1], "p0.1")], pill("ok", "differentiable up to the threshold and the close step"), "the differentiable reference: geometry as good, topology precision lower")
    row("8", "E303, handle scales + snap", "E287 pair head with two learned handle scales per pair", C_E303S, [None, None], pill("info", "snap + threshold"), "kept inside E312")
    row("9", "E305, maps and pairs trained together + snap", "the pair loss reaches the maps head through the soft-argmax", C_E305S, [None, None], pill("info", "snap + threshold"), "kept inside E312")
    row("10", "E287 p2, pair head (no snap, p &gt;= 0.5)", "junction maps (E286) + pair head v2", C_E287, [None, None], pill("ok", "up to threshold"), "first working junction layer; superseded")
    row("11", "E284, end head on B's node tokens", "the first junction end head", C_E284, [None, None], pill("ok", "up to threshold"), "baseline of the junction thread; superseded")
    return tbl(["rank", "approach", "topo F1 (s0 / s1)", "vs plain graph", "lanes found vs plain", "maneuvers realised, %", "end-to-end differentiable?", "use"], rows, num=(2, 3, 4, 5))


# ------------------------------------------------------------------------------------------------------------------ the experiment table
PHASES = [("Lane head B and its read-out (E273 - E283)", 273, 283), ("Junction thread: GT, maps, pairs, connectors (E284 - E314)", 284, 314), ("Night of 2026-11-12 / 13: longer training, the differentiable attachment, the one TEST use (E315 - E329)", 315, 329), ("System inventory, simplification arms, stack check (E330 - E340)", 330, 340), ("Update 2026-11-15: removals on the system of record, the TEST of E324 s, plans (E341 - E350)", 341, 350)]
VCLS = {"KEEP": "ok", "TIE": "info", "CUT": "no", "ADOPT": "ok", "NOT_ADOPTED": "no", "ON_HOLD": "info"}
D87_KEPT = {"E320": "the only differentiable replacement of the snap geometry; topo F1 ties the snap; the duplicates are to be fixed (E324)", "E322": "the only end-to-end replacement of the output selection; to be built at the equal-accuracy bar (the gate G0 stays as its result)"}
VNAME = {"PLANNED_OR_RUNNING": "planned / running", "BUILT_NOT_RUN": "built, not run", "DIAGNOSTIC": "diagnostic", "STOPPED": "stopped", "INCONCLUSIVE": "inconclusive", "SUPERSEDED": "superseded", "ADOPT": "ADOPT", "NOT_ADOPTED": "NOT adopted", "ON_HOLD": "on hold", "WITHDRAWN": "withdrawn", "GATE_NOT_MET": "gate not met, closed", "CLOSED_BY_DATA": "closed by data", "PART_A_INSTALLED": "part A installed, part B open", "PLAN": "plan", "PLANNED": "planned", "RUNNING": "running", "BUILT_NOT_RUN": "built, not run"}


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
            if e['id'] in D87_KEPT:
                vp += f"<br><span class='cap'><b>D87: E2E-required, kept and refined</b> - {esc(D87_KEPT[e['id']])}</span>"
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
    rows = [["E315, E316", "maps trained 6000 steps (E315); the E312 pair head trained 6000 steps (E316)", "yes, same model", "0 ms (training only)", "measured: TIE, kept (E316 +0.0073 / +0.0074 snapped over E312); the 6000 pair steps are part of E324 s"],
            ["E317", "geometry loss on the delivered connector: Chamfer between the Hermite curve and the true connector", "yes: the loss on the output reaches positions, headings, scales and, through the soft-argmax, the maps", "0 ms", "measured: TIE, kept (snapped +0.0077 / +0.0098 over E312); part of the E324 s recipe"],
            ["E318", "E317 + a position scale per keypoint (Laplace NLL)", "yes", "about 0 ms", "measured: CUT by the informativeness clause (Spearman +0.247 / +0.234 against 0.3); no topo F1 gain"],
            ["E319", "heading-guided forward / backward exchange in the junction maps head", "yes", "+0.33 GMACs (+10 % of the maps head), +33 k parameters", "closed at its gate (maps keypoint F1 +0.0026 / +0.0082, bar +0.01); no pair stage run"],
            ["E320", "learned attachment of the connector ends to B's nodes + the E317 geometry loss: replaces the geometry of the snap", "yes (the one-piece wiring on the output stays)", "+9.6 k parameters, +3.9 MMACs", "MEASURED: written token CUT by the duplicates guard (dup_frac +0.0184 on seed 0, +0.0093 on seed 1; topo F1 ties the snap); E323: not the curve count; D87: kept and refined, follow-up E324 (the duplicates)"],
            ["E322, E325", "a learned choice of the delivered piece a connector end is wired to (E322 gate, E325 head)", "yes (the final choice is an argmax: a named selection)", "545 parameters, at most 0.12 MMACs per frame", "E322 G0: no accuracy headroom; E325 head: KEEP on E312 (+0.0023 / +0.0032 snapped); E350 transfers it to E324 s (planned)"],
            ["E331 - E334", "simplification arms (D84 point 4): evidence off, no paint branch, slim maps head, one joint stage", "removals", "lower cost", "E331 NOT adopted (unsnapped -0.0232 / -0.0124), E332 arm a NOT adopted (-0.0189 / -0.0106), E333 withdrawn (D92), E334 NOT adopted (one guard, dup_frac +0.0104 against 0.01)"],
            ["E341, E348", "the evidence without the 0.5 m paint branch (-1.477 GMACs): on the E312 recipe (E341) and on the E324 s recipe (E348)", "removal", "-1.477 GMACs, -0.037 M parameters", "E341 ADOPT on the E312 recipe; E348 NOT adopted by its written rule (PRIMARY seed 1 dup_frac +0.0109 against 0.01)"],
            ["E338", "stack check: E324 s heads on the graph of c2", "n/a (read-out, CPU)", "0", "TIE, kept: sc 0.7066 / 0.7016, additive prediction hit; c2 and E324 s stay separate candidates"],
            ["E343", "the one TEST of E324 s (D95)", "n/a", "CPU scoring, GPU dumps", "ON HOLD after the failed dress rehearsal (vertices per 100 m, one number); no TEST number"],
            ["E345", "pruning by importance, step 0 (D99): a sensitivity table without retraining", "n/a (measurement)", "GPU about 8 h in four jobs", "running; no table yet"],
            ["E342, E349", "host time of the pair head (E342 part A installed, part B open); a GT cache for the pair stage (E349, plan)", "n/a", "E342 A: -1.4 ms per frame", "E342 part A installed; part B and E349 not run"],
            ["E344, E346, E347, E350", "Q-Lane plan; ceiling of the one-graph code; ends on the 2 m node grid; the wiring head on E324 s", "plans", "-", "E344, E347, E350 plan cards; E346 built in scratch, smoke on 31 frames"],
            ["E299, E336", "time: the previous keyframe, ego-warped", "yes", "small", "E336 closed by data: the CPU ceiling of an end prior misses its bar (4.83 % of the GT ends matched, bar 0.05); E299 not built"],
            ["D84 items", "TEST pipeline for E312, simplification by one-at-a-time removals (E330 - E335)", "n/a", "removals lower the cost", "TEST of E312 done once (E321); removals: E341 adopted, E331 / E332 / E334 not adopted, E333 withdrawn, E335 CUT"]]
    return tbl(["id", "question", "differentiable", "run-time cost", "state (2026-11-15)"], rows)


def next_table():
    rows = [["1", "E343 HOLD (Mirko)", "the dress-rehearsal check failed by one guard number (vertices per 100 m 4.9e-3 against 3e-3, a device effect of B's dump); Mirko decides between (a) a kill and (b) a recorded tolerance of 1e-2; until his word no log row, no grant, no TEST"],
            ["2", "E346 (CPU, scratch build running, no frozen file)", "the 873-frame arms of the one-graph ceiling; only the 31-frame smoke exists so far"],
            ["2", "E318 / E348 results", "in the Update of 2026-11-15: E318 CUT by the written rule; E348 NOT adopted (rule log; the card's result section is still to be written)"],
            ["3", "WAIT for the TEST end of E343 (freeze of its 71 code files)", "E349 (GT cache), E350 phase B (wiring head on E324 s), E347 (staged copies in temp/e347, install after), D97 (removal of the old reference chain), dead code (D93)"]]
    return ("<p>From the line <b>Next</b> of HANDOVER.md (2026-11-15): <code>1. E343 HOLD (Mirko); 2. E346 (CPU, scratch build running; no frozen file); E318 / E348 results; 3. WAIT for E343's TEST end (freeze): E349, E350 phase B, E347 (staged copies in temp/e347, install after), D97, dead code.</code></p>"
            + tbl(["#", "item", "state"], rows))


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


# ------------------------------------------------------------------------------------------------------------------ update 2026-11-15 (patch_total_5.py)
C_E324S = cards("runs/e324_s_s{s}/ends_card_snap_thr0.1.json")


def pf(vals, fmt=f3):
    return "-" if any(v is None for v in vals) else " / ".join(fmt(v) for v in vals)


def tv(path, key="topo_F1"):
    c = J(path)
    return None if not c else c["numbers"]["head"][key]


def readout_table():
    """topo F1 and dup_frac of the arms of the last days against their controls, read from the result files (VAL, 873 frames, p >= 0.1)"""
    r12s, r12u = "runs/e312_p_s{s}/ends_card_snap_thr0.1.json", "runs/e312_p_s{s}/ends_card_thr0.1.json"
    r324 = "runs/e324_s_s{s}/ends_card_snap_thr0.1.json"
    spec = [("E324 s (system of record)", "snapped, own keypoints", "E312", r324, r12s),
            ("E317 g (geometry loss)", "snapped", "E312", "runs/e317_g_s{s}/ends_card_snap_thr0.1.json", r12s),
            ("E317 g", "unsnapped", "E312", "runs/e317_g_s{s}/ends_card_thr0.1.json", r12u),
            ("E318 u (position scale)", "snapped", "E317 g", "runs/e318_u_s{s}/ends_card_snap_thr0.1.json", "runs/e317_g_s{s}/ends_card_snap_thr0.1.json"),
            ("E325 w (learned wiring head)", "snapped", "E312", "runs/e312_p_s{s}/ends_card_e325w_snap_thr0.1.json", r12s),
            ("E341 p (no 0.5 m paint branch)", "snapped", "E312", "runs/e341_p_s{s}/ends_card_snap_thr0.1.json", r12s),
            ("E341 p", "unsnapped", "E312", "runs/e341_p_s{s}/ends_card_thr0.1.json", r12u),
            ("E334 o (one joint stage)", "snapped", "E312", "runs/e334_o_s{s}/ends_card_snap_thr0.1.json", r12s),
            ("E334 o", "unsnapped", "E312", "runs/e334_o_s{s}/ends_card_thr0.1.json", r12u),
            ("E331 e (no evidence)", "unsnapped", "E312", "runs/e331_e_s{s}/ends_card_thr0.1.json", r12u),
            ("E332 a (no paint branch, no evidence)", "unsnapped", "E312", "runs/e332_pa_s{s}/ends_card_thr0.1.json", r12u),
            ("E338 sc (E324 s heads on c2's graph)", "snapped, own keypoints", "E324 s", "runs/e324_s_s{s}/ends_card_e338_snap_thr0.1.json", r324),
            ("E348 w (E324 s recipe, no paint branch)", "snapped, own keypoints", "E324 s", "runs/e348_w_s{s}/ends_card_snap_thr0.1.json", r324),
            ("E348 w", "PRIMARY (attached ends, one-piece wiring)", "E324 s", "runs/e348_w_s{s}/ends_card_snap_nogeom_attach_thr0.1.json", "runs/e324_s_s{s}/ends_card_snap_nogeom_attach_thr0.1.json"),
            ("E348 w", "attached ends, proximity wiring", "E324 s", "runs/e348_w_s{s}/ends_card_attach_thr0.1.json", "runs/e324_s_s{s}/ends_card_attach_thr0.1.json")]
    rows = []
    for arm, ro, ctl, pa, pb in spec:
        va, vb = [tv(pa.format(s=s)) for s in (0, 1)], [tv(pb.format(s=s)) for s in (0, 1)]
        da, db = [tv(pa.format(s=s), "dup_frac") for s in (0, 1)], [tv(pb.format(s=s), "dup_frac") for s in (0, 1)]
        d = [None if None in (x, y) else x - y for x, y in zip(va, vb)]
        dd = [None if None in (x, y) else x - y for x, y in zip(da, db)]
        dds = " / ".join("-" if v is None else (f"<span style='color:var(--bad)'><b>{v:+.4f}</b></span>" if v > 0.01 else f"{v:+.4f}") for v in dd)
        rows.append([f"<b>{esc(arm)}</b>", esc(ro), esc(ctl), pf(va), pf(vb), pf(d, sg), dds])
    return tbl(["arm", "read-out", "control (same seed)", "topo F1 arm (s0 / s1)", "topo F1 control", "difference", "dup_frac difference (guard +0.01)"], rows, num=(3, 4, 5, 6))


def update_section():
    E = esc
    res = [["E317", "geometry loss on the delivered connector (E312 recipe)", pill("info", "TIE, kept"),
            "snapped topo F1 over E312 +0.0077 / +0.0098 (KEEP needs more than +0.01 on both seeds); snap-nogeom read-out over u' +0.0199 / +0.0212; unsnapped +0.0159 / +0.0183; every guard held",
            "the loss makes the shape a constraint and is already part of the E324 s recipe; no separate adoption"],
           ["E318", "position scale per keypoint (Laplace NLL) on top of E317", pill("no", "CUT"),
            "snapped topo F1 against the control E317 g -0.0009 / -0.0037 (inside the band); Spearman(sigma, end error) +0.247 / +0.234 against the bar 0.3, so clause 1 of the informativeness requirement fails on both seeds; clause 2 holds (+0.3995 / +0.3510 against +0.03)",
            "the scale is information but moves no graph number; the same loss and head is not proposed again"],
           ["E319", "heading-guided forward / backward exchange in the maps head", pill("dg", "gate not met, closed"),
            "maps keypoint F1 against E307 +0.0026 / +0.0082 (gate +0.01); entry recall beyond 35 m -0.0035 / -0.0073; the pair stage was not run, so there is no topo F1",
            "the head without the block stays"],
           ["E325", "learned wiring head (545 parameters) replaces the nearest-piece rule", pill("ok", "KEEP"),
            "snapped topo F1 over E312 +0.0023 / +0.0032 (paired intervals exclude 0, stay below +0.005); agreement with the oracle choice on the real choices 0.780 / 0.786 against the rule's 0.757 / 0.751; guards held",
            "trained on stored records of the frozen system, not jointly; the final choice is still an argmax (a named output selection); transfer to E324 s = E350 (planned)"],
           ["E341", "evidence without the 0.5 m paint branch (E312 recipe)", pill("ok", "ADOPT"),
            "unsnapped topo F1 against E312 +0.0000 / -0.0080, snapped +0.0005 / -0.0060 (bound 0.010); saves 1.477 GMACs of the system's 7.955 (-18.6 %) and 0.037 M of 2.632 M parameters",
            "seed 1 unsnapped sits 0.002 inside the bound (interval [-0.0167, -0.0001]): ADOPT says within the band, not no loss; the ms are an estimate until E342 part B"],
           ["E348", "the E341 removal on the E324 s recipe", pill("no", "NOT adopted"),
            "the one failing guard: PRIMARY seed 1 dup_frac +0.0109 against the bound +0.01; topo F1 against E324 s inside the band: snapped -0.0028 / -0.0006, PRIMARY -0.0014 / +0.0022, attached -0.0013 / +0.0014",
            "source: the rule log temp/e348/rule.log (the card has no result section yet); E348.w_s1 crashed at step 5038 (ENOMEM) and was resumed from its checkpoint (named deviation)"],
           ["E334", "one joint training stage instead of two", pill("no", "NOT adopted"),
            "one guard: dup_frac +0.0104 against +0.01 on seed 0, unsnapped; topo F1 inside the band (unsnapped -0.0011 / -0.0034, snapped -0.0024 / -0.0049); pair precision at E312's recall -0.057 / -0.077",
            "the pair head is worse in a way the rule does not see; no re-run on the strength of a near miss"],
           ["E331, E332", "pair stage without the evidence; maps head without paint branch and evidence", pill("no", "NOT adopted (both)"),
            "unsnapped topo F1 against E312: E331 -0.0232 / -0.0124, E332 arm a -0.0189 / -0.0106 (bound 0.010); snapped -0.0029 / -0.0007 and -0.0045 / -0.0015",
            "the loss is the evidence, not the paint branch, which opened E341 (both results date from 2026-11-14, after the last build)"],
           ["E333, E339, E340", "width ladder of the maps head (40, 40 taught by the wide head, 32)", pill("dg", "withdrawn (D92)"), "never run", "the width stays 64; compute is saved by other means"],
           ["E338", "stack check: E324 s heads on the graph of c2 (CPU)", pill("info", "TIE, kept"),
            "sc 0.7066 / 0.7016 against s 0.6981 / 0.7010: +0.0085 / +0.0006 (seed 1 misses the KEEP bound); the additive prediction is a hit on every number; against E312 snapped +0.0242 / +0.0152",
            "c2's own gain on seed 1 is small (+0.0020 over E312); c2 and E324 s stay two separate candidates; adoption is Mirko's"],
           ["E337", "attached ends against the snap's geometry (stored cards)", pill("dg", "answered, no training"),
            "the gap +0.006932 / +0.006155 topo F1 is geometry: the attached end and the keypoint wire the same piece in 96.2 % / 96.6 % of the ends",
            "the sharpening lever stays a stub, not scheduled"],
           ["E342", "where the 17.09 ms of the junction forward go", pill("info", "part A installed, B open"),
            "jgt._groups vectorised: 0.80 ms to 0.095 ms per set, labels equal on 3 322 real sets; about 2.2 ms to 0.75 ms per frame (-1.4 ms of 17.09 ms)",
            "installed by commit 27fee60b; part B (idle-GPU profile of the pair head) is not run"],
           ["E343", "the one TEST of E324 s (D95)", pill("info", "ON HOLD"),
            "no TEST number exists. Dress rehearsal on VAL: GPU branch seed 0 PASS, seed 1 FAIL by one number: vertices per 100 m differs by 4.87e-3 (snapped) against the tolerance 3e-3",
            "decision with Mirko: kill, or a recorded tolerance of 1e-2; no log row, no grant, no TEST until then"],
           ["E345", "pruning by importance, step 0 of D99", pill("dg", "running"), "no table yet (the card folder holds no results.md)", "a measurement on the stored E312 and E324 s models; no training"]]
    hard = ("<ul class='tight'>"
            "<li><b>Wiring of the connector ends</b> (the one-piece, nearest-piece rule: 4 m / 45 degrees, at most 8 candidate places): now REPLACEABLE by the learned head of E325 (KEEP on E312). <b>The final choice is still an argmax</b>, a named output selection; the weights are not edge probabilities; the head is trained on stored records, not jointly. The PRIMARY read-out of E324 s keeps the nearest-piece rule; E350 plans the transfer to the snapped read-out of E324 s.</li>"
            "<li><b>Probability threshold p &gt;= 0.1</b>: a connector is delivered above it; the planner never sees the probability.</li>"
            "<li><b>Top-K keypoint choice</b> (K = 32, 3 x 3 peaks of the end heat): the position inside the window is a soft-argmax, the existence of a token is a hard choice.</li>"
            "<li><b>Close-node step 1.5 m</b> on the read-out of B: greedy removal of the lower-probability node; the soft versions tried so far are E277 (CUT), E279 (KEEP at 1.25 m, Mirko's choice open) and E288 (TIE as a read-out replay); the layer E292 is not covered by this update.</li>"
            "<li><b>Douglas-Peucker 0.3 m and the node threshold</b>: the budget of 14.3 vertices per 100 m is not learned.</li>"
            "<li><b>Snap geometry</b> in the read-out of record (own keypoints snapped): the attached ends of E320 / E324 give up +0.0069 / +0.0062 topo F1 against it, and that gap is geometry (E337).</li>"
            "<li><b>Frozen sensor features</b> (D50): the loss never reaches the camera / radar encoder.</li></ul>")
    plans = [["E342 part B", "idle-GPU profile of the pair head (Sunday night)", "not run"],
             ["E344", "plan from five rule steps to none: the query head on the 1 m grid (Q-Lane); Q0 is E346, Q1 and later wait for Mirko's decision on D84 point 6", "plan only"],
             ["E346", "ceiling of the one-graph code on all 873 VAL frames (B's node + link code against a clean control; 1 m grid; node chains through the junction)", "built in scratch, 31-frame smoke only; open kill item named in the card"],
             ["E347", "the junction ends on the 2 m node grid: can one grid carry the ends, and does a one-to-one target make the peak rule unnecessary?", "plan card, nothing started"],
             ["E349", "the per-frame ground truth of the pair stage computed once (cache + pair-only item path)", "plan card, not built"],
             ["E350", "the E325 wiring head trained on E324 s and read out at its snapped graph", "plan card; phase B waits for the TEST end of E343"]]
    eff = ("<ul class='tight'>"
           "<li><b>GPU utilisation of the trainings is low:</b> about 15 % over 12 s on 2026-11-14 (0 % with peaks to 100 %); the main processes of the two E334 trainings sit at 13 - 17 % CPU, the six data workers (3 per job) at about 95 % each, 12 cores with about 44 % idle. The step (about 2 s) is bound by the loader's CPU work, not by the GPU (E342 section 4).</li>"
           "<li><b>D101:</b> 4 data workers per new training start; against 3 workers every field the pair stage reads is bit-identical (only the simulated map is unseeded, and the pair and maps stages do not read it). Loader per batch of 8 on the loaded box: 2.4 - 2.5 s with 3 workers, 1.8 s with 4.</li>"
           "<li><b>E349</b> plans a cache of the per-frame geometry and a pair-only item path (not built): control 1.83 s per batch, the card predicts 1.5 s less, the rule asks for at least 0.9 s.</li>"
           "<li><b>Saved so far:</b> E341 1.477 GMACs and 0.037 M parameters (ms estimated, not measured); E342 part A about 1.4 ms per frame of the 17.09 ms.</li></ul>")
    dec = [["D92", "no blunt narrowing: the width of the heads stays 64; compute is saved otherwise; E333, E339, E340 withdrawn"],
           ["D93", "dead code may be deleted, only in Spurmitte, in batches after a tag, git history is the way back"],
           ["D94", "system of record = E324 s (with c2; the stack E338 the combined candidate); new arms are compared with E324 s of the same seed; PROBLEM.md and the registry change only after the TEST"],
           ["D95", "ONE TEST use of E324 s (both seeds), control E312's stored TEST cards; the readings R1 - R3 of card E343 confirmed (Confirmed lines of 2026-11-14 22:25 and 23:00); the technical gates of the card stay"],
           ["D96", "the holds on E317, E318 and E319 are lifted; they run on the recipe their cards name"],
           ["D97", "removal of the old reference chain of loops 8 - 18 (option B of D89), in steps: freeze the reference numbers and tag first; waits for the TEST end of E343"],
           ["D98", "weekend rule (GPU jobs by day only one at a time, nice, host gate): void since D100"],
           ["D99", "pruning by importance instead of narrowing: a size-against-loss table (step 0 = E345), then cards for two or three sizes"],
           ["D100", "the weekend rule D98 is void: chains run freely up to the two GPU slots, no host gate; the hard rules and the weekday GPU window stay"],
           ["D101", "4 data workers for every new training start (checked bit-identical against 3)"]]
    return ('<section id="update"><h2>Update 2026-11-15</h2>'
            "<p>State of 2026-11-15; the report was last built on 2026-11-14. <b>The system of record is unchanged:</b> E324 s (the E320 recipe at 6000 pair steps), VAL snapped topo F1 " + pair(head(C_E324S, "topo_F1")) + " against " + pair(head(C_E312S, "topo_F1")) + " for E312; c2 (B fine-tuned) and the stack E338 are candidates next to it (D94). TEST has been used once, for E312 (E321); there is no TEST number for E324 s. Everything below is VAL, two seeds, 873 frames; the tokens are the cards' and the written rule of each card decides.</p>"
            "<h3>What was measured since the last build</h3>"
            + tbl(["experiment", "what", "verdict (written rule)", "key numbers (card, VAL, seed 0 / seed 1)", "reading"], [[f"<b>{a}</b>", E(b), c, E(d), E(e)] for a, b, c, d, e in res])
            + "<h3>The same arms read from the result files</h3>" + readout_table()
            + "<p class='cap'>Read from runs/*/ends_card*.json (topo F1 and dup_frac of the delivered graph, p &gt;= 0.1, VAL 873 frames); the difference is arm minus control of the same seed. E348 and E338 are read against E324 s, the others against the control of their card. dup_frac differences above the guard +0.01 are marked red.</p>"
            "<h3>Hard steps still in the system</h3>" + hard
            + "<div class='note warn'><b>Open item: E343 is on hold.</b> The dress-rehearsal check of the one TEST use of E324 s failed by one guard number (vertices per 100 m, 4.87e-3 against 3e-3, on seed 1 of the GPU branch; the stack arm reproduces exactly). By the letter of the card the check failed and nothing was started; Mirko decides between a kill and a recorded tolerance of 1e-2. <b>No TEST number of E324 s exists.</b></div>"
            "<h3>Efficiency</h3>" + eff
            + "<h3>Planned and running cards</h3>" + tbl(["card", "what", "state"], [[f"<b>{a}</b>", E(b), E(c)] for a, b, c in plans])
            + "<h3>Decisions D92 - D101</h3>" + tbl(["decision", "what (as recorded in experiment_execution/decisions/)"], [[f"<b>{a}</b>", E(b)] for a, b in dec])
            + "</section>")


# ------------------------------------------------------------------------------------------------------------------ page
head_html = open(os.path.join(HERE, "head.html"), encoding="utf-8").read().replace("Spurmitte Lane Report", "Spurmitte Total Report")
head_html = head_html.replace("</style>", """svg .box.new { fill: var(--surface); stroke: var(--accent); stroke-dasharray: 6 4; stroke-width: 1.6; }
svg .tag { fill: var(--warn); font-family: var(--font-mono); font-size: 10px; font-weight: 500; }
.pill.dg { background: var(--surface2); color: var(--muted); }
table.exp td { font-size: 13.5px; } table.exp tr[data-v="KEEP"] td:first-child { border-left: 3px solid var(--good); } table.exp tr[data-v="TIE"] td:first-child { border-left: 3px solid var(--warn); } table.exp tr[data-v="CUT"] td:first-child { border-left: 3px solid var(--bad); } table.exp tr[data-v="ADOPT"] td:first-child { border-left: 3px solid var(--good); } table.exp tr[data-v="NOT_ADOPTED"] td:first-child { border-left: 3px solid var(--bad); } table.exp tr[data-v="ON_HOLD"] td:first-child { border-left: 3px solid var(--warn); }
.flt { margin: 4px 0 10px; } .flt button { font: 500 12.5px var(--font-mono); padding: 3px 10px; margin-right: 6px; border: 1px solid var(--line); background: var(--surface); color: var(--ink); border-radius: 4px; cursor: pointer; } .flt button.on { background: var(--accent); color: #fff; border-color: var(--accent); }
</style>""", 1)

d312s, d312u = diff(C_E312S, "topo_F1"), diff(C_E312U, "topo_F1")
cov_s, cov_u, cov_p = [cov(MAN[s]) for s in (0, 1)], [cov(MAN[s], "p0.1") for s in (0, 1)], [cov(MAN[s], "plain") for s in (0, 1)]
VC = verdict_counts()
EXPMAX = max([int(re.match(r"E([0-9]+)", e["id"]).group(1)) for e in EXP] or [319])
vc_txt = ", ".join(f"{n} {VNAME.get(k, k)}" for k, n in sorted(VC.items(), key=lambda kv: -kv[1])) if VC else "pending"
tiles = f"""<div class="tiles">
<div class="tile good"><div class="k">best system E312, snapped: topo F1 over the plain graph (s0 / s1)</div><div class="v">{pair(d312s, lambda v: '+%.3f' % v)}</div><div class="n">0.6825 / 0.6864 against 0.6259 / 0.6249; VAL, 873 frames; the one TEST use (E321) in section TEST</div></div>
<div class="tile good"><div class="k">system of record E324 s, snapped: topo F1 (s0 / s1)</div><div class="v">{pair(head(C_E324S, "topo_F1"))}</div><div class="n">E312 recipe + geometry loss + learned attachment + 6000 pair steps (D94); VAL, two seeds; no TEST yet (E343 on hold)</div></div>
<div class="tile good"><div class="k">the same without the snap rule (the differentiable read-out)</div><div class="v">{pair(d312u, lambda v: '+%.3f' % v)}</div><div class="n">the snap, a rule on the output, carries most of the topology gain; E320 replaced its geometry (CUT by a guard)</div></div>
<div class="tile good"><div class="k">GT maneuvers realised: plain graph to E312 snapped</div><div class="v">{f2(cov_p[0])} to {f2(cov_s[0])} %</div><div class="n">seed 0, 10991 maneuvers in 826 junction frames; not snapped {f2(cov_u[0])} %</div></div>
<div class="tile"><div class="k">experiments in this report</div><div class="v">{len(EXP)}</div><div class="n">E273 - E{EXPMAX}: {vc_txt}</div></div>
<div class="tile"><div class="k">not end-to-end differentiable yet</div><div class="v">5</div><div class="n">snap + wiring, threshold, top-K choice, close-node step, read-out simplification; plus the frozen sensor features (D50)</div></div>
<div class="tile"><div class="k">the whole lane head</div><div class="v">{f"{COST['total_macs_g']:.1f} GMACs" if COST else "pending"}</div><div class="n">{f"{COST['params_m']['total']:.2f} M parameters; B alone {TIMEB['total']['median']:.0f} ms per frame (idle GPU)" if COST and TIMEB else ""}</div></div>
</div>"""

body = f"""<div class="wrap">
<header class="top">
  <div class="eyebrow">Spurmitte - total report - state of 2026-11-15 (first built 2026-11-12) - lane-centre graph from camera + radar on frozen GaussianCaR features - VAL, plus the one TEST use of E312 (E321); the TEST of E324 s (E343) is on hold</div>
  <h1>All experiments, the architecture, the best approaches, and what is still not end-to-end differentiable</h1>
  <p class="lead">One page for the whole lane thread since the plain lane head B (E273): every experiment with its question, verdict and key numbers, the architecture with the non-differentiable steps marked <b>to change</b>, the best approaches side by side, the results, and all pictures and videos. The system of record since D94 is E324 s ({pair(head(C_E324S, "topo_F1"))} snapped topo F1 on VAL against {pair(head(C_E312S, "topo_F1"))} for E312), a refinement of E312; the base system E312 (junction maps with a narrow end heat, a pair head trained together with them, learned connector shapes, connectors delivered at a low probability threshold, ends snapped to the delivered pieces). It is end-to-end differentiable except for a few named rule steps on the output; the snap is the one that matters, because without it two thirds of the topology gain are gone.</p>
  {tiles}
</header>
<nav class="idx"><a href="#sum">Summary</a><a href="#update">Update 2026-11-15</a><a href="#best">Best approaches</a><a href="#change">To change</a><a href="#arch">Architecture</a><a href="#exps">All experiments</a><a href="#numbers">Results</a><a href="#test">TEST</a><a href="#compute">Compute</a><a href="#look">Pictures and videos</a><a href="#why">Why the shape does not emerge</a><a href="#next">Next</a><a href="#weak">Weaknesses</a></nav>

<section id="sum"><h2>Summary</h2>
<ul class="tight">
<li><b>Where we started.</b> B (E273) is a pixel-map lane head: nodes on a 2 m grid, Hermite links, soft sections and a soft map assignment, end to end on cached GaussianCaR features. Its read-out has a hard close-node step (E275). At junctions it fails: it has no notion of which entry connects to which exit.</li>
<li><b>What we built.</b> A junction layer: a 1 m maps head (E286) with entry / exit heat and heading, a pair head over keypoints with the order rule (E287), joint training of both (E305), learned connector shapes (E303), a narrow heat of 1.0 m (E307, the largest single gain), a low delivery threshold (E309) and the attachment of connector ends to the delivered pieces (E311). Together: E312.</li>
<li><b>What it gives (VAL, two seeds).</b> Topo F1 {pair(head(C_E312S, "topo_F1"))} against {pair(base(C_E312S, "topo_F1"))} for the plain graph ({pair(d312s, sg)}), lanes found {pair(diff(C_E312S, "lanes_found"), sg)}, GT maneuvers realised {f2(cov_p[0])} % to {f2(cov_s[0])} % (seed 0). Pictures and videos agree with the numbers; the strict match radius of 1 m lowers the absolute level.</li>
<li><b>What is not differentiable.</b> The attachment of the connector ends (snap), the probability threshold, the top-K keypoint choice, the close-node step and the output simplification; and the sensor features are frozen by decision (D50). Section "To change" lists each one with its cost and its replacement; E320 (learned attachment) is measured (written token CUT by a guard, a tie with the snap; D87: kept and refined); the learned wiring head E325 is KEEP on E312 (its final choice is still an argmax, a named output selection); E317 is a TIE, E318 CUT, E319 closed at its gate. Section Update 2026-11-15 has the numbers.</li>
<li><b>Experiments.</b> {len(EXP)} cards: {vc_txt}. Ties are kept and refined (D82), never worse.</li>
<li><b>TEST.</b> Used once, for E312 only (E321, D84): the gain holds on the 21 unseen scenes, see section TEST. <b>Not done:</b> no later design has been run on TEST: E343 (the one TEST use of E324 s, D95) is on hold after its dress rehearsal failed by one guard number, so there is no TEST number for E324 s; everything after E321 is VAL.</li>
<li><b>System of record since D94.</b> E324 s (the E320 recipe at 6000 pair steps), VAL snapped topo F1 {pair(head(C_E324S, "topo_F1"))}; c2 (B fine-tuned) and the stack E338 are candidates next to it; every new arm is compared with E324 s of the same seed. The protected files (PROBLEM.md, registry) change only after the TEST check.</li></ul></section>

{update_section()}

<section id="best"><h2>Best approaches</h2>
<p>All rows are scored the same way: delivered graph = B plus the hard close-node step plus the connectors of the head, paired scorer, VAL, two seeds, differences to the plain delivered graph of the same seed. Ranked by what we would deploy today. Maneuvers realised is the share of GT entry-to-exit maneuvers the delivered graph covers (E310, <code>evaluation/jmaneuver.py</code>, p &gt;= 0.1 snapped) where it was measured.</p>
{best_table()}
<div class="note"><b>Recommendation.</b> E324 s (own keypoints, snapped) is the system of record since D94 (VAL); E312 is the control of the one TEST use so far and the older reference. For a planner that cares about the ego's own maneuvers take E298; for lanes found over junction precision take E314. The snap's geometry has a differentiable replacement (E320: a tie with the snap on topo F1, CUT by a duplicates guard); the one-piece wiring can be replaced by the learned head of E325 (KEEP on E312: +0.0023 / +0.0032 snapped topo F1; its final choice is still an argmax, a named selection on the output). The simplification arms are measured (E331, E332, E334 not adopted, E333 withdrawn, E341 adopted); the further gains come from the stack with c2 (E338, a candidate), the learned wiring head on the system of record (E350, planned) and the plans listed in the Update. The earlier loops' results (route head, spurs, ring, prototype Q 0.727 on TEST in loop 22) belong to another system and are not repeated here.</div>
<h3>The ladder: what each step adds to the delivered graph</h3>{ladder_table()}
<p class="cap">topo F1 is the directed reachability score (seeds every 5 m on the GT, reach 30 m, match 1 m, direction within 60 degrees); lanes found is the share of GT pieces with at least 80 % of their points within 1 m of delivered points. Both are strict.</p>
<h3>All approaches with their soft and hard parts</h3>{audit_table()}</section>

<section id="change"><h2>Not end-to-end differentiable: what has to change</h2>
<p>The rule of the project (D55, confirmed by Mirko on 2026-11-12 and again in D84): everything on the training path is soft, a loss on the output reaches every module, hard selection only on the output and named in every report; from now on only end-to-end differentiable solutions are pursued. The table lists every step that breaks the rule, in the order in which it is worth replacing.</p>
{hard_table()}
<h3>What the snap does: wiring and geometry (E312, VAL, p &gt;= 0.1)</h3>
<p>The snap does two things: it moves the connector ends onto the ends of the delivered pieces (geometry) and it wires each end to ONE nearest piece (wiring); the unsnapped read-out wires every piece within 4 m / 45 degrees. Scoring the same E312 checkpoints with each half alone (E311 options <code>--snap-nogeom</code> and <code>--snap-all</code>, scored in the night of 2026-11-13) splits the gain:</p>
{wiring_table()}
<div class="note warn"><b>The wiring is the larger part.</b> Of the gain of the snap over the unsnapped read-out {' / '.join(f'{100 * w[0]:.0f} %' for w in wiring_share() if w)} (s0 / s1) is the wiring ({' / '.join(f'{w[3]:+.4f}' for w in wiring_share() if w)} topo F1), {' / '.join(f'{100 * w[1]:.0f} %' for w in wiring_share() if w)} the end geometry ({' / '.join(f'{w[4]:+.4f}' for w in wiring_share() if w)}). With the proximity wiring even perfect end geometry (snap-all) reaches only {pair(head(WC['all'], 'topo_F1'))}. So the attachment of E320 (geometry) could recover at most about a third of the gap, and it did (table below; by D87 E320 is kept and refined, not cut: the duplicates are the defect to remove). The wiring is not a which-piece effect: E322 gate G0 (oracle selection at the matched GT end) gains {G0TXT}; the one-piece rule is a COUNT effect (one piece instead of every piece in reach), a named selection on the output that D55 allows; the question the card carried for Mirko (end-to-end replacement of the output selection against a simple system) is answered by D87: E2E has priority where no E2E alternative exists, so the learned wiring head is built at the equal-accuracy bar (topo F1 >= c - 0.005 on both seeds) and its price in parameters and ms is reported.</div>
<h3>E324 arm s: the E320 recipe at 6000 pair steps (the best read-outs so far)</h3>
{e324_table()}
<p class="cap">E324 s = geometry loss + learned attachment + 6000 pair steps (card E324, VAL, two seeds). The primary read-out (attached ends, the one-piece wiring on the output) beats the snapped geometry of E312 by +0.0087 / +0.0084 topo F1; the own-keypoint snapped read-out of the same run is +0.0156 / +0.0146 over E312 snapped with the guards held and is named a system-of-record CANDIDATE (reporting only; the adoption and any TEST use need Mirko's recorded OK). The duplicates guard of the primary read-out breaks again on seed 0 (+0.0150 against 0.01), so the written token is not KEEP; by D87 it is kept and refined (follow-up card E337: the attached end is a convex combination of the keypoint and raw B nodes, not a delivered piece end). Cost: +9.6 k parameters, +3.9 MMACs, no ms at the read-out; training 3.2 h per seed.</p>
<h3>E320: the learned attachment, read-outs on VAL</h3>
{e320_table()}
<p class="cap">Rule of the card (written before the run): KEEP when the primary read-out reaches u' + 0.005 on both seeds with the guards held; the duplicates guard breaks on seed 0 (dup_frac +0.0184 against the limit +0.01), so the verdict is CUT by the written rule. Bootstraps (22 scenes): a - c -0.0021 [-0.0079, +0.0041] / +0.0025 [-0.0078, +0.0106], a tie with the snap; junction F1 -0.0256 / -0.0137, significantly lower but inside the guard. The tight 1.5 m / 1.0 m wiring fails on both seeds: a contact threshold does not replace the one-piece rule. E323 reads E312 and E320 at equal curve counts: the duplicate excess is not the curve count.</p>
<div class="note warn"><b>Reading.</b> Without the snap E312 gains {pair(d312u, sg)} topo F1 over the plain graph, with it {pair(d312s, sg)}: topology precision falls from {pair(head(C_E312S, "topo_P"))} to {pair(head(C_E312U, "topo_P"))} because connectors that do not start and end exactly at the delivered pieces break reachability, while the geometric coverage does not change (lanes found {pair(head(C_E312U, "lanes_found"))} unsnapped against {pair(head(C_E312S, "lanes_found"))} snapped; maneuvers realised {f2(cov_u[0])} against {f2(cov_s[0])} %). The first step was E320 with E317's geometry loss as its training signal (both measured, see above and the Update of 2026-11-15); what is left is the wiring (the learned head E325 / E350, whose output is still an argmax), delivery of probabilities instead of a threshold, and the close-node step; the unstructured set head stays deferred (D84).</div></section>

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
<p>Every card from E273 to E{EXPMAX}, from the cards of <code>experiments/</code> (question, design, verdict as the card states it, key numbers copied from the card; all VAL unless the card says otherwise). Colour at the left edge: green KEEP or ADOPT, amber TIE (kept and refined) or on hold, red CUT or NOT adopted (the removal arms are ADOPT / NOT adopted by their written rule). "In E312" marks what the best system uses. <b>D87 (Mirko, 2026-11-13):</b> an experiment that is the end-to-end replacement of a rule step with no other end-to-end alternative is not cut by a missed bound or guard: the written token stays in the card and in this table, the experiment is kept and refined (marked D87). Read E273 with care: its own card is CUT because the topology broke the guards against the older head (topo F1 0.597 against 0.741), yet it is the base B of the whole thread, because the gain of the soft map assignment holds; E275 repaired its read-out. Verdicts "by hand" carry the label of the card (hard rule 4 of the project); the numbers are copied from the cards, never recomputed.</p>
<div class="flt"><button class="on" data-v="ALL">all</button><button data-v="KEEP">KEEP</button><button data-v="TIE">TIE</button><button data-v="CUT">CUT</button><button data-v="ADOPT">ADOPT</button><button data-v="NOT_ADOPTED">NOT adopted</button><button data-v="OTHER">other</button></div>
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
<div class="note"><b>Reading.</b> The VAL gain holds on unseen scenes: snapped topo F1 over the plain graph {pair([None if not c else c['variants']['snapped']['difference']['topo_F1'] for c in TS], sg)} (VAL {pair(d312s, sg)}); not snapped {pair([None if not c else c['variants']['unsnapped']['difference']['topo_F1'] for c in TS], sg)} (VAL {pair(d312u, sg)}). The snap, the rule step, carries about 60 % of the gain on TEST as well; the differentiable read-out does not improve the junction F1 (its precision falls). The plain graph is harder on TEST than on VAL (topo F1 {pair([None if not c else c['variants']['snapped']['numbers']['base']['topo_F1'] for c in TS])} against {pair(base(C_E312S, 'topo_F1'))}). The paired scene-bootstrap intervals (all above zero for the snapped read-out) are in the card <code>experiments/E321_test-e312/experiment.md</code>, section 4; no promotion (the champion decision stays with Mirko); since D94 the VAL reference is E324 s, while the protected champion entries still name E312 until the TEST check of E324 s (D95).</div>
<div class="note warn"><b>The second TEST use is on hold (E343).</b> Mirko's OK of D95 allows ONE use for E324 s (with the stack E338 as a second arm), control E312's stored TEST cards. The VAL gate of the card (the code must reproduce the stored VAL cards, then a dress rehearsal with fresh dumps on the TEST device) failed by one guard number: vertices per 100 m differs by 4.87e-3 on seed 1 against the pre-registered 3e-3 (the residual is the CPU dump of B); by the card's letter the check failed. <b>No TEST run was started and no TEST number of E324 s exists</b>; Mirko decides between a kill and a recorded tolerance of 1e-2 on that guard metric.</div></section>

<section id="compute"><h2>Compute</h2>{compute_table()}
<p class="cap">MACs and parameters by counting, ms only on an idle GPU (the timing job of the night chain fills the pending cells on the next build). Training: maps 0.9 s per step, pair stage with joint training 1.9 s per step (3000 steps = 1.6 h per seed). The pair head costs 0.02 GMACs, the maps head 3.3 GMACs on top of the pyramid that B already runs; the read-out runs on the CPU. Since the last build: E341 removes 1.477 GMACs and 0.037 M parameters (the ms are an estimate until E342 part B), E342 part A took about 1.4 ms of the 17.09 ms per frame (the pair head's host bookkeeping), and the efficiency facts of the trainings (GPU about 15 % busy, loader-bound step, E349) are in the Update of 2026-11-15.</p></section>

<section id="look"><h2>Pictures and videos</h2>{media_total()}</section>

<section id="why"><h2>Why the right shape does not emerge by itself</h2>
<div class="note"><b>Short answer from the data.</b> On ordinary lane stretches the shape does emerge: median lateral offset 0.2 m, jitter 0.13 m, 98 % of the delivered length within 2 m of the GT. What does not emerge is what the evidence does not fix: which entry joins which exit, where an end sits to a metre, what lies beyond the far ring. Mirko's three hints name three reasons; each had a differentiable test (measured: E317 TIE, E318 CUT, E319 closed at its gate; see the Update of 2026-11-15).</div>
<div class="scroll"><table><thead><tr><th>hint</th><th>what we measured</th><th>differentiable test</th></tr></thead><tbody>
<tr><td><b>smoothness is no constraint</b></td><td>the connector curve itself has no loss. Kinks at the joins: median {f"{LS['B']['join_med']:.1f}" if LS else "?"} degrees in B, {f"{LS['e312']['join_med']:.1f}" if LS else "?"} in E312, GT {f"{LS['GT']['join_med']:.1f}" if LS else "?"}</td><td><b>E317</b>: Chamfer loss between the delivered Hermite curve and the true connector; training only. Measured: TIE, kept; it lifts the end geometry by +0.0199 / +0.0212 topo F1 and is part of the E324 s recipe</td></tr>
<tr><td><b>position uncertainty is too small</b></td><td>the end position is a 1 m cell plus a soft-argmax and carries no uncertainty; before E307 the ends were 1.44 m off on average (E310) and only 52 % of the GT pairs had both ends matched strictly</td><td><b>E318</b>: a log sigma per end with a Laplace NLL; sigma enters the pair logit and goes to the planner. Measured: CUT (Spearman of sigma with the end error +0.247 / +0.234 against the bar 0.3; no topo F1 gain)</td></tr>
<tr><td><b>forward-backward exchange is missing</b></td><td>on the E305 keypoints the pair decision equals a perfect classifier (+0.0304 vs +0.0314, E310): the ends are the limit; 53 % of the GT ends lie beyond 35 m, pair recall there 0.23 vs 0.38 near (E300); the context of the maps head is isotropic</td><td><b>E319</b>: heading-guided exchange ahead / behind along the lane; <b>E299</b> in time (pair flips 19 %). Measured: E319 closed at its gate (maps keypoint F1 +0.0026 / +0.0082, bar +0.01); E336 (recursive state) closed by data</td></tr></tbody></table></div>
<p>Splines, point types and classes are what a transformer's output is made of: a Hermite curve between two typed points is what the planner and the loss can use, and the order rule (1899 of 1899) is a prior a generic decoder would have to learn from data. The direct test of "a transformer can do it all" is an unstructured set-prediction head on the same cached features; D84 defers it until it is planned further.</p></section>

<section id="next"><h2>Next: queued, running, decided</h2>{next_table()}<h3>Queued, running and decided cards</h3>{q_table_total()}
<ul class="tight"><li><b>D84 (Mirko, 2026-11-12):</b> TEST E312 once (done as E321 after the pipeline reproduced the VAL cards; snapped and unsnapped read-outs, control = the plain graph of the same seed); only end-to-end differentiable solutions; simplify the system by one-at-a-time removals with a rule written before the run; plan and run more experiments; the unstructured transformer head is deferred.</li>
<li><b>Rule of every new card:</b> KEEP when the snapped topo F1 over its control rises by more than 0.01 on both seeds with the guards held; since D94 the control is E324 s of the same seed (E312 for the cards written before); the removal arms are ADOPT / NOT adopted by the removal rule of D84 point 4 (loss at most 0.01 on both seeds and both read-outs, the differentiable one first, guards held).</li></ul></section>

<section id="weak"><h2>Weaknesses and honest negatives</h2>
<ul class="tight"><li>The headline numbers are strict (1 m, direction) and scored against the graph GT in a 48 m box, not directly against l6; the 2 m cross-check keeps the order of the variants.</li>
<li>The snap and the close-node step are rules on the output; the delivered graph is end to end only up to these named rules (the snap's geometry has a differentiable replacement, E320, which a duplicates guard rejected; the one-piece wiring and the close-node step stay named selections on the output until a soft close-node step, E292, is measured).</li>
<li>E312 has false connectors near the ego vehicle at the low threshold and weak far-ring ends; pair flips of 19 % between keyframes; single keyframes only.</li>
<li>The best junction results come from the GT-heavy junction layer; the sensor features are frozen, so the loss never reaches the camera / radar encoder.</li>
<li>Cuts and stops are part of the table: E277 (soft step) CUT, E286 jj (joint pyramid) stopped, E304 stopped; ties are kept.</li>
<li>TEST was used once, for E312 only (E321); no later design has been run on TEST: the second use (E343, E324 s, D95) is on hold after the failed dress rehearsal and would need its own log row; no TEST number of E324 s exists.</li></ul>
<h3>Open decisions</h3><ul class="tight"><li>E343 hold: kill, or a recorded tolerance of 1e-2 on vertices per 100 m (Mirko); TEST for any later design (its own recorded OK), champion promotion and the change of PROBLEM.md (a D entry), the soft step of 1.25 m (E279), D4 / D5, push, whether to build the unstructured set head.</li></ul></section>
<footer>Built by temp/jr/build_total_report.py (prefix: build_e2e_report.py; patched by patch_total_2 - 5) from runs/**/ends_card*.json, runs/e312_p_s*/maneuver_val.json, runs/e*/jmaps_val.json, temp/e309/*.json, temp/jr/model_cost.json, runs/e273/e273_p_s0/timing.json and the experiment table temp/jr/experiments_table.json (made from experiments/E*/experiment.md). Pictures and videos: results/l6_view_E312, results/Junction.report.</footer>
</div>
<script>
document.querySelectorAll('.flt button').forEach(function (b) {{
  b.addEventListener('click', function () {{
    var v = b.dataset.v;
    document.querySelectorAll('.flt button').forEach(function (x) {{ x.classList.toggle('on', x === b); }});
    document.querySelectorAll('table.exp tr[data-v]').forEach(function (r) {{
      var show = v === 'ALL' || r.dataset.v === v || (v === 'OTHER' && ['KEEP', 'TIE', 'CUT', 'ADOPT', 'NOT_ADOPTED'].indexOf(r.dataset.v) < 0);
      r.style.display = show ? '' : 'none';
    }});
  }});
}});
</script>"""

open(os.path.join(OUT, "index.html"), "w", encoding="utf-8").write(head_html + "\n" + body)
print("written", os.path.join(OUT, "index.html"), len(body), "experiments:", len(EXP))
