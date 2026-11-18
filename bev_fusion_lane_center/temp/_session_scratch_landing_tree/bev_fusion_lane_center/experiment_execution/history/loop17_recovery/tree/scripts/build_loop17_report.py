"""Builds the loop-17 report (HTML, A4): robustness and temporal fusion of the fully differentiable lane-graph chain under sensor failure (night, blur, lost cameras, ...), the learned gated memory, map assignment under failure,
the map-conditioned builder, the joint end-to-end fine-tune and the latency. Every number is read from out/loop2/e17/results_master.json (scripts/p17_master.py), the route JSONs, the latency JSONs and the ledger when the report is built;
the narrative is loop17_text.py. Usage (gcar): build_loop17_report.py -> report_final/build/Loop17_Robust_Differentiable_Lane_Graph_Report_EN.html (PDF: finish_report6.ps1 <html> <pdf>)"""
import html
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_e2e_condensed as B  # noqa: E402

OUT = "/mnt/c/ML-Experiment/report_final/build/Loop17_Robust_Differentiable_Lane_Graph_Report_EN.html"
FIG = "/mnt/c/ML-Experiment/report_final/figures"
E17 = "/mnt/c/ML-Experiment/out/loop2/e17"
esc = html.escape
MJ = json.load(open(f"{E17}/results_master.json"))
EX = json.load(open(f"{E17}/results_extra.json")) if os.path.exists(f"{E17}/results_extra.json") else dict(cleanref={}, unseen={}, route_boot={}, maponly=None, consumer={})
PERT3 = ["dark", "blur", "cam_front3_off"]
PERT4 = ["fog", "radar_off", "extr3", "cam_front_off"]
NAME = dict(dark="night (dark images)", blur="motion blur", cam_front3_off="three front cameras off", fog="fog", radar_off="radar off", extr3="extrinsics error", cam_front_off="front camera off")
LABEL = dict(ctrl_single="control E94, single frame", ctrl_hold="control E94 + spatial hold", ctrl_ema="control E94 + EMA / monitor hold", champ_single="champion, single frame", champ_hold="champion + spatial hold (recursion on)",
             champ_ema="champion + EMA / monitor hold", norec_single="champion, recursion off", norec_hold="champion, recursion off + spatial hold", gate="champion, recursion off + learned gate",
             ftall_single="failure-trained dense head, single frame", ftall_hold="failure-trained dense head + hold", ft_single="builder fine-tuned on failure frames, single", ft_hold="builder fine-tuned on failure frames + hold",
             ft_ema="builder fine-tuned on failure frames + EMA hold")


# ------------------------------------------------------------------ accessors
def card(p, k, ph):
    return MJ["perts"][p]["cards"][k][ph]


def S(p, k, ph="fail", m="S"):
    return card(p, k, ph)[m]


def boot(p, key):
    return MJ["perts"][p]["boot"].get(key)


def f3(x, n=3):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{n}f}"


def bs(p, key):
    b = boot(p, key)
    return "-" if b is None else f"{b['diff']:+.3f} [{b['lo']:+.3f}, {b['hi']:+.3f}]"


def rb(label):
    c = EX.get("route_boot", {}).get(label)
    return "-" if not c or "diff" not in c else f"{c['diff']:+.2f} [{c['lo']:+.2f}, {c['hi']:+.2f}]"


def table(head, rows, nums=None):
    nums = nums or set()
    h = "<table><tr>" + "".join(f"<th class='{'n' if i in nums else ''}'>{c}</th>" for i, c in enumerate(head)) + "</tr>"
    for r in rows:
        h += "<tr>" + "".join(f"<td class='{'n' if i in nums else ''}'>{c}</td>" for i, c in enumerate(r)) + "</tr>"
    return h + "</table>"


def figure(name, cap, width=None):
    if not os.path.exists(f"{FIG}/{name}"):
        return f"<p class='small'>[figure {esc(name)} is not built yet]</p>"
    return B.fig(name, cap, width)


# ------------------------------------------------------------------ tables
def tbl_main(p, keys, ref=None):
    rows = []
    for k in keys:
        if k not in MJ["perts"][p]["cards"]:
            continue
        cl, fa = card(p, k, "cl"), card(p, k, "fail")
        b = (bs(p, f"{k} - {ref[k]}") + " vs " + LABEL.get(ref[k], ref[k]).replace(" (recursion on)", "")) if ref and k in ref and boot(p, f"{k} - {ref[k]}") else ""
        rows.append([esc(LABEL.get(k, k)), f3(cl["S"]), f3(fa["S"]), f3(fa["F1@1.0"]), f3(fa["P@1.0"]), f3(fa["R@1.0"]), f3(fa["J"]), f3(fa["lanes_found"], 2), f3(fa["vert100"], 1), f3(fa["false_len_frac"]), b])
    return table(["system", "S clean half", "S failure half", "F1@1", "P@1", "R@1", "J", "lanes found", "v / 100 m", "false length", "F1@1 difference to the reference [95 % interval]"], rows, nums=set(range(1, 10)))


def tbl_more():
    rows = []
    for p in PERT4:
        a, b = MJ["perts"][p]["cards"]["champ_single"], MJ["perts"][p]["cards"]["champ_hold"]
        bb = bs(p, "champ_hold - champ_single") if boot(p, "champ_hold - champ_single") else "-"
        rows.append([esc(NAME[p]), f3(a["cl"]["S"]), f3(a["fail"]["S"]), f3(b["fail"]["S"]), f3(a["fail"]["F1@1.0"]), f3(b["fail"]["F1@1.0"]), f3(a["fail"]["J"]), f3(b["fail"]["J"]), f3(a["fail"]["lanes_found"], 2), f3(b["fail"]["lanes_found"], 2)])
    return table(["failure type", "S clean half", "S failure, single", "S failure, hold", "F1@1 single", "F1@1 hold", "J single", "J hold", "found single", "found hold"], rows, nums=set(range(1, 10)))


def route_cell(d, key, sig="0", ph="fail", route="s0"):
    c = MJ["route"].get(d)
    if not c:
        return None
    for k, v in c.items():
        if k.startswith(f"sigma{sig}_") and k.endswith(f"_{ph}") and f"_{route}" in k:
            return v["route_success"]
    return None


def tbl_route():
    """route success under failure, the two route-head seeds shown separately (they differ by 0.05 - 0.09); single-frame rows are recursion ON, hold and gate rows recursion OFF (the files of E135 were overwritten by E135b)"""
    rows = []
    for tag, nm, rec in (("e135_route/dark_single", "night, single frame", "on"), ("e135_route/dark_maxdecay_0.95", "night, spatial hold", "off"), ("e135_route/dark_gate", "night, learned gate", "off"),
                         ("e135_route/blur_single", "blur, single frame", "on"), ("e135_route/blur_maxdecay_0.95", "blur, spatial hold", "off"), ("e135_route/blur_gate", "blur, learned gate", "off"),
                         ("e135_route/cam_front3_off_single", "three front cameras off, single frame", "on"), ("e135_route/cam_front3_off_maxdecay_0.95", "three front cameras off, spatial hold", "off"), ("e135_route/cam_front3_off_gate", "three front cameras off, learned gate", "off")):
        c = MJ["route"].get(tag)
        if not c:
            continue
        def cell(sg, ph, rt):
            return next((v["route_success"] for k, v in c.items() if k.startswith(f"sigma{sg}_") and k.endswith(f"_route_{rt}_{ph}")), None)
        n = next((v["n"] for k, v in c.items() if k.endswith("_fail")), 0)
        clm = np.mean([cell("0", "cl", "s0"), cell("0", "cl", "s1")])
        f0, f1 = cell("0", "fail", "s0"), cell("0", "fail", "s1")
        f2 = np.mean([cell("2", "fail", "s0"), cell("2", "fail", "s1")])
        rows.append([esc(nm), rec, f3(clm), f3(f0), f3(f1), f3(f2), str(n)])
    return table(["series", "time recursion", "clean half (mean of 2 seeds)", "failure half, seed 0", "failure half, seed 1", "failure half, map error 2 m (mean)", "frames with a route / with a graph (failure half)"], rows, nums={2, 3, 4, 5, 6})


def tbl_map():
    rows = []
    for p in PERT3:
        for v, nm in (("single", "single frame"), ("hold", "spatial hold")):
            base = "norec_single" if v == "single" else "norec_hold"
            r = [esc(NAME[p]), nm, f3(S(p, base))]
            for mode in ("m0", "m2", "mnone", "mwrong"):
                k = f"map_{v}_{mode}"
                r.append(f3(S(p, k)) if k in MJ["perts"][p]["cards"] else "-")
            r.append(f3(S(p, f"map_{v}_m0", "fail", "lanes_found")) if f"map_{v}_m0" in MJ["perts"][p]["cards"] else "-")
            rows.append(r)
    return table(["failure", "input", "S champion (no map)", "S map builder, exact map", "map error 2 m", "no map", "wrong map", "lanes found (exact map)"], rows, nums=set(range(2, 8)))


def tbl_latency():
    d = f"{E17}/latency"
    rows = []
    if os.path.isdir(d):
        for f in sorted(os.listdir(d)):
            j = json.load(open(f"{d}/{f}"))
            g = lambda k: j[k]["median_ms"] if k in j else float("nan")
            tot = g("total") + g("memory")
            rows.append([f"M {j['M']} / K0 {j['K0']}", "on" if j.get("rec") else "off", f3(g("memory"), 1), f3(g("builder"), 1), f3(g("assembly"), 1), f3(g("route_input"), 1), f3(g("route_head"), 1), f3(g("consumer_prep") + g("consumer"), 1), f3(tot, 1)])
    return table(["beam", "time recursion", "memory (warp + gate)", "builder", "assembly", "route input", "route head", "consumer", "total, ms"], rows, nums=set(range(2, 9))) if rows else "<p class='small'>[latency not measured yet]</p>"


def _latency_total():
    try:
        j = json.load(open(f"{E17}/latency/latency_M288_rec0.json"))
        return f"{j['total']['median_ms'] + j['memory']['median_ms']:.0f} ms (beam 288 / 144, median; the rasterisation of the SD map is not timed)"
    except Exception:
        return "[latency not measured]"


latency_total = _latency_total()
has_e139 = any(k.startswith("m139b") for k in MJ["perts"]["dark"]["cards"])


def e139_html():
    import loop17_text_e139 as T9

    return T9.html(globals())


def build():
    import loop17_text as TX

    h = [f"<html><head><meta charset='utf-8'><title>Loop 17</title><style>{B.CSS}</style></head><body>"]
    h += TX.sections(globals())
    h.append("</body></html>")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w", encoding="utf-8").write("\n".join(h))
    print("written", OUT, len("\n".join(h)))


if __name__ == "__main__":
    build()
