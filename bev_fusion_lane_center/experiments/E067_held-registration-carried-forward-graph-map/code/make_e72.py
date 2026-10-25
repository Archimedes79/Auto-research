"""Bericht E72 (Schleife 5): intermittierende Sensorausfaelle (drei kurze Ausfallfenster je Szene) mit der empfohlenen Registrierungsregel; ohne Sprung / mit einem Sprung von 3 m mitten im zweiten Fenster.  Liest out/loop2/e72_<ort>_j3/metrics.json (combo_eval.py --windows); Text loop2/loop5_text_e72.py (TEXT72).  Aufruf (gcar): make_e72.py [figs|report]"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arch_figs5 as AF5  # noqa: E402
from report2 import f3, make_report2  # noqa: E402

OUT = "/mnt/c/ML-Experiment/bev_fusion_lane_center/runs"
FIG = f"{OUT}/figs"
os.makedirs(FIG, exist_ok=True)
_ns = {}
if os.path.exists("/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/reports_build/loop5_text_e72.py"):
    exec(open("/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/reports_build/loop5_text_e72.py", encoding="utf-8").read(), _ns)
TEXT = _ns.get("TEXT72", {})
RUNS = [("e72_none_j3", "no jump (three bursts)"), ("e72_during_j3", "3 m jump in the second burst")]
LAB = {"dark": "night", "cam_front3_off": "3 forward cameras off", "blur": "blur", "cam_front_off": "front camera off"}
PL = {"perframe": "per-frame registration", "track": "tracked, every frame (E61)", "hold": "tracked + monitor-gated hold (E67)", "holdrestart": "hold + restart when healthy and the score is low (E70)", "holdverify": "hold + restart + VERIFY with the wide search when the sensors return after a hold", "holdstrict": "hold + restart + STRICT verify (replace only if the wide search scores better)", "trackrestart": "restart on a low score WITHOUT the monitor gate", "gt": "true registration (floor)"}
PS = {"perframe": "per-frame registration", "track": "tracked (E61)", "hold": "tracked + hold (E67)", "holdrestart": "hold + restart (E70)", "holdverify": "hold + restart + verify (recommended)", "holdstrict": "hold + restart + strict verify", "trackrestart": "restart without monitor gate", "gt": "true registration (floor)"}
ES = {"e72_none_j3": "no jump", "e72_during_j3": "3 m jump in burst 2"}
COL = {"perframe": "#1f77b4", "track": "#ff7f0e", "hold": "#d62728", "holdrestart": "#2ca02c", "holdverify": "#17becf", "holdstrict": "#8c564b", "trackrestart": "#9467bd", "gt": "#7f7f7f"}
PH = ["before the first burst", "during the bursts", "after a burst: 0-1.5 s", "after a burst: 1.5-3 s", "after a burst: 3-5 s", "after a burst: 5 s or more"]
WIN = [(0.10, 0.25), (0.40, 0.55), (0.70, 0.85)]  # Ausfallfenster als Anteile der Szene (combo_eval.py --windows)


def jl(p):
    return json.load(open(p)) if os.path.exists(p) else None


def tables():
    runs = [(k, lab, jl(f"{OUT}/{k}/metrics.json")) for k, lab in RUNS]
    runs = [(k, lab, m) for k, lab, m in runs if m and "mean" in m]
    if not runs:
        return []
    h1 = ["event", "method"] + [f"lateral error (m): {p}" for p in PH[:4]]
    PH2 = ["before", "during", "after 0-1.5 s", "after 1.5-3 s"]
    r1, r2 = [], []
    for k, lab, m in runs:
        for p in m["pols"]:
            r1.append([ES[k], PS[p]] + [f3(x, 2) for x in m["mean"]["lata"][p][:4]])
            r2.append([ES[k], PS[p]] + [f3(x, 2) for x in m["mean"]["err"][p][:4]])
    h2 = ["event", "method"] + [f"offset error (m): {p}" for p in PH2]
    # je Bedingung (Sprung im Ausfall)
    k, lab, m = [x for x in runs if x[0] == "e72_during_j3"][0] if any(x[0] == "e72_during_j3" for x in runs) else runs[-1]
    h3 = ["condition", "method"] + [f"lateral error (m): {p}" for p in PH[:4]]
    r3 = []
    for c in m["conds"]:
        for p in ("perframe", "track", "hold", "holdrestart", "holdverify", "holdstrict", "trackrestart"):
            r3.append([LAB.get(c, c), PS[p]] + [f3(x, 2) for x in m["conds"][c]["lata"][p][:4]])
    h4 = ["event", "restarts of 'hold + restart' (before / during / after the failure)", "restarts and verifications of 'hold + restart + verify'", "restarts of the ungated restart (before / during / after)"]
    r4 = []
    for k, lab, m in runs:
        a = {"fail": 0, "before": 0, "after": 0}
        b = {"fail": 0, "before": 0, "after": 0}
        v = {"fail": 0, "before": 0, "after": 0}
        for c in m["conds"]:
            for ph in a:
                a[ph] += m["conds"][c]["restarts"]["holdrestart"][ph]
                b[ph] += m["conds"][c]["restarts"]["trackrestart"][ph]
                v[ph] += m["conds"][c]["restarts"].get("holdverify", {}).get(ph, 0)
        r4.append([ES[k], f"{a['before']} / {a['fail']} / {a['after']}", f"{v['before']} / {v['fail']} / {v['after']}", f"{b['before']} / {b['fail']} / {b['after']}"])
    return (h1, r1), (h2, r2), (h3, r3), (h4, r4), burst_table(runs)


def burst_table(runs):
    """Querfehler je Ausfallfenster (im Fenster / in den ersten 1.5 s danach) aus den Bild-Aufzeichnungen; nach dem dritten Fenster fehlen die letzten 3 s der Szene (keine Zukunftspositionen)"""
    import pickle
    pols = ("hold", "holdrestart", "holdverify", "holdstrict", "trackrestart", "gt")
    h = ["event", "method", "during burst 1", "0-1.5 s after burst 1", "during burst 2", "0-1.5 s after burst 2", "during burst 3"]
    rows = []
    for k, lab, m in runs:
        acc = {}
        for c in ("dark", "cam_front3_off", "blur", "cam_front_off"):
            fp = f"{OUT}/{k}/frames_{c}.pkl"
            if not os.path.exists(fp):
                continue
            for sc in pickle.load(open(fp, "rb")):
                if not sc:
                    continue
                n = len(sc)
                win = [(int(n * lo), int(n * hi)) for lo, hi in WIN]
                for r in sc:
                    f = r["f"]
                    if r["lata"] is None:
                        continue
                    if r["phase"] == "fail":
                        b = [i for i, (lo, hi) in enumerate(win) if lo <= f < hi][0]
                        key = ("fail", b)
                    elif r["phase"] == "after" and r["since_end"] < 3:
                        key = ("after", len([1 for lo, hi in win if hi <= f]) - 1)
                    else:
                        continue
                    for pol in pols:
                        acc.setdefault((pol,) + key, []).append(r["lata"][pol])
        for pol in pols:
            g = lambda ph, b: f3(float(np.mean(acc[(pol, ph, b)])), 2) if (pol, ph, b) in acc else "-"
            rows.append([ES[k], PS[pol], g("fail", 0), g("after", 0), g("fail", 1), g("after", 1), g("fail", 2)])
    return h, rows


def line_fig():
    runs = [(k, lab, jl(f"{OUT}/{k}/metrics.json")) for k, lab in RUNS]
    runs = [(k, lab, m) for k, lab, m in runs if m and "mean" in m]
    if not runs:
        return None
    fig, axs = plt.subplots(1, len(runs), figsize=(5.2 * len(runs), 4.6), sharey=True)
    axs = np.atleast_1d(axs).ravel()
    for ax, (k, lab, m) in zip(axs, runs):
        for p in m["pols"]:
            ax.plot(range(4), m["mean"]["lata"][p][:4], "o-", color=COL[p], label=PL[p], lw=1.8)
        ax.set_xticks(range(4))
        ax.set_xticklabels(["before", "during", "after\n0-1.5 s", "after\n1.5-3 s"], fontsize=9)
        ax.set_title(lab, fontsize=10)
        ax.grid(alpha=0.3)
    axs[0].set_ylabel("mean lateral error (m)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=9)
    fig.tight_layout(rect=[0, 0.2, 1, 1])
    p = f"{FIG}/e72_lines.png"
    fig.savefig(p, dpi=100)
    plt.close(fig)
    return p


def trace_fig(run="e72_during_j3", cond="dark"):
    """Zeitverlauf des Registrierungsfehlers in zwei Szenen (3 m Sprung mitten im Ausfall, Nacht): die Szene mit dem mittleren und die mit dem groessten Fehler 'holdverify' in den ersten 1.5 s nach der Rueckkehr der Sensoren (feste Regel, keine Handauswahl)"""
    import pickle
    fp = f"{OUT}/{run}/frames_{cond}.pkl"
    if not os.path.exists(fp):
        return None
    scenes = [sc for sc in pickle.load(open(fp, "rb")) if sc]
    def after_err(sc):
        v = [r["err"]["holdverify"] for r in sc if r["phase"] == "after" and r["since_end"] < 3]
        return float(np.mean(v)) if v else float("nan")
    ae = np.array([after_err(sc) for sc in scenes])
    ok = np.where(np.isfinite(ae))[0]
    order = ok[np.argsort(ae[ok])]
    picks = [(order[len(order) // 2], "typical scene (median error after the return)"), (order[-1], "worst scene (largest error after the return)")]
    fig, axs = plt.subplots(2, 2, figsize=(11, 5.6), sharex="col", gridspec_kw=dict(height_ratios=[3, 1]))
    for j, (idx, ttl) in enumerate(picks):
        sc = scenes[idx]
        n = len(sc)
        fj = int(n * 0.50)
        t = np.arange(n) * 0.5
        ax, ax2 = axs[0, j], axs[1, j]
        for lo, hi in WIN:
            ax.axvspan(int(n * lo) * 0.5, int(n * hi) * 0.5, color="#f2d0d0", alpha=0.6, lw=0)
            ax2.axvspan(int(n * lo) * 0.5, int(n * hi) * 0.5, color="#f2d0d0", alpha=0.6, lw=0)
        ax.axvline(fj * 0.5, color="k", ls="--", lw=1)
        for p in ("hold", "holdrestart", "holdverify", "trackrestart", "gt"):
            ax.plot(t, [r["err"][p] for r in sc], "-", color=COL[p], lw=1.6, label=PL[p])
            rs = [i for i, r in enumerate(sc) if p in r["restart"]]
            if rs and p != "gt":
                ax.plot(t[rs], [sc[i]["err"][p] for i in rs], "v", color=COL[p], ms=5)
        ax.set_title(f"{ttl} - TEST scene #{idx + 1} of {len(scenes)} ({n} frames)", fontsize=8)
        ax.set_ylabel("registration offset error (m)")
        ax.grid(alpha=0.3)
        ax2.axvline(fj * 0.5, color="k", ls="--", lw=1)
        ax2.plot(t, [r["pf"] for r in sc], color="#555555", lw=1.2)
        ax2.axhline(0.5, color="#999999", ls=":", lw=1)
        ax2.set_ylim(-0.05, 1.05)
        ax2.set_ylabel("monitor p(fail)")
        ax2.set_xlabel("time in the scene (s)")
        ax2.grid(alpha=0.3)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8)
    fig.tight_layout(rect=[0, 0.11, 1, 1])
    p = f"{FIG}/e72_trace.png"
    fig.savefig(p, dpi=100)
    plt.close(fig)
    return p


def main(mode):
    tabs_ = tables()
    if mode == "figs":
        for t in tabs_:
            for r in t[1]:
                print(r)
        return
    images = []
    try:
        p = line_fig()
        if p:
            images.append((p, "Lateral error of the registered map route before the first burst, during the three sensor-failure bursts and in the 0-1.5 s / 1.5-3 s after a burst (mean of night, three forward cameras off, blur and front camera off), without a jump and with a 3 m jump in the last third of the second burst."))
    except Exception as ex:  # noqa: BLE001
        print("line_fig Fehler:", ex)
    try:
        p = trace_fig()
        if p:
            images.append((p, "Registration offset error over time in two TEST scenes with three night-failure bursts (red bands) and a 3 m jump in the last third of the second burst (dashed line); triangles mark restarts / verifications. Left: the scene with the median, right: the scene with the largest error of 'hold + restart + verify' in the first 1.5 s after the bursts (fixed rule, not hand-picked). Below: the monitor p(fail)."))
    except Exception as ex:  # noqa: BLE001
        print("trace_fig Fehler:", ex)
    t = TEXT
    caps = ["Lateral error (m, anchored) of the registered map route by phase (mean over four failure types, the 1.5-3 s column over three because the blur scenes have no frames there; 21 TEST scenes; three bursts per scene at 10-25 %, 40-55 % and 70-85 % of the scene). Event: no jump / a 3 m jump in the last third of the second burst (at 50 %, the sensors are blind). 'After a burst' = time since the end of the most recent burst",
            "Registration offset error (m) at the vehicle by phase (mean over four failure types; events as above)",
            "Per failure type (3 m jump in the second burst)",
            "Restart events (all four failure types together; events as above)",
            "Lateral error (m) per burst (mean over four failure types): during each burst and in the first 1.5 s after it; the 3 m jump happens in the last third of burst 2; after burst 3 the last 3 s of each scene have no future positions and are not evaluated"]
    tabs = [(cap, tb[0], tb[1]) for cap, tb in zip(caps, tabs_)]
    make_report2("E72", t.get("title", "E72"), t.get("question", ""), t.get("setup", []), tabs, images, t.get("findings", ["(to be written)"]), t.get("limits", []), arch=AF5.burst_arch, arch_height=3.0)
    print("E72 written")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "figs")
