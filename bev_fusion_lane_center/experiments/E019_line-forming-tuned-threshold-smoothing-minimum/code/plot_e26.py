"""Bilder fuer die Sparsifizierung (Linienverfolger, E26 und Folgeversuche): Beispielbilder, Ablauf mit Linienpruefung, Zahlenuebersicht.
Aufruf: plot_e26.py [crn|gcar] [overlay] [steps] [numbers]   (Familie = Fundament; liest out/crn/exp_trace_<quelle>.json und trace_ex_<quelle>.pkl, schreibt out/crn/viz_trace_<familie>_*.png)"""
import json
import os
import pickle
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_trace as ET  # noqa: E402
import line_tracer as LT  # noqa: E402

OUT = "/mnt/c/ML-Experiment/bev_fusion_lane_center/runs/_loop1_crn"
FAM = {"crn": ("crn_orakel", "crn_unet64", "Kamera+Radar (CRN) U-Netz 64"), "gcar": ("crn_orakel", "gcar_unet64", "Kamera+Radar (GaussianCaR) U-Netz 64")}
RED, BLUE = "#d93025", "#1a73e8"
REASON_MARK = {0: ("x", "#d93025", "Feld zu hoch: Linie endet"), 1: ("D", "#1a73e8", "auf andere Linie getroffen: verschmolzen"), 2: ("s", "#5f6368", "Bildrand"), 3: ("o", "#f9ab00", "Schrittgrenze"), 4: ("o", "#f9ab00", "Speichergrenze")}


def f1_both(P, G):
    """Punkt-F1 und 1:1-F1 bei 1 m."""
    if len(P) == 0 or len(G) == 0:
        return 0.0, 0.0
    tG = cKDTree(G)
    dp = tG.query(P)[0]
    dg = cKDTree(P).query(G)[0]
    pr, rc = float((dp <= 1).mean()), float((dg <= 1).mean())
    m = ET.match_1to1(P, G, tG, taus=(1.0,))[0]
    return 2 * pr * rc / max(pr + rc, 1e-9), 2 * m / (len(P) + len(G))


def skel_pts(sk):
    P = np.argwhere(sk).astype(np.float32)
    P = np.stack([-50 + (P[:, 0] + 0.5) * 0.5, -50 + (P[:, 1] + 0.5) * 0.5], 1)
    return P[(np.abs(P) < 48).all(1)]


def gt_px(G):
    return np.stack([(G[:, 0] + 50) / 0.5 - 0.5, (G[:, 1] + 50) / 0.5 - 0.5], 1)


def style(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 199)
    ax.set_ylim(199, 0)


def overlay_fig(fam, path):
    _, key, label = FAM[fam]
    D, g, gts = ET.load_source(key)
    ex = pickle.load(open(f"{OUT}/trace_ex_{key}.pkl", "rb"))
    ids = sorted(ex)
    has_v1 = all(ex[i].get("polys_v1") is not None for i in ids)
    ncol = 5 if has_v1 else 4
    fig, axs = plt.subplots(len(ids), ncol, figsize=(4.2 * ncol, 4.25 * len(ids)))
    for r, i in enumerate(ids):
        e = ex[i]
        gp = gt_px(gts[i])
        G = gts[i]
        c = 0
        ax = axs[r, c]
        ax.imshow(np.clip(D[i], 0, 4), cmap="gray", vmin=0, vmax=4)
        ax.set_title(f"Val-Sample {i}: Abstandsfeld (Pixel)", fontsize=9)
        c += 1
        ax = axs[r, c]
        ax.imshow(np.ones((200, 200)), cmap="gray", vmin=0, vmax=1)
        ax.scatter(gp[:, 1], gp[:, 0], s=1.0, c=BLUE)
        yy, xx = np.nonzero(e["sk_tun"])
        ax.scatter(xx, yy, s=1.6, c=RED)
        Ps = skel_pts(e["sk_tun"])
        a1, a2 = f1_both(Ps, G)
        ax.set_title(f"Skelett (abgestimmt): {len(Ps)} Pixel\nF1@1m {a1:.2f} / 1:1 {a2:.2f}", fontsize=9)
        c += 1
        if has_v1:
            ax = axs[r, c]
            ax.imshow(np.ones((200, 200)), cmap="gray", vmin=0, vmax=1)
            ax.scatter(gp[:, 1], gp[:, 0], s=1.0, c=BLUE)
            for a in e["polys_v1"]:
                ax.plot(a[:, 1], a[:, 0], c=RED, lw=1.5)
            a1, a2 = f1_both(LT.polys_points_m(e["polys_v1"], 0.5), G)
            ax.set_title(f"Verfolger v1 (Punkt-F1 abgestimmt): {len(e['polys_v1'])} Linien\nF1@1m {a1:.2f} / 1:1 {a2:.2f}", fontsize=9)
            c += 1
        ax = axs[r, c]
        ax.imshow(np.ones((200, 200)), cmap="gray", vmin=0, vmax=1)
        ax.scatter(gp[:, 1], gp[:, 0], s=1.0, c=BLUE)
        for a in e["polys"]:
            ax.plot(a[:, 1], a[:, 0], c=RED, lw=1.5)
        a1, a2 = f1_both(LT.polys_points_m(e["polys"], 0.5), G)
        ax.set_title(f"Verfolger v2 (1:1 abgestimmt): {len(e['polys'])} Linien\nF1@1m {a1:.2f} / 1:1 {a2:.2f}", fontsize=9)
        c += 1
        ax = axs[r, c]
        ax.imshow(np.ones((200, 200)), cmap="gray", vmin=0, vmax=1)
        ax.scatter(gp[:, 1], gp[:, 0], s=1.0, c=BLUE)
        nn = 0
        for a in LT.sparse_nodes(e["polys"], 3.0):
            ax.plot(a[:, 1], a[:, 0], c=RED, lw=1.0)
            ax.scatter(a[:, 1], a[:, 0], s=8, c=RED)
            nn += len(a)
        a1, a2 = f1_both(LT.polys_points_m(LT.sparse_nodes(e["polys"], 3.0), 0.5), G)
        ax.set_title(f"Verfolger v2, Knoten alle 3 m: {nn} Knoten\nF1@1m {a1:.2f} / 1:1 {a2:.2f}", fontsize=9)
        for ax in axs[r]:
            style(ax)
    axs[0, 1].scatter([], [], s=6, c=BLUE, label="Soll (Karte)")
    axs[0, 1].scatter([], [], s=6, c=RED, label="geschaetzt")
    axs[0, 1].legend(fontsize=7, loc="lower right")
    fig.suptitle(f"{label}: geschaetzte Mittellinien (rot) gegen Soll (blau); F1@1m = Punkt-F1, 1:1 = F1 mit 1:1-Zuordnung", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=75)
    plt.close(fig)


def calib_lenient(D, gts, bins, step=4):
    """Feldwert am Linienpunkt gegen den wahren Abstand zur Karte, mit nachsichtiger Einstellung (t_on 1,5 m, Luecken bis 4 Schritte), damit auch hohe Feldwerte vorkommen."""
    par = dict(LT.DEFAULT, t_on=1.5, t_seed=0.6, gap_max=4, min_len=4.0, beta=0.0, dup_m=0.0)
    nb = len(bins) + 1
    n, ok1, ok05 = np.zeros(nb), np.zeros(nb), np.zeros(nb)
    for i in range(1, len(D), step):
        P = LT.polys_points_m(LT.trace_field(D[i], par)["polys"], 0.5)
        if len(P) == 0 or len(gts[i]) == 0:
            continue
        dt = cKDTree(gts[i]).query(P)[0]
        v = ndi.map_coordinates(D[i], [(P[:, 0] + 50) / 0.5 - 0.5, (P[:, 1] + 50) / 0.5 - 0.5], order=1, mode="nearest")
        b = np.digitize(v, bins)
        n += np.bincount(b, minlength=nb)
        ok1 += np.bincount(b, weights=(dt <= 1.0).astype(float), minlength=nb)
        ok05 += np.bincount(b, weights=(dt <= 0.5).astype(float), minlength=nb)
    return dict(n=n, ok1=ok1, ok05=ok05)


def steps_fig(fam, path, sample=1100):
    _, key, label = FAM[fam]
    D, g, gts = ET.load_source(key)
    J = json.load(open(f"{OUT}/exp_trace_{key}.json"))
    par = J["arms"]["Verfolger abgestimmt"]["params"]
    res = LT.trace_field(D[sample], par, want_debug=True)
    polys, meta = res["polys"], res["info"]["meta"]
    sr, sc = res["info"]["seeds"]
    gp = gt_px(gts[sample])
    cm = plt.get_cmap("viridis")
    fig, axs = plt.subplots(1, 4, figsize=(18, 4.6), gridspec_kw=dict(width_ratios=[1, 1, 1.15, 0.9]))
    ax = axs[0]
    ax.imshow(np.clip(D[sample], 0, 4), cmap="gray", vmin=0, vmax=4)
    ax.scatter(sc, sr, s=5, c="#f9ab00", label=f"Kandidaten ({len(sr)})")
    for k, m in enumerate(meta):
        ax.scatter([m["seed"][1]], [m["seed"][0]], s=26, c=[cm(k / max(len(meta) - 1, 1))], edgecolors="w", linewidths=0.6, zorder=3)
        ax.text(m["seed"][1] + 2, m["seed"][0] - 2, str(k + 1), color="#00e5ff", fontsize=7, zorder=4)
    ax.set_title(f"{label}, Sample {sample}\n1 Saatpunkte: lokale Minima des Feldes, beste zuerst", fontsize=9)
    ax.legend(fontsize=7, loc="lower right")
    ax = axs[1]
    ax.imshow(np.ones((200, 200)), cmap="gray", vmin=0, vmax=1)
    ax.scatter(gp[:, 1], gp[:, 0], s=0.8, c=BLUE, alpha=0.6)
    for k, a in enumerate(polys):
        ax.plot(a[:, 1], a[:, 0], c=cm(k / max(len(polys) - 1, 1)), lw=1.8)
        m = meta[k]
        for pt, rs in ((a[0], m["reason_start"]), (a[-1], m["reason_end"])):
            mk, col, _ = REASON_MARK[rs]
            ax.scatter([pt[1]], [pt[0]], s=34, marker=mk, c=col, zorder=3)
    for rs in (0, 1, 2):
        mk, col, lab = REASON_MARK[rs]
        ax.scatter([], [], s=34, marker=mk, c=col, label=lab)
    ax.legend(fontsize=6.5, loc="lower right")
    ax.set_title(f"2 Linien nach Reihenfolge ({len(polys)}), Marken = Grund des Endes", fontsize=9)
    ax = axs[2]
    order = np.argsort([-m["length"] for m in meta])[:3]
    tG = cKDTree(gts[sample])
    cols = ["#188038", "#a142f4", "#e8710a"]
    for c_, k in zip(cols, order):
        a = LT.resample(polys[k], 1.0)
        s_ = np.arange(len(a)) * 0.5
        v = ndi.map_coordinates(D[sample], [a[:, 0], a[:, 1]], order=1, mode="nearest")
        dt = tG.query(LT.px_to_m(a))[0]
        ax.plot(s_, v, c=c_, lw=1.4, label=f"Linie {k + 1}: Feldwert (Pruefung)")
        ax.plot(s_, dt, c=c_, lw=1.0, ls=":", label=f"Linie {k + 1}: wahrer Abstand zur Karte")
    ax.axhline(par["t_on"], c="k", ls="--", lw=0.9)
    ax.text(0.5, par["t_on"] + 0.04, f"Schwelle t_on = {par['t_on']} m".replace(".", ","), fontsize=7)
    ax.axhline(1.0, c="#999", ls="-", lw=0.6)
    ax.set_ylim(0, 3.2)
    ax.set_xlabel("Weg entlang der Linie [m]", fontsize=8)
    ax.set_ylabel("Abstand [m]", fontsize=8)
    ax.legend(fontsize=6, loc="upper right", ncol=1)
    ax.grid(alpha=0.3)
    ax.set_title("3 Pruefung: Feldwert am Punkt (durchgezogen) gegen wahren Abstand (gepunktet)", fontsize=9)
    ax = axs[3]
    bins = np.array([0.05, 0.15, 0.3, 0.6, 1.0, 1.5])
    cal = calib_lenient(D, gts, bins)
    lab = [f"<{bins[0]}".replace(".", ",")] + [f"{bins[i]}-{bins[i + 1]}".replace(".", ",") for i in range(len(bins) - 1)] + [f">={bins[-1]}".replace(".", ",")]
    n = cal["n"]
    ok1 = cal["ok1"] / np.maximum(n, 1)
    ok05 = cal["ok05"] / np.maximum(n, 1)
    x = np.arange(len(n))
    ax.bar(x - 0.2, ok1, 0.4, color="#1a73e8", label="wahr auf Karten-Linie (<= 1 m)")
    ax.bar(x + 0.2, ok05, 0.4, color="#9ac1f7", label="wahr auf Karten-Linie (<= 0,5 m)")
    for xi, (a_, nn) in enumerate(zip(ok1, n)):
        ax.text(xi - 0.2, a_ + 0.01, f"{a_:.2f}".replace(".", ","), ha="center", fontsize=6.5)
        ax.text(xi, 0.02, f"n={nn / 1000:.1f}k", ha="center", fontsize=6, color="w", rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(lab, fontsize=7)
    ax.set_xlabel("Feldwert am Linienpunkt [m]", fontsize=8)
    ax.set_ylim(0, 1.08)
    ax.legend(fontsize=6.5, loc="upper right")
    ax.set_title("4 Wie verlaesslich ist die Pruefung?\n(nachsichtige Einstellung t_on = 1,5 m, jedes 4. Sample)", fontsize=9)
    for a_ in axs[:2]:
        style(a_)
    fig.suptitle("Wie der Verfolger vorgeht: Saatpunkt, Schritt, Zentrieren, auf Linie pruefen, Ende oder Verschmelzen", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)
    json.dump(dict(bins=bins.tolist(), n=cal["n"].tolist(), ok1=cal["ok1"].tolist(), ok05=cal["ok05"].tolist()), open(f"{OUT}/trace_calib_{fam}.json", "w"), indent=1)


def numbers_fig(fam, path):
    ork, key, label = FAM[fam]
    keys = [ork, key]
    Js = {k: json.load(open(f"{OUT}/exp_trace_{k}.json")) for k in keys}
    names = {ork: "Soll-Feld (Orakel)", key: label.replace(" U-Netz 64", "\nU-Netz 64")}
    arms = [("Skelett Standard (0,5 m)", "Skelett Standard", "#bdc1c6"), ("Skelett abgestimmt", "Skelett abgestimmt", "#5f6368"), ("Verfolger v1 (Punkt-F1 abgestimmt)", "Verfolger v1", "#f6aea9"),
            ("Verfolger abgestimmt", "Verfolger v2", RED), ("Verfolger abgestimmt, 3-m-Knoten", "Verfolger v2, 3-m-Knoten", "#8c1d18")]
    fig, axs = plt.subplots(2, 3, figsize=(17, 9.0))
    labels = [names[k] for k in keys]
    x = np.arange(len(keys))
    nb = len(arms)
    w = 0.8 / nb

    def val(k, arm, getter):
        if arm not in Js[k]["arms"]:
            return np.nan
        if k.endswith("orakel") and arm == "Skelett abgestimmt":
            return np.nan  # fuer plateau-artige gelernte Felder abgestimmt, am exakten Soll-Feld nicht uebertragbar
        return getter(Js[k]["arms"][arm]["ungesehen"])

    panels = [
        ("F1 bei 1 m, Punkt-F1 (ungesehene Samples)", lambda u: u["1.0"][2], (0, 1.05), True),
        ("F1 bei 1 m mit 1:1-Zuordnung (doppelte Linien zaehlen nicht)", lambda u: u["o2o"]["1.0"][2], (0, 1.05), True),
        ("Spuren gefunden (>= 80 % einer Soll-Spur innerhalb 1 m)", lambda u: u["lane_recall"], (0, 1.05), True),
        ("Schleifen je Sample (eingeschlossene Flaechen >= 4 m2)", lambda u: u["topo"][1], None, False),
        ("Zusammenhangskomponenten je Sample", lambda u: u["topo"][0], None, False),
    ]
    for ax, (ttl, getter, ylim, pct) in zip(axs.ravel()[:5], panels):
        for j, (arm, lab, col) in enumerate(arms):
            vals = [val(k, arm, getter) for k in keys]
            xs = x + (j - (nb - 1) / 2) * w
            ax.bar(xs, vals, w, color=col, label=lab)
            for xi, v in zip(xs, vals):
                if not np.isnan(v):
                    ax.text(xi, v + (0.008 if pct else 0.1), (f"{v:.2f}" if pct else f"{v:.1f}").replace(".", ","), ha="center", fontsize=6, rotation=90 if not pct else 0)
        if not pct:
            gi = 1 if "Schleifen" in ttl else 0
            for xi, k in enumerate(keys):
                ax.hlines(Js[k]["gt"]["topo"][gi], xi - 0.4, xi + 0.4, colors=BLUE, linestyles="--", lw=1.2)
            ax.plot([], [], c=BLUE, ls="--", label="Karte (gezeichnet)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_title(ttl, fontsize=9.5)
        ax.grid(axis="y", alpha=0.3)
    hs, ls = axs[1, 0].get_legend_handles_labels()
    ax = axs[1, 2]
    names2 = [("Skelett abgestimmt", "Skelett-Pixel", "#5f6368"), ("Verfolger abgestimmt", "Verfolger 1 m", RED), ("Verfolger abgestimmt, 3-m-Knoten", "Verfolger 3 m", "#f28b82"), ("Verfolger abgestimmt, 6-m-Knoten", "Verfolger 6 m", "#fbc4bf")]
    for j, (arm, lab, col) in enumerate(names2):
        v = Js[key]["arms"][arm]["ungesehen"]["nodes"]
        ax.bar(j, v, 0.7, color=col, label=lab)
        ax.text(j, v + 12, f"{v:.0f}", ha="center", fontsize=8)
    gtn = Js[key]["gt"]["nodes"]
    ax.hlines(gtn, -0.5, len(names2) - 0.5, colors=BLUE, linestyles="--", lw=1.0)
    ax.text(len(names2) - 0.5, gtn + 15, f"Karte: {gtn:.0f} Knoten (alle 3 m)", color=BLUE, fontsize=7.5, ha="right")
    ax.set_xticks(range(len(names2)))
    ax.set_xticklabels([n[1] for n in names2], fontsize=8)
    ax.set_title("Punkte je Sample: dichtes Skelett gegen sparse Knoten", fontsize=9.5)
    ax.grid(axis="y", alpha=0.3)
    fig.legend(hs, ls, loc="upper center", ncol=6, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=80)
    plt.close(fig)


if __name__ == "__main__":
    args = sys.argv[1:]
    fam = "gcar" if "gcar" in args else "crn"
    what = [a for a in args if a not in ("crn", "gcar")] or ["overlay", "steps", "numbers"]
    if "overlay" in what:
        overlay_fig(fam, f"{OUT}/viz_trace_{fam}_beispiele.png")
    if "steps" in what:
        steps_fig(fam, f"{OUT}/viz_trace_{fam}_schritte.png")
    if "numbers" in what:
        numbers_fig(fam, f"{OUT}/viz_trace_{fam}_zahlen.png")
    print("FERTIG")
