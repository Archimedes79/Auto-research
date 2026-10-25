"""Bericht E40: Karten-Vorwissen am Gitter (dichter Kopf mit drei Zusatzkanaelen): Ausrichtung, Leck-Pruefung, Ablationen (ohne Karte, Pose-Rauschen, veraltete Karte), Kontrollen (Karte am falschen Ort, Orakel).
Aufruf (gcar): make_e40.py [figs|report].  Liest out/loop2/e40_*/metrics.json, sys_e40_*/metrics.json (Systemwirkung: Vorschlaege + Verfeinerer), align_eval.json, leak_stats.json. Text: loop2/text_e37_e42.py TEXT['e40']."""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arch_figs2 as AF2  # noqa: E402
import dense_report_lib as DR  # noqa: E402
from report2 import f3, make_report2  # noqa: E402

OUT = "/mnt/c/ML-Experiment/bev_fusion_lane_center/runs"
FIG = f"{OUT}/figs"
os.makedirs(FIG, exist_ok=True)
HOME = os.path.expanduser("~")
_ns = {}
if os.path.exists("/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/reports_build/text_e37_e42.py"):
    exec(open("/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/reports_build/text_e37_e42.py", encoding="utf-8").read(), _ns)
TEXT = _ns.get("TEXT", {})

# (Lauf, Beschriftung, Gruppe)
RUNS = [
    ("e37_final", "E37 final head (no fine-tune, reference)", "ref"),
    ("e40_nomap_ft", "no map (control: same extra fine-tuning steps)", "ref"),
    ("e40_clean", "map prior, exact pose", "map"),
    ("e40_clean_s03_noalign", "  same exact-pose model, test noise 0.3 m / 0.6 deg, NO alignment", "map"),
    ("e40_clean_s03_align", "  same exact-pose model, test noise 0.3 m / 0.6 deg, WITH alignment", "map"),
    ("e40_clean_s05_noalign", "  same exact-pose model, test noise 0.5 m / 1 deg, NO alignment", "map"),
    ("e40_clean_s05_align", "  same exact-pose model, test noise 0.5 m / 1 deg, WITH alignment", "map"),
    ("e40_clean_s1_noalign", "  same exact-pose model, test noise 1 m / 2 deg, NO alignment", "map"),
    ("e40_clean_s1_align", "  same exact-pose model, test noise 1 m / 2 deg, WITH alignment", "map"),
    ("e40_clean_s2_noalign", "  same exact-pose model, test noise 2 m / 4 deg, NO alignment", "map"),
    ("e40_clean_s2_align", "  same exact-pose model, test noise 2 m / 4 deg, WITH alignment", "map"),
    ("e40_clean_stale3", "  same exact-pose model, outdated map (30 % of the prior replaced)", "map"),
    ("e40_clean_stale6", "  same exact-pose model, outdated map (60 % of the prior replaced)", "map"),
    ("e40_noise", "map prior, trained + tested with pose noise (1 m / 2 deg)", "map"),
    ("e40_noise_align", "map prior, trained + tested with pose noise 1 m / 2 deg AND alignment", "map"),
    ("e40_noise_s0_noalign", "  same model, test with exact pose", "map"),
    ("e40_noise_s2_noalign", "  same model, test noise 2 m / 4 deg, NO alignment", "map"),
    ("e40_noise_s2_align", "  same model, test noise 2 m / 4 deg, WITH alignment", "map"),
    ("e40_noise_s3_noalign", "  same model, test noise 3 m / 6 deg, NO alignment", "map"),
    ("e40_noise_s3_align", "  same model, test noise 3 m / 6 deg, WITH alignment", "map"),
    ("e40_noise_stale3", "  same model, outdated map (30 % of the prior replaced)", "map"),
    ("e40_noise_stale6", "  same model, outdated map (60 % of the prior replaced)", "map"),
    ("e40_wrong", "CONTROL: prior from a different place", "ctl"),
    ("e40_oracle", "CONTROL (leak): ground-truth centerline as prior channel", "ctl"),
]


def jl(p):
    return json.load(open(p)) if os.path.exists(p) else None


def best(res, st="val"):
    return DR.best_tau(res, on=st if st in res else "val")


def align_example_fig(path, sigma=(2.0, 4.0), n=3):
    import align_tools as AT
    import lf_inputs as LI

    sp = json.load(open("/mnt/c/ML-Experiment/bev_fusion_lane_center/data/splits.json"))
    ids = np.array(sp["val"])
    sel = ids[np.linspace(15, len(ids) - 16, n).astype(int)]
    prov = LI.PriorProvider("b123", "map", sigma[0], sigma[1], 0.0, False)
    lg = np.load(f"{HOME}/exp_gcar_b123/feats/val_logit7.npy", mmap_mode="r")
    fig, axs = plt.subplots(n, 3, figsize=(11.5, 3.8 * n), squeeze=False)
    for r, i in enumerate(sel):
        pr, noise = prov.get("val", int(i))
        raw = torch.from_numpy(np.asarray(lg[int(i)], np.float32))[None].cuda()
        obs = LI.obs_from_logits(raw)
        al, est = AT.align(pr[None].cuda(), obs)
        ob = obs[0, 1].cpu().numpy()
        for c, (ttl, p) in enumerate((("prior with pose noise", pr[1].numpy()), ("after alignment", al[0, 1].cpu().numpy()))):
            ax = axs[r, c]
            ax.imshow(ob.T, origin="lower", extent=(-50, 50, -50, 50), cmap="gray_r", vmin=0, vmax=1)
            yy, xx = np.nonzero(p.T > 0.5)
            ax.plot(-50 + (xx + 0.5) * 0.5, -50 + (yy + 0.5) * 0.5, ".", color="#d62728" if c == 0 else "#1a9850", ms=1.0)
            ax.set_title(f"{ttl} - sample {int(i)}" if r == 0 else f"sample {int(i)}", fontsize=8)
            ax.set_xlim(-50, 50)
            ax.set_ylim(-50, 50)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=6)
        ax = axs[r, 2]
        ax.axis("off")
        e = est[0].cpu().numpy()
        t_ = noise.numpy()
        ax.text(0.02, 0.6, f"applied pose noise: dx {t_[0]:+.2f} m, dy {t_[1]:+.2f} m, dtheta {t_[2]:+.2f} deg\nestimated correction: dx {e[0]:+.2f} m, dy {e[1]:+.2f} m, dtheta {e[2]:+.2f} deg\n"
                f"(a perfect estimate equals the applied noise)", fontsize=8, va="top")
    fig.suptitle("alignment by correlation search: foundation's divider prediction (grey) vs. map dividers (dots)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return path


def main(mode):
    res = {n: DR.load(n) for n, _, _ in RUNS}
    res = {k: v for k, v in res.items() if v and "val" in v}
    head = ["condition", "TRAIN_EVAL F1@1m", "VAL F1@0.5m", "VAL F1@1m", "VAL F1@1.5m", "VAL P@1m", "VAL R@1m", "TEST F1@1m", "TEST P / R @1m", "d VAL F1@1m vs no-map control", "alignment error (median, m) before -> after"]
    rows = []
    base = res.get("e40_nomap_ft") or res.get("e37_final")
    bt0 = best(base) if base else "0.5"
    f_base = base["val"][bt0]["F1@1.0"] if base else None
    for n, lab, g in RUNS:
        if n not in res:
            continue
        r = res[n]
        bt = best(r)
        v, t = r["val"][bt], r["test"][bt]
        tr = f3(r["train_eval"][bt]["F1@1.0"]) if "train_eval" in r else "-"
        al = r.get("val_align")
        a_txt = f"{al['pos_err_before_median']:.2f} -> {al['pos_err_median']:.2f}" if al else "-"
        d = f"{v['F1@1.0'] - f_base:+.3f}" if f_base is not None else "-"
        rows.append([lab, tr, f3(v["F1@0.5"]), f3(v["F1@1.0"]), f3(v["F1@1.5"]), f3(v["P@1.0"]), f3(v["R@1.0"]), f3(t["F1@1.0"]), f"{f3(t['P@1.0'])} / {f3(t['R@1.0'])}", d, a_txt])
    # Tabelle 2: Ausrichtungsgenauigkeit (Schleife 2)
    ae = jl(f"{OUT}/align_eval.json") or {}
    h2 = ["pose noise (m / deg)", "outdated map", "median position error before (m)", "after alignment (m)", "share within 0.5 m before", "after", "share within 1 m after", "median |dx| / |dy| / |dtheta| after (m, m, deg)"]
    r2 = []
    for k, v in ae.items():
        r2.append([f"{v['sigma_xy']} / {v['sigma_deg']}", f"{v['stale']:.0%}", f3(v["pos_err_before_median"], 2), f3(v["pos_err_after_median"], 2), f3(v["within_0p5_before"], 2), f3(v["within_0p5_after"], 2), f3(v["within_1p0_after"], 2),
                   f"{v['x_err_median']:.2f} / {v['y_err_median']:.2f} / {v['th_err_median']:.2f}"])
    # Tabelle 3: Leck-Statistik
    lk = jl(f"{OUT}/leak_stats.json") or {}
    h3 = ["statistic (300 VAL samples, ground-truth centerline points)", "value"]
    r3 = []
    if lk:
        r3 = [["share of GT centerline points within 0.5 m of a map divider / road-edge line used as prior", f3(lk["share_within_0p5m"], 3)], ["... within 1 m", f3(lk["share_within_1p0m"], 3)], ["... within 2 m", f3(lk["share_within_2p0m"], 3)],
              ["median distance from a GT centerline point to the nearest prior divider (m)", f3(lk["divider_dist_median_m"], 2)], ["share of GT centerline points inside the prior's drivable area", f3(lk["share_inside_drivable"], 3)]]
    # Tabelle 4: Systemwirkung
    h4 = ["dense head used for proposals", "VAL F1@1m (proposals)", "VAL F1@1m (refined)", "VAL route F (refined)", "VAL split / merge F1 6 m (refined)", "TEST F1@1m (refined)", "TEST route F (refined)"]
    r4 = []
    for n, lab, g in RUNS:
        s = jl(f"{OUT}/sys_{n}/metrics.json")
        if not s or "val" not in s:
            continue
        v, t = s["val"]["refined"], s.get("test", {}).get("refined", {})
        r4.append([lab, f3(s["val"]["proposals"]["F1@1.0"]), f3(v["F1@1.0"]), f3(v["route_f"]), f"{f3(v['split6_F1'])} / {f3(v['merge6_F1'])}", f3(t.get("F1@1.0")), f3(t.get("route_f"))])
    for key, lab in (("e40_noise_align", "map prior, trained + tested with pose noise 1 m / 2 deg AND alignment"), ("e43_joint", "map prior (exact pose) + previous-frame features (E41 A), joint head")):
        s_ = jl(f"{OUT}/sys_{key}/metrics.json")
        e_ = jl(f"{OUT}/e43/metrics_{key}_noprop.json")
        if s_ and "val" in s_:
            v, t = s_["val"]["refined"], s_.get("test", {}).get("refined", {})
            r4.append([lab, f3(s_["val"]["proposals"]["F1@1.0"]), f3(v["F1@1.0"]), f3(v["route_f"]), f"{f3(v['split6_F1'])} / {f3(v['merge6_F1'])}", f3(t.get("F1@1.0")), f3(t.get("route_f"))])
        elif e_ and "val" in e_:
            v, t = e_["val"]["rule"]["metrics"], e_.get("test", {}).get("rule", {}).get("metrics", {})
            r4.append([lab, "-", f3(v["F1@1.0"]), f3(v["route_f"]), f"{f3(v['split6_F1'])} / {f3(v['merge6_F1'])}", f3(t.get("F1@1.0")), f3(t.get("route_f"))])
    images = []
    # Abbildung 1: Balken
    sel = [(n, lab) for n, lab, g in RUNS if n in res]
    if sel:
        fig, ax = plt.subplots(figsize=(11.5, 4.2))
        x = np.arange(len(sel))
        for j, (st, col) in enumerate((("val", "#1f77b4"), ("test", "#ff7f0e"))):
            vals = [res[n][st][best(res[n])]["F1@1.0"] for n, _ in sel]
            ax.bar(x + (j - 0.5) * 0.38, vals, 0.38, label=st.upper(), color=col)
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace("e40_", "").replace("e37_final", "final") for n, _ in sel], rotation=30, ha="right", fontsize=8)
        lo = min(res[n][st][best(res[n])]["F1@1.0"] for n, _ in sel for st in ("val", "test"))
        ax.set_ylim(max(lo - 0.04, 0), None)
        ax.set_ylabel("F1 @ 1 m (dense head)")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_title("map prior ablations (threshold chosen on VAL per condition)", fontsize=9)
        fig.tight_layout()
        p = f"{FIG}/e40_bars.png"
        fig.savefig(p, dpi=100)
        plt.close(fig)
        images.append((p, "Dense-head F1@1m under each condition."))
    # Abbildung 2: Pose-Rauschen -> F1 (VAL) fuer die Modelle, mit / ohne Ausrichtung
    series = {
        "exact-pose model, no alignment": [(0, "e40_clean"), (0.3, "e40_clean_s03_noalign"), (0.5, "e40_clean_s05_noalign"), (1, "e40_clean_s1_noalign")],
        "exact-pose model, WITH alignment": [(0.3, "e40_clean_s03_align"), (0.5, "e40_clean_s05_align"), (1, "e40_clean_s1_align")],
        "noise-trained model (1 m), no alignment": [(1, "e40_noise"), (2, "e40_noise_s2_noalign"), (3, "e40_noise_s3_noalign")],
        "noise-trained model, WITH alignment": [(1, "e40_noise_s1_align"), (2, "e40_noise_s2_align"), (3, "e40_noise_s3_align")],
        "noise + alignment trained model": [(1, "e40_noise_align")],
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for lab, pts in series.items():
        xy = [(x, res[n]["val"][best(res[n])]["F1@1.0"]) for x, n in pts if n in res]
        if xy:
            ax.plot([a for a, _ in xy], [b for _, b in xy], "-o", label=lab)
    if base:
        ax.axhline(f_base, color="gray", ls="--", label="no map (control)")
    ax.set_xlabel("test pose noise sigma_xy (m); sigma_theta = 2 sigma_xy (deg)")
    ax.set_ylabel("VAL F1 @ 1 m (dense head)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    p = f"{FIG}/e40_noise.png"
    fig.savefig(p, dpi=100)
    plt.close(fig)
    images.append((p, "Robustness to pose noise: a model trained with an exact pose loses its gain beyond ~0.5 m and falls below the no-map control at 1 m; alignment (residual ~0.5 m) keeps it above; a noise-trained model is safe but gains little."))
    try:
        images.append((align_example_fig(f"{FIG}/e40_align_example.png"), "Alignment examples (VAL): the noisy prior (red) is moved onto the foundation's own divider observation; the result (green) is compared with the observation (grey)."))
    except Exception as ex:  # noqa: BLE001
        print("Ausrichtungsbild:", ex)
    # Qualitativ: Heatmaps
    if "e40_clean" in res and "e40_nomap_ft" in res:
        pr, ids = DR.load_prob("e40_clean", "val")
        selx = list(np.linspace(5, len(ids) - 6, 5).astype(int))
        tau = {n: float(best(res[n])) for n in ("e40_nomap_ft", "e40_clean")}
        p = DR.qualitative(["e40_nomap_ft", "e40_clean"], {"e40_nomap_ft": "no map", "e40_clean": "map prior"}, "val", selx, tau, f"{FIG}/e40_qual.png")
        images.append((p, "VAL examples: heat map without and with the map prior."))
    if mode == "figs":
        for r in rows + r2 + r3 + r4:
            print(r)
        return
    t = TEXT.get("e40", {})
    make_report2("E40", t.get("title", "E40"), t.get("question", ""), t.get("setup", []),
                 [("Map prior at the dense head: ablations and controls (threshold chosen on VAL per row)", head, rows), ("Alignment accuracy of the correlation search (300 VAL samples, from the alignment evaluation)", h2, r2),
                  ("Leakage statistics: how much of the ground truth is spelled out by the prior channels", h3, r3), ("System effect: dense head with prior -> proposals -> refiner E39 (plug-in, no retraining)", h4, r4)],
                 images, t.get("findings", ["(to be written)"]), t.get("limits", []), arch=AF2.prior_grid_arch)
    print("E40 written")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "figs")
