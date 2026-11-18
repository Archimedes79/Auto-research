"""Pictures and a video of the grid chain on real VAL frames (E271): the 1 m centre-lane map, the sparse vertices against the ground truth and the prototype's, the assignment of the
vertices to the SD-map roads (START-value assigner), and the front camera.

    wsl bash tools/env/_py14.sh grid_view.py RUN [--scene 7] [--frames 16] [--every 2] [--out DIR] [--video] [--device cpu]

RUN = a grid run folder name under runs/ (e.g. e271_a_s0, or a chain run e272_j_s0: then the FUSED map, the hold and the TRAINED assigner; model.pt and the read-out threshold of runs/e27x/RUN/card.json). Four panels per frame (ego frame, metres, x right, y forward):
  heat      sigmoid(p) of the head at 1 m (white = lane), ground truth in green
  vertices  the read-out vertices of the head (blue, arrow = heading, size = weight), the prototype's node ids (orange cross) and the ground truth (green); per frame cov5 / prec1 / vertices per 100 m
  assignment  the SD roads (grey, arrow = direction), the head's vertices coloured by the road they are assigned to, red cross = no road (START-value assigner of training/grid/assign.py: not trained)
  camera    front camera (orientation only)
All from cached features and stored maps; CPU is enough. The prototype row is the registry entry recall_sp0_dp (`--registry` to change).
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")  # a shared machine
import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for sub in ("training/grid", "evaluation", "training/data", "tools/prototype", "tools/viz", "tools/video"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import assign as AS  # noqa: E402
import chain as CH  # noqa: E402
import chain_data as CD  # noqa: E402
import head as GH  # noqa: E402
import soft_lines as SL  # noqa: E402
import vertex_eval as VE  # noqa: E402
from lane_task import measures as M  # noqa: E402

CMAP = plt.get_cmap("tab10")


def to_grid(p):
    p = np.asarray(p, np.float32)
    return np.stack([p[..., 1], -p[..., 0]], -1)


def scene_ks(scene, frames, every):
    import evaluate as EV
    rows = np.array(json.load(open(f"{EV.R}/data/splits.json"))[EV.SPLITS["val"][1]])
    pz = np.load(f"{EV.HOME}/exp_gcar_b123/poses_val.npz")
    sc, ts = pz["scene"][rows], pz["ts"][rows]
    ix = sorted(np.nonzero(sc == scene)[0], key=lambda i: ts[i])[::every]
    return [int(i) for i in ix][:frames]


def counts_text(V, H, G, T, n_ids=None):
    c = M.vertex_numbers(M.vertex_counts(V, H, G, T, ids=n_ids))
    return f"cov5 {c['cov5']:.2f}  prec1 {c['prec1']:.2f}  {c['vgt100']:.1f}/100 m"


def draw(fig, d):
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1], hspace=0.18, wspace=0.10)
    ext = (-50, 50, -50, 50)

    def base(ax, title):
        ax.set_xlim(-50, 50), ax.set_ylim(-50, 50), ax.set_aspect("equal")
        ax.set_title(title, fontsize=8, loc="left")
        ax.tick_params(labelsize=6)
        ax.scatter([0], [0], marker="^", c="k", s=30, zorder=9)

    def gt(ax):
        for p in d["gt"]["pts"]:
            p = np.asarray(p)
            if len(p) > 1:
                ax.plot(p[:, 0], p[:, 1], color=(0.35, 0.8, 0.35), lw=2.0, zorder=2)

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(d["heat"].T, origin="lower", extent=ext, cmap="gray", vmin=0, vmax=1, zorder=0)
    gt(ax)
    base(ax, "centre-lane map p (1 m), ground truth green")
    ax = fig.add_subplot(gs[0, 1])
    base(ax, "vertices: head (blue), prototype nodes (orange x), GT (green)")
    gt(ax)
    V, H, w = d["V"], d["H"], d["w"]
    if len(V):
        ax.scatter(V[:, 0], V[:, 1], s=14 + 50 * w, c="tab:blue", edgecolor="k", linewidth=0.4, zorder=5)
        ax.quiver(V[:, 0], V[:, 1], H[:, 0], H[:, 1], color="tab:blue", scale=40, width=0.004, zorder=4)
    if len(d["rV"]):
        ax.scatter(d["rV"][:, 0], d["rV"][:, 1], marker="x", c="tab:orange", s=26, zorder=6)
    ax.text(-49, -48, f"head: {d['txt_a']}\nprototype: {d['txt_r']}", fontsize=6.5, bbox=dict(fc="white", alpha=0.8, lw=0), zorder=10)
    ax = fig.add_subplot(gs[0, 2])
    if d["cam"] is not None:
        ax.imshow(d["cam"])
    ax.axis("off")
    ax.set_title("front camera", fontsize=8, loc="left")
    mp = d["mp"]
    col = {}
    ax = fig.add_subplot(gs[1, 0])
    base(ax, "SD map: one colour per road (arrow = direction, n = lanes)")
    for k, (poly, cnt) in enumerate(d["sd"]["roads"]):
        poly = np.asarray(poly)
        if len(poly) < 2:
            continue
        col[k] = CMAP(k % 10)
        ax.plot(poly[:, 0], poly[:, 1], color=col[k], lw=2.4, solid_capstyle="round", zorder=2)
        m = len(poly) // 2
        ax.annotate("", xy=poly[min(m + 1, len(poly) - 1)], xytext=poly[m], arrowprops=dict(arrowstyle="-|>", color=col[k], lw=1.4), zorder=3)
        ax.text(poly[m, 0], poly[m, 1], f" {k}  n={int(round(float(np.mean(cnt))))}", fontsize=6.5, color=col[k], zorder=4)
    ax = fig.add_subplot(gs[1, 1])
    base(ax, "system: vertices in the colour of the road they are assigned to")
    for k, (poly, cnt) in enumerate(d["sd"]["roads"]):
        poly = np.asarray(poly)
        if len(poly) > 1:
            ax.plot(poly[:, 0], poly[:, 1], color="0.85", lw=1.0, zorder=0)
    if len(V) and d["P"] is not None:
        P, R = d["P"], mp["n_ids"]
        best = P[:, :R].argmax(1)
        un = P[:, R] > 0.5
        ok = ~un
        slot = d["slot"].argmax(1) if d.get("slot") is not None else np.zeros(len(V), int)
        for sl, mk in enumerate(("o", "s", "^", "D")):
            sel = ok & (slot == sl)
            if sel.any():
                ax.scatter(V[sel, 0], V[sel, 1], marker=mk, s=22 + 45 * w[sel], c=[col.get(int(b_), "0.5") for b_ in best[sel]], edgecolor="k", linewidth=0.4, zorder=5)
        ax.scatter(V[un, 0], V[un, 1], marker="x", c="red", s=34, zorder=6)
        ax.text(-49, -48, f"{int(ok.sum())} of {len(V)} vertices on a road; red x = no road\nmarker = lane slot (circle 0, square 1, triangle 2, diamond 3)", fontsize=6.5, bbox=dict(fc="white", alpha=0.8, lw=0), zorder=10)
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    ax.text(0, 1, d["title"], va="top", fontsize=8, family="monospace", wrap=True)
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--scene", type=int, default=7)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--registry", default="recall_sp0_dp")
    ap.add_argument("--out", default=None)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--thr", type=float, default=None)
    a = ap.parse_args()
    import evaluate as EV
    import lane_prototype as LP
    import make_results_video as RV
    import data as GD
    import p8_lib as L
    from PIL import Image

    out_dir = os.path.join(ROOT, a.out) if a.out else os.path.join(ROOT, "runs", "grid_view", a.run)
    os.makedirs(out_dir, exist_ok=True)
    ks = scene_ks(a.scene, a.frames, a.every)
    chain = a.run.startswith("e272")                                       # a chain run: the fused map and the TRAINED assigner
    card_p = os.path.join(ROOT, "runs", "e272" if chain else "e271", a.run, "card.json")
    thr = a.thr if a.thr is not None else (json.load(open(card_p))["thr"] if os.path.exists(card_p) else 0.3)
    width = json.load(open(os.path.join(ROOT, "runs", a.run, "config.json"))).get("width", 48)
    model = CH.GridChain(width=width) if chain else GH.GridHead(width=width)
    model.load_state_dict(torch.load(os.path.join(ROOT, "runs", a.run, "model.pt"), map_location="cpu")["state"])
    model.eval().to(a.device)
    ds, dl = CD.loader("val_clean", bs=4, workers=0, shuffle=False, ks=ks) if chain else GD.loader("val_clean", bs=4, workers=0, shuffle=False, ks=ks)
    e = EV.registry()[a.registry]
    g_r, conf, gts, info = LP.delivered(e, "val", ks)
    ft = RV.frame_table("val")
    sds = L.cache("sd", ds.bd.part)
    mod = model.assigner if chain else AS.Assigner()
    frames, n = [], 0
    with torch.no_grad():
        for b in dl:
            if chain:
                o_ = model(b["x"].to(a.device), b["step"].to(a.device), b["has"].to(a.device))
                outp, sp = o_["outs"][-1].cpu(), o_["fused"].cpu()
            else:
                outp = model(b["x"].to(a.device).float()).cpu()
                sp = SL.sparsify(outp)
            for i in range(outp.shape[0]):
                k, row = int(b["k"][i]), int(b["row"][i])
                j = ks.index(k)
                V, H, w = VE.read_out(sp[i], thr)
                m = (b["m05"][i, -1] if chain else b["m05"][i]) > 0
                g_, t_ = (b["g05"][i, -1], b["t05"][i, -1]) if chain else (b["g05"][i], b["t05"][i])
                G, T = g_[m].numpy().astype(np.float64), t_[m].numpy().astype(np.float64)
                rV, rH, rids = VE.node_vertices(g_r[j])
                sd = sds[row]
                P, mp, slot = None, None, None
                if sd is not None:
                    mp = AS.map_tokens(dict(roads=sd["roads"], route=[]))
                    if len(V) and len(mp["pos"]):
                        o = mod(torch.from_numpy(to_grid(V)), torch.from_numpy(to_grid(H)), torch.from_numpy(w.astype(np.float32)), AS.to_torch(mp))
                        P = o["P_road"].numpy()
                        slot = o["P_slot"].numpy()
                cam = None
                try:
                    cam = np.asarray(Image.open(f"{EV.HOME}/data/nuscenes_full/{ft['info'][k]['cam_infos']['CAM_FRONT']['filename']}").convert("RGB").resize((480, 270)))
                except Exception:  # noqa: BLE001
                    pass
                d = dict(heat=torch.sigmoid(outp[i, 0]).numpy(), gt=gts[j], V=V, H=H, w=w, rV=rV, txt_a=counts_text(V, H, G, T), txt_r=counts_text(rV, rH, G, T, rids),
                         mp=mp, slot=slot, sd=sd if sd is not None else dict(roads=[], route=[]), P=P, cam=cam,
                         title=f"{a.run}  VAL scene {a.scene}\nframe {n + 1}/{len(ks)}  (k {k})\nread-out thr {thr:.3f}\n\n" + ("map: fused over 3 frames\nassignment: trained" if chain else "assignment: start values,\nnot trained"))
                fig = plt.figure(figsize=(14, 8), dpi=80)
                draw(fig, d)
                path = os.path.join(out_dir, f"scene{a.scene}_f{n:02d}.png")
                fig.savefig(path, bbox_inches="tight")
                fig.canvas.draw()
                frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
                plt.close(fig)
                print(f"frame k {k}: head {d['txt_a']} | prototype {d['txt_r']} -> {os.path.relpath(path, ROOT)}")
                n += 1
    if a.video and frames:
        import make_video as MV
        h, w_ = min(f.shape[0] for f in frames), min(f.shape[1] for f in frames)
        MV.frames_to_video([f[:h, :w_] for f in frames], os.path.join(out_dir, f"scene{a.scene}.mp4"), fps=2)
        print("video", os.path.join(out_dir, f"scene{a.scene}.mp4"))


if __name__ == "__main__":
    main()
