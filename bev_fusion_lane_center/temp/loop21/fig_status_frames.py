"""Loop 21 status report, figures A and B (CPU, cached raw outputs, 1 thread).

    wsl bash tools/env/_py14.sh temp/loop21/fig_status_frames.py

A  s21_frames_<city>.png   three VAL frames per city, chosen by rank of the per-frame F1@1.0 of recall_dp inside the city (quantiles 0.85 / 0.50 / 0.15, a different scene each, not hand-picked):
                           camera with the graph projected onto the road (ground truth green, recall_dp red) | ego-frame graphs: control E94, fast_recall_dp, recall_dp (prediction red over ground truth green)
B  s21_inter_<city>.png    the stages inside `recall` for the median frame of the city: camera | BEV features | dense heat | dense direction | builder hypothesis tree with path validity
                           | junction head (split / merge) | assembled graph (node ids) | graph after the output stage simplify_dp (0.2 m)
"""
import json
import os
import pickle
import sys

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "loop20"))
import e255_matched as E  # noqa: E402
import evaluate as EV  # noqa: E402
import p16_data as PD  # noqa: E402
import p16_infer as PI  # noqa: E402
import sparse_metrics as SM  # noqa: E402
import viz_graph as VG  # noqa: E402
import matplotlib.colors as mc  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

FIG = f"{EV.R}/reports/source/figures"
NAMES = ("recall", "recall_dp", "fast_recall_dp")
G, RAW, TAU = {}, {}, {}
for n in NAMES:
    E.load(n)
    S = E.S
    G[n] = [PI.assemble(r, None, S["tau"], S["cfg"], **S["asm"]) for r in S["raw"]]
    RAW[n], TAU[n] = S["raw"], S["tau"]
    gts = S["gts"]
    stored = json.load(open(f"{EV.R}/runs/eval/{n}/val.json", encoding="utf-8-sig"))["metrics"]
    m = SM.card_any(G[n], gts, extra=True)
    assert abs(m["Q"] - stored["Q"]) <= 5e-4, f"{n}: stored card not reproduced"
    print(f"{n}: Q {m['Q']:.4f} vert100 {m['vert100']:.2f} (stored card reproduced)")
ctrl = pickle.load(open(f"{EV.R}/runs/e94_a/graphs_val_clean.pkl", "rb"))
assert [int(k) for k in ctrl["ks"]] == list(range(len(gts))), "control graphs are not in frame order"
G["control"] = ctrl["preds"]


def f1(g, gt):
    try:
        return float(SM.card_any([g], [gt])["F1@1.0"])
    except Exception:  # noqa: BLE001
        return float("nan")


def nv(g):
    return len({k for nd in g["nodes"] for k in nd}) if g.get("nodes") is not None else sum(len(p) for p in g["pts"])


F = np.array([f1(g, gt) for g, gt in zip(G["recall_dp"], gts)])
sp = json.load(open(f"{EV.R}/data/splits.json"))
rows = np.array(sp["val"])
pz = np.load(f"{EV.HOME}/exp_gcar_b123/poses_val.npz")
scene, loc, ts = pz["scene"][rows], pz["loc"][rows], pz["ts"][rows]
LOCN = {0: "Boston seaport", 1: "Singapore One-North", 2: "Singapore Holland Village", 3: "Singapore Queenstown"}
CITY = {"boston": [0], "singapore": [1, 2, 3]}
infos = pickle.load(open(f"{EV.HOME}/data/nuscenes_full/nuscenes_infos_val_crn.pkl", "rb"))
INFO_TS = np.array([i["timestamp"] for i in infos], dtype=np.float64) / 1e6


def info_of(i):
    j = int(np.argmin(np.abs(INFO_TS - ts[i])))
    assert abs(INFO_TS[j] - ts[i]) < 0.06
    return infos[j]


def quat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def project(info, pts):
    """ego-frame (lidar-frame) ground points (x right, y forward, on the road plane) -> pixels of CAM_FRONT; points behind the camera are dropped"""
    lc = info["lidar_infos"]["LIDAR_TOP"]["calibrated_sensor"]
    cc = info["cam_infos"]["CAM_FRONT"]["calibrated_sensor"]
    Rl, tl = quat(lc["rotation"]), np.array(lc["translation"])
    Rc, tc = quat(cc["rotation"]), np.array(cc["translation"])
    K = np.array(cc["camera_intrinsic"])
    p = np.c_[pts[:, 0], pts[:, 1], np.full(len(pts), -tl[2])]  # the road plane is z = 0 of the vehicle frame
    pe = p @ Rl.T + tl
    pc = (pe - tc) @ Rc  # R_c^T (p - t_c)
    ok = pc[:, 2] > 2.0
    uv = (pc @ K.T)
    uv = uv[:, :2] / np.maximum(uv[:, 2:3], 1e-6)
    return uv, ok


def dense(p, step=0.5):
    out = [p[0]]
    for a, b in zip(p[:-1], p[1:]):
        n = max(int(np.linalg.norm(b - a) / step), 1)
        out += [a + (b - a) * (k + 1) / n for k in range(n)]
    return np.array(out)


def overlay(ax, i, g, gt):
    info = info_of(i)
    ax.imshow(Image.open(f"{EV.HOME}/data/nuscenes_full/{info['cam_infos']['CAM_FRONT']['filename']}").convert("RGB"))
    for gg, col, lw, z in ((gt, "#1fd05a", 4.0, 2), (g, "#ff3b30", 1.8, 3)):
        for p in gg["pts"]:
            if len(p) < 2:
                continue
            uv, ok = project(info, dense(np.asarray(p, np.float64)))
            uv[~ok] = np.nan
            ax.plot(uv[:, 0], uv[:, 1], "-", color=col, lw=lw, alpha=0.75 if z == 2 else 0.95, zorder=z, solid_capstyle="round")
    ax.set_xlim(0, 1600)
    ax.set_ylim(900, 250)
    ax.axis("off")


def pick_city(city):
    ix = [i for i in range(len(F)) if int(loc[i]) in CITY[city] and not np.isnan(F[i])]
    ix.sort(key=lambda i: F[i])
    out, used = [], set()
    for q in (0.85, 0.50, 0.15):
        k = int(q * (len(ix) - 1))
        for d in range(len(ix)):
            c = [kk for kk in (k + d, k - d) if 0 <= kk < len(ix) and int(scene[ix[kk]]) not in used]
            if c:
                out.append(ix[c[0]])
                used.add(int(scene[ix[c[0]]]))
                break
    return out


# ---------------------------------------------------------------- figure A
meta = {}
PICK = {c: pick_city(c) for c in CITY}
for city, pick in PICK.items():
    fig, axs = plt.subplots(len(pick), 4, figsize=(11.5, 3.05 * len(pick)), gridspec_kw=dict(width_ratios=[1.7, 1, 1, 1]))
    for r, i in enumerate(pick):
        overlay(axs[r, 0], i, G["recall_dp"][i], gts[i])
        axs[r, 0].set_title(f"VAL frame {i}, scene {int(scene[i])}, {LOCN[int(loc[i])]}: camera, GT green, recall_dp red", fontsize=7)
        for c, (nm, lab) in enumerate((("control", "control E94 (trace network, 628 ms)"), ("fast_recall_dp", "fast_recall_dp"), ("recall_dp", "recall_dp"))):
            ax = axs[r, c + 1]
            VG.panel(ax, G[nm][i], gts[i], title=f"{lab}\nF1@1.0 {f1(G[nm][i], gts[i]):.2f}, {nv(G[nm][i])} vertices")
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(f"{FIG}/s21_frames_{city}.png", dpi=110)
    plt.close(fig)
    meta[city] = dict(frames=pick, f1_recall_dp=[float(F[i]) for i in pick], scenes=[int(scene[i]) for i in pick])
    print("saved", f"s21_frames_{city}.png", meta[city])

# ---------------------------------------------------------------- figure B
ds = PD.BuilderData(EV.SPLITS["val"][0], ks=None, train=False)
ds._open()


def dir_check(dirf, gt):
    cs = []
    for p in gt["pts"]:
        for u in range(1, len(p) - 1):
            t = p[u + 1] - p[u - 1]
            t = t / (np.linalg.norm(t) + 1e-9)
            ix, iy = int(np.clip((p[u, 0] + 50) / 0.5, 0, 199)), int(np.clip((p[u, 1] + 50) / 0.5, 0, 199))
            v = dirf[:, ix, iy]
            cs.append(float(v @ t / (np.linalg.norm(v) + 1e-9)))
    return float(np.mean(cs)) if cs else 0.0


for city, pick in PICK.items():
    i = pick[1]
    row = int(ds.rows[i])
    feat = np.asarray(ds.mm[0][row], np.float32)
    heat = np.asarray(ds.es.prob[i], np.float32)
    dirf = np.asarray(ds.es.dirf[i], np.float32)
    cs = dir_check(dirf, gts[i])
    print(f"{city} frame {i}: mean cos(direction field, GT tangent) {cs:+.2f}")
    ang = np.arctan2(dirf[1], dirf[0])
    hsv = np.ones((200, 200, 3), np.float32)
    hsv[..., 0] = (ang + np.pi) / (2 * np.pi)
    hsv[..., 2] = np.clip(heat / max(heat.max(), 1e-6), 0, 1) ** 0.6
    rgb = mc.hsv_to_rgb(hsv)
    r = RAW["recall"][i]
    pi_, par, pos = np.asarray(r["pi"]), np.asarray(r["parent"]), np.asarray(r["pos"])
    tau = TAU["recall"]
    junc = np.asarray(r["junc"], np.float32)
    fig, axs = plt.subplots(2, 4, figsize=(12.5, 7.0))
    ext = dict(origin="lower", extent=(-50, 50, -50, 50))
    overlay(axs[0, 0], i, G["recall_dp"][i], gts[i])
    axs[0, 0].set_title(f"camera, frame {i}, {LOCN[int(loc[i])]}", fontsize=9)
    axs[0, 1].imshow(np.abs(feat).mean(0).T, cmap="magma", **ext)
    axs[0, 1].set_title("1  BEV features (frozen), mean |value|", fontsize=9)
    axs[0, 2].imshow(heat.T, cmap="viridis", **ext)
    axs[0, 2].set_title("2  dense lane heat", fontsize=9)
    axs[0, 3].imshow(np.transpose(rgb, (1, 0, 2)), **ext)
    axs[0, 3].set_title("3  dense direction (hue = heading)", fontsize=9)
    ax = axs[1, 0]
    VG.draw_graph(ax, gts[i], color="#1a9850", lw=3.0, arrows=False, marks=False, alpha=0.35, zorder=1)
    for c in np.nonzero(par >= 0)[0]:
        a, b = pos[par[c]], pos[c]
        keep = pi_[c] >= tau
        ax.plot([a[0], b[0]], [a[1], b[1]], "-", color=plt.cm.plasma(float(pi_[c])) if keep else "#bbbbbb", lw=1.3 if keep else 0.5, zorder=3 if keep else 2)
    s0 = np.nonzero(par < 0)[0]
    ax.scatter(pos[s0, 0], pos[s0, 1], c=pi_[s0], cmap="plasma", vmin=0, vmax=1, s=9 + 20 * (pi_[s0] >= tau), zorder=4, edgecolors="none")
    ax.set_title("4  hypothesis tree, colour = path validity", fontsize=9)
    axs[1, 1].imshow(junc[0].T, cmap="Reds", vmin=0, vmax=0.8, **ext)
    axs[1, 1].imshow(np.ma.masked_less(junc[1].T, 0.05), cmap="Blues", alpha=0.75, vmin=0, vmax=0.8, **ext)
    VG.draw_graph(axs[1, 1], gts[i], color="#1a9850", lw=1.0, arrows=False, marks=True, alpha=0.9, zorder=3)
    axs[1, 1].set_title("5  junction heat: split red, merge blue", fontsize=9)
    for ax2, g, ttl in ((axs[1, 2], G["recall"][i], f"6  assembled graph (path union, anchoring, pruning):\n{nv(G['recall'][i])} vertices (node ids)"),
                        (axs[1, 3], G["recall_dp"][i], f"7  after the output stage simplify_dp 0.2 m:\n{nv(G['recall_dp'][i])} vertices")):
        VG.draw_graph(ax2, gts[i], color="#1a9850", lw=3.0, arrows=False, marks=False, alpha=0.4, zorder=1)
        VG.draw_graph(ax2, g, color="#d73027", lw=1.3, arrows=True, marks=True, zorder=3)
        pts = np.concatenate([p for p in g["pts"] if len(p)], 0)
        ax2.plot(pts[:, 0], pts[:, 1], ".", color="k", ms=3.4, zorder=5)
        ax2.set_title(ttl + f", F1 {f1(g, gts[i]):.2f}", fontsize=9)
    for ax2 in axs.ravel()[1:]:
        ax2.set_xlim(-50, 50)
        ax2.set_ylim(-50, 50)
        ax2.set_aspect("equal")
        ax2.set_xticks([-40, -20, 0, 20, 40])
        ax2.set_yticks([-40, -20, 0, 20, 40])
        ax2.tick_params(labelsize=7)
        ax2.plot([0], [0], "^", color="w" if ax2 in (axs[0, 1], axs[0, 2], axs[0, 3]) else "k", ms=5, zorder=9)
    fig.suptitle("system recall on one VAL frame (ego frame, +-50 m): 1 - 3 inputs, 4 - 5 builder outputs, 6 - 7 assembly", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{FIG}/s21_inter_{city}.png", dpi=110)
    plt.close(fig)
    print("saved", f"s21_inter_{city}.png", "| pi >= tau:", int((pi_ >= tau).sum()), "of", len(pi_))
    meta[city]["inter_frame"] = i
json.dump(meta, open(f"{FIG}/s21_frames.json", "w"))
