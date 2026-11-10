"""t5 lane probe v2 (experiments/t5_lane_probe/experiment.md section 4): fresh VAL frames, measures only at 2-30 m flat-ground
distance, recall@1 + corridor precision@1 + F1 pre-registered, 50 epochs, threshold by TRAIN F1, control gate (rgb > prior + 0.05).
Reuses temp/t5/lane_probe.py (projection, heads, feature extraction) and the v1 TRAIN features in temp/t5/feats/.
Stages (from the bev_fusion folder, CPU, 2 threads):
  python temp/t5/lane_probe_v2.py frames     300 VAL keyframes not used by v1 -> temp/t5/frames_v2.json
  python temp/t5/lane_probe_v2.py features   their features + distance maps -> temp/t5/feats_v2/; TRAIN distance maps
  python temp/t5/lane_probe_v2.py train ARM  ARM in ctx16 fpn16 res8 res4 rgb -> temp/t5/results_v2/ARM.json
  python temp/t5/lane_probe_v2.py eval       prior, gate, bootstrap, results_v2.md, picture
  python temp/t5/lane_probe_v2.py selftest
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lane_probe as v1                                                          # noqa: E402
from evaluation.metrics import SegMeter, dilate                                 # noqa: E402

OUT = v1.OUT
FEAT2 = os.path.join(OUT, "feats_v2")
RES2 = os.path.join(OUT, "results_v2")
NEAR = (2.0, 30.0)
EPOCHS = 50
GATE = 0.05
torch.set_num_threads(2)


# ------------------------------------------------------------------------------------------------ frames and distance maps
def choose_frames():
    cfg = json.load(open(os.path.join(v1.ROOT, "training", "configs", "t4_fpn_order.json")))
    meta = os.path.join(v1.DATA, "v1.0-trainval")
    scenes = {s["name"]: s for s in json.load(open(os.path.join(meta, "scene.json")))}
    samples = {s["token"]: s for s in json.load(open(os.path.join(meta, "sample.json")))}
    files = v1.gt_files()
    old = json.load(open(os.path.join(OUT, "frames.json")))
    used = {f[1] for f in old["val"]}
    per_scene = []
    for name in cfg["val_scenes"]:
        toks, t = [], scenes[name]["first_sample_token"]
        while t:
            toks.append(t)
            t = samples[t]["next"]
        inner = [x for x in toks[2:-2] if x not in used and x in files]
        idx = np.linspace(0, len(inner) - 1, 6).round().astype(int)[1:-1] if len(inner) >= 6 else range(len(inner))
        per_scene.append([[name, inner[i], files[inner[i]]] for i in sorted(set(idx))])
    val = []
    for k in range(4):                                     # round robin over scenes: every val scene is used before any repeats
        val += [p[k] for p in per_scene if k < len(p)]
    frames = {"train": old["train"], "val": val[:v1.N_VAL]}
    assert not used & {f[1] for f in frames["val"]}, "v2 VAL frames must be new"
    json.dump(frames, open(os.path.join(OUT, "frames_v2.json"), "w"), indent=0)
    print(len(frames["val"]), "fresh VAL frames from", len({f[0] for f in frames["val"]}), "scenes; overlap with v1: 0")


def dist_map(meta):
    """(64, 176) float32: flat-ground forward distance x (m) of each stride-4 cell centre; -1 above the horizon."""
    K, T = np.asarray(meta["K"], np.float64), np.asarray(meta["T_c2e"], np.float64)
    v, u = np.mgrid[0:v1.H4, 0:v1.W4]
    pix = np.stack([(u + 0.5) * v1.S, (v + 0.5) * v1.S, np.ones_like(u, dtype=np.float64)], -1).reshape(-1, 3)
    d = pix @ np.linalg.inv(v1.IDA).T @ np.linalg.inv(K).T @ T[:3, :3].T
    t = T[:3, 3]
    s = np.where(d[:, 2] < -1e-6, -t[2] / np.minimum(d[:, 2], -1e-6), -1.0)
    x = np.where(s > 0, t[0] + s * d[:, 0], -1.0)
    return x.reshape(v1.H4, v1.W4).astype(np.float32)


@torch.no_grad()
def extract():
    frames = json.load(open(os.path.join(OUT, "frames_v2.json")))
    os.makedirs(FEAT2, exist_ok=True)
    # TRAIN distance maps (geometry only)
    p = os.path.join(FEAT2, "train_dist.npy")
    if not os.path.exists(p):
        np.save(p, np.stack([dist_map(v1.load_meta(f[2])) for f in frames["train"]]).astype(np.float16))
        print("TRAIN distance maps", len(frames["train"]))
    # VAL features: v1's extractor writes into its FEAT folder and frames.json, so point it at the v2 ones temporarily
    v1.FEAT = FEAT2
    real = v1.OUT
    tmp = os.path.join(FEAT2, "_frames")
    os.makedirs(tmp, exist_ok=True)
    json.dump({"train": [], "val": frames["val"]}, open(os.path.join(tmp, "frames.json"), "w"))
    v1.OUT = tmp
    try:
        v1.extract()
    finally:
        v1.OUT = real
    p = os.path.join(FEAT2, "val_dist.npy")
    if not os.path.exists(p):
        np.save(p, np.stack([dist_map(v1.load_meta(f[2])) for f in frames["val"]]).astype(np.float16))
    print("VAL features and distance maps", len(frames["val"]))


# ------------------------------------------------------------------------------------------------ measures
def near_mask(dist):
    return (dist >= NEAR[0]) & (dist < NEAR[1])


def counts_v2(prob, masks, dists, scenes, thr):
    """per scene: SegMeter counts at tolerances 0 / 1 / 2 cells for 'all30' (every 2-30 m pixel) and 'corr30' (2-30 m pixels
    inside a 4-cell corridor around the labels)."""
    meters = {}
    for p, g, d, sc in zip(prob, masks, dists, scenes):
        m = meters.setdefault(sc, SegMeter(v1.GRID, tolerances=v1.TOL, bins=v1.ONE_BIN))
        g = g.astype(np.float32)
        pred = (p > thr).astype(np.float32)
        near = near_mask(d.astype(np.float32))
        m.add({"all30": pred}, {"all30": g}, mask=near)
        m.add({"corr30": pred}, {"corr30": g}, mask=near & dilate(g > 0, v1.CORRIDOR))
    return {sc: {n: m.c[(0.0, n, "all", "all")]["cnt"].tolist() for n in ("all30", "corr30")} for sc, m in meters.items()}


def rpf1(counts, tol=1):
    a = np.sum([np.asarray(c["all30"]) for c in counts.values()], 0)[tol]
    k = np.sum([np.asarray(c["corr30"]) for c in counts.values()], 0)[tol]
    r = a[2] / max(a[2] + a[3], 1e-9)
    p = k[0] / max(k[0] + k[1], 1e-9)
    return float(r), float(p), float(2 * p * r / max(p + r, 1e-9))


def bootstrap(ca, cb, n=1000, seed=0):
    sc = sorted(set(ca) & set(cb))
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        j = [sc[k] for k in rng.integers(0, len(sc), len(sc))]
        d.append(rpf1({f"{s}#{i}": cb[s] for i, s in enumerate(j)})[2] - rpf1({f"{s}#{i}": ca[s] for i, s in enumerate(j)})[2])
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def best_threshold(prob, masks, dists):
    best = (-1.0, 0.5)
    for thr in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        best = max(best, (rpf1(counts_v2(prob, masks, dists, ["train"] * len(prob), thr))[2], thr))
    return best[1], best[0]


# ------------------------------------------------------------------------------------------------ data
def data():
    fr = json.load(open(os.path.join(OUT, "frames_v2.json")))
    nt, nv = len(fr["train"]), len(fr["val"])
    tr = v1.open_arrays(len(fr["train"]) + v1.N_VAL, "r")             # v1 feats: rows 0..599 are the TRAIN frames
    va = {k: np.load(os.path.join(FEAT2, k + ".npy"), mmap_mode="r") for k in v1.feature_shapes(1)}
    return fr, nt, nv, tr, va, np.load(os.path.join(FEAT2, "train_dist.npy")), np.load(os.path.join(FEAT2, "val_dist.npy"))


def train(arm, epochs=EPOCHS, bs=8, seed=0):
    fr, nt, nv, tr, va, tr_dist, va_dist = data()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = v1.Probe(v1.ARMS[arm][0], rgb=(arm == "rgb"))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    pw = torch.tensor(10.0)
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        order, tot = rng.permutation(nt), 0.0
        for k in range(0, nt, bs):
            idx = order[k:k + bs]
            x = v1.batch_input(tr, arm, idx)
            y = torch.from_numpy(np.ascontiguousarray(tr["mask"][np.sort(idx)])).float()[:, None]
            lg = model(x)
            p = torch.sigmoid(lg)
            loss = F.binary_cross_entropy_with_logits(lg, y, pos_weight=pw) + 1 - (2 * (p * y).sum() + 1) / (p.sum() + y.sum() + 1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        sched.step()
        if ep % 5 == 4 or ep == epochs - 1:
            print(f"{arm} epoch {ep + 1}/{epochs} loss {tot / nt:.4f} ({time.time() - t0:.0f} s)", flush=True)
    tri = np.arange(nt)
    thr, tr_f1 = best_threshold(v1.predict(model, tr, arm, tri), tr["mask"][:nt], tr_dist)
    vai = np.arange(nv)
    prob = v1.predict(model, va, arm, vai)
    counts = counts_v2(prob, va["mask"], va_dist, [f[0] for f in fr["val"]], thr)
    os.makedirs(RES2, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(RES2, arm + ".pt"))
    np.save(os.path.join(RES2, arm + "_prob_val.npy"), prob[:12].astype(np.float16))
    res = {"arm": arm, "threshold": thr, "train_f1": tr_f1, "val_rpf1": rpf1(counts), "val_rpf1_tol0": rpf1(counts, 0),
           "counts": counts, "epochs": epochs, "seconds": time.time() - t0}
    json.dump(res, open(os.path.join(RES2, arm + ".json"), "w"))
    print(arm, "thr", thr, "train F1 %.3f" % tr_f1, "VAL recall / corridor precision / F1", [round(v, 3) for v in res["val_rpf1"]])


def evaluate():
    fr, nt, nv, tr, va, tr_dist, va_dist = data()
    res = {a: json.load(open(os.path.join(RES2, a + ".json"))) for a in v1.ARMS if os.path.exists(os.path.join(RES2, a + ".json"))}
    prior = tr["mask"][:nt].astype(np.float32).mean(0)
    prior = prior / max(prior.max(), 1e-9)
    thr, tr_f1 = best_threshold(np.repeat(prior[None], nt, 0), tr["mask"][:nt], tr_dist)
    pc = counts_v2(np.repeat(prior[None], nv, 0), va["mask"], va_dist, [f[0] for f in fr["val"]], thr)
    res["prior"] = {"arm": "prior", "threshold": thr, "train_f1": tr_f1, "val_rpf1": rpf1(pc), "val_rpf1_tol0": rpf1(pc, 0),
                    "counts": pc}
    g_lo, g_hi = bootstrap(res["prior"]["counts"], res["rgb"]["counts"])
    g_d = res["rgb"]["val_rpf1"][2] - res["prior"]["val_rpf1"][2]
    gate = g_d >= GATE and g_lo > 0
    near_cells = va["mask"].astype(bool) & near_mask(va_dist.astype(np.float32))
    lines = [f"# t5 lane probe v2 - results (VAL: {nv} fresh keyframes of {len({f[0] for f in fr['val']})} scenes; TRAIN: {nt} "
             f"keyframes; measures at {NEAR[0]:g}-{NEAR[1]:g} m flat-ground distance)", "",
             "Recall@1 of the labelled line cells, corridor precision@1 (inside 4 cells of a label), F1 of the two "
             "(evaluation/metrics.SegMeter counts, 1 cell = 4 x 4 input pixels); threshold = best TRAIN F1; 50 epochs; difference "
             "of F1 to `ctx16` with the paired scene bootstrap (1000 draws, 95 %).", "",
             "| arm | features | recall@1 | corridor precision@1 | F1 | F1 at 0 cells | d F1 vs ctx16 [95 %] | thr | TRAIN F1 |",
             "|---|---|---|---|---|---|---|---|---|"]
    desc = {"ctx16": "context 80 ch, stride 16 (lift input)", "fpn16": "neck 512 ch, stride 16", "res8": "ResNet layer 2, stride 8",
            "res4": "ResNet layer 1, stride 4", "rgb": "RGB CNN from scratch (control)", "prior": "per-pixel prior (null)"}
    for a in ["prior", "ctx16", "fpn16", "res8", "res4", "rgb"]:
        if a not in res:
            continue
        r, p, f1 = res[a]["val_rpf1"]
        d = "-"
        if a != "ctx16":
            lo, hi = bootstrap(res["ctx16"]["counts"], res[a]["counts"])
            d = f"{f1 - res['ctx16']['val_rpf1'][2]:+.3f} [{lo:+.3f}, {hi:+.3f}]"
        lines.append(f"| {a} | {desc[a]} | {r:.3f} | {p:.3f} | {f1:.3f} | {res[a]['val_rpf1_tol0'][2]:.3f} | {d} | "
                     f"{res[a]['threshold']} | {res[a]['train_f1']:.3f} |")
    lines += ["", f"Control gate (rgb F1 - prior F1 >= {GATE} with the interval excluding 0): {g_d:+.3f} [{g_lo:+.3f}, {g_hi:+.3f}] -> "
              f"{'PASSES' if gate else 'FAILS'}.",
              f"Labelled line cells at 2-30 m: {int(near_cells.sum())} in {nv} frames (mean {near_cells.sum() / nv:.0f} per frame); "
              f"frames without one: {int((near_cells.reshape(nv, -1).sum(1) == 0).sum())}."]
    open(os.path.join(OUT, "results_v2.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    picture(va, res)


def picture(va, res, shown=(0, 3, 6, 9)):
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    arms = [a for a in ["ctx16", "fpn16", "res8", "res4", "rgb"] if os.path.exists(os.path.join(RES2, a + "_prob_val.npy"))]
    probs = {a: np.load(os.path.join(RES2, a + "_prob_val.npy")).astype(np.float32) for a in arms}
    fig, ax = plt.subplots(len(shown), 1 + len(arms), figsize=(3.2 * (1 + len(arms)), 1.6 * len(shown)))
    for r, j in enumerate(shown):
        img = va["rgb"][j].transpose(1, 2, 0)
        m = cv2.resize(va["mask"][j], (v1.W, v1.H), interpolation=cv2.INTER_NEAREST)
        ax[r, 0].imshow(img)
        ax[r, 0].imshow(np.ma.masked_where(m == 0, m), cmap="autumn", alpha=0.8)
        for c, a in enumerate(arms):
            ax[r, 1 + c].imshow(probs[a][j], vmin=0, vmax=1, cmap="magma")
            if r == 0:
                ax[r, 1 + c].set_title(f"{a}: F1 {res[a]['val_rpf1'][2]:.2f}", fontsize=9)
        if r == 0:
            ax[r, 0].set_title("fresh val image + labels", fontsize=9)
    for a_ in ax.ravel():
        a_.axis("off")
    plt.tight_layout()
    p = os.path.join(v1.ROOT, "experiments", "t5_lane_probe", "images", "probe_v2_predictions.png")
    plt.savefig(p, dpi=110)
    print("picture", p)


def selftest():
    meta = v1.load_meta(next(iter(v1.gt_files().values())))
    d = dist_map(meta)
    uv, _ = v1.project(np.array([[10.0, 0.0, 0.0], [25.0, 0.0, 0.0]]), np.asarray(meta["K"]), np.asarray(meta["T_c2e"]))
    for (u, v), want in zip(uv, (10.0, 25.0)):
        got = d[int(v // v1.S), int(u // v1.S)]
        assert abs(got - want) < 0.15 * want, (got, want)
    assert (d[:4] < 0).all(), "the top rows are sky"
    g = np.zeros((v1.H4, v1.W4), np.float32)
    g[40, 50:80] = 1
    dd = np.full_like(g, 10.0)
    perfect = rpf1(counts_v2(g[None], g[None], dd[None], ["s"], 0.5))
    far = np.zeros_like(g)
    far[10, 100:130] = 1                                                                 # unlabelled paint outside the corridor
    extra = rpf1(counts_v2((g + far)[None], g[None], dd[None], ["s"], 0.5))
    shifted = np.roll(g, 1, 0)
    sh = rpf1(counts_v2(shifted[None], g[None], dd[None], ["s"], 0.5))
    assert perfect == (1.0, 1.0, 1.0) and extra == (1.0, 1.0, 1.0) and sh[2] == 1.0, (perfect, extra, sh)
    beyond = rpf1(counts_v2(g[None], g[None], np.full_like(g, 50.0)[None], ["s"], 0.5))
    assert beyond[0] == 0.0, "pixels beyond 30 m are not counted"
    print(f"lane_probe_v2 selftest ok: ground distance at 10 / 25 m = {d[int(uv[0, 1] // 4), int(uv[0, 0] // 4)]:.1f} / "
          f"{d[int(uv[1, 1] // 4), int(uv[1, 0] // 4)]:.1f} m, sky rows -1, perfect = 1, unlabelled paint outside the corridor "
          "not a false positive, 1-cell shift tolerated, pixels beyond 30 m excluded")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    {"frames": choose_frames, "features": extract, "eval": evaluate, "selftest": selftest,
     "train": lambda: train(sys.argv[2])}[stage]()
