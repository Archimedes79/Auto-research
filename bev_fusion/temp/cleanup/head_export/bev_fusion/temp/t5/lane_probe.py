"""t5 lane probe (experiments/t5_lane_probe/experiment.md): do the frozen image features hold the lane lines?

Image-plane probe: the exact divider polylines of the GT cache, projected onto the ground of the 704 x 256 front image and drawn on
the stride-4 grid (64 x 176); one identical small head per feature source (frozen DualBEV ResNet-50 stages / neck / context, an RGB CNN
control, a prior); IoU by evaluation/metrics.SegMeter, per val scene for a paired scene bootstrap.
Stages (from the bev_fusion folder, CPU, 2 threads):
  python temp/t5/lane_probe.py frames      choose 600 TRAIN + 300 VAL keyframes -> temp/t5/frames.json
  python temp/t5/lane_probe.py overlay     10 val images with the projected GT (validity check of the card)
  python temp/t5/lane_probe.py features    one frozen-encoder pass per frame -> temp/t5/feats/*.npy (resumable)
  python temp/t5/lane_probe.py train ARM   ARM in ctx16 fpn16 res8 res4 rgb -> temp/t5/results/ARM.json
  python temp/t5/lane_probe.py eval        prior baseline, bootstrap vs ctx16, results.md, picture
  python temp/t5/lane_probe.py selftest
"""
import json
import math
import os
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))   # the bev_fusion folder
sys.path.insert(0, os.path.join(ROOT, "training"))
sys.path.insert(0, ROOT)
from camera_encoder import IMG_MEAN, IMG_STD, FrontCameraEncoder, camera_vector, nuscenes_front_ida   # noqa: E402
from evaluation.metrics import SegMeter, dilate                                                      # noqa: E402

OUT = os.path.join(ROOT, "temp", "t5")
FEAT = os.path.join(OUT, "feats")
RES = os.path.join(OUT, "results")
DATA = os.path.expanduser("~/data/nuscenes_full")
H, W, S = 256, 704, 4                          # input image, probe stride
H4, W4 = H // S, W // S                        # 64 x 176
IDA = nuscenes_front_ida().numpy()
BANDS = ((2.0, 20.0), (20.0, 40.0), (40.0, 60.0))
TOL = (0.0, 1.0, 2.0)                          # cells of the stride-4 grid
ARMS = {"ctx16": (80, 16), "fpn16": (512, 16), "res8": (512, 8), "res4": (256, 4), "rgb": (3, 1)}
GRID = SimpleNamespace(x_min=0.0, res=1.0)     # SegMeter on the image grid: one cell = one stride-4 pixel
ONE_BIN = ((-1e9, 1e9),)
N_TRAIN, N_VAL = 600, 300
CORRIDOR = 4                                   # post hoc: cells around the labelled lines in which precision is counted
torch.set_num_threads(2)


# ------------------------------------------------------------------------------------------------ frames and GT
def gt_files():
    d = os.path.join(ROOT, "data", "cache", "gt")
    return {f.split("_")[0]: os.path.join(d, f) for f in os.listdir(d) if f.endswith(".npz")}


def choose_frames():
    cfg = json.load(open(os.path.join(ROOT, "training", "configs", "t4_fpn_order.json")))
    meta = os.path.join(DATA, "v1.0-trainval")
    scenes = {s["name"]: s for s in json.load(open(os.path.join(meta, "scene.json")))}
    samples = {s["token"]: s for s in json.load(open(os.path.join(meta, "sample.json")))}
    files = gt_files()

    def pick(names, per):
        out = []
        for name in names:
            toks, t = [], scenes[name]["first_sample_token"]
            while t:
                toks.append(t)
                t = samples[t]["next"]
            inner = toks[2:-2]
            idx = np.linspace(0, len(inner) - 1, per).round().astype(int) if len(inner) >= per else range(len(inner))
            out += [[name, inner[i], files[inner[i]]] for i in sorted(set(idx)) if inner[i] in files]
        return out
    tr = cfg["train_scenes"]
    tr = [tr[i] for i in np.linspace(0, len(tr) - 1, N_TRAIN // 4).round().astype(int)]
    frames = {"train": pick(tr, 4)[:N_TRAIN], "val": pick(cfg["val_scenes"], 4)[:N_VAL]}
    json.dump(frames, open(os.path.join(OUT, "frames.json"), "w"), indent=0)
    print({k: len(v) for k, v in frames.items()}, "frames;", len({f[0] for f in frames["val"]}), "val scenes")


def load_meta(path):
    return np.load(path, allow_pickle=True)["gt"].item()["cur"]["meta"]


def project(pts_ego, K, T_c2e):
    """(N, 3) ego points -> (N, 2) input-image pixels (after the image transform), camera depth (N,)."""
    T = np.linalg.inv(T_c2e)
    pc = np.c_[pts_ego, np.ones(len(pts_ego))] @ T.T
    z = pc[:, 2]
    uv = (pc[:, :3] / np.maximum(z, 1e-6)[:, None]) @ K.T
    uv = uv @ IDA.T
    return uv[:, :2], z


def densify(pl, step=0.25):
    out = [pl[0]]
    for a, b in zip(pl[:-1], pl[1:]):
        n = max(1, int(math.ceil(np.linalg.norm(b - a) / step)))
        out += [a + (b - a) * (k / n) for k in range(1, n + 1)]
    return np.asarray(out)


def divider_mask(meta, x_range=(2.0, 60.0)):
    """(64, 176) uint8: lane and road dividers on the ground (z = 0) projected and drawn 1 cell wide on the stride-4 grid."""
    m = np.zeros((H4, W4), np.uint8)
    K, T = np.asarray(meta["K"], np.float64), np.asarray(meta["T_c2e"], np.float64)
    for layer in ("lane_divider", "road_divider"):
        for pl in meta["vectors"].get(layer, []):
            p = densify(np.asarray(pl, np.float64))
            ok = (p[:, 0] >= x_range[0]) & (p[:, 0] <= x_range[1])
            uv, z = project(np.c_[p, np.zeros(len(p))], K, T)
            ok &= z > 0.5
            q = uv / S
            for i in range(len(p) - 1):
                if ok[i] and ok[i + 1]:
                    a, b = q[i], q[i + 1]
                    if max(abs(a[0]), abs(b[0])) < 4 * W4 and max(abs(a[1]), abs(b[1])) < 4 * H4:
                        cv2.line(m, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), 1, 1)
    return m


def band_map(meta):
    """(64, 176) int8: flat-ground distance band of each stride-4 cell centre (0, 1, 2; -1 above the horizon / outside)."""
    K, T = np.asarray(meta["K"], np.float64), np.asarray(meta["T_c2e"], np.float64)
    v, u = np.mgrid[0:H4, 0:W4]
    pix = np.stack([(u + 0.5) * S, (v + 0.5) * S, np.ones_like(u, dtype=np.float64)], -1).reshape(-1, 3)
    rays = pix @ np.linalg.inv(IDA).T @ np.linalg.inv(K).T            # camera frame directions
    d = rays @ T[:3, :3].T
    t = T[:3, 3]
    s = np.where(d[:, 2] < -1e-6, -t[2] / np.minimum(d[:, 2], -1e-6), -1.0)
    x = t[0] + s * d[:, 0]
    out = np.full(len(x), -1, np.int8)
    for b, (lo, hi) in enumerate(BANDS):
        out[(s > 0) & (x >= lo) & (x < hi)] = b
    return out.reshape(H4, W4)


def load_image(meta):
    path = meta["cam_file"] if os.path.isabs(meta["cam_file"]) else os.path.join(DATA, meta["cam_file"])
    img = Image.open(path).convert("RGB")
    s = float(IDA[0, 0])
    w, h = int(round(img.width * s)), int(round(img.height * s))
    return np.asarray(img.resize((w, h), Image.BILINEAR).crop((0, h - H, W, h)))       # (256, 704, 3) uint8


# ------------------------------------------------------------------------------------------------ features
def feature_shapes(n):
    return {"ctx16": (n, 80, H // 16, W // 16), "fpn16": (n, 512, H // 16, W // 16), "res8": (n, 512, H // 8, W // 8),
            "res4": (n, 256, H // 4, W // 4), "rgb": (n, 3, H, W), "mask": (n, H4, W4), "band": (n, H4, W4)}


def open_arrays(n, mode):
    dt = {"rgb": np.uint8, "mask": np.uint8, "band": np.int8}
    out = {}
    for k, shp in feature_shapes(n).items():
        p = os.path.join(FEAT, k + ".npy")
        if mode == "w+" and not os.path.exists(p):
            out[k] = np.lib.format.open_memmap(p, "w+", dt.get(k, np.float16), shp)
        else:
            out[k] = np.load(p, mmap_mode="r+" if mode == "w+" else "r")
    return out


@torch.no_grad()
def extract():
    frames = json.load(open(os.path.join(OUT, "frames.json")))
    allf = frames["train"] + frames["val"]
    os.makedirs(FEAT, exist_ok=True)
    arr = open_arrays(len(allf), "w+")
    prog = os.path.join(FEAT, "done.txt")
    start = int(open(prog).read()) if os.path.exists(prog) else 0
    enc = FrontCameraEncoder()
    rep = enc.load_pretrained(os.path.join(ROOT, "models", "pretrained", "dualbev-r50-cbgs.pth"))
    assert not rep["missing"] and not rep["shape_mismatch"], rep
    enc.eval()
    b = enc.img_backbone
    mean, std = torch.tensor(IMG_MEAN).view(3, 1, 1), torch.tensor(IMG_STD).view(3, 1, 1)
    ida = torch.tensor(IDA, dtype=torch.float32)
    for i in range(start, len(allf)):
        meta = load_meta(allf[i][2])
        rgb = load_image(meta)
        x = ((torch.tensor(rgb, dtype=torch.float32).permute(2, 0, 1) - mean) / std)[None]
        K, T = torch.tensor(meta["K"], dtype=torch.float32), torch.tensor(meta["T_c2e"], dtype=torch.float32)
        cv = camera_vector(K[None], ida[None, :2, :2], ida[None, :2, 2], T[None])
        y = b.maxpool(b.relu(b.bn1(b.conv1(x))))
        l1 = b.layer1(y)
        l2 = b.layer2(l1)
        l3 = b.layer3(l2)
        fpn = enc.img_neck(l3, b.layer4(l3))
        _, ctx = enc.depth_net(fpn, cv)
        for k, t in (("ctx16", ctx), ("fpn16", fpn), ("res8", l2), ("res4", l1)):
            arr[k][i] = t[0].numpy().astype(np.float16)
        arr["rgb"][i] = rgb.transpose(2, 0, 1)
        arr["mask"][i] = divider_mask(meta)
        arr["band"][i] = band_map(meta)
        if i % 20 == 19 or i == len(allf) - 1:
            for a in arr.values():
                a.flush()
            open(prog, "w").write(str(i + 1))
            print(f"features {i + 1} / {len(allf)}", flush=True)


# ------------------------------------------------------------------------------------------------ probe heads
class Probe(nn.Module):
    """The same head for every arm: 1 x 1 to 32, 3 x 3, 1 x 1 to one logit, bilinear up-sampling to stride 4. The rgb arm has a
    3-layer CNN stem (stride 4) in front, trained from scratch."""

    def __init__(self, cin, rgb=False):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(True),
                                  nn.Conv2d(32, 64, 3, 2, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
                                  nn.Conv2d(64, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True)) if rgb else None
        c = 64 if rgb else cin
        self.net = nn.Sequential(nn.Conv2d(c, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(True),
                                 nn.Conv2d(32, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(True), nn.Conv2d(32, 1, 1))

    def forward(self, x):
        if self.stem is not None:
            x = self.stem(x)
        y = self.net(x)
        return y if y.shape[-2:] == (H4, W4) else F.interpolate(y, (H4, W4), mode="bilinear", align_corners=False)


def batch_input(arr, arm, idx):
    x = torch.from_numpy(np.ascontiguousarray(arr[arm][np.sort(idx)])).float()
    if arm == "rgb":
        x = (x - torch.tensor(IMG_MEAN).view(1, 3, 1, 1)) / torch.tensor(IMG_STD).view(1, 3, 1, 1)
    return x


def predict(model, arr, arm, idx, bs=8):
    model.eval()
    out = []
    with torch.no_grad():
        for k in range(0, len(idx), bs):
            out.append(torch.sigmoid(model(batch_input(arr, arm, idx[k:k + bs]))[:, 0]).numpy())
    return np.concatenate(out)


def scene_counts(prob, arr, idx, scenes, thr):
    """per val scene: SegMeter counts (tolerances x [tp_pred, fp, tp_gt, fn]) for 'all' and each distance band."""
    meters = {}
    for p, i, sc in zip(prob, idx, scenes):
        m = meters.setdefault(sc, SegMeter(GRID, tolerances=TOL, bins=ONE_BIN))
        g, band = arr["mask"][i].astype(np.float32), arr["band"][i]
        pred = (p > thr).astype(np.float32)
        m.add({"all": pred}, {"all": g})
        for b in range(len(BANDS)):
            m.add({f"band{b}": pred}, {f"band{b}": g}, mask=band == b)
        # post hoc (labels are incomplete: visible paint is often unlabelled): precision only inside a corridor of CORRIDOR
        # cells around the labelled lines, so detected but unlabelled paint far from them does not count as a false positive
        m.add({"corr": pred}, {"corr": g}, mask=dilate(g > 0, CORRIDOR))
    out = {}
    for sc, m in meters.items():
        out[sc] = {name: m.c[(0.0, name, "all", "all")]["cnt"].tolist()
                   for name in ["all", "corr"] + [f"band{b}" for b in range(3)] if (0.0, name, "all", "all") in m.c}
    return out


def recall_precision_f1(counts, tol=1):
    """post hoc: recall of the labelled line cells (all pixels) and corridor precision, at tolerance tol cells -> (r, p, f1)."""
    a = np.sum([np.asarray(c["all"]) for c in counts.values()], 0)[tol]
    k = np.sum([np.asarray(c["corr"]) for c in counts.values()], 0)[tol]
    r = a[2] / max(a[2] + a[3], 1e-9)
    p = k[0] / max(k[0] + k[1], 1e-9)
    return float(r), float(p), float(2 * p * r / max(p + r, 1e-9))


def iou_from(counts, names=("all", "band0", "band1", "band2")):
    tot = {n: np.sum([np.asarray(c[n]) for c in counts.values()], 0) for n in names}
    res = {}
    for n, a in tot.items():
        tp = (a[:, 0] + a[:, 2]) / 2
        res[n] = [float(x) for x in tp / np.maximum(tp + a[:, 1] + a[:, 3], 1e-9)]
    return res


def choose_threshold(prob, arr, idx):
    best = (-1.0, 0.5)
    for thr in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        iou = iou_from(scene_counts(prob, arr, idx, ["train"] * len(idx), thr), ("all",))["all"][1]
        best = max(best, (iou, thr))
    return best[1], best[0]


def train(arm, epochs=15, bs=8, seed=0):
    frames = json.load(open(os.path.join(OUT, "frames.json")))
    nt, nv = len(frames["train"]), len(frames["val"])
    arr = open_arrays(nt + nv, "r")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    cin = ARMS[arm][0]
    model = Probe(cin, rgb=(arm == "rgb"))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    pw = torch.tensor(10.0)
    import time
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        order, tot = rng.permutation(nt), 0.0
        for k in range(0, nt, bs):
            idx = order[k:k + bs]
            x = batch_input(arr, arm, idx)
            y = torch.from_numpy(np.ascontiguousarray(arr["mask"][np.sort(idx)])).float()[:, None]
            lg = model(x)
            p = torch.sigmoid(lg)
            dice = 1 - (2 * (p * y).sum() + 1) / (p.sum() + y.sum() + 1)
            loss = F.binary_cross_entropy_with_logits(lg, y, pos_weight=pw) + dice
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        print(f"{arm} epoch {ep + 1}/{epochs} loss {tot / nt:.4f} ({time.time() - t0:.0f} s)", flush=True)
    tr_idx = np.arange(min(nt, 200))
    thr, tr_iou = choose_threshold(predict(model, arr, arm, tr_idx), arr, tr_idx)
    va_idx = np.arange(nt, nt + nv)
    prob = predict(model, arr, arm, va_idx)
    counts = scene_counts(prob, arr, va_idx, [f[0] for f in frames["val"]], thr)
    os.makedirs(RES, exist_ok=True)
    np.save(os.path.join(RES, arm + "_prob_val.npy"), prob[:12].astype(np.float16))     # for the picture only
    torch.save(model.state_dict(), os.path.join(RES, arm + ".pt"))
    res = {"arm": arm, "threshold": thr, "train_iou1": tr_iou, "val_iou": iou_from(counts), "counts": counts,
           "val_rpf1": recall_precision_f1(counts),
           "epochs": epochs, "seconds": time.time() - t0, "params": sum(q.numel() for q in model.parameters())}
    json.dump(res, open(os.path.join(RES, arm + ".json"), "w"))
    print(arm, "threshold", thr, "val IoU@0/1/2", [round(v, 3) for v in res["val_iou"]["all"]])


# ------------------------------------------------------------------------------------------------ prior, bootstrap, report
def prior_result(arr, nt, nv, frames):
    prior = arr["mask"][:nt].astype(np.float32).mean(0)
    tr_idx = np.arange(min(nt, 200))
    thr, tr_iou = choose_threshold(np.repeat(prior[None], len(tr_idx), 0) / max(prior.max(), 1e-9), arr, tr_idx)
    va_idx = np.arange(nt, nt + nv)
    prob = np.repeat(prior[None], nv, 0) / max(prior.max(), 1e-9)
    counts = scene_counts(prob, arr, va_idx, [f[0] for f in frames["val"]], thr)
    return {"arm": "prior", "threshold": thr, "train_iou1": tr_iou, "val_iou": iou_from(counts), "counts": counts,
            "val_rpf1": recall_precision_f1(counts)}


def bootstrap_f1(ca, cb, n=1000, seed=0):
    """paired scene bootstrap of the post-hoc F1 (recall + corridor precision at 1 cell): b - a, 95 % interval."""
    sc = sorted(set(ca) & set(cb))
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        j = [sc[k] for k in rng.integers(0, len(sc), len(sc))]
        pick = lambda c: {f"{s}#{i}": c[s] for i, s in enumerate(j)}
        d.append(recall_precision_f1(pick(cb))[2] - recall_precision_f1(pick(ca))[2])
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def bootstrap(ca, cb, n=1000, seed=0, tol=1):
    """paired scene bootstrap of IoU@tol(all): b - a, 95 % interval."""
    sc = sorted(set(ca) & set(cb))
    rng = np.random.default_rng(seed)
    A = np.asarray([ca[s]["all"][tol] for s in sc])
    B = np.asarray([cb[s]["all"][tol] for s in sc])

    def iou(a):
        tp = (a[:, 0] + a[:, 2]) / 2
        return tp.sum() / max((tp + a[:, 1] + a[:, 3]).sum(), 1e-9)
    d = []
    for _ in range(n):
        j = rng.integers(0, len(sc), len(sc))
        d.append(iou(B[j]) - iou(A[j]))
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def evaluate():
    frames = json.load(open(os.path.join(OUT, "frames.json")))
    nt, nv = len(frames["train"]), len(frames["val"])
    arr = open_arrays(nt + nv, "r")
    res = {a: json.load(open(os.path.join(RES, a + ".json"))) for a in ARMS if os.path.exists(os.path.join(RES, a + ".json"))}
    res["prior"] = prior_result(arr, nt, nv, frames)
    lines = [f"# t5 lane probe - results (VAL: {nv} keyframes of {len({f[0] for f in frames['val']})} scenes; TRAIN: {nt} keyframes)",
             "", "Divider IoU on the stride-4 image grid (1 cell = 4 x 4 input pixels) at tolerance 0 / 1 / 2 cells "
             "(evaluation/metrics.SegMeter); bands = flat-ground distance of the pixel; threshold chosen on TRAIN frames; difference to "
             "`ctx16` with the paired scene bootstrap (1000 draws, 95 % interval) on IoU@1.", "",
             "| arm | features | IoU@0 | IoU@1 | IoU@2 | IoU@1 2-20 m | 20-40 m | 40-60 m | d IoU@1 vs ctx16 [95 %] | thr | train IoU@1 |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    desc = {"ctx16": "context 80 ch, stride 16 (lift input)", "fpn16": "neck 512 ch, stride 16", "res8": "ResNet layer 2, stride 8",
            "res4": "ResNet layer 1, stride 4", "rgb": "RGB CNN from scratch (control)", "prior": "per-pixel prior (null)"}
    for a in ["prior", "ctx16", "fpn16", "res8", "res4", "rgb"]:
        if a not in res:
            continue
        r = res[a]
        v = r["val_iou"]
        d = "-"
        if a != "ctx16" and "ctx16" in res:
            lo, hi = bootstrap(res["ctx16"]["counts"], r["counts"])
            diff = v["all"][1] - res["ctx16"]["val_iou"]["all"][1]
            d = f"{diff:+.3f} [{lo:+.3f}, {hi:+.3f}]"
        lines.append(f"| {a} | {desc[a]} | {v['all'][0]:.3f} | {v['all'][1]:.3f} | {v['all'][2]:.3f} | {v['band0'][1]:.3f} | "
                     f"{v['band1'][1]:.3f} | {v['band2'][1]:.3f} | {d} | {r['threshold']} | {r['train_iou1']:.3f} |")
    lines += ["", "Post hoc (added after the run, because the labels miss much visible paint; NOT pre-registered): recall of the "
              f"labelled line cells at 1 cell over all pixels, precision at 1 cell only inside a corridor of {CORRIDOR} cells around "
              "the labelled lines (unlabelled paint further away is not counted), their F1; same threshold, heads, frames; "
              "difference of F1 to `ctx16` with the paired scene bootstrap.", "",
              "| arm | recall@1 | corridor precision@1 | F1 | d F1 vs ctx16 [95 %] |", "|---|---|---|---|---|"]
    for a in ["prior", "ctx16", "fpn16", "res8", "res4", "rgb"]:
        if a not in res or "val_rpf1" not in res[a]:
            continue
        r, p, f1 = res[a]["val_rpf1"]
        d = "-"
        if a != "ctx16" and "ctx16" in res:
            lo, hi = bootstrap_f1(res["ctx16"]["counts"], res[a]["counts"])
            d = f"{f1 - res['ctx16']['val_rpf1'][2]:+.3f} [{lo:+.3f}, {hi:+.3f}]"
        lines.append(f"| {a} | {r:.3f} | {p:.3f} | {f1:.3f} | {d} |")
    gt = arr["mask"][nt:nt + nv]
    lines += ["", f"GT divider cells per val frame: mean {gt.reshape(nv, -1).sum(1).mean():.0f} of {H4 * W4}; frames without a "
              f"divider: {int((gt.reshape(nv, -1).sum(1) == 0).sum())}."]
    open(os.path.join(OUT, "results.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    picture(arr, nt, res)


def picture(arr, nt, res, frames_shown=(0, 4, 8)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    arms = [a for a in ["ctx16", "fpn16", "res8", "res4", "rgb"] if os.path.exists(os.path.join(RES, a + "_prob_val.npy"))]
    probs = {a: np.load(os.path.join(RES, a + "_prob_val.npy")).astype(np.float32) for a in arms}
    fig, ax = plt.subplots(len(frames_shown), 2 + len(arms), figsize=(3.2 * (2 + len(arms)), 1.6 * len(frames_shown)))
    for r, j in enumerate(frames_shown):
        i = nt + j
        ax[r, 0].imshow(arr["rgb"][i].transpose(1, 2, 0))
        ax[r, 1].imshow(arr["rgb"][i].transpose(1, 2, 0))
        m = cv2.resize(arr["mask"][i], (W, H), interpolation=cv2.INTER_NEAREST)
        ax[r, 1].imshow(np.ma.masked_where(m == 0, m), cmap="autumn", alpha=0.9)
        for c, a in enumerate(arms):
            ax[r, 2 + c].imshow(probs[a][j], vmin=0, vmax=1, cmap="magma")
            if r == 0:
                ax[r, 2 + c].set_title(f"{a}: IoU@1 {res[a]['val_iou']['all'][1]:.2f}", fontsize=9)
        if r == 0:
            ax[r, 0].set_title("front image (val)", fontsize=9)
            ax[r, 1].set_title("GT dividers projected", fontsize=9)
    for a_ in ax.ravel():
        a_.axis("off")
    plt.tight_layout()
    os.makedirs(os.path.join(ROOT, "experiments", "t5_lane_probe", "images"), exist_ok=True)
    p = os.path.join(ROOT, "experiments", "t5_lane_probe", "images", "probe_predictions.png")
    plt.savefig(p, dpi=110)
    print("picture", p)


def overlay(n=10):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    frames = json.load(open(os.path.join(OUT, "frames.json")))["val"]
    pick = [frames[i] for i in np.linspace(0, len(frames) - 1, n).round().astype(int)]
    fig, ax = plt.subplots(n // 2, 2, figsize=(14, 2.2 * n // 2))
    for a_, (sc, tok, f) in zip(ax.ravel(), pick):
        meta = load_meta(f)
        img = load_image(meta)
        m = cv2.resize(divider_mask(meta), (W, H), interpolation=cv2.INTER_NEAREST)
        a_.imshow(img)
        a_.imshow(np.ma.masked_where(m == 0, m), cmap="autumn", alpha=0.6)
        a_.set_title(f"{sc} {tok[:8]}", fontsize=8)
        a_.axis("off")
    plt.tight_layout()
    p = os.path.join(OUT, "gt_overlay.png")
    plt.savefig(p, dpi=90)
    print("overlay", p)


# ------------------------------------------------------------------------------------------------ self-test
def selftest():
    files = gt_files()
    meta = load_meta(next(iter(files.values())))
    K, T = np.asarray(meta["K"], np.float64), np.asarray(meta["T_c2e"], np.float64)
    # a ground point 20 m ahead projects into the image below the horizon, and the band map's ray-ground intersection returns it
    uv, z = project(np.array([[20.0, 0.0, 0.0]]), K, T)
    assert 0 <= uv[0, 0] < W and 0 <= uv[0, 1] < H and z[0] > 0, uv
    u4, v4 = int(uv[0, 0] // S), int(uv[0, 1] // S)
    assert band_map(meta)[v4, u4] == 1, band_map(meta)[v4, u4]          # 20 m lies in band 20-40 m (edge: cell centre)
    uv10, _ = project(np.array([[10.0, 0.0, 0.0]]), K, T)
    assert uv10[0, 1] > uv[0, 1], "nearer ground is lower in the image"
    # a straight divider 2-60 m ahead draws a connected run of cells
    m = divider_mask({"K": K, "T_c2e": T, "vectors": {"lane_divider": [np.array([[0.0, 1.8], [70.0, 1.8]])]}})
    rows = np.where(m.any(1))[0]
    assert m.sum() > 20 and (np.diff(rows) == 1).all(), (m.sum(), rows)
    # probe heads give stride-4 logits for every arm
    for a, (c, s) in ARMS.items():
        x = torch.zeros(2, c, H // s, W // s)
        assert Probe(c, rgb=(a == "rgb"))(x).shape == (2, 1, H4, W4), a
    # metric: a perfect prediction gives IoU 1, an empty one 0
    arr = {"mask": m[None], "band": band_map(meta)[None]}
    c1 = scene_counts(m[None].astype(np.float32), arr, [0], ["s"], 0.5)
    c0 = scene_counts(np.zeros_like(m, np.float32)[None], arr, [0], ["s"], 0.5)
    assert abs(iou_from(c1)["all"][0] - 1) < 1e-9 and iou_from(c0)["all"][0] == 0.0
    print(f"lane_probe selftest ok: ground point 20 m ahead at pixel ({uv[0, 0]:.0f}, {uv[0, 1]:.0f}), band 20-40 m on the way back, "
          f"straight divider -> {int(m.sum())} connected cells over {len(rows)} rows, head shapes, metric 1 / 0")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    stage = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    {"frames": choose_frames, "overlay": overlay, "features": extract, "eval": evaluate, "selftest": selftest,
     "train": lambda: train(sys.argv[2])}[stage]()
