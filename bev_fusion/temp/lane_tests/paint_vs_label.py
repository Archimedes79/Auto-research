"""scratch (diagnostic, no verdict): do the ground-truth lane lines lie on the paint in the front camera image?

For frames of the validation scenes the stored map lines (lane_divider, road_divider, GT signature of t20) are laid on the ground (ego z = 0) and projected into the original
front image (K, T_c2e of the frame). At every line point (every 0.5 m, 5-30 m ahead) the image brightness is sampled ACROSS the line on the ground, +-1 m in 4 cm steps
(a rectified profile: a paint stripe is the same width in metres at every range). Reading:
  * mean profile: a registered line gives a narrow bump at offset 0 (the paint is ~0.15 m wide); a bump away from 0 = a systematic shift, a wide or no bump = the labels are
    not on the paint (registration error of the map, or no paint there)
  * paint hit rate: share of line points with a bright stripe (contrast >= 0.25 over the asphalt next to it) within +-0.2 m / +-0.5 m of the label, against the same on the
    control lines (the label moved 1.75 m sideways, i.e. into the middle of the lane or onto the next marking)
  * per-frame best offset: the lateral shift that maximises the frame's mean brightness bump; its spread over frames = how far the labels are off the paint frame by frame,
    against the spread on the control lines (= what a random answer looks like)
Not corrected: vehicles in front hide paint, the ground is taken flat, the camera time differs from the annotation time by up to ~40 ms.
python temp/lane_tests/paint_vs_label.py [--n-per-scene 3] [--seed 0] [--split val] [--out temp/lane_tests/paint_out] [--max-frames 0]"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
SIG = "6a7aa7864a"
LAYERS = ("lane_divider", "road_divider")
RANGES = ((5.0, 12.0), (12.0, 20.0), (20.0, 30.0))
D = np.linspace(-1.0, 1.0, 51)                       # lateral offsets across the line, metres
NULL_SHIFT = 1.75                                    # control lines: the label moved sideways by this (middle of a lane)
CONTRAST_HIT = 0.25                                  # a stripe is 25 % brighter than the asphalt next to it


def densify(pl, step=0.5):
    pl = np.asarray(pl, float)
    if len(pl) < 2:
        return np.zeros((0, 2)), np.zeros((0, 2))
    pts, tans = [], []
    for a, b in zip(pl[:-1], pl[1:]):
        L = float(np.hypot(*(b - a)))
        if L < 1e-6:
            continue
        n = max(int(np.ceil(L / step)), 1)
        t = (np.arange(n) + 0.5) / n
        pts.append(a[None] + t[:, None] * (b - a)[None])
        tans.append(np.repeat(((b - a) / L)[None], n, 0))
    if not pts:
        return np.zeros((0, 2)), np.zeros((0, 2))
    return np.concatenate(pts), np.concatenate(tans)


def sample_gray(img, K, T_c2e, xy):
    """gray values of the image at ground points (ego x, y, z=0): (N,) with NaN where the point is behind the camera or outside the image"""
    from scipy.ndimage import map_coordinates
    P = np.c_[xy, np.zeros(len(xy)), np.ones(len(xy))] @ np.linalg.inv(T_c2e).T
    z = P[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u, v = K[0, 0] * P[:, 0] / z + K[0, 2], K[1, 1] * P[:, 1] / z + K[1, 2]
    ok = (z > 1.0) & (u > 2) & (u < img.shape[1] - 3) & (v > 2) & (v < img.shape[0] - 3)
    out = np.full(len(xy), np.nan, np.float32)
    if ok.any():
        out[ok] = map_coordinates(img, [v[ok], u[ok]], order=1, mode="nearest")
    return out


def profiles(img, meta, pts, tans, shift=0.0):
    """(N, len(D)) brightness profiles across the line points (moved sideways by `shift` m first)"""
    K, T = np.asarray(meta["K"], float), np.asarray(meta["T_c2e"], float)
    nrm = np.c_[-tans[:, 1], tans[:, 0]]
    flat = (pts[:, None, :] + (D[None, :, None] + shift) * nrm[:, None, :]).reshape(-1, 2)
    return sample_gray(img, K, T, flat).reshape(len(pts), len(D))


def contrast_hits(prof):
    """per profile: contrast of the brightest smoothed stripe within +-0.2 m and within +-0.5 m over the asphalt (median of the outer 0.5-1.0 m)"""
    ker = np.ones(3) / 3.0                                              # ~ a 12 cm box: the paint width
    sm = np.array([np.convolve(p, ker, mode="same") for p in prof])
    bg = np.nanmedian(np.where((np.abs(D) >= 0.5)[None, :], sm, np.nan), axis=1)
    out = {}
    for name, w in (("hit02", 0.2), ("hit05", 0.5)):
        pk = np.nanmax(np.where((np.abs(D) <= w)[None, :], sm, np.nan), axis=1)
        out[name] = (pk - bg) / (bg + 8.0) >= CONTRAST_HIT
    return out


def bump(prof):
    """brightness bump profile (gray levels over the row's median), (N, len(D))"""
    return prof - np.nanmedian(prof, axis=1, keepdims=True)


def best_offset(mean_bump):
    ker = np.exp(-0.5 * (np.arange(-4, 5) * 0.04 / 0.05) ** 2)
    ker /= ker.sum()
    sm = np.convolve(np.nan_to_num(mean_bump), ker, mode="same")
    i = int(np.argmax(sm))
    return float(D[i]), float(sm[i]), float(sm[len(D) // 2])


def fwhm(mean_bump):
    ker = np.exp(-0.5 * (np.arange(-4, 5) * 0.04 / 0.05) ** 2)
    ker /= ker.sum()
    sm = np.convolve(np.nan_to_num(mean_bump), ker, mode="same")
    base = float(np.median(sm))
    i = int(np.argmax(sm))
    half = base + 0.5 * (sm[i] - base)
    lo = i
    while lo > 0 and sm[lo] > half:
        lo -= 1
    hi = i
    while hi < len(sm) - 1 and sm[hi] > half:
        hi += 1
    return float(D[hi] - D[lo]), float(D[i]), float(sm[i] - base)


def choose_frames(split, n_per_scene, seed, max_frames):
    root = os.path.expanduser("~/data/nuscenes_full")
    from nuscenes.utils.splits import create_splits_scenes
    scenes = {s["token"]: s["name"] for s in json.load(open(os.path.join(root, "v1.0-trainval", "scene.json")))}
    samples = {s["token"]: s["scene_token"] for s in json.load(open(os.path.join(root, "v1.0-trainval", "sample.json")))}
    names = set(create_splits_scenes()[split])
    files = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "cache", "gt", f"*_{SIG}.npz"))):
        tok = os.path.basename(f).split("_")[0]
        sc = scenes.get(samples.get(tok, ""), "")
        if sc in names:
            files.setdefault(sc, []).append(f)
    rng = np.random.RandomState(seed)
    out = []
    for sc in sorted(files):
        fl = files[sc]
        pick = np.unique(np.linspace(0, len(fl) - 1, n_per_scene).round().astype(int))
        out += [(sc, fl[i]) for i in pick]
    rng.shuffle(out)
    return out[:max_frames] if max_frames else out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n-per-scene", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="val", choices=["val", "train"])
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "temp", "lane_tests", "paint_out"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from PIL import Image
    froot = os.path.expanduser("~/data/nuscenes_full")
    frames = choose_frames(a.split, a.n_per_scene, a.seed, a.max_frames)
    print(f"{len(frames)} frames of {len({s for s, _ in frames})} {a.split} scenes (GT {SIG})", flush=True)
    # accumulators: layer -> range bin -> {cond -> list of bump arrays / hit flags}
    acc = {(l, r): {"bump": [], "bump_null": [], "hit02": [], "hit05": [], "n02": [], "n05": []} for l in LAYERS for r in RANGES}
    per_frame = {l: [] for l in LAYERS}
    for k, (sc, f) in enumerate(frames):
        meta = np.load(f, allow_pickle=True)["gt"].item()["cur"]["meta"]
        im = np.asarray(Image.open(os.path.join(froot, meta["cam_file"])).convert("L"), np.float32)
        for layer in LAYERS:
            P, T = [], []
            for pl in meta["vectors"].get(layer, []):
                p, t = densify(pl)
                if len(p):
                    P.append(p)
                    T.append(t)
            if not P:
                continue
            P, T = np.concatenate(P), np.concatenate(T)
            keep = (P[:, 0] >= 5.0) & (P[:, 0] <= 30.0) & (np.abs(P[:, 1]) < 14.0)
            P, T = P[keep], T[keep]
            if len(P) == 0:
                continue
            real = profiles(im, meta, P, T, 0.0)
            nulls = [profiles(im, meta, P, T, s) for s in (NULL_SHIFT, -NULL_SHIFT)]
            okr = ~np.isnan(real).any(1)
            if okr.sum() < 4:
                continue
            for (lo, hi) in RANGES:
                sel = okr & (P[:, 0] >= lo) & (P[:, 0] < hi)
                if sel.sum() == 0:
                    continue
                A = acc[(layer, (lo, hi))]
                A["bump"].append(bump(real[sel]))
                nb = [bump(n[sel & ~np.isnan(n).any(1)]) for n in nulls]
                A["bump_null"].append(np.concatenate([x for x in nb if len(x)]) if any(len(x) for x in nb) else np.zeros((0, len(D))))
                h = contrast_hits(real[sel])
                A["hit02"].append(h["hit02"])
                A["hit05"].append(h["hit05"])
                for n in nulls:
                    ns = sel & ~np.isnan(n).any(1)
                    if ns.sum():
                        hn = contrast_hits(n[ns])
                        A["n02"].append(hn["hit02"])
                        A["n05"].append(hn["hit05"])
            # per-frame best offset (frame mean bump), real line and control lines
            if okr.sum() >= 8:
                d_real, pk_real, at0 = best_offset(np.nanmean(bump(real[okr]), axis=0))
                nd = []
                for n in nulls:
                    nok = ~np.isnan(n).any(1)
                    if nok.sum() >= 8:
                        nd.append(best_offset(np.nanmean(bump(n[nok]), axis=0))[0])
                per_frame[layer].append(dict(frame=os.path.basename(f), scene=sc, n=int(okr.sum()), d_best=d_real, peak=pk_real, at_zero=at0, null_best=nd))
        if (k + 1) % 50 == 0:
            print(f"  {k + 1}/{len(frames)}", flush=True)

    res = {"split": a.split, "n_frames": len(frames), "gt_signature": SIG, "contrast_hit": CONTRAST_HIT, "null_shift_m": NULL_SHIFT, "layers": {}}
    lines = [f"paint vs label: {len(frames)} frames of {a.split} scenes, GT {SIG}; stripe = {CONTRAST_HIT:.2f} contrast over the asphalt next to it; control = label moved {NULL_SHIFT} m sideways", ""]
    lines.append(f"{'layer':13s} {'range m':8s} {'points':>7s} {'bump peak':>10s} {'peak at m':>10s} {'FWHM m':>7s} | {'hit +-0.2':>9s} {'ctl':>6s} | {'hit +-0.5':>9s} {'ctl':>6s}")
    for layer in LAYERS:
        res["layers"][layer] = {"bins": {}}
        for r in RANGES:
            A = acc[(layer, r)]
            if not A["bump"]:
                continue
            B = np.concatenate(A["bump"])
            mb = np.nanmean(B, axis=0)
            w, at, pk = fwhm(mb)
            h02, h05 = np.concatenate(A["hit02"]), np.concatenate(A["hit05"])
            n02 = np.concatenate(A["n02"]) if A["n02"] else np.zeros(0, bool)
            n05 = np.concatenate(A["n05"]) if A["n05"] else np.zeros(0, bool)
            e = dict(points=int(len(B)), bump_peak_gray=pk, peak_at_m=at, fwhm_m=w, hit02=float(h02.mean()), hit02_ctl=float(n02.mean()) if len(n02) else None,
                     hit05=float(h05.mean()), hit05_ctl=float(n05.mean()) if len(n05) else None, mean_bump=[float(x) for x in mb])
            res["layers"][layer]["bins"][f"{r[0]:.0f}-{r[1]:.0f}"] = e
            lines.append(f"{layer:13s} {r[0]:3.0f}-{r[1]:<4.0f} {len(B):7d} {pk:10.1f} {at:10.2f} {w:7.2f} | {e['hit02']:9.2f} {e['hit02_ctl'] if e['hit02_ctl'] is not None else float('nan'):6.2f} | {e['hit05']:9.2f} {e['hit05_ctl'] if e['hit05_ctl'] is not None else float('nan'):6.2f}")
    lines += ["", f"{'layer':13s} {'frames':>7s} {'best offset m: mean':>20s} {'SD':>6s} {'MAD':>6s} {'|d|>0.25':>9s} {'|d|>0.5':>8s} | control: {'SD':>5s} {'|d|>0.25':>9s}"]
    for layer in LAYERS:
        pf = per_frame[layer]
        if not pf:
            continue
        d = np.array([x["d_best"] for x in pf])
        nd = np.array([v for x in pf for v in x["null_best"]]) if any(x["null_best"] for x in pf) else np.zeros(0)
        mad = float(1.4826 * np.median(np.abs(d - np.median(d))))
        res["layers"][layer]["per_frame"] = dict(n=len(d), mean=float(d.mean()), sd=float(d.std()), mad=mad, frac_gt025=float((np.abs(d) > 0.25).mean()), frac_gt05=float((np.abs(d) > 0.5).mean()),
                                                  null_sd=float(nd.std()) if len(nd) else None, null_frac_gt025=float((np.abs(nd) > 0.25).mean()) if len(nd) else None,
                                                  gain_best_over_zero=float(np.mean([x["peak"] - x["at_zero"] for x in pf])))
        lines.append(f"{layer:13s} {len(d):7d} {d.mean():20.3f} {d.std():6.3f} {mad:6.3f} {(np.abs(d) > 0.25).mean():9.2f} {(np.abs(d) > 0.5).mean():8.2f} | {nd.std() if len(nd) else float('nan'):13.3f} {(np.abs(nd) > 0.25).mean() if len(nd) else float('nan'):9.2f}")
    res["per_frame_rows"] = per_frame
    json.dump(res, open(os.path.join(a.out, "paint_result.json"), "w"), indent=1)
    open(os.path.join(a.out, "paint_table.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 3, figsize=(14, 7))
        for i, layer in enumerate(LAYERS):
            for r in RANGES:
                e = res["layers"][layer]["bins"].get(f"{r[0]:.0f}-{r[1]:.0f}")
                if e:
                    ax[i, 0].plot(D, e["mean_bump"], label=f"{r[0]:.0f}-{r[1]:.0f} m")
            ax[i, 0].axvline(0, color="k", lw=0.5)
            ax[i, 0].set_title(f"{layer}: mean brightness across the label (gray levels over the row median)")
            ax[i, 0].set_xlabel("offset across the line on the ground (m)")
            ax[i, 0].legend()
            pf = per_frame[layer]
            if pf:
                d = [x["d_best"] for x in pf]
                nd = [v for x in pf for v in x["null_best"]]
                bins = np.linspace(-1, 1, 26)
                ax[i, 1].hist(d, bins=bins, alpha=0.7, label="label lines", density=True)
                ax[i, 1].hist(nd, bins=bins, alpha=0.5, label="control (moved 1.75 m)", density=True)
                ax[i, 1].set_title(f"{layer}: best offset per frame")
                ax[i, 1].set_xlabel("offset that puts the line on the brightest stripe (m)")
                ax[i, 1].legend()
            hit = [(res["layers"][layer]["bins"].get(f"{r[0]:.0f}-{r[1]:.0f}") or {}) for r in RANGES]
            xs = np.arange(len(RANGES))
            ax[i, 2].bar(xs - 0.2, [h.get("hit02", 0) for h in hit], 0.2, label="label +-0.2 m")
            ax[i, 2].bar(xs, [h.get("hit05", 0) for h in hit], 0.2, label="label +-0.5 m")
            ax[i, 2].bar(xs + 0.2, [h.get("hit05_ctl") or 0 for h in hit], 0.2, label="control +-0.5 m")
            ax[i, 2].set_xticks(xs)
            ax[i, 2].set_xticklabels([f"{r[0]:.0f}-{r[1]:.0f} m" for r in RANGES])
            ax[i, 2].set_title(f"{layer}: share of points with a stripe")
            ax[i, 2].legend()
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, "paint_vs_label.png"), dpi=110)
    except Exception as e:                                               # noqa: BLE001
        print("figure failed:", e)


if __name__ == "__main__":
    main()
