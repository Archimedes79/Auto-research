"""scratch (analysis, no verdict) stage 1 of the recall-by-paint test: the frames and label points of the paint check (temp/lane_tests/paint_vs_label.py, imported:
same frame choice, same densify / profiles / contrast_hits), saved per point: ego x, y, the stripe flags (contrast >= 0.25 within +-0.2 / +-0.5 m), the frame and scene.
Output: temp/overview/rp_points.npz  (arrays over all points of all frames, per layer) + the frame list (token, scene, file) in rp_frames.json
python temp/overview/rp_stage1_points.py [--max-frames 0]"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [os.path.join(ROOT, "temp", "lane_tests")]
import paint_vs_label as P  # noqa: E402

LAYERS = P.LAYERS
RANGES = P.RANGES


def choose_frames_as_then(split, n_per_scene, seed, max_frames, exclude):
    """paint_vs_label.choose_frames with the GT files left out that did not exist when the paint check ran (the 41 files of this signature newer than
    paint_result.json, listed in rp_newer_files.txt): the frames of one scene are picked by position in the sorted file list, so new files shift the picks"""
    import glob
    root = os.path.expanduser("~/data/nuscenes_full")
    scenes = {s["token"]: s["name"] for s in json.load(open(os.path.join(root, "v1.0-trainval", "scene.json")))}
    samples = {s["token"]: s["scene_token"] for s in json.load(open(os.path.join(root, "v1.0-trainval", "sample.json")))}
    from nuscenes.utils.splits import create_splits_scenes
    names = set(create_splits_scenes()[split])
    files = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "cache", "gt", f"*_{P.SIG}.npz"))):
        if os.path.basename(f) in exclude:
            continue
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--now", action="store_true", help="use the GT files that exist now (default: leave out the 41 newer than the paint check)")
    ap.add_argument("--out", default=os.path.join(HERE, "rp_points.npz"))
    a = ap.parse_args()
    from PIL import Image
    froot = os.path.expanduser("~/data/nuscenes_full")
    excl = set(open(os.path.join(HERE, "rp_newer_files.txt")).read().split()) if not a.now else set()
    frames = choose_frames_as_then("val", 3, 0, a.max_frames, excl)
    print(f"{len(frames)} frames of {len({s for s, _ in frames})} val scenes (GT {P.SIG})", flush=True)
    rows = {l: {"fi": [], "x": [], "y": [], "tx": [], "ty": [], "hit02": [], "hit05": []} for l in LAYERS}
    finfo = []
    for k, (sc, f) in enumerate(frames):
        g = np.load(f, allow_pickle=True)["gt"].item()["cur"]
        meta = g["meta"]
        tok = os.path.basename(f).split("_")[0]
        finfo.append({"i": k, "scene": sc, "token": tok, "file": os.path.basename(f)})
        im = np.asarray(Image.open(os.path.join(froot, meta["cam_file"])).convert("L"), np.float32)
        for layer in LAYERS:
            Pp, Tt = [], []
            for pl in meta["vectors"].get(layer, []):
                p, t = P.densify(pl)
                if len(p):
                    Pp.append(p)
                    Tt.append(t)
            if not Pp:
                continue
            Pp, Tt = np.concatenate(Pp), np.concatenate(Tt)
            keep = (Pp[:, 0] >= 5.0) & (Pp[:, 0] <= 30.0) & (np.abs(Pp[:, 1]) < 14.0)
            Pp, Tt = Pp[keep], Tt[keep]
            if len(Pp) == 0:
                continue
            real = P.profiles(im, meta, Pp, Tt, 0.0)
            okr = ~np.isnan(real).any(1)
            if okr.sum() < 4:                                     # as the paint check: such a frame / layer is skipped
                continue
            h = P.contrast_hits(real[okr])
            r = rows[layer]
            n = int(okr.sum())
            r["fi"].append(np.full(n, k, np.int32))
            r["x"].append(Pp[okr, 0])
            r["y"].append(Pp[okr, 1])
            r["tx"].append(Tt[okr, 0])
            r["ty"].append(Tt[okr, 1])
            r["hit02"].append(h["hit02"])
            r["hit05"].append(h["hit05"])
        if (k + 1) % 25 == 0:
            print(f"  {k + 1}/{len(frames)}", flush=True)
    out = {}
    for layer in LAYERS:
        for key, v in rows[layer].items():
            out[f"{layer}/{key}"] = np.concatenate(v) if v else np.zeros(0)
    np.savez(a.out, **out)
    json.dump(finfo, open(os.path.join(HERE, "rp_frames.json"), "w"), indent=0)
    print("saved", a.out)
    # reproduction of the paint check table (points per range bin, hit +-0.5 / +-0.2)
    print(f"{'layer':13s} {'range':8s} {'points':>7s} {'hit+-0.2':>9s} {'hit+-0.5':>9s}   (paint_table.txt: lane 1100/3190/5533, 0.46/0.41/0.30 at +-0.5; 0.35/0.34/0.21 at +-0.2)")
    for layer in LAYERS:
        x, h2, h5 = out[f"{layer}/x"], out[f"{layer}/hit02"], out[f"{layer}/hit05"]
        for lo, hi in RANGES:
            s = (x >= lo) & (x < hi)
            print(f"{layer:13s} {lo:3.0f}-{hi:<4.0f} {int(s.sum()):7d} {h2[s].mean():9.2f} {h5[s].mean():9.2f}")


if __name__ == "__main__":
    main()
