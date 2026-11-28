"""How close a new extraction of the foundation features is to the stored ones (`<data>/lane_center/b123/feats`, read only): a few frames are
extracted again by `gcar_features.frames` and compared per channel group (dec32, logit7): bit-equal or not, the largest and the mean absolute
difference, the correlation. Every frame is extracted twice; the second against the first is the noise of the rasteriser itself, the bar a
comparison with the stored rows is read against. GPU, batch 1, about a minute plus 0.4 s per frame and run.

    python -P -m lane_center.net.prep.foundation.check_features [--n 8] [--parts val train] [--clone DIR] [--weights FILE] [--out FILE]

Frames (`pick_rows`): only rows of TRAIN and VAL of data/splits.json (the TEST rows of the part "val" stay closed); of each part n scenes at
even steps through its scenes, of each the middle row, moved by one row where needed so that even and odd rows alternate. The sets are reported
apart (val, train even rows, train odd rows): the stored arrays were written in that order by two versions of the code (README.md here).
Writes the comparison as json below the run root (`bflc_paths.SCRATCH`/foundation/check_features.json), never below the data root.
"""
import argparse
import hashlib
import json
import os
import statistics
import time

import numpy as np

from lane_center.net.prep.foundation import gcar_features as GF
from lane_center.training import bflc_paths as BP

KINDS = {"stored": "new against stored", "repeat": "new against new (the rasteriser's own noise)"}


def open_rows():
    """{part: the rows the check may read} = TRAIN and VAL of data/splits.json; the part "val" also holds the TEST rows, which stay closed"""
    with open(os.path.join(BP.ROOT, "data", "splits.json"), encoding="utf-8") as f:
        splits = json.load(f)
    return {"train": splits["train"], "val": splits["val"]}


def pick_rows(scene, rows, n):
    """scene [all rows of a part] = the scene id of every row, rows = the rows that may be read -> n of them (fewer when they have fewer scenes),
    one per picked scene"""
    rows = np.asarray(sorted(rows))
    scene = np.asarray(scene)[rows]
    ids = np.unique(scene)
    out = []
    for k, i in enumerate(np.linspace(0, len(ids) - 1, min(n, len(ids))).round().astype(int)):
        own = rows[scene == ids[i]]
        row = int(own[len(own) // 2])
        if row % 2 != k % 2:                                          # the k-th pick is an even row for even k
            row = next((r for r in (row + 1, row - 1) if r in own), row)
        out.append(row)
    return out


def row_set(part, row):
    return part if part != "train" else f"train {'odd' if row % 2 else 'even'} rows"


def compare(new, ref):
    """two arrays of one channel group -> how far apart they are; corr is None for a constant array"""
    a, b = np.asarray(new, np.float64), np.asarray(ref, np.float64)
    d = np.abs(a - b)
    corr = float(np.corrcoef(a.ravel(), b.ravel())[0, 1]) if a.std() > 0 and b.std() > 0 else None
    return {"equal": bool(np.array_equal(a, b)), "max_abs": float(d.max()), "mean_abs": float(d.mean()), "ref_mean_abs": float(np.abs(b).mean()), "corr": corr}


def summarise(rows):
    """the comparisons of several frames as one: how many are bit-equal, the worst and the mean difference, the median and the worst correlation"""
    corr = [r["corr"] for r in rows if r["corr"] is not None]
    return {"frames": len(rows), "equal": sum(r["equal"] for r in rows), "max_abs": max(r["max_abs"] for r in rows),
            "mean_abs": statistics.fmean(r["mean_abs"] for r in rows), "ref_mean_abs": statistics.fmean(r["ref_mean_abs"] for r in rows),
            "corr_median": statistics.median(corr) if corr else None, "corr_min": min(corr) if corr else None}


def scratch(path):
    """the file the comparison is written to; refused below the data root"""
    path, data = os.path.abspath(path), os.path.abspath(os.path.dirname(BP.DATA))
    if (path + os.sep).startswith(data + os.sep):
        raise GF.Missing(f"{path} is below the data root {data}: the check writes below the run root only")
    return path


def stored(parts):
    """the stored arrays {part: {group: memmap}}, read only, after a check that their rows are the frames of the repository's list"""
    feats, tokens, out = os.path.join(BP.B123, "feats"), GF.sample_tokens(), {}
    for part in parts:
        files = {g: os.path.join(feats, f"{part}_{g}.npy") for g in GF.GROUPS}
        gone = [f for f in list(files.values()) + [os.path.join(feats, f"{part}_tokens.json")] if not os.path.isfile(f)]
        if gone:
            raise GF.Missing(f"no stored features to compare with: {gone[0]} is missing")
        with open(os.path.join(feats, f"{part}_tokens.json"), "rb") as f:
            if hashlib.sha256(f.read()).hexdigest() != GF.SAMPLES_SHA256[part]:
                raise GF.Missing(f"{feats}/{part}_tokens.json is not the list of b123_samples.json: the stored rows are other frames")
        out[part] = {g: np.load(f, mmap_mode="r") for g, f in files.items()}
        assert all(a.shape == (len(tokens[part]), GF.GROUPS[g], 200, 200) for g, a in out[part].items()), {g: a.shape for g, a in out[part].items()}
    return out


def parse(argv=None):
    ap = argparse.ArgumentParser(description="a new extraction of a few frames against the stored foundation features")
    ap.add_argument("--n", type=int, default=8, help="frames per part")
    ap.add_argument("--parts", nargs="+", default=list(GF.PARTS), choices=GF.PARTS)
    ap.add_argument("--clone", default=os.environ.get("GCAR_REPO"), help="the GaussianCaR clone (default: GCAR_REPO)")
    ap.add_argument("--weights", default=os.environ.get("GCAR_WEIGHTS"), help="nuscenes_map.ckpt (default: GCAR_WEIGHTS)")
    ap.add_argument("--workers", type=int, default=0, help="processes that read pictures and radar (0: in this process)")
    ap.add_argument("--out", default=os.path.join(BP.SCRATCH, "foundation", "check_features.json"))
    return ap.parse_args(argv)


def main():
    import torch

    a = parse()
    try:
        out, ref = scratch(a.out), stored(a.parts)
        may = open_rows()
        picks = [(part, row) for part in a.parts for row in pick_rows(np.load(os.path.join(BP.B123, f"poses_{part}.npz"))["scene"], may[part], a.n)]
        GF.need_raw(BP.NUSC)
        module, cfg = GF.load(a.clone, a.weights)
    except GF.Missing as e:
        raise SystemExit(f"check_features: {e}")
    tokens = GF.sample_tokens()
    toks = [tokens[part][row] for part, row in picks]
    t0 = time.time()
    new = list(GF.frames(module, cfg, toks + toks, a.workers))             # every frame twice, as two runs
    seconds = time.time() - t0
    frames = []
    for k, (part, row) in enumerate(picks):
        first, second = new[k], new[k + len(picks)]
        frames.append({"part": part, "row": row, "token": toks[k], "set": row_set(part, row),
                       "stored": {g: compare(first[g], ref[part][g][row]) for g in GF.GROUPS},
                       "repeat": {g: compare(second[g], first[g]) for g in GF.GROUPS}})
    sets = {name: {kind: {g: summarise([f[kind][g] for f in frames if f["set"] == name]) for g in GF.GROUPS} for kind in KINDS}
            for name in dict.fromkeys(f["set"] for f in frames)}
    report = {"made": time.strftime("%Y-%m-%dT%H:%M:%S"), "stored": os.path.join(BP.B123, "feats"), "clone_commit": GF.COMMIT,
              "weights_sha256": GF.WEIGHTS["sha256"], "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "frames_per_part": a.n,
              "seconds_for_two_runs": round(seconds, 1), "kinds": KINDS, "sets": sets, "frames": frames}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    for name, kinds in sets.items():
        for kind, groups in kinds.items():
            for g, s in groups.items():
                print(f"{name:16s} {g:7s} {KINDS[kind]:46s} {s['equal']}/{s['frames']} bit-equal, |diff| max {s['max_abs']:.3f} mean {s['mean_abs']:.4f} "
                      f"(mean |value| {s['ref_mean_abs']:.3f}), corr median {s['corr_median']:.5f} min {s['corr_min']:.5f}")
    print("written:", out)


if __name__ == "__main__":
    main()
