"""The card numbers of the champion, scored by the project's own scorer (`evaluation/qgraph_eval.py`, imported, not re-written), next to the stored
per-frame counts of the cards of record (`runs/e324_s_s<seed>/ends_card_snap_thr0.1` = snapped, `ends_card_snap_nogeom_attach_thr0.1` = attached
ends), for both read-outs from one forward; plus the link-candidate counts per batch of 4 / 8 frames (the original LinkHead subsamples a batch above
80,000 candidates at random; B's stored dump ran at batch 4).

    bash tools/env/_py14.sh training/lanegraph/scripts/lg_eval.py --seed 0 --frames 48 [--device cuda] [--out result.json]   # --frames 873 = all of VAL
"""
import argparse
import dataclasses
import json
import os
import pickle
import sys
import time

sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."), os.path.dirname(os.path.abspath(__file__))]
import numpy as np  # noqa: E402
import torch  # noqa: E402

from lg_config import CHAMPION, PROJECT  # noqa: E402
from lg_features import FeatureStore  # noqa: E402
from lg_model import LaneGraphNet, per_frame  # noqa: E402
from lg_project import use_project_code  # noqa: E402
from lg_readout import deliver  # noqa: E402

KEYS = ("topo_F1", "F1@1.0", "lanes_found", "junction_F1", "dup_frac", "vert100_dp", "Q")
CARDS = {"snapped": "ends_card_snap_thr0.1", "attached": "ends_card_snap_nogeom_attach_thr0.1"}
CAP = 80000


def same(a, b) -> bool:
    """equal per-frame counts (nested dicts / arrays / scalars, 1e-9)"""
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    try:
        return bool(np.allclose(np.asarray(a, float), np.asarray(b, float), atol=1e-9, equal_nan=True))
    except (ValueError, TypeError):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frames", type=int, default=48, help="the first N VAL frames")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None, help="json file of the numbers")
    a = ap.parse_args()
    use_project_code()
    import qgraph_eval as QE
    net = LaneGraphNet.from_runs(a.seed, dataclasses.replace(CHAMPION, attached_ends=True)).to(a.device)
    cfgs = {"snapped": dataclasses.replace(CHAMPION, attached_ends=False), "attached": net.cfg}
    store, frames = FeatureStore(), QE.val_frames()[: a.frames]
    run_dir = PROJECT / "runs" / f"e324_s_s{a.seed}"
    stored = {k: pickle.load(open(run_dir / f"{c}.counts.pkl", "rb"))["head"][: len(frames)] for k, c in CARDS.items()}
    mine, ncand, t_fwd = {k: [] for k in CARDS}, [], 0.0
    for i in range(0, len(frames), 4):
        chunk = [frames[k] for k in range(i, min(i + 4, len(frames)))]
        t0 = time.perf_counter()
        with torch.no_grad():
            soft = per_frame(net(store([f.row for f in chunk]).to(a.device)))
        t_fwd += time.perf_counter() - t0
        for k in range(i, i + len(chunk)):
            frames[k] = None                                                       # the frame's lane GT is freed with it
        for f, (lanes, pairs) in zip(chunk, soft):
            ncand.append(len(lanes["src"]))
            for name, cfg in cfgs.items():
                mine[name].append(QE.graph_counts(deliver(lanes, pairs, cfg).raw(), f))
    sub = ("rec", "topo", "lane", "dp")
    res = dict(seed=a.seed, frames=len(ncand), device=a.device, load_forward_ms_per_frame_batch4=1000 * t_fwd / len(ncand), candidates_per_frame=float(np.mean(ncand)))
    for b in (4, 8):
        sums = [sum(ncand[i:i + b]) for i in range(0, len(ncand), b)]
        res[f"candidates_batch{b}"] = dict(max=int(max(sums)), over_cap=int(sum(s > CAP for s in sums)), batches=len(sums))
    for name in CARDS:
        res[name] = dict(champion={k: float(v) for k, v in QE.graph_numbers(mine[name]).items() if k in KEYS},
                         stored={k: float(v) for k, v in QE.graph_numbers(stored[name]).items() if k in KEYS},
                         identical_frames=int(sum(same({k: m[k] for k in sub}, {k: s[k] for k in sub}) for m, s in zip(mine[name], stored[name]))))
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
