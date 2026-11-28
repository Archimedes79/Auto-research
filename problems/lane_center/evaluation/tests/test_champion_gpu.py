"""The champion through `lane_net.LaneNet`, on the GPU, by hand (not in the list of `research tests`: it needs the GPU, the token cache and the stored run):

    python -P -m lane_center.evaluation.tests.test_champion_gpu [--run E396.champ_s0] [--n 16]      (with the module path: README.md)

1. The network built from the run's files (`lane_predict.live_source` -> `LaneNet.from_checkpoint`) gives on the first --n VAL frames (a multiple of 8), in the predict step's
   batching (B' at 4, the head at 8: `lane_predict.live_dump`), exactly the records that the stored live dump of the run holds: every field of every record bit for bit (the
   frame's own tokens among them; a cache-trained run's also equal the cache). So the delivered graphs at the run's decode (`lane_readout.graph`, as evaluation/adapter.py
   decodes them) are the stored ones.
2. `LaneNet.graph(x, decode)`, features -> graph in one forward at batch 4, delivers the same nodes (the tokens' own fields bit for bit) and the same links; the written
   existence 1 - p_none is compared to 1e-4 (the head runs at batch 4 there, at 8 in the dump: float noise of the batch size, reported).
Exit 1 on a difference. A diagnostic of the code, never a record."""
import argparse
import json
import os
import sys

import numpy as np
import torch

from lane_center.training import bflc_paths as BP
from lane_center.evaluation import lane_dump as DU
from lane_center.evaluation import lane_predict as PR
from lane_center.net.model import lane_readout as RO

E_TOL = 1e-4


def same(x, y):
    """two values of a record, deeply: dicts key by key, arrays with dtype and content (NaN = NaN)"""
    if isinstance(x, dict):
        return isinstance(y, dict) and x.keys() == y.keys() and all(same(x[k], y[k]) for k in x)
    if isinstance(x, np.ndarray):
        return isinstance(y, np.ndarray) and x.dtype == y.dtype and np.array_equal(x, y, equal_nan=x.dtype.kind == "f")
    return x == y


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="E396.champ_s0", help="a measured run below the run root (its model.pt, config.json and stored live dump of VAL)")
    ap.add_argument("--n", type=int, default=16)
    a = ap.parse_args(argv)
    assert torch.cuda.is_available(), "no GPU visible: the stored dump was made on the GPU (the CPU differs by more than float noise)"
    assert a.n > 0 and a.n % PR.HEAD_BS == 0, f"--n {a.n}: a multiple of the head's batch {PR.HEAD_BS} (a short last batch is not a batch of the dump)"
    cfg = json.load(open(os.path.join(BP.RUNS, a.run, "config.json"), encoding="utf-8"))
    ns = argparse.Namespace(run=a.run, seed=cfg["seed"], split="val", live=True, every=1, bs=None)
    r = DU.read_dump(ns, BP.RUNS, 0)                                                          # the stored dump, checked against the run and the B' that ran
    assert (r["meta"]["bs_b"], r["meta"]["bs_head"], r["meta"]["device"]) == (PR.LIVE_BS, PR.HEAD_BS, "cuda"), r["meta"]
    idx = r["idx"][:a.n]
    frames, meta = PR.live_dump(ns, r["ck"], r["kw"], r["ds"], idx, "cuda", BP.RUNS)
    assert all(np.isfinite(f.pop("ms")) for f in frames) and meta["cost"]["peak_MB"] > 0 and meta["cost"]["params_M"] > 2, meta.get("cost")     # the measurement of the step: a time per frame (never equal), memory, parameters
    older = sorted({k for old in r["fr"][:a.n] for k in old} - set(frames[0]) - {"ms"})         # fields of removed diagnostics that an older dump still holds
    assert all(set(f) <= set(old) for f, old in zip(frames, r["fr"])), "a record holds a field the stored dump lacks"
    bad = [j for j, (f, old) in enumerate(zip(frames, r["fr"])) if not all(same(f[k], old[k]) for k in f)]
    fields = sorted({k for j in bad for k in frames[j] if not same(frames[j][k], r["fr"][j][k])})
    d = RO.Decode(**cfg["decode"])
    want = [RO.graph(t, f, v, d) for t, f, v in zip(r["toks"][:a.n], r["fr"][:a.n], r["valid"][:a.n])]
    got = [RO.graph(f["tokens_np"], f, f["tokens_np"]["valid"], d) for f in frames]
    g_bad = [j for j, (p, q) in enumerate(zip(got, want)) if not (same(p["nodes"], q["nodes"]) and same({k: v for k, v in p["links"].items() if v is not None}, {k: v for k, v in q["links"].items() if v is not None}))]
    print(f"{a.run}: {len(frames)} VAL frames through LaneNet in the predict batching: records equal to the stored dump on {len(frames) - len(bad)}"
          + (f" (differing fields {fields}, frames {bad[:8]})" if bad else "") + (f" (the stored dump also holds {older}: fields of removed diagnostics)" if older else "")
          + f", tokens equal to the cache on {meta['tokens_exact_frames']} (the stored dump: tokens_exact {r['meta']['tokens_exact']}); graphs at the decode {dict(d._asdict())} equal on {len(got) - len(g_bad)}; "
          f"{sum(len(p['nodes']['e']) for p in want)} nodes, {sum(len(p['links']['src']) for p in want)} links")
    net = PR.live_source(os.path.join(BP.RUNS, a.run, "model.pt"), r["ck"], r["kw"], r["ds"].K)[0].to("cuda")
    data = PR.live_data("val", r["kw"]["seed"], r["kw"]["ident"])
    one, e_gap = [], 0.0
    for c in range(0, a.n, PR.LIVE_BS):
        js = range(c, c + PR.LIVE_BS)
        for j, p in zip(js, net.graph(torch.stack([data[int(idx[j])]["x"] for j in js]), d)):
            q = want[j]
            ok = all(same(p["nodes"][k], q["nodes"][k]) for k in p["nodes"] if k != "e") and all(same(p["links"][k], q["links"][k]) for k in ("src", "dst", "long", "p"))
            if ok:
                e_gap = max(e_gap, float(np.abs(p["nodes"]["e"] - q["nodes"]["e"]).max(initial=0.0)), float(np.abs(p["links"]["a"] - q["links"]["a"]).max(initial=0.0)),
                            float(np.abs(p["links"]["b"] - q["links"]["b"]).max(initial=0.0)))
            else:
                one.append(j)
    print(f"LaneNet.graph, one forward at batch {PR.LIVE_BS}: the same nodes and links on {a.n - len(one)} of {a.n} frames" + (f" (frames {one[:8]} differ)" if one else "")
          + f"; largest difference of the written existence and the handle scales {e_gap:.2e} (bound {E_TOL})")
    exact = meta["tokens_exact"] or not r["meta"]["tokens_exact"]                             # a joint run's tokens are its own B''s, not the cache's
    ok = not bad and not g_bad and exact and not one and e_gap <= E_TOL
    print("CHAMPION " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
