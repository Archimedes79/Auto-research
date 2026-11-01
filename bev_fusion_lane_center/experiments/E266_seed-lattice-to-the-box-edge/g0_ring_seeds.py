"""E266 gate G0 (does the ring flag change the output?): the builder of a registry entry (arm c, default `recall`) and of its ring entry (arm b,
default `recall_ring`, `lattice_ring` 47.5) on the first n VAL frames, both re-inferred on one device (builder_raw fresh: no cache read or
written). Reports
  1. the cache files of the two entries (must differ: the cache-key fix of evaluate.py),
  2. the share of kept seeds (level-0 valid nodes) with max(|x|,|y|) >= 45.5 m in each arm (G0 passes if arm b >= 2 x arm c),
  3. the share of arm c's inner seeds (max(|x|,|y|) < 45 m) that arm b no longer keeps at the same place / within 1 m (the ring candidates
     enter the same top-K0 selection),
  4. with --sanity: the share of arm c's cached seeds that the fresh arm c reproduces within 0.05 m (the same network as the cache).
Usage (from the problem folder): wsl bash tools/env/_py14.sh experiments/E266_seed-lattice-to-the-box-edge/g0_ring_seeds.py [--n 50] [--dev cuda|cpu] [--sanity]"""
import argparse
import gc
import os

import numpy as np
from scipy.spatial import cKDTree

import evaluate as EV


def kept_seeds(raw):
    """positions of the kept seeds of every frame (each seed appears twice, as the forward and the backward tip)"""
    return [r["pos"][(r["level"] == 0) & (r["valid"] > 0)] for r in raw]


def share_matched(A, B, r, sel=lambda p: np.ones(len(p), bool)):
    """over frames: share of the points of A (selected by sel) that have a point of B in the same frame within r"""
    hit = tot = 0
    for a, b in zip(A, B):
        a = a[sel(a)]
        tot += len(a)
        if len(a) and len(b):
            hit += int((cKDTree(b).query(a)[0] <= r).sum())
    return hit / max(tot, 1)


def ring_share(S, edge=45.5):
    p = np.concatenate(S)
    return float((np.abs(p).max(-1) >= edge).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry", default="recall")
    ap.add_argument("--ring", default="recall_ring")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--dev", default="cuda")
    ap.add_argument("--sanity", action="store_true", help="also compare arm c's fresh seeds with its cache (2026-11-01, CPU, 50 frames: 0.9943 within 0.05 m)")
    a = ap.parse_args()
    reg = EV.registry()
    c, b = reg[a.entry], reg[a.ring]
    pb = EV.cache_paths(b, "val")
    pc = next((p for p in EV.cache_paths(c, "val") if os.path.exists(p)), None)  # the file arm c reads
    if pc is None:
        raise SystemExit(f"G0 not run: no VAL cache of {a.entry} (looked for {EV.cache_paths(c, 'val')}); evaluate.py {a.entry} on the GPU writes it")
    print(f"cache files: {a.entry} {pc}\n             {a.ring} {pb}\n  -> {'different' if pc not in pb else 'THE SAME (cache-key bug)'}")
    ks = list(range(a.n))
    cached = None
    if a.sanity:  # loads the whole cache of arm c (about 1 GB more memory)
        cached = kept_seeds(EV.builder_raw(c, "val", ks=ks, require_cache=True)[0])
        gc.collect()
    sc = kept_seeds(EV.builder_raw(c, "val", ks=ks, fresh=True, dev=a.dev)[0])  # both arms re-inferred on the same device
    sb = kept_seeds(EV.builder_raw(b, "val", ks=ks, fresh=True, dev=a.dev)[0])
    rc, rb = ring_share(sc), ring_share(sb)
    inner = lambda p: np.abs(p).max(-1) < 45.0  # noqa: E731
    print(f"kept seeds per frame: {a.entry} {np.mean([len(s) for s in sc]) / 2:.1f}, {a.ring} {np.mean([len(s) for s in sb]) / 2:.1f} ({a.n} frames, device {a.dev})")
    print(f"share of kept seeds at max(|x|,|y|) >= 45.5 m: {a.entry} {rc:.4f}, {a.ring} {rb:.4f} (ratio {rb / max(rc, 1e-9):.2f})")
    print(f"inner seeds of {a.entry} not kept by {a.ring} at the same place (1 mm): {1 - share_matched(sc, sb, 1e-3, inner):.4f}, "
          f"none within 1 m: {1 - share_matched(sc, sb, 1.0, inner):.4f} (with the time recursion the warped previous-frame nodes also compete)")
    if cached is not None:
        print(f"sanity: cached seeds of {a.entry} reproduced by the fresh run within 0.05 m: {share_matched(cached, sc, 0.05):.4f}")
    ok = pc not in pb and rb >= 2 * rc
    print(f"G0 {'PASS' if ok else 'FAIL'} (different cache files and ring share of arm b >= 2 x arm c)")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
