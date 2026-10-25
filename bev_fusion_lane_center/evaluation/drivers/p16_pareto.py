"""Loop 16b: quality against compute time of the builder variants (VAL250, final assembly, tau 0.7): time recursion on / off, beam M / seeds K0.
Time per frame, batch 1: CUDA-graph forward incl. transfer + CPU assembly (cached foundation features as input, same for all variants).
Usage (gcar, GPU, idle machine): p16_pareto.py [--n 60]"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p16_common as C  # noqa: E402
import p16_data as PD  # noqa: E402
import p16_infer as PI  # noqa: E402
import p16_runtime as RT  # noqa: E402
import p16_timing as TM  # noqa: E402

CONFIGS = [("e123_D7_r3_f1", 288, 144), ("e123_D7_r3_f1", 192, 96), ("e123_D7_r3_f1", 144, 72), ("e123_D7_r3_f1", 96, 48),
           ("e123_D7_r3_j1", 288, 144), ("e123_D7_r3_j1", 192, 96), ("e123_D7_r3_j1", 144, 72), ("e123_D7_r3_j1", 96, 48), ("e123_D7_r3_j1", 64, 32),
           ("e123_D7_r3_s2", 288, 144), ("e123_D7_r3_s2", 144, 72)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    rows = []
    for name, M, K0 in CONFIGS:
        if a.only and a.only not in name:
            continue
        net, cfg = TM.load_builder(M, K0, name)
        ks = C.val_subset()
        ds = PD.BuilderData("val_clean", ks=ks, train=False, with_prev=cfg.temporal)
        # timing
        batches = [TM.to_gpu(PD.collate([ds[i]])) for i in range(a.n)]
        fb = RT.FastBuilder(net, cfg, batches[0])
        tf, raws = [], []
        for rep in range(2):
            tf.clear()
            raws.clear()
            for b in batches:
                torch.cuda.synchronize()
                t0 = time.time()
                r = fb.raw(b)
                tf.append(time.time() - t0)
                raws.append(r)
        ta = []
        for rep in range(2):
            ta.clear()
            for r in raws:
                t0 = time.time()
                PI.assemble(r, None, 0.7, cfg, **TM.ASM)
                ta.append(time.time() - t0)
        # quality on VAL250 (eager path, batch 8)
        raw = []
        _, gts, _ = PI.predict(net, ds, ks, taus=(0.7,), bs=8, workers=2, raw=raw)
        graphs = [PI.assemble(r, None, 0.7, cfg, **TM.ASM) for r in raw]
        m = C.card(graphs, gts)
        row = dict(name=name, M=M, K0=K0, temporal=bool(cfg.temporal), forward_ms=float(np.median(tf)) * 1e3, assembly_ms=float(np.median(ta)) * 1e3, **{k: float(m[k]) for k in ("S", "F1@1.0", "F1@0.5", "J", "vert100", "false_len_frac", "lanes_found") if k in m})
        row["total_ms"] = row["forward_ms"] + row["assembly_ms"]
        rows.append(row)
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
        del fb, net
        torch.cuda.empty_cache()
        json.dump(rows, open(f"{C.LADDER}/diag/pareto.json", "w"), indent=1)
    print("done")


if __name__ == "__main__":
    main()
