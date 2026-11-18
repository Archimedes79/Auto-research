import argparse, sys, time, os
import numpy as np
ROOT = os.environ["BFLC_ROOT"]
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import jtoken_eval as JE
quiet = lambda *x, **k: None

def stats(split, every, run="e367_b25_s0"):
    a = argparse.Namespace(run=run, model="model.pt", seed=0, split=split, every=every, drop=0.0, drop_seed=0, live=False, max=0)
    t0 = time.time(); r = JE.read_dump(a, JE.RUNS, 0); print(split, "read_dump", round(time.time() - t0, 1), "s", len(r["fr"]), "frames", flush=True)
    for gname in ("perclass", "default"):
        cells = JE.SWEEP_GRIDS[gname]
        tot = 0; uniq = 0; uniq_nodes = 0
        for f, v in zip(r["fr"], r["valid"]):
            keys = set(); nk = set()
            for lt, ltl, kt, g in cells:
                d = JE.decode_variant(f, v, lt, kt, g, ltl)
                keys.add((d["keep"].tobytes(), d["rec"]["src"].tobytes(), d["rec"]["dst"].tobytes()))
            tot += len(cells); uniq += len(keys)
        print(f"  grid {gname}: {len(cells)} cells x {len(r['fr'])} frames = {tot} frame-cells, distinct predictions {uniq} ({100*uniq/tot:.0f}%) -> max speed-up {tot/uniq:.1f}x", flush=True)
    # score() with a variant: rec0 + variant + ref_cell: how many of the three coincide per frame
    cellA = (0.3, 0.2, 0.6, False); cellB = (0.3, 0.3, 0.5, True)
    same = 0
    for f, v in zip(r["fr"], r["valid"]):
        da = JE.decode_variant(f, v, cellA[0], cellA[2], cellA[3], cellA[1]); db = JE.decode_variant(f, v, cellB[0], cellB[2], cellB[3], cellB[1])
        same += (da["keep"].tobytes(), da["rec"]["src"].tobytes(), da["rec"]["dst"].tobytes()) == (db["keep"].tobytes(), db["rec"]["src"].tobytes(), db["rec"]["dst"].tobytes())
    print(f"  D115 cell vs record cell: identical prediction in {same}/{len(r['fr'])} frames", flush=True)

stats("train", 3)
stats("val", 1)
