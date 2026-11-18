"""E77: SD map (roads, lane counts, centred route) of every labelled row -> ~/exp_gcar_b123/sd_{train,val}.pkl (list; None where the row has no route).
Usage: sd_precompute.py [<tag>]   (tag: b123 by default; b456 or a check set: ~/exp_gcar_<tag>/)"""
import os, pickle, sys, time
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sd_lib as SL

D = os.path.expanduser(f"~/exp_gcar_{sys.argv[1] if len(sys.argv) > 1 else 'b123'}")


def work(args):
    part, i = args
    g, rt = _G[part][i], _RT[part][i]
    if rt is None:
        return None
    return SL.build_sd(g, rt["route"])


if __name__ == "__main__":
    _G, _RT = {}, {}
    for part in ("train", "val"):
        _G[part] = pickle.load(open(f"{D}/graph_gt_{part}.pkl", "rb"))
        _RT[part] = pickle.load(open(f"{D}/route_gt_{part}.pkl", "rb"))
    for part in ("val", "train"):
        t0 = time.time()
        with Pool(6) as p:
            out = p.map(work, [(part, i) for i in range(len(_G[part]))], chunksize=40)
        pickle.dump(out, open(f"{D}/sd_{part}.pkl", "wb"))
        print(part, len(out), sum(o is not None for o in out), f"{time.time() - t0:.0f} s", flush=True)
