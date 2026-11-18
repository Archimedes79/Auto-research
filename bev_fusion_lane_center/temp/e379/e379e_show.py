"""E379 scratch (one use): the layer-1 ensemble cards of temp/e379/e379e_layer1/ with six decimals and their paired intervals against the reference run. usage: e379e_show.py"""
import glob
import json
import os

for p in sorted(glob.glob(os.path.join(os.environ["BFLC_ROOT"], "temp", "e379", "e379e_layer1", "*.json"))):
    c = json.load(open(p))
    m, i = c["metrics"], c["info"]
    pr = i["paired_vs_ref_run"]
    print(os.path.basename(p), c["frames"], "frames, runs", i["runs"])
    for k in ("topo_F1", "topo_P", "topo_R", "F1@1.0", "lanes_found", "Q", "junction_F1", "dup_frac", "vert100_dp", "junction_P", "junction_R"):
        print(f"  {k:<12} E {m[k]:.6f}  ref {i['ref_run'][k]:.6f}  E - ref {m[k] - i['ref_run'][k]:+.6f}" + (f"  paired {pr[k]['diff']:+.4f} [{pr[k]['lo']:+.4f}, {pr[k]['hi']:+.4f}]" if k in pr else ""))
