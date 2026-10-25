"""Loop 18 (one-off): fork pruning on another trained builder of loop 16 (default rc1 = the recall-weighted fine-tune of the champion, which paid for recall with J 0.18). Raw outputs cached in ~/exp_gcar_b123/p18/raw_val873_<name>.pkl.
Usage (gcar): p18_variant_prune.py --name e123_D7_r3_rc1 [--M 288 --K0 144]"""
import bflc_paths as BP  # project paths from the location of this checkout
import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p16_data as PD  # noqa: E402
import p16_infer as PI  # noqa: E402
import p16_timing as TM  # noqa: E402
import sparse_metrics as SM  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="e123_D7_r3_rc1")
ap.add_argument("--M", type=int, default=288)
ap.add_argument("--K0", type=int, default=144)
ap.add_argument("--thr", default="0,0.2,0.3,0.4")
ap.add_argument("--weight", default="-", help="recall loss weight of the run (for the table)")
ap.add_argument("--json_thr", type=float, default=0.4, help="threshold whose card is written to runs/e18/rc_eval/<tag>.json")
ap.add_argument("--tag", default="rc")
a = ap.parse_args()
net, cfg = TM.load_builder(a.M, a.K0, a.name)
cache = os.path.expanduser(f"~/exp_gcar_b123/p18/raw_val873_{a.name}_{a.M}.pkl")
if os.path.exists(cache):
    raw, gts = pickle.load(open(cache, "rb"))
else:
    ks = list(range(873))
    ds = PD.BuilderData("val_clean", ks=ks, train=False, with_prev=cfg.temporal)
    raw = []
    _, gts, _ = PI.predict(net, ds, ks, taus=(0.7,), bs=8, workers=2, raw=raw)
    pickle.dump((raw, gts), open(cache, "wb"))
for t in a.thr.split(","):
    gs = [PI.assemble(r, None, 0.7, cfg, fast=2, fork_prune=float(t), **TM.ASM) for r in raw]
    m = SM.card_any(gs, gts)
    print(f"{a.name} {a.M}/{a.K0} prune {t}: S {m['S']:.3f} Q {SM.quality(m):.3f} J {m['J']:.3f} (split P {m['split6_P']:.2f} R {m['split6_R']:.2f}) F1@1 {m['F1@1.0']:.3f} R@1 {m['R@1.0']:.3f} P@1 {m['P@1.0']:.3f} found {m['lanes_found']:.3f} FL {m['false_len_frac']:.3f} v100 {m['vert100']:.1f}", flush=True)
    if abs(float(t) - a.json_thr) < 1e-9:
        od = BP.ROOT + "/runs/e18/rc_eval"
        os.makedirs(od, exist_ok=True)
        import json
        json.dump(dict(name=a.name, weight=a.weight, thr=float(t), **{k: float(m[k]) for k in ("S", "J", "F1@1.0", "R@1.0", "P@1.0", "lanes_found", "false_len_frac", "vert100", "split6_P", "split6_R", "merge6_P", "merge6_R")}, Q=float(SM.quality(m))), open(f"{od}/{a.tag}.json", "w"), indent=1)
