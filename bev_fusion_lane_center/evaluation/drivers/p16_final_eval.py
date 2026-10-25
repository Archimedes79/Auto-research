"""Loop 16: evaluate a trained builder with explicit assembly settings on a whole split (val_clean 873 frames, test_clean 833 frames), threshold sweep, writes a metric card.
The settings (assembly, threshold) are chosen on the VAL250 subset / full VAL; for TEST pass the chosen threshold with --tau (no sweep, nothing is selected on TEST).
Usage (gcar, GPU): p16_final_eval.py --design D7 --name e123_D7_r3_j1 --split val_clean --asm cov_r=0.5 vtx_r=1.5 junc_thr=0.5 [--tau 0.6] [--tag final]  -> cards/<name>.<tag>_<split>.json
Also saves the node outputs of every frame (pickle, for frames and figures) with --keep."""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p16_common as C  # noqa: E402
import p16_data as PD  # noqa: E402
import p16_infer as PI  # noqa: E402
import p16_train as PT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--split", default="val_clean")
    ap.add_argument("--asm", nargs="*", default=[])
    ap.add_argument("--tau", type=float, default=None, help="fixed threshold (TEST); default: sweep and pick the best S with R@1 >= 0.3")
    ap.add_argument("--tag", default="final")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--bs", type=int, default=4)
    a = ap.parse_args()
    asm = {k: float(v) for k, v in (x.split("=") for x in a.asm)}
    c = json.load(open(f"{C.LADDER}/cards/{a.name}.json", encoding="utf-8-sig"))
    extra = dict(c["cfg"])
    extra.update(PT.parse_set(a.set))
    net, cfg = PT.build(a.design, extra)
    net = net.cuda()
    net.load_state_dict(torch.load(f"{C.OUT}/e123/{a.name}/model.pt", map_location="cuda"), strict=False)
    ds = PD.BuilderData(a.split, ks=None, train=False, with_prev=cfg.temporal)
    ks = list(range(len(ds)))
    raw = []
    t0 = time.time()
    _, gts, _ = PI.predict(net, ds, ks, taus=(0.5,), bs=a.bs, workers=2, raw=raw)
    taus = [a.tau] if a.tau is not None else [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    cards = {}
    for t in taus:
        g = [PI.assemble(r, None, t, cfg, **asm) for r in raw]
        m = C.card(g, gts)
        m["tau"] = t
        cards[t] = m
        print(f"tau {t:.2f} | " + " ".join(f"{k} {m[k]:.3f}" for k in ("S", "F1@1.0", "F1@0.5", "J", "split6_F1", "merge6_F1", "Sp", "vert100", "false_len_frac", "R@1.0", "P@1.0", "lanes_found")), flush=True)
        if a.keep and (a.tau is not None):
            pickle.dump(dict(graphs=g, gts=gts, ks=ks, asm=asm, tau=t), open(f"{C.LADDER}/diag/{a.name}.{a.tag}_{a.split}.pkl", "wb"))
    curve = [dict(thr=t, vert100=m["vert100"], **{k: m[k] for k in ("F1@1.0", "R@1.0", "false_len_frac")}) for t, m in cards.items()]
    fr = C.frontier_scalars(curve) if len(curve) > 2 else {}
    sane = [m for m in cards.values() if m["R@1.0"] >= 0.3] or list(cards.values())
    best = max(sane, key=lambda m: m["S"])
    card = dict(run=a.name, kind="final-eval", design=a.design, cfg=cfg.asdict(), split=a.split, n=len(gts), asm=asm, metrics=best, sweep={str(t): {k: m[k] for k in ("S", "F1@1.0", "F1@0.5", "J", "Sp", "false_len_frac", "vert100", "R@1.0", "P@1.0", "lanes_found", "split6_F1", "merge6_F1")} for t, m in cards.items()}, eval_seconds=time.time() - t0, **fr)
    C.write_card(f"{a.name}.{a.tag}_{a.split}", card)
    print("best:", a.name, a.split, "tau", best["tau"], "S %.4f F1@1 %.4f J %.3f v100 %.1f" % (best["S"], best["F1@1.0"], best["J"], best["vert100"]))


if __name__ == "__main__":
    main()
