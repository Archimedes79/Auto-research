"""Loop 16 control arm R (E122): evaluate a trace network E94 trained at a ladder fidelity on the fixed VAL subset (or full VAL / TEST) with the standard wavefront protocol
(192 seeds, near order, orientation by the direction head, as the final E94 evaluation) and write the metric card.
Usage (gcar, GPU): p16_ctrl.py --name e122_s0_1u [--split val_clean] [--subset val250|all] [--u 1] [--seed 0]"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import p10_infer as PI  # noqa: E402
import p10_net as PN  # noqa: E402
import p16_common as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--split", default="val_clean")
    ap.add_argument("--subset", default="val250")
    ap.add_argument("--u", type=float, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tau_on", type=float, default=0.5)
    ap.add_argument("--tag", default="")
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    dev = "cuda"
    net = PN.TraceNet().to(dev)
    net.load_state_dict(torch.load(f"{C.OUT}/{a.name}/model.pt", map_location=dev))
    es = PI.EvalSet(a.split)
    ks = C.val_subset() if (a.subset == "val250" and a.split == "val_clean") else list(range(len(es.rows)))
    t0 = time.time()
    preds, stats = PI.run_eval(net, es, ks, k_seed=192, nms=5, tau_seed=0.3, tau_on=a.tau_on, wave_kw=dict(r_cov=1.2, r_head=1.2, order="near"), orient=True, verbose=False)
    gts = [es.gt[es.rows[k]] for k in ks]
    if a.keep:
        import pickle

        pickle.dump(dict(graphs=preds, gts=gts, ks=ks), open(f"{C.LADDER}/diag/{a.name}.{a.tag or (a.split + '_' + a.subset)}.pkl", "wb"))
    m = C.card(preds, gts)
    m["eval_seconds"] = time.time() - t0
    tag = a.tag or f"{a.split}_{a.subset}"
    run_id = f"{a.name}.{tag}"
    C.write_card(run_id, dict(run=a.name, kind="control", u=a.u, seed=a.seed, split=a.split, subset=a.subset, n=len(ks), tau_on=a.tau_on, metrics=m, stats=stats))
    print(run_id, json.dumps({k: round(v, 4) for k, v in m.items() if k in ("F1@0.5", "F1@1.0", "J", "Sp", "false_len_frac", "vert100", "S", "split6_F1", "merge6_F1", "lanes_found")}), flush=True)


if __name__ == "__main__":
    main()
