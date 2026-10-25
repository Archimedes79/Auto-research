"""Schleife 5 / E57: Dichter Kopf (Standard e37_final, trainiert auf sauberen Merkmalen) auf gestoerten Merkmalen der TEST-Zeilen (gcar_pert.py): Pixel-F1 bei festen tau (auf sauberem VAL gewaehlt) + Graph (strenger Tracer + Zusammenbau).
Schreibt out/loop2/e57_<stoerung>/{prob_val.npy, dir_val.npy, ids_val.npy, metrics.json} (wie ein dichter Lauf; damit laeuft auch das Gesamtsystem final_system.py --dense e57_<stoerung>).
Aufruf (gcar): eval_pert.py --pert dark|...|stored  [--head e37_final] [--workers 6]    stored = gespeicherte saubere Merkmale (Referenz)"""
import bflc_paths as BP  # project paths from the location of this checkout
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dense_head as DH  # noqa: E402
import graph_from_dense as GD  # noqa: E402

HOME, OUT = DH.HOME, DH.OUT


def load_net(run):
    ck = torch.load(f"{OUT}/{run}/model.pt", map_location="cuda", weights_only=False)
    a = ck["args"]
    net = DH.UNet(len(ck["mean"]), a["base"], 3 if a["dir"] > 0 else 1, a["drop"]).cuda()
    net.load_state_dict(ck["state"])
    net.eval()
    return net, ck["mean"].cuda(), ck["std"].cuda()


@torch.no_grad()
def infer(net, mean, std, dec, lg, bs=16):
    P, D = [], []
    for i in range(0, len(dec), bs):
        x = torch.cat([torch.from_numpy(np.asarray(dec[i : i + bs], np.float32)), torch.from_numpy(np.asarray(lg[i : i + bs], np.float32))], 1).cuda()
        x = (x - mean[None, :, None, None]) / std[None, :, None, None]
        with torch.autocast("cuda", dtype=torch.float16):
            o = net(x).float()
        P.append(torch.sigmoid(o[:, 0]).half().cpu())
        D.append(o[:, 1:3].half().cpu())
    return torch.cat(P), torch.cat(D)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pert", required=True)
    ap.add_argument("--head", default="e37_final")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--taus", default="0.5,0.6,0.7")
    ap.add_argument("--no_graph", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1, help="nur fuer 'stored': jede k-te TEST-Zeile (Vergleichsmenge der halbierten Stoerungen)")
    a = ap.parse_args()
    t0 = time.time()
    rows = np.array(json.load(open(BP.ROOT + "/data/splits.json"))["val"])
    stride = a.stride
    if a.pert != "stored":
        d = f"{HOME}/exp_gcar_b123/pert/{a.pert}"
        assert os.path.exists(f"{d}/DONE"), f"{a.pert}: Extraktion nicht fertig"
        pr = np.load(f"{d}/val_rows.npy")
        stride = 1
        rows = pr
    rows = rows[::stride]
    if a.pert == "stored":
        d = f"{HOME}/exp_gcar_b123/feats"
        dec, lg = np.load(f"{d}/val_dec32.npy", mmap_mode="r")[rows], np.load(f"{d}/val_logit7.npy", mmap_mode="r")[rows]
    else:
        dec, lg = np.load(f"{d}/val_dec32.npy", mmap_mode="r"), np.load(f"{d}/val_logit7.npy", mmap_mode="r")
    net, mean, std = load_net(a.head)
    prob, dirf = infer(net, mean, std, dec, lg)
    out_dir = f"{OUT}/e57_{a.pert}_valset" + ("_s2" if (a.pert == "stored" and stride == 2) else "")
    os.makedirs(out_dir, exist_ok=True)
    np.save(f"{out_dir}/prob_val.npy", prob.numpy())
    np.save(f"{out_dir}/dir_val.npy", dirf.numpy())
    np.save(f"{out_dir}/ids_val.npy", rows)
    gts = DH.load_gt_points("b123", "val", rows)
    taus = tuple(float(t) for t in a.taus.split(","))
    res = dict(pert=a.pert, head=a.head, n=len(rows), pixel=DH.eval_probs(prob, gts, taus=taus))
    print(f"[{a.pert}] Pixel-F1@1 m " + " ".join(f"tau {t}: {res['pixel'][str(t)]['F1@1.0']:.3f} (P {res['pixel'][str(t)]['P@1.0']:.3f} R {res['pixel'][str(t)]['R@1.0']:.3f})" for t in taus), flush=True)
    if not a.no_graph:
        ggt = GD.load_gt_graphs("val", rows)
        m, preds = GD.eval_variant(prob.float().numpy(), dirf.float().numpy(), ggt, "trace", {"t_seed": 0.9, "t_on": 1.1}, {"lead_m": 5.0}, a.workers, True)
        res["graph"] = {k: float(v) for k, v in m.items() if isinstance(v, (int, float, np.floating))}
        import pickle

        pickle.dump(dict(graphs=preds, ids=[int(i) for i in rows]), open(f"{out_dir}/graphs_val.pkl", "wb"))
        print(f"[{a.pert}] Graph: F1@1 {m['F1@1.0']:.3f} route F {m['route_f']:.3f} lanes found {m['lanes_found']:.3f} split6/merge6 {m['split6_F1']:.3f}/{m['merge6_F1']:.3f} dir {m.get('dir_acc', 0):.3f}", flush=True)
    json.dump(res, open(f"{out_dir}/metrics.json", "w"), indent=1)
    print(f"fertig {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
