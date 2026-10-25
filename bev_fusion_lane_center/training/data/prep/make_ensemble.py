"""Loop 3, E37: Mittel der dichten Koepfe (Endkopf + drei Kreuz-Anpassungs-Koepfe) auf VAL/TEST: Heatmap und Richtungsfeld gemittelt -> out/loop2/e37_ens (wie ein dichter Lauf: prob_/dir_/ids_ + metrics.json mit Pixel-Kennzahlen).
Aufruf (gcar): make_ensemble.py [lauf1,lauf2,...] [name]"""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dense_head as DH

OUT = DH.OUT
runs = (sys.argv[1] if len(sys.argv) > 1 else "e37_final,e37_fold0,e37_fold1,e37_fold2").split(",")
name = sys.argv[2] if len(sys.argv) > 2 else "e37_ens"
os.makedirs(f"{OUT}/{name}", exist_ok=True)
res = dict(runs=runs)
for st, part in (("val", "val"), ("test", "val")):
    P, D, ids = [], [], None
    for r in runs:
        P.append(np.load(f"{OUT}/{r}/prob_{st}.npy").astype(np.float32))
        D.append(np.load(f"{OUT}/{r}/dir_{st}.npy").astype(np.float32))
        i = np.load(f"{OUT}/{r}/ids_{st}.npy")
        assert ids is None or (ids == i).all()
        ids = i
    prob = np.mean(P, 0)
    dirf = np.mean(D, 0)
    np.save(f"{OUT}/{name}/prob_{st}.npy", prob.astype(np.float16))
    np.save(f"{OUT}/{name}/dir_{st}.npy", dirf.astype(np.float16))
    np.save(f"{OUT}/{name}/ids_{st}.npy", ids)
    gts = DH.load_gt_points("b123", part, ids)
    res[st] = DH.eval_probs(torch.from_numpy(prob).half(), gts, taus=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8))
    best = max(res[st].items(), key=lambda kv: kv[1]["F1@1.0"])
    print(f"[{st}] Mittel aus {len(runs)} Koepfen: bester tau {best[0]}: F1@0.5/1/1.5 {best[1]['F1@0.5']:.3f}/{best[1]['F1@1.0']:.3f}/{best[1]['F1@1.5']:.3f} P/R@1 {best[1]['P@1.0']:.3f}/{best[1]['R@1.0']:.3f}", flush=True)
json.dump(res, open(f"{OUT}/{name}/metrics.json", "w"), indent=1)
