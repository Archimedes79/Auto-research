"""E394 v1 design on TRAIN only (scratch): per-frame raw statistics of the 800 fit draws, then which label-free feature set and threshold make the gate reject the map of another scene
while trusting >= 97.5 % of the other draws; and the expected lanes_all of the gated head on the fit draws (wrong scene vs the rest) against z. No holdout, no VAL."""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("BFLC_ROOT") or os.path.normpath(os.path.join(HERE, "..", "..", "..", ".."))   # the problem folder (this copy lies in experiments/E394_*/scripts/e394/; it ran from temp/)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import maptrust_gate as G  # noqa: E402

torch.set_num_threads(2)
MEP = __import__("maperror_probe")
MP, enc, heads6, f1, z = G._models(0)
fr = MP.train_frames(0)
pick = sorted(np.random.default_rng(G.FIT_SEED).choice(fr, min(800, len(fr)), replace=False).tolist())
specs = {i: G.ME.train_spec(np.random.default_rng([394, i, 1])) for i in pick}
plan = MEP.plan_for("holdout", 0)


class FitData(MEP.ErrAssocData):
    def __getitem__(self, j):
        self.spec = specs[self.frames[j]]
        return super().__getitem__(j)


dl = torch.utils.data.DataLoader(FitData("train", 0, {}, frames=pick, plan=plan), batch_size=8, shuffle=False, num_workers=0, collate_fn=MEP.collate)
rows = []
with torch.no_grad():
    for b in dl:
        h, sv = MP.encode(enc, b, "cpu")
        z6 = heads6.assoc(h, b["pa"].float(), sv)
        o1, oz = f1(b, h, sv, b["pa"]), z(b)
        N = h.shape[1]
        for r in range(len(b["frame"])):
            k = int(b["frame"][r])
            kept = b["kept"][r].numpy().astype(bool)
            zl = oz["lanes"][r].argmax(-1).numpy()
            l1 = o1["lanes"][r].argmax(-1).numpy()
            x4 = G.frame_features(o1["assoc"][r].numpy(), MP.p6_assign(z6, r, N), zl, kept, b["lanes_in"][r].numpy(), b["seg_kind"][r].numpy())
            p6 = torch.softmax(z6[r].double(), -1).numpy()[kept]
            extra = [float((1 - p6[:, -1]).mean()) if len(p6) else 0.0, float(p6[:, :-1].max(-1).mean()) if len(p6) and p6.shape[1] > 1 else 0.0]
            la = b["lanes_all"][r].numpy()
            m = kept & (la >= 0)
            rows.append(dict(k=k, ws=int(bool(specs[k].get("wrong_scene"))), x=list(x4) + extra, n=int(m.sum()), c1=int((l1[m] == la[m]).sum()), cz=int((zl[m] == la[m]).sum())))
json.dump(rows, open(os.path.join(ROOT, "temp", "e394_smoke", "explore_rows_s0.json"), "w"))
X = np.array([r_["x"] for r_ in rows])
ws = np.array([r_["ws"] for r_ in rows])
n, c1, cz = (np.array([r_[k] for r_ in rows]) for k in ("n", "c1", "cz"))
harm = (c1 < cz).astype(int)
print(f"frames {len(rows)}, wrong scene {ws.sum()}, harm frames {harm.sum()} (of them wrong scene {int((harm & ws).sum())})")
for name, cols in (("4 feats", [0, 1, 2, 3]), ("4 + p6 mass/max", [0, 1, 2, 3, 4, 5]), ("p6 mass/max + agree", [3, 4, 5]), ("p6 rate only", [2]), ("p6 mass only", [4])):
    for lab_name, y in (("wrong_scene", ws), ("harm", harm)):
        f = G.fit_logistic(X[:, cols], y)
        p = G.gate_prob(f, X[:, cols])
        thr = np.quantile(p[ws == 0], G.TRUST_Q)                              # trust 97.5 % of the draws that are NOT the map of another scene
        ok = p <= thr
        acc = lambda sel, use: (np.where(use[sel], c1[sel], cz[sel]).sum() / max(n[sel].sum(), 1))  # noqa: E731
        lw = acc(ws == 1, ok)
        lr = acc(ws == 0, ok)
        zw, zr = cz[ws == 1].sum() / n[ws == 1].sum(), cz[ws == 0].sum() / n[ws == 0].sum()
        aw, ar = c1[ws == 1].sum() / n[ws == 1].sum(), c1[ws == 0].sum() / n[ws == 0].sum()
        print(f"{name:20s} label {lab_name:11s}: AUROC(ws) {G.roc_auc(ws, p) if hasattr(G, 'roc_auc') else __import__('sklearn.metrics', fromlist=['roc_auc_score']).roc_auc_score(ws, p):.3f} "
              f"rejects ws {1 - ok[ws == 1].mean():.3f}, trusts rest {ok[ws == 0].mean():.3f} | lanes on ws: g {lw:.3f} z {zw:.3f} a {aw:.3f} (g - z {lw - zw:+.3f}) | rest: g {lr:.3f} z {zr:.3f} a {ar:.3f}")
