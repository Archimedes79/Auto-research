"""Loop 16: re-evaluate a trained design checkpoint (assembly or threshold changes without retraining).
Usage (gcar, GPU): p16_eval.py --design D3 --name e123_D3_r0_00 [--eval val250|val] [--tag reeval]  -> card <name>[.tag].json"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p16_common as C  # noqa: E402
import p16_data as PD  # noqa: E402
import p16_infer as PI  # noqa: E402
import p16_train as PT  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--design", required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--eval", default="val250")
ap.add_argument("--split", default="val_clean")
ap.add_argument("--tag", default="")
ap.add_argument("--set", nargs="*", default=[])
ap.add_argument("--workers", type=int, default=4)
a = ap.parse_args()
_card = f"{C.LADDER}/cards/{a.name}.json"
_extra = dict(json.load(open(_card, encoding="utf-8-sig"))["cfg"]) if os.path.exists(_card) else {}
_extra.update(PT.parse_set(a.set))
net, cfg = PT.build(a.design, _extra)
net = net.cuda()
net.load_state_dict(torch.load(f"{C.OUT}/e123/{a.name}/model.pt", map_location="cuda"), strict=False)  # (cfg of the run: pass its --set overrides)
ks = C.val_subset() if (a.eval == "val250" and a.split == "val_clean") else None
ds = PD.BuilderData(a.split, ks=ks, train=False, with_prev=cfg.temporal)
t0 = time.time()
graphs, gts, sec = PI.predict(net, ds, ks, workers=a.workers)
ev = PI.evaluate_sweep(graphs, gts)
m = ev["best"]
card = dict(run=a.name, kind="design-reeval", design=a.design, cfg=cfg.asdict(), eval=a.eval, split=a.split, n=len(gts), metrics=m, metrics_tau05=ev["at05"], sweep=ev["sweep"], eval_seconds=time.time() - t0, **{k: ev[k] for k in ("F1@Sp17", "FL@R85")})
C.write_card(a.name + (("." + a.tag) if a.tag else ".reeval"), card)
print(a.name, f"S {m['S']:.4f} tau {m['tau']} F1@1 {m['F1@1.0']:.4f} F1@.5 {m['F1@0.5']:.4f} J {m['J']:.3f} (split {m['split6_F1']:.3f} merge {m['merge6_F1']:.3f}) Sp {m['Sp']:.3f} v100 {m['vert100']:.1f} FL {m['false_len_frac']:.3f} dir {m.get('dir_acc', 0):.3f}")
for t, s in ev["sweep"].items():
    print(" ", t, {k: round(v, 3) for k, v in s.items() if k in ("S", "F1@1.0", "J", "Sp", "vert100", "R@1.0", "P@1.0", "split6_F1", "merge6_F1")})
