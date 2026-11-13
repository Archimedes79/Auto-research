"""E334 (S5): a maps checkpoint with the pyramid of the baseline and an UNTRAINED junction head (the seeded initialisation `jmaps_train.py` starts from), so that `jends_train.py --e2e --maps <this>` trains the head from scratch together
with the pair head in ONE stage. Writes runs/e334_init_s<seed>/model.pt (state + args with ctx / d / base / seed, as jmaps_train.py writes them). usage: wsl bash tools/env/_py14.sh e334/make_init.py <seed>"""
import os
import sys

import torch

ROOT = os.environ["BFLC_ROOT"]
for sub in ("training", "training/grid", "training/data", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import jmaps as JM  # noqa: E402
import trainer as T  # noqa: E402

seed = int(sys.argv[1])
T.seed_all(seed)
model = JM.JModel(width=48, d=64, ctx="dil")
base = os.path.join(ROOT, "runs", f"e273_p_s{seed}", "model.pt")
n, tot = model.load_base(base)
out = os.path.join(ROOT, "runs", f"e334_init_s{seed}")
os.makedirs(out, exist_ok=True)
args = dict(name=f"e334_init_s{seed}", seed=seed, base=base, ctx="dil", d=64, joint=False, sigma=1.0, raster="legacy", far_w=0.0, dirx=False, no_paint=False, no_zone=False, steps=0, lr=1e-3, bs=8, workers=3, smoke=False)
torch.save(dict(state=model.state_dict(), args=args), os.path.join(out, "model.pt"))
print(f"written {out}/model.pt: pyramid tensors loaded {n} of {tot} from {base}, the junction head is untrained")
