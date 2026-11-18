"""scratch (E379 v2, Fable's measurement): the real teachers' own GT loss (fp32, eval, none weight 0.25) on the student's first 5 TRAIN batches (cache seed 0, head seed 2, bs 8: E373 cb's order), the student's init loss on
batch 1 against E373's logged first loss 4.5516, the init hash; CPU only"""
import hashlib
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
for sub_ in ("training", "training/grid", "training/train", "training/data", "training/data/prep", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub_))
import torch  # noqa: E402
torch.set_num_threads(2)
import jtoken as JK  # noqa: E402
import jtoken_train as JKT  # noqa: E402
import trainer as T  # noqa: E402
import lane_ptok_data as LP  # noqa: E402

t0 = time.time()
ea = torch.load(os.path.join(ROOT, "runs/e373_cb_s0/model.pt"), map_location="cpu", weights_only=False)["args"]
teachers = JKT.load_teachers(dict(distill_from="runs/e373_cb_s0,runs/e367_b25_s0", seed=0, ident="e292_a", K=None), "cpu")
print("teacher steps", [m._step_py for m in teachers], "args steps", ea["steps"])
T.seed_all(2)
_, loader = LP.loader("train", 0, 8, 0, ident="e292_a")
m = JKT.build_head(ea)
h = hashlib.sha256(b"".join(v.detach().cpu().numpy().tobytes() for v in m.state_dict().values())).hexdigest()
T.seed_all(2)
it = iter(loader)
bs = [next(it) for _ in range(5)]
print("first batch tokens", list(bs[0]["token"]))
with torch.no_grad():
    l0 = float(JK.jtoken_loss(m(bs[0]), bs[0], none_weight=0.25)[0])
    gl = [[float(JK.jtoken_loss(t(b), b, none_weight=0.25)[0]) for b in bs] for t in teachers]
print(f"student init hash {h[:16]}, init GT loss on batch 1 (CPU fp32) {l0:.4f} (E373 logged 4.5516)")
for n, g in zip(("A e373_cb_s0", "B e367_b25_s0"), gl):
    print(n, "per batch", " ".join(f"{v:.4f}" for v in g), f"mean of 5 {sum(g) / 5:.4f}")
print(f"{time.time() - t0:.0f} s")
