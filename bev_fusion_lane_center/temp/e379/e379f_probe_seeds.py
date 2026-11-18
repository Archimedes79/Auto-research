"""scratch (E379 round 2, Fable's measurement): per cache seed S in 0, 1 the real teachers e373_cb_s<S> / e367_b25_s<S> through the live trainer's own start-check path (`load_teachers`, `first_batches` with the head seed of
e373_cb_s<S>, `teacher_start_loss`: float32, eval, none weight 0.25) on the first 20 TRAIN batches (bs 8): the first batch and the 20-batch mean; CPU only"""
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
import jtoken_train as JKT  # noqa: E402
import lane_ptok_data as LP  # noqa: E402

for s in (0, 1):
    t0 = time.time()
    runs = [f"runs/e373_cb_s{s}", f"runs/e367_b25_s{s}"]
    ea = torch.load(os.path.join(ROOT, runs[0], "model.pt"), map_location="cpu", weights_only=False)["args"]
    hs = ea["seed"] if ea.get("head_seed") is None else ea["head_seed"]
    teachers = JKT.load_teachers(dict(distill_from=",".join(runs), seed=s, ident="e292_a", K=None, cache=None), "cpu")
    _, loader = LP.loader("train", s, 8, 0, ident="e292_a")
    bs = JKT.first_batches(loader, None, hs)
    tl = JKT.teacher_start_loss(teachers, bs, "cpu", none_weight=0.25, pos_loss="sum", none_weight_near=None)
    print(f"cache seed {s}, head seed {hs}, first batch {list(bs[0]['token'])[:2]}..., step counters {[t._step_py for t in teachers]}")
    for r, row in zip(runs, tl):
        print(f"  {r}: first batch {row[0]:.4f}, first 5 {' '.join(f'{v:.4f}' for v in row[:5])}, mean of 20 {row.mean():.4f} (one batch {row.min():.4f} - {row.max():.4f})")
    print(f"  {time.time() - t0:.0f} s")
