"""scratch: side-by-side pictures of the same VAL frames for several lane-head variants (CPU), for the morning review: B plain | B + hard close-node step (E275) | the head trained with the soft step (E277) | ...
    wsl env CUDA_VISIBLE_DEVICES= bash tools/env/_py14.sh compare_views.py <out dir> <frame indices comma separated> <variant> [<variant> ...]
A variant is `<run>` (plain read-out at 0.5) or `<run>+close` (the hard step 1.5 m on top), e.g. e273_p_s0 e273_p_s0+close e277_u_s0. Each panel is the scorer's figure of `qgraph_view.render_frame` without the cameras
(delivered nodes / links against the GT graph, false = red halo, missed = blue halo; the scorer's numbers in the title of the frame figure)."""
import os
import sys

import numpy as np

ROOT = os.environ["BFLC_ROOT"]
for sub in ("training", "training/grid", "training/data", "evaluation", "tools/viz"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import qgraph_eval as QE  # noqa: E402
import qgraph_readout as RO  # noqa: E402
import qgraph_view as QV  # noqa: E402
import qpredict as QP  # noqa: E402

out_dir, idx, variants = sys.argv[1], [int(x) for x in sys.argv[2].split(",")], sys.argv[3:]
os.makedirs(out_dir, exist_ok=True)
frames = QE.val_frames()
fns = {}
for v in variants:
    run = v.split("+")[0]
    if run not in fns:
        fns[run] = QP.make_pred_fn(run, device="cpu", keep_maps=True)
spec = RO.parse("close:1.5@0.5")
for i in idx:
    tok = frames[i].token
    panels = []
    for v in variants:
        pred = fns[v.split("+")[0]](tok)
        if v.endswith("+close"):
            pred = RO.apply(pred, spec)
        arr = QV.render_frame(tok, pred, None, cams=False, note=v)
        panels.append((v, arr))
    fig, ax = plt.subplots(len(panels), 2, figsize=(14, 6.6 * len(panels)), dpi=75)
    ax = np.atleast_2d(ax)
    for r, (v, arr) in enumerate(panels):
        H, W = arr.shape[:2]
        ax[r, 0].imshow(arr[int(0.03 * H):int(0.52 * H), int(0.655 * W):])                       # the nodes panel: delivered nodes and links, red halo = false, blue halo = missed
        ax[r, 1].imshow(arr[int(0.52 * H):int(0.96 * H), int(0.33 * W):int(0.655 * W)])         # the delivered graph in the colour of the assigned map piece
        ax[r, 0].set_title(f"{v}  frame {i}: nodes / links against the GT", fontsize=12)
        ax[r, 1].set_title(f"{v}: delivered graph, colour = assigned map piece", fontsize=12)
        ax[r, 0].axis("off")
        ax[r, 1].axis("off")
    fig.tight_layout()
    path = os.path.join(out_dir, f"frame_{i:03d}.png")
    fig.savefig(path, dpi=75)
    plt.close(fig)
    print("written", path, flush=True)
