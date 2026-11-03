"""Spec test of the grid head (E271, G1): real VAL frames, CPU. Gradient from the total loss reaches every parameter group, the loss falls when one batch is overfitted,
the old dense-head channels are not read, the targets and the read-out agree on the frame.

    wsl bash tools/env/_py14.sh training/grid/test_grid.py
"""
import os
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for sub in ("training/grid", "evaluation", "training/data"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import data as GD  # noqa: E402
import head as GH  # noqa: E402
import losses as GL  # noqa: E402
import soft_lines as SL  # noqa: E402
import vertex_eval as VE  # noqa: E402
from lane_task import measures as M  # noqa: E402


def main():
    torch.manual_seed(0)
    ds = GD.GridData("val_clean", ks=[3, 400])
    b = GD.collate([ds[0], ds[1]])
    assert b["x"].shape == (2, 39, 200, 200) and b["g05"].shape[2] == 2 and b["m05"].sum() > 100, (b["x"].shape, b["g05"].shape)
    assert float(b["g05"][b["m05"] > 0].abs().max()) < 48.0, "GT points must lie in the scored box"
    # the same frame, the old way: BuilderData channels 0..38 are identical to the new loader's (one source of truth for the features)
    import p16_data as PD
    old = PD.BuilderData("val_clean", train=False)[3]["x"][:39].float()
    assert torch.equal(old, b["x"][0].float()), float((old - b["x"][0].float()).abs().max())
    # the targets decode to the lane: centre + offset is a GT point itself, for both axes (a sign error of one offset axis stays under 1 m and would go unseen by the metrics)
    import targets as TG
    d, off, head, _ = TG.cell_targets(b["g05"], b["t05"])
    c = TG.centres().view(100, 100, 2)
    pts = c + off[0].permute(1, 2, 0)                                           # (100, 100, 2) nearest lane point of every cell, metres
    gp = b["g05"][0][b["m05"][0] > 0]
    sel = (d[0] < 2.0).nonzero()[:200]
    q = pts[sel[:, 0], sel[:, 1]]
    assert float(torch.cdist(q, gp, compute_mode="donot_use_mm_for_euclid_dist").min(1).values.max()) < 1e-3, "centre + offset must be a ground-truth point"
    m = GH.GridHead(width=16)
    x, bt = b["x"].float(), {k: b[k] for k in ("g05", "t05", "m05")}
    out = m(x)
    loss, info = GL.grid_loss(out, bt)
    assert torch.isfinite(loss), info
    loss.backward()
    groups = {}
    for n_, p in m.named_parameters():
        g = n_.split(".")[0]
        groups[g] = groups.get(g, 0.0) + (0.0 if p.grad is None else float(p.grad.abs().sum()))
    print("G1 gradient per group", {k: f"{v:.2e}" for k, v in groups.items()})
    assert all(v > 0 for v in groups.values()), "a parameter group without gradient"
    # overfit one batch: the loss must fall clearly
    m = GH.GridHead(width=16)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    first = None
    for it in range(40):
        opt.zero_grad()
        loss, info = GL.grid_loss(m(x), bt)
        loss.backward()
        opt.step()
        first = first if first is not None else float(loss)
    print("overfit 40 steps: loss", round(first, 3), "->", round(float(loss), 3), {k: round(v, 3) for k, v in info.items()})
    assert float(loss) < 0.8 * first, "the loss does not fall on one batch"
    # read-out of the overfitted head on its own frame gives vertices near the lane
    with torch.no_grad():
        sp = SL.sparsify(m(x))
    V, H, w = VE.read_out(sp[0], 0.2)
    G = b["g05"][0][b["m05"][0] > 0].numpy().astype(np.float64)
    T = b["t05"][0][b["m05"][0] > 0].numpy().astype(np.float64)
    num = M.vertex_numbers(M.vertex_counts(V, H, G, T))
    print("read-out after overfitting:", {k: round(v, 3) for k, v in num.items()}, "vertices", len(V))
    assert len(V) > 0 and num["prec1"] > 0.6, num
    print("grid spec test ok")


if __name__ == "__main__":
    main()
