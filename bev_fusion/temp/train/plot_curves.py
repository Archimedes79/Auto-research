"""One-off (report of experiment t1): training curves of several runs from runs/<name>/log.jsonl -> one PNG.
Panels: train loss per task group (moving mean over 5 records), validation total, Kendall / UW-SO s per group, gradient
norm per group at the BEV feature (diagnostic every 500 steps).
Usage: python temp/train/plot_curves.py out.png run1 [run2 ...]"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

out, runs = sys.argv[1], sys.argv[2:]
GROUPS = ("areas", "lines", "objects", "motion")
fig, axs = plt.subplots(2, 3, figsize=(16, 8.5))
styles = ["-", "--", ":", "-."]
for ri, run in enumerate(runs):
    recs = [json.loads(line) for line in open(f"runs/{run}/log.jsonl")]
    tr = [r for r in recs if "val" not in r and "total" in r]
    va = [r for r in recs if "val" in r]
    st = np.array([r["step"] for r in tr])
    ls = styles[ri % len(styles)]

    def smooth(y, k=5):
        y = np.asarray(y, float)
        if len(y) <= k:
            return y
        return np.convolve(y, np.ones(k), mode="same") / np.convolve(np.ones_like(y), np.ones(k), mode="same")
    for gi, g in enumerate(GROUPS):
        ax = axs[0, 0] if g in ("areas", "lines", "motion") else axs[0, 1]
        ax.plot(st, smooth([r.get("group." + g, np.nan) for r in tr]), ls, color=f"C{gi}", lw=1.2,
                label=f"{g} ({run})")
        axs[1, 0].plot(st, [r.get("s." + g, np.nan) for r in tr], ls, color=f"C{gi}", lw=1.2, label=f"{g} ({run})")
    if va:
        axs[0, 2].plot([r["step"] for r in va], [r["val"]["total"] for r in va], ls + "o", color="k", ms=4, label=f"val total ({run})")
        for gi, g in enumerate(GROUPS):
            axs[0, 2].plot([r["step"] for r in va], [r["val"].get("group." + g, np.nan) for r in va], ls + ".",
                           color=f"C{gi}", ms=4, lw=0.8)
    dg = [r for r in tr if "diag" in r]
    for gi, g in enumerate(GROUPS):
        y = [dict(zip(r["diag"]["groups"], r["diag"]["norm"])).get(g, np.nan) for r in dg]
        axs[1, 1].semilogy([r["step"] for r in dg], y, ls + "o", color=f"C{gi}", ms=4, label=f"{g} ({run})")
    axs[1, 2].plot(st, smooth([r["grad_norm"] for r in tr]), ls, color="0.3", lw=1, label=f"grad norm ({run})")
titles = ["train loss: areas, lines, motion (dt-weighted)", "train loss: objects", "validation loss (200 samples, EMA)",
          "task weights s (weight = exp(-s))", "gradient norm per group at the BEV feature", "total gradient norm (clip 5)"]
for ax, t in zip(axs.flat, titles):
    ax.set_title(t, fontsize=10)
    ax.set_xlabel("optimiser step", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6, ncol=2)
for ax in (axs[0, 0], axs[0, 1], axs[0, 2]):
    ax.axvspan(0.2 * 5000, 0.4 * 5000, color="0.9", zorder=0)      # dt != 0 terms ramp in (warmup_hold, warmup_ramp)
fig.suptitle("training curves: " + ", ".join(runs) + " (grey band: the dt = +-0.5 s terms ramp in)", fontsize=11)
fig.tight_layout()
fig.savefig(out, dpi=90)
print("wrote", out)
