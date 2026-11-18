"""Loop 18 figure: the systems on the validation frames side by side (S, Q, J, lanes found) - control E94, champion f1, champion + fork pruning, single-frame j1 + pruning, recall variants rc1 / rc2 (+ pruning).
Reads evaluation/examples/result.json, runs/e18/fork_prune_val/results.json, runs/e18/rc_eval/*.json.  Writes reports/source/figures/p18_final_val.png.  Usage (gcar): fig_p18_final.py"""
import bflc_paths as BP  # project paths from the location of this checkout
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

R = BP.ROOT
ex = json.load(open(f"{R}/evaluation/examples/result.json"))
fp = json.load(open(f"{R}/runs/e18/fork_prune_val/results.json"))["cards"]
rcd = {n: json.load(open(f"{R}/runs/e18/rc_eval/{n}.json")) for n in ("rc1", "rc2", "j1p", "rc4") if os.path.exists(f"{R}/runs/e18/rc_eval/{n}.json")}
rows = [("control E94", ex["control"]), ("champion f1", ex["champion"]), ("f1 + pruning", fp["prune 0.3"])]
for nm, lab in (("j1p", "j1 192/96 + pruning"), ("rc1", "rc1 + pruning"), ("rc2", "rc2 + pruning"), ("rc4", "rc4 + pruning")):
    if nm in rcd:
        rows.append((lab, rcd[nm]))
keys = (("S", "S (composite)", "#c9a441"), ("Q", "Q (quality, no sparsity term)", "#2a9d4a"), ("J", "J (split / merge F1)", "#1f5fa8"), ("lanes_found", "lanes found", "#a85f9e"))
fig, ax = plt.subplots(1, 1, figsize=(11.5, 4.3))
w = 0.2
for k, (key, lab, col) in enumerate(keys):
    ax.bar(np.arange(len(rows)) + (k - 1.5) * w, [r[1][key] for r in rows], w, color=col, label=lab)
    for i, r in enumerate(rows):
        ax.text(i + (k - 1.5) * w, r[1][key] + 0.008, f"{r[1][key]:.2f}", ha="center", fontsize=6.5, rotation=90)
ax.set_xticks(np.arange(len(rows)))
ax.set_xticklabels([f"{r[0]}\n{r[1]['vert100']:.1f} vertices / 100 m" for r in rows], fontsize=8)
ax.set_ylim(0, 0.95)
ax.legend(fontsize=8, frameon=False, ncol=4, loc="upper center")
ax.set_title("Validation, 873 frames: the systems of loop 18 against the control", fontsize=10, loc="left")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(f"{R}/reports/source/figures/p18_final_val.png", dpi=140)
print("saved")
