#!/usr/bin/env python3
"""plots: one clear chart for a report, from a small json (no numbers typed into the picture by hand).

    python plots.py bars spec.json out.png

spec.json: {"title": "...", "xlabel": "topo F1 (VAL)", "xmin": 0.5, "xmax": 0.72,
            "series": ["seed 0", "seed 1"],
            "rows": [{"label": "plain lane head B", "values": [0.6259, 0.6249]},
                     {"label": "token head S1", "values": [0.5486, 0.5743]},
                     {"label": "champion", "values": null}],              # null = PENDING (a hatched empty bar)
            "lines": [{"x": [0.6359, 0.6349], "label": "gate floor"}]}     # optional reference lines (one x per series, or one number)
Horizontal bars, one group per row, one bar per series, the value written at the end of each bar, pending rows hatched, no decoration.
Needs matplotlib. Standalone."""
import json, sys


def bars(spec, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows, series = spec["rows"], spec.get("series", ["s0", "s1"])
    cols = ["#1f5fa8", "#7fa7d6", "#b07a00", "#2f6b3c"][:len(series)]
    n, k = len(rows), len(series)
    h = 0.8 / k
    fig, ax = plt.subplots(figsize=(8.4, 0.55 * n * k * 0.62 + 1.6), dpi=150)
    for i, r in enumerate(rows):
        y0 = n - 1 - i
        for j, s in enumerate(series):
            y = y0 + 0.4 - h * (j + 0.5)
            v = None if r["values"] is None else r["values"][j]
            if v is None:                                                     # pending: a short hatched stub, never a long bar that looks like a result
                stub = (spec["xmax"] - spec["xmin"]) * 0.06
                ax.barh(y, stub, left=spec["xmin"], height=h * 0.85, fill=False, hatch="////", edgecolor="#b07a00", linewidth=0.6)
                ax.text(spec["xmin"] + stub + 0.001, y, "PENDING", va="center", fontsize=7, color="#7a5a00")
            else:
                ax.barh(y, v - spec["xmin"], left=spec["xmin"], height=h * 0.85, color=cols[j], label=s if i == 0 else None)
                ax.text(v - 0.0012, y, "%.4f" % v, va="center", ha="right", fontsize=7, color="white")   # inside the bar: nothing crosses the reference line
    for ln in spec.get("lines", []):
        x = max(ln["x"]) if isinstance(ln["x"], list) else ln["x"]        # one line: the larger floor of the seeds
        ax.axvline(x, color="#444", ls=(0, (4, 3)), lw=0.9, zorder=0)
        ax.text(x, n - 0.25, ln["label"], fontsize=7, color="#444", ha="center", va="bottom")
    ax.set_yticks([n - 1 - i for i in range(n)])
    ax.set_yticklabels([r["label"] for r in rows], fontsize=8)
    ax.set_xlim(spec["xmin"], spec["xmax"])
    ax.set_ylim(-0.6, n - 0.2 + (0.35 if spec.get("lines") else 0))
    ax.set_xlabel(spec.get("xlabel", ""), fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title(spec.get("title", ""), fontsize=9, loc="left")
    ax.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=len(series))
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "bars":
        sys.exit(__doc__)
    bars(json.load(open(sys.argv[2], encoding="utf-8")), sys.argv[3])
    print(sys.argv[3])
