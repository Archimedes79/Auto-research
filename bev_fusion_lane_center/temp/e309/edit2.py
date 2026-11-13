p = "temp/e309/render.py"
s = open(p).read()
s = s.replace('(("B", "B (E273 + hard step)"), ("e284", "B + E284 end head"), ("e287p2", "B + E287 p2 pair head"))', '(("B", "B (E273 + hard step) = the plain graph"), ("e312", "E312 final system (E307 maps + pair head + snapped p>=0.1)"), ("e287p2", "B + E287 p2 pair head (old)"))')
s = s.replace("gs = fig.add_gridspec(1, 4, width_ratios=[1.25, 1, 1, 1], wspace=0.08)", "gs = fig.add_gridspec(1, 4, width_ratios=[1.25, 1, 1, 1], wspace=0.08)")
# arrows on delivered lines every 15 m
s = s.replace("    ax.plot([0], [0], marker=\"^\", color=\"k\", ms=7, zorder=6)", "    for q in lines:\n        for m in range(15, len(q) - 1, 30):\n            ax.annotate(\"\", xy=q[m + 1], xytext=q[m], arrowprops=dict(arrowstyle=\"-|>\", color=\"#0b4d12\", lw=1.0), zorder=5)\n    ax.plot([0], [0], marker=\"^\", color=\"k\", ms=7, zorder=6)")
open(p, "w").write(s)
