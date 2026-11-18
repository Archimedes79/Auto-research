"""Loss curves of one or several runs on one page (needs matplotlib; the rest of the suite does not)."""
import math


def moving_mean(v, n):
    out, acc = [], 0.0
    for i, x in enumerate(v):
        acc += x
        if i >= n:
            acc -= v[i - n]
        out.append(acc / min(i + 1, n))
    return out


def default_keys(run, limit=7):
    ks = [run.loss_key] + [k for k in run.keys() if k.startswith("group.")]
    ks += [k for k in ("grad_norm",) if k in run.keys()]
    return ks[:limit]


def plot(runs, keys, out, smooth=10):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = 2 if len(keys) > 1 else 1
    rows = math.ceil(len(keys) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6.2 * cols, 2.9 * rows), squeeze=False)
    colours = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b"]
    for ax, key in zip(axes.flat, keys):
        for i, run in enumerate(runs):
            data = [(s, v) for s, v in run.series(key) if math.isfinite(v)]
            if not data:
                continue
            xs, ys = zip(*data)
            c = colours[i % len(colours)]
            if key.startswith("val."):
                ax.plot(xs, ys, "o-", color=c, label=run.name)
            else:
                ax.plot(xs, ys, color=c, alpha=0.18, lw=0.8)
                ax.plot(xs, moving_mean(list(ys), smooth), color=c, lw=1.6, label=run.name)
        ax.set_title(key, fontsize=9)
        ax.set_xlabel("step", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    for ax in list(axes.flat)[len(keys):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=7, frameon=False)
    fig.suptitle(f"training curves (moving mean of {smooth} logged rows; faint = raw)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out
