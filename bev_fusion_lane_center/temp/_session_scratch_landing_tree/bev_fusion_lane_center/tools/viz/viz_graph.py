"""Zeichnen von Spurmitten-Graphen (E27 ff.): Soll und Vorhersage im Fahrzeug-System (x rechts -> Bildachse waagerecht, y vorwaerts -> senkrecht), mit Pfeilen, Splits (Dreieck) und Merges (Raute).
Benutzung: import viz_graph as VG; VG.draw_graph(ax, g, color, ...); VG.panel(ax, g_pred, g_gt, title=...)
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import graph_eval as GE  # noqa: E402


def _deg(g):
    n = len(g["pts"])
    o, i = np.zeros(n, int), np.zeros(n, int)
    for a, b in g["edges"]:
        o[a] += 1
        i[b] += 1
    return o, i


def draw_graph(ax, g, color="r", lw=1.4, arrows=True, marks=True, alpha=1.0, zorder=3, label=None, link=True):
    """x (rechts) waagerecht, y (vorwaerts) senkrecht."""
    first = True
    for p in g["pts"]:
        if len(p) < 2:
            continue
        ax.plot(p[:, 0], p[:, 1], "-", color=color, lw=lw, alpha=alpha, zorder=zorder, label=label if first else None)
        first = False
        if arrows and len(p) >= 2:
            k = len(p) // 2
            a, b = p[max(k - 1, 0)], p[min(k + 1, len(p) - 1)]
            d = b - a
            if np.linalg.norm(d) > 1e-6:
                ax.annotate("", xy=(a + 0.5 * d + 0.02 * d / np.linalg.norm(d) * 5)[:2], xytext=a[:2] + 0.5 * d - 0.02 * d, arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=7), zorder=zorder + 1)
    if link:
        for i, j in g["edges"]:
            if len(g["pts"][i]) and len(g["pts"][j]):
                a, b = g["pts"][i][-1], g["pts"][j][0]
                if np.linalg.norm(a - b) > 0.3:
                    ax.plot([a[0], b[0]], [a[1], b[1]], ":", color=color, lw=lw, alpha=alpha, zorder=zorder)
    if marks:
        o, i = _deg(g)
        sp = [g["pts"][k][-1] for k in range(len(g["pts"])) if o[k] >= 2 and len(g["pts"][k])]
        mg = [g["pts"][k][0] for k in range(len(g["pts"])) if i[k] >= 2 and len(g["pts"][k])]
        if sp:
            sp = np.array(sp)
            ax.plot(sp[:, 0], sp[:, 1], "^", ms=7, mfc="none", mec=color, mew=1.6, zorder=zorder + 2)
        if mg:
            mg = np.array(mg)
            ax.plot(mg[:, 0], mg[:, 1], "D", ms=6, mfc="none", mec=color, mew=1.6, zorder=zorder + 2)


def setup_ax(ax, title="", extent=50):
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.25, lw=0.4)
    ax.plot([0], [0], "k^", ms=6, zorder=6)  # Ego
    ax.tick_params(labelsize=7)


def panel(ax, pred, gt, title="", background=None, bg_cmap="viridis"):
    """Vorhersage (rot) ueber Soll (gruen). background: [200,200] Karte (Zeile = x, Spalte = y) wird im selben System gezeigt."""
    if background is not None:
        ax.imshow(np.asarray(background, np.float32).T, origin="lower", extent=(-50, 50, -50, 50), cmap=bg_cmap, alpha=0.55, zorder=0)
    if gt is not None:
        draw_graph(ax, gt, color="#1a9850", lw=2.4, arrows=False, marks=True, alpha=0.8, zorder=2)
    if pred is not None:
        draw_graph(ax, pred, color="#d73027", lw=1.3, arrows=True, marks=True, zorder=4)
    setup_ax(ax, title)


def grid(preds, gts, ids, path, titles=None, backgrounds=None, ncol=4, figsize=3.4, metrics_line=None):
    n = len(preds)
    nrow = int(np.ceil(n / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(figsize * ncol, figsize * nrow + (0.3 if metrics_line else 0)))
    axs = np.atleast_1d(axs).ravel()
    for k in range(len(axs)):
        if k >= n:
            axs[k].axis("off")
            continue
        panel(axs[k], preds[k], gts[k], title=(titles[k] if titles else f"Sample {ids[k]}"), background=None if backgrounds is None else backgrounds[k])
    if metrics_line:
        fig.suptitle(metrics_line, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
