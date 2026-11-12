p = "temp/jr/fail_gallery.py"
s = open(p, encoding="utf-8").read()
i = s.index("# ---------------------------------------------------------------------------------------------- error videos")
head = s[:i]
tail = '''# ---------------------------------------------------------------------------------------------- contact sheet: 12 random junction frames, the delivered graph against GT
def sheet(name, sel, ncol=4):
    nrow = (len(sel) + ncol - 1) // ncol
    fig, axs = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 5.4 * nrow), squeeze=False)
    for ax in axs.ravel():
        ax.axis("off")
    for ax, r in zip(axs.ravel(), sel):
        ax.axis("on")
        f = frames[r["k"]]
        ref = f.ref()
        P, hp, G, cg = cover(raws[r["k"]], ref)
        ax.scatter(G[~cg, 0], G[~cg, 1], s=2, color="#1f77b4", zorder=2)
        ax.scatter(P[hp, 0], P[hp, 1], s=2, color="#2ca02c", zorder=3)
        ax.scatter(P[~hp, 0], P[~hp, 1], s=3, color="#d62728", zorder=4)
        ax.set_title(f"{f.token[:8]}: coverage {r['recall_head']:.2f}, precision {r['precision_head']:.2f}, pairs {r['n_true']}/{r['n_gt_pairs']}", fontsize=8)
        ax.set_xlim(-BOX, BOX)
        ax.set_ylim(-BOX, BOX)
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
    fig.suptitle("12 random junction frames (>= 6 GT pairs): green = delivered near GT, red = delivered without GT, blue = GT not covered", fontsize=10)
    fig.tight_layout()
    path = os.path.join(OUT, f"{name}.png")
    fig.savefig(path, dpi=70)
    plt.close(fig)
    print("wrote", path)


rng = np.random.default_rng(1)
pool = [r for r in rec if r["n_gt_pairs"] >= 6]
sheet("random12", [pool[i] for i in sorted(rng.choice(len(pool), 12, replace=False))])


# ---------------------------------------------------------------------------------------------- error videos (same three panels per keyframe of a scene)
def video(name, fps=2):
    import cv2
    sid, nm, toks = QV.resolve_scene(name)
    rows = [r for r in rec if r["token"] in set(toks)]
    if not rows:
        print("no frames for", name)
        return
    path = os.path.join(OUT, f"{nm}_errors.webm")
    wr = None
    for r in rows:
        fig, axs = plt.subplots(1, 3, figsize=(16.5, 5.6))
        draw_frame(axs, r)
        fig.tight_layout()
        fig.canvas.draw()
        img = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba())[:, :, :3], cv2.COLOR_RGB2BGR)
        plt.close(fig)
        w = 1400
        h = int(round(img.shape[0] * w / img.shape[1] / 2) * 2)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        if wr is None:
            wr = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"vp09"), fps, (w, h))
        wr.write(img)
    wr.release()
    print("wrote", path, len(rows), "frames", round(os.path.getsize(path) / 1e6, 2), "MB")


for nm in ("scene-0096", "scene-0221", "scene-0017", "scene-0108"):
    video(nm)
print("videos done")
'''
open(p, "w", encoding="utf-8").write(head + tail)
print("ok")
