p = "temp/e276/compare_views.py"
s = open(p, encoding="utf-8").read()
a = s.index("    h = max(a.shape[0]")
b = s.index("    path = os.path.join(out_dir")
new = '''    fig, ax = plt.subplots(len(panels), 2, figsize=(14, 6.6 * len(panels)), dpi=75)
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
'''
s = s[:a] + new + s[b:]
s = s.replace("fig.savefig(path, dpi=80)", "fig.savefig(path, dpi=75)")
open(p, "w", encoding="utf-8").write(s)
print("patched")
