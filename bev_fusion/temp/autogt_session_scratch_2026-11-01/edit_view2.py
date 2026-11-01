p = "autogt/view.py"
s = open(p, encoding="utf-8").read()
s = s.replace("def figure(gt_gen, name, k, out_dir, total=False):", "def figure(gt_gen, name, k, out_dir, total=False, gt=None):\n    \"\"\"one sample figure; out_dir = None returns the matplotlib figure (the caller closes it: no file is written), else a png.\"\"\"")
s = s.replace("    gt = gt_gen.keyframe(tok)\n", "    gt = gt if gt is not None else gt_gen.keyframe(tok)\n", 1)
s = s.replace('''    p = os.path.join(out_dir, f"gt_sample_{name}_k{k:02d}.png")
    fig.savefig(p, dpi=72)
    plt.close(fig)
    print("saved", p, flush=True)
    return p''', '''    if out_dir is None:
        return fig
    p = os.path.join(out_dir, f"gt_sample_{name}_k{k:02d}.png")
    fig.savefig(p, dpi=72)
    plt.close(fig)
    print("saved", p, flush=True)
    return p''')
open(p, "w", encoding="utf-8").write(s)
print("ok")
