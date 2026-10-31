"""scratch (gallery of the GT quality overview): draw the picked keyframes from the STORED GT.

  render.py SIGNATURE picks.json   -> temp/agt_gallery/pics/<scene>_k<kk>.png (4-panel figure of autogt.view, dpi 80)
                                      and temp/agt_gallery/thumbs_<n>.png (BEV thumbnails of the 'thumbs' list)

picks.json: {"examples": [{"scene", "k", "tok", "title", "note"}], "thumbs": [{"scene", "k", "tok"}]}
No nuScenes devkit: the figure needs only the scene list, the keyframe order and the dataroot.
"""
import json
import os
import sys
import types

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, ".")
from autogt import view  # noqa: E402
from autogt.cli import load_keyframe  # noqa: E402

sig, picks_file = sys.argv[1], sys.argv[2]
root = os.path.expanduser("~/data/nuscenes_full")
tab = os.path.join(root, "v1.0-trainval")
scenes = json.load(open(os.path.join(tab, "scene.json")))
samples = {r["token"]: r for r in json.load(open(os.path.join(tab, "sample.json")))}


class Stub:
    """what autogt.view.figure needs from the generator."""
    dataroot = root
    nusc = types.SimpleNamespace(scene=scenes)

    def keyframes(self, name):
        sc = next(s for s in scenes if s["name"] == name)
        out, tok = [], sc["first_sample_token"]
        while tok:
            out.append(tok)
            tok = samples[tok]["next"]
        return out


D = os.path.join("data", "autogt", sig)
picks = json.load(open(picks_file))
os.makedirs("temp/agt_gallery/pics", exist_ok=True)
stub = Stub()
for p in picks["examples"]:
    gt = load_keyframe(D, p["tok"])
    assert stub.keyframes(p["scene"])[p["k"]] == p["tok"], p
    fig = view.figure(stub, p["scene"], p["k"], None, gt=gt)
    path = f"temp/agt_gallery/pics/{p['scene']}_k{p['k']:02d}.png"
    fig.savefig(path, dpi=80)
    plt.close(fig)
    print("saved", path, flush=True)

th = picks.get("thumbs", [])
cols = 8
for part, chunk in enumerate([th[i:i + 48] for i in range(0, len(th), 48)]):
    rows = (len(chunk) + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 3.3))
    for ax in np.atleast_1d(axs).ravel():
        ax.axis("off")
    for ax, p in zip(np.atleast_1d(axs).ravel(), chunk):
        gt = load_keyframe(D, p["tok"])
        img = view.semantic_rgb(gt)
        for grp, c in view.COL.items():
            img = view.blend(img, gt[grp], c)
        img = view.blend(img, gt["ignore"], (0.95, 0.85, 0.2), 0.7)
        img = np.clip(img * (1 - 0.6 * gt["undefined"][..., None]), 0, 1)
        view.show(ax, img, f"{p['scene']} k{p['k']}")
        ax.axis("off")
        ax.set_title(f"{p['scene']} k{p['k']}", fontsize=8)
        for b in gt["meta"]["boxes"]:
            v = b["vel"]
            if b["observed"] and np.hypot(*v) > 1.0 and 0 < b["center"][0] < 60 and abs(b["center"][1]) < 26:
                ax.annotate("", (b["center"][1] + v[1], b["center"][0] + v[0]), (b["center"][1], b["center"][0]),
                            arrowprops=dict(arrowstyle="->", color="cyan", lw=0.8))
        ax.set_xlim(26, -26)
        ax.set_ylim(0, 60)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.01, wspace=0.05, hspace=0.18)
    path = f"temp/agt_gallery/thumbs_{part}.png"
    fig.savefig(path, dpi=70)
    plt.close(fig)
    print("saved", path, flush=True)
