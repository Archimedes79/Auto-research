"""throw-away: video of the fused velocity model over every keyframe of held-out scenes (tests/test_flow_fused.py,
the chosen arm: labels + boxes +-dt + static, no correlation, seed 0). Each keyframe is predicted from itself and the one
before (the model has no carried state). Panels: maps + radar Doppler at k, GT velocity, learned velocity, error on objects.
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python temp/checks/sup/video_fused.py [model tag] [n scenes]"""
import json
import os
import sys

TAG = sys.argv[1] if len(sys.argv) > 1 else "_vel+box+static_nocorr"
NSC = int(sys.argv[2]) if len(sys.argv) > 2 else 2
sys.argv = [sys.argv[0]] + (["--corr"] if "corr" in TAG and "nocorr" not in TAG else [])
sys.path.insert(0, "studies/early")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import flow_fused as tf  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402
from flow_viz import flow_rgb, wheel  # noqa: E402

torch.set_num_threads(2)
NS = os.path.expanduser("~/data/nuscenes_full/v1.0-trainval")
samples = {s["token"]: s for s in json.load(open(f"{NS}/sample.json"))}
scenes = {s["token"]: s for s in json.load(open(f"{NS}/scene.json"))}
for f in sorted(os.listdir("data/cache/radar")):
    tf.RADAR.setdefault(f.split("_")[0], []).append(os.path.join("data/cache/radar", f))


def gt_path(tok):
    return f"data/cache/gt/{tok}_0.5_4_{tf.SIG}.npz"


def item_all(path):
    """tf.item without the moving-vehicle filter (every keyframe of the scene)."""
    keep = tf.maps_of
    g = np.load(path, allow_pickle=True)["gt"].item()
    if g.get("prev") is None or g.get("next") is None:
        return None
    vel = g["cur"]["velocity"]
    if not ((keep(g["cur"])[1] > 0.5) & (np.hypot(*vel) > 1.0)).any():   # tf.item would drop it: use a copy with v set
        g2 = dict(g)
        return _item_from(g2, path)
    return tf.item(path)


def _item_from(g, path):
    p, c, n = g["prev"], g["cur"], g["next"]
    token = os.path.basename(path).split("_")[0]
    rf = tf.RADAR.get(token)
    if not rf:
        return None
    Db, Mb = tf.box_motion(c, p)
    Df, Mf = tf.box_motion(c, n)
    static = ((c["instance"] < 0) & (c["ignore"] < 0.5)).astype(np.float32)
    return {"token": token,
            "maps": torch.from_numpy(np.stack([tf.maps_of(p), tf.maps_of(c), tf.maps_of(n)])).half(),
            "vel": torch.from_numpy(c["velocity"].astype(np.float32)),
            "inst": torch.from_numpy(c["instance"].astype(np.int16)),
            "D": torch.from_numpy(np.stack([Db, Df])), "M": torch.from_numpy(np.stack([Mb, Mf])).bool(),
            "static": torch.from_numpy(static).bool(),
            "radar": torch.from_numpy(tf.radar_raster(np.load(rf[0]))),
            "s_back": [tuple(map(float, s)) for s in p["meta"]["steps"]],
            "s_fwd": [tuple(map(float, s)) for s in n["meta"]["steps"]],
            "s_prev": [(-float(dt), float(v), float(w)) for dt, v, w in reversed(p["meta"]["steps"])],
            "dt": (float(p["meta"]["dt"]), float(n["meta"]["dt"]))}


def scene_tokens(sc):
    out, tok = [], scenes[sc]["first_sample_token"]
    while tok:
        out.append(tok)
        tok = samples[tok]["next"]
    return out


# held-out scenes ranked by how many of their keyframes are in the held-out evaluation set (= most moving vehicles)
test = tf.items(True, 48)
count = {}
for x in test:
    sc = samples[x["token"]]["scene_token"]
    count[sc] = count.get(sc, 0) + 1
chosen = sorted(count, key=lambda s: -count[s])[:NSC]

model = tf.Model().to(tf.DEV)
model.load_state_dict(torch.load(f"temp/checks/model_fused{TAG}.pt", map_location=tf.DEV))
model.eval()
os.makedirs("temp/checks/video_fused", exist_ok=True)
frames = []
for sc in chosen:
    name = scenes[sc]["name"]
    toks = [t for t in scene_tokens(sc) if os.path.exists(gt_path(t))]
    errs = []
    for i, tok in enumerate(toks):
        x = item_all(gt_path(tok))
        if x is None:
            continue
        t = tf.batch_of([x], [0])
        with torch.no_grad():
            v = model(t)[-1][0].cpu()
        mp = x["maps"].float()
        occ = (mp[1, 1:].max(0).values > 0.3).float()
        veh = mp[1, 1] > 0.5
        mov = veh & (x["vel"].norm(dim=0) > 1.0)
        e = (v - x["vel"]).norm(dim=0)
        em = float(e[mov].mean()) if mov.any() else float("nan")
        errs.append(em)
        rad = x["radar"]
        dop = torch.stack([(rad[2] / 10).clamp(0, 1), 0.25 * rad[0], (-rad[2] / 10).clamp(0, 1)], -1)
        lane = 0.35 * mp[1, 0].clamp(0, 1)[..., None]
        dop = torch.maximum(torch.maximum(dop, 0.3 * mp[1, 1:].max(0).values[..., None]), lane).flip(1).numpy()
        err = torch.stack([(e / 3).clamp(0, 1) * occ, 0.2 * occ, 0.2 * occ], -1)
        err = torch.maximum(err, lane).flip(1).numpy()
        ims = [dop, flow_rgb(x["vel"], occ, 15.0), flow_rgb(v, occ, 15.0), err]
        cols = ["maps + radar Doppler (red +, blue -)", "GT velocity", "predicted velocity", "error on objects (red = 3 m/s)"]
        fig, ax = plt.subplots(1, 5, figsize=(13, 3.4), gridspec_kw={"width_ratios": [1, 1, 1, 1, 0.5]})
        for c, im in enumerate(ims):
            ax[c].imshow(im, origin="lower")
            ax[c].set_title(cols[c], fontsize=8)
            ax[c].axis("off")
        ax[4].axis("off")
        wheel(ax[4], vmax=15.0)
        ax[4].set_title("direction: forward = up", fontsize=7)
        fig.suptitle(f"{name}  keyframe {i + 1}/{len(toks)}  (t = {0.5 * i:.1f} s)   moving-vehicle error "
                     f"{em:.2f} m/s   |  60 x 52 m, ego at the bottom centre, x forward = up; hue = direction, "
                     f"brightness = speed (15 m/s full)", fontsize=8)
        fig.tight_layout()
        fp = f"temp/checks/video_fused/{name}_{i:03d}.png"
        fig.savefig(fp, dpi=90)
        plt.close(fig)
        frames.append(Image.open(fp).convert("RGB"))
    ok = [e for e in errs if e == e]
    print(f"{name}: {len(errs)} keyframes, moving-vehicle error mean {np.mean(ok):.2f} m/s over {len(ok)} frames with a "
          f"moving vehicle (held-out scene)", flush=True)
out = f"temp/checks/video_fused{TAG}.gif"
frames[0].save(out, save_all=True, append_images=frames[1:], duration=500, loop=0)
print("video", out, len(frames), "frames")
