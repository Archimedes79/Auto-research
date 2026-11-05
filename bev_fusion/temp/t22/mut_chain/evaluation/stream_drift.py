"""t21 (D51): how an unbounded stream (one frame per step, the state carried) differs from its references, PER STEP t = the number of frames
since the cold start (t = 1, 2, 3, 4, 6, 9, 12, 18, 24, 36, 48 by default), at dt 0:

  exact   the independent reference that must agree to float round-off: recurrent = the batch unroll cold-started at the first frame over all
          t frames (a batch of t frames, built here from the scene's frames and their ego chains; the unroll exists up to --max-ref frames);
          pair = the batch of the two last frames given the stream's own state of the frame before (the memory carried)
  cold    the evaluation harness's way: the same frame with a COLD buffer: recurrent = the batch of the last 3 frames, pair = the last 2
          (what a per-sample `model(batch)` does) - the train / deploy gap: training chains up to 9 frames, the stream is unbounded
reported per t: max and mean absolute difference of the semantic logits (all channels) and of the motion field `v_hat` (B, 2, X, Y).

Frames: the scene's keyframe samples (n_cam 3, stride 2) put end to end are a gapless 6 Hz stream: frames (t-4, t-2, t) of sample i, the previous
key frame 2 sweeps before the oldest one (the chain from the dataset's gap steps). The dataset caches the radar of the KEY frame only: every frame of a
sample gets that radar (stream and references alike, so the comparison stays exact; the radar is mis-timed at the two earlier frames of a
sample). The lane branch is off (the cache holds the key-frame image only; it adds to the camera BEV after the temporal block).
Frames that are not reported go through `model.advance` (lift + temporal block), as a deployment that needs outputs only now and then would.

  python evaluation/stream_drift.py                          # CPU smoke: tiny random model, nuScenes mini scene-0061, 12 frames
  python evaluation/stream_drift.py --config training/configs/t21_rec.json --ckpt runs/t21_rec/ckpt_00002000.pt --scene scene-0003 --device cuda
Run from the bev_fusion folder. Result: a table on stdout and temp/stream/drift_<tag>.json."""
import argparse
import json
import os
import sys
import time

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = os.environ.get("STREAM_OUT") or os.path.join(ROOT, "temp", "stream")      # where the json of a run is written (scratch)
for sub in ("tests", "training"):
    sys.path.insert(0, os.path.join(ROOT, sub))
from feature_cache import CachedBEVFusionModel, CachedFrontRadarDataset, FeatureStore, collate_cached  # noqa: E402
from model import ModelConfig  # noqa: E402

TINY = dict(pretrained="", trunk_pretrained="", lane_pretrained="", grid_kwargs={"x_max": 16.0, "y_min": -8.0, "y_max": 8.0}, oversample=1,
            c_cam=16, c_rad=8, c_fuse=24, trunk_width=0.25, p_drop=0.0, motion_block=True, box_vel_from_field=True)
PER_FRAME = ("K", "T_cam_car", "prob", "ctx", "prob2", "ctx2", "img", "cam_vec")


def dataset(a, cfg):
    from nuscenes.nuscenes import NuScenes
    if cfg is None:
        root, version, kw = os.path.expanduser("~/data/nuscenes"), "v1.0-mini", dict(store=FeatureStore(os.path.join(ROOT, "data", "cache", "cam_feats_mini")))
    else:
        root, version = os.path.expanduser(cfg["data_root"]), cfg["version"]
        kw = dict(store=FeatureStore(os.path.join(ROOT, cfg["feature_cache"])), radar_cache=os.path.join(ROOT, cfg["radar_cache"]) or None,
                  camera_features=cfg["model"].get("camera_features", "dualbev"))
        if cfg.get("feature_cache2"):
            kw["store2"] = FeatureStore(os.path.join(ROOT, cfg["feature_cache2"]))
    nusc = NuScenes(version, dataroot=root, verbose=False)
    tok = [s["token"] for s in nusc.scene if s["name"] == a.scene][0]
    return CachedFrontRadarDataset(root, version, [tok], n_cam=3, cam_stride=2, with_gt=False, nusc=nusc, **kw)


def scene_frames(ds, n_frames):
    """the scene's frames in time order: {"b": the sample's collated batch, "k": its frame index (0 = key frame), "chain": the ego steps from
    the previous frame, "key": is the key frame}"""
    out, prev, i = [], None, 0
    while len(out) < n_frames and i < len(ds):
        b = collate_cached([ds[i]])
        steps = b["steps"]
        F = len(steps)
        for k in range(F - 1, -1, -1):
            if k == F - 1:      # the previous key frame -> the oldest frame: the gap chain of the previous sample less the oldest frame's own chain
                g = prev["gap_steps"][0]["next"] if prev is not None else []
                chain = [tuple(torch.full((1,), float(x)) for x in s) for s in g[:max(0, len(g) - len(steps[F - 1]))]] if prev is not None else None
            else:
                chain = steps[k + 1][:len(steps[k + 1]) - len(steps[k])]
            out.append({"b": b, "k": k, "chain": chain, "key": k == 0})
        prev, i = b, i + 1
    return out[:n_frames]


def one(f):
    b = dict(f["b"])
    for key in PER_FRAME:
        if key in b:
            b[key] = b[key][:, [f["k"]]]
    b.pop("lane_img", None)
    b["steps"] = [[]]
    return b


def many(frames, j0, j1):
    """the batch of frames j0..j1 (newest first) with their chains to frame j1; the other fields are those of frame j1's sample"""
    b = one(frames[j1])
    parts = [one(frames[j]) for j in range(j1, j0 - 1, -1)]
    for key in PER_FRAME:
        if key in b:
            b[key] = torch.cat([p[key] for p in parts], 1)
    b["steps"] = [sum((frames[j]["chain"] for j in range(jj + 1, j1 + 1)), []) for jj in range(j1, j0 - 1, -1)]
    return b


def diffs(a, b):
    out = {}
    for name, x, y in (("sem", a["out"][0.0]["sem"], b["out"][0.0]["sem"]), ("v", a["out"][0.0]["v_hat"], b["out"][0.0]["v_hat"])):
        d = (x.float() - y.float()).abs()
        out[name + "_max"], out[name + "_mean"] = float(d.max()), float(d.mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--buffer", default="", help="pair | recurrent (default: the config's, else recurrent)")
    ap.add_argument("--scene", default="scene-0061")
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--steps", default="1,2,3,4,6,9,12,18,24,36,48")
    ap.add_argument("--max-ref", type=int, default=24, help="the longest batch unroll used as the exact reference")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    torch.set_num_threads(2)
    dev = torch.device(a.device)
    cfg = json.load(open(os.path.join(ROOT, a.config))) if a.config else None
    buf = a.buffer or (cfg["model"].get("cam_buffer", "recurrent") if cfg else "recurrent")
    if buf not in ("pair", "recurrent"):
        sys.exit("the drift of a window model is not defined: --buffer pair | recurrent")
    mc = dict(TINY, camera_features="dualbev") if cfg is None else dict(cfg["model"], pretrained="", trunk_pretrained="", lane_branch="", lane_pretrained="")
    torch.manual_seed(0)
    m = CachedBEVFusionModel(ModelConfig(**dict(mc, cam_buffer=buf))).eval()
    if a.ckpt:
        ck = torch.load(os.path.join(ROOT, a.ckpt), map_location="cpu", weights_only=False)
        src = {k[6:]: v for k, v in ck["state"].items() if k.startswith("model.")}
        src.update(ck["ema"])
        miss, unexp = m.load_state_dict({k: v for k, v in src.items() if k in m.state_dict()}, strict=False)
        print(f"checkpoint {a.ckpt}: {len(miss)} model keys missing, {len(src) - len(src.keys() & m.state_dict().keys())} checkpoint keys not used (lane branch off)")
    else:
        with torch.no_grad():
            for q in m.parameters():
                if q.requires_grad and q.abs().max() == 0:
                    q.normal_(0, 0.05)
    m.to(dev)
    def to(b):      # the radar dict and the ego steps are nested: the trainer's to_device does the same
        out = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
        if "radar" in out:
            out["radar"] = {x: y.to(dev) for x, y in out["radar"].items()}
        if "steps" in out:
            out["steps"] = [[tuple(t.to(dev) if torch.is_tensor(t) else t for t in s) for s in fr] for fr in out["steps"]]
        return out
    ds = dataset(a, cfg)
    frames = scene_frames(ds, a.frames)
    wanted = [t for t in map(int, a.steps.split(",")) if t <= len(frames)]
    print(f"{a.scene}: {len(frames)} frames ({len(ds)} key samples), cam_buffer {buf}, reported at t = {wanted}")
    state, rows = None, []
    t0 = time.time()
    with torch.no_grad():
        for j, f in enumerate(frames):
            t, step = j + 1, f["chain"] if j else None
            if t not in wanted:
                state = m.advance(to(one(f)), state, step)
                continue
            res = m(to(one(f)), dts=(0.0,), state=state, step=step)
            if buf == "recurrent":
                exact = m(to(many(frames, 0, j)), dts=(0.0,)) if t <= a.max_ref else None
                cold = m(to(many(frames, j - 2, j)), dts=(0.0,)) if t >= 3 else None
            else:
                exact = m(to(many(frames, j - 1, j)), dts=(0.0,), state=state, step=step) if t >= 2 else None
                cold = m(to(many(frames, j - 1, j)), dts=(0.0,)) if t >= 2 else None
            row = {"t": t, "key_frame": f["key"]}
            for name, ref in (("exact", exact), ("cold", cold)):
                row[name] = diffs(res, ref) if ref is not None else None
            rows.append(row)
            state = res["state"]
    fmt = lambda d, k: "%9.2e" % d[k] if d else "        -"      # noqa: E731
    print(f"\n{'t':>3} key | exact (stream vs batch unroll / pair batch)      | cold (harness: last {3 if buf == 'recurrent' else 2} frames from cold)\n"
          f"{'':>7} | sem max   sem mean  v max     v mean    | sem max   sem mean  v max     v mean")
    for r in rows:
        print(f"{r['t']:>3} {'*' if r['key_frame'] else ' '}   | " + " ".join(fmt(r["exact"], k) for k in ("sem_max", "sem_mean", "v_max", "v_mean")) + " | "
              + " ".join(fmt(r["cold"], k) for k in ("sem_max", "sem_mean", "v_max", "v_mean")))
    print(f"({time.time() - t0:.0f} s; * = a key frame of the dataset, its own radar)")
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, f"drift_{a.scene}_{buf}_{'tiny' if cfg is None else os.path.basename(a.config)[:-5]}.json")
    json.dump({"scene": a.scene, "cam_buffer": buf, "config": a.config, "ckpt": a.ckpt, "rows": rows}, open(out, "w"), indent=1)
    print("written", out)


if __name__ == "__main__":
    main()
