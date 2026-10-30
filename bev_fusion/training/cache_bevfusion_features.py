"""Cache of the BEVFusion camera-only-seg image side's outputs (feature-swap test, plan.md 2.8 step 0): the same 198 channels as the DualBEV cache
(118 depth probabilities + 80 context, float16) but at stride 8 (32 x 88 cells) from the MAP-trained Swin-T + FPN of `training/bevfusion_camera.py`.

Same shard format as `training/feature_cache.py` (so `FeatureStore(root)` reads it, N_PROB = 118, N_CTX = 80): one `<scene>.npy` (n, 198, 32, 88)
float16 + `<scene>.index.npz` (the sample_data tokens, row order) per scene. The frames are the ones of an existing DualBEV cache (`--src`), so both caches cover
the same frames; `--keyframes-only` keeps only the key frames (20 k frames, about 22 GB) instead of all cached frames (58 k, about 64 GB: 1.1 MB per frame
against 0.28 MB at stride 16). Images as BEVFusion's test pipeline reads them (`bevfusion_camera.load_image`: 0.48 scale, bicubic, crop 32 / 176). The
calibration side of a model that reads this cache: `ida = bevfusion_ida()`, depth bin centres 1.0 + 0.5 i, `LiftPD(frustum_cells=True)`.
The token -> file name map is read from `sample_data.json` once (a trainval parse peaks at about 3 GB for a minute; do not run it next to a training round
without checking free memory). Resumable per scene, 2 loader threads.
Build: python training/cache_bevfusion_features.py build --src data/cache/cam_feats [--out data/cache/cam_feats_bevfusion] [--keyframes-only] [--device cuda]
Self-test (nuScenes mini, CPU): python training/cache_bevfusion_features.py
"""
import argparse
import glob
import json
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bevfusion_camera import BEVFusionImageSide, load_image  # noqa: E402
from feature_cache import N_CTX, N_PROB, FeatureStore  # noqa: E402

FEAT_HW = (32, 88)


def filemap(dataroot, version, tokens):
    """{sample_data token: (relative file name, is_key_frame)} for the given tokens, from sample_data.json."""
    want = set(tokens)
    with open(os.path.join(dataroot, version, "sample_data.json")) as f:
        recs = json.load(f)
    out = {r["token"]: (r["filename"], bool(r["is_key_frame"])) for r in recs if r["token"] in want}
    del recs
    return out


def source_tokens(src_root):
    """scene -> tokens (store order) of an existing cache."""
    out = {}
    for f in sorted(glob.glob(os.path.join(src_root, "*.index.npz"))):
        out[os.path.basename(f)[:-len(".index.npz")]] = [str(t) for t in np.load(f)["tokens"]]
    return out


def build(src_root, out_root, dataroot, version, device="cpu", batch=8, keyframes_only=False, threads=2, max_scenes=None,
          max_frames=None, verbose=True):
    os.makedirs(out_root, exist_ok=True)
    src = source_tokens(src_root)
    fmap = filemap(dataroot, version, [t for ts in src.values() for t in ts])
    model = BEVFusionImageSide().eval()
    rep = model.load_pretrained()
    assert not [k for k in rep["missing"] if not k.endswith("relative_position_index")] and not rep["unexpected"]
    model = model.to(device)
    pool = ThreadPoolExecutor(threads)
    t0, n_done = time.time(), 0
    for si, (scene, toks) in enumerate(sorted(src.items())):
        if max_scenes is not None and si >= max_scenes:
            break
        toks = [t for t in toks if t in fmap and (fmap[t][1] or not keyframes_only)]
        if max_frames is not None:
            toks = toks[:max_frames]
        if not toks:
            continue
        idx_path = os.path.join(out_root, scene + ".index.npz")
        if os.path.exists(idx_path) and set(np.load(idx_path)["tokens"].tolist()) >= set(toks):
            continue
        arr = np.zeros((len(toks), N_PROB + N_CTX) + FEAT_HW, np.float16)
        for i in range(0, len(toks), batch):
            chunk = toks[i:i + batch]
            imgs = list(pool.map(lambda t: load_image(os.path.join(dataroot, fmap[t][0])), chunk))
            with torch.no_grad():
                prob, ctx = model.image_features(torch.stack(imgs).to(device))
            arr[i:i + len(chunk)] = torch.cat([prob, ctx], 1).cpu().numpy().astype(np.float16)
        np.save(os.path.join(out_root, scene + ".npy"), arr)
        np.savez(idx_path, tokens=np.array(toks, dtype="U40"))
        n_done += len(toks)
        if verbose:
            print(f"{scene}: {len(toks)} frames ({n_done} done, {time.time() - t0:.0f} s)", flush=True)
    return FeatureStore(out_root)


def _selftest():
    torch.manual_seed(0)
    root = os.path.expanduser("~/data/nuscenes")
    src = "data/cache/cam_feats_mini"
    with tempfile.TemporaryDirectory() as tmp:
        store = build(src, tmp, root, "v1.0-mini", max_scenes=1, max_frames=4, verbose=False)
        assert len(store) == 4, len(store)
        tok = next(iter(store.index))
        prob, ctx = store.get(tok)
        assert prob.shape == (N_PROB,) + FEAT_HW and ctx.shape == (N_CTX,) + FEAT_HW and prob.dtype == torch.float16
        assert torch.allclose(prob.float().sum(0), torch.ones(FEAT_HW), atol=5e-3), "probabilities sum to 1"
        # equals a direct forward of the same image (float16 storage)
        fm = filemap(root, "v1.0-mini", [tok])
        model = BEVFusionImageSide().eval()
        model.load_pretrained()
        with torch.no_grad():
            p, c = model.image_features(load_image(os.path.join(root, fm[tok][0]))[None])
        assert (p[0] - prob.float()).abs().max() < 2e-3 and (c[0] - ctx.float()).abs().max() < 5e-2 * max(1.0, float(c.abs().max())), "cache = direct forward"
        # key frames only keeps a subset (mini: the cached frames of one scene include sweeps)
        n_all = len([t for t in source_tokens(src)[sorted(source_tokens(src))[0]] if t in fm or True])
    keys = filemap(root, "v1.0-mini", [t for ts in source_tokens(src).values() for t in ts])
    n_key = sum(1 for v in keys.values() if v[1])
    print(f"cache_bevfusion_features selftest ok: shards in the DualBEV cache format ((n, 198, 32, 88) float16 + token index, read by FeatureStore), "
          f"probabilities sum to 1, equal to a direct forward within float16; {n_key} of {len(keys)} cached mini frames are key frames "
          f"(--keyframes-only), source scene has {n_all} cached frames")


def main():
    if len(sys.argv) < 2 or sys.argv[1] != "build":
        return _selftest()
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--src", default="data/cache/cam_feats")
    ap.add_argument("--out", default="data/cache/cam_feats_bevfusion")
    ap.add_argument("--dataroot", default="~/data/nuscenes_full")
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--keyframes-only", action="store_true")
    a = ap.parse_args()
    store = build(a.src, a.out, os.path.expanduser(a.dataroot), a.version, a.device, a.batch, a.keyframes_only, a.threads)
    print(f"done: {len(store)} frames in {a.out}")


if __name__ == "__main__":
    main()
