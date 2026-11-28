"""Cache of the BEVFusion camera-only-seg image side's outputs (the feature-swap test): the same 198 channels as the DualBEV cache
(118 depth probabilities + 80 context, float16) but at stride 8 (32 x 88 cells) from the MAP-trained Swin-T + FPN of `net/model/bevfusion_camera.py`.

Same shard format as `net/prep/feature_cache.py` (so `FeatureStore(root)` reads it, N_PROB = 118, N_CTX = 80): one `<scene>.npy` (n, 198, 32, 88)
float16 + `<scene>.index.npz` (the sample_data tokens, row order) per scene. The frames are the ones of an existing DualBEV cache (`--src`), so both caches
cover the same frames; `--keyframes-only` keeps only the key frames (20 k frames, about 22 GB) instead of all cached frames (58 k, about 64 GB: 1.1 MB per frame
against 0.28 MB at stride 16). Images as BEVFusion's test pipeline reads them (`bevfusion_camera.load_image`: 0.48 scale, bicubic, crop 32 / 176). The
calibration side of a model that reads this cache: `ida = bevfusion_ida()`, depth bin centres 1.0 + 0.5 i, `LiftPD(frustum_cells=True)`.
The token -> file name map is read from `sample_data.json` once (a trainval parse peaks at about 3 GB for a minute; do not run it next to a training round
without checking free memory). Resumable per scene, 2 loader threads.
Build: python -P -m bev_fusion.net.prep.cache_bevfusion_features build --src data/cache/cam_feats [--out data/cache/cam_feats_bevfusion] [--keyframes-only]
       [--device cuda]
Tests: tests/test_cache_bevfusion_features.py
"""
import argparse
import glob
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from bev_fusion.net import netpath
from bev_fusion.net.model.bevfusion_camera import BEVFusionImageSide
from bev_fusion.net.prep.images import load_image
from bev_fusion.net.prep.feature_cache import FEAT_HW_OF, N_CTX, N_PROB, FeatureStore, shard_complete, write_shard

FEAT_HW = FEAT_HW_OF["bevfusion"]


def _stream_records(path, chunk=1 << 24):
    """the records of a big JSON array one by one with a bounded buffer (sample_data.json of trainval parses to about 3 GB at once)."""
    dec = json.JSONDecoder()
    with open(path, encoding="utf-8") as f:
        buf = f.read(chunk)
        i = buf.index("[") + 1
        while True:
            while i < len(buf) and buf[i] in " \n\r\t,":
                i += 1
            if i >= len(buf):
                more = f.read(chunk)
                if not more:
                    return
                buf, i = buf[i:] + more, 0
                continue
            if buf[i] == "]":
                return
            try:
                obj, j = dec.raw_decode(buf, i)
            except json.JSONDecodeError:
                more = f.read(chunk)
                if not more:
                    raise
                buf, i = buf[i:] + more, 0
                continue
            yield obj
            i = j
            if i > chunk:
                buf, i = buf[i:], 0


def filemap(dataroot, version, tokens, cache_path=None):
    """{sample_data token: (relative file name, is_key_frame)} for the given tokens, streamed from sample_data.json (bounded memory); a small
    json file `cache_path` remembers the answer, so the extraction job itself never parses the big file."""
    want = set(tokens)
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as f:
            got = {k: (v[0], v[1]) for k, v in json.load(f).items()}
        if want <= set(got):
            return {t: got[t] for t in want}
    out = {}
    for r in _stream_records(os.path.join(dataroot, version, "sample_data.json")):
        if r["token"] in want:
            out[r["token"]] = (r["filename"], bool(r["is_key_frame"]))
    if cache_path:
        with open(cache_path, "w") as f:
            json.dump({k: list(v) for k, v in out.items()}, f)
    return out


def source_tokens(src_root):
    """scene -> tokens (store order) of an existing cache."""
    out = {}
    for f in sorted(glob.glob(os.path.join(src_root, "*.index.npz"))):
        out[os.path.basename(f)[:-len(".index.npz")]] = [str(t) for t in np.load(f)["tokens"]]
    return out


def build(src_root, out_root, dataroot, version, device="cpu", batch=8, keyframes_only=False, threads=2, max_scenes=None,
          max_frames=None, verbose=True, filemap_path=None, dry_run=False):
    os.makedirs(out_root, exist_ok=True)
    src = source_tokens(src_root)
    fmap = filemap(dataroot, version, [t for ts in src.values() for t in ts], filemap_path)
    if dry_run:
        n = sum(1 for ts in src.values() for t in ts if t in fmap and (fmap[t][1] or not keyframes_only))
        print(f"dry run: {n} frames in {len(src)} scenes, {n * (N_PROB + N_CTX) * FEAT_HW[0] * FEAT_HW[1] * 2 / 1e9:.1f} GB of float16, {len(fmap)} of the source tokens found in sample_data.json")
        return None
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
        if shard_complete(out_root, scene, set(toks)):
            continue
        arr = np.zeros((len(toks), N_PROB + N_CTX) + FEAT_HW, np.float16)
        for i in range(0, len(toks), batch):
            chunk = toks[i:i + batch]
            imgs = list(pool.map(lambda t: load_image(os.path.join(dataroot, fmap[t][0])), chunk))
            with torch.no_grad():
                prob, ctx = model.image_features(torch.stack(imgs).to(device))
            arr[i:i + len(chunk)] = torch.cat([prob, ctx], 1).cpu().numpy().astype(np.float16)
        write_shard(out_root, scene, arr, toks)
        n_done += len(toks)
        if verbose:
            print(f"{scene}: {len(toks)} frames ({n_done} done, {time.time() - t0:.0f} s)", flush=True)
    return FeatureStore(out_root)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--src", default="data/cache/cam_feats")
    ap.add_argument("--out", default="data/cache/cam_feats_bevfusion")
    ap.add_argument("--dataroot", default=netpath.nuscenes_root())
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--keyframes-only", action="store_true")
    ap.add_argument("--filemap", default="data/cache/cam_front_filemap.json")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    store = build(a.src, a.out, os.path.expanduser(a.dataroot), a.version, a.device, a.batch, a.keyframes_only, a.threads,
                  filemap_path=a.filemap, dry_run=a.dry_run)
    if store is not None:
        print(f"done: {len(store)} frames in {a.out}")


if __name__ == "__main__":
    main()
