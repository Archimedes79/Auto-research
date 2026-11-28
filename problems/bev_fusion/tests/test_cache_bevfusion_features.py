"""Tests of net/prep/cache_bevfusion_features.py (its former self-test).

    python -P -m bev_fusion.tests.test_cache_bevfusion_features      (from the problem folder, with its module path)
"""
import json
import os
import tempfile

import torch

from bev_fusion.net import netpath
from bev_fusion.net.model.bevfusion_camera import BEVFusionImageSide
from bev_fusion.net.prep.cache_bevfusion_features import FEAT_HW, _stream_records, build, filemap, source_tokens
from bev_fusion.net.prep.feature_cache import N_CTX, N_PROB
from bev_fusion.net.prep.images import load_image


def _selftest():
    os.chdir(netpath.SITE)                                       # the data folder: the mini feature cache and the BEVFusion weights are below it
    torch.manual_seed(0)
    root = netpath.nuscenes_root(mini=True)
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
        # the streamed parse equals a full json parse (bounded-memory reader, tiny chunks force records across chunk borders)
        full = {r["token"]: r for r in json.load(open(os.path.join(root, "v1.0-mini", "sample_data.json")))}
        streamed = {r["token"]: r for r in _stream_records(os.path.join(root, "v1.0-mini", "sample_data.json"), chunk=1 << 12)}
        assert streamed == full, "streamed records equal json.load"
        # key frames only keeps a subset (mini: the cached frames of one scene include sweeps)
        n_all = len(source_tokens(src)[sorted(source_tokens(src))[0]])
    keys = filemap(root, "v1.0-mini", [t for ts in source_tokens(src).values() for t in ts])
    n_key = sum(1 for v in keys.values() if v[1])
    print(f"cache_bevfusion_features selftest ok: shards in the DualBEV cache format ((n, 198, 32, 88) float16 + token index, read by FeatureStore), "
          f"probabilities sum to 1, equal to a direct forward within float16; {n_key} of {len(keys)} cached mini frames are key frames "
          f"(--keyframes-only), source scene has {n_all} cached frames")


if __name__ == "__main__":
    _selftest()
