"""Tests of net/prep/build_feature_cache.py (its former self-test).

    python -P -m bev_fusion.tests.test_build_feature_cache      (from the problem folder, with its module path)
"""
import os
import time

import torch

from bev_fusion.net import netpath
from bev_fusion.net.model.cached_model import CachedBEVFusionModel
from bev_fusion.net.prep.build_feature_cache import build
from bev_fusion.net.prep.feature_cache import FEAT_HW, N_CTX, N_PROB, CachedFrontRadarDataset, collate_cached, used_frames
from bev_fusion.net.prep.nuscenes_dataset import FrontRadarDataset, collate


def _selftest():
    import tempfile
    os.chdir(netpath.SITE)                                       # the data folder: the default weights are named relative to it
    from bev_fusion.net.model.model import BEVFusionModel
    torch.manual_seed(0)
    root_nusc = netpath.nuscenes_root(mini=True)
    ds = FrontRadarDataset(root_nusc, "v1.0-mini", n_cam=3, cam_stride=2, with_gt=False)
    tokens = [ds.samples[10], ds.samples[200]]
    used = used_frames(ds, tokens)
    n_frames = sum(len(v) for v in used.values())
    assert 3 <= n_frames <= 6, n_frames
    with tempfile.TemporaryDirectory() as tmp:
        t0 = time.time()
        store = build(ds, tmp, tokens, device="cpu", batch=4, verbose=False)
        assert len(store) == n_frames and all(store.has(t) for v in used.values() for t in v)
        # storage: float16, shapes, probabilities sum to one
        t = next(iter(used.values())).copy().pop()
        prob, ctx = store.get(t)
        assert prob.dtype == torch.float16 and prob.shape == (N_PROB,) + FEAT_HW and ctx.shape == (N_CTX,) + FEAT_HW
        assert torch.allclose(prob.float().sum(0), torch.ones(FEAT_HW), atol=2e-2), "probabilities sum to one"
        build(ds, tmp, tokens, device="cpu", verbose=False)                                  # resumable: nothing recomputed
        # equivalence: the cached model's camera BEVs = the standard model's (same weights downstream)
        cds = CachedFrontRadarDataset(root_nusc, "v1.0-mini", store=store, n_cam=3, cam_stride=2, with_gt=False)
        ia, ib = ds.samples.index(tokens[0]), ds.samples.index(tokens[1])
        std = BEVFusionModel().eval()
        cached = CachedBEVFusionModel().eval()
        missing = cached.load_state_dict({k: v for k, v in std.state_dict().items() if not k.startswith("camera.")},
                                         strict=False)
        assert not missing.unexpected_keys and not missing.missing_keys, "the cached model has every non-camera weight"
        b_std = collate([ds[ia], ds[ib]])
        b_cac = collate_cached([cds[ia], cds[ib]])
        assert b_cac["img"].numel() < b_std["img"].numel() / 1e4, "no images are loaded"
        with torch.no_grad():
            bev_std, bev_cac = std.camera_bevs(b_std), cached.camera_bevs(b_cac)
        err = (bev_std - bev_cac).abs().max().item()
        scale = bev_std.abs().max().item()
        assert bev_std.shape == bev_cac.shape and err < 0.02 * scale + 1e-4, f"cached vs standard: {err:.4g} of {scale:.4g}"
        # the whole chain runs from the cached batch and gives the same outputs as the standard model
        with torch.no_grad():
            o_std = std(b_std)["out"][0.0]["sem"]
            o_cac = cached(b_cac)["out"][0.0]["sem"]
        err_o = (o_std - o_cac).abs().max().item()
        assert err_o < 0.05 * max(1.0, o_std.abs().max().item()), f"outputs: {err_o:.4g}"
        n_cached = sum(p.numel() for p in cached.parameters())
        n_std = sum(p.numel() for p in std.parameters())
        print(f"feature_cache selftest ok ({time.time() - t0:.0f} s): {n_frames} frames cached as float16 (0.28 MB each), "
              f"resumable, probabilities sum to 1, no images read, camera BEVs equal the standard model's within "
              f"{err / scale:.1e} of their range, chain outputs within {err_o:.1e}; the cached model has "
              f"{(n_std - n_cached) / 1e6:.1f} M fewer parameters (frozen heads not instantiated)")


if __name__ == "__main__":
    _selftest()
