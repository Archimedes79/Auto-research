"""Tests of net/model/cached_model.py (its former self-test).

    python tests/test_cached_model.py      (from the problem folder, with its module path)
"""
import torch
import torch.utils.checkpoint

from net.model.cached_model import CachedBEVFusionModel


def _model_cfg():
    from net.model.model import ModelConfig
    return ModelConfig()


def _selftest_camera_off():
    """the flags of ModelConfig must reach the cached model: camera_off gives a zero camera BEV and never reads the features."""
    from net.prep.feature_cache import FEAT_HW, N_CTX, N_PROB
    m = CachedBEVFusionModel(__import__("dataclasses").replace(_model_cfg(), camera_off=True)).eval()
    batch = {"prob": torch.rand(2, 3, N_PROB, *FEAT_HW), "ctx": torch.rand(2, 3, N_CTX, *FEAT_HW)}
    out = m.camera_bevs(batch)
    assert out.shape == (2, 3, m.cfg.c_cam, m.grid.nx, m.grid.ny) and float(out.abs().sum()) == 0.0, "camera_off ignored by the cached model"
    print("camera_off selftest ok: zero camera BEV from the cached model")


if __name__ == "__main__":
    _selftest_camera_off()
