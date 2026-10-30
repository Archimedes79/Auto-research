p = "training/feature_cache.py"
s = open(p, encoding="utf-8").read()
old = '''            prob, ctx = batch["prob"].float(), batch["ctx"].float()
            B, Fn = prob.shape[:2]
            if self.cam_adapter is not None:'''
new = '''            prob, ctx = batch["prob"].float(), batch["ctx"].float()
            B, Fn = prob.shape[:2]
            if self.cfg.camera_off:                                     # radar-only system (was ignored here until 2026-10-30:
                return torch.zeros(B, Fn, self.cfg.c_cam, self.grid.nx, self.grid.ny, device=prob.device)   # t2_cam_off ran WITH camera)
            if self.cam_adapter is not None:'''
assert old in s
s = s.replace(old, new)
# selftest guard
old2 = "def _selftest():"
new2 = '''def _selftest_camera_off():
    """the flags of ModelConfig must reach the cached model: camera_off gives a zero camera BEV and never reads the features."""
    m = CachedBEVFusionModel(__import__("dataclasses").replace(_model_cfg(), camera_off=True)).eval()
    batch = {"prob": torch.rand(2, 3, N_PROB, *FEAT_HW), "ctx": torch.rand(2, 3, N_CTX, *FEAT_HW)}
    out = m.camera_bevs(batch)
    assert out.shape == (2, 3, m.cfg.c_cam, m.grid.nx, m.grid.ny) and float(out.abs().sum()) == 0.0, "camera_off ignored by the cached model"
    print("camera_off selftest ok: zero camera BEV from the cached model")


def _model_cfg():
    try:
        from .model import ModelConfig
    except ImportError:
        from model import ModelConfig
    return ModelConfig()


def _selftest():'''
assert old2 in s
s = s.replace(old2, new2, 1)
s = s.replace('''    if argv[:1] == ["radar-selftest"]:''', '''    if argv[:1] == ["camera-off-selftest"]:
        return _selftest_camera_off()
    if argv[:1] == ["radar-selftest"]:''', 1)
open(p, "w", encoding="utf-8", newline="\n").write(s)
