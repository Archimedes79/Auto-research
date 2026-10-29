p = "training/blocks_new.py"
s = open(p, encoding="utf-8").read()
s = s.replace('''Run the self-test: python training/blocks_new.py
"""''', '''ZSumReduce     the standard LSS / DualBEV collapse of the height axis: SUM the depth-weighted features over the height slices
               (80 channels, no parameters), then 1x1 conv + BN + ReLU to c_out. Replaces the 1x1 conv over Z x 80 = 640 channels
               of the reference (`cam_reduce`), which has 10x more parameters learned from scratch (try 7 of t3_improve_loop).
Run the self-test: python training/blocks_new.py
"""''', 1)
s = s.replace('''def _selftest():
    torch.manual_seed(0)''', '''class ZSumReduce(nn.Module):
    def __init__(self, nz, c_in, c_out):
        super().__init__()
        self.nz, self.c_in = nz, c_in
        self.proj = nn.Sequential(nn.Conv2d(c_in, c_out, 1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(True))

    def forward(self, vol):                                # (B, Z * C, X, Y), z-major as lift_pd.height_to_channels
        B, _, X, Y = vol.shape
        return self.proj(vol.view(B, self.nz, self.c_in, X, Y).sum(1))


def _selftest():
    torch.manual_seed(0)''', 1)
s = s.replace('''    print(f"blocks_new selftest ok:''', '''    zr = ZSumReduce(8, 80, 64)
    xv = torch.randn(2, 8 * 80, 6, 5)
    ref = xv.view(2, 8, 80, 6, 5).sum(1)
    assert torch.allclose(zr.proj[0](ref), zr.proj[0](xv.view(2, 8, 80, 6, 5).sum(1))) and zr(xv).shape == (2, 64, 6, 5)
    print(f"blocks_new selftest ok:''', 1)
s = s.replace("CtxAdapter (identity at init, gradients)\")", "CtxAdapter (identity at init, gradients), ZSumReduce\")")
open(p, "w", encoding="utf-8", newline="\n").write(s)
p = "training/model.py"
s = open(p, encoding="utf-8").read()
s = s.replace("import CtxAdapter, RadarPillars, SecondTrunk", "import CtxAdapter, RadarPillars, SecondTrunk, ZSumReduce")
s = s.replace('''    cam_adapter: bool = False ''', '''    cam_collapse: str = "zconv"        # try 7: "zconv" (1x1 conv over Z x 80 channels) | "zsum" (sum over heights, then 1x1; blocks_new.ZSumReduce)
    cam_adapter: bool = False ''', 1)
s = s.replace('''        self.cam_reduce = nn.Sequential(nn.Conv2d(nz * n_ctx, cfg.c_cam, 1, bias=False), nn.BatchNorm2d(cfg.c_cam),
                                        nn.ReLU(True))''', '''        if cfg.cam_collapse == "zsum":
            self.cam_reduce = ZSumReduce(nz, n_ctx, cfg.c_cam)
        elif cfg.cam_collapse == "zconv":
            self.cam_reduce = nn.Sequential(nn.Conv2d(nz * n_ctx, cfg.c_cam, 1, bias=False), nn.BatchNorm2d(cfg.c_cam),
                                            nn.ReLU(True))
        else:
            raise ValueError(cfg.cam_collapse)''', 1)
s = s.replace('{"cam_adapter": True}]', '{"cam_adapter": True}, {"cam_collapse": "zsum"}]')
open(p, "w", encoding="utf-8", newline="\n").write(s)
