import sys
sys.path.insert(0, "training")
import torch
import torch.nn as nn
torch.set_num_threads(2)
import distill_student as D

cfg = D.DistillConfig(split="small", workers=0, batch=16)
sets, _, val_loader = D.build_data(cfg)


class AsNull(nn.Module):
    grid8 = torch.ones(1)

    def __init__(self, null):
        super().__init__()
        self.null = null

    def forward(self, img):
        return D.null_output(self.null, len(img), "cpu")


for floor in (1e-8, 1e-6, 1e-4):
    D.NULL_FLOOR = floor
    null = D.null_baseline(sets["train"], 512, 0)
    res = D.validate(AsNull(null), val_loader, "cpu")
    print(f"floor {floor:g}: kl16 {res['kl16']:.3f} (near {res['kl16.near']:.2f} mid {res['kl16.mid']:.2f} far {res['kl16.far']:.2f}) kl8 {res['kl8']:.3f} (near {res['kl8.near']:.2f} mid {res['kl8.mid']:.2f} far {res['kl8.far']:.2f})", flush=True)
