"""Losses of the junction maps head: the loss side of net/model/jmaps.py (D152; the functions are unchanged, `jmaps` holds the network)."""
import torch
import torch.nn.functional as F



def _dice(logit, y, eps=1.0):
    p = logit.sigmoid()
    inter = (p * y).sum((-2, -1))
    return (1 - (2 * inter + eps) / (p.sum((-2, -1)) + y.sum((-2, -1)) + eps)).mean()


def jmaps_loss(out, batch, far_w=0.0, far_r=35.0):
    dev = out["j1"].device
    j1 = out["j1"].float()
    j05 = out["j05"].float() if "j05" in out else None                                                  # E332: a head without the paint branch has no j05 and no paint loss
    jz, jh, jd, jp = (batch[k].to(dev).float() for k in ("jz", "jheat", "jdir", "jpaint"))
    info = {}
    if "zone" in out.get("skip", ()):                                                                   # E332: the zone / core maps are not supervised (nothing reads them)
        l_z = 0.0
    else:
        l_z = F.binary_cross_entropy_with_logits(j1[:, :2], jz) + _dice(j1[:, :2], jz)
    info["zone"] = float(l_z)
    pw = torch.tensor(20.0, device=dev)
    if far_w > 0:                                                                                      # E304: the cells beyond far_r metres from the ego count 1 + far_w times in the heat BCE (53 % of the GT ends are there, the recall is 0.55)
        cc = -50.0 + (torch.arange(100, device=dev).float() + 0.5)
        wmap = 1.0 + far_w * ((cc[:, None] ** 2 + cc[None, :] ** 2).sqrt() > far_r).float()
        l_h = (F.binary_cross_entropy_with_logits(j1[:, 2:4], jh, pos_weight=pw, reduction="none") * wmap).mean() + _dice(j1[:, 2:4], jh)
    else:
        l_h = F.binary_cross_entropy_with_logits(j1[:, 2:4], jh, pos_weight=pw) + _dice(j1[:, 2:4], jh)
    info["heat"] = float(l_h)
    l_d = 0.0
    for k in range(2):                                                       # heading where the target has one: cosine loss on the normalised raw output
        tgt = jd[:, 2 * k: 2 * k + 2]
        m = (tgt.abs().sum(1) > 0.5).float()
        pr = j1[:, 4 + 2 * k: 6 + 2 * k]
        pr = pr / pr.norm(dim=1, keepdim=True).clamp_min(1e-4)
        l_d = l_d + ((1 - (pr * tgt).sum(1)) * m).sum() / m.sum().clamp_min(1.0)
    info["dir"] = float(l_d)
    l_p = F.binary_cross_entropy_with_logits(j05, jp, pos_weight=torch.tensor(10.0, device=dev)) + _dice(j05, jp) if j05 is not None else 0.0
    info["paint"] = float(l_p)
    return l_z + l_h + 0.5 * l_d + l_p, info
