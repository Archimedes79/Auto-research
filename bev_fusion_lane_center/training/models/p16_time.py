"""Loop 16 rung 3, T.2 time recursion (one-step memory): the nodes of the previous frame (own single-frame prediction, no gradient), warped by the ego motion, are extra seed candidates (validity x 0.9)
and their validity is fused with the current one by log-odds (a = 0.9, b = 1.0) where a current node coincides with a previous node. Pose noise and a share of frames without a prior in training.
Usage: imported by p16_train.py / p16_infer.py."""
import torch

import p16_builder as PB


def prior_from_prev(net, b, train=False, p_none=0.1, sig_t=0.15, sig_yaw_deg=0.5, P=64):
    was = net.training
    net.eval()
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.float16):
            rp = net(b["x_prev"].float())
    net.train(was)
    pi = (rp["pi"] * rp["valid"]).float()
    val, ix = pi.topk(min(P, pi.shape[1]), dim=1)
    prior = dict(pos=PB.gather1(rp["pos"].float(), ix), th=PB.gather1(rp["th_travel"].float(), ix), pi=val, h=PB.gather1(rp["h"].float(), ix).detach(), valid=(val > 0.3).float())
    pose = dict(Rp=b["Rp"].float(), Rc=b["Rc"].float(), tp=b["tp"].float(), tc=b["tc"].float())
    keep = b["has_prev"].float()
    if train:  # pose noise of the odometry and frames without a prior (single-frame mode)
        B = pose["tc"].shape[0]
        pose["tc"] = pose["tc"] + torch.randn_like(pose["tc"]) * sig_t
        a = torch.randn(B, device=pose["tc"].device) * (sig_yaw_deg * 3.14159265 / 180.0)
        rot = torch.stack([torch.stack([torch.cos(a), -torch.sin(a)], -1), torch.stack([torch.sin(a), torch.cos(a)], -1)], -2)
        pose["Rc"] = rot @ pose["Rc"]
        keep = keep * (torch.rand(B, device=keep.device) >= p_none).float()
    prior = PB.Builder.warp_prior(prior, pose)
    prior["valid"] = prior["valid"] * keep[:, None]
    return prior
