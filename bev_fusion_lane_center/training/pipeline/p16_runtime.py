"""Loop 16b: fast inference runtime of the builder (batch 1, latency).
The forward of the builder (and, for the time recursion, the prior built from the previous frame) is captured in CUDA graphs (static shapes, no host synchronisation), all outputs of a frame leave the GPU in ONE transfer, the node arrays are the same as those of p16_infer.to_numpy
(so the assembly gives identical graphs). Usage as a library: FastBuilder(net, cfg, sample_batch); fb.raw(batch) -> node-array dict (the same as p16_infer.to_numpy(res, 0))."""
import numpy as np
import torch

import p16_builder as PB

KEYS = ("pos", "th_travel", "flag", "pi", "nu", "valid", "parent", "level", "pos_rev")


def prior_from_res(rp, pose, keep, P=64):
    """the prior of the time recursion from the single-frame output of the previous frame (same arithmetic as p16_time.prior_from_prev with train=False)"""
    pi = (rp["pi"] * rp["valid"]).float()
    val, ix = pi.topk(min(P, pi.shape[1]), dim=1)
    prior = dict(pos=PB.gather1(rp["pos"].float(), ix), th=PB.gather1(rp["th_travel"].float(), ix), pi=val, h=PB.gather1(rp["h"].float(), ix).detach(), valid=(val > 0.3).float())
    prior = PB.Builder.warp_prior(prior, pose)
    prior["valid"] = prior["valid"] * keep[:, None]
    return prior


class FastBuilder:
    def __init__(self, net, cfg, sample):
        self.net, self.cfg = net.eval(), cfg
        self.sx = sample["x"].clone()
        self.temporal = bool(cfg.temporal)
        self.layout = None
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._single(self.sx)
        torch.cuda.current_stream().wait_stream(s)
        # graph 0: single-frame forward (also the producer of the prior for the next frame)
        self.g0 = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.g0):
            self.res0, self.out0 = self._single(self.sx)
        if self.temporal:
            self.xp = sample["x_prev"].clone()
            self.pose = {k: sample[k].clone().float() for k in ("Rp", "Rc", "tp", "tc")}
            self.has = sample["has_prev"].clone().float()
            self.gp = torch.cuda.CUDAGraph()  # graph P: forward of the previous frame (no prior)
            with torch.cuda.graph(self.gp):
                self.resp, _ = self._single(self.xp)
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):
                    self._with_prior()
            torch.cuda.current_stream().wait_stream(s)
            self.g1 = torch.cuda.CUDAGraph()  # graph 1: the prior from graph P's output and the forward with the prior
            with torch.cuda.graph(self.g1):
                self.out1 = self._with_prior()

    def _pack(self, res):
        self.h_static = res["h"][0]  # node features stay on the GPU (static output of the graph that was captured last); only the vertices that survive the assembly are gathered from it
        parts, lay = [], []
        for k in KEYS:
            if k in res:
                t = res[k][0].float()
                t = t.reshape(t.shape[0], -1)
                parts.append(t.flatten())
                lay.append((k, tuple(res[k][0].shape)))
        if "junc_logit" in res:
            j = torch.sigmoid(res["junc_logit"][0].float()).half().float()
            parts.append(j.flatten())
            lay.append(("junc", tuple(j.shape)))
        if "link_ki" in res:
            p = torch.sigmoid(res["link_logit"][0]).float() * res["link_mask"][0].float()
            parts.append(p.flatten())
            lay.append(("link_p", tuple(p.shape)))
            parts.append(res["link_ki"][0].float().flatten())
            lay.append(("link_ki", tuple(p.shape)))
            self.link_off = res["link_off"]
        if self.layout is None:
            self.layout = lay
        return torch.cat(parts)

    def _single(self, x):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            res = self.net(x.float(), gt=None, share=0.0, prior=None)
        return res, self._pack(res)

    def _with_prior(self):
        prior = prior_from_res(self.resp, self.pose, self.has)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            res = self.net(self.sx.float(), gt=None, share=0.0, prior=prior)
        return self._pack(res)

    def _unpack(self, flat):
        r, o = {}, 0
        for k, shp in self.layout:
            n = int(np.prod(shp))
            r[k] = flat[o : o + n].reshape(shp)
            o += n
        out = {k: r[k] for k in KEYS if k in r}
        out["parent"] = out["parent"].astype(np.int64)
        if "junc" in r:
            out["junc"] = r["junc"].astype(np.float16)
        if "link_p" in r:
            p, ki = r["link_p"], r["link_ki"].astype(np.int64)
            aa, cc = np.nonzero(p > 0.05)
            off = self.link_off
            out["link_p"] = [(off + int(a), off + int(ki[a, c]), float(p[a, c])) for a, c in zip(aa, cc)]
        return out

    def raw(self, batch):
        """node arrays of one frame (batch of size 1)"""
        self.sx.copy_(batch["x"])
        if self.temporal:
            self.xp.copy_(batch["x_prev"])
            for k in self.pose:
                self.pose[k].copy_(batch[k].float())
            self.has.copy_(batch["has_prev"].float())
            self.gp.replay()
            self.g1.replay()
            flat = self.out1.cpu().numpy()
        else:
            self.g0.replay()
            flat = self.out0.cpu().numpy()
        return self._unpack(flat)
