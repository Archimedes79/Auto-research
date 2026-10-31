"""Loop 17 / E137 (copy of p16_pipeline.py): the whole chain per frame, plus the memory stage (warp + learned gate) and the option of the time recursion on / off. Original text: the whole chain per frame, fast (reference implementation of the deployment path) and its stage times.
frame features -> builder (CUDA graph) -> assembly (CPU) -> route input (vertex tokens, points every <= 4 m) -> route head along the edges (CUDA graphs per padded size) -> planner consumer (CUDA graph).
Usage (gcar, GPU, idle machine): p16_pipeline.py --name e123_D7_r3_f1 [--n 60]"""
import bflc_paths as BP  # project paths from the location of this checkout
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p8_lib as L  # noqa: E402
import p10_net as PN  # noqa: E402
import p14_chain_consumer as CC  # noqa: E402
import p16_common as C  # noqa: E402
import p16_data as PD  # noqa: E402
import p16_infer as PI  # noqa: E402
import p16_runtime as RT  # noqa: E402
import p16_timing as TM  # noqa: E402
import p16_tok_resample as TR  # noqa: E402
import p16_tokens as TK  # noqa: E402
import sd_lib as SL  # noqa: E402

BUCKETS = (96, 160, 224, 288, 384, 512, 768)


class FastRoute:
    """RouteXNet with one CUDA graph per padded number of points (the masked padding does not change the result for the real points)"""

    def __init__(self, net):
        self.net = net.eval()
        self.g, self.st = {}, {}
        for n in BUCKETS:
            st = dict(tok=torch.zeros(1, n, 128, device="cuda"), xy=torch.zeros(1, n, 2, device="cuda"), cp=torch.zeros(1, n, 7, 2, device="cuda"), sd=torch.zeros(1, 5, 200, 200, device="cuda"), mask=torch.zeros(1, n, dtype=torch.bool, device="cuda"))
            st["mask"][:, 0] = True
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    self._f(st)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                out = self._f(st)
            self.g[n], self.st[n] = g, dict(st, out=out)

    def _f(self, st):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            return torch.sigmoid(self.net(st["tok"], st["xy"], st["cp"], st["sd"], st["mask"]).float())

    def __call__(self, tok, xy, cp, sd, n):
        b = next((k for k in BUCKETS if k >= n), None)
        if b is None:  # larger than the largest bucket: eager
            m = torch.ones(1, n, dtype=torch.bool, device="cuda")
            return self._f(dict(tok=tok, xy=xy, cp=cp, sd=sd, mask=m))[0]
        st = self.st[b]
        st["tok"].zero_()
        st["xy"].zero_()
        st["cp"].zero_()
        st["mask"].zero_()
        st["tok"][0, :n] = tok
        st["xy"][0, :n] = xy
        st["cp"][0, :n] = cp
        st["mask"][0, :n] = True
        st["sd"].copy_(sd)
        self.g[b].replay()
        return st["out"][0, :n]


def route_input(r, g_aux, g, h_static, step, keep):
    """the token item of the route head from the assembly output (same as p16_tokens.py + p16_tok_resample.py)"""
    V = np.asarray(g_aux["V"], np.float32).reshape(-1, 2)
    src = np.asarray(g_aux["src"], int)
    n = len(V)
    tok = np.zeros((n, 128), np.float16)
    has = src >= 0
    if has.any():
        h = h_static[torch.as_tensor(src[has], device=h_static.device)].float().cpu().numpy()
        tok[has, : h.shape[1]] = h
    it = dict(xy=V, tok=tok, has=has, cp=TK.cross_section(V, g_aux["edges"]), pieces=[list(map(int, p)) for p in g["nodes"]], edges=sorted((int(x), int(y)) for x, y in g_aux["edges"]), keep=np.where(has, keep[np.clip(src, 0, None)], 0.5).astype(np.float32))
    return TR.resample_item(it, step)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="e123_D7_r3_f1")
    ap.add_argument("--M", type=int, default=288)
    ap.add_argument("--K0", type=int, default=144)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--route", default="p16_f2hr_route_s0")
    ap.add_argument("--step", type=float, default=4.0)
    ap.add_argument("--rec", type=int, default=0, help="1: time recursion on (the champion as trained), 0: off")
    a = ap.parse_args()
    net, cfg = TM.load_builder(a.M, a.K0, a.name)
    if not a.rec:
        cfg.temporal = False
        net.cfg.temporal = False
    ks = C.val_subset()[: a.n]
    ds = PD.BuilderData("val_clean", ks=ks, train=False, with_prev=cfg.temporal)
    batches = [TM.to_gpu(PD.collate([ds[i]])) for i in range(a.n)]
    sdl = L.cache("sd", ds.part if hasattr(ds, "part") else "val")
    rows = [int(ds.rows[int(k)]) if hasattr(ds, "rows") else int(k) for k in ks]
    ras = []
    for row in rows:
        sd = sdl[row]
        if sd is None:
            ras.append(torch.zeros(1, 5, 200, 200, device="cuda"))
        else:
            sdm, _, _ = L.draw_sd(sd, row, 0.0, 0.0, 1)
            ras.append(torch.from_numpy(SL.raster_sd(sdm))[None].cuda())
    fb = RT.FastBuilder(net, cfg, batches[0])
    rnet = PN.RouteXNet().cuda()
    rnet.load_state_dict(torch.load(f"{L.OUT}/{a.route}/model.pt", map_location="cuda"))
    fr = FastRoute(rnet)
    cons = CC.ConsB(CC.nfeat("route+ego"), False).cuda().eval()
    cst = dict(x=torch.zeros(1, 320, CC.nfeat("route+ego"), device="cuda"), t=None, m=torch.ones(1, 320, dtype=torch.bool, device="cuda"))
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                cons(cst["x"], None, cst["m"])
    torch.cuda.current_stream().wait_stream(s)
    cg = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cg):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            cout = cons(cst["x"], None, cst["m"])
    T = {k: [] for k in ("builder", "assembly", "route_input", "route_head", "consumer_prep", "consumer", "total")}
    nv = []
    for rep in range(2):
        for v in T.values():
            v.clear()
        nv.clear()
        for b, ra in zip(batches, ras):
            torch.cuda.synchronize()
            t0 = time.time()
            r = fb.raw(b)
            t1 = time.time()
            g, ax = PI.assemble(r, None, 0.7, cfg, aux=True, **TM.ASM)
            t2 = time.time()
            if not len(ax["V"]):
                continue
            it = route_input(r, ax, g, fb.h_static, a.step, r["pi"])
            t3 = time.time()
            n = len(it["xy"])
            p = fr(torch.from_numpy(it["tok"].astype(np.float32))[None].cuda(), torch.from_numpy(it["xy"])[None].cuda(), torch.from_numpy(it["cp"])[None].cuda(), ra, n).cpu().numpy()
            t4 = time.time()
            # consumer input (edge geometry, route probability per edge, ego motion): numpy
            ia = np.array([e[0] for pth in it["pieces"] for e in zip(pth[:-1], pth[1:])], int)[:320]
            ib = np.array([e[1] for pth in it["pieces"] for e in zip(pth[:-1], pth[1:])], int)[:320]
            xy = it["xy"]
            x = np.zeros((1, 320, CC.nfeat("route+ego")), np.float32)
            m_ = len(ia)
            x[0, :m_, :6] = np.concatenate([xy[ia], 0.5 * (xy[ia] + xy[ib]), xy[ib]], 1) / 50.0
            x[0, :m_, 6] = 0.5 * (p[ia] + p[ib])
            cst["x"].copy_(torch.from_numpy(x))
            t5 = time.time()
            cg.replay()
            cout.float().cpu()
            t6 = time.time()
            for k, (u, w) in zip(T, ((t1, t0), (t2, t1), (t3, t2), (t4, t3), (t5, t4), (t6, t5), (t6, t0))):
                T[k].append(u - w)
            nv.append((len(g["pts"]), n))
    res = {k: dict(median_ms=float(np.median(v)) * 1e3, p90_ms=float(np.percentile(v, 90)) * 1e3) for k, v in T.items()}
    res["points_per_frame_mean"] = float(np.mean([x[1] for x in nv]))
    res["name"], res["M"], res["K0"] = a.name, a.M, a.K0
    import p17_gate as GT

    mem = GT.Stabilizer().cuda().eval()
    Pz = dict(np.load(f"{PD.D}/poses_val.npz"))
    tm = []
    with torch.no_grad():
        for rep in range(2):
            tm.clear()
            for b, row in zip(batches, rows):
                y = b["x"].float()
                torch.cuda.synchronize()
                t_ = time.time()
                m_, v_ = GT.warp_mem(y, Pz, row, row)
                mem(y, m_, v_)
                torch.cuda.synchronize()
                tm.append(time.time() - t_)
    res["memory"] = dict(median_ms=float(np.median(tm)) * 1e3, p90_ms=float(np.percentile(tm, 90)) * 1e3)
    res["rec"] = a.rec
    print(json.dumps(res, indent=1))
    os.makedirs(BP.ROOT + "/runs/e17/latency", exist_ok=True)
    json.dump(res, open(f"{BP.ROOT}/runs/e17/latency/latency_M{a.M}_rec{a.rec}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
