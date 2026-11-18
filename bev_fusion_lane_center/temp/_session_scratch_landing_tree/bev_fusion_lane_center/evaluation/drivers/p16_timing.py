"""Loop 16b: compute time of the whole chain, per frame, on one GPU (cached foundation features as input for both systems; the frozen foundation network is the same for both and not timed).
Stages: builder forward (batch 1 and 8; prior of the time recursion precomputed = streaming state), assembly (CPU), route head along the edges (points every 8 / 4 / 2 m and the vertex-based one), planner consumer; control: trace network wavefront (GPU + CPU) and its route head.
Usage (gcar, GPU, idle machine): p16_timing.py [--quality] [--n 48]"""
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
import p10_infer as P10  # noqa: E402
import p10_net as PN  # noqa: E402
import p14_chain_consumer as CC  # noqa: E402
import p16_common as C  # noqa: E402
import p16_data as PD  # noqa: E402
import p16_infer as PI  # noqa: E402
import p16_time as PTM  # noqa: E402
import p16_train as PT  # noqa: E402
import sd_lib as SL  # noqa: E402

HOME = os.path.expanduser("~")
OUT = f"{C.LADDER}/diag/timing.json"
ASM = dict(cov_r=0.5, vtx_r=1.5, junc_thr=0.5, reloc_thr=0.4, reloc_shift=12.0, reloc_tol=1.6)


def sync():
    torch.cuda.synchronize()


def ms(xs):
    return dict(mean=float(np.mean(xs)) * 1e3, median=float(np.median(xs)) * 1e3, p90=float(np.percentile(xs, 90)) * 1e3)


def load_builder(M, K0, name="e123_D7_r3_f1"):
    c = json.load(open(f"{C.LADDER}/cards/{name}.json", encoding="utf-8-sig"))
    extra = dict(c["cfg"])
    extra.update(PT.parse_set([f"M={M}", f"K0={K0}"]))
    net, cfg = PT.build("D7", extra)
    net = net.cuda()
    net.load_state_dict(torch.load(f"{C.OUT}/e123/{name}/model.pt", map_location="cuda"), strict=False)
    net.eval()
    return net, cfg


def to_gpu(b):
    return {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in b.items()}


def time_builder(M, K0, n, quality):
    net, cfg = load_builder(M, K0)
    ks = C.val_subset()
    ds = PD.BuilderData("val_clean", ks=ks[:n], train=False, with_prev=cfg.temporal)
    items = [ds[i] for i in range(n)]
    out = dict(M=M, K0=K0, params_M=sum(p.numel() for p in net.parameters()) / 1e6)
    for bs in (1, 8):
        batches = [to_gpu(PD.collate(items[j : j + bs])) for j in range(0, n - bs + 1, bs)]
        priors = [PTM.prior_from_prev(net, b, train=False) if cfg.temporal else None for b in batches]
        tp = []
        for b in batches:  # cost of the prior if it were recomputed from the previous frame
            sync()
            t0 = time.time()
            if cfg.temporal:
                PTM.prior_from_prev(net, b, train=False)
            sync()
            tp.append((time.time() - t0) / len(b["x"]))
        torch.cuda.reset_peak_memory_stats()
        tf, tn = [], []
        for rep in range(2):  # the first pass warms up
            tf.clear()
            tn.clear()
            for b, pr in zip(batches, priors):
                sync()
                t0 = time.time()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                    res = net(b["x"].float(), gt=None, share=0.0, prior=pr)
                sync()
                t1 = time.time()
                raws = [PI.to_numpy(res, i) for i in range(b["x"].shape[0])]
                t2 = time.time()
                tf.append((t1 - t0) / len(raws))
                tn.append((t2 - t1) / len(raws))
        out[f"forward_bs{bs}_ms_per_frame"] = ms(tf)
        out[f"to_numpy_bs{bs}_ms_per_frame"] = ms(tn)
        out[f"prior_if_recomputed_bs{bs}_ms_per_frame"] = ms(tp)
        out[f"peak_gpu_MB_bs{bs}"] = torch.cuda.max_memory_allocated() / 2**20
        if bs == 1:
            raws1 = []
            for b, pr in zip(batches, priors):
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                    res = net(b["x"].float(), gt=None, share=0.0, prior=pr)
                raws1.append(PI.to_numpy(res, 0))
    ta, nv = [], []
    for rep in range(2):
        ta.clear()
        nv.clear()
        for r in raws1:
            t0 = time.time()
            g = PI.assemble(r, None, 0.7, cfg, **ASM)
            ta.append(time.time() - t0)
            nv.append(len(g["pts"]))
    out["assembly_cpu_ms_per_frame"] = ms(ta)
    if quality:
        ks250 = C.val_subset()
        ds2 = PD.BuilderData("val_clean", ks=ks250, train=False, with_prev=cfg.temporal)
        raw = []
        _, gts, _ = PI.predict(net, ds2, ks250, taus=(0.7,), bs=8, workers=2, raw=raw)
        graphs = [PI.assemble(r, None, 0.7, cfg, **ASM) for r in raw]
        m = C.card(graphs, gts)
        out["quality_val250"] = {k: float(m[k]) for k in ("S", "F1@1.0", "J", "vert100", "false_len_frac", "lanes_found") if k in m}
    del net
    torch.cuda.empty_cache()
    return out


def prep_route_items(tokfile, n):
    z = pickle.load(open(f"{HOME}/exp_gcar_b123/p10/{tokfile}.pkl", "rb"))
    part = z["part"]
    sdl = L.cache("sd", part)
    its = []
    for it in z["items"]:
        if len(it["xy"]) == 0 or sdl[it["row"]] is None:
            continue
        sdm, _, _ = L.draw_sd(sdl[it["row"]], it["row"], 0.0, 0.0, 1)
        its.append((it, torch.from_numpy(SL.raster_sd(sdm))[None].cuda()))
        if len(its) >= n:
            break
    return its


def time_route(tokfile, model, n):
    net = PN.RouteXNet().cuda()
    net.load_state_dict(torch.load(f"{L.OUT}/{model}/model.pt", map_location="cuda"))
    net.eval()
    its = prep_route_items(tokfile, n)
    data = []
    for it, ras in its:
        nn_ = len(it["xy"])
        data.append((torch.from_numpy(it["tok"].astype(np.float32))[None].cuda(), torch.from_numpy(it["xy"])[None].cuda(), torch.from_numpy(it["cp"])[None].cuda(), ras, torch.ones(1, nn_, dtype=torch.bool).cuda()))
    t = []
    for rep in range(2):
        t.clear()
        for tok, xy, cp, ras, mk in data:
            sync()
            t0 = time.time()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                lg = net(tok, xy, cp, ras, mk)
            lg.float().cpu()
            sync()
            t.append(time.time() - t0)
    return dict(points_per_frame=float(np.mean([len(it["xy"]) for it, _ in its])), params_M=sum(p.numel() for p in net.parameters()) / 1e6, ms_per_frame=ms(t))


def time_consumer(n=256):
    z = torch.load(f"{HOME}/exp_gcar_b123/p14/p16chain_f2hr_val.pt", weights_only=False)
    st, p, k, cell = CC.to_dev(z)
    net = CC.ConsB(CC.nfeat("route+ego"), False).cuda().eval()
    t = []
    for rep in range(2):
        t.clear()
        for i in range(n):
            kk = k[i : i + 1]
            sync()
            t0 = time.time()
            x, tk, m = CC.feats(st, kk, p[i : i + 1], "route+ego")
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                o = net(x, tk, m)
            o.float().cpu()
            sync()
            t.append(time.time() - t0)
    return dict(params_M=sum(q.numel() for q in net.parameters()) / 1e6, ms_per_frame=ms(t))


def time_control(n):
    net = PN.TraceNet().cuda()
    net.load_state_dict(torch.load(f"{C.OUT}/e94_a/model.pt", map_location="cuda"))
    net.eval()
    es = P10.EvalSet("val_clean")
    ks = C.val_subset()[:n]
    out = dict(params_M=sum(p.numel() for p in net.parameters()) / 1e6)
    for bs in (1, 16):
        tl = []
        for s in range(0, len(ks), bs):  # time of reading the cached features (disk), reported separately
            sync()
            t0 = time.time()
            es.batch(ks[s : s + bs])
            sync()
            tl.append((time.time() - t0) / len(ks[s : s + bs]))
        P10.run_eval(net, es, ks[:bs], bs=bs, k_seed=192, nms=5, tau_seed=0.3, tau_on=0.5, wave_kw=dict(r_cov=1.2, r_head=1.2, order="near"), orient=True, verbose=False)  # warm-up
        sync()
        t0 = time.time()
        P10.run_eval(net, es, ks, bs=bs, k_seed=192, nms=5, tau_seed=0.3, tau_on=0.5, wave_kw=dict(r_cov=1.2, r_head=1.2, order="near"), orient=True, verbose=False)
        sync()
        tot = (time.time() - t0) / len(ks)
        out[f"wavefront_total_bs{bs}_ms_per_frame"] = tot * 1e3
        out[f"feature_read_bs{bs}_ms_per_frame"] = float(np.mean(tl)) * 1e3
        out[f"wavefront_compute_bs{bs}_ms_per_frame"] = (tot - float(np.mean(tl))) * 1e3
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--quality", action="store_true")
    ap.add_argument("--beams", default="96/48,144/72,192/96,288/144")
    a = ap.parse_args()
    res = dict(device=torch.cuda.get_device_name(0), n=a.n)
    res["builder"] = []
    for bm in a.beams.split(","):
        M, K0 = (int(x) for x in bm.split("/"))
        r = time_builder(M, K0, a.n, a.quality)
        res["builder"].append(r)
        print(json.dumps(r), flush=True)
    res["route"] = {}
    for key, tok, model in (("champion_vertex_tokens", "tok_p16_f2h_val_clean", "p16_f2h_route_s0"), ("champion_8m", "tok_p16_f2hr8_val_clean", "p16_f2hr8_route_s0"), ("champion_4m", "tok_p16_f2hr_val_clean", "p16_f2hr_route_s0"), ("champion_2m", "tok_p16_f2hr2_val_clean", "p16_f2hr2_route_s0"),
                            ("control_vertex_tokens", "tok_e94_a_val_clean", "e104b"), ("control_4m", "tok_e94_ar_val_clean", "e104br")):
        try:
            res["route"][key] = time_route(tok, model, a.n)
            print(key, json.dumps(res["route"][key]), flush=True)
        except Exception as e:  # model not trained yet
            print(key, "skipped:", repr(e), flush=True)
    res["consumer"] = time_consumer()
    print("consumer", json.dumps(res["consumer"]), flush=True)
    res["control_trace"] = time_control(a.n)
    print("control", json.dumps(res["control_trace"]), flush=True)
    json.dump(res, open(OUT, "w"), indent=1)
    print("written", OUT)


if __name__ == "__main__":
    main()
