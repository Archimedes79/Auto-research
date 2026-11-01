"""Route success, the failure series and the timing of the chain as named parts of `evaluate.py` (roadmap W11): one registry entry yields them without a
per-loop driver. The numerics are those of the drivers (imported, not re-written); this module only wires them to a registry entry and to the card.

    evaluate.py champion --route                                   # route success on VAL: builder -> assembly -> route input -> route head (GPU)
    evaluate.py champion --route --route-tokens tok_p18_f1pr_val_clean   # the same from STORED route tokens (CPU, no builder): reproduces pipeeval_f1_288_s0_fp0.3
    evaluate.py champion --failure dark,blur,cam_front3_off        # failure series: the builder on the series input (GPU), graph metrics per phase (cl = clean half, fail)
    evaluate.py champion --failure dark --series-dir e130_champion # the same from the STORED graphs of a series run (runs/e17/<dir>/<pert>_<variant><suffix>.pkl, CPU)
    evaluate.py champion --failure dark --route                    # + route success per phase of the series (GPU)
    evaluate.py champion --timing                                  # ms per frame of the whole chain (idle GPU only)

Keys in the card (`metrics`; the direction of each is in `better`):
  route_success / route_cov / route_prec / route_frames    keys of drivers/p16_pipeline_eval.py (mean over the frames with a route; route_frames = their number)
  series.<pert>_<variant>.<cl|fail>.<S|Q|F1@1.0|J|topo_F1|lanes_found|vert100>   per phase of a failure series (graph metrics as evaluate.py); with --route also
  series.<tag>.<phase>.route_success                       (p17_route.py: the route head along the assembled edges, exact map)
  ms_per_frame (= ms.total.median) and ms.<stage>.<median|p90>   stages of drivers/p18_latency.py: builder, assembly, route_input, route_head, consumer_prep, consumer, total, memory
`info` carries the sources and the full per-phase cards (`info.series`), the driver's timing json (`info.timing`).
"""
import bflc_paths as BP  # project paths from the location of this checkout
import glob
import json
import os
import pickle
import re
import sys

import numpy as np

R = BP.ROOT
E17 = f"{R}/runs/e17"
HOME = os.path.expanduser("~")
ROUTE_HEAD = "p16_f2hr_route_s0"  # the route head of the drivers (seed 0; seed 1 = p16_f2hr_route_s1: a route comparison needs both)
SERIES_KEYS = ("S", "Q", "F1@1.0", "J", "topo_F1", "lanes_found", "vert100")
PHASES = (("cl", 0), ("fail", 1))  # phase of a series frame: first half clean, second half perturbed (p17_series.py)
STAGES = ("builder", "assembly", "route_input", "route_head", "consumer_prep", "consumer", "total", "memory")
COUNTS = ("route_frames",)  # numbers without a direction
SPLIT_DS = {"val": "val_clean", "test": "test_clean"}  # the BuilderData name of a split (evaluate.SPLITS[split][0])


# ------------------------------------------------------------------ route success

def route_frame(p, it, rp):
    """route success of one frame (None without route points): the edges of the token item `it` whose mean vertex probability p is >= 0.5, as points, against
    the route points rp: {succ, cov, prec} (p8_lib.route_success). The rule of drivers/p16_pipeline_eval.py, loops/loop17/p17_route.py"""
    import p10_route_eval2 as RE
    import p16_route_diag as RD
    import p8_lib as L
    xy = it["xy"].astype(np.float64)
    return L.route_success(RD.pts_of(xy, [(a, b, ln) for (a, b, ln) in RE.edges_of(it) if 0.5 * (p[a] + p[b]) >= 0.5]), rp)


def route_metrics(results):
    """the route keys of the card from the per-frame results of route_frame (route_frames = the frames with a route)"""
    out = dict(route_frames=len(results), route_success=float(np.mean([r["succ"] for r in results])) if results else float("nan"))
    if results:
        out.update(route_cov=float(np.mean([r["cov"] for r in results])), route_prec=float(np.mean([r["prec"] for r in results])))
    return out


def route_over(items, probs, part):
    """per-frame route results and the number of frames: items yields (row, token item or None); probs(it, sd raster) -> vertex probabilities. A frame without
    vertices, route points or SD map is not counted (as the drivers)"""
    import torch
    import p8_lib as L
    import sd_lib as SL
    sdl, rptl = L.cache("sd", part), L.cache("rpts", part)
    res, n = [], 0
    for row, it in items:
        n += 1
        rp, sd = rptl[row], sdl[row]
        if it is None or len(it["xy"]) == 0 or rp is None or len(rp) == 0 or sd is None:
            continue
        sdm, _, _ = L.draw_sd(sd, row, 0.0, 0.0, 1)
        res.append(route_frame(probs(it, torch.from_numpy(SL.raster_sd(sdm))[None]), it, rp))
    return res, n


def load_head(head, dev):
    import torch
    import p10_net as PN
    import p8_lib as L
    net = PN.RouteXNet().to(dev)
    net.load_state_dict(torch.load(f"{L.OUT}/{head}/model.pt", map_location=dev))
    return net.eval()


def head_probs(net, dev="cpu"):
    """vertex probabilities of the route head (eager, fp16 autocast on the GPU) for one token item and its SD raster"""
    import torch

    def probs(it, ras):
        n = len(it["xy"])
        t = lambda a: torch.from_numpy(a)[None].to(dev)
        with torch.no_grad(), torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
            return torch.sigmoid(net(t(it["tok"].astype(np.float32)), t(it["xy"]), t(it["cp"]), ras.to(dev), torch.ones(1, n, dtype=torch.bool, device=dev)).float())[0].cpu().numpy()
    return probs


def route_from_tokens(tok, head=ROUTE_HEAD):
    """route metrics from STORED token items (~/exp_gcar_b123/p10/<tok>.pkl: the vertex tokens of a builder on a split), CPU; (metrics, info)"""
    import torch
    torch.set_num_threads(2)
    z = pickle.load(open(f"{HOME}/exp_gcar_b123/p10/{tok}.pkl", "rb"))
    res, n = route_over(((it["row"], it) for it in z["items"]), head_probs(load_head(head, "cpu")), z["part"])
    return route_metrics(res), dict(source=f"tokens {tok}", head=head, frames=n)


def builder_items(net, cfg, e, split, step=4.0):
    """(row, route token item) of every frame of the split from the builder as deployed (GPU): CUDA-graph builder -> assembly (aux) -> route input at points every <= step m
    (the loop of drivers/p16_pipeline_eval.py; p16_pipeline.route_input)"""
    import p16_data as PD
    import p16_infer as PI
    import p16_pipeline as PP
    import p16_runtime as RT
    import p16_timing as TM
    ds = PD.BuilderData(split, ks=None, train=False, with_prev=cfg.temporal)
    fb = RT.FastBuilder(net, cfg, TM.to_gpu(PD.collate([ds[0]])))
    for i in range(len(ds)):
        b = TM.to_gpu(PD.collate([ds[i]]))
        row = int(ds.rows[int(b["k"][0])])
        r = fb.raw(b)
        g, ax = PI.assemble(r, None, e["tau"], cfg, aux=True, **e["asm"])
        yield row, (PP.route_input(r, ax, g, fb.h_static, step, r["pi"]) if len(ax["V"]) else None)


def route_from_builder(net, cfg, e, split, head=ROUTE_HEAD, step=4.0):
    """route metrics of a builder entry on the split (GPU, CUDA graphs as the deployed path); (metrics, info)"""
    import torch
    import p16_pipeline as PP
    fr = PP.FastRoute(load_head(head, "cuda"))

    def probs(it, ras):
        t = lambda a: torch.from_numpy(a)[None].cuda()
        return fr(t(it["tok"].astype(np.float32)), t(it["xy"]), t(it["cp"]), ras.cuda(), len(it["xy"])).cpu().numpy()

    res, n = route_over(builder_items(net, cfg, e, SPLIT_DS[split], step), probs, split)
    return route_metrics(res), dict(source="builder", head=head, step=step, frames=n)


# ------------------------------------------------------------------ failure series

def phase_cards(graphs, gts, phase):
    """{cl, fail}: the metric card (sparse_metrics.card_any, extra) of the clean half and the perturbed half of a series"""
    import sparse_metrics as SM
    phase = np.asarray(phase)
    return {nm: SM.card_any([graphs[i] for i in np.where(phase == v)[0]], [gts[i] for i in np.where(phase == v)[0]], extra=True) for nm, v in PHASES}


def series_stored(series_dir, tag, suffix=""):
    """(graphs, gts, phase) of a stored series run: runs/e17/<series_dir>/<tag><suffix>.pkl (loops/loop17/p17_eval.py)"""
    z = pickle.load(open(f"{E17}/{series_dir}/{tag}{suffix}.pkl", "rb"))
    return z["graphs"], z["gts"], z["phase"], f"runs/e17/{series_dir}/{tag}{suffix}.pkl"


def series_from_builder(net, cfg, e, tag, bs=4, workers=2):
    """(graphs, gts, phase) of the builder on the series input of p17_series.py (GPU): the loop of loops/loop17/p17_eval.py with the entry's assembly"""
    import p16_infer as PI
    import p17_eval as EV17
    ds = EV17.SeriesData(tag, with_prev=bool(cfg.temporal))
    raw = []
    _, gts, _ = PI.predict(net, ds, list(range(len(ds))), taus=(0.5,), bs=bs, workers=workers, raw=raw)
    return [PI.assemble(r, None, e["tau"], cfg, **e["asm"]) for r in raw], gts, ds.meta["phase"], f"series input {tag}"


def series_route(net, cfg, e, tag, head=ROUTE_HEAD):
    """route success per phase of a series, exact map (GPU): loops/loop17/p17_route.run_series (aux assembly, route head along the edges)"""
    import p17_eval as EV17
    import p17_route as RT17
    ds = EV17.SeriesData(tag, with_prev=bool(cfg.temporal))
    out, meta, _ = RT17.run_series(net, cfg, ds, [load_head(head, "cuda")], [0.0], e["tau"], e["asm"])
    v = np.asarray([np.nan if x is None else x for x in out[(0, 0)]], float)
    ph = np.asarray([m[1] for m in meta])
    return {nm: (float(np.nanmean(v[ph == k])), int(np.isfinite(v[ph == k]).sum())) for nm, k in PHASES}


def failure_metrics(tag, cards):
    """flat card keys of the per-phase cards of one series"""
    return {f"series.{tag}.{nm}.{k}": float(c[k]) for nm, c in cards.items() for k in SERIES_KEYS if k in c}


# ------------------------------------------------------------------ timing

def timing_argv(e, reg, n=60, tag=""):
    """the command line of drivers/p18_latency.py for a registry entry (time recursion off, the setting of every stored latency); ValueError when the driver cannot
    time the entry as it is (another design, another network setting, other assembly settings than its fixed ones): the card must not claim a timing of another system"""
    import p16_timing as TM
    ent = [e] if e["kind"] == "builder" else [reg[m] for m in e.get("members", [])]
    if e["kind"] not in ("builder", "union") or not 1 <= len(ent) <= 2:
        raise ValueError("timing: builder entries and unions of two builders only")
    for x in ent:
        if x["design"] != "D7" or set(x["set"]) != {"M", "K0"}:
            raise ValueError("timing: p18_latency times design D7 with M / K0 only")
    base = {k: v for k, v in e["asm"].items() if k not in ("fork_prune", "fast")}
    if e["tau"] != 0.7 or {k: float(v) for k, v in base.items()} != {k: float(v) for k, v in TM.ASM.items()}:
        raise ValueError(f"timing: p18_latency times tau 0.7 and the assembly {TM.ASM} (+ fork_prune / fast); the entry differs")
    a = ["p18_latency.py", "--name", ent[0]["run"], "--M", str(ent[0]["set"]["M"]), "--K0", str(ent[0]["set"]["K0"]), "--n", str(n), "--rec", "0",
         "--union", str(int(e["asm"].get("fast", 2))), "--fork_prune", str(float(e["asm"].get("fork_prune", 0.0))), "--tag", tag]
    return a + (["--name2", ent[1]["run"]] if len(ent) == 2 else [])


def timing_metrics(res):
    """flat card keys of a latency json (drivers/p18_latency.py): ms_per_frame = the median of the whole chain, ms.<stage>.<median|p90>"""
    out = {"ms_per_frame": float(res["total"]["median_ms"])}
    for st in STAGES:
        if st in res:
            out.update({f"ms.{st}.median": float(res[st]["median_ms"]), f"ms.{st}.p90": float(res[st]["p90_ms"])})
    return out


def timing_run(e, reg, n, tag):
    """runs p18_latency on an idle GPU for the entry; (latency json, its file)"""
    import p18_latency as LT
    argv, old = timing_argv(e, reg, n, tag), sys.argv
    sys.argv = argv
    try:
        LT.main()
    finally:
        sys.argv = old
    f = sorted(glob.glob(f"{R}/runs/e18/latency/latency_*{tag}.json"), key=os.path.getmtime)[-1]
    return json.load(open(f)), f


# ------------------------------------------------------------------ the parts of one card

def direction(keys):
    """{key: 'lower' | 'higher'} of the part keys (time lower; counts none; the rest as sparse_metrics.better)"""
    import sparse_metrics as SM
    out = {}
    for k in keys:
        if k in COUNTS:
            continue
        base = re.fullmatch(r"series\.(.+?)\.(?:cl|fail)\.(.+)", k)  # the metric of a series key (the tag may hold a dot: maxdecay_0.95; so may the metric: F1@1.0)
        base = base.group(2) if base else k
        out[k] = "lower" if k.startswith("ms") else SM.better({base: 0}).get(base, "higher")
    return out


def run(e, reg, split, parts, build_net):
    """the numbers of the requested parts for the (effective) registry entry e: dict(metrics, better, info). parts = {route: {head, tokens, step}, failure: {perts, variant,
    series_dir, suffix, bs, workers}, timing: {n, tag}}; build_net(e) -> (net, cfg) on the GPU, called once, only when a part needs the builder"""
    m, info, held = {}, {}, {}

    def net():
        if "net" not in held:
            held["net"], held["cfg"] = build_net(e)
        return held["net"], held["cfg"]

    r, f = parts.get("route"), parts.get("failure")
    if (r and not r.get("tokens")) or (f and not f.get("series_dir")):
        if e["kind"] != "builder":
            raise ValueError("route (without --route-tokens) and failure (without --series-dir) run the builder: builder entries only")
    if r:
        rm, ri = route_from_tokens(r["tokens"], r["head"]) if r.get("tokens") else route_from_builder(*net(), e, split, r["head"], r["step"])
        m.update(rm)
        info["route"] = ri
    if f:
        if split != "val":
            raise ValueError("the failure series are built on VAL scenes")
        info["series"] = {}
        for pert in f["perts"]:
            tag = f"{pert}_{f['variant']}"
            if f.get("series_dir"):
                graphs, gts, phase, src = series_stored(f["series_dir"], tag, f.get("suffix", ""))
            else:
                graphs, gts, phase, src = series_from_builder(*net(), e, tag, f["bs"], f["workers"])
            cards = phase_cards(graphs, gts, phase)
            m.update(failure_metrics(tag, cards))
            info["series"][tag] = dict(source=src, n={nm: int((np.asarray(phase) == v).sum()) for nm, v in PHASES}, cards=cards)
            if r and not f.get("series_dir"):  # route per phase of the series (not from stored graphs: they carry no route): the builder runs again, with the aux assembly of the route chain
                for nm, (v, cnt) in series_route(*net(), e, tag, r["head"]).items():
                    m[f"series.{tag}.{nm}.route_success"] = v
                    info["series"][tag].setdefault("route_n", {})[nm] = cnt
    t = parts.get("timing")
    if t:
        res, path = timing_run(e, reg, t["n"], t["tag"])
        m.update(timing_metrics(res))
        info["timing"] = dict(file=os.path.relpath(path, R), **res)
    return dict(metrics=m, better=direction(m), info=info)
