"""E282 CPU mini-pilot (card experiments/E282_zone-matching-node-loss, section 3): can the node head learn the SPARSE row under the zone-matching loss? Four arms on the same frames, order, seed and
initialisation, the node head alone on CACHED outputs of the frozen pyramid (the phase-3 recipe: `trainer.fit`, lr 1e-3, one-cycle, on the CPU), read on held-out VAL frames with the zone metrics.

    python qnode_pilot.py cache  [--n N] [--cache NAME]                                     # once: 601 TRAIN + 110 VAL frames -> the pyramid outputs (fp16 memory maps) + the target maps, temp/e282/cache[_NAME]/
    python qnode_pilot.py train  --arm d|s|m|m5 [--steps 2000] [--seed 0] [--cache NAME] [--tag NAME] [--resume]   # trains the arm, evaluates it, writes runs/e282_p_<arm>_<steps>[_<tag>]/result.json
    python qnode_pilot.py gate   [--steps 2000] [--cache NAME] [--tag NAME]                  # the outcome of card section 3 from the four result files

Arms: d = the DENSE per-cell loss (E273 phase 3); s = the per-cell loss on the SPARSE target (0.3 m / 7.5 m), no tier switches; m = the zone-matching loss (bias 0.05); m5 = as m with bias 0.5 (reported).
Data: TRAIN `ks[::14]` of the frames with a stored lane GT (601 of 8414), VAL `ks[::8]` (110 of 873); the pyramid is `runs/e273_p_s0_p2/model.pt` (its pyramid tensors only). The run directory carries the
number of steps in its name; an existing checkpoint is REFUSED unless `--resume` is given (the trainer would otherwise continue the old weights and train nothing). The read-outs: every arm at e >= 0.5 and at its
MATCHED-DENSITY threshold (fitted on a spread of TRAIN frames so that it delivers 14.4 nodes per 100 m of GT lane; the control s is judged there). Start it through `autoresearch/tools/run.py` (CPU,
`--id E282.p_<arm>`): the arms close or open the card. A diagnostic of the card, in-sample VAL, never a verdict on the graph. Tests: `test_qnode_pilot.py`.
"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")        # a CPU job must not open a CUDA context on the shared GPU (the trainer's checkpoint code asks torch for the CUDA rng state)

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for sub in ("training/train", "training/grid", "training/data", "training", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import lane_gt_data as LD  # noqa: E402
import qgraph as QG  # noqa: E402
import qnodes as QN  # noqa: E402
import qtargets as QT  # noqa: E402
import qzones as QZ  # noqa: E402
import trainer as T  # noqa: E402

SPARSE = (0.3, 7.5)
STRIDE = dict(train=14, val=8)
FEATS = ("feat2", "feat1", "feat05", "road2", "centre1")
SPARSE_KEYS = ("exists", "flags", "offset", "dirv", "sstart", "uout", "uin")
ZONE_KEYS = LD.ZONE_NAMES
ARMS = dict(d=dict(loss="e273", dense=True, bias=0.0), s=dict(loss="e273", dense=False, bias=0.0), m=dict(loss="zone", dense=False, bias=0.05), m5=dict(loss="zone", dense=False, bias=0.5))
TARGET_DENS, MATCH_TOL_VAL = 14.4, 3.0                      # the matched density (nodes per 100 m of GT lane) and how far s may be from it on the VAL frames before the control counts as not matched


def cache_dir(tag):
    return os.path.join(ROOT, "temp", "e282", "cache" + (f"_{tag}" if tag else ""))


def run_name(arm, steps, tag=""):
    return f"e282_p_{arm}_{steps}" + (f"_{tag}" if tag else "")


# ------------------------------------------------------------------------------------------------------------------------------ cache
def build_cache(tag, n_max=0):
    import qgraph_eval as QE
    out = cache_dir(tag)
    os.makedirs(out, exist_ok=True)
    model = QG.QGraph(width=48)
    sd = torch.load(os.path.join(ROOT, "runs", "e273_p_s0_p2", "model.pt"), map_location="cpu")["state"]
    own = model.state_dict()
    fit = {k: v for k, v in sd.items() if k.startswith("pyr.") and k in own and own[k].shape == v.shape}
    assert len(fit) == sum(1 for k in own if k.startswith("pyr.")), "not every pyramid tensor was found in the phase-2 checkpoint"
    model.load_state_dict(fit, strict=False)
    model.eval()
    meta = dict(sparse=SPARSE, stride=STRIDE, pyramid="runs/e273_p_s0_p2/model.pt", shapes={})
    for split, part in (("train", "train"), ("val", "val_clean")):
        ds = LD.QData(part, pixels=False, map_mode="exact", sparse=SPARSE, zone=True)
        sel = list(range(0, len(ds), STRIDE[split]))
        if n_max:
            sel = sel[:n_max]
        N = len(sel)
        mm, tg, tokens, lens = {}, {}, [], []
        t0 = time.time()
        for b0 in range(0, N, 4):
            ids = sel[b0:b0 + 4]
            items = [ds[i] for i in ids]
            x = torch.stack([it["x"] for it in items]).float()
            with torch.no_grad():
                P = model.pyr(x)
            for name in FEATS:
                arr = P[name].detach().float().numpy().astype(np.float16)
                if name not in mm:
                    meta["shapes"][name] = list(arr.shape[1:])
                    mm[name] = np.lib.format.open_memmap(os.path.join(out, f"{split}_{name}.npy"), mode="w+", dtype=np.float16, shape=(N,) + arr.shape[1:])
                mm[name][b0:b0 + len(ids)] = arr
            for it, i in zip(items, ids):
                token = ds.tokens[int(ds.bd.rows[ds.ks[i]])]
                g = ds.lg.load(token)
                dense = QT.grid_targets(QT.node_table(g))
                for k in SPARSE_KEYS + ZONE_KEYS:
                    tg.setdefault(k, []).append(np.asarray(it[k]))
                for k in SPARSE_KEYS:
                    tg.setdefault("d_" + k, []).append(np.asarray(dense[k]))
                tokens.append(token)
                lens.append(float(QE.LaneRef(g).length))
            print(f"  {split} {b0 + len(ids)}/{N}  {time.time() - t0:.0f} s", flush=True)
        for name in mm:
            mm[name].flush()
        np.savez_compressed(os.path.join(out, f"{split}_targets.npz"), tokens=np.array(tokens), lens=np.array(lens), **{k: np.stack(v) for k, v in tg.items()})
        meta[split] = dict(n=N, signature=ds.lg.signature())
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
    print("cache ready:", out, meta["shapes"])


class Cache:
    def __init__(self, tag, split):
        d = cache_dir(tag)
        self.split = split
        self.mm = {n: np.load(os.path.join(d, f"{split}_{n}.npy"), mmap_mode="r") for n in FEATS}
        z = np.load(os.path.join(d, f"{split}_targets.npz"))
        self.t = {k: z[k] for k in z.files}
        self.n = len(self.t["tokens"])
        self.meta = json.load(open(os.path.join(d, "meta.json")))

    def batch(self, idx, arm):
        b = {n: torch.from_numpy(np.asarray(self.mm[n][idx], dtype=np.float32)) for n in FEATS}
        for k in SPARSE_KEYS:
            b[k] = torch.from_numpy(self.t[("d_" + k) if ARMS[arm]["dense"] else k][idx].copy())
        if ARMS[arm]["loss"] == "zone":
            for k in ZONE_KEYS:
                b[k] = torch.from_numpy(self.t[k][idx].copy())
        return b


class NodeOnly(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.nodes = QN.NodeHead(c2=144, c1=48, c05=32, d=64)                                  # the sizes of QGraph(width=48, f05=32, d=64)

    def forward(self, b):
        return self.nodes(b["feat2"], b["feat1"], b["feat05"], b["road2"], b["centre1"])


# ------------------------------------------------------------------------------------------------------------------------------ train + evaluate
def train_arm(a):
    arm = ARMS[a.arm]
    QG.CFG.update(node_loss=arm["loss"], zone_bias=arm["bias"])
    T.seed_all(a.seed)
    model = NodeOnly()
    tr = Cache(a.cache, "train")
    name = run_name(a.arm, a.steps, a.tag)
    run = T.Run(name)
    if os.path.exists(run.path("ckpt.pt")) and not a.resume:
        raise SystemExit(f"{run.dir}: a checkpoint exists; the trainer would resume it and train nothing. Use another --tag, delete the run folder deliberately, or pass --resume.")
    run.log(f"E282 pilot arm {a.arm}: loss {arm['loss']}, dense target {arm['dense']}, bias {arm['bias']}, {a.steps} steps, {tr.n} train frames, seed {a.seed}, CPU, resume {a.resume}")

    def batch_fn(step):
        rng = np.random.RandomState((a.seed * 1000003 + step) % (2 ** 31 - 1))
        return tr.batch(np.sort(rng.choice(tr.n, 4, replace=False)), a.arm)

    def loss_fn(m, b):
        return QG.node_loss(m(b), b)

    torch.set_num_threads(a.threads)
    hist = T.fit(model, loss_fn, run, steps=a.steps, batch_fn=batch_fn, lr=1e-3, ckpt_every=500, seed=a.seed, args=vars(a), device="cpu", resume=a.resume,
                 config=dict(design="qnode_pilot", arm=a.arm, cfg=dict(QG.CFG)))
    torch.save(dict(state=model.state_dict(), args=vars(a)), run.path("model.pt"))
    if a.no_eval:                                                                               # smoke: the training loop alone
        return
    try:
        res = evaluate(model, a, tr)
    except Exception:                                                                           # the trainer marked the run done before the evaluation: a false flag is renamed (project rule)
        done = run.path(".done")
        if os.path.exists(done):
            os.replace(done, done + ".FALSE")
        raise
    res["trained_steps"] = len(hist)
    json.dump(res, open(run.path("result.json"), "w"), indent=1, default=float)
    run.log("done", json.dumps({k: res[k] for k in ("arm", "steps", "trained_steps")}), "metrics@0.5", json.dumps(res["at_0.5"]["pool"], default=float))


@torch.no_grad()
def node_maps(model, c, idx):
    model.eval()
    out = []
    for b0 in range(0, len(idx), 4):
        ids = np.asarray(idx[b0:b0 + 4])
        b = c.batch(ids, "s")
        no = model(b)
        out.append((no["e"].float().sigmoid().numpy(), no["pos"].float().numpy(), no["e"].float().numpy()))
    return [np.concatenate(z, 0) for z in zip(*out)]


def delivered(e, pos, t):
    """-> list of (k, 2) metre positions of the nodes with e >= t, one array per frame"""
    res = []
    for f in range(e.shape[0]):
        ii, jj = np.nonzero(e[f] >= t)
        res.append(np.stack([pos[f, 0, ii, jj], pos[f, 1, ii, jj]], 1) if len(ii) else np.zeros((0, 2)))
    return res


def evaluate(model, a, tr):
    import zone_metrics as ZM
    from lane_task import measures as MS
    import qgraph_eval as QE
    va = Cache(a.cache, "val")
    lg, _ = LD.lane_store()
    sig = lg.signature()
    assert tr.meta["val"]["signature"] == sig and tr.meta["train"]["signature"] == sig, f"the lane GT store changed since the cache was built ({tr.meta['val']['signature']} -> {sig})"
    e_v, pos_v, logit_v = node_maps(model, va, np.arange(va.n))
    spread = np.arange(0, tr.n, 5)                                                              # the matched-density threshold is fitted on TRAIN frames spread over the split
    e_t, _, logit_t = node_maps(model, tr, spread)
    mt = ZM.matched_threshold([e_t[i].ravel() for i in range(len(spread))], list(tr.t["lens"][spread]), target=TARGET_DENS, tol=2.0)
    t_match, dens_match = mt["thr"], mt["dens"]
    graphs = [lg.load(str(tok)) for tok in va.t["tokens"]]
    zs = [QZ.zones(g, SPARSE) for g in graphs]
    out = dict(arm=a.arm, steps=a.steps, seed=a.seed, cache=a.cache, tag=a.tag, frames=int(va.n), lane_signature=sig, t_match=float(t_match), dens_match_train=float(dens_match), t_match_ok=bool(mt["ok"]))
    for label, t in (("at_0.5", 0.5), ("at_match", t_match)):
        P = delivered(e_v, pos_v, t)
        counts = [ZM.frame_counts(z, float(L), p) for z, L, p in zip(zs, va.t["lens"], P)]
        out[label] = dict(t=float(t), pool=ZM.pool(counts), counts=counts)
    # the node gate of E273 phase 3 against the DENSE GT (node precision / recall within 2 m, the scorer's definitions): the validity of the pilot (arm d)
    P5 = delivered(e_v, pos_v, 0.5)
    tp = nd_n = gt_n = 0
    for z, p in zip(zs, P5):
        G = z["nd"]["pos"]
        pr = MS.match_1to1(p, G, QE.NODE_TOL) if len(p) and len(G) else np.zeros((0, 2), int)
        tp, nd_n, gt_n = tp + len(pr), nd_n + len(p), gt_n + len(G)
    out["dense_node_gate"] = dict(precision=tp / max(nd_n, 1), recall=tp / max(gt_n, 1), delivered=nd_n, gt=gt_n)
    if ARMS[a.arm]["loss"] == "zone":                                                           # the share of free zones whose winner is not the GT cell, at the end of training (spread TRAIN frames)
        shares = []
        for b0 in range(0, len(spread), 4):
            ids = spread[b0:b0 + 4]
            _, info = QG.zone_targets(torch.from_numpy(logit_t[b0:b0 + len(ids)]), tr.batch(ids, a.arm), ARMS[a.arm]["bias"])
            shares.append((float(info["win_off_gt"]), len(ids)))
        out["win_off_gt_end"] = float(sum(s * n for s, n in shares) / sum(n for _, n in shares))
    return out


# ------------------------------------------------------------------------------------------------------------------------------ gate
def gate(a):
    import zone_metrics as ZM
    res = {}
    for arm in ARMS:
        p = os.path.join(ROOT, "runs", run_name(arm, a.steps, a.tag), "result.json")
        if os.path.exists(p):
            res[arm] = json.load(open(p))
    if any(k not in res for k in ("d", "s", "m")):
        print("missing result files for", [k for k in ("d", "s", "m") if k not in res])
        return None
    d, s, m = res["d"], res["s"], res["m"]
    nan = lambda *v: any(x != x for x in v)  # noqa: E731
    out = dict(arms={k: dict(at_0_5=v["at_0.5"]["pool"], at_match=v["at_match"]["pool"], t_match=v["t_match"], node_gate=v["dense_node_gate"], win_off_gt_end=v.get("win_off_gt_end")) for k, v in res.items()})
    valid = d["dense_node_gate"]["precision"] >= 0.75 and d["dense_node_gate"]["recall"] >= 0.5
    d_rec, d_off = d["at_0.5"]["pool"]["free_zone_recall"], d["at_0.5"]["pool"]["off_zone"]
    s_m = s["at_match"]["pool"]
    s_dens_ok = abs(s_m["nodes_per_100m"] - TARGET_DENS) <= MATCH_TOL_VAL
    s_passes = s_m["free_zone_exact"] >= 0.50 and s_m["free_zone_recall"] >= d_rec - 0.10 and s_m["off_zone"] <= 0.15

    def floors(r):
        p = r["at_0.5"]["pool"]
        bd = ZM.bootstrap_diff(r["at_0.5"]["counts"], s["at_match"]["counts"], "free_zone_exact", n=2000, seed=0)
        f = dict(free_recall=p["free_zone_recall"] >= d_rec - 0.10, free_exact=p["free_zone_exact"] >= 0.50, off=p["off_zone"] <= 0.15, density=8.0 <= p["nodes_per_100m"] <= 22.0,
                 fixed=p["fixed_zone_recall"] >= s_m["fixed_zone_recall"] - 0.05, gain=(p["free_zone_exact"] - s_m["free_zone_exact"]) >= 0.15 and bd["lo90"] > 0.05,
                 left_gt=(r.get("win_off_gt_end") or 0.0) >= 0.25)
        return {k: bool(v) for k, v in f.items()}, bd

    fm, bd = floors(m)
    out.update(valid=bool(valid), s_matched_density_ok=bool(s_dens_ok), s_density_val=s_m["nodes_per_100m"], s_passes=bool(s_passes), floors=fm, bootstrap_m_minus_s_matched=bd, d_reference=dict(free_zone_recall=d_rec, off_zone=d_off))
    if "m5" in res:
        out["floors_m5"], out["bootstrap_m5_minus_s_matched"] = floors(res["m5"])
    core = ("free_recall", "free_exact", "off", "density", "fixed")
    if not valid:
        out["outcome"] = "INVALID"
    elif (not s_dens_ok) or nan(s_m["free_zone_exact"], s_m["free_zone_recall"], s_m["off_zone"], s_m["fixed_zone_recall"], bd["lo90"]):
        out["outcome"] = "INCONCLUSIVE-control"                                                  # the control s is not at the matched density on the VAL frames (or a metric is undefined): no closure
    elif s_passes:
        out["outcome"] = "S-PASSES"
    elif all(fm.values()):
        out["outcome"] = "GO"
    elif all(fm[k] for k in core):
        out["outcome"] = "STOP-B"
    else:
        out["outcome"] = "STOP-A"
    path = os.path.join(ROOT, "temp", "e282", "pilot_outcome" + (f"_{a.tag}" if a.tag else "") + ".json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(out, open(path, "w"), indent=1, default=float)
    print(json.dumps(dict(outcome=out["outcome"], valid=out["valid"], s_matched_density_ok=out["s_matched_density_ok"], s_density_val=out["s_density_val"], s_passes=out["s_passes"], floors=out["floors"],
                          floors_m5=out.get("floors_m5"), bootstrap=bd), indent=1, default=float))
    for k, v in out["arms"].items():
        print(k, "at 0.5:", {x: round(float(y), 3) for x, y in v["at_0_5"].items()}, "| at matched density t=%.2f:" % v["t_match"], {x: round(float(y), 3) for x, y in v["at_match"].items() if x in ("free_zone_recall", "free_zone_exact", "off_zone", "nodes_per_100m")})
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cache")
    c.add_argument("--n", type=int, default=0, help="only the first N frames of each split (smoke)")
    c.add_argument("--cache", default="", help="name of the cache folder temp/e282/cache[_NAME]")
    t = sub.add_parser("train")
    t.add_argument("--arm", choices=sorted(ARMS), required=True)
    t.add_argument("--steps", type=int, default=2000)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--threads", type=int, default=2)
    t.add_argument("--no-eval", action="store_true", help="smoke: train only")
    t.add_argument("--resume", action="store_true", help="continue an existing checkpoint of the same run folder (refused otherwise)")
    t.add_argument("--cache", default="")
    t.add_argument("--tag", default="", help="suffix of the run folder name (the cache is chosen by --cache)")
    g = sub.add_parser("gate")
    g.add_argument("--steps", type=int, default=2000)
    g.add_argument("--cache", default="")
    g.add_argument("--tag", default="")
    a = ap.parse_args()
    if a.cmd == "cache":
        build_cache(a.cache, a.n)
    elif a.cmd == "train":
        train_arm(a)
    else:
        gate(a)


if __name__ == "__main__":
    main()
