"""E325 step 2: the data of the wiring head `Wire` (`training/grid/jwire_head.py`): per frame a compact RECORD of what a connector end sees (the kept pieces of the delivered graph and their attributes, the pairs, the keypoint probabilities,
the GT ends the keypoints match), and the instances / agreement statistics made from it.

    wsl bash tools/env/_py14.sh jwire_data.py val --seed 0 [--run e312_p_s0] [--thr 0.1] [--max N]     # VAL records from the stored dumps (CPU: runs/<run>/ends_val.pkl, B's full dump temp/e276/predfull_e273_p_s<seed>.pkl, the GT ends
                                                                                                         # temp/jwiring/gt_ends_val.pkl) -> temp/jwire/val_records_s<seed>.pkl and the rule's agreement with the oracle label
    wsl bash tools/env/_py14.sh jwire_data.py live --seed 0 [--split train|val_clean] [--max N]               # the same records from a LIVE pass of B and the pair run (GPU) + the pilot numbers
    wsl bash tools/env/_py14.sh jwire_data.py selftest

A record = dict(row, scene, ends (n, 8) = last point, heading there, first point, heading there of every kept piece, attrs (n, 6), pairs (m, 8) = entry position / heading, exit position / heading, kp_p (m, 2), gts (m, 2, 4) = the GT end of the
entry / exit keypoint (x, y, tx, ty) or NaN). `instances_of(rec, jitter)` -> the instances of `jwire_head.build_instances`. The pieces are the pieces KEPT by the cover drop of the unjittered pairs (the deployed read-out): a jitter changes
the keypoints, not the delivered graph. The agreement is counted on the ends with an ORACLE label (a matched GT end that finds a piece in reach): overall and on the ends with two or more places (`real choices`); the rule = place 0.
VAL only, TEST never; the TRAIN records (a live pass of B and the pair run, GPU) are a separate step of the card.
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for sub in ("evaluation", "training", "training/grid", "training/data", "training/data/prep", "training/train", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub)))
import jwire as JW  # noqa: E402
import jwire_head as JH  # noqa: E402
import qends_eval as QEV  # noqa: E402

MATCH_POS, MATCH_ANG = 4.0, 30.0
RECORD_VERSION = 1                                                                                   # in the cache file names: change it when `record_of` / `piece_attrs` change (the attrs are stored at record time)
SPLITS = ("train", "val_clean")                                                                       # a live pass never reads another split (TEST is never touched)


def match(kp, kd, valid, G, m):
    """numpy copy of qendnet.match_keypoints for one frame and one kind (see `jwiring.match`): -> (n,) index of the GT end a keypoint is (within 4 m and 30 degrees, nearest, one-to-one), -1 = none"""
    res = np.full(len(kp), -1, int)
    if not m.any() or not len(kp):
        return res
    idx = np.where(m)[0]
    G = G[idx]
    dk = np.linalg.norm(kp[:, None, :] - G[None, :, :2], axis=-1)
    ang = (kd[:, None, :] * G[None, :, 2:]).sum(-1)
    ok = (dk <= MATCH_POS) & (ang >= np.cos(np.radians(MATCH_ANG))) & valid[:, None]
    dk = np.where(ok, dk, 1e6)
    a = dk.argmin(1)
    best = dk.argmin(0)
    mine = best[a] == np.arange(len(kp))
    return np.where(ok.any(1) & mine, idx[a], -1)


def record_of(pred, graph, bk, e, g, thr=0.1, row=-1, scene=-1):
    """the record of one frame: pred / graph / bk = the delivered graph of B read out as the card of record does (`qgraph_readout.apply` close:1.5@0.5, `qgraph_eval.readout`), e = the pair run's frame dict (kp_pos, kp_dir, kp_p,
    valid, logit, allowed), g = the GT ends of the frame (ent, ent_m, ext, ext_m)"""
    k = e["kp_pos"].shape[0] // 2
    p = 1.0 / (1.0 + np.exp(-e["logit"])) * e["allowed"]
    ia, ib = np.nonzero(p >= thr)
    pairs = QEV.make_pairs(e, ia, ib, k)
    keep, ends = JW.kept_ends(graph.lines, bk["kind"], bk["piece_links"], np.asarray(pred["links"]["long"], bool), pairs, QEV.cover_pos())
    attrs = JH.piece_attrs(graph.lines, bk["kind"], bk["piece_nodes"], bk["piece_links"], np.asarray(pred["nodes"]["e"], float).reshape(-1), np.asarray(pred["links"]["p"], float).reshape(-1), keep)
    valid = np.asarray(e["valid"], bool)
    m_in = match(e["kp_pos"][:k], e["kp_dir"][:k], valid[0], g["ent"], g["ent_m"])
    m_out = match(e["kp_pos"][k:], e["kp_dir"][k:], valid[1], g["ext"], g["ext_m"])
    gts = np.full((len(pairs), 2, 4), np.nan)
    for n, (i, j) in enumerate(zip(ia, ib)):
        if m_in[i] >= 0:
            gts[n, 0] = g["ent"][m_in[i]]
        if m_out[j] >= 0:
            gts[n, 1] = g["ext"][m_out[j]]
    return dict(row=row, scene=scene, ends=np.array([np.r_[en[0], en[1], en[2], en[3]] for en in ends], float).reshape(-1, 8), attrs=attrs,
                pairs=np.array([np.r_[pr[0], pr[1], pr[2], pr[3]] for pr in pairs], float).reshape(-1, 8), kp_p=np.array([(e["kp_p"][i], e["kp_p"][k + j]) for i, j in zip(ia, ib)], float).reshape(-1, 2), gts=gts)


def instances_of(rec, jitter=None, wire_pos=JH.REACH):
    """-> the instances of a record (`jwire_head.build_instances`)"""
    ends = [(r[0:2], r[2:4], r[4:6], r[6:8]) for r in rec["ends"]]
    pairs = [(r[0:2], r[2:4], r[4:6], r[6:8]) for r in rec["pairs"]]
    gts = [tuple(None if np.isnan(g).any() else g for g in rec["gts"][n]) for n in range(len(pairs))]
    return JH.build_instances(ends, rec["attrs"], pairs, [tuple(x) for x in rec["kp_p"]], gts, wire_pos, jitter)


def agreement(inst):
    """the rule's (place 0) agreement with the oracle label: dict(ends, labelled, oracle, rule_same, multi, multi_same, outside, none) over a list of instances"""
    o = [i for i in inst if i["kind"] == "oracle"]
    m = [i for i in o if len(i["piece"]) >= 2]
    return dict(ends=len(inst), oracle=len(o), same=sum(i["label"] == 0 for i in o), multi=len(m), multi_same=sum(i["label"] == 0 for i in m), outside=sum(i["kind"] == "outside" for i in inst), rule_label=sum(i["kind"] == "rule" for i in inst))


def val_records(seed, run, thr=0.1, nmax=0):
    import jwiring as JWI
    QE, QR, dump, ends, frames, spec = JWI._load(argparse.Namespace(seed=seed, run=run))
    gt = pickle.load(open(JWI.GT_PATH, "rb"))
    out, t0 = [], time.time()
    for b, (f, d) in enumerate(zip(frames, dump)):
        if nmax and b >= nmax:
            break
        pred, graph, bk = JWI._graph(QE, QR, d, spec)
        out.append(record_of(pred, graph, bk, ends[f.row], gt[f.row], thr, f.row, f.scene))
        f._lane = f._ref = None
        if (b + 1) % 200 == 0:
            print(f"  {b + 1}/{len(frames)}  {time.time() - t0:.0f} s", flush=True)
    return out


def live_records(seed, split="train", thr=0.1, nmax=0, workers=3, bs=4, device=None, pair_run=None):
    """the records of the frames of a split from a LIVE pass of B (`runs/e273_p_s<seed>`, exact-map pass, the graph read out with close:1.5@0.5 as the card of record does) and the pair run (`runs/<pair_run>`, default
    e312_p_s<seed>) on the loader of `split` ('train' for the training of the head, 'val_clean' to check that this pass reproduces the records of the stored dumps): GPU, about 10 minutes for the 8414 TRAIN frames"""
    assert split in SPLITS, f"split {split!r}: only {SPLITS} (TEST is never read by a pass of this module)"
    import torch
    import jends_eval as JE
    import lane_gt_data as LD
    import qgraph_eval as QE
    import qgraph_readout as QR
    import qpredict as QP
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    mb = QP.load_model(os.path.join(ROOT, "runs", f"e273_p_s{seed}"), dev)
    mp, _ = JE.load_model(pair_run or f"e312_p_s{seed}", dev)
    LD.QData.DEFAULT_ENDS = True
    ds, dl = LD.loader(split, bs=bs, workers=workers, shuffle=False, map_mode="exact", pixels=False, strict=True, ends=True)
    assert ds.missing == 0, f"{ds.missing} frames of {split} have no stored lane GT"
    spec = QR.parse("close:1.5@0.5")
    out, t0 = [], time.time()
    for b in dl:
        preds = QP.predict(mb, b)
        with torch.no_grad():
            _, _, o = mp(b["x"].to(dev).float(), gt=None)
        for i, pr in enumerate(preds):
            e = dict(kp_pos=o["kp_pos"][i].cpu().numpy(), kp_dir=o["kp_dir"][i].cpu().numpy(), kp_p=o["kp_p"][i].cpu().numpy(), valid=o["valid"][i].cpu().numpy(), logit=o["logit"][i].float().cpu().numpy(),
                     allowed=o["allowed"][i].cpu().numpy(), scale=o["scale"][i].float().cpu().numpy() if "scale" in o else None)
            g = dict(ent=b["ent"][i].numpy(), ent_m=b["ent_m"][i].numpy(), ext=b["ext"][i].numpy(), ext_m=b["ext_m"][i].numpy())
            pred = QR.apply(dict(nodes=dict(pos=pr["nodes"]["pos"], dirv=pr["nodes"]["dirv"], e=pr["nodes"]["e"], flags=pr["nodes"]["flags"], section=pr["nodes"]["section"], s=pr["nodes"].get("s")),
                                 links=dict(src=pr["links"]["src"], dst=pr["links"]["dst"], p=pr["links"]["p"], long=pr["links"]["long"], a=pr["links"]["a"], b=pr["links"]["b"])), spec)
            graph, bk = QE.readout(pred, 0.5, 0.5)
            out.append(record_of(pred, graph, bk, e, g, thr, int(b["row"][i]), -1))
        if nmax and len(out) >= nmax:
            break
        if len(out) % 400 < bs:
            print(f"  {len(out)}/{len(ds)}  {time.time() - t0:.0f} s", flush=True)
    out = out[: nmax or None]
    assert len({r["row"] for r in out}) == len(out), "a row was visited twice"
    assert nmax or len(out) == len(ds), f"{len(out)} records for {len(ds)} frames of {split}"
    return out


def cache_path(kind, seed, nmax=0, thr=0.1):
    """temp/jwire/<kind>_records_v<version>_s<seed>[_max<N>][_thr<t>].pkl: the name carries everything that changes the content"""
    return os.path.join(ROOT, "temp", "jwire", f"{kind}_records_v{RECORD_VERSION}_s{seed}{'_max' + str(nmax) if nmax else ''}{'_thr' + str(thr) if thr != 0.1 else ''}.pkl")


def pilot_stats(records):
    """the numbers of the pilot (card section 1, in-sample guard): the keypoint error of the matched ends (distance of the pair end to its GT end: mean / median / 90 %) and the rule's agreement with the oracle label"""
    err = []
    for r in records:
        for n in range(len(r["pairs"])):
            for side, sl in ((0, slice(0, 2)), (1, slice(4, 6))):
                g = r["gts"][n, side]
                if not np.isnan(g).any():
                    err.append(float(np.linalg.norm(r["pairs"][n, sl] - g[:2])))
    err = np.array(err)
    tot = {}
    for r in records:
        for k, v in agreement(instances_of(r)).items():
            tot[k] = tot.get(k, 0) + v
    kp = np.concatenate([r["kp_p"].reshape(-1) for r in records if len(r["kp_p"])])
    at = np.concatenate([r["attrs"] for r in records if len(r["attrs"])])
    return dict(matched_ends=len(err), err_mean=float(err.mean()), err_median=float(np.median(err)), err_p90=float(np.percentile(err, 90)), overall=tot["same"] / max(tot["oracle"], 1), multi=tot["multi_same"] / max(tot["multi"], 1),
                kp_p_mean=float(kp.mean()), kp_p_p10=float(np.percentile(kp, 10)), node_e_mean=float(at[:, 2:4].mean()), link_p_mean=float(at[:, 4:6].mean()), multi_share=tot["multi"] / max(tot["oracle"], 1), counts=tot)


def make_jitter(scale, seed=0, heading_deg=4.0):
    """a heavy-tailed keypoint jitter (Student t with 3 degrees of freedom per coordinate, times `scale` m, clipped at 8 m; a Gaussian heading rotation of `heading_deg` degrees): emulates the keypoint error of frames the networks have not been trained on"""
    rng = np.random.default_rng(seed)

    def jit(pos, tan):
        d = np.clip(scale * rng.standard_t(3, 2), -8.0, 8.0)
        a = np.radians(rng.normal(0.0, heading_deg))
        c, s = np.cos(a), np.sin(a)
        return pos + d, np.array([c * tan[0] - s * tan[1], s * tan[0] + c * tan[1]])
    return jit


def calibrate_jitter(records, target_median, seed=0, iters=12):
    """the jitter scale (m) at which the MEDIAN keypoint error of the matched ends of `records` (pair end to GT end) reaches `target_median` (the VAL value): the in-sample guard of the card, matched on the error quantile
    (TRAIN keypoints are better than VAL ones, the networks were trained on them); the agreement of the rule with the oracle label cannot be matched at the same time (see the card), it is reported -> (scale, median error)"""
    def med(scale):
        err = []
        j = make_jitter(scale, seed)                                                                  # ONE stream over all records, as the training draws use
        for r in records:
            for n in range(len(r["pairs"])):
                for side, sl in ((0, slice(0, 2)), (1, slice(4, 6))):
                    g = r["gts"][n, side]
                    if not np.isnan(g).any():
                        p, _ = j(r["pairs"][n, sl], np.array([0.0, 1.0]))
                        err.append(np.linalg.norm(p - g[:2]))
        return float(np.median(err))
    lo, hi = 0.0, 3.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if med(mid) < target_median:
            lo = mid
        else:
            hi = mid
    s = 0.5 * (lo + hi)
    return s, med(s)


def _selftest():
    import jwire_head as H
    up = np.array([0.0, 1.0])
    # a record built by hand: two pieces, one pair, the entry keypoint matches a GT end at the second piece
    ends = np.array([np.r_[[0.0, -1.0], up, [0.0, -20.0], up], np.r_[[0.0, -3.0], up, [0.0, -30.0], up]])
    rec = dict(row=1, scene=2, ends=ends, attrs=np.zeros((2, 6), np.float32), pairs=np.array([np.r_[[0.2, 0.0], up, [5.0, 20.0], up]]), kp_p=np.array([[0.7, 0.6]]),
               gts=np.array([[[0.0, -3.05, 0.0, 1.0], [np.nan] * 4]]))
    inst = instances_of(rec)
    ent = [i for i in inst if i["side"] == 0][0]
    assert ent["label"] == 1 and ent["kind"] == "oracle" and len(ent["piece"]) == 2
    ag = agreement(inst)
    assert ag["oracle"] == 1 and ag["same"] == 0 and ag["multi"] == 1 and ag["multi_same"] == 0, ag
    inst0 = instances_of(dict(rec, gts=np.full((1, 2, 4), np.nan)))
    ag0 = agreement(inst0)
    assert ag0["oracle"] == 0 and ag0["rule_label"] == len(inst0), ag0
    # match(): the same one-to-one rule as the loader's
    G = np.array([[0.0, 0.0, 0.0, 1.0], [10.0, 0.0, 0.0, 1.0]])
    mm = match(np.array([[0.5, 0.0], [0.6, 0.0], [10.2, 0.0]]), np.tile(up, (3, 1)), np.ones(3, bool), G, np.array([True, True]))
    assert list(mm) == [0, -1, 1], mm                                                                     # the second keypoint is a duplicate of the first end: no positive
    assert H.FEAT == 15
    # record_of on a toy graph: the entry GT end goes to the entry slot and the exit GT end to the exit slot (swapping them must fail), and `instances_of(record_of(...))` equals `build_instances` on the same objects
    from types import SimpleNamespace as NS
    lines = [np.c_[np.zeros(6), np.linspace(-20, -1, 6)], np.c_[np.zeros(6), np.linspace(2, 15, 6)]]
    graph = NS(lines=lines)
    bk = dict(kind=["section", "section"], piece_links=[[0], [1]], piece_nodes=[[0, 1], [2, 3]])
    pred = dict(links=dict(long=np.zeros(2, bool), p=np.array([0.9, 0.8])), nodes=dict(e=np.array([0.9, 0.8, 0.7, 0.6])))
    e = dict(kp_pos=np.array([[0.2, 0.0], [0.1, 1.0]]), kp_dir=np.tile(up, (2, 1)), kp_p=np.array([0.7, 0.6]), valid=np.ones((2, 1), bool), logit=np.array([[5.0]]),
             allowed=np.array([[1.0]]), scale=None)
    g = dict(ent=np.array([[0.0, -1.1, 0.0, 1.0]]), ent_m=np.array([True]), ext=np.array([[0.0, 2.1, 0.0, 1.0]]), ext_m=np.array([True]))
    rec2 = record_of(pred, graph, bk, e, g, 0.1)
    assert rec2["pairs"].shape == (1, 8) and np.allclose(rec2["gts"][0, 0], g["ent"][0]) and np.allclose(rec2["gts"][0, 1], g["ext"][0]), rec2["gts"]
    assert not np.allclose(rec2["gts"][0, 0], g["ext"][0])                                                     # an entry / exit swap in record_of would put the exit GT in the entry slot
    pa = [(np.array([0.2, 0.0]), up, np.array([0.1, 1.0]), up)]
    keep_, en_ = JW.kept_ends(lines, bk["kind"], bk["piece_links"], np.zeros(2, bool), pa, QEV.cover_pos())
    at_ = H.piece_attrs(lines, bk["kind"], bk["piece_nodes"], bk["piece_links"], pred["nodes"]["e"], pred["links"]["p"], keep_)
    direct = H.build_instances(en_, at_, pa, [(0.7, 0.6)], [(g["ent"][0], g["ext"][0])])
    viarec = instances_of(rec2)
    assert len(direct) == len(viarec) and all(np.allclose(x["feat"], y["feat"], atol=1e-5) and x["label"] == y["label"] for x, y in zip(direct, viarec)), "records and direct instances differ"
    print("jwire_data selftest ok")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("val")
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--run", default=None)
    s.add_argument("--thr", type=float, default=0.1)
    s.add_argument("--max", type=int, default=0)
    t = sub.add_parser("live")
    t.add_argument("--seed", type=int, required=True)
    t.add_argument("--split", default="train")
    t.add_argument("--max", type=int, default=0)
    t.add_argument("--workers", type=int, default=3)
    t.add_argument("--device", default=None)
    t.add_argument("--out", default=None)
    sub.add_parser("selftest")
    a = ap.parse_args()
    if a.cmd == "live":
        recs = live_records(a.seed, a.split, 0.1, a.max, a.workers, device=a.device)
        path = a.out or cache_path(a.split, a.seed, a.max)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pickle.dump(recs, open(path, "wb"))
        print(f"{len(recs)} records -> {os.path.relpath(path, ROOT)}")
        print(pilot_stats(recs))
        return
    if a.cmd == "selftest":
        _selftest()
        return
    run = a.run or f"e312_p_s{a.seed}"
    recs = val_records(a.seed, run, a.thr, a.max)
    path = cache_path("val", a.seed, a.max, a.thr)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pickle.dump(recs, open(path, "wb"))
    tot = {}
    for r in recs:
        for k, v in agreement(instances_of(r)).items():
            tot[k] = tot.get(k, 0) + v
    print(f"{len(recs)} records -> {os.path.relpath(path, ROOT)}")
    print(tot)
    if tot["oracle"]:
        print(f"rule agreement with the oracle label: overall {tot['same'] / tot['oracle']:.3f} ({tot['oracle']} ends), real choices {tot['multi_same'] / max(tot['multi'], 1):.3f} ({tot['multi']} ends); label outside the keypoint's places: {tot['outside']}")


if __name__ == "__main__":
    main()
