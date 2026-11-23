"""E325 step 2: the data of the wiring head `Wire` (`training/grid/jwire_head.py`): per frame a compact RECORD of what a connector end sees (the kept pieces of the delivered graph and their attributes, the pairs, the keypoint probabilities,
the GT ends the keypoints match), and the instances / agreement statistics made from it.

    wsl bash tools/env/_py14.sh jwire_data.py val --seed 0 [--pair-run e312_p_s0] [--ends kp|att] [--thr 0.1] [--max N]  # VAL records from the stored dumps (CPU: runs/<pair run>/ends_val.pkl, B's full dump runs/kept/e276/predfull_e273_p_s<seed>.pkl, the GT ends
                                                                                                         # runs/kept/jwiring/gt_ends_val.pkl) -> runs/kept/jwire/val_records_v1_s<seed>[_pair_<run>][_ends_att].pkl and the rule's agreement with the oracle label (--run = --pair-run)
    wsl bash tools/env/_py14.sh jwire_data.py live --seed 0 [--pair-run e312_p_s0] [--ends kp|att] [--split train|val_clean] [--max N]   # the same records from a LIVE pass of B and the pair run (GPU) + the pilot numbers
    wsl bash tools/env/_py14.sh jwire_data.py selftest

A record = dict(row, scene, ends (n, 8) = last point, heading there, first point, heading there of every kept piece, attrs (n, 6), pairs (m, 8) = entry position / heading, exit position / heading, kp_p (m, 2), gts (m, 2, 4) = the GT end of the
entry / exit keypoint (x, y, tx, ty) or NaN). `instances_of(rec, jitter)` -> the instances of `jwire_head.build_instances`. The pieces are the pieces KEPT by the cover drop of the unjittered pairs (the deployed read-out): a jitter changes
the keypoints, not the delivered graph. The agreement is counted on the ends with an ORACLE label (a matched GT end that finds a piece in reach): overall and on the ends with two or more places (`real choices`); the rule = place 0.
VAL only, TEST never; the TRAIN records (a live pass of B and the pair run, GPU) are a separate step of the card.
The PAIR RUN (the run whose keypoints the records hold) is E312's `e312_p_s<seed>` by default (E325); another one (E350: `e324_s_s<seed>`) gets its own cache names (`cache_path`), every cache file carries
`meta = dict(pair_run, seed, record_version)` that a load checks (`load_records`: a file without meta is E312's, valid for E312's pair run only), a cache file is never overwritten, and the seed of the pair run is
checked against the seed asked for (`check_seed`, as `jends_eval.score` does).
The END KIND (E358, `--ends`): `kp` (default: E312's, E325's and E350's records) builds the pairs from the keypoints of the pair run, `att` from its ATTACHED ends (`att_pos` / `att_dir`, the ends of the primary read-out of E324 s; a pair run
without them, E312's, is refused); only the pair ends change, the matching to the GT end stays on the keypoints. `att` adds `_ends_att` to the cache names and `ends='att'` to the meta; the names and the meta of `kp` are what they were (no
`ends` field = `kp`), so a kp cache is refused for `att` and an att cache for `kp`.
"""
import argparse
import contextlib
import os
import pickle
import re
import sys
import tempfile
import time

import numpy as np

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for sub in ("evaluation", "training", "training/grid", "training/data", "training/data/prep", "training/train", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub)))
import jwire as JW  # noqa: E402
import jwire_head as JH  # noqa: E402
import qends_eval as QEV  # noqa: E402

MATCH_POS, MATCH_ANG = 4.0, 30.0
NEAR_M = 0.5                                                                                          # `pilot_stats`: an end is NEAR when its nearest candidate piece end is within this many metres
ENDS = ("kp", "att")                                                                                  # the connector ends the pairs of a record are built from: the keypoints (E312, E325, E350) or the ATTACHED ends of the pair run (E324 s, E358)
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


def record_of(pred, graph, bk, e, g, thr=0.1, row=-1, scene=-1, ends="kp"):
    """the record of one frame: pred / graph / bk = the delivered graph of B read out as the card of record does (`qgraph_readout.apply` close:1.5@0.5, `qgraph_eval.readout`), e = the pair run's frame dict (kp_pos, kp_dir, kp_p,
    valid, logit, allowed; ends = att also att_pos, att_dir), g = the GT ends of the frame (ent, ent_m, ext, ext_m). ends = which ends the pairs hold (`check_ends`); the match to the GT end is always the keypoint's"""
    attach = check_ends(ends) == "att"
    assert not attach or ("att_pos" in e and "att_dir" in e), "ends att: the frame has no attached ends (att_pos / att_dir): the pair run is not an attach model or its dump is older than E320 (no fall back to the keypoints)"
    k = e["kp_pos"].shape[0] // 2
    p = 1.0 / (1.0 + np.exp(-e["logit"])) * e["allowed"]
    ia, ib = np.nonzero(p >= thr)
    pairs = QEV.make_pairs(e, ia, ib, k, attach=attach)
    keep, kept = JW.kept_ends(graph.lines, bk["kind"], bk["piece_links"], np.asarray(pred["links"]["long"], bool), pairs, QEV.cover_pos())
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
    return dict(row=row, scene=scene, ends=np.array([np.r_[en[0], en[1], en[2], en[3]] for en in kept], float).reshape(-1, 8), attrs=attrs,
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


def default_pair_run(seed):
    """the pair run of E325's records and heads (E312)"""
    return f"e312_p_s{seed}"


def resolve_pair_run(seed, pair_run=None):
    """the pair run asked for: None = E312's (the default); anything else must be a run folder name (an empty string is refused, not read as the default)"""
    if pair_run is None:
        return default_pair_run(seed)
    assert re.fullmatch(r"[A-Za-z0-9_-]+", pair_run), f"pair run {pair_run!r}: a run folder name"
    return pair_run


def check_ends(ends, seed=None, pair_run=None):
    """-> `ends` when it is 'kp' or 'att' (the one rule of the end kind); with a seed also: 'att' needs a pair run that has attached ends, and E312's own pair run (the default) has none (its dump has no `att_pos`)"""
    assert ends in ENDS, f"ends {ends!r}: one of {ENDS}"
    assert ends == "kp" or seed is None or resolve_pair_run(seed, pair_run) != default_pair_run(seed), f"ends att needs the pair run of a run with attached ends (--pair-run, e.g. e324_s_s{seed}): E312's `{default_pair_run(seed)}` has none"
    return ends


def check_seed(args, seed, run):
    """the pair run's own seed (the args of its checkpoint) must be the seed asked for, as `jends_eval.score` checks it: B's delivered graph is seed-specific and would be joined to another seed's keypoints"""
    m = re.search(r"_s(\d+)[/\\]", str(args.get("maps", "")).replace("\\", "/") + "/")
    assert args.get("seed", seed) == seed and (m is None or int(m.group(1)) == seed), f"--seed {seed} does not match the pair run {run} (seed {args.get('seed')}, maps {args.get('maps')}): its keypoints belong to another seed"


def run_args(run):
    """the args of the checkpoint of a run (`runs/<run>/model.pt`)"""
    import torch
    return torch.load(os.path.join(ROOT, "runs", run, "model.pt"), map_location="cpu")["args"]


def val_records(seed, run, thr=0.1, nmax=0, ends="kp"):
    check_ends(ends, seed, run)
    check_seed(run_args(run), seed, run)
    import jwiring as JWI
    QE, QR, dump, frames_ends, frames, spec = JWI._load(argparse.Namespace(seed=seed, run=run))
    gt = pickle.load(open(JWI.GT_PATH, "rb"))
    out, t0 = [], time.time()
    for b, (f, d) in enumerate(zip(frames, dump)):
        if nmax and b >= nmax:
            break
        pred, graph, bk = JWI._graph(QE, QR, d, spec)
        out.append(record_of(pred, graph, bk, frames_ends[f.row], gt[f.row], thr, f.row, f.scene, ends))
        f._lane = f._ref = None
        if (b + 1) % 200 == 0:
            print(f"  {b + 1}/{len(frames)}  {time.time() - t0:.0f} s", flush=True)
    return out


def load_models(seed, pair_run, dev):
    """-> (B, the pair run) for a live pass: B = `runs/e273_p_s<seed>`, the pair run = `runs/<pair_run>` (the attach model of E324 s included); the seed of the pair run is checked"""
    import jends_eval as JE
    import qpredict as QP
    mb = QP.load_model(os.path.join(ROOT, "runs", f"e273_p_s{seed}"), dev)
    mp, args = JE.load_model(pair_run, dev)
    check_seed(args, seed, pair_run)
    return mb, mp


def frame_dict(o, i, ends="kp"):
    """the frame dict `record_of` reads (the keys of a frame of the stored dump, `jends_eval.py dump`) of batch item i of the pair run's output `o`; ends = att also takes the attached ends of the model (`att_pos`, `att_dir`):
    an output without them is an error, never a fall back to the keypoints"""
    e = dict(kp_pos=o["kp_pos"][i].cpu().numpy(), kp_dir=o["kp_dir"][i].cpu().numpy(), kp_p=o["kp_p"][i].cpu().numpy(), valid=o["valid"][i].cpu().numpy(), logit=o["logit"][i].float().cpu().numpy(),
             allowed=o["allowed"][i].cpu().numpy(), scale=o["scale"][i].float().cpu().numpy() if "scale" in o else None)
    if check_ends(ends) == "att":
        assert "att_pos" in o and "att_dir" in o, "ends att: the output of the pair run has no attached ends (att_pos / att_dir): it is not an attach model (no fall back to the keypoints)"
        e.update(att_pos=o["att_pos"][i].float().cpu().numpy(), att_dir=o["att_dir"][i].float().cpu().numpy())
    return e


# E358 attempt 4 (card section 5, pre-stated before the run): the settings the two STORED dumps were made with, constants and not parameters; the check `jwire_check.py live --faithful` replays both
DUMP_B = dict(bs=4, map_mode="exact")                      # B's full dump `runs/kept/e276/predfull_e273_p_s<seed>.pkl`: `qgraph_run_eval.py:89` (`LD.loader("val_clean", bs=4, ..., map_mode=<exact>, pixels=False, strict=True)`)
DUMP_PAIR = dict(bs=8, map_mode="error", workers=2)        # the pair dump `runs/<pair run>/ends_val.pkl`: `jends_eval.py:42` (`dump`: `DEFAULT_JUNCTION = True; LD.loader("val_clean", bs=8, workers=2, shuffle=False, map_mode="error", pixels=False)`)
FAITHFUL_MAX = 64                                          # the faithful pass keeps B's predictions of all its frames in memory: a check, not a TRAIN pass


def live_records(seed, split="train", thr=0.1, nmax=0, workers=3, bs=4, device=None, pair_run=None, ends="kp", faithful=False):
    """the records of the frames of a split from a LIVE pass of B (`runs/e273_p_s<seed>`, exact-map pass, the graph read out with close:1.5@0.5 as the card of record does) and the pair run (`runs/<pair_run>`, default
    e312_p_s<seed>) on the loader of `split` ('train' for the training of the head, 'val_clean' to check that this pass reproduces the records of the stored dumps): GPU, about 10 minutes for the 8414 TRAIN frames"""
    assert split in SPLITS, f"split {split!r}: only {SPLITS} (TEST is never read by a pass of this module)"
    check_ends(ends, seed, pair_run)
    assert not faithful or (split == "val_clean" and 0 < nmax <= FAITHFUL_MAX), f"faithful (the two-loader replay of the stored dumps' settings, E358) is a check of at most {FAITHFUL_MAX} val_clean frames, not a TRAIN pass"
    import torch
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    mb, mp = load_models(seed, resolve_pair_run(seed, pair_run), dev)
    if faithful:
        return _live_records_faithful(thr, nmax, workers, dev, mb, mp, ends)
    import lane_gt_data as LD
    import qgraph_eval as QE
    import qgraph_readout as QR
    import qpredict as QP
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
            e = frame_dict(o, i, ends)
            g = dict(ent=b["ent"][i].numpy(), ent_m=b["ent_m"][i].numpy(), ext=b["ext"][i].numpy(), ext_m=b["ext_m"][i].numpy())
            pred = QR.apply(dict(nodes=dict(pos=pr["nodes"]["pos"], dirv=pr["nodes"]["dirv"], e=pr["nodes"]["e"], flags=pr["nodes"]["flags"], section=pr["nodes"]["section"], s=pr["nodes"].get("s")),
                                 links=dict(src=pr["links"]["src"], dst=pr["links"]["dst"], p=pr["links"]["p"], long=pr["links"]["long"], a=pr["links"]["a"], b=pr["links"]["b"])), spec)
            graph, bk = QE.readout(pred, 0.5, 0.5)
            out.append(record_of(pred, graph, bk, e, g, thr, int(b["row"][i]), -1, ends))
        if nmax and len(out) >= nmax:
            break
        if len(out) % 400 < bs:
            print(f"  {len(out)}/{len(ds)}  {time.time() - t0:.0f} s", flush=True)
    out = out[: nmax or None]
    assert len({r["row"] for r in out}) == len(out), "a row was visited twice"
    assert nmax or len(out) == len(ds), f"{len(out)} records for {len(ds)} frames of {split}"
    return out


def _live_records_faithful(thr, nmax, workers, dev, mb, mp, ends):
    """E358 attempt 4: the live records of the first `nmax` val_clean frames with BOTH models run the way their stored dumps were made: B on a loader of its dump's settings (`DUMP_B`), the pair model on a loader of the
    pair dump's settings (`DUMP_PAIR`), the two joined by the frame's `row` (the GT ends of the record come from B's loader, as in `live_records`). The record construction after the two forwards is the same as there."""
    import torch
    import lane_gt_data as LD
    import qgraph_eval as QE
    import qgraph_readout as QR
    import qpredict as QP
    spec = QR.parse("close:1.5@0.5")
    old_ends = getattr(LD.QData, "DEFAULT_ENDS", False)
    LD.QData.DEFAULT_ENDS = True
    ds, dl = LD.loader("val_clean", bs=DUMP_B["bs"], workers=workers, shuffle=False, map_mode=DUMP_B["map_mode"], pixels=False, strict=True, ends=True)
    assert ds.missing == 0, f"{ds.missing} frames of val_clean have no stored lane GT"
    b_side = {}
    for b in dl:                                                                                           # pass 1: B (QP.predict) on its dump's loader
        for i, pr in enumerate(QP.predict(mb, b)):
            b_side[int(b["row"][i])] = (pr, dict(ent=b["ent"][i].numpy(), ent_m=b["ent_m"][i].numpy(), ext=b["ext"][i].numpy(), ext_m=b["ext_m"][i].numpy()))
        if len(b_side) >= nmax:
            break
    rows = list(b_side)[:nmax]
    assert len(rows) == nmax, f"B's loader gave {len(rows)} frames, {nmax} asked"
    LD.QData.DEFAULT_ENDS = old_ends
    old_junction = getattr(LD.QData, "DEFAULT_JUNCTION", False)
    LD.QData.DEFAULT_JUNCTION = True                                                                       # as `jends_eval.dump` sets it before its loader; put back below
    try:
        ds2, dl2 = LD.loader("val_clean", bs=DUMP_PAIR["bs"], workers=DUMP_PAIR["workers"], shuffle=False, map_mode=DUMP_PAIR["map_mode"], pixels=False)
        assert ds2.missing == 0, f"{ds2.missing} frames of val_clean have no stored lane GT (the dump asserts it too)"
        by_row = {}
        for b in dl2:                                                                                      # pass 2: the pair model on its dump's loader; a frame is kept when B's pass saw it
            with torch.no_grad():
                _, _, o = mp(b["x"].to(dev).float(), gt=None)
            for i in range(len(b["row"])):
                if int(b["row"][i]) in b_side and int(b["row"][i]) not in by_row:
                    by_row[int(b["row"][i])] = frame_dict(o, i, ends)
            if all(r in by_row for r in rows):
                break
    finally:
        LD.QData.DEFAULT_JUNCTION = old_junction
    assert all(r in by_row for r in rows), f"the pair loader did not reach the rows {[r for r in rows if r not in by_row]} of B's pass"
    out = []
    for r in rows:
        pr, g = b_side[r]
        pred = QR.apply(dict(nodes=dict(pos=pr["nodes"]["pos"], dirv=pr["nodes"]["dirv"], e=pr["nodes"]["e"], flags=pr["nodes"]["flags"], section=pr["nodes"]["section"], s=pr["nodes"].get("s")),
                             links=dict(src=pr["links"]["src"], dst=pr["links"]["dst"], p=pr["links"]["p"], long=pr["links"]["long"], a=pr["links"]["a"], b=pr["links"]["b"])), spec)
        graph, bk = QE.readout(pred, 0.5, 0.5)
        out.append(record_of(pred, graph, bk, by_row[r], g, thr, r, -1, ends))
    assert len({x["row"] for x in out}) == len(out) == nmax, "a row was visited twice or a frame is missing"
    return out


def cache_path(kind, seed, nmax=0, thr=0.1, pair_run=None, ends="kp"):
    """runs/kept/jwire/<kind>_records_v<version>_s<seed>[_max<N>][_thr<t>][_pair_<run>][_ends_att].pkl: the name carries everything that changes the content; E312's own pair run (the default) and the keypoint ends add nothing, so
    E325's and E350's names stay"""
    pair_run = resolve_pair_run(seed, pair_run)
    check_ends(ends, seed, pair_run)
    other = pair_run != default_pair_run(seed)
    return os.path.join(ROOT, "runs", "kept", "jwire", f"{kind}_records_v{RECORD_VERSION}_s{seed}{'_max' + str(nmax) if nmax else ''}{'_thr' + str(thr) if thr != 0.1 else ''}{'_pair_' + pair_run if other else ''}{'_ends_' + ends if ends != 'kp' else ''}.pkl")


def meta_of(seed, pair_run=None, ends="kp"):
    """the meta of a cache file: `ends` is in it for `att` only, so the meta (and the caches) of E325 / E350 stay what they were: no `ends` field = kp"""
    pair_run = resolve_pair_run(seed, pair_run)
    meta = dict(pair_run=pair_run, seed=seed, record_version=RECORD_VERSION)
    if check_ends(ends, seed, pair_run) != "kp":
        meta["ends"] = ends
    return meta


def save_records(path, records, seed, pair_run=None, ends="kp"):
    """writes `dict(meta, records)`; a cache file that exists is never overwritten (move it away to rebuild it): the file is written under a name of its own and linked to `path`, which fails when `path` exists, so
    two builders of the same cache cannot overwrite each other"""
    assert not os.path.exists(path), f"{path} exists: a cache is never overwritten"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    try:
        with open(tmp, "wb") as f:
            pickle.dump(dict(meta=meta_of(seed, pair_run, ends), records=records), f)
        os.link(tmp, path)
    except FileExistsError:
        raise AssertionError(f"{path} exists: a cache is never overwritten (another builder was faster)") from None
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def load_records(path, seed, pair_run=None, ends="kp"):
    """the records of a cache file, its meta must be the one asked for (the end kind included); a file without meta (E325's caches of 2026-11-14, a plain list) holds E312's records and is valid for E312's pair run only (and so for `kp`)"""
    obj = pickle.load(open(path, "rb"))
    want = meta_of(seed, pair_run, ends)
    if isinstance(obj, list):
        assert want["pair_run"] == default_pair_run(seed), f"{path} has no meta (E312's records, `{default_pair_run(seed)}`): refused for the pair run {want['pair_run']}"
        return obj
    assert obj["meta"] == want, f"{path}: meta {obj['meta']} is not the one asked for {want} (no `ends` field = kp)"
    return obj["records"]


def cached_records(path, seed, pair_run, make, ends="kp"):
    """the records of the cache file `path` (`load_records`) or, when there is none, `make()` written to it"""
    if os.path.exists(path):
        return load_records(path, seed, pair_run, ends)
    recs = make()
    save_records(path, recs, seed, pair_run, ends)
    return recs


def is_near(d_feat):
    """True where the nearest candidate piece end of an end is NEAR: `d_feat` = feature 0 of its place 0 (distance / reach; a float, an array or a tensor) within `NEAR_M` metres: the one rule of `pilot_stats` (from the records) and of the
    jittered share `jwire_train` logs (from the features the head trains on)"""
    return d_feat * JH.REACH <= NEAR_M


def pilot_stats(records):
    """the numbers of the pilot (card section 1, in-sample guard): the error of the matched pair ends (keypoint or attached end: distance of the pair end to its GT end: mean / median / 90 %), the rule's agreement with the oracle label,
    and `near05` = the share of the ends with an oracle label whose nearest candidate piece end lies within `NEAR_M` (0.5 m) (E358: the attached ends are bimodal, a TRAIN share far from the VAL share is a shift of the problem)"""
    err = []
    for r in records:
        for n in range(len(r["pairs"])):
            for side, sl in ((0, slice(0, 2)), (1, slice(4, 6))):
                g = r["gts"][n, side]
                if not np.isnan(g).any():
                    err.append(float(np.linalg.norm(r["pairs"][n, sl] - g[:2])))
    err = np.array(err)
    tot, near = {}, []
    for r in records:
        inst = instances_of(r)
        for k, v in agreement(inst).items():
            tot[k] = tot.get(k, 0) + v
        near += [is_near(float(i["feat"][0, 0])) for i in inst if i["kind"] == "oracle"]                   # place 0 = the nearest candidate; feature 0 = its distance / reach
    kp = np.concatenate([r["kp_p"].reshape(-1) for r in records if len(r["kp_p"])] + [np.zeros(0)])
    at = np.concatenate([r["attrs"] for r in records if len(r["attrs"])] + [np.zeros((0, 6))])
    q = lambda x, f: float(f(x)) if np.size(x) else float("nan")                                          # noqa: E731  (a subset with no matched end or no pair: NaN, not a crash)
    return dict(matched_ends=len(err), err_mean=q(err, np.mean), err_median=q(err, np.median), err_p90=q(err, lambda x: np.percentile(x, 90)), overall=tot["same"] / max(tot["oracle"], 1), multi=tot["multi_same"] / max(tot["multi"], 1),
                kp_p_mean=q(kp, np.mean), kp_p_p10=q(kp, lambda x: np.percentile(x, 10)), node_e_mean=q(at[:, 2:4], np.mean), link_p_mean=q(at[:, 4:6], np.mean), multi_share=tot["multi"] / max(tot["oracle"], 1), near05=q(near, np.mean), counts=tot)


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


def toy_record(row=1, scene=2):
    """a record built by hand (the tests of this module and of `jwire_train`): two pieces, one pair, the entry keypoint matches a GT end at the second piece"""
    up = np.array([0.0, 1.0])
    ends = np.array([np.r_[[0.0, -1.0], up, [0.0, -20.0], up], np.r_[[0.0, -3.0], up, [0.0, -30.0], up]])
    return dict(row=row, scene=scene, ends=ends, attrs=np.zeros((2, 6), np.float32), pairs=np.array([np.r_[[0.2, 0.0], up, [5.0, 20.0], up]]), kp_p=np.array([[0.7, 0.6]]), gts=np.array([[[0.0, -3.05, 0.0, 1.0], [np.nan] * 4]]))


def toy_frame():
    """one frame built by hand (the tests of this module): two pieces on the line x = 0 (the first ends at y = -1, the second starts at y = 2), one pair of keypoints 1 m beside the piece ends, the GT ends 0.1 m from them
    -> (pred, graph, bk, e, g) as `record_of` takes them; `e` holds the keypoints only (a test adds `att_pos` / `att_dir`)"""
    from types import SimpleNamespace as NS
    up = np.array([0.0, 1.0])
    graph = NS(lines=[np.c_[np.zeros(6), np.linspace(-20, -1, 6)], np.c_[np.zeros(6), np.linspace(2, 15, 6)]])
    bk = dict(kind=["section", "section"], piece_links=[[0], [1]], piece_nodes=[[0, 1], [2, 3]])
    pred = dict(links=dict(long=np.zeros(2, bool), p=np.array([0.9, 0.8])), nodes=dict(e=np.array([0.9, 0.8, 0.7, 0.6])))
    e = dict(kp_pos=np.array([[0.2, 0.0], [0.1, 1.0]]), kp_dir=np.tile(up, (2, 1)), kp_p=np.array([0.7, 0.6]), valid=np.ones((2, 1), bool), logit=np.array([[5.0]]), allowed=np.array([[1.0]]), scale=None)
    g = dict(ent=np.array([[0.0, -1.1, 0.0, 1.0]]), ent_m=np.array([True]), ext=np.array([[0.0, 2.1, 0.0, 1.0]]), ext_m=np.array([True]))
    return pred, graph, bk, e, g


def refusal(f, *a, **k):
    """the message of the AssertionError that `f(*a, **k)` raises ('' when the message is empty), None when it passes"""
    try:
        f(*a, **k)
    except AssertionError as e:
        return str(e) or "refused"
    return None


@contextlib.contextmanager
def tmp_root():
    """a test's ROOT: a temp folder instead of the problem folder, so that no test reads or writes the real caches (`cache_path`, `run_args`)"""
    global ROOT
    old = ROOT
    with tempfile.TemporaryDirectory() as d:
        ROOT = d
        try:
            yield d
        finally:
            ROOT = old


class _Reached(Exception):
    """a stub that records its arguments and stops the code under test at this point"""


def _test_pair_run():
    """the pair run (E350): cache names, meta, seed checks, the CLI and the live pass reach it; no test touches a real cache"""
    import io
    import jends_eval as JE
    import jwiring as JWI
    import qpredict as QP
    import torch
    # 1. the names: E325's stay bit for bit, another pair run (or the other seed's E312 run) gets its own
    with tmp_root() as d:
        name = lambda *a, **k: os.path.basename(cache_path(*a, **k))  # noqa: E731
        assert [name("val", 0), name("val", 1), name("train", 0), name("train", 1), name("train", 0, 200), name("val_clean", 0, 24), name("val", 0, 0, 0.083)] == [
            "val_records_v1_s0.pkl", "val_records_v1_s1.pkl", "train_records_v1_s0.pkl", "train_records_v1_s1.pkl", "train_records_v1_s0_max200.pkl", "val_clean_records_v1_s0_max24.pkl", "val_records_v1_s0_thr0.083.pkl"]
        assert name("val", 0, pair_run="e312_p_s0") == "val_records_v1_s0.pkl" and name("train", 1, pair_run="e312_p_s1") == "train_records_v1_s1.pkl"
        assert name("val", 0, pair_run="e324_s_s0") == "val_records_v1_s0_pair_e324_s_s0.pkl" and name("train", 1, 40, pair_run="e324_s_s1") == "train_records_v1_s1_max40_pair_e324_s_s1.pkl"
        assert name("val", 0, pair_run="e312_p_s1") == "val_records_v1_s0_pair_e312_p_s1.pkl"                 # E312's run of the OTHER seed is not the default of this seed
        assert os.path.dirname(cache_path("val", 0)) == os.path.join(d, "runs", "kept", "jwire")
        for bad in ("../x", "a/b", "", "a b"):
            for f in (lambda b: cache_path("val", 0, pair_run=b), lambda b: meta_of(0, b), lambda b: resolve_pair_run(0, b)):                    # an empty name is NOT the default
                try:
                    f(bad)
                    raise AssertionError(f"pair run {bad!r} must be refused")
                except AssertionError as e:
                    assert "run folder name" in str(e), e
        assert resolve_pair_run(1) == "e312_p_s1" and resolve_pair_run(0, "e324_s_s0") == "e324_s_s0"
    # 2. the content: meta is checked on load, a file without meta is E312's, a cache is never overwritten
    with tmp_root() as d:
        recs = [toy_record(1, 2)]
        p = cache_path("val", 0)
        save_records(p, recs, 0)
        assert load_records(p, 0)[0]["row"] == 1 and load_records(p, 0, "e312_p_s0")[0]["row"] == 1 and pickle.load(open(p, "rb"))["meta"] == dict(pair_run="e312_p_s0", seed=0, record_version=RECORD_VERSION)

        def refused(f, *a, **k):
            try:
                f(*a, **k)
            except AssertionError as e:
                return str(e) or True
            return False
        assert refused(load_records, p, 0, "e324_s_s0") and refused(load_records, p, 1), "a cache of another pair run / seed must be refused"
        assert refused(save_records, p, recs, 0), "an existing cache must not be overwritten"
        assert os.listdir(os.path.dirname(p)) == [os.path.basename(p)], "no temp file is left behind"
        race = cache_path("val", 0, 7)

        class Racer:                                                                                          # pickling it creates the cache: another builder finishes between the check and the link
            def __reduce__(self):
                open(race, "wb").write(b"the other builder's cache")
                return (list, ())
        assert refused(save_records, race, [Racer()], 0) and open(race, "rb").read() == b"the other builder's cache" and sorted(os.listdir(os.path.dirname(p))) == sorted([os.path.basename(p), os.path.basename(race)]), "two builders of one cache: the first one stays"
        old = cache_path("val", 1)                                                                            # E325's caches of 2026-11-14: a plain list, no meta
        pickle.dump(recs, open(old, "wb"))
        assert load_records(old, 1)[0]["row"] == 1 and refused(load_records, old, 1, "e324_s_s1"), "a file without meta is E312's records"
        stale = cache_path("val", 0, 5)
        pickle.dump(dict(meta=dict(meta_of(0), record_version=RECORD_VERSION + 1), records=recs), open(stale, "wb"))
        assert refused(load_records, stale, 0), "another record version must be refused"
        n = []
        assert cached_records(cache_path("train", 0), 0, None, lambda: n.append(1) or recs) is recs and cached_records(cache_path("train", 0), 0, None, lambda: n.append(1) or recs)[0]["row"] == 1 and len(n) == 1
    # 3. the seed of the pair run: as jends_eval.score checks it
    assert not refused(check_seed, dict(seed=0, maps="/mnt/c/x/runs/e307_n_s0/model.pt"), 0, "r") and not refused(check_seed, dict(maps="C:\\x\\e307_n_s1\\model.pt"), 1, "r") and not refused(check_seed, {}, 0, "r")
    assert refused(check_seed, dict(seed=1, maps="/x/e307_n_s1/model.pt"), 0, "r") and refused(check_seed, dict(seed=0, maps="/x/e307_n_s1/model.pt"), 0, "r") and refused(check_seed, dict(seed=1), 0, "r")
    with tmp_root() as d:                                                                                     # val_records reads the args of runs/<run>/model.pt before it loads anything
        os.makedirs(os.path.join(d, "runs", "e324_s_s1"))
        torch.save(dict(args=dict(seed=1, maps="/x/runs/e307_n_s1/model.pt")), os.path.join(d, "runs", "e324_s_s1", "model.pt"))
        assert refused(val_records, 0, "e324_s_s1"), "the pair run of seed 1 must not be read for seed 0"
        old_load = JWI._load
        JWI._load = lambda a: (_ for _ in ()).throw(_Reached(a.seed, a.run))
        try:
            val_records(1, "e324_s_s1")
            raise AssertionError("val_records did not reach jwiring._load")
        except _Reached as e:
            assert e.args == (1, "e324_s_s1")
        finally:
            JWI._load = old_load
    # 4. the live pass: the pair run reaches the model loader, its seed is checked
    seen = []
    old_je, old_qp, old_lm = JE.load_model, QP.load_model, load_models
    try:
        QP.load_model = lambda path, dev: seen.append(("B", os.path.basename(path))) or "B"
        JE.load_model = lambda run, dev: seen.append(("pair", run)) or ("model", dict(seed=1, maps="/x/e307_n_s1/model.pt"))
        assert load_models(1, "e324_s_s1", "cpu") == ("B", "model") and seen == [("B", "e273_p_s1"), ("pair", "e324_s_s1")], seen
        assert refused(load_models, 0, "e324_s_s1", "cpu"), "a pair run of another seed must be refused"
        got = []
        globals()["load_models"] = lambda seed, pair_run, dev: got.append((seed, pair_run)) or (_ for _ in ()).throw(_Reached())
        for kw, want in ((dict(pair_run="e324_s_s0"), "e324_s_s0"), (dict(), "e312_p_s0")):
            try:
                live_records(0, "val_clean", 0.1, 4, device="cpu", **kw)
            except _Reached:
                pass
            assert got[-1] == (0, want), got
        assert refused(live_records, 0, "test"), "TEST is never read"
        n_got = len(got)
        assert refused(live_records, 0, "val_clean", 0.1, 4, device="cpu", pair_run="") and len(got) == n_got, "an empty pair run is not the default"
    finally:
        JE.load_model, QP.load_model = old_je, old_qp
        globals()["load_models"] = old_lm
    # 5. the CLI: `val --run <pair run>` writes ITS cache name and never touches E325's, `--run` and `--pair-run` agree, `live` takes the pair run
    with tmp_root() as d:
        e325 = cache_path("val", 0)
        os.makedirs(os.path.dirname(e325))
        open(e325, "wb").write(b"E325 cache")
        stamp = (open(e325, "rb").read(), os.stat(e325).st_mtime_ns)
        calls = []
        old_vr, old_lr = globals()["val_records"], globals()["live_records"]
        none = dict(row=5, scene=1, ends=np.zeros((0, 8)), attrs=np.zeros((0, 6), np.float32), pairs=np.zeros((0, 8)), kp_p=np.zeros((0, 2)), gts=np.zeros((0, 2, 4)))   # a subset with no pair and no matched end
        assert np.isnan(pilot_stats([none])["err_median"]), "pilot_stats of a subset without a matched end is NaN, not a crash"
        globals()["val_records"] = lambda seed, run, thr=0.1, nmax=0, ends="kp": calls.append(("val", seed, run, thr, nmax, ends)) or ([none] if nmax == 1 else [toy_record(1, 2), toy_record(2, 3)])
        globals()["live_records"] = lambda seed, split="train", thr=0.1, nmax=0, workers=3, bs=4, device=None, pair_run=None, ends="kp": calls.append(("live", seed, split, nmax, pair_run, ends)) or [toy_record(1, 2)]
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                main(["val", "--seed", "0", "--run", "e324_s_s0"])
                main(["val", "--seed", "0", "--pair-run", "e324_s_s0"])                                       # the cache exists: loaded, not rebuilt
                main(["val", "--seed", "1"])
                main(["live", "--seed", "0", "--split", "val_clean", "--max", "24", "--pair-run", "e324_s_s0"])
                main(["val", "--seed", "1", "--max", "1"])                                                    # no matched end: the statistics print NaN
            assert calls == [("val", 0, "e324_s_s0", 0.1, 0, "kp"), ("val", 1, "e312_p_s1", 0.1, 0, "kp"), ("live", 0, "val_clean", 24, "e324_s_s0", "kp"), ("val", 1, "e312_p_s1", 0.1, 1, "kp")], calls
            for argv in (["val", "--seed", "0", "--pair-run", ""], ["val", "--seed", "0", "--run", ""], ["live", "--seed", "0", "--pair-run", ""], ["live", "--seed", "0", "--pair-run", "", "--out", os.path.join(d, "x.pkl")]):
                assert "run folder name" in (refused(main, argv) or ""), argv                                   # an empty pair run is refused, not read as the default
            assert len(calls) == 4 and not os.path.exists(os.path.join(d, "x.pkl")), "a refused call computed or wrote something"
            assert (open(e325, "rb").read(), os.stat(e325).st_mtime_ns) == stamp, "E325's cache was touched"
            for f, pr, sd in (("val_records_v1_s0_pair_e324_s_s0.pkl", "e324_s_s0", 0), ("val_records_v1_s1.pkl", "e312_p_s1", 1), ("val_clean_records_v1_s0_max24_pair_e324_s_s0.pkl", "e324_s_s0", 0)):
                assert pickle.load(open(os.path.join(d, "runs", "kept", "jwire", f), "rb"))["meta"] == dict(pair_run=pr, seed=sd, record_version=RECORD_VERSION), f
            try:
                main(["val", "--seed", "0", "--run", "e324_s_s0", "--pair-run", "e324_s_s1"])
                raise AssertionError("--run and --pair-run that differ must be refused")
            except AssertionError as e:
                assert "name the same thing" in str(e), e
        finally:
            globals()["val_records"], globals()["live_records"] = old_vr, old_lr


def _test_ends():
    """E358: the end kind (kp / att): the names and the meta of kp are what they were and att differs, a cache of the other kind is refused, the kind reaches `record_of` from `val_records` and `live_records` and the pairs of the
    record, att records differ from kp records on a frame with att_pos != kp_pos while the GT match stays on the keypoints, a frame without attached ends is refused for att (no fall back), and the command line"""
    import io
    import types
    from types import SimpleNamespace as NS
    import jwiring as JWI
    import torch
    up = np.array([0.0, 1.0])
    # 1. names and meta: kp (spelled out or not) is E325's / E350's bit for bit, att adds `_ends_att` / `ends`; a wrong kind or att on E312's own pair run is refused
    with tmp_root():
        name = lambda *a, **k: os.path.basename(cache_path(*a, **k))  # noqa: E731
        for a, k in ((("val", 0), {}), (("train", 1, 40), dict(pair_run="e324_s_s1")), (("val", 0, 0, 0.083), {}), (("val_clean", 0, 24), dict(pair_run="e312_p_s0"))):
            assert name(*a, **k, ends="kp") == name(*a, **k) and "ends" not in name(*a, **k), (a, k)
        assert name("val", 0, pair_run="e324_s_s0", ends="att") == "val_records_v1_s0_pair_e324_s_s0_ends_att.pkl"
        assert name("train", 1, 40, pair_run="e324_s_s1", ends="att") == "train_records_v1_s1_max40_pair_e324_s_s1_ends_att.pkl"
        assert name("val", 0, 0, 0.083, pair_run="e324_s_s0", ends="att") == "val_records_v1_s0_thr0.083_pair_e324_s_s0_ends_att.pkl"
        for bad in ("attached", "KP", "", None, 1):
            assert "one of" in (refusal(cache_path, "val", 0, pair_run="e324_s_s0", ends=bad) or ""), bad
            assert "one of" in (refusal(meta_of, 0, "e324_s_s0", bad) or ""), bad
        for pr in (None, "e312_p_s0"):
            assert "attached ends" in (refusal(cache_path, "val", 0, pair_run=pr, ends="att") or ""), pr         # E312's pair run has no att_pos
            assert "attached ends" in (refusal(meta_of, 0, pr, "att") or "")
        assert meta_of(0) == dict(pair_run="e312_p_s0", seed=0, record_version=RECORD_VERSION) == meta_of(0, None, "kp") and meta_of(1, "e324_s_s1") == dict(pair_run="e324_s_s1", seed=1, record_version=RECORD_VERSION)
        assert meta_of(1, "e324_s_s1", "att") == dict(pair_run="e324_s_s1", seed=1, record_version=RECORD_VERSION, ends="att")
    # 2. the content: a cache of the other kind is refused on load (its meta), a plain list (E325's caches) is E312's kp records only, a cache is built once
    with tmp_root():
        recs = [toy_record(1, 2)]
        pk, pa = cache_path("val", 0, pair_run="e324_s_s0"), cache_path("val", 0, pair_run="e324_s_s0", ends="att")
        save_records(pk, recs, 0, "e324_s_s0")
        save_records(pa, recs, 0, "e324_s_s0", "att")
        assert pickle.load(open(pk, "rb"))["meta"] == meta_of(0, "e324_s_s0") and pickle.load(open(pa, "rb"))["meta"] == meta_of(0, "e324_s_s0", "att")
        assert load_records(pk, 0, "e324_s_s0")[0]["row"] == 1 and load_records(pa, 0, "e324_s_s0", "att")[0]["row"] == 1
        assert "meta" in (refusal(load_records, pa, 0, "e324_s_s0") or ""), "an att cache must be refused for kp"
        assert "meta" in (refusal(load_records, pk, 0, "e324_s_s0", "att") or ""), "a kp cache must be refused for att"
        wrong = cache_path("train", 0, pair_run="e324_s_s0", ends="att")                                       # a kp cache that sits under the att name
        pickle.dump(dict(meta=meta_of(0, "e324_s_s0"), records=recs), open(wrong, "wb"))
        assert "meta" in (refusal(load_records, wrong, 0, "e324_s_s0", "att") or "") and "meta" in (refusal(cached_records, wrong, 0, "e324_s_s0", lambda: recs, "att") or "")
        old = cache_path("val", 1)
        pickle.dump(recs, open(old, "wb"))
        assert load_records(old, 1)[0]["row"] == 1 and refusal(load_records, old, 1, "e324_s_s1", "att"), "a plain list is E312's kp records"
        n = []
        make = lambda: n.append(1) or recs  # noqa: E731
        assert cached_records(cache_path("val", 0, 7, pair_run="e324_s_s0", ends="att"), 0, "e324_s_s0", make, "att") is recs and len(n) == 1
        assert cached_records(cache_path("val", 0, 7, pair_run="e324_s_s0", ends="att"), 0, "e324_s_s0", make, "att")[0]["row"] == 1 and len(n) == 1, "built once"
    # 3. the frame dict of the live pass: kp = the keys of a stored dump frame, att adds the attached ends of THIS batch item, an output without them is refused
    o = dict(kp_pos=torch.arange(8.0).reshape(2, 2, 2), kp_dir=torch.ones(2, 2, 2), kp_p=torch.rand(2, 2), valid=torch.ones(2, 2, 1, dtype=torch.bool), logit=torch.zeros(2, 1, 1), allowed=torch.ones(2, 1, 1))
    oa = dict(o, att_pos=100 + torch.arange(8.0).reshape(2, 2, 2), att_dir=0.5 + torch.arange(8.0).reshape(2, 2, 2) / 10)
    assert sorted(frame_dict(oa, 1)) == sorted(frame_dict(oa, 1, "kp")) == sorted(["kp_pos", "kp_dir", "kp_p", "valid", "logit", "allowed", "scale"]) and frame_dict(oa, 1)["scale"] is None
    fa = frame_dict(oa, 1, "att")
    assert np.array_equal(fa["att_pos"], oa["att_pos"][1].numpy()) and np.array_equal(fa["att_dir"], oa["att_dir"][1].numpy()) and np.array_equal(fa["kp_pos"], o["kp_pos"][1].numpy()) and fa["att_pos"].dtype == np.float32
    assert "attached ends" in (refusal(frame_dict, o, 1, "att") or "") and "attached ends" in (refusal(frame_dict, dict(o, att_pos=oa["att_pos"]), 1, "att") or ""), "no fall back to the keypoints"
    assert "one of" in (refusal(frame_dict, oa, 1, "attached") or "")
    # 4. the plumbing: the kind reaches `record_of` from `val_records` and from `live_records` (stubs for everything else; the attached ends of the model output reach the frame dict of the record)
    seen = []
    real_record_of = record_of
    globals()["record_of"] = lambda pred, graph, bk, e, g, thr=0.1, row=-1, scene=-1, ends="kp": seen.append((e, g, thr, row, scene, ends)) or toy_record(row, scene)

    class DS:
        missing = 0

        def __len__(self):
            return 2
    batch = dict(x=torch.zeros(2, 1), ent=torch.zeros(2, 1, 4), ent_m=torch.zeros(2, 1, dtype=torch.bool), ext=torch.zeros(2, 1, 4), ext_m=torch.zeros(2, 1, dtype=torch.bool), row=torch.tensor([5, 6]))
    fakes = dict(lane_gt_data=NS(QData=NS(), loader=lambda *a, **k: (DS(), [batch])), qgraph_eval=NS(readout=lambda pred, a, b: ("graph", "bk")), qgraph_readout=NS(parse=lambda s: s, apply=lambda d, spec: "pred"),
                 qpredict=NS(predict=lambda mb, b: [dict(nodes=dict(pos=0, dirv=0, e=0, flags=0, section=0), links=dict(src=0, dst=0, p=0, long=0, a=0, b=0))] * 2))
    saved = {k: sys.modules.get(k) for k in fakes}
    old_jwi, old_lm = (JWI._load, JWI._graph, JWI.GT_PATH), globals()["load_models"]
    try:
        with tmp_root() as d:
            os.makedirs(os.path.join(d, "runs", "e324_s_s0"))
            torch.save(dict(args=dict(seed=0, maps="/x/runs/e307_n_s0/model.pt")), os.path.join(d, "runs", "e324_s_s0", "model.pt"))
            pickle.dump({7: "gt7"}, open(os.path.join(d, "gt.pkl"), "wb"))
            JWI._load = lambda a: ("QE", "QR", ["dump"], {7: "e7"}, [NS(row=7, scene=3, _lane=1, _ref=1)], "spec")
            JWI._graph = lambda QE_, QR_, d_, spec: ("pred", "graph", "bk")
            JWI.GT_PATH = os.path.join(d, "gt.pkl")
            val_records(0, "e324_s_s0", ends="att")
            val_records(0, "e324_s_s0")
            assert seen == [("e7", "gt7", 0.1, 7, 3, "att"), ("e7", "gt7", 0.1, 7, 3, "kp")], seen
            assert "attached ends" in (refusal(val_records, 0, "e312_p_s0", ends="att") or "") and len(seen) == 2, "att on E312's pair run is refused before anything is read"
            seen.clear()
            for k, v in fakes.items():
                sys.modules[k] = types.ModuleType(k)
                sys.modules[k].__dict__.update(vars(v))
            for out_has_att, ends in ((True, "att"), (True, "kp"), (False, "kp")):
                model = lambda x, gt=None: (None, None, oa if out_has_att else o)  # noqa: E731
                globals()["load_models"] = lambda seed, pair_run, dev: ("B", model)
                with contextlib.redirect_stdout(io.StringIO()):
                    recs_ = live_records(0, "val_clean", 0.1, 0, device="cpu", pair_run="e324_s_s0", ends=ends)
                assert [r["row"] for r in recs_] == [5, 6] and [s[5] for s in seen[-2:]] == [ends, ends] and [s[3] for s in seen[-2:]] == [5, 6], seen
                for i, s in enumerate(seen[-2:]):
                    assert np.array_equal(s[0]["kp_pos"], o["kp_pos"][i].numpy()) and (("att_pos" in s[0]) == (ends == "att")), (ends, sorted(s[0]))
                    if ends == "att":
                        assert np.array_equal(s[0]["att_pos"], oa["att_pos"][i].numpy()) and np.array_equal(s[0]["att_dir"], oa["att_dir"][i].numpy()), "the attached ends of batch item i"
            n_seen = len(seen)
            globals()["load_models"] = lambda seed, pair_run, dev: ("B", lambda x, gt=None: (None, None, o))
            assert "attached ends" in (refusal(live_records, 0, "val_clean", 0.1, 0, device="cpu", pair_run="e324_s_s0", ends="att") or "") and len(seen) == n_seen, "att without attached ends in the output: refused, no fall back"
            loaded = []
            globals()["load_models"] = lambda seed, pair_run, dev: loaded.append(pair_run) or ("B", lambda x, gt=None: (None, None, oa))             # a model WITH attached ends: only the check of the pair run can refuse here
            assert "attached ends" in (refusal(live_records, 0, "val_clean", 0.1, 0, device="cpu", ends="att") or "") and not loaded and len(seen) == n_seen, "att on E312's pair run: refused before the model loads"
    finally:
        JWI._load, JWI._graph, JWI.GT_PATH = old_jwi
        globals()["record_of"], globals()["load_models"] = real_record_of, old_lm
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    # 5. the records: a frame with att_pos != kp_pos; kp (spelled out or not, with or without att_pos in the frame) is the old record bit for bit, att holds the attached ends in its pairs and the keypoints' GT match
    pred, graph, bk, e, g = toy_frame()
    att = dict(e, att_pos=np.array([[0.0, -0.8], [0.0, 1.9]]), att_dir=np.tile(up, (2, 1)))
    far = dict(e, att_pos=np.array([[0.0, -8.0], [0.0, 9.0]]), att_dir=np.tile(up, (2, 1)))                  # attached ends 7 m from the GT ends: only a match on the keypoints finds them
    rk, ra, rf = record_of(pred, graph, bk, e, g, 0.1), record_of(pred, graph, bk, att, g, 0.1, ends="att"), record_of(pred, graph, bk, far, g, 0.1, ends="att")
    for r in (record_of(pred, graph, bk, e, g, 0.1, ends="kp"), record_of(pred, graph, bk, att, g, 0.1, ends="kp")):
        assert all(np.array_equal(r[k], rk[k], equal_nan=True) for k in ("ends", "attrs", "pairs", "kp_p", "gts")), "kp records must not depend on the attached ends"
    assert np.allclose(rk["pairs"][0], np.r_[[0.2, 0.0], up, [0.1, 1.0], up]) and np.allclose(ra["pairs"][0], np.r_[[0.0, -0.8], up, [0.0, 1.9], up]) and np.allclose(rf["pairs"][0], np.r_[[0.0, -8.0], up, [0.0, 9.0], up])
    for r in (ra, rf):
        assert np.array_equal(r["gts"], rk["gts"]) and not np.isnan(r["gts"]).any() and np.allclose(r["gts"][0, 0], g["ent"][0]) and np.array_equal(r["kp_p"], rk["kp_p"]), "the match to the GT end stays on the keypoints"
    ik, ia = instances_of(rk), instances_of(ra)
    assert len(ik) == len(ia) == 2 and all(abs(x["feat"][0, 0] * JH.REACH - d) < 1e-5 for x, d in zip(ik, (np.hypot(0.2, 1.0), np.hypot(0.1, 1.0)))) and all(abs(x["feat"][0, 0] * JH.REACH - d) < 1e-5 for x, d in zip(ia, (0.2, 0.1))), "the features are relative to the pair end"
    assert pilot_stats([rk])["near05"] == 0.0 and pilot_stats([ra])["near05"] == 1.0, "near05: 1.02 / 1.00 m (keypoints) against 0.2 / 0.1 m (attached ends)"
    assert "attached ends" in (refusal(record_of, pred, graph, bk, e, g, 0.1, ends="att") or "") and "attached ends" in (refusal(record_of, pred, graph, bk, dict(e, att_pos=att["att_pos"]), g, 0.1, ends="att") or ""), "no fall back to the keypoints"
    assert "one of" in (refusal(record_of, pred, graph, bk, att, g, 0.1, ends="attached") or "")
    # 6. the pilot's share within 0.5 m counts the ends with an oracle label: the nearest piece end 0.2 / 0.48 m away (near), 0.52 / 1.02 m away (far), no GT end (not counted)
    at_m = lambda d, row: dict(toy_record(row, row), pairs=np.array([np.r_[[0.0, -1.0 + d], up, [5.0, 20.0], up]]))  # noqa: E731  (the entry d metres from the end of the first piece)
    nogt = dict(toy_record(5, 5), gts=np.full((1, 2, 4), np.nan))
    assert pilot_stats([at_m(0.2, 1), at_m(0.48, 2), at_m(0.52, 3), toy_record(4, 4), nogt])["near05"] == 0.5 and pilot_stats([at_m(0.2, 1)])["near05"] == 1.0 and pilot_stats([toy_record(4, 4)])["near05"] == 0.0
    assert np.isnan(pilot_stats([nogt])["near05"]), "no labelled end: NaN, not a crash"
    # 7. the command line: --ends att reaches val_records / live_records, the cache name and the meta; att without an attach pair run is refused before anything is computed; another kind than kp / att is not a choice
    with tmp_root() as d:
        calls = []
        old_vr, old_lr = globals()["val_records"], globals()["live_records"]
        globals()["val_records"] = lambda seed, run, thr=0.1, nmax=0, ends="kp": calls.append(("val", seed, run, ends)) or [toy_record(1, 2), toy_record(2, 3)]
        globals()["live_records"] = lambda seed, split="train", thr=0.1, nmax=0, workers=3, bs=4, device=None, pair_run=None, ends="kp": calls.append(("live", seed, split, pair_run, ends)) or [toy_record(1, 2)]
        try:
            with contextlib.redirect_stdout(io.StringIO()) as out:
                main(["val", "--seed", "0", "--pair-run", "e324_s_s0", "--ends", "att"])
                main(["val", "--seed", "0", "--run", "e324_s_s0"])
                main(["live", "--seed", "0", "--split", "val_clean", "--max", "24", "--pair-run", "e324_s_s0", "--ends", "att"])
            assert calls == [("val", 0, "e324_s_s0", "att"), ("val", 0, "e324_s_s0", "kp"), ("live", 0, "val_clean", "e324_s_s0", "att")], calls
            assert "(att ends)" in out.getvalue() and "attached-end error" in out.getvalue() and "(kp ends)" in out.getvalue() and "keypoint error" in out.getvalue()
            for f, ends in (("val_records_v1_s0_pair_e324_s_s0_ends_att.pkl", "att"), ("val_records_v1_s0_pair_e324_s_s0.pkl", "kp"), ("val_clean_records_v1_s0_max24_pair_e324_s_s0_ends_att.pkl", "att")):
                assert pickle.load(open(os.path.join(d, "runs", "kept", "jwire", f), "rb"))["meta"] == meta_of(0, "e324_s_s0", ends), f
            for argv in (["val", "--seed", "0", "--ends", "att"], ["val", "--seed", "0", "--pair-run", "e312_p_s0", "--ends", "att"], ["live", "--seed", "0", "--ends", "att"], ["live", "--seed", "0", "--ends", "att", "--out", os.path.join(d, "x.pkl")]):
                assert "attached ends" in (refusal(main, argv) or ""), argv
            assert len(calls) == 3 and not os.path.exists(os.path.join(d, "x.pkl")), "a refused call computed or wrote something"
            with contextlib.redirect_stderr(io.StringIO()):
                for argv in (["val", "--seed", "0", "--ends", "attached"], ["live", "--seed", "0", "--ends", ""]):
                    try:
                        main(argv)
                        raise AssertionError(f"{argv} must be refused")
                    except SystemExit as ex:
                        assert ex.code == 2, argv
        finally:
            globals()["val_records"], globals()["live_records"] = old_vr, old_lr


def _test_faithful():
    """E358 attempt 4 (`live_records(..., faithful=True)`): both models run on loaders with the settings of THEIR stored dumps (`DUMP_B`, `DUMP_PAIR`), the two joined by the frame's row although the batches differ in size, the
    GT ends of the record come from B's loader, the flags of the loader class are set for the pair loader the way the dump set them, it is a check of at most FAITHFUL_MAX val_clean frames, and the default path is untouched"""
    import types
    from types import SimpleNamespace as NS
    import torch
    rows = [10, 11, 12, 13, 14, 15]
    calls, seen, pcalls = [], [], []

    miss = []                                                                                              # the pair loader's dataset has frames without stored lane GT
    reverse = []                                                                                           # the pair loader hands the frames out in REVERSED order (the join must not depend on it)

    def make(bs, coded):
        out = []
        order = rows[::-1] if (coded and reverse) else rows
        for a in range(0, len(order), bs):
            r = order[a:a + bs]
            t = torch.tensor(r, dtype=torch.float32)
            out.append(dict(x=t[:, None], row=torch.tensor(r), ent=t[:, None, None].repeat(1, 1, 4), ent_m=torch.zeros(len(r), 1, dtype=torch.bool), ext=t[:, None, None].repeat(1, 1, 4), ext_m=torch.zeros(len(r), 1, dtype=torch.bool)))
        return out
    QData = NS(DEFAULT_ENDS=False, DEFAULT_JUNCTION=False)

    def loader(split, bs=0, workers=0, shuffle=None, map_mode=None, pixels=None, strict=False, ends=False):
        calls.append(dict(split=split, bs=bs, workers=workers, shuffle=shuffle, map_mode=map_mode, pixels=pixels, strict=strict, ends=ends, flags=(QData.DEFAULT_ENDS, QData.DEFAULT_JUNCTION)))
        return NS(missing=1 if (len(calls) == 2 and miss) else 0), make(bs, len(calls) == 2)
    fakes = dict(lane_gt_data=NS(QData=QData, loader=loader), qgraph_eval=NS(readout=lambda pred, a, b: (pred, "bk")), qgraph_readout=NS(parse=lambda s: s, apply=lambda d, spec: d["nodes"]["pos"]),
                 qpredict=NS(predict=lambda mb, b: pcalls.append(len(b["row"])) or [dict(nodes=dict(pos=float(b["x"][i, 0]), dirv=0, e=0, flags=0, section=0), links=dict(src=0, dst=0, p=0, long=0, a=0, b=0)) for i in range(len(b["row"]))]))

    def model(x, gt=None):                                                                                 # the pair model: every output is coded with the frame's row
        n = x.shape[0]
        o = dict(kp_pos=x[:, :, None].repeat(1, 4, 2), kp_dir=torch.zeros(n, 4, 2), kp_p=torch.zeros(n, 4), valid=torch.ones(n, 2, 2, dtype=torch.bool), logit=torch.zeros(n, 2, 2), allowed=torch.ones(n, 2, 2, dtype=torch.bool),
             att_pos=100.0 + x[:, :, None].repeat(1, 4, 2), att_dir=torch.zeros(n, 4, 2))
        return None, None, o
    saved = {k: sys.modules.get(k) for k in fakes}
    real_record_of, real_lm = record_of, load_models
    try:
        for k, v in fakes.items():
            sys.modules[k] = types.ModuleType(k)
            sys.modules[k].__dict__.update(vars(v))
        globals()["record_of"] = lambda pred, graph, bk, e, g, thr=0.1, row=-1, scene=-1, ends="kp": seen.append((pred, float(e["kp_pos"][0, 0]), float(g["ent"][0, 0]), row, ends, float(e["att_pos"][0, 0]) if "att_pos" in e else None)) or toy_record(row, scene)
        globals()["load_models"] = lambda seed, pair_run, dev: ("B", model)
        recs = live_records(0, "val_clean", 0.1, 5, device="cpu", pair_run="e324_s_s0", ends="att", faithful=True)
        assert [r["row"] for r in recs] == rows[:5], [r["row"] for r in recs]
        assert seen == [(float(r), float(r), float(r), r, "att", 100.0 + r) for r in rows[:5]], seen             # B's prediction, the pair model's output (keypoints AND attached ends) and the GT ends of every record belong to the SAME row
        assert len(calls) == 2, calls
        b_call, p_call = calls
        assert (b_call["split"], b_call["bs"], b_call["map_mode"], b_call["pixels"], b_call["strict"], b_call["ends"], b_call["shuffle"]) == ("val_clean", DUMP_B["bs"], DUMP_B["map_mode"], False, True, True, False) == ("val_clean", 4, "exact", False, True, True, False), b_call
        assert (p_call["split"], p_call["bs"], p_call["workers"], p_call["map_mode"], p_call["pixels"], p_call["shuffle"]) == ("val_clean", DUMP_PAIR["bs"], DUMP_PAIR["workers"], DUMP_PAIR["map_mode"], False, False) == ("val_clean", 8, 2, "error", False, False), p_call
        assert b_call["flags"] == (True, False) and p_call["flags"] == (False, True), (b_call["flags"], p_call["flags"])      # B's loader: ends flag on; the pair loader as the dump set it: junction flag on, the ends flag back where it was
        assert (QData.DEFAULT_ENDS, QData.DEFAULT_JUNCTION) == (False, False), "both loader flags are put back after the pass"
        reverse.append(1)
        calls.clear()
        recs_r = live_records(0, "val_clean", 0.1, 5, device="cpu", pair_run="e324_s_s0", ends="att", faithful=True)
        assert [r["row"] for r in recs_r] == rows[:5] and sorted(seen[-5:]) == sorted((float(r), float(r), float(r), r, "att", 100.0 + r) for r in rows[:5]) and [s[3] for s in seen[-5:]] == rows[:5], seen[-5:]      # a reversed pair loader joins the same way
        reverse.clear()
        calls.clear()
        miss.append(1)
        assert "no stored lane GT" in (refusal(live_records, 0, "val_clean", 0.1, 5, device="cpu", pair_run="e324_s_s0", ends="att", faithful=True) or ""), "the pair loader's dataset is checked like the dump does"
        miss.clear()
        assert (QData.DEFAULT_ENDS, QData.DEFAULT_JUNCTION) == (False, False), "the flags are put back also when the pair pass is refused"
        calls.clear()
        pcalls[:] = [4, 2]
        calls.extend([b_call, p_call])
        assert pcalls == [4, 2], pcalls                                                                        # B's loader was read until nmax = 5 frames were there (two batches)
        pcalls.clear()
        r4 = live_records(0, "val_clean", 0.1, 4, device="cpu", pair_run="e324_s_s0", ends="att", faithful=True)
        assert [r["row"] for r in r4] == rows[:4] and pcalls == [4], (pcalls, "B's loader stops as soon as nmax frames are there")
        calls[:] = calls[:2]
        assert DUMP_B == dict(bs=4, map_mode="exact") and DUMP_PAIR == dict(bs=8, map_mode="error", workers=2) and FAITHFUL_MAX == 64
        # the guards: only a check of val_clean frames, never a TRAIN pass, never without nmax or above FAITHFUL_MAX
        n_calls = len(calls)
        for kw in (dict(split="train", nmax=5), dict(split="val_clean", nmax=0), dict(split="val_clean", nmax=FAITHFUL_MAX + 1)):
            assert "faithful" in (refusal(live_records, 0, kw["split"], 0.1, kw["nmax"], device="cpu", pair_run="e324_s_s0", ends="att", faithful=True) or ""), kw
        assert len(calls) == n_calls, "refused before any loader was built"
        # a pair loader that never reaches a row of B's pass is an error, not a short result
        globals()["load_models"] = lambda seed, pair_run, dev: ("B", model)
        short_rows, rows[:] = rows[:], rows[:4]
        try:
            fakes_loader = sys.modules["lane_gt_data"].loader
            first = []

            def loader2(split, bs=0, **k):
                ds, b = fakes_loader(split, bs=bs, **k)
                first.append(bs)
                if len(first) == 2:
                    b = b[:0]                                                                              # the pair loader gives nothing
                return ds, b
            sys.modules["lane_gt_data"].loader = loader2
            rows[:] = short_rows
            assert "pair loader" in (refusal(live_records, 0, "val_clean", 0.1, 5, device="cpu", pair_run="e324_s_s0", ends="att", faithful=True) or ""), "missing rows on the pair side"
        finally:
            sys.modules["lane_gt_data"].loader = fakes_loader
            rows[:] = short_rows
        # the default path (faithful off) builds ONE loader, exactly as before
        calls.clear()
        live_records(0, "val_clean", 0.1, 4, device="cpu", pair_run="e324_s_s0", ends="att")
        assert len(calls) == 1 and calls[0]["bs"] == 4 and calls[0]["map_mode"] == "exact", calls
    finally:
        globals()["record_of"], globals()["load_models"] = real_record_of, real_lm
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def _selftest():
    import jwire_head as H
    up = np.array([0.0, 1.0])
    rec = toy_record()
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
    pred, graph, bk, e, g = toy_frame()
    lines = graph.lines
    rec2 = record_of(pred, graph, bk, e, g, 0.1)
    assert rec2["pairs"].shape == (1, 8) and np.allclose(rec2["gts"][0, 0], g["ent"][0]) and np.allclose(rec2["gts"][0, 1], g["ext"][0]), rec2["gts"]
    assert not np.allclose(rec2["gts"][0, 0], g["ext"][0])                                                     # an entry / exit swap in record_of would put the exit GT in the entry slot
    pa = [(np.array([0.2, 0.0]), up, np.array([0.1, 1.0]), up)]
    keep_, en_ = JW.kept_ends(lines, bk["kind"], bk["piece_links"], np.zeros(2, bool), pa, QEV.cover_pos())
    at_ = H.piece_attrs(lines, bk["kind"], bk["piece_nodes"], bk["piece_links"], pred["nodes"]["e"], pred["links"]["p"], keep_)
    direct = H.build_instances(en_, at_, pa, [(0.7, 0.6)], [(g["ent"][0], g["ext"][0])])
    viarec = instances_of(rec2)
    assert len(direct) == len(viarec) and all(np.allclose(x["feat"], y["feat"], atol=1e-5) and x["label"] == y["label"] for x, y in zip(direct, viarec)), "records and direct instances differ"
    _test_pair_run()
    _test_ends()
    _test_faithful()
    print("jwire_data selftest ok")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("val")
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--pair-run", default=None, help="the pair run whose stored dump gives the keypoints (default e312_p_s<seed>)")
    s.add_argument("--run", default=None, help="the older name of --pair-run")
    s.add_argument("--ends", default="kp", choices=ENDS, help="the pair ends of the records: the keypoints (default) or the ATTACHED ends of the pair run (E358: needs --pair-run of an attach model)")
    s.add_argument("--thr", type=float, default=0.1)
    s.add_argument("--max", type=int, default=0)
    t = sub.add_parser("live")
    t.add_argument("--seed", type=int, required=True)
    t.add_argument("--pair-run", default=None, help="the pair run of the live pass (default e312_p_s<seed>)")
    t.add_argument("--ends", default="kp", choices=ENDS, help="the pair ends of the records: the keypoints (default) or the ATTACHED ends of the pair run (E358: needs --pair-run of an attach model)")
    t.add_argument("--split", default="train")
    t.add_argument("--max", type=int, default=0)
    t.add_argument("--workers", type=int, default=3)
    t.add_argument("--device", default=None)
    t.add_argument("--out", default=None)
    sub.add_parser("selftest")
    a = ap.parse_args(argv)
    if a.cmd == "selftest":
        _selftest()
        return
    if a.cmd == "live":
        pr = a.pair_run
        resolve_pair_run(a.seed, pr)                                                                       # an empty name is refused here, not read as the default
        check_ends(a.ends, a.seed, pr)                                                                     # ... and so is att without an attach pair run, before the pass
        path = a.out or cache_path(a.split, a.seed, a.max, 0.1, pr, a.ends)
        built = not os.path.exists(path)
        recs = cached_records(path, a.seed, pr, lambda: live_records(a.seed, a.split, 0.1, a.max, a.workers, device=a.device, pair_run=pr, ends=a.ends), a.ends)
        print(f"{len(recs)} records ({a.ends} ends) -> {os.path.relpath(path, ROOT)}" + ("" if built else " (the cache exists: loaded, not rebuilt)"))
        print(pilot_stats(recs))
        return
    assert a.run is None or a.pair_run is None or a.run == a.pair_run, f"--run {a.run} and --pair-run {a.pair_run} name the same thing and differ"
    pr = a.pair_run if a.pair_run is not None else a.run
    path = cache_path("val", a.seed, a.max, a.thr, pr, a.ends)
    built = not os.path.exists(path)
    recs = cached_records(path, a.seed, pr, lambda: val_records(a.seed, resolve_pair_run(a.seed, pr), a.thr, a.max, a.ends), a.ends)
    tot = {}
    for r in recs:
        for k, v in agreement(instances_of(r)).items():
            tot[k] = tot.get(k, 0) + v
    print(f"{len(recs)} records ({a.ends} ends) -> {os.path.relpath(path, ROOT)}" + ("" if built else " (the cache exists: loaded, not rebuilt)"))
    print(tot)
    if tot["oracle"]:
        print(f"rule agreement with the oracle label: overall {tot['same'] / tot['oracle']:.3f} ({tot['oracle']} ends), real choices {tot['multi_same'] / max(tot['multi'], 1):.3f} ({tot['multi']} ends, share of the labelled ends {tot['multi'] / tot['oracle']:.3f}); label outside the keypoint's places: {tot['outside']}")
    ps = pilot_stats(recs)
    print(f"{'keypoint' if a.ends == 'kp' else 'attached-end'} error of the matched ends: mean {ps['err_mean']:.2f} / median {ps['err_median']:.2f} / p90 {ps['err_p90']:.2f} m ({ps['matched_ends']} ends); share of the labelled ends whose nearest piece end is within {NEAR_M} m: {ps['near05']:.3f}")


if __name__ == "__main__":
    main()
