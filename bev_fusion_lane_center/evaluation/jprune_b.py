"""Step 0 of E352 (no training): what do the E345 hook edits of B's pyramid trunk do to B's OWN graph, and to the junction system on B's re-dumped graph? The probe applies one point set {unit: f} (E345 semantics: the lowest-importance share f of the
channels of the unit's tensor is replaced by its calibration mean; ranking and means from `temp/e345/e324_s_s<S>/calib.pt` through `jprune_probe`) to the pyramid of the junction system (`runs/e324_s_s<S>`) AND to the pyramid of B (`runs/e273_p_s<S>`, the QGraph that makes the
delivered graph), in ONE pass over the 873 VAL frames, and writes a point folder: model.pt (a link to the unmodified E324 s checkpoint), ends_val.pkl (the junction dump on the edited pyramid, the format of `jends_eval.py dump`), predfull_<tag>_s<S>.pkl
(B' = B's full prediction on the edited pyramid, the format of `temp/e276/predfull_e273_p_s<S>.pkl`: `qpredict.predict` + `qgraph_readout.dump_full`, exact map, as `qgraph_run_eval.score --dump-full`), point.json (the set, the channels, the saved cost, the sha256 of both pickles and of the
calibration, the device, the seconds, `partial`; written LAST: a folder without it is incomplete). Nothing in a stored run is touched; every file goes through a temporary file and `os.replace`; a complete folder is never overwritten; a partial dump (--max) needs --out and cannot land in `runs/`.

    wsl bash tools/env/_py14.sh jprune_b.py plan     [--out temp/e352/points.json]            # CPU, no data: the 9 point sets (null, T1, T2, random draws), channel counts, saved GMACs; the point-set file
    wsl bash tools/env/_py14.sh jprune_b.py dump     --seed 0 --point t1 --groups [--threads N] [--device cuda] [--workers N] [--max N --out DIR]    # one point of one seed -> runs/e352g_t1_s0/
    wsl bash tools/env/_py14.sh jprune_b.py check    --seed 0 [--max 24]                       # the EMPTY set against the stored B dump (temp/e276) and the stored E345 junction null, as arrays
    wsl bash tools/env/_py14.sh jprune_b.py gate     --seed 0 [--null-run e352_null_s0]        # the NULL gate: the null's cards against the stored E324 s cards, every number against its tolerance; exit 1 when one fails
    wsl bash tools/env/_py14.sh jprune_b.py table    [--seeds 0 1] [--out FILE]                # the step-0 table from the cards of all point folders, as markdown, with the rule's lines
    wsl bash tools/env/_py14.sh jprune_b.py commands --seed 0 [--point t1] [--groups] [--readout snap]   # the scoring lines (jends_eval.py score --full --base --tag, `$R` = the problem folder)
    wsl bash tools/env/_py14.sh jprune_b.py selftest

Sets (E352 card, step 0): T1 = {pyr.e2: 0.5, pyr.d2: 0.25, pyr.e3: 0.25}, T2 = T1 + {pyr.d0: 0.25, pyr.e1: 0.25, pyr.d1: 0.25}; fractions of the E345 ladder. TWO VARIANTS: `--groups` (the JUDGED set: WHOLE GroupNorm groups, ranked by the summed channel importance, a single channel inside a GroupNorm(8)
group cannot be built physically; same counts as the channel variant) and the channel variant (E345's per-channel ranking: the ANCHOR to the E345 table, context only). Controls: for each set 3 RANDOM draws of the same count per unit (channels, or whole groups in the groups variant); an independent RandomState
per (variant, draw, unit) from a CRC32 of the names (E345 used `1000 * draw + sum(ord(c))`, which gives `pyr.d2` and `pyr.e1` the same seed 543 + 1000 draw: its single-unit draws are other draws than these; a draw here is nested: the 0.25 set of a unit is inside its 0.5 set). Points and folders:
null (variant-free: `e352_null_s<S>`, tag null), groups `e352g_<pt>_s<S>` (tag g<pt>), channels `e352c_<pt>_s<S>` (tag c<pt>), pt in t1, t2, r1d0..r1d2 (draws of T1), r2d0..r2d2 (draws of T2). Run ids of the chain: E352.s0_null_s<S>, E352.s0g_<pt>_s<S>, E352.s0c_<pt>_s<S>, each card
job with the read-out before `_s<S>` (`E352.s0g_t1_snap_s0`). B's plain graph is `numbers.base` of any card; the junction numbers use B' as the delivered graph (`--full / --base / --tag` of `qends_eval.cmd_score`, default read-out `close:1.5@0.5` = the hard close-node step of B's delivery).
A pass runs BOTH models on the same batch: a hook that does not fire once per forward or changes nothing stops the dump (E345's `Watch`). On the CPU the pass takes about 0.7 s per frame (about 11 minutes per point of 873 frames, 2 threads); B's section stage needs the single-threaded solve of
`cpu_solve_fix` there (the threaded oneMKL LU of this box fails for 512 x 512). B' on the CPU matches the stored dump of the GPU as sets (the order of equal-valued nodes differs) and the scorer's numbers to the tolerances of `gate`. The tests need the scratch inputs temp/e276/predfull_e273_p_s0.pkl,
temp/e345/e324_s_s0/calib.pt (and seed 1's) and runs/e345_p000_e324_s_s0/ends_val.pkl.
"""
import argparse
import contextlib
import copy
import hashlib
import json
import os
import pickle
import sys
import time
import zlib

import numpy as np
import torch
import torch.nn as nn

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for sub_ in ("evaluation", "training", "training/grid", "training/data", "training/data/prep", "training/train", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub_)))
import jends_eval as JEV  # noqa: E402
import jprune_cost as PC  # noqa: E402
import jprune_probe as PP  # noqa: E402
from jprune_cost import Edit  # noqa: E402

T1 = {"pyr.e2": 0.5, "pyr.d2": 0.25, "pyr.e3": 0.25}
T2 = dict(T1, **{"pyr.d0": 0.25, "pyr.e1": 0.25, "pyr.d1": 0.25})
POINTS = dict([("null", ({}, "rank", None)), ("t1", (T1, "rank", None)), ("t2", (T2, "rank", None))]
              + [(f"r{k}d{d}", (s, "random", d)) for k, s in ((1, T1), (2, T2)) for d in range(3)])                    # name -> (set, how, draw)
IDS = {"null": 0, "t1": 201, "t2": 202, **{f"r{k}d{d}": 200 + 10 * k + d + 1 for k in (1, 2) for d in range(3)}}          # point ids of the point-set file (E345 uses 1 - 120, 900)
RANDOM_OF = {"t1": ("r1d0", "r1d1", "r1d2"), "t2": ("r2d0", "r2d1", "r2d2")}                                              # the random draws that belong to a ranked set
READOUTS = ("snap", "primary", "attach")
KEYS = ("Q", "F1@1.0", "lanes_found", "topo_F1", "topo_P", "topo_R", "junction_F1", "junction_P", "junction_R", "dup_frac", "vert100_dp")      # the numbers of a card (`qends_eval.KEYS`)
STORED = {"snap": "ends_card_snap_thr0.1.json", "primary": "ends_card_snap_nogeom_attach_thr0.1.json", "attach": "ends_card_attach_thr0.1.json"}    # the stored cards of E324 s that the null must reproduce
GATE_TOPO_F1 = 2e-4                                                                                                        # the null gate: topo F1 of base and head, absolute
GATE_ABS, GATE_REL = 2e-3, 1e-3                                                                                            # every other number: |difference| <= max(GATE_ABS, GATE_REL * |stored|) (observed, 2 seeds x 3 read-outs x base / head: at most 8.6e-4 on the 0 - 1 numbers = 0.43 of GATE_ABS, 4.5e-3 on vert100_dp (about 19) = 0.23 of GATE_REL * 19.6)


def runs_of(seed):
    return f"e324_s_s{seed}", f"e273_p_s{seed}"


def variant(name, groups):
    """'' for the null (no edit: the same pass in both variants), g = whole groups (judged), c = channels (the E345 anchor)"""
    return "" if name == "null" else ("g" if groups else "c")


def point_run(seed, name, groups=False):
    """the folder name under `runs/`: e352_null_s<S>, e352g_<pt>_s<S> (whole groups), e352c_<pt>_s<S> (channels)"""
    return f"e352{variant(name, groups)}_{name}_s{seed}"


def point_tag(name, groups=False):
    """the `--tag` of the scoring lines (lower-case letters and digits): null, g<pt>, c<pt>"""
    return f"{variant(name, groups)}{name}"


def run_id(seed, name, groups=False, readout=None):
    """the run.py id of a dump (readout None) or a card job: E352.s0_null_s0, E352.s0g_t1_s0, E352.s0g_t1_snap_s0, E352.s0c_t2_s1"""
    return f"E352.s0{variant(name, groups)}_{name}{'_' + readout if readout else ''}_s{seed}"


def sha256_file(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


# ---------------------------------------------------------------- 1. the point sets and their edits
def rng_for(unit, draw, groups):
    """an independent RandomState per (variant, draw, unit): a CRC32 of the names (E345's `1000 * draw + sum(ord(c))` collides: `pyr.d2` and `pyr.e1` both sum to 543)"""
    return np.random.RandomState(zlib.crc32(f"e352|{'groups' if groups else 'channels'}|{draw}|{unit}".encode()))


def group_edits(model, unit, f, calib, how="rank", rng=None):
    """the edits of a unit at share f that remove WHOLE GroupNorm groups (the block is conv GroupNorm ReLU conv GroupNorm ReLU: the GroupNorm in front of the last ReLU); ranked by the summed importance of the group's channels, or random groups"""
    out = []
    for tap in unit.taps:
        gn = PC.sub(model, tap)[-2]
        assert isinstance(gn, nn.GroupNorm), tap
        G, gs = gn.num_groups, gn.num_channels // gn.num_groups
        n = int(f * G + 0.5)
        gone = rng.permutation(G)[:n].tolist() if how == "random" else torch.argsort(calib.s[tap].double().view(G, gs).sum(1), descending=True, stable=True)[:n].tolist()
        idx = sorted(c for g in gone for c in range(g * gs, (g + 1) * gs))
        out.append(Edit("chan", tap=tap, idx=tuple(idx), val=calib.mean[tap][idx]))
    return out


def point_edits(model, calib, name, groups=False):
    """-> the edits of a point: ranked (E345 `edits_for`, or whole groups) or the random draw of the point, per unit with its own RandomState (`rng_for`)"""
    spec, how, draw = POINTS[name]
    units = {u.name: u for u in PP.build_units(model)}
    out = []
    for unit, f in spec.items():
        u = units[unit]
        assert u.kind == "chan" and unit.startswith("pyr.") and f in u.fracs, f"{unit}@{f} is not a pyramid point of the E345 ladder"
        rng = rng_for(unit, draw, groups) if how == "random" else None
        out += group_edits(model, u, f, calib, how, rng) if groups else PP.edits_for(u, f, calib, how, rng)
    return out


class Shim(nn.Module):
    """B's QGraph under the names of the E345 tap table: `PP.ablate` resolves a tap `pyr.e2` as `jmodel.pyr.e2`. The QGraph's own pyramid is reachable under both paths (a deepcopy keeps the alias), so the hooks of the copy sit on the QGraph that predicts"""

    def __init__(self, qgraph):
        super().__init__()
        self.qg = qgraph
        self.jmodel = nn.Module()
        self.jmodel.pyr = qgraph.pyr


def same_pyramid(qgraph, jmodel):
    """B's pyramid tensors are the junction system's (the premise of the probe): -> the number of tensors compared; asserts"""
    sb, sj = qgraph.state_dict(), jmodel.state_dict()
    keys = [k for k in sb if k.startswith("pyr.")]
    assert len(keys) == 57 and all(torch.equal(sb[k], sj["jmodel." + k]) for k in keys), "B's pyramid is not the junction system's: the same edit would not be the same network"
    return len(keys)


@contextlib.contextmanager
def cpu_solve_fix(device):
    """B's section stage (`sections.py`) solves a batched 512 x 512 linear system with `torch.linalg.solve`. On this WSL box the threaded oneMKL LU fails for it ("Parameter 6 was incorrect on entry to SLASWP") and takes minutes per batch; with one thread it takes 0.03 s and
    gives the residual of float32 (6e-7). On the CPU the solve therefore runs single-threaded for the duration of the pass (the same numbers up to the rounding of the other algorithm; no digested file is edited); on the GPU nothing changes"""
    if str(device) != "cpu":
        yield
        return
    orig = torch.linalg.solve

    def solve1(A, B, **kw):
        n = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            return orig(A, B, **kw)
        finally:
            torch.set_num_threads(n)
    torch.linalg.solve = solve1
    try:
        yield
    finally:
        torch.linalg.solve = orig


def load_models(seed, device):
    import qpredict as QP
    j, b = runs_of(seed)
    jm, args = JEV.load_model(j, device)
    qg = QP.load_model(os.path.join(ROOT, "runs", b), device)
    same_pyramid(qg, jm)
    return jm, args, qg


def calib_path(seed):
    return PP.rootpath(f"temp/e345/e324_s_s{seed}/calib.pt")


def check_out(max_frames, out):
    """a partial dump (--max) needs --out and must not land in `runs/`: a partial point.json would read as a complete point"""
    if not max_frames:
        return
    if not out:
        raise SystemExit("--max writes a partial dump: give --out (a folder outside runs/)")
    if os.path.realpath(PP.rootpath(out)).startswith(os.path.realpath(os.path.join(ROOT, "runs")) + os.sep):
        raise SystemExit(f"--max writes a partial dump: {out} is inside runs/ (give a folder outside it)")


# ---------------------------------------------------------------- 2. one pass: B' and the junction dump on the edited pyramid
def run_pass(seed, name, device="cpu", max_frames=0, groups=False, workers=0, log=print):
    """-> dict(frames (the junction dump), preds + rows (B'), edits, saved cost, seconds, total (the frames of the loader)): both models see the same edited pyramid and the same batches (B's loader: exact map, batch 4, strict, as `jtest_eval.dump_b`; the junction model reads only x)"""
    import lane_gt_data as LD
    import qpredict as QP
    PP.wait_for_gpu(f"jprune_b {name}_s{seed}", device)
    jm, args, qg = load_models(seed, device)
    calib = PP.load_calib(calib_path(seed), "cpu")
    edits = point_edits(jm, calib, name, groups)
    taps = PP.tap_table(jm)
    cost = PC.Cost(copy.deepcopy(jm).cpu()).saved(edits)
    mj, wj = PP.ablate(jm, edits, taps)
    shim, wb = PP.ablate(Shim(qg), edits, taps)
    mb = shim.qg
    assert mb.pyr is shim.jmodel.pyr and next(mb.parameters()).device == next(mj.parameters()).device
    ds, dl = LD.loader("val_clean", bs=4, workers=workers, shuffle=False, map_mode="exact", pixels=False, strict=True)
    rows = [int(ds.bd.rows[k]) for k in ds.ks]
    frames, preds, nb, t0 = [], [], 0, time.time()
    with torch.no_grad(), cpu_solve_fix(device):
        for b in dl:
            out = mj(b["x"].to(device).float(), gt=None)[2]
            frames += PP.frames_of({k: (v.cpu() if torch.is_tensor(v) else v) for k, v in out.items()}, b)
            preds += QP.predict(mb, b)
            nb += 1
            if max_frames and len(preds) >= max_frames:
                break
            if nb % 25 == 0:
                log(f"  {name} s{seed}: {len(preds)}/{len(rows)} frames  {time.time() - t0:.0f} s", flush=True)
    wj.check(nb, calib)
    wb.check(nb, calib)
    n = max_frames or len(preds)
    frames, preds = frames[:n], preds[:n]
    assert [f["row"] for f in frames] == rows[:n], "the junction frames are not in the order of B's loader"
    return dict(frames=frames, preds=preds, rows=rows[:n], total=len(rows), edits=edits, saved=cost, seconds=time.time() - t0, batches=nb, groups=groups, device=str(device), calib=calib)


def write_point(folder, seed, name, res):
    """the point folder (see the module docstring); point.json LAST; a complete folder is never overwritten, a folder without point.json is redone"""
    import qgraph_readout as RO
    if os.path.exists(os.path.join(folder, "point.json")):
        raise SystemExit(f"{folder} is complete already: a point folder is never overwritten")
    os.makedirs(folder, exist_ok=True)
    j, b = runs_of(seed)
    tag = point_tag(name, res["groups"])
    link = os.path.join(folder, "model.pt")
    if not os.path.lexists(link):
        os.symlink(os.path.abspath(os.path.join(ROOT, "runs", j, "model.pt")), link)
    pj, pb = os.path.join(folder, "ends_val.pkl"), os.path.join(folder, f"predfull_{tag}_s{seed}.pkl")
    PP.atomic_write(pj, pickle.dumps(res["frames"]))
    RO.dump_full(res["preds"], res["rows"], pb + ".tmp")
    os.replace(pb + ".tmp", pb)
    spec, how, draw = POINTS[name]
    meta = dict(point=name, tag=tag, id=IDS[name], seed=seed, set=spec, how=how, draw=draw, groups=res["groups"], frames=len(res["frames"]), frames_total=res["total"], partial=len(res["frames"]) != res["total"], device=res["device"], seconds=res["seconds"], torch=torch.__version__,
                edits=[dict(tap=e.tap, f=len(e.idx) / res["calib"].mean[e.tap].numel(), n=len(e.idx), idx=list(e.idx)) for e in res["edits"]], saved={k: float(v) for k, v in res["saved"].items()},
                sha256=dict(ends_val=sha256_file(pj), predfull=sha256_file(pb), model_junction=sha256_file(os.path.join(ROOT, "runs", j, "model.pt")), model_b=sha256_file(os.path.join(ROOT, "runs", b, "model.pt")), calib=sha256_file(calib_path(seed))))
    PP.atomic_write(os.path.join(folder, "point.json"), json.dumps(meta, indent=1).encode())
    return meta


# ---------------------------------------------------------------- 3. the equality of the unedited pass: arrays, then the null gate on the cards
def match_points(pa, da, pb, db, tol=0.05):
    """mutual nearest neighbours in (position, 5 x heading) within `tol` -> (ia, ib) indices. The order of the active nodes / keypoints is a top-K order where equal values tie differently on the CPU and on the GPU, and an item near its threshold (node e = 0.2, keypoint p) is in one dump
    and not in the other; the scorer reads them as sets (the links and the pair logits name them by index), so the dumps are compared as sets"""
    fa = np.c_[pa, 5.0 * da]
    fb = np.c_[pb, 5.0 * db]
    d = np.sqrt(((fa[:, None, :] - fb[None, :, :]) ** 2).sum(-1))
    ja, jb = d.argmin(1), d.argmin(0)
    ia = np.flatnonzero((jb[ja] == np.arange(len(fa))) & (d[np.arange(len(fa)), ja] < tol))
    return ia, ja[ia]


def match_nodes(a, b, tol=0.05):
    return match_points(a["pos"], a["dirv"], b["pos"], b["dirv"], tol)


def compare_dumps(ref, new):
    """B' dumps frame by frame, nodes and links as sets (`match_nodes`): -> dict(frames, nodes_matched / _only_in_ref / _only_in_new, e_max_unmatched (the existence of the nodes in one dump only: they sit at the threshold e >= 0.2), nodes (max abs per key over the matched nodes),
    sections_compared / sections_differ (the section start of a matched node = the argmax of its membership row M: matched nodes whose start is another node, or is a kept node in one dump and not in the other: near-tied M entries flip between devices),
    links_matched / _only_in_ref / _only_in_new (among the links of matched nodes; p_max_one_sided: their probability sits at the threshold p >= 0.05), links (max abs of p / a / b), long_equal). The assignment rows
    are not compared: they depend on the map layers of the loader, and the stored dump of 2026-11-08 predates the GT merge of 2026-11-11 (v9 -> v13); the scorer does not read them"""
    r = dict(frames=len(ref), nodes_matched=0, nodes_only_in_ref=0, nodes_only_in_new=0, e_max_unmatched=0.0, nodes={}, sections_compared=0, sections_differ=0, links_matched=0, links_only_in_ref=0, links_only_in_new=0, p_max_one_sided=0.0, links={}, long_equal=True)
    for x, y in zip(ref, new):
        assert x["row"] == y["row"]
        ia, ib = match_nodes(x, y)
        r["nodes_matched"] += len(ia)
        um_a, um_b = np.setdiff1d(np.arange(len(x["e"])), ia), np.setdiff1d(np.arange(len(y["e"])), ib)
        r["nodes_only_in_ref"] += len(um_a)
        r["nodes_only_in_new"] += len(um_b)
        r["e_max_unmatched"] = max([r["e_max_unmatched"], *x["e"][um_a].tolist(), *y["e"][um_b].tolist()])
        for k in ("e", "pos", "dirv", "flags", "s"):
            if x.get(k) is not None:
                r["nodes"][k] = max(r["nodes"].get(k, 0.0), float(np.abs(np.asarray(x[k])[ia].astype(float) - np.asarray(y[k])[ib].astype(float)).max()) if len(ia) else 0.0)
        mapa, mapb = -np.ones(len(x["e"]), int), -np.ones(len(y["e"]), int)
        mapa[ia] = ib
        mapb[ib] = ia
        where_a = -np.ones(int(max(x["idx"].max(initial=0), np.asarray(x["section"]).max(initial=0))) + 1, int)         # `dump_full` keeps the section start as an index of the unfiltered node list (`idx` holds the kept ones)
        where_a[x["idx"]] = np.arange(len(x["idx"]))
        where_b = -np.ones(int(max(y["idx"].max(initial=0), np.asarray(y["section"]).max(initial=0))) + 1, int)
        where_b[y["idx"]] = np.arange(len(y["idx"]))
        sa, sb = where_a[np.asarray(x["section"])[ia]], where_b[np.asarray(y["section"])[ib]]
        both = (sa >= 0) & (sb >= 0)
        r["sections_compared"] += len(ia)
        r["sections_differ"] += int((mapa[sa[both]] != sb[both]).sum() + ((sa >= 0) != (sb >= 0)).sum())
        la = {(int(s), int(d)): j for j, (s, d) in enumerate(zip(mapa[x["link_src"]], mapa[x["link_dst"]])) if s >= 0 and d >= 0}
        lb = {(int(s), int(d)): j for j, (s, d) in enumerate(zip(y["link_src"], y["link_dst"])) if mapb[s] >= 0 and mapb[d] >= 0}
        com = [(la[k], lb[k]) for k in la if k in lb]
        ja, jb = np.array([c[0] for c in com], int), np.array([c[1] for c in com], int)
        r["links_matched"] += len(com)
        one_a, one_b = [j for k, j in la.items() if k not in lb], [j for k, j in lb.items() if k not in la]
        r["links_only_in_ref"] += len(one_a)
        r["links_only_in_new"] += len(one_b)
        r["p_max_one_sided"] = max([r["p_max_one_sided"], *x["link_p"][one_a].tolist(), *y["link_p"][one_b].tolist()])
        for k, key in (("p", "link_p"), ("a", "link_a"), ("b", "link_b")):
            r["links"][k] = max(r["links"].get(k, 0.0), float(np.abs(x[key][ja] - y[key][jb]).max()) if len(com) else 0.0)
        r["long_equal"] &= bool(np.array_equal(x["link_long"][ja], y["link_long"][jb]))
    return r


def compare_frames(ref, new):
    """junction dumps frame by frame, the valid keypoints of each kind as sets (`match_points`): -> dict(frames, keypoints_matched / _only_in_ref / _only_in_new, frames_with_unmatched, p_max_unmatched (the probability of the keypoints in one dump only), max_abs per key over the
    matched keypoints and over the logit / scale entries of the matched pairs of the frames without an unmatched keypoint, allowed_equal, att_moved (matched keypoints whose attached end moved by more than 0.05 m: the attachment is a softmax over the nodes in reach, a near tie flips the node it attaches to), att_moved_max_m)"""
    r = dict(frames=len(ref), keypoints_matched=0, keypoints_only_in_ref=0, keypoints_only_in_new=0, frames_with_unmatched=0, p_max_unmatched=0.0, max_abs={}, allowed_equal=True, att_moved=0, att_moved_max_m=0.0)

    def put(k, v):
        r["max_abs"][k] = max(r["max_abs"].get(k, 0.0), float(v))
    for a, b in zip(ref, new):
        assert a["token"] == b["token"] and a["row"] == b["row"]
        K = a["valid"].shape[1]
        m, lone = [], 0
        for kind in (0, 1):
            ja, jb = np.flatnonzero(a["valid"][kind]), np.flatnonzero(b["valid"][kind])
            ia, ib = match_points(a["kp_pos"][kind * K + ja], a["kp_dir"][kind * K + ja], b["kp_pos"][kind * K + jb], b["kp_dir"][kind * K + jb])
            m.append((ja[ia], jb[ib]))
            r["keypoints_matched"] += len(ia)
            ua, ub = np.setdiff1d(np.arange(len(ja)), ia), np.setdiff1d(np.arange(len(jb)), ib)
            r["keypoints_only_in_ref"] += len(ua)
            r["keypoints_only_in_new"] += len(ub)
            lone += len(ua) + len(ub)
            r["p_max_unmatched"] = max([r["p_max_unmatched"], *a["kp_p"][kind * K + ja[ua]].tolist(), *b["kp_p"][kind * K + jb[ub]].tolist()])
        ra, rb = np.r_[m[0][0], K + m[1][0]], np.r_[m[0][1], K + m[1][1]]
        for k in ("kp_pos", "kp_dir", "kp_p", "att_pos", "att_dir", "att_p", "logsig"):
            if isinstance(a.get(k), np.ndarray) and len(ra):
                put(k, np.abs(a[k][ra] - b[k][rb]).max())
        if isinstance(a.get("att_pos"), np.ndarray) and len(ra):
            d = np.linalg.norm(a["att_pos"][ra] - b["att_pos"][rb], axis=1)
            r["att_moved"] += int((d > 0.05).sum())
            r["att_moved_max_m"] = max(r["att_moved_max_m"], float(d.max()))
        (ea, eb), (xa, xb) = m
        r["frames_with_unmatched"] += int(lone > 0)
        if len(ea) and len(xa) and not lone:                                                                    # a keypoint in one dump only shifts the rank features of the pair head (group sizes, order): the logits are compared in the frames without one
            for k in ("logit", "scale"):
                if isinstance(a.get(k), np.ndarray):
                    put(k, np.abs(a[k][np.ix_(ea, xa)] - b[k][np.ix_(eb, xb)]).max())
            r["allowed_equal"] &= bool(np.array_equal(a["allowed"][np.ix_(ea, xa)], b["allowed"][np.ix_(eb, xb)]))
    return r


def check(seed, n=24, device="cpu", out=None):
    """the EMPTY set through the pass against the stored B dump (`temp/e276`) and the stored E345 null of the junction system (`runs/e345_p000_e324_s_s<S>`): the build's equality as arrays; nothing is written but the scratch pickle under `out`"""
    res = run_pass(seed, "null", device, max_frames=n)
    ref_b = pickle.load(open(os.path.join(ROOT, "temp", "e276", f"predfull_e273_p_s{seed}.pkl"), "rb"))[:n]
    import qgraph_readout as RO
    out = out or os.path.join(ROOT, "temp", "e352", f"check_s{seed}")
    os.makedirs(out, exist_ok=True)
    RO.dump_full(res["preds"], res["rows"], os.path.join(out, f"predfull_null_s{seed}.pkl"))
    new_b = pickle.load(open(os.path.join(out, f"predfull_null_s{seed}.pkl"), "rb"))
    ref_j = pickle.load(open(os.path.join(ROOT, "runs", f"e345_p000_e324_s_s{seed}", "ends_val.pkl"), "rb"))[:n]
    return dict(b=compare_dumps(ref_b, new_b), junction=compare_frames(ref_j, res["frames"]), seconds_per_frame=res["seconds"] / n, device=res["device"])


def tolerance(key, stored):
    """the gate's bound of one number: topo F1 2e-4, every other number max(GATE_ABS, GATE_REL * |stored|)"""
    return GATE_TOPO_F1 if key == "topo_F1" else max(GATE_ABS, GATE_REL * abs(stored))


def gate(seed, null_run=None, runs_dir=None):
    """the NULL gate of the E352 rule: the cards of the null (`runs/e352_null_s<S>/ends_card_null_*`: B' of the empty set on the CPU, scored by the stored scorer) against the stored cards of E324 s (`runs/e324_s_s<S>`, B's stored dump), every number of `base` (B's plain graph) and `head` (the junction system
    on it) of the three read-outs against its tolerance. -> (rows (seed, readout, which, key, stored, new, difference, tolerance, ok), all ok)"""
    runs_dir = runs_dir or os.path.join(ROOT, "runs")
    null_run = null_run or point_run(seed, "null")
    rows = []
    for k in READOUTS:
        a = json.load(open(os.path.join(runs_dir, runs_of(seed)[0], STORED[k])))["numbers"]
        b = json.load(open(os.path.join(runs_dir, null_run, card_files("null")[k])))["numbers"]
        for which in ("base", "head"):
            for key in KEYS:
                d = b[which][key] - a[which][key]
                tol = tolerance(key, a[which][key])
                rows.append((seed, k, which, key, a[which][key], b[which][key], d, tol, abs(d) <= tol))
    return rows, all(r[-1] for r in rows)


def gate_md(rows):
    """the gate rows as markdown: per read-out and which, the largest |difference| / tolerance and the topo F1 difference"""
    out = ["| seed | read-out | graph | topo F1 stored | topo F1 null | diff | worst number | diff | tolerance | ok |", "|---|---|---|---|---|---|---|---|---|---|"]
    for seed in sorted({r[0] for r in rows}):
        for k in READOUTS:
            for which in ("base", "head"):
                sel = [r for r in rows if r[0] == seed and r[1] == k and r[2] == which]
                if not sel:
                    continue
                t = next(r for r in sel if r[3] == "topo_F1")
                w = max(sel, key=lambda r: abs(r[6]) / r[7])
                out.append(f"| {seed} | {k} | {which} | {t[4]:.5f} | {t[5]:.5f} | {t[6]:+.1e} | {w[3]} | {w[6]:+.1e} | {w[7]:.1e} | {'yes' if all(r[-1] for r in sel) else 'NO'} |")
    return "\n".join(out)


# ---------------------------------------------------------------- 4. the step-0 table, the scoring lines, the plan
def score_lines(seed, name, groups=False):
    """the scoring lines of one point (`$R` = the problem folder): the three judged read-outs of the junction system on B' and the unsnapped one; B's plain graph is `numbers.base` of any card. The default read-out of B's delivered graph is `close:1.5@0.5` (`qends_eval.DEFAULT_SPEC`)"""
    run, tag = point_run(seed, name, groups), point_tag(name, groups)
    head = f"bash tools/env/_py14.sh jends_eval.py score --run {run} --seed {seed} --full $R/runs/{run}/predfull_{tag}_s{seed}.pkl --base $R/runs/e273_p_s{seed}/model.pt --spec close:1.5@0.5 --tag {tag}"
    return {"snap": f"{head} --snap --thr 0.1", "primary": f"{head} --snap --snap-nogeom --attach --thr 0.1", "attach": f"{head} --attach --thr 0.1", "unsnapped": f"{head} --thr 0.1"}


def card_files(tag):
    """the card files these lines write (in the point folder), by the tag"""
    return {"snap": f"ends_card_{tag}_snap_thr0.1.json", "primary": f"ends_card_{tag}_snap_nogeom_attach_thr0.1.json", "attach": f"ends_card_{tag}_attach_thr0.1.json", "unsnapped": f"ends_card_{tag}_thr0.1.json"}


def read_point(seed, name, groups, runs_dir, null_fmt="e352_null_s{seed}"):
    """-> {readout: numbers dict of the card} of a point (None for a card that is not there)"""
    run = null_fmt.format(seed=seed) if name == "null" else point_run(seed, name, groups)
    files = card_files(point_tag(name, groups))
    out = {}
    for k in READOUTS:
        p = os.path.join(runs_dir, run, files[k])
        out[k] = json.load(open(p))["numbers"] if os.path.exists(p) else None
    return out


def values(cards, key="topo_F1"):
    """-> [B plain, snap, primary, attach] of a point's cards (B plain from the first card that exists); None where a card is missing"""
    have = [c for c in cards.values() if c is not None]
    return [have[0]["base"][key] if have else None] + [cards[k]["head"][key] if cards[k] else None for k in READOUTS]


WINDOWS = {"t1_b": (-0.012, 0.001), "t2_b": (-0.025, -0.002), "rand_b": (0.0, 0.02), "t1_snap": (-0.010, 0.002), "t2_snap": (-0.018, -0.002)}     # the hit windows of the step-0 predictions (E352 card; the T2 junction window is the coder's proposal)


def table_data(seeds=(0, 1), runs_dir=None, null_fmt="e352_null_s{seed}", key="topo_F1"):
    """-> dict seed -> dict(null, rows {(variant, point): [B plain, snap, primary, attach]}, random_mean {(variant, set): [...]}, checks) from the cards of the point folders; every Delta is value - null (negative = worse), 'ranked minus random' is the ranked value minus the mean of the random draws"""
    runs_dir = runs_dir or os.path.join(ROOT, "runs")
    out = {}
    for seed in seeds:
        null = values(read_point(seed, "null", False, runs_dir, null_fmt), key)
        d = dict(null=null, rows={}, random_mean={}, checks={})
        for g, v in ((True, "groups"), (False, "channels")):
            for name in [n for n in POINTS if n != "null"]:
                if any(c is not None for c in read_point(seed, name, g, runs_dir).values()):
                    d["rows"][(v, name)] = values(read_point(seed, name, g, runs_dir), key)
            for s, draws in RANDOM_OF.items():
                got = [d["rows"][(v, n)] for n in draws if (v, n) in d["rows"]]
                if len(got) == len(draws) and all(None not in x for x in got):
                    d["random_mean"][(v, s)] = [float(np.mean([x[i] for x in got])) for i in range(4)]
        out[seed] = d
    return out


def checks(data):
    """the lines of the rule on the groups variant (the judged one), per seed: the predictions with their hit windows and the kill. -> list of strings"""
    lines = []
    for seed, d in data.items():
        n = d["null"]
        if None in n:
            lines.append(f"seed {seed}: the null cards are missing")
            continue
        for s in ("t1", "t2"):
            r, m = d["rows"].get(("groups", s)), d["random_mean"].get(("groups", s))
            if r is None or None in r:
                continue
            db = r[0] - n[0]
            lines.append(f"seed {seed} {s}: B plain {db:+.4f} (window {WINDOWS[s + '_b'][0]:+.3f} .. {WINDOWS[s + '_b'][1]:+.3f}: {'HIT' if WINDOWS[s + '_b'][0] <= db <= WINDOWS[s + '_b'][1] else 'MISS'}), junction snapped {r[1] - n[1]:+.4f} (window {WINDOWS[s + '_snap'][0]:+.3f} .. {WINDOWS[s + '_snap'][1]:+.3f}: {'HIT' if WINDOWS[s + '_snap'][0] <= r[1] - n[1] <= WINDOWS[s + '_snap'][1] else 'MISS'})")
            if m is not None:
                gap = r[0] - m[0]
                lines.append(f"seed {seed} {s}: ranked minus mean of the 3 random draws on B plain {gap:+.4f} (prediction at least +0.003, window {WINDOWS['rand_b'][0]:+.3f} .. {WINDOWS['rand_b'][1]:+.3f}: {'HIT' if WINDOWS['rand_b'][0] <= gap <= WINDOWS['rand_b'][1] and gap >= 0.003 else 'MISS'})")
                if s == "t1":
                    kill = (r[0] - n[0]) < -0.03 and (m[0] - n[0]) >= (r[0] - n[0])
                    lines.append(f"seed {seed} KILL (T1 loses more than 0.03 on B plain and the random mean loses no more): {'YES: stop, report' if kill else 'no'}")
    return lines


def table_md(data, key="topo_F1"):
    """the step-0 table as markdown (per seed: value and Delta against the null of B plain and of the three junction read-outs, ranked rows, the random draws and their mean, ranked minus random) and the rule's lines"""
    out = []
    for seed, d in data.items():
        n = d["null"]
        if None in n:
            out += [f"### seed {seed}: the null cards are missing", ""]
            continue
        out += [f"### seed {seed}, {key} (Delta = value - null: negative = worse; 'ranked - random' = ranked value - mean of the draws: positive = ranked better)", "", "| variant | point | B plain | Delta | snap | Delta | primary | Delta | attach | Delta |", "|---|---|---|---|---|---|---|---|---|---|"]

        def row(v, name, vals, delta=True):
            cells = []
            for x, b in zip(vals, n):
                cells += ["-", "-"] if x is None else [f"{x:.4f}", f"{x - b:+.4f}" if delta else "-"]
            return f"| {v} | {name} | " + " | ".join(cells) + " |"
        out.append(row("-", "null", n, delta=False))
        for v in ("groups", "channels"):
            for s in ("t1", "t2"):
                for name in (s,) + RANDOM_OF[s]:
                    if (v, name) in d["rows"]:
                        out.append(row(v, name, d["rows"][(v, name)]))
                if (v, s) in d["random_mean"]:
                    out.append(row(v, f"{s} random mean", d["random_mean"][(v, s)]))
                    if (v, s) in d["random_mean"] and (v, s) in d["rows"] and None not in d["rows"][(v, s)]:
                        diff = [a - b for a, b in zip(d["rows"][(v, s)], d["random_mean"][(v, s)])]
                        out.append(f"| {v} | {s} ranked - random | " + " | ".join(f"{x:+.4f} | -" for x in diff) + " |")
        out.append("")
    return "\n".join(out + ["### the rule (groups variant, the judged one)", ""] + ["- " + line for line in checks(data)])


def plan(out):
    """the point-set file (the format of `jprune_probe.py --score-points`: [{id, name, set}], plus how / draw for this tool; E345's own sweep would dump a random entry as the RANKED set: this file is for `jprune_b.py` only) and the table of channel counts and saved cost"""
    jm, _ = JEV.load_model(runs_of(0)[0], "cpu")
    calib = PP.load_calib(calib_path(0), "cpu")
    cost = PC.Cost(copy.deepcopy(jm))
    rows, entries = [], []
    for name, (spec, how, draw) in POINTS.items():
        edits = point_edits(jm, calib, name)
        sv = cost.saved(edits)
        rows.append(f"{name:5s} {how:6s} draw {draw!s:4s} " + " ".join(f"{e.tap.split('.')[1]}:{len(e.idx)}" for e in edits) + f"  saved {sv['macs'] / 1e9:.4f} GMACs (pyramid {sv['macs_pyr'] / 1e9:.4f}, maps {sv['macs_maps'] / 1e9:.4f}), {sv['params']} params")
        entries.append(dict(id=IDS[name], name=name, set=spec, how=how, draw=draw))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(entries, open(out, "w"), indent=1)
    return "\n".join(rows + [f"of {cost.all['macs'] / 1e9:.3f} GMACs / {cost.all['params'] / 1e6:.3f} M parameters of the cost table (pyramid + maps head + pair head; B's node head and the attachment are not counted: the node head reads d0 and d1 too)"])


# ---------------------------------------------------------------- selftest and CLI
def selftest():
    """no data, CPU: the point table against the E345 ladder on a random-init model, the nested random draws, the group variant, the shim's alias"""
    import jends_train as JT
    import qgraph as QG
    torch.manual_seed(0)
    m = JT.JEndModel(dict(d=64, ctx="dil"), order=True, evidence=True, scales=True).eval()
    taps = PP.tap_table(m)
    units = {u.name: u for u in PP.build_units(m)}
    for name, (spec, how, draw) in POINTS.items():
        assert all(u in units and f in units[u].fracs for u, f in spec.items()), name
    x = torch.randn(2, 39, 200, 200)
    calib = PP.calibrate(m, [dict(x=x)], "cpu", taps)
    for name in POINTS:
        e = point_edits(m, calib, name)
        eg = point_edits(m, calib, name, groups=True)
        assert [len(a.idx) for a in e] == [len(a.idx) for a in eg], "the group variant removes the same counts"
        for a in eg:
            gs = PC.sub(m, a.tap)[-2].num_channels // 8
            assert len(a.idx) % gs == 0 and all(set(range(c // gs * gs, c // gs * gs + gs)) <= set(a.idx) for c in a.idx), "whole groups"
    r, r5 = point_edits(m, calib, "r1d0"), point_edits(m, calib, "r1d1")
    assert [a.idx for a in r] != [a.idx for a in r5] and [a.idx for a in point_edits(m, calib, "r1d0")] == [a.idx for a in r], "a draw is fixed, the draws differ"
    assert len({rng_for(u, 0, False).randint(1 << 30) for u in T2}) == len(T2) and rng_for("pyr.d2", 0, True).randint(1 << 30) != rng_for("pyr.d2", 0, False).randint(1 << 30), "independent seeds per unit and variant"
    qg = QG.QGraph().eval()
    sh, w = PP.ablate(Shim(qg), point_edits(m, calib, "t1"), taps)
    assert sh.qg.pyr is sh.jmodel.pyr and sh.qg is not qg
    seen = []
    sh.qg.pyr.e2.register_forward_hook(lambda _m, _a, o: seen.append(o.detach().clone()))
    with torch.no_grad():
        sh.qg.pyr(x)
    w.check(1, calib)
    idx = list(next(e for e in w.edit.values() if e.tap == "pyr.e2").idx)
    assert len(idx) == 72 and float(seen[0][:, idx].std(dim=(0, 2, 3)).max()) < 1e-6, "the edited channels of B's pyramid are constant"
    print(f"jprune_b selftest ok ({len(POINTS)} points)")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("plan")
    s.add_argument("--out", default="temp/e352/points.json")
    s = sp.add_parser("dump")
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--point", required=True, choices=sorted(POINTS))
    s.add_argument("--device", default="cpu")
    s.add_argument("--groups", action="store_true", help="whole GroupNorm groups (the judged set); without it E345's per-channel ranking (the anchor)")
    s.add_argument("--workers", type=int, default=0)
    s.add_argument("--threads", type=int, default=2, help="torch threads of the pass (the section solve always runs on one)")
    s.add_argument("--max", type=int, default=0, help="only the first N frames (a test of the pipeline; needs --out outside runs/)")
    s.add_argument("--out", default=None)
    s = sp.add_parser("check")
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--max", type=int, default=24)
    s.add_argument("--device", default="cpu")
    s.add_argument("--threads", type=int, default=2)
    s = sp.add_parser("gate")
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--null-run", default=None)
    s = sp.add_parser("table")
    s.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    s.add_argument("--null-fmt", default="e352_null_s{seed}", help="the run folder of the null (e352diag_null_s{seed} for the hand-made diagnostic)")
    s.add_argument("--key", default="topo_F1")
    s.add_argument("--out", default=None)
    s = sp.add_parser("commands")
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--point", default=None, choices=sorted(POINTS))
    s.add_argument("--groups", action="store_true")
    s.add_argument("--readout", default=None, choices=("snap", "primary", "attach", "unsnapped"), help="print only the line of this read-out (no header)")
    sp.add_parser("selftest")
    a = ap.parse_args()
    torch.set_num_threads(getattr(a, "threads", 2))
    if a.cmd == "selftest":
        selftest()
    elif a.cmd == "plan":
        print(plan(PP.rootpath(a.out)))
    elif a.cmd == "commands":
        for name in ([a.point] if a.point else POINTS):
            for k, line in score_lines(a.seed, name, a.groups).items():
                if a.readout is None:
                    print(f"# {name} {k}\n{line}")
                elif a.readout == k:
                    print(line)
    elif a.cmd == "check":
        print(json.dumps(check(a.seed, a.max, a.device), indent=1, default=float))
    elif a.cmd == "gate":
        rows, ok = gate(a.seed, a.null_run)
        print(gate_md(rows))
        print(f"null gate seed {a.seed}: {'PASS' if ok else 'FAIL: ' + ', '.join(f'{r[1]}/{r[2]}/{r[3]} {r[6]:+.2e} > {r[7]:.1e}' for r in rows if not r[-1])}")
        sys.exit(0 if ok else 1)
    elif a.cmd == "table":
        text = table_md(table_data(tuple(a.seeds), null_fmt=a.null_fmt, key=a.key), a.key)
        print(text)
        if a.out:
            open(PP.rootpath(a.out), "w").write(text + "\n")
    else:
        check_out(a.max, a.out)
        folder = PP.rootpath(a.out) if a.out else os.path.join(ROOT, "runs", point_run(a.seed, a.point, a.groups))
        if os.path.exists(os.path.join(folder, "point.json")):
            sys.exit(f"{folder} is complete already: a point folder is never overwritten")
        res = run_pass(a.seed, a.point, a.device, a.max, a.groups, a.workers)
        meta = write_point(folder, a.seed, a.point, res)
        print(f"{a.point} seed {a.seed}: {meta['frames']} frames on {meta['device']} in {meta['seconds']:.0f} s -> {folder}; saved {meta['saved']['macs'] / 1e9:.4f} GMACs")


if __name__ == "__main__":
    main()
