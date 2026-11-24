"""Step 0 of E352 (no training): what do the E345 hook edits of B's pyramid trunk do to B's OWN graph, and to the junction system on B's re-dumped graph? The probe applies one point set {unit: f} (E345 semantics: the lowest-importance share f of the
channels of the unit's tensor is replaced by its calibration mean; ranking and means from `temp/e345/e324_s_s<S>/calib.pt` through `jprune_probe`) to the pyramid of the junction system (`runs/e324_s_s<S>`) AND to the pyramid of B (`runs/e273_p_s<S>`, the QGraph that makes the
delivered graph), in ONE pass over the 873 VAL frames, and writes the folder `runs/e352_<point>_s<S>/`:  model.pt (a link to the unmodified E324 s checkpoint), ends_val.pkl (the junction dump on the edited pyramid, the format of `jends_eval.py dump`), predfull_<point>_s<S>.pkl
(B' = B's full prediction on the edited pyramid, the format of `temp/e276/predfull_e273_p_s<S>.pkl`: `qpredict.predict` + `qgraph_readout.dump_full`, exact map, as `qgraph_run_eval.score --dump-full`), point.json (the set, the channels, the saved cost, the sha256 of both pickles,
the device, the seconds; written LAST: a folder without it is incomplete). Nothing in a stored run is touched; every file goes through a temporary file and `os.replace`; a complete folder is never overwritten; a partial dump (--max) cannot land in `runs/`.

    wsl bash tools/env/_py14.sh jprune_b.py plan     [--out temp/e352/points.json]            # CPU, no data: the 9 point sets (null, T1, T2, random draws), channel counts, saved GMACs; the point-set file
    wsl bash tools/env/_py14.sh jprune_b.py dump     --seed 0 --point t1 [--device cuda] [--groups] [--workers N] [--max N --out DIR]    # one point of one seed -> runs/e352_t1_s0/
    wsl bash tools/env/_py14.sh jprune_b.py check    --seed 0 [--max 24]                       # the EMPTY set against the stored B dump (temp/e276) and the stored E345 junction null: the build's equality
    wsl bash tools/env/_py14.sh jprune_b.py commands --seed 0 [--point t1]                     # the scoring lines (jends_eval.py score --full --base --tag) of the read-outs snap / primary / attach
    wsl bash tools/env/_py14.sh jprune_b.py selftest

Sets (E352 card, step 0): T1 = {pyr.e2: 0.5, pyr.d2: 0.25, pyr.e3: 0.25}, T2 = T1 + {pyr.d0: 0.25, pyr.e1: 0.25, pyr.d1: 0.25}; fractions of the E345 ladder. Controls: for each set 3 RANDOM draws of the same channel counts per unit (`RandomState(1000 * draw + sum(ord(c) of the unit name))`
per unit, as the E345 random control: a draw is nested, the 0.25 set is inside the 0.5 set of the same unit); `--groups` removes WHOLE GroupNorm groups (ranked by the summed channel importance; random: random groups) instead of single channels: the channel sets the physical
narrowing of E352 will remove (same counts). Points: null, t1, t2, r1d0..r1d2 (draws of T1), r2d0..r2d2 (draws of T2). The B plain graph is `numbers.base` of any of the cards; the junction numbers use B' as the delivered graph (`--full / --base / --tag` of `qends_eval.cmd_score`, default read-out
`close:1.5@0.5` = the hard close-node step of B's delivery). A pass runs BOTH models on the same batch: a hook that does not fire once per forward or changes nothing stops the dump (E345's `Watch`).
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
FRAME_KEYS = ("kp_pos", "kp_dir", "kp_p", "valid", "logit", "allowed", "scale", "logsig", "att_pos", "att_dir", "att_p")


def runs_of(seed):
    return f"e324_s_s{seed}", f"e273_p_s{seed}"


def sha256_file(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


# ---------------------------------------------------------------- 1. the point sets and their edits
def group_edits(model, unit, f, calib, how="rank", rng=None):
    """the edits of a unit at share f that remove WHOLE GroupNorm groups (the block is conv GroupNorm ReLU conv GroupNorm ReLU: the GroupNorm in front of the last ReLU); ranked by the summed importance of the group's channels, or random groups"""
    out = []
    for tap in unit.taps:
        gn = PC.sub(model, tap)[-2]
        assert isinstance(gn, nn.GroupNorm), tap
        G, gs = gn.num_groups, gn.num_channels // gn.num_groups
        n = int(f * G + 0.5)
        gone = rng.permutation(G)[:n].tolist() if how == "random" else torch.argsort(calib.s[tap].double().view(G, gs).sum(1), stable=True)[:n].tolist()
        idx = sorted(c for g in gone for c in range(g * gs, (g + 1) * gs))
        out.append(Edit("chan", tap=tap, idx=tuple(idx), val=calib.mean[tap][idx]))
    return out


def point_edits(model, calib, name, groups=False):
    """-> the edits of a point: ranked (E345 `edits_for`) or the random draw of the point, per unit with its own RandomState"""
    spec, how, draw = POINTS[name]
    units = {u.name: u for u in PP.build_units(model)}
    out = []
    for unit, f in spec.items():
        u = units[unit]
        assert u.kind == "chan" and unit.startswith("pyr.") and f in u.fracs, f"{unit}@{f} is not a pyramid point of the E345 ladder"
        rng = np.random.RandomState(1000 * draw + sum(map(ord, unit))) if how == "random" else None
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


# ---------------------------------------------------------------- 2. one pass: B' and the junction dump on the edited pyramid
def run_pass(seed, name, device="cpu", max_frames=0, groups=False, workers=0, log=print):
    """-> dict(frames (the junction dump), preds + rows (B'), edits, saved cost, seconds, hooks fired): both models see the same edited pyramid and the same batches (B's loader: exact map, batch 4, strict, as `jtest_eval.dump_b`; the junction model reads only x)"""
    import lane_gt_data as LD
    import qpredict as QP
    PP.wait_for_gpu(f"jprune_b {name}_s{seed}", device)
    jm, args, qg = load_models(seed, device)
    calib = PP.load_calib(PP.rootpath(f"temp/e345/e324_s_s{seed}/calib.pt"), "cpu")
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
    return dict(frames=frames, preds=preds, rows=rows[:n], edits=edits, saved=cost, seconds=time.time() - t0, batches=nb, groups=groups, device=str(device), calib=calib)


def write_point(folder, seed, name, res):
    """the point folder (see the module docstring); point.json LAST; a complete folder is never overwritten, a folder without point.json is redone"""
    import qgraph_readout as RO
    if os.path.exists(os.path.join(folder, "point.json")):
        raise SystemExit(f"{folder} is complete already: a point folder is never overwritten")
    os.makedirs(folder, exist_ok=True)
    j, b = runs_of(seed)
    link = os.path.join(folder, "model.pt")
    if not os.path.lexists(link):
        os.symlink(os.path.abspath(os.path.join(ROOT, "runs", j, "model.pt")), link)
    pj, pb = os.path.join(folder, "ends_val.pkl"), os.path.join(folder, f"predfull_{name}_s{seed}.pkl")
    PP.atomic_write(pj, pickle.dumps(res["frames"]))
    RO.dump_full(res["preds"], res["rows"], pb + ".tmp")
    os.replace(pb + ".tmp", pb)
    spec, how, draw = POINTS[name]
    meta = dict(point=name, id=IDS[name], seed=seed, set=spec, how=how, draw=draw, groups=res["groups"], frames=len(res["frames"]), device=res["device"], seconds=res["seconds"], torch=torch.__version__,
                edits=[dict(tap=e.tap, f=len(e.idx) / res["calib"].mean[e.tap].numel(), n=len(e.idx), idx=list(e.idx)) for e in res["edits"]], saved={k: float(v) for k, v in res["saved"].items()},
                sha256=dict(ends_val=sha256_file(pj), predfull=sha256_file(pb), model_junction=sha256_file(os.path.join(ROOT, "runs", j, "model.pt")), model_b=sha256_file(os.path.join(ROOT, "runs", b, "model.pt")),
                            calib=sha256_file(PP.rootpath(f"temp/e345/e324_s_s{seed}/calib.pt"))))
    PP.atomic_write(os.path.join(folder, "point.json"), json.dumps(meta, indent=1).encode())
    return meta


# ---------------------------------------------------------------- 3. the equality of the unedited pass, the cost table, the scoring lines
def compare_dumps(ref, new):
    """B' dumps frame by frame: -> dict(frames, shape_mismatch (frames whose node or link arrays differ in length), max_abs (per float key, over the frames of equal shape), equal_int (the integer keys are equal))"""
    mx, bad, ints = {}, [], True
    for i, (a, b) in enumerate(zip(ref, new)):
        assert a["row"] == b["row"]
        if len(a["e"]) != len(b["e"]) or len(a["link_p"]) != len(b["link_p"]):
            bad.append(i)
            continue
        for k, v in a.items():
            if isinstance(v, np.ndarray) and k in b:
                if v.dtype.kind in "iub":
                    ints &= bool(np.array_equal(v, b[k]))
                else:
                    mx[k] = max(mx.get(k, 0.0), float(np.abs(v.astype(float) - b[k].astype(float)).max()) if v.size else 0.0)
    return dict(frames=len(ref), shape_mismatch=bad, max_abs=mx, equal_int=ints)


def compare_frames(ref, new):
    """junction dumps frame by frame: -> dict(max_abs per float key, equal_valid, equal_kp_cells)"""
    mx, valid = {}, True
    for a, b in zip(ref, new):
        assert a["token"] == b["token"] and a["row"] == b["row"]
        valid &= bool(np.array_equal(a["valid"], b["valid"]) and np.array_equal(a["allowed"], b["allowed"]))
        for k in FRAME_KEYS:
            if isinstance(a.get(k), np.ndarray) and k not in ("valid", "allowed"):
                mx[k] = max(mx.get(k, 0.0), float(np.abs(a[k].astype(float) - b[k].astype(float)).max()))
    return dict(max_abs=mx, equal_valid_allowed=valid)


def check(seed, n=24, device="cpu", out=None):
    """the EMPTY set through the pass against the stored B dump (`temp/e276`) and the stored E345 null of the junction system (`runs/e345_p000_e324_s_s<S>`): the build's equality; nothing is written but the scratch pickle under `out`"""
    res = run_pass(seed, "null", device, max_frames=n)
    ref_b = pickle.load(open(os.path.join(ROOT, "temp", "e276", f"predfull_e273_p_s{seed}.pkl"), "rb"))[:n]
    import qgraph_readout as RO
    out = out or os.path.join(ROOT, "temp", "e352", f"check_s{seed}")
    os.makedirs(out, exist_ok=True)
    RO.dump_full(res["preds"], res["rows"], os.path.join(out, f"predfull_null_s{seed}.pkl"))
    new_b = pickle.load(open(os.path.join(out, f"predfull_null_s{seed}.pkl"), "rb"))
    ref_j = pickle.load(open(os.path.join(ROOT, "runs", f"e345_p000_e324_s_s{seed}", "ends_val.pkl"), "rb"))[:n]
    return dict(b=compare_dumps(ref_b, new_b), junction=compare_frames(ref_j, res["frames"]), seconds_per_frame=res["seconds"] / n, device=res["device"])


def score_lines(seed, name):
    """the scoring lines of one point (`$R` = the problem folder): the three judged read-outs of the junction system on B' and the unsnapped one; B's plain graph is `numbers.base` of any card. The default read-out of B's delivered graph is `close:1.5@0.5` (`qends_eval.DEFAULT_SPEC`)"""
    run = f"e352_{name}_s{seed}"
    head = f"bash tools/env/_py14.sh jends_eval.py score --run {run} --seed {seed} --full $R/runs/{run}/predfull_{name}_s{seed}.pkl --base $R/runs/e273_p_s{seed}/model.pt --spec close:1.5@0.5 --tag {name}"
    return {"snap": f"{head} --snap --thr 0.1", "primary": f"{head} --snap --snap-nogeom --attach --thr 0.1", "attach": f"{head} --attach --thr 0.1", "unsnapped": f"{head} --thr 0.1"}


def card_files(name):
    """the card files these lines write (in runs/e352_<name>_s<S>/)"""
    return {"snap": f"ends_card_{name}_snap_thr0.1.json", "primary": f"ends_card_{name}_snap_nogeom_attach_thr0.1.json", "attach": f"ends_card_{name}_attach_thr0.1.json", "unsnapped": f"ends_card_{name}_thr0.1.json"}


def plan(out):
    """the point-set file (the format of `jprune_probe.py --score-points`: [{id, name, set}], plus how / draw / groups for this tool; E345's own sweep would dump a random entry as the RANKED set: this file is for `jprune_b.py` only) and the table of channel counts and saved cost"""
    jm, _ = JEV.load_model(runs_of(0)[0], "cpu")
    calib = PP.load_calib(PP.rootpath("temp/e345/e324_s_s0/calib.pt"), "cpu")
    cost = PC.Cost(copy.deepcopy(jm))
    rows, entries = [], []
    for name, (spec, how, draw) in POINTS.items():
        edits = point_edits(jm, calib, name)
        sv = cost.saved(edits)
        rows.append(f"{name:5s} {how:6s} draw {draw!s:4s} " + " ".join(f"{e.tap.split('.')[1]}:{len(e.idx)}" for e in edits) + f"  saved {sv['macs'] / 1e9:.4f} GMACs (pyramid {sv['macs_pyr'] / 1e9:.4f}, maps {sv['macs_maps'] / 1e9:.4f}), {sv['params']} params")
        entries.append(dict(id=IDS[name], name=name, set=spec, how=how, draw=draw))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(entries, open(out, "w"), indent=1)
    return "\n".join(rows + [f"of {cost.all['macs'] / 1e9:.3f} GMACs / {cost.all['params'] / 1e6:.3f} M parameters of the cost table (the node head and the attachment are not counted)"])


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
    s.add_argument("--groups", action="store_true")
    s.add_argument("--workers", type=int, default=0)
    s.add_argument("--max", type=int, default=0, help="only the first N frames (a test of the pipeline; needs --out, never lands in runs/)")
    s.add_argument("--out", default=None)
    s = sp.add_parser("check")
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--max", type=int, default=24)
    s.add_argument("--device", default="cpu")
    s = sp.add_parser("commands")
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--point", default=None, choices=sorted(POINTS))
    sp.add_parser("selftest")
    a = ap.parse_args()
    torch.set_num_threads(2)
    if a.cmd == "selftest":
        selftest()
    elif a.cmd == "plan":
        print(plan(PP.rootpath(a.out)))
    elif a.cmd == "commands":
        for name in ([a.point] if a.point else POINTS):
            for k, line in score_lines(a.seed, name).items():
                print(f"# {name} {k}\n{line}")
    elif a.cmd == "check":
        r = check(a.seed, a.max, a.device)
        print(json.dumps(r, indent=1, default=float))
    else:
        assert not a.max or a.out, "--max writes a partial dump: give --out (a folder outside runs/)"
        folder = PP.rootpath(a.out) if a.out else os.path.join(ROOT, "runs", f"e352_{a.point}_s{a.seed}")
        if os.path.exists(os.path.join(folder, "point.json")):
            sys.exit(f"{folder} is complete already: a point folder is never overwritten")
        res = run_pass(a.seed, a.point, a.device, a.max, a.groups, a.workers)
        meta = write_point(folder, a.seed, a.point, res)
        print(f"{a.point} seed {a.seed}: {meta['frames']} frames on {meta['device']} in {meta['seconds']:.0f} s -> {folder}; saved {meta['saved']['macs'] / 1e9:.4f} GMACs")


if __name__ == "__main__":
    main()
