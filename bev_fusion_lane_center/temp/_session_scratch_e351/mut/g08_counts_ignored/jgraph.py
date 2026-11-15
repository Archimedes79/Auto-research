"""E351 lever g1: the front of the junction forward (pyramid + maps head + B's node head) as ONE CUDA graph, the pair head eager on its static outputs; the outputs must be bit-identical to the eager forward. Inference path only (batch 1,
bf16 autocast, no_grad); the model is a loaded `jends_train.JEndModel` and is wrapped from outside (no frozen file is edited). The timing of the two arms is `jtiming.py graph` (it uses `GraphedModel` and `eager_forward` of this file).

    wsl bash tools/env/_py14.sh jgraph.py check --run e312_p_s0 [--dump] [--max N] [--cpu] [--runs-dir DIR] [--tag T]   # GPU: all 873 VAL frames through the eager forward twice (the NULL control) and the graph forward; torch.equal of EVERY tensor
                                                                                                                        # of the returned trees; --dump also writes the two dumps runs/e351_eager_<run>/ and runs/e351_g1_<run>/ (new directories, never an overwrite)
    wsl bash tools/env/_py14.sh jgraph.py commands --run e312_p_s0 [--tag e351]                                       # CPU: the exact `jends_eval.py score` lines for the two dumps + a dry check that `score` would find everything
    wsl bash tools/env/_py14.sh jgraph.py cards --run e312_p_s0 [--tag e351]                                          # CPU: the cards scored from the two dumps (and their per-frame counts) compared exactly
    wsl bash tools/env/_py14.sh jgraph.py selftest                                                                    # CPU, toy model: the pieces that need no CUDA, planted differences and mutation checks

What is compared (card E351 section 0): `check` runs per frame e1 = eager, e2 = eager again (the null: must give 0 differences), g = graph forward (clones: see below) and compares the trees (P, out, pair-head output) leaf by leaf (same key, same dtype and shape,
the same bits by the integer view of floating-point tensors, which implies `torch.equal` except for NaN: +0.0 and -0.0 differ, a NaN with the same bits is equal). Tier 1 holds when the graph column has no differing key (and the null none) over ALL 873 frames: the last line
says so (`verdict`: a run over fewer frames, e.g. `--max`, is PARTIAL and has no verdict; `--max` needs its own `--tag` or `--runs-dir`) and the result file `<runs dir>/eval/jgraph/<tag>.json` holds the reports; `check` exits 0 also when
tensors differ (a finding is a result, not a crash: the launcher would retry it). `--runs-dir` is relative to the problem folder (the scripts run from evaluation/). The dump of an arm is the `ends_val.pkl` of `jends_eval.dump` (the frames are built by
`jprune_probe.frames_of`, which is tested frame for frame against `jends_eval.dump`) in a NEW directory with a copy of `model.pt` and `config.json` of the source run and `e351.json` (provenance, written LAST: a directory without it is incomplete), so that
`jends_eval.py score --run e351_<arm>_<run> ...` works unchanged (it reads runs/ only: the score lines fit dumps in runs/); the runs' own directories are never written. `commands` exits 1 when a dump directory is missing (NOT CHECKED) or a dry
check fails; `cards` requires exactly the cards of the system in BOTH directories (E312: the snapped one, E324 s: the snapped and the PRIMARY one) and exits 1 otherwise.
Every tensor the graph forward returns is a CLONE: the outputs of the graph live in its private pool and the next replay overwrites them (`out["heat"]` is even a view of the static maps tensor). `PoolRunner` is the CPU stand-in with the same overwrite
semantics (the selftest, `check --cpu`). `JEndModel.forward` is mirrored stage by stage in `GraphedModel._front`; its source hash is pinned (`FORWARD_SHA`), a change of the forward stops the construction.
"""
import argparse
import contextlib
import hashlib
import inspect
import json
import os
import pickle
import re
import shutil
import sys
import tempfile
import time
from types import SimpleNamespace

import numpy as np
import torch

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for sub in ("evaluation", "training", "training/grid", "training/data", "training/data/prep", "training/train", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub)))

FORWARD_SHA = "b2fabf9ca5bddd114e1bd6234398ef061bacd8ab71abff79c1289af090749f3d"                     # sha256 of the source of `JEndModel.forward` (lines right-stripped), 2026-11-15: `_front` mirrors it
PREFIX = "e351"                                                                                     # the names of the dump directories: runs/e351_<arm>_<run>
ARMS = ("eager", "g1")
THR = 0.1                                                                                           # the read-out threshold of the stored cards
PRIMARY = ("--attach", "--snap", "--snap-nogeom")                                                   # the PRIMARY read-out of E324 s (D94); the snapped one is ("--snap",)
COPIED_FILES = ("model.pt", "config.json")                                                          # a dump directory holds copies of these two files of the source run


# ---------------------------------------------------------------- 1. trees: leaves, clones, equality
def leaves(t, prefix=""):
    """{path: leaf} of a tree of dicts / lists / tuples (a tensor, a number, a string, None is a leaf)"""
    if isinstance(t, dict):
        return {p: v for k, x in t.items() for p, v in leaves(x, f"{prefix}/{k}").items()}
    if isinstance(t, (list, tuple)):
        return {p: v for i, x in enumerate(t) for p, v in leaves(x, f"{prefix}[{i}]").items()}
    return {prefix: t}


def map_tensors(fn, t):
    """the tree t with fn applied to every tensor (the structure and the other leaves as they are)"""
    if torch.is_tensor(t):
        return fn(t)
    if isinstance(t, dict):
        return {k: map_tensors(fn, v) for k, v in t.items()}
    if isinstance(t, (list, tuple)):
        return type(t)(map_tensors(fn, v) for v in t)
    return t


def clone_tree(t):
    return map_tensors(lambda v: v.clone(), t)


def copy_into(dst, src):
    """src copied IN PLACE into the tensors of dst (the same structure): what a replay does to the static outputs"""
    s = leaves(src)
    for p, d in leaves(dst).items():
        if torch.is_tensor(d):
            d.copy_(s[p])


INT_VIEW = {torch.bfloat16: torch.int16, torch.float16: torch.int16, torch.float32: torch.int32, torch.float64: torch.int64}


def bits(t):
    """the bit pattern of a floating-point tensor (its integer view), other tensors as they are"""
    return t.view(INT_VIEW[t.dtype]) if t.dtype in INT_VIEW else t


def diff_kind(a, b):
    """None when the two tensors have the same device, dtype, shape and bits (the integer view: a NaN with the same bits counts as equal, +0.0 and -0.0 differ; for equal bits `torch.equal` holds except for NaN); else 'dtype or shape',
    'value' (`torch.equal` says they differ) or 'bits' (`torch.equal` says equal, the bit patterns differ)"""
    if a.device != b.device or a.dtype != b.dtype or a.shape != b.shape:
        return "dtype or shape"
    if torch.equal(bits(a), bits(b)):
        return None
    return "bits" if torch.equal(a, b) else "value"


class Compare:
    """frame after frame, a reference tree against a tested one: `add(ref, test)`; `report()` names every key that differs (frames, first frame, what, the largest absolute difference, the most differing elements)"""

    def __init__(self):
        self.frames = self.tensors = self.other = 0
        self.bad = {}

    def _flag(self, key, what, ref=None, test=None):
        e = self.bad.setdefault(key, dict(frames=0, first=self.frames - 1, what=set(), max_abs=0.0, n_diff=0))
        e["frames"] += 1
        e["what"].add(what)
        if what in ("value", "bits"):
            e["n_diff"] = max(e["n_diff"], int((bits(ref) != bits(test)).sum()))
        if what == "value":
            e["max_abs"] = max(e["max_abs"], float(torch.nan_to_num((ref.double() - test.double()).abs()).max()))

    def add(self, ref, test):
        r, t = leaves(ref), leaves(test)
        self.frames += 1
        for k in sorted(r.keys() | t.keys()):
            if k not in t or k not in r:
                self._flag(k, "missing" if k not in t else "extra")
                continue
            a, b = r[k], t[k]
            if torch.is_tensor(a) != torch.is_tensor(b):
                self._flag(k, "kind")
            elif torch.is_tensor(a):
                self.tensors += 1
                kind = diff_kind(a, b)
                if kind:
                    self._flag(k, kind, a, b)
            else:
                self.other += 1
                if not bool(a == b):
                    self._flag(k, "leaf")

    def report(self):
        return dict(frames=self.frames, tensors=self.tensors, other_leaves=self.other, identical=not self.bad, keys_differing=len(self.bad),
                    differing={k: dict(v, what=sorted(v["what"])) for k, v in self.bad.items()})

    def lines(self, name):
        out = [f"{name}: {self.tensors} tensors (+ {self.other} other leaves) compared over {self.frames} frames: " + ("0 differences (torch.equal and the same bits)" if not self.bad else f"{len(self.bad)} keys differ")]
        for k, v in sorted(self.bad.items(), key=lambda kv: -kv[1]["frames"])[:12]:
            out.append(f"    {k}: {v['frames']} frames (first {v['first']}), {'/'.join(sorted(v['what']))}, max abs diff {v['max_abs']:.3e}, most differing elements {v['n_diff']}")
        return out


# ---------------------------------------------------------------- 2. the graph of the front
@contextlib.contextmanager
def inference(device_type, dtype=torch.bfloat16, cache=True):
    """no_grad + autocast (dtype None: none) as inference runs the forward"""
    with torch.no_grad(), (torch.autocast(device_type, dtype=dtype, cache_enabled=cache) if dtype is not None else contextlib.nullcontext()):
        yield


def eager_forward(model, x, dtype=torch.bfloat16):
    """the reference (and the c arm of the timing): the model's own forward as inference runs it -> (P, out, pair-head output), fresh tensors"""
    with inference(x.device.type, dtype):
        return model(x, gt=None)


def forward_sha(model):
    return hashlib.sha256("\n".join(line.rstrip() for line in inspect.getsource(type(model).forward).splitlines()).encode()).hexdigest()


class CudaGraphRunner:
    """the CUDA side: `capture(fn, ctx)` runs fn 3 times on a side stream (warm-up) and once under `torch.cuda.graph`, inside the contexts `ctx()` makes; `out` is what fn returned at capture (static tensors of the graph's pool); `replay()` runs the kernels again"""

    def capture(self, fn, ctx, warmup=3):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), ctx():
            for _ in range(warmup):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with ctx(), torch.cuda.graph(self.graph):
            self.out = fn()

    def replay(self):
        self.graph.replay()


class PoolRunner:
    """the CPU stand-in of `CudaGraphRunner` (selftest, `check --cpu`): `capture` keeps the result of one call as the static outputs, `replay` calls fn again and copies the result INTO them, so that an earlier result is overwritten as in a CUDA graph"""

    def capture(self, fn, ctx, warmup=3):
        self.fn, self.ctx = fn, ctx
        with ctx():
            self.out = fn()

    def replay(self):
        with self.ctx():
            copy_into(self.out, self.fn())


class GraphedModel:
    """a loaded `JEndModel` whose front (`jmodel(x, joint=False)` + the node head of the attach / ntok models) runs as one graph on a static input; the pair head `je` runs eager on the static outputs. `capture(x)` once (batch 1, the shape of x),
    then `forward(x)` -> (P, out, pair-head output) as `JEndModel.forward` returns it, every tensor a clone (see the module docstring). The pair head's own tensors are cloned as well: `out["heat"]` is a view of the static maps tensor."""

    def __init__(self, model, runner=None, dtype=torch.bfloat16):
        assert not model.training, "the graph is the inference path: model.eval() first (the normalisation layers must not be in training mode)"
        assert not model.e2e, "the graph is the inference path: the maps head gets no gradient here (e2e=False)"
        assert forward_sha(model) == FORWARD_SHA, "JEndModel.forward changed: `_front` mirrors the old text; check it and update FORWARD_SHA (`jgraph.py --sha` prints it)"
        assert not torch.backends.cudnn.benchmark, "torch.backends.cudnn.benchmark must be False (the algorithm of a captured convolution is fixed at capture)"
        self.model, self.runner, self.dtype, self.static_x = model, runner or CudaGraphRunner(), dtype, None

    def _front(self):
        """JEndModel.forward up to the pair head, on the static input"""
        m = self.model
        P, out = m.jmodel(self.static_x, joint=False)
        tok = nd = None
        if m.nodes is not None:
            nd = m.nodes(P["feat2"], P["feat1"], P["feat05"], P["road2"], P["centre1"])
            tok = nd["tok"]
        return P, out, tok, nd

    def capture(self, x):
        assert self.static_x is None, "captured already: one graph per instance"
        assert not torch.backends.cudnn.benchmark
        self.static_x = x.detach().clone()
        dev = x.device.type
        self.runner.capture(self._front, lambda: inference(dev, self.dtype, cache=False))
        return self

    def forward(self, x):
        assert self.static_x is not None, "capture(x) first"
        assert x.shape == self.static_x.shape and x.dtype == self.static_x.dtype and x.device == self.static_x.device, f"the graph is of {tuple(self.static_x.shape)} {self.static_x.dtype} {self.static_x.device}, got {tuple(x.shape)} {x.dtype} {x.device}"
        self.static_x.copy_(x)
        self.runner.replay()
        P, out, tok, nd = self.runner.out
        with inference(x.device.type, self.dtype):
            oe = self.model.je(out, P, gt=None, tok=tok, nodes=nd if self.model.je.attach is not None else None)
        return clone_tree((P, out, oe))


# ---------------------------------------------------------------- 3. the dumps and their scoring
VAL_FRAMES = 873                                                                                    # the frames of the VAL split a score needs


def rel(path):
    """path relative to the problem folder with forward slashes (a path outside it as it is): what the result files record"""
    return os.path.relpath(path, ROOT).replace(os.sep, "/") if os.path.abspath(path).startswith(ROOT + os.sep) else path


def resolve_dir(path):
    """a directory option relative to the problem folder (the scripts run from evaluation/), None as None"""
    return os.path.join(ROOT, path) if path else None


def dump_dir(run, arm, runs_dir=None):
    return os.path.join(runs_dir or os.path.join(ROOT, "runs"), f"{PREFIX}_{arm}_{run}")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def preflight(run, runs_dir=None):
    """before any GPU work: the source files are there and neither dump directory exists"""
    src = os.path.join(ROOT, "runs", run)
    for f in COPIED_FILES:
        assert os.path.isfile(os.path.join(src, f)), f"{os.path.join(src, f)} is missing"
    for arm in ARMS:
        assert not os.path.exists(dump_dir(run, arm, runs_dir)), f"{dump_dir(run, arm, runs_dir)} exists: a dump is never overwritten (move it by hand or use another --runs-dir)"


def write_dump(src, dst, frames, meta):
    """dst (a NEW directory) = copies of model.pt and config.json of src + ends_val.pkl (the frames of `jends_eval.dump`) + e351.json (LAST: the provenance, the sha256 of every file)"""
    os.makedirs(dst)                                                                                  # raises when it exists: nothing is overwritten
    for f in COPIED_FILES:
        shutil.copyfile(os.path.join(src, f), os.path.join(dst, f))
    with open(os.path.join(dst, "ends_val.pkl"), "wb") as f:
        pickle.dump(frames, f)
    meta = dict(meta, source=rel(src), frames=len(frames), **{f"{f}_sha256": sha256(os.path.join(dst, f)) for f in COPIED_FILES + ("ends_val.pkl",)})
    with open(os.path.join(dst, "e351.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return dst


def run_info(src):
    """(seed, attach) of a run from its config.json"""
    a = json.load(open(os.path.join(src, "config.json")))["args"]
    return int(a["seed"]), bool(a.get("attach", False))


def score_flags(attach):
    """the read-outs of the card: the snapped one; E324 s (attach) also its PRIMARY one"""
    return [("--snap",)] + ([PRIMARY] if attach else [])


def score_lines(run, seed, attach, tag=PREFIX):
    """the `jends_eval.py score` command lines for both dumps of run (the eager reference first)"""
    return [f"wsl bash tools/env/_py14.sh jends_eval.py score --run {os.path.basename(dump_dir(run, arm))} --seed {seed} --thr {THR} {' '.join(fl)} --tag {tag}" for fl in score_flags(attach) for arm in ARMS]


def score_ns(name, seed, flags, tag):
    """the options of `jends_eval.py score --run <name> --seed <seed> <flags> --tag <tag>` as `qends_eval.score_target` / `card_name` read them"""
    return SimpleNamespace(run=name, seed=seed, thr=THR, tag=tag, snap="--snap" in flags, attach="--attach" in flags, snap_nogeom="--snap-nogeom" in flags)


def expected_cards(run, seed, attach, tag=PREFIX):
    """the card files the score lines of `score_lines` write into each dump directory: E312 one (snapped), E324 s two (snapped and PRIMARY)"""
    import qends_eval as QE
    return sorted(QE.card_name(score_ns(run, seed, fl, tag)) for fl in score_flags(attach))


def check_score_ready(dst, seed, flags, tag, src=None, base_pt=None, full=None):
    """a dry check (nothing runs, nothing is written): would `jends_eval.py score --run <dst> --seed <seed> <flags> --tag <tag>` find everything? -> (problems, name of the card it would write); problems [] = ready.
    The checks of `jends_eval.score` and `qends_eval.cmd_score` before the long loop (the seed of the run, the base checkpoint, the full dump of B, the attached ends, the tag, a card that exists) + the dump's own completeness
    (e351.json, frame count, the sha256 of every file) and, when src is given, that model.pt / config.json are the source's and the frames are the source's VAL frames in the same order"""
    import qends_eval as QE
    bad = []
    for f in COPIED_FILES + ("ends_val.pkl", "e351.json"):
        if not os.path.isfile(os.path.join(dst, f)):
            bad.append(f"{f} is missing")
    if bad:
        return bad, None
    meta = json.load(open(os.path.join(dst, "e351.json")))
    frames = pickle.load(open(os.path.join(dst, "ends_val.pkl"), "rb"))
    bad += [f"{f}: sha256 differs from e351.json (the file changed after the dump)" for f in COPIED_FILES + ("ends_val.pkl",) if sha256(os.path.join(dst, f)) != meta.get(f"{f}_sha256")]
    if meta.get("frames") != len(frames) or len(frames) != VAL_FRAMES:
        bad.append(f"{len(frames)} frames in ends_val.pkl, e351.json says {meta.get('frames')}: a score needs all {VAL_FRAMES} VAL frames")
    if src is not None:
        bad += [f"{f} is not the source's" for f in COPIED_FILES if sha256(os.path.join(dst, f)) != sha256(os.path.join(src, f))]
        ref = pickle.load(open(os.path.join(src, "ends_val.pkl"), "rb"))
        if [(f["row"], f["token"]) for f in frames] != [(f["row"], f["token"]) for f in ref]:
            bad.append("the rows / tokens of ends_val.pkl are not the source dump's (the VAL frames in the same order)")
    args = torch.load(os.path.join(dst, "model.pt"), map_location="cpu")["args"]
    m = re.search(r"_s(\d+)[/\\]", str(args.get("maps", "")).replace("\\", "/") + "/")
    if not (args.get("seed", seed) == seed and (m is None or int(m.group(1)) == seed)):
        bad.append(f"--seed {seed} does not match the run (seed {args.get('seed')}, maps {args.get('maps')}): jends_eval.score would refuse")
    for what, p in (("the base checkpoint", base_pt or os.path.join(ROOT, "runs", f"e273_p_s{seed}", "model.pt")), ("the full dump of B", full or os.path.join(ROOT, "temp", "e276", f"predfull_e273_p_s{seed}.pkl"))):
        if not os.path.isfile(p):
            bad.append(f"{what} {p} is missing")
    ns = score_ns(os.path.basename(dst), seed, flags, tag)
    if ns.attach and frames and "att_pos" not in frames[0]:
        bad.append("the dump has no attached ends (att_pos) but --attach is asked")
    name = None
    try:
        name = QE.score_target(ns, dst)[0]
        QE.check_wire(ns)
    except AssertionError as e:
        bad.append(f"qends_eval refuses: {e}")
    return bad, name


def same_tree(a, b):
    """exact equality of nested dicts / lists / arrays / numbers (NaN equal to NaN)"""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same_tree(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same_tree(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(a == b) or (a != a and b != b)


def compare_cards(dir_a, dir_b, expected, tag=PREFIX):
    """both directories hold EXACTLY the cards `expected` (the names of `expected_cards`; a missing or an extra `ends_card_<tag>_*.json` is a difference), and each card and its `.counts.pkl` (per-frame counts) is exactly equal in the two:
    the key `run` (the directory's name) apart -> differences"""
    diffs = []
    for d in (dir_a, dir_b):
        found = sorted(f for f in os.listdir(d) if f.startswith(f"ends_card_{tag}_") and f.endswith(".json")) if os.path.isdir(d) else None
        if found != sorted(expected):
            diffs.append(f"{d}: cards {found}, expected {sorted(expected)}")
    for n in expected:
        pa, pb = os.path.join(dir_a, n), os.path.join(dir_b, n)
        if not (os.path.isfile(pa) and os.path.isfile(pb)):
            continue
        ca, cb = json.load(open(pa)), json.load(open(pb))
        diffs += [f"{n}: {k} differs" for k in sorted(set(ca) | set(cb)) if k != "run" and not same_tree(ca.get(k), cb.get(k))]
        ka, kb = pa[:-5] + ".counts.pkl", pb[:-5] + ".counts.pkl"
        if False:
            diffs.append(f"{n[:-5]}.counts.pkl: missing or not equal")
    return diffs


# ---------------------------------------------------------------- 4. the drivers
def cpu_frames(model_out, b):
    from jprune_probe import frames_of
    return frames_of({k: (v.cpu() if torch.is_tensor(v) else v) for k, v in model_out.items()}, b)


def verdict(null, graph, total=VAL_FRAMES):
    """the verdict line of `check` from the reports of the null control and of the graph column (`Compare.report()`): a run that did not compare all `total` frames, or compared no tensor, has no verdict"""
    if graph["frames"] != total or graph["tensors"] == 0 or null["tensors"] == 0:
        return f"PARTIAL ({graph['frames']} of {total} frames, {graph['tensors']} tensors): no verdict" + ("" if graph["identical"] and null["identical"] else f"; differing keys: null {null['keys_differing']}, graph {graph['keys_differing']}")
    if not null["identical"]:
        return "NULL NOT CLEAN: the eager forward is not reproducible in this process; the graph can only be judged at that level"
    if graph["identical"]:
        return f"TIER 1: EQUAL (torch.equal) and the same bits (integer view) on every tensor of all {total} frames"
    return "NOT IDENTICAL: tier 2 / CUT of the card"


def check_options(a):
    """the rules of `check` before any work -> (runs_dir, result file): a partial run (--max) never takes the default result name or writes into runs/ (it needs its own --tag or --runs-dir; a dump of a part only into another --runs-dir);
    a result file that exists is never overwritten"""
    runs_dir = resolve_dir(a.runs_dir)
    assert not a.max or a.tag or runs_dir, "--max (a part of the frames) needs its own --tag or --runs-dir: the default result file is the one of the full run"
    assert not (a.dump and a.max) or runs_dir, "--dump with --max would write a part of the frames: only to another --runs-dir than runs/"
    out = os.path.join(runs_dir or os.path.join(ROOT, "runs"), "eval", "jgraph", f"{a.tag or f'{PREFIX}_equal_{a.run}'}.json")
    assert not os.path.exists(out), f"{out} exists: another --tag (a measurement is never overwritten)"
    return runs_dir, out


def check(a):
    import jends_eval as JE
    import lane_gt_data as LD
    dev = "cpu" if a.cpu else "cuda"
    assert a.cpu or torch.cuda.is_available(), "no GPU (--cpu runs the stand-in graph on the CPU: a smoke test of the pipeline)"
    torch.backends.cudnn.benchmark = False
    runs_dir, out_json = check_options(a)
    if a.dump:
        preflight(a.run, runs_dir)
    if dev == "cuda":
        import gpu_gate
        gpu_gate.wait_for_room(who=f"jgraph check {a.run}")
    print(f"loadavg {open('/proc/loadavg').read().split()[0] if os.path.exists('/proc/loadavg') else 'n/a'} | cudnn.benchmark {torch.backends.cudnn.benchmark} | torch {torch.__version__} | {torch.cuda.get_device_name(0) if dev == 'cuda' else 'cpu'}")
    model, _ = JE.load_model(a.run, dev)
    LD.QData.DEFAULT_JUNCTION = True
    ds, dl = LD.loader("val_clean", bs=1, workers=a.workers, shuffle=False, map_mode="error", pixels=False)
    assert ds.missing == 0, f"{ds.missing} VAL frames have no stored lane GT"
    x0 = LD.collate([ds[0]])["x"].to(dev).float()
    for _ in range(5):                                                                                # warm-up of the eager path (handles, lazy init), then the capture
        eager_forward(model, x0)
    gm = GraphedModel(model, runner=PoolRunner() if a.cpu else None).capture(x0)
    null, graph, frames, t0 = Compare(), Compare(), {arm: [] for arm in ARMS}, time.time()
    for n, b in enumerate(dl):
        if a.max and n >= a.max:
            break
        x = b["x"].to(dev).float()
        e1, e2, g = eager_forward(model, x), eager_forward(model, x), gm.forward(x)
        null.add(e1, e2)
        graph.add(e1, g)
        if a.dump:
            frames["eager"] += cpu_frames(e1[2], b)
            frames["g1"] += cpu_frames(g[2], b)
        if (n + 1) % 200 == 0:
            print(f"  {n + 1} frames  {time.time() - t0:.0f} s  differing keys: null {len(null.bad)}, graph {len(graph.bad)}", flush=True)
    print("\n".join(null.lines("NULL control, eager against eager (two passes of one process)") + graph.lines("graph forward against eager")))
    v = verdict(null.report(), graph.report())
    print(f"{a.run}: {v}")
    res = dict(run=a.run, frames=graph.frames, max=a.max, device=dev, torch=torch.__version__, cudnn_benchmark=torch.backends.cudnn.benchmark, precision="bf16 autocast, batch 1, no_grad", null=null.report(), graph=graph.report(), verdict=v, dumps={})
    if a.dump:
        for arm in ARMS:
            res["dumps"][arm] = rel(write_dump(os.path.join(ROOT, "runs", a.run), dump_dir(a.run, arm, runs_dir), frames[arm],
                                               dict(arm=arm, source_run=a.run, precision=res["precision"], graph=arm == "g1", device=dev, torch=torch.__version__, cudnn_benchmark=False, null_identical=null.report()["identical"], graph_identical=graph.report()["identical"])))
            print("written", res["dumps"][arm])
        if not runs_dir:
            print("next (CPU, one at a time; `jgraph.py commands --run <run>` is the dry check, `jgraph.py cards --run <run>` the comparison afterwards):\n" + "\n".join(score_lines(a.run, *run_info(os.path.join(ROOT, "runs", a.run)), PREFIX)))
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    res["result_file"] = rel(out_json)
    json.dump(res, open(out_json, "w"), indent=1)
    print("written", rel(out_json))


def dry_report(run, tag, runs_dir=None, **kw):
    """-> (lines, ok): the score lines of `run` and the dry check of every dump directory and read-out; a directory that is not there is NOT CHECKED and not ok (kw: base_pt, full for the tests)"""
    src = os.path.join(ROOT, "runs", run)
    seed, attach = run_info(src)
    lines, ok = score_lines(run, seed, attach, tag), True
    for arm in ARMS:
        dst = dump_dir(run, arm, runs_dir)
        if not os.path.isdir(dst):
            lines.append(f"  [{arm}] NOT CHECKED: {rel(dst)} is not there (run `jgraph.py check --run {run} --dump` first)")
            ok = False
            continue
        for fl in score_flags(attach):
            bad, name = check_score_ready(dst, seed, fl, tag, src=src, **kw)
            ok &= not bad
            lines.append(f"  [{arm}] {' '.join(fl)}: " + (f"ready, would write {name}" if not bad else "NOT READY: " + "; ".join(bad)))
    if runs_dir:
        lines.append("note: the score lines read runs/ only (`jends_eval.py score` has no --runs-dir): they fit dumps in runs/, not these")
    return lines, ok


def commands(a):
    lines, ok = dry_report(a.run, a.tag, resolve_dir(a.runs_dir))
    print("\n".join(lines))
    sys.exit(0 if ok else 1)


def cards(a):
    seed, attach = run_info(os.path.join(ROOT, "runs", a.run))
    expected = expected_cards(a.run, seed, attach, a.tag)
    diffs = compare_cards(dump_dir(a.run, "eager", resolve_dir(a.runs_dir)), dump_dir(a.run, "g1", resolve_dir(a.runs_dir)), expected, a.tag)
    print(f"{len(expected)} cards expected ({', '.join(expected)}): " + ("present in both directories and exactly equal (difference 0)" if not diffs else f"{len(diffs)} differences"))
    print("\n".join("  " + d for d in diffs))
    sys.exit(0 if not diffs else 1)


# ---------------------------------------------------------------- 5. the selftest (CPU, toy model)
def _toy(**kw):
    import jends_train as JT
    torch.manual_seed(0)
    return JT.JEndModel(dict(d=64, ctx="none"), order=True, evidence=True, **kw).eval()


def _refused(call, what):
    try:
        call()
    except AssertionError:
        return
    raise SystemExit(f"{what}: accepted")


@contextlib.contextmanager
def _root(td):
    """ROOT is td meanwhile (the paths the code builds from it lie in the test directory)"""
    global ROOT
    saved, ROOT = ROOT, td
    try:
        yield
    finally:
        ROOT = saved


def planted_misses(cls=Compare):
    """the planted differences, one case each: the names of the cases that `cls` does not report (must be none)"""
    f = lambda *s: torch.arange(float(np.prod(s))).reshape(*s)  # noqa: E731
    base = dict(heat=f(2, 3), idx=torch.arange(4), valid=torch.tensor([True, False, True]), pair=(f(2), dict(z=f(1, 2))), skip=("zone",), n=3)
    ref = lambda: {k: (v.clone() if torch.is_tensor(v) else (clone_tree(v) if isinstance(v, (tuple, dict)) else v)) for k, v in base.items()}  # noqa: E731

    def with_(**ch):
        t = ref()
        t.update(ch)
        return t
    flipped = f(2, 3).clone()
    flipped[1, 2] += 0.5
    negzero = f(2, 3).clone()
    negzero[0, 0] = -0.0                                                                              # equal to +0.0 for torch.equal, other bits
    bf = torch.tensor([True, True, True])
    nan_a, nan_b = f(2, 3).clone(), f(2, 3).clone()
    nan_a[0, 0] = nan_b[0, 0] = float("nan")
    nan_c = nan_b.clone()
    nan_c[1, 1] = float("nan")
    t_pair = ref()
    t_pair["pair"] = (f(2), dict(z=f(1, 2) + 1))
    t_extra, t_miss = ref(), ref()
    t_extra["new"] = f(1)
    del t_miss["idx"]
    cases = {"value": (ref(), with_(heat=flipped), "/heat"), "signed zero": (ref(), with_(heat=negzero), "/heat"), "bool": (ref(), with_(valid=bf), "/valid"), "dtype": (ref(), with_(heat=f(2, 3).to(torch.bfloat16)), "/heat"),
             "shape": (ref(), with_(heat=f(3, 2)), "/heat"), "nested": (ref(), t_pair, "/pair[1]/z"), "extra key": (ref(), t_extra, "/new"), "missing key": (ref(), t_miss, "/idx"), "leaf": (ref(), with_(skip=("core",)), "/skip[0]"),
             "number": (ref(), with_(n=4), "/n"), "kind": (ref(), with_(heat=3.0), "/heat"), "NaN only in one": (with_(heat=nan_a), with_(heat=nan_c), "/heat")}
    misses = []
    for name, (r, t, key) in cases.items():
        c = cls()
        c.add(r, t)
        if key not in c.bad:
            misses.append(name)
    c = cls()
    c.add(with_(heat=nan_a), with_(heat=nan_b))                                                       # NaN at the same place is no difference
    c.add(ref(), ref())
    if c.bad:
        misses.append("identical trees reported as different: " + ",".join(c.bad))
    return misses


def _test_compare():
    """every planted difference is named by its key, identical trees (a NaN at the same place included) give none, the counts, the sizes and the kinds are right; a comparison that finds nothing is caught by the same cases (mutation)"""
    global diff_kind
    assert planted_misses() == [], planted_misses()
    c = Compare()
    c.add(dict(a=torch.zeros(2), b=(torch.ones(3), "x")), dict(a=torch.zeros(2), b=(torch.ones(3), "x")))
    assert (c.tensors, c.other, c.bad) == (2, 1, {}), (c.tensors, c.other, c.bad)
    c = Compare()
    c.add(dict(a=torch.tensor([1.0, 2.0, 4.0]), z=torch.zeros(2)), dict(a=torch.tensor([1.0, 2.5, 3.0]), z=torch.tensor([0.0, -0.0])))
    assert c.bad["/a"]["max_abs"] == 1.0 and c.bad["/a"]["n_diff"] == 2 and c.bad["/a"]["what"] == {"value"}, c.bad
    assert c.bad["/z"]["what"] == {"bits"} and c.bad["/z"]["n_diff"] == 1 and c.bad["/z"]["max_abs"] == 0.0, c.bad
    saved = diff_kind
    try:
        diff_kind = lambda a, b: None                                                                 # noqa: E731  mutation
        assert planted_misses() != [], "the planted cases did not catch a comparison that finds nothing"
    finally:
        diff_kind = saved


def _test_verdict():
    """the verdict line: only a run over all 873 frames with tensors in both columns has one (a loader without frames must not say TIER 1)"""
    rep = lambda frames, tensors, bad=0: dict(frames=frames, tensors=tensors, identical=not bad, keys_differing=bad)  # noqa: E731
    assert verdict(rep(873, 20000), rep(873, 20000)).startswith("TIER 1: EQUAL (torch.equal) and the same bits")
    assert verdict(rep(0, 0), rep(0, 0)).startswith("PARTIAL (0 of 873 frames, 0 tensors)")
    assert verdict(rep(8, 176), rep(8, 176)).startswith("PARTIAL (8 of 873 frames") and "differing keys" in verdict(rep(8, 176), rep(8, 176, 3))
    assert verdict(rep(873, 0), rep(873, 0)).startswith("PARTIAL") and verdict(rep(873, 5), rep(873, 0)).startswith("PARTIAL")
    assert verdict(rep(873, 20000), rep(873, 20000, 2)).startswith("NOT IDENTICAL") and verdict(rep(873, 20000, 1), rep(873, 20000)).startswith("NULL NOT CLEAN")


def _test_options(td):
    """`check` options: a part of the frames never takes the default result name or writes into runs/; a result file that exists is never overwritten; a relative --runs-dir is relative to the problem folder"""
    a = lambda **kw: SimpleNamespace(**dict(dict(run="e312_p_s0", max=0, dump=False, tag=None, runs_dir=None), **kw))  # noqa: E731
    with _root(td):
        assert check_options(a())[1] == os.path.join(td, "runs", "eval", "jgraph", "e351_equal_e312_p_s0.json")
        _refused(lambda: check_options(a(max=8)), "--max without a tag or a runs dir")
        assert check_options(a(max=8, tag="smoke"))[1].endswith("smoke.json") and check_options(a(max=8, runs_dir="temp/x"))[0] == os.path.join(td, "temp", "x")
        _refused(lambda: check_options(a(max=8, dump=True, tag="smoke")), "--max --dump without a runs dir")
        assert check_options(a(max=8, dump=True, runs_dir="temp/x"))[0] == os.path.join(td, "temp", "x")
        os.makedirs(os.path.join(td, "runs", "eval", "jgraph"))
        open(os.path.join(td, "runs", "eval", "jgraph", "old.json"), "w").close()
        _refused(lambda: check_options(a(tag="old")), "an existing result file")
        assert resolve_dir(None) is None and resolve_dir("/abs/x") == "/abs/x" and resolve_dir("temp/x") == os.path.join(td, "temp", "x")
        assert rel(os.path.join(td, "runs", "e351_g1_x")) == "runs/e351_g1_x" and rel("/elsewhere/x") == "/elsewhere/x"


def _test_graph():
    """the graph forward on the CPU stand-in (the pool is overwritten by the next replay) equals the eager forward, for a plain model and for the attach model (the node head: the `nd` / `tok` path); every result is a clone (mutation: without it an earlier result changes)"""
    global clone_tree
    for kw in (dict(), dict(attach=True, scales=True)):
        m = _toy(**kw)
        xs = [torch.randn(1, 39, 200, 200) for _ in range(3)]
        gm = GraphedModel(m, runner=PoolRunner()).capture(xs[0])
        cmp, cmp_other = Compare(), Compare()
        for x in xs:
            cmp.add(eager_forward(m, x), gm.forward(x))
        cmp_other.add(eager_forward(m, xs[1]), gm.forward(xs[2]))                                      # the input matters: another frame is another output
        assert cmp.report()["identical"] and cmp.tensors > 50 and not cmp_other.report()["identical"], (kw, cmp.report(), len(cmp_other.bad))
        r1 = gm.forward(xs[1])
        before = clone_tree(r1)
        gm.forward(xs[2])
        keep = Compare()
        keep.add(before, r1)
        assert keep.report()["identical"], "an earlier result changed after the next replay"
        saved = clone_tree
        try:
            clone_tree = lambda t: t                                                                  # noqa: E731  mutation
            r1 = gm.forward(xs[1])
            before = saved(r1)
            gm.forward(xs[2])
            keep = Compare()
            keep.add(before, r1)
            assert not keep.report()["identical"] and any(k.startswith(("[0]", "[1]")) for k in keep.bad), "the clone test did not catch results that alias the static pool"
        finally:
            clone_tree = saved


def _test_guards():
    """a changed forward, the e2e model, a model in training mode and cudnn.benchmark stop the construction; forward before capture, a second capture and another input shape stop the call"""
    m = _toy()
    changed = _toy()
    changed.__class__ = type("Changed", (type(changed),), {"forward": lambda self, x, gt=None: None})
    _refused(lambda: GraphedModel(changed, runner=PoolRunner()), "a changed forward")
    _refused(lambda: GraphedModel(_toy(e2e=True), runner=PoolRunner()), "the e2e model")
    _refused(lambda: GraphedModel(_toy().train(), runner=PoolRunner()), "a model in training mode")
    torch.backends.cudnn.benchmark = True
    try:
        _refused(lambda: GraphedModel(m, runner=PoolRunner()), "cudnn.benchmark True")
    finally:
        torch.backends.cudnn.benchmark = False
    gm = GraphedModel(m, runner=PoolRunner())
    z = torch.zeros(1, 39, 200, 200)
    _refused(lambda: gm.forward(z), "forward before capture")
    gm.capture(z)
    _refused(lambda: gm.capture(z), "a second capture")
    _refused(lambda: gm.forward(torch.zeros(2, 39, 200, 200)), "another batch size")


def _test_dump_and_score(td):
    """the frames of a dump (the keys `score` reads), the dump directory (copies, never an overwrite, e351.json last), the dry check with its planted problems (an empty dump included), the dry report (a missing directory is NOT CHECKED), the score lines"""
    m = _toy(attach=True, scales=True)
    out = eager_forward(m, torch.randn(1, 39, 200, 200))[2]
    fr = cpu_frames(out, dict(token=["t0"], row=torch.tensor([7])))[0]
    assert fr["row"] == 7 and fr["token"] == "t0" and {"kp_pos", "kp_dir", "kp_p", "valid", "logit", "allowed", "scale", "att_pos", "att_dir", "att_p"} <= set(fr) and all(fr[k].dtype == np.float32 for k in ("kp_pos", "logit", "att_pos")), sorted(fr)
    src = os.path.join(td, "runs", "e312_p_s0")
    os.makedirs(src)
    torch.save(dict(args=dict(seed=0, maps=os.path.join(td, "e307_n_s0", "model.pt")), state={}), os.path.join(src, "model.pt"))
    json.dump(dict(args=dict(seed=0, attach=True)), open(os.path.join(src, "config.json"), "w"))
    frames = [dict(fr, row=i, token=f"t{i}") for i in range(VAL_FRAMES)]
    pickle.dump(frames, open(os.path.join(src, "ends_val.pkl"), "wb"))
    base, full = os.path.join(td, "base.pt"), os.path.join(td, "full.pkl")
    open(base, "w").close()
    open(full, "w").close()
    assert run_info(src) == (0, True)
    with _root(td):
        preflight("e312_p_s0")
        d = write_dump(src, dump_dir("e312_p_s0", "g1"), frames, dict(arm="g1"))
        assert json.load(open(os.path.join(d, "e351.json")))["source"] == "runs/e312_p_s0"
        for what, call in (("an existing dump directory", lambda: write_dump(src, d, frames, {})), ("preflight over it", lambda: preflight("e312_p_s0"))):
            try:
                call()
                raise SystemExit(f"{what} was accepted")
            except (FileExistsError, AssertionError):
                pass
        assert sorted(os.listdir(d)) == ["config.json", "e351.json", "ends_val.pkl", "model.pt"] and json.load(open(os.path.join(d, "e351.json")))["frames"] == VAL_FRAMES
        ok = dict(src=src, base_pt=base, full=full)
        ready = {fl: check_score_ready(d, 0, fl, PREFIX, **ok) for fl in score_flags(True)}
        assert ready[("--snap",)] == ([], "ends_card_e351_snap_thr0.1.json") and ready[PRIMARY] == ([], "ends_card_e351_snap_nogeom_attach_thr0.1.json"), ready
        for name, kw in {"seed": dict(seed=1), "reserved tag": dict(tag="snap"), "base checkpoint": dict(base_pt=os.path.join(td, "none.pt")), "full dump": dict(full=os.path.join(td, "none.pkl"))}.items():
            call = dict(seed=0, flags=("--snap",), tag=PREFIX, **ok)
            call.update(kw)
            assert check_score_ready(d, call["seed"], call["flags"], call["tag"], src=src, base_pt=call["base_pt"], full=call["full"])[0], (name, "not caught")
        empty = write_dump(src, os.path.join(td, "empty_dump"), [], {})                                                                            # an empty dump is reported, not an IndexError
        assert any("0 frames" in p for p in check_score_ready(empty, 0, PRIMARY, PREFIX, **ok)[0]) and any("0 frames" in p for p in check_score_ready(empty, 0, PRIMARY, PREFIX)[0])
        pickle.dump([{k: v for k, v in f_.items() if k != "att_pos"} for f_ in frames], open(os.path.join(d, "ends_val.pkl"), "wb"))                  # a dump without the attached ends, changed after e351.json
        bad = check_score_ready(d, 0, PRIMARY, PREFIX, **ok)[0]
        assert any("sha256" in p for p in bad) and any("att_pos" in p for p in bad), bad
        pickle.dump(frames, open(os.path.join(d, "ends_val.pkl"), "wb"))
        open(os.path.join(d, "ends_card_e351_snap_thr0.1.json"), "w").close()                                                                      # a card that exists: score refuses to overwrite a tagged card
        assert any("overwritten" in p for p in check_score_ready(d, 0, ("--snap",), PREFIX, **ok)[0])
        os.remove(os.path.join(d, "ends_card_e351_snap_thr0.1.json"))
        lines, all_ok = dry_report("e312_p_s0", PREFIX, None, base_pt=base, full=full)                                                           # the eager directory is not there: NOT CHECKED, not ok
        assert not all_ok and any("[eager] NOT CHECKED" in ln for ln in lines) and any("[g1] --snap: ready" in ln for ln in lines), lines
        write_dump(src, dump_dir("e312_p_s0", "eager"), frames, dict(arm="eager"))
        lines, all_ok = dry_report("e312_p_s0", PREFIX, None, base_pt=base, full=full)
        checks = [ln for ln in lines if ln.startswith("  [")]
        assert all_ok and len(checks) == 4 and all("ready, would write" in ln for ln in checks) and not any(ln.startswith("note") for ln in lines), lines
        assert dry_report("e312_p_s0", PREFIX, os.path.join(td, "elsewhere"), base_pt=base, full=full)[1] is False and dry_report("e312_p_s0", PREFIX, d, base_pt=base, full=full)[0][-1].startswith("note: the score lines read runs/ only")
        assert score_lines("e312_p_s0", 0, False) == [f"wsl bash tools/env/_py14.sh jends_eval.py score --run e351_{arm}_e312_p_s0 --seed 0 --thr 0.1 --snap --tag e351" for arm in ARMS], score_lines("e312_p_s0", 0, False)
        lines = score_lines("e324_s_s0", 0, True)
        assert len(lines) == 4 and lines[3].endswith("--run e351_g1_e324_s_s0 --seed 0 --thr 0.1 --attach --snap --snap-nogeom --tag e351"), lines


def _test_cards(td):
    """the cards expected per system (E312 one, E324 s two); both directories hold exactly those and they are equal except `run`; a planted difference in a number and one in the per-frame counts, a missing card, an extra card and a missing directory are found"""
    snap, primary = "ends_card_e351_snap_thr0.1.json", "ends_card_e351_snap_nogeom_attach_thr0.1.json"
    assert expected_cards("e312_p_s0", 0, False) == [snap] and expected_cards("e324_s_s0", 0, True) == sorted([snap, primary])
    da, db = os.path.join(td, "a"), os.path.join(td, "b")
    for p_ in (da, db):
        os.makedirs(p_)
        json.dump(dict(run=os.path.basename(p_), numbers=dict(head=dict(topo_F1=0.7)), difference=dict(topo_F1=0.0)), open(os.path.join(p_, snap), "w"))
        pickle.dump(dict(scenes=["s"], head=[dict(n=np.array([1.0, np.nan]))]), open(os.path.join(p_, snap[:-5] + ".counts.pkl"), "wb"))
    assert compare_cards(da, db, [snap]) == []
    assert len(compare_cards(da, db, [snap, primary])) == 2                                                                                       # E324 s expects two cards: one directory pair holding only the snapped one
    assert compare_cards(da, os.path.join(td, "none"), [snap])
    open(os.path.join(db, "ends_card_e351_snap_all_thr0.1.json"), "w").close()                                                                    # an extra card in one directory
    assert any(db in d_ for d_ in compare_cards(da, db, [snap]))
    os.remove(os.path.join(db, "ends_card_e351_snap_all_thr0.1.json"))
    json.dump(dict(run="b", numbers=dict(head=dict(topo_F1=0.7000001)), difference=dict(topo_F1=0.0)), open(os.path.join(db, snap), "w"))
    pickle.dump(dict(scenes=["s"], head=[dict(n=np.array([1.0, 2.0]))]), open(os.path.join(db, snap[:-5] + ".counts.pkl"), "wb"))
    diffs = compare_cards(da, db, [snap])
    assert len(diffs) == 2 and "numbers differs" in diffs[0] and "counts.pkl" in diffs[1], diffs


def selftest():
    torch.set_num_threads(2)
    _test_compare()
    _test_verdict()
    _test_graph()
    _test_guards()
    with tempfile.TemporaryDirectory() as td:
        _test_options(os.path.join(td, "o"))
        _test_dump_and_score(os.path.join(td, "d"))
        _test_cards(os.path.join(td, "c"))
    print("jgraph selftest ok")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    c = sub.add_parser("check")
    c.add_argument("--run", required=True)
    c.add_argument("--dump", action="store_true", help="also write runs/e351_eager_<run>/ and runs/e351_g1_<run>/ (new directories)")
    c.add_argument("--max", type=int, default=0, help="only the first N frames (a test of the pipeline); with --dump only into another --runs-dir")
    c.add_argument("--cpu", action="store_true", help="the CPU stand-in graph (a smoke test of the pipeline, no CUDA)")
    c.add_argument("--workers", type=int, default=2)
    c.add_argument("--tag", default=None, help="name of the result file runs/eval/jgraph/<tag>.json (default e351_equal_<run>); never overwritten")
    for p in (c, sub.add_parser("commands"), sub.add_parser("cards")):
        p.add_argument("--runs-dir", default=None, help="where the dump directories (and, for `check`, eval/jgraph/<tag>.json) are written / read, relative to the problem folder (default runs/)")
        if p is not c:
            p.add_argument("--run", required=True)
            p.add_argument("--tag", default=PREFIX, help="the --tag of the score commands")
    sub.add_parser("selftest")
    ap.add_argument("--sha", action="store_true", help="print the source hash of JEndModel.forward (FORWARD_SHA) and stop")
    a = ap.parse_args()
    if a.cmd is None and not a.sha:
        ap.error("a command: check | commands | cards | selftest")
    if a.sha:
        import jends_train as JT
        print(forward_sha(JT.JEndModel(dict(d=64, ctx="none"))))
        return
    {"check": check, "commands": commands, "cards": cards, "selftest": lambda _: selftest()}[a.cmd](a)


if __name__ == "__main__":
    main()
