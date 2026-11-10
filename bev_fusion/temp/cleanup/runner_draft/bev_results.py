"""bev_results - run a trained network over whole scenes and store a result set (runner/resultset.py) that the evaluation reads without the network.

    from bev_results import Runner                     # from runner/, or: python runner/bev_results.py <command>
    r = Runner()                                       # the project folder (default: this one)
    r.check("runs/t20_base_s0", "scene-0003:3")        # the frames, the id, does it exist already; the network is not built, nothing is written          -> dict
    r.run("runs/t20_base_s0", "scene-0003:3", taps=["fused"])    # run and store (the same content is never made twice)                          -> dict
    r.list()  /  r.get("3f9a0c1d2e", verify=True)      # the stored result sets / one of them (verify: read every frame and check it)                    -> list / dict
    rs = r.open("3f9a0c1d2e")                          # a ResultSet: rs.keys(), rs.load("scene-0003/000"), rs.subset([...]), rs.check()

    python runner/bev_results.py run runs/t20_base_s0 --frames scene-0003:3 --taps fused [--split val] [--dts -0.5 0 0.5] [--device cpu] [--code training] [--json]
    python runner/bev_results.py check <run> --frames ...   |   list   |   get ID [--verify]   |   taps          (--json: print exactly what the method returns)

frames, comma-separated:  scene-0003 (the whole scene)   scene-0003:3 (its first 3 frames)   all (every scene of the split)   random:5@1 (5 random scenes, seed 1: the seed is required).
The runner always runs WHOLE scenes in time order with the memory carried from keyframe to keyframe (D70: a network with a memory is initialised by its own cold start at the start
of every scene and then uses it; there is no cold mode); `frames` only selects what to run and keep. A result set holds, per frame and dt, what the evaluation's live subject returns
(maps, distance fields, velocity fields, detections: made by its own functions) plus the taps asked for; no ground truth is read. Results go to <run>/results/<id>/.
The run's own code snapshot (<run>/code) is used; --code training runs its checkpoint under the live code. Like bev_runner the process works from the project folder.
Exit codes: 0 ok, 1 a check found problems, 2 an error. Tests: tests/test_bev_results.py (real run, CPU, minutes), tests/test_resultset.py (the format, seconds).
"""
import argparse
import contextlib
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
AUTORESEARCH = os.path.abspath(os.path.join(ROOT, "..", "autoresearch"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import resultset as RS  # noqa: E402
from resultset import ResultSet, ResultSetError as RunnerError  # noqa: E402

DTS = (-0.5, 0.0, 0.5)                                     # what the evaluation scores: now and +-0.5 s
TAPS = ("lift", "cam_reduce", "cam_temporal", "temporal_agg", "memory_conf", "radar_raster", "radar_feat", "radar_state", "radar_memory", "fused",
        "motion_l0", "motion_l1", "motion_l2", "trunk")     # the stages of evaluation/bev_adapter.bev_hooks() and the trunk's features
MEMORY_EMPTY = 0.05                                        # the memory confidence of a scene's first frame must not exceed this (evaluation tap memory_conf: "0 on the first frame")


# ---- which frames
def parse_frames(spec):
    """'scene-0003' | 'scene-0003:3' | 'all' | 'random:5@1' | a comma-separated list (or a list of such texts) -> [("scene", name, first | None) | ("all",) | ("random", n, seed)]"""
    items = []
    for part in ([spec] if isinstance(spec, str) else list(spec)):
        for t in (x.strip() for x in part.split(",") if x.strip()):
            if t == "all":
                items.append(("all",))
            elif re.fullmatch(r"random:\d+", t):
                raise RunnerError(f"{t!r}: a random draw needs its seed, write {t}@1")
            elif re.fullmatch(r"random:\d+@\d+", t):
                n, seed = re.findall(r"\d+", t)
                items.append(("random", int(n), int(seed)))
            elif re.fullmatch(r"scene-\d+(:\d+)?", t):
                name, _, first = t.partition(":")
                items.append(("scene", name, int(first) if first else None))
            else:
                raise RunnerError(f"{t!r}: frames are scene-0003, scene-0003:3, all or random:5@1")
    if not items:
        raise RunnerError("no frames given")
    return items


def plan_frames(items, scenes):
    """items + {scene name: dataset indices in time order} -> [{"scene", "index" (position inside the scene), "di" (dataset index)}]: whole scenes in the order of the items, a scene
    named twice once; `:N` keeps its first N frames"""
    chosen = {}                                            # scene -> how many frames to keep (None = all)
    for it in items:
        if it[0] == "scene":
            if it[1] not in scenes:
                raise RunnerError(f"{it[1]} is not a scene of this split")
            chosen.setdefault(it[1], it[2])
            continue
        names = sorted(scenes)
        for s in names if it[0] == "all" else random.Random(it[2]).sample(names, min(it[1], len(names))):
            chosen.setdefault(s, None)
    return [{"scene": s, "index": k, "di": di} for s, first in chosen.items() for k, di in enumerate(scenes[s][:first])]


def scene_index(ds):
    """{scene name: dataset indices in time order}"""
    names = {s["token"]: s["name"] for s in ds.nusc.scene}
    by = {}
    for i, token in enumerate(ds.samples):
        smp = ds.nusc.get("sample", token)
        by.setdefault(names[smp["scene_token"]], []).append((smp["timestamp"], i))
    return {n: [i for _, i in sorted(v)] for n, v in by.items()}


# ---- the data and the network's outputs
def open_data(config, split):
    """(cfg, dataset, item): item(i) = (the network input of dataset item i, the row bev_runner.Network needs). The input is the dataset item through the precompute step, without
    ground truth, checked against interface.INPUTS: bev_runner.network_inputs does the same for one scene per call and would reload the data for every scene."""
    import train as T
    from feature_cache import collate_cached
    from interface import check_inputs
    from model import ModelConfig
    from nuscenes_dataset import collate
    from precompute import Precompute
    cfg = T.read_config(config)
    train_ds, val_ds = T.make_datasets(cfg)
    ds = val_ds if split == "val" else train_ds
    getattr(ds, "dataset", ds).with_gt = False
    fn = collate_cached if cfg.feature_cache else collate
    pre = Precompute.from_config(ModelConfig(**cfg.model))

    def item(i):
        batch = pre(fn([ds[i]]))
        batch.pop("gt", None)
        check_inputs(batch, raise_on=True)
        token = batch["tokens"][0]
        return batch, {"index": i, "token": token, "next": ds.nusc.get("sample", token)["next"]}
    return cfg, ds, item


def frame_outputs(model, res, cfg, dts):
    """the raw output of one frame (batch 1) -> {"dt": {dt: {prob, dist, dir, mass, vhat, objvel, dets, detr}}}: what the evaluation's live subject (evaluation/bev_adapter.py)
    returns, made by its own functions"""
    from evaluation.bev_adapter import predictions, predictions_1m
    from losses import NUSC_GROUPS
    from object_head import decode
    refine = bool(getattr(cfg, "model", {}).get("refine")) and hasattr(model, "boxes")
    out = {}
    for dt in dts:
        p = res["out"][dt]
        prob, dist = predictions(p, 0, model.spec)
        o = {"prob": prob, "dist": dist, "dir": {}, "mass": p["mass"][0, 0].float().cpu().numpy() if p.get("mass") is not None else None,
             "dets": decode({k: v[:1].float().cpu() for k, v in p["obj"].items()}, model.grid, NUSC_GROUPS, k=100, threshold=0.05)[0],
             "detr": list(model.boxes(res, dt, k=100, threshold=0.05, rescore=True)[0]) if refine else None,
             "vhat": p["v_hat"][0].float().cpu().numpy() if p.get("v_hat") is not None else None,
             "objvel": p["obj"]["vel"][0].float().cpu().numpy() if "vel" in p["obj"] else None}
        if getattr(model, "spec_1m", None):                # the 1 m centre-line head: its presence is a layer of `prob`, its direction goes to `dir`
            p1, o["dir"] = predictions_1m(p, 0, model.spec_1m)
            prob.update(p1)
        out[dt] = o
    return {"dt": out}


def memory_warnings(conf):
    """{scene: [mean memory confidence per frame]} -> warnings. The memory must be empty at the first frame of a scene (it was initialised) and must grow afterwards (it is used)"""
    out = []
    for scene, c in conf.items():
        if c[0] > MEMORY_EMPTY:
            out.append(f"{scene}: the memory is not empty at the first frame (confidence {c[0]:.3f})")
        if len(c) > 1 and c[-1] <= c[0] + 1e-6:
            out.append(f"{scene}: the memory confidence does not grow ({c[0]:.3f} -> {c[-1]:.3f}): the memory is not used")
    return out


def file_sha(path, n=12):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()[:n]


def code_state(code):
    """the run's own snapshot is fixed by the run; live code is named by the repository's commit (+dirty when it has uncommitted changes)"""
    rel = os.path.relpath(code, ROOT)
    if rel.startswith("runs" + os.sep):
        return rel
    git = lambda *a: subprocess.run(["git", "-C", ROOT, *a], capture_output=True, text=True).stdout.strip()      # noqa: E731
    return f"{rel}@{git('rev-parse', '--short', 'HEAD')}" + ("+dirty" if git("status", "--porcelain", "--", rel) else "")


# ---- the class
class Runner:
    """runs a trained network over whole scenes and stores result sets under <run>/results/<id>/"""

    def __init__(self, project=ROOT):
        self.project = os.path.abspath(project)

    @contextlib.contextmanager
    def _in_project(self):
        """the data paths are relative to the project folder (as in bev_runner.main)"""
        here = os.getcwd()
        os.chdir(self.project)
        try:
            yield
        finally:
            os.chdir(here)

    def _prepare(self, run, frames, split, dts, taps, code):
        """everything the id depends on, without the network: the data is opened, the frames are resolved"""
        import bev_runner as BR
        run = os.path.abspath(os.path.join(self.project, run))
        if not os.path.isfile(os.path.join(run, "config.json")):
            raise RunnerError(f"{run} is no run folder (no config.json)")
        if split not in ("val", "train"):
            raise RunnerError(f"split is val or train, not {split!r}")
        bad = [t for t in taps if t not in TAPS]
        if bad:
            raise RunnerError(f"unknown taps {bad}; the taps are: {', '.join(TAPS)}")
        ckpts = sorted(f for f in os.listdir(run) if re.fullmatch(r"ckpt_\d+\.pt", f))
        if not ckpts:
            raise RunnerError(f"no checkpoint in {run}")
        items = parse_frames(frames)
        try:
            codedir = BR.use_code(run, os.path.join(self.project, code) if code else "")
        except SystemExit as e:                            # use_code refuses a snapshot from before the five-layer build with a hint
            raise RunnerError(str(e))
        loaded = sys.modules.get("train")
        if loaded is not None and os.path.dirname(os.path.abspath(loaded.__file__)) != codedir:
            raise RunnerError(f"this process has loaded the code of {os.path.dirname(loaded.__file__)} already: use one process per code folder")
        for p in (ROOT, AUTORESEARCH):                     # the evaluation's helpers and the taps; the code folder keeps the first place
            if p not in sys.path:
                sys.path.append(p)
        cfg, ds, item = open_data(os.path.join(run, "config.json"), split)
        rows = plan_frames(items, scene_index(ds))
        frames_ = [{"scene": r["scene"], "index": r["index"], "token": ds.samples[r["di"]]} for r in rows]
        with open(os.path.join(run, "config.json"), "rb") as f:
            config_sha = hashlib.sha256(f.read()).hexdigest()[:12]
        identity = {"network": os.path.basename(run), "checkpoint": ckpts[-1], "checkpoint_sha": file_sha(os.path.join(run, ckpts[-1])), "step": int(re.findall(r"\d+", ckpts[-1])[0]),
                    "weights": "EMA", "code": code_state(codedir), "config": config_sha, "split": split, "memory": "carried"}
        rid, identity = RS.make_id(identity, [f["token"] for f in frames_], dts, taps)
        root = os.path.join(run, "results")
        return SimpleNamespace(run=run, cfg=cfg, item=item, rows=rows, frames=frames_, identity=identity, id=rid, root=root, final=os.path.join(root, rid),
                               dts=[float(d) for d in dts], taps=list(taps))

    def check(self, run, frames, split="val", dts=DTS, taps=(), code=""):
        """the frames, the id and whether that result set exists; the network is not built and nothing is written (the data is opened: a minute or so)
        -> {ok, id, folder, exists, network, step, frames, scenes, dts, taps}"""
        with self._in_project():
            p = self._prepare(run, frames, split, dts, taps, code)
        return {"ok": True, "id": p.id, "folder": p.final, "exists": os.path.isdir(p.final), "network": p.identity["network"], "step": p.identity["step"],
                "frames": len(p.frames), "scenes": len({f["scene"] for f in p.frames}), "dts": p.dts, "taps": sorted(p.taps)}

    def run(self, run, frames, split="val", dts=DTS, taps=(), code="", device=None):
        """run the network over the scenes and store the result set; the same content is never made twice (an existing set is returned) -> summary + {existed, seconds, warnings}"""
        with self._in_project():
            p = self._prepare(run, frames, split, dts, taps, code)
            if os.path.isdir(p.final):
                return dict(ResultSet.open(p.final).summary(), existed=True, seconds=0.0, warnings=[])
            t0 = time.time()
            rs, warnings = self._produce(p, device)
        return dict(rs.summary(), existed=False, seconds=round(time.time() - t0, 1), warnings=warnings)

    def _produce(self, p, device):
        import torch
        import bev_runner as BR
        from evalsuite.torch_taps import TapRecorder, resolve
        from evaluation.bev_adapter import bev_hooks
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        net = BR.Network(p.run, device, p.dts)             # warm: it carries the state from keyframe to keyframe and starts cold by itself where a scene begins or a frame is missing
        hooks = bev_hooks()
        missing = [t for t in p.taps if t != "trunk" and resolve(net.model, hooks[t].path) is None]
        if missing:
            raise RunnerError(f"this network has no {missing}")
        held = {}
        net.model.register_forward_hook(lambda m, i, o: held.update(res=o))
        rec = TapRecorder(net.model, hooks)
        arm = sorted({t for t in p.taps if t != "trunk"} | {"memory_conf"})
        writer = RS.ResultWriter(p.root, {k: v for k, v in p.identity.items() if k not in ("frames", "n_frames")}, p.frames, p.dts, p.taps)
        conf, scene, last = {}, None, time.time()
        for r, f in zip(p.rows, p.frames):
            batch, row = p.item(r["di"])
            rec.arm(arm)
            net(batch, row)
            res = held["res"]
            arrays = rec.read(0)
            with torch.no_grad():
                outputs = frame_outputs(net.model, res, net.cfg, p.dts)
                taps = {t: (res["features"][0].float().cpu().numpy() if t == "trunk" else arrays[t]) for t in p.taps if t == "trunk" or t in arrays}
            info = {}
            if "memory_conf" in arrays:
                info["memory_conf"] = float(arrays["memory_conf"].mean())
                conf.setdefault(f["scene"], []).append(info["memory_conf"])
            writer.add(RS.frame_key(f["scene"], f["index"]), outputs, taps, info)
            if f["scene"] != scene or time.time() - last > 30:
                scene, last = f["scene"], time.time()
                print(f"runner: {f['scene']} frame {f['index']}", file=sys.stderr, flush=True)
        return writer.finish(), memory_warnings(conf)

    def _folders(self):
        root = os.path.join(self.project, "runs")
        found = [os.path.join(root, run, "results", rid) for run in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, run, "results"))
                 for rid in sorted(os.listdir(os.path.join(root, run, "results")))]
        return [d for d in found if os.path.isfile(os.path.join(d, "result.json"))]

    def open(self, id_or_folder):
        """a ResultSet by its id (a unique start of it is enough) or its folder"""
        if os.path.isdir(id_or_folder):
            return ResultSet.open(id_or_folder)
        hits = [d for d in self._folders() if os.path.basename(d).startswith(id_or_folder)]
        if len(hits) != 1:
            raise RunnerError(f"{id_or_folder!r}: {'no result set' if not hits else f'{len(hits)} result sets'} of that id under runs/*/results/")
        return ResultSet.open(hits[0])

    def list(self):
        """the summaries of all stored result sets, oldest first"""
        return sorted((ResultSet.open(d).summary() for d in self._folders()), key=lambda s: s["created"])

    def get(self, id_or_folder, verify=False):
        """one result set: summary + identity (+ with verify every frame read and checked: `problems`)"""
        rs = self.open(id_or_folder)
        out = dict(rs.summary(), identity=rs.identity)
        if verify:
            out["problems"] = rs.check()
        return out


# ---- the command line
def headline(d):
    """one line for every dict that describes a result set (summary, check, run, get)"""
    dts = " ".join(f"{x:+g}" for x in d["dts"])
    return f"{d['id']}  {d['network']} step {d['step']}  {d['frames']} frames, {d['scenes']} scenes  dts {dts}  taps {','.join(d['taps']) or '-'}"


def text(cmd, d):
    """the short text a command prints (without --json)"""
    if cmd == "taps":
        return "\n".join(d)
    if cmd == "list":
        return "\n".join(f"{headline(s)}  {s['created']}" for s in d) or "no result sets"
    if cmd == "get":
        lines = [headline(d), f"  {d['folder']}", *(f"  {k}: {v}" for k, v in d["identity"].items())]
        if "problems" in d:
            lines += [f"  PROBLEM {x}" for x in d["problems"]] or ["  sound"]
        return "\n".join(lines)
    if cmd == "check":
        status = "exists" if d["exists"] else "would be new"
    else:
        status = "existed already" if d["existed"] else f"{d['seconds']} s"
    return f"{headline(d)}  {status}  -> {d['folder']}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("check", "run"):
        s = sub.add_parser(name, help=("plan: the frames and the id, nothing is run" if name == "check" else "run the network and store the result set"))
        s.add_argument("run", help="a run folder, e.g. runs/t20_base_s0")
        s.add_argument("--frames", required=True, help="scene-0003 | scene-0003:3 | all | random:5@1, comma-separated")
        s.add_argument("--split", default="val", choices=["val", "train"])
        s.add_argument("--dts", type=float, nargs="+", default=list(DTS))
        s.add_argument("--taps", default="", help="comma-separated: " + ", ".join(TAPS))
        s.add_argument("--code", default="", help="a code folder instead of <run>/code, e.g. training")
        if name == "run":
            s.add_argument("--device", default=None, help="cuda (default when there is one) or cpu")
    sub.add_parser("list", help="the stored result sets")
    g = sub.add_parser("get", help="one result set")
    g.add_argument("id", help="its id (a unique start is enough) or its folder")
    g.add_argument("--verify", action="store_true", help="read every frame and check it")
    sub.add_parser("taps", help="the names of the taps")
    for s in sub.choices.values():
        s.add_argument("--json", action="store_true", help="print exactly what the method returns")
    a = ap.parse_args(argv)
    r = Runner()
    taps = [t for t in getattr(a, "taps", "").split(",") if t] if a.cmd in ("check", "run") else []
    try:
        if a.cmd == "check":
            d = r.check(a.run, a.frames, a.split, a.dts, taps, a.code)
        elif a.cmd == "run":
            d = r.run(a.run, a.frames, a.split, a.dts, taps, a.code, a.device)
        elif a.cmd == "list":
            d = r.list()
        elif a.cmd == "get":
            d = r.get(a.id, a.verify)
        else:
            d = list(TAPS)
    except RunnerError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(json.dumps(d, indent=1) if a.json else text(a.cmd, d))
    for w in (d.get("warnings", []) if isinstance(d, dict) else []):
        print(f"warning: {w}", file=sys.stderr)
    return 1 if isinstance(d, dict) and d.get("problems") else 0


if __name__ == "__main__":
    sys.exit(main())
