"""The lane suite of this project: the general evaluation suite (../../autoresearch/evalsuite) on the lane-centre-line data of VAL.

    tools/env/_py14.sh lane_suite.py <system> <command ...>
    tools/env/_py14.sh lane_suite.py dense:e37_final qual random 6 --seed 1          # 6 random frames: a video first, then a frame
    tools/env/_py14.sh lane_suite.py dense:e37_final quant run                       # numbers over all 873 frames, stored per frame
    tools/env/_py14.sh lane_suite.py dense:e37_final quant table <run> --all         # picture level first (img.*), then the graph read from it (geo.*, topo.*)
    tools/env/_py14.sh lane_suite.py dense:e37_final probe scan <run>                # where do the errors come from: budget, situations, random examples
    tools/env/_py14.sh lane_suite.py graphs:champion quant run                       # a registry model: sparse graphs + the project's own S / Q / topo numbers (lcr.*)
    tools/env/_py14.sh lane_suite.py selftest [--quick]                              # alignment, orientation, the project's numbers reproduced

systems
  dense:<run>        stored probability maps of a dense head (runs/<run>/prob_val.npy, ids_val.npy): scored as a PICTURE first (img.*), then as the
                     graph read from the thinned picture (geo.*, topo.*)
  graphs:<registry>  a model of models/registry.json: its cached network outputs assembled to graphs on the CPU (evaluate.graphs_of): scored as GRAPHS
                     (geo.*, topo.*, and lcr.* = the numbers of evaluate.py: S, Q, J, vertices per 100 m ...) and drawn as a picture (img.*)
commands: everything of the evalsuite (list, qual frame | debug | random | videos, quant run | table | per-scene | outliers | report | trace | compare,
probe budget | situations | sample | scan); global options --out DIR (default runs/suite/<system>) --state. VAL only: the TEST split is used through
evaluate.py / run.py with its log entry.

Frames: `scene-NN/index` (the index is the position in the video, frames of a scene in time order); the key is `row<N>` (the row of the split
file). Coordinates: ego frame, u = x to the right, v = y forward, 200 x 200 cells of 0.5 m, evaluated |x|, |y| < 48 m like the project's metrics.
No camera picture yet: the ground-truth file carries no sample tokens.
"""
import bflc_paths as BP  # project paths from the location of this checkout
import importlib.util
import json
import os
import pickle
import sys
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, os.path.join(BP.REPO, "autoresearch"))
from evalsuite.contracts import FrameRef, Metric, Prediction, Sample, Source, Subject  # noqa: E402
from evalsuite.lanes import Grid, LaneConfig, LaneGraph, LaneTask, rasterize  # noqa: E402

HOME = os.path.expanduser("~")
B123 = f"{HOME}/exp_gcar_b123"
GRID = Grid(u0=-50.0, v0=-50.0, res=0.5, nu=200, nv=200)
BOX = (-48.0, 48.0, -48.0, 48.0)


def config() -> LaneConfig:
    return LaneConfig(GRID, taus=(0.5, 1.0, 1.5), ds=0.5, box=BOX, thr=0.5, clip=5.0, far_m=40.0, origin=(0.0, 0.0))


def _evaluate_module():
    """evaluation/evaluate.py (registry, graphs_of) loaded by path: the name 'evaluate' is also a popular package"""
    spec = importlib.util.spec_from_file_location("lcr_evaluate", f"{BP.ROOT}/evaluation/evaluate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------------------------------------- Source
class LaneSource(Source):
    """the VAL frames of the project grouped into scenes in time order; ground truth = the graph of the row"""

    def __init__(self, split: str = "val"):
        if split != "val":
            raise SystemExit("the lane suite reads VAL only; TEST goes through evaluate.py / run.py with its TEST_USAGE_LOG entry")
        self.split = split
        sp = json.load(open(f"{BP.ROOT}/data/splits.json"))
        self.rows = np.asarray(sp[split], int)
        pz = np.load(f"{B123}/poses_{split}.npz")
        self.pz = {k: pz[k] for k in ("scene", "ts", "t", "prev", "dt")}
        by: Dict[int, List[int]] = {}
        for r in self.rows:
            by.setdefault(int(self.pz["scene"][r]), []).append(int(r))
        self.by_scene = {f"scene-{s:02d}": sorted(v, key=lambda r: self.pz["ts"][r]) for s, v in sorted(by.items())}
        self._gt = None

    @property
    def gt(self):
        if self._gt is None:
            self._gt = pickle.load(open(f"{B123}/graph_gt_{self.split}.pkl", "rb"))
        return self._gt

    def scenes(self) -> List[str]:
        return list(self.by_scene)

    def frames(self, scene: str) -> List[FrameRef]:
        return [FrameRef(scene, i, key=f"row{r}") for i, r in enumerate(self.by_scene[scene])]

    def describe(self) -> Dict[str, Any]:
        return {"class": "LaneSource", "split": self.split, "frames": int(len(self.rows)), "scenes": len(self.by_scene)}

    def speed(self, row: int) -> float:
        p = int(self.pz["prev"][row])
        dt = float(self.pz["dt"][row])
        return float(np.linalg.norm(self.pz["t"][row] - self.pz["t"][p]) / dt) if p >= 0 and dt > 0 else float("nan")

    def load(self, ref: FrameRef) -> Sample:
        row = self.by_scene[ref.scene][ref.index]
        raw = self.gt[row]
        g = LaneGraph.of({"pts": raw["pts"], "edges": raw.get("edges")})
        return Sample(ref, inputs={"row": row}, reference={"lanes": g, "mask": rasterize(g, GRID), "raw": raw},
                      display={}, meta={"row": row, "speed": self.speed(row)})


# --------------------------------------------------------------------------------------------------------- Subjects
class DenseSubject(Subject):
    """a dense head's stored probability maps; the maps are [row = x (right), col = y (forward)], the suite wants [row = v (forward), col = u (right)]"""
    warmup = 0

    def __init__(self, run: str, split: str = "val"):
        self.run = run
        self.prob = np.load(f"{BP.ROOT}/runs/{run}/prob_{split}.npy", mmap_mode="r")
        ids = np.load(f"{BP.ROOT}/runs/{run}/ids_{split}.npy")
        self.pos = {int(r): i for i, r in enumerate(ids)}

    def reset(self) -> None:
        pass

    def taps(self):
        return []

    def describe(self) -> Dict[str, Any]:
        return {"name": f"dense_{self.run}", "class": "DenseSubject", "run": self.run}

    def predict(self, sample: Sample, taps=()) -> Prediction:
        return Prediction({"prob": np.ascontiguousarray(np.asarray(self.prob[self.pos[sample.inputs["row"]]], np.float32).T)})


class GraphSubject(Subject):
    """a registry model's graphs (cached network outputs assembled on the CPU, or the stored graphs of a system that is not re-run)"""
    warmup = 0

    def __init__(self, name: str, split: str = "val"):
        ev = _evaluate_module()
        self.name = name
        graphs, gts, info = ev.graphs_of(name, ev.registry()[name], split, require_cache=True)
        rows = np.asarray(json.load(open(f"{BP.ROOT}/data/splits.json"))[split], int)
        assert len(graphs) == len(rows), (len(graphs), len(rows))
        self.graphs = {int(r): g for r, g in zip(rows, graphs)}
        self.gts = {int(r): g for r, g in zip(rows, gts)}
        self.info = {k: v for k, v in info.items() if isinstance(v, (int, float, str))}

    def reset(self) -> None:
        pass

    def taps(self):
        return []

    def describe(self) -> Dict[str, Any]:
        return {"name": f"graphs_{self.name}", "class": "GraphSubject", "registry": self.name, **self.info}

    def predict(self, sample: Sample, taps=()) -> Prediction:
        g = self.graphs[sample.inputs["row"]]
        return Prediction({"lanes": LaneGraph.of(g), "raw": g})


# --------------------------------------------------------------------------------------------------------- the project's own metrics, additive
class LcrMetric(Metric):
    """the numbers of evaluate.py (sparse_lanes/bootstrap.frame_records + pooled: S, Q, J, F1@t, lanes found, vertices per 100 m, len_ratio ...):
    the per-frame records are stored as arrays, any set of frames is pooled again by the project's own function"""
    name = "lcr"
    headline = ("Q", "S", "F1@1.0", "J", "lanes_found", "vert100")
    better = {"vert100": "low"}

    def _sb(self):
        import bootstrap as SB
        return SB

    def frame(self, sample, pred):
        rec = self._sb().frame_records([pred.outputs["raw"]], [sample.reference["raw"]])[0]
        rs = rec["rs"]
        misc = [rs["n_p"], rs["n_g"], rs.get("lanes_found", 0), rs.get("lanes_total", 0), rs.get("false_len", 0.0), rs.get("pred_len", 0.0),
                rs.get("dir_ok", 0), rs.get("dir_n", 0), rec["nv"], rec["length"]]
        return {"cnt": np.asarray(rs["cnt"], float), "m1": np.asarray(rs["m1"], float), "misc": np.asarray(misc, float),
                "junc": np.asarray([rs[k] for k in ("spl", "mrg", "spl6", "mrg6")], float)}

    def reduce(self, counts, keys=None):
        if not counts:
            return {}
        recs = []
        for c in counts:
            m = c["misc"]
            rs = dict(cnt=c["cnt"], m1=c["m1"], n_p=m[0], n_g=m[1], lanes_found=m[2], lanes_total=m[3], false_len=m[4], pred_len=m[5], dir_ok=m[6], dir_n=m[7],
                      spl=tuple(c["junc"][0]), mrg=tuple(c["junc"][1]), spl6=tuple(c["junc"][2]), mrg6=tuple(c["junc"][3]))
            recs.append(dict(rs=rs, nv=m[8], length=m[9]))
        return {k: float(v) for k, v in self._sb().pooled(recs).items()}

    def score(self, counts):
        c = counts["cnt"][1]
        p, r = (c[0] / c[1] if c[1] else 1.0), (c[2] / c[3] if c[3] else 1.0)
        return {"F1@1.0": 2 * p * r / (p + r) if p + r else 0.0}

    def support(self, counts):
        return {"F1@1.0": float(counts["cnt"][1][3])}


class LcrTopoMetric(Metric):
    """the connectivity numbers of evaluate.py (topo_metrics: topo P / R / F1, duplicate share, recall / precision by range)"""
    name = "lcr_topo"
    headline = ("topo_F1", "R@1_far", "dup_frac")
    better = {"dup_frac": "low"}

    def frame(self, sample, pred):
        import topo_metrics as TM
        r = TM.frame_record(pred.outputs["raw"], sample.reference["raw"])
        return {"topo": np.asarray(r["topo"], float), "dup": np.asarray(r["dup"], float), "rng": np.asarray(r["rng"], float)}

    def reduce(self, counts, keys=None):
        import topo_metrics as TM
        return {k: float(v) for k, v in TM.pooled(list(counts)).items()} if counts else {}

    def score(self, counts):
        t = counts["topo"]
        p, r = (t[0] / t[1] if t[1] else 1.0), (t[0] / t[2] if t[2] else 1.0)
        return {"topo_F1": 2 * p * r / (p + r) if p + r else 0.0}

    def support(self, counts):
        return {"topo_F1": float(counts["topo"][2])}


# --------------------------------------------------------------------------------------------------------- build
def build(system: str = "dense:e37_final", split: str = "val", **_):
    kind, _, name = system.partition(":")
    src = LaneSource(split)

    def tags(sample):
        sp = sample.meta.get("speed", float("nan"))
        return {"ego_speed": sp} if sp == sp else {}
    if kind == "dense":
        return src, DenseSubject(name, split), LaneTask(config(), extra_tags=tags)
    if kind == "graphs":
        return src, GraphSubject(name, split), LaneTask(config(), extra_tags=tags, extra_metrics=[LcrMetric(), LcrTopoMetric()])
    raise SystemExit(f"unknown system {system!r}: dense:<run> or graphs:<registry name>")


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    system = argv.pop(0)
    if system == "selftest":
        import test_lane_suite as ST
        return ST.main(argv)
    from evalsuite.cli import main as cli
    out = f"{BP.ROOT}/runs/suite/{system.replace(':', '_')}"
    return cli(["--adapter", "lane_suite:build", "--arg", f"system={system}", "--out", out] + argv)


if __name__ == "__main__":
    raise SystemExit(main())
