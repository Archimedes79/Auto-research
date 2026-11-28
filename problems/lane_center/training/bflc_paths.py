"""Where the lane problem's code, data and runs are. The code is this checkout; data and run outputs are where the machine says (a run of the research
system gets RESEARCH_DATA / RESEARCH_RUNS from its worker; by hand they come from the machine file, ~/.research/machine.json or RESEARCH_MACHINE).

    import bflc_paths as BP            # flat import like every module of this problem (the entry scripts put training/ on the path)
    BP.ROOT     the problem folder (problems/lane_center)          BP.REPO     the repository (autoresearch/, GtGenerator/, problems/)
    BP.RUNS     the problem's run root (old run folders keep their names below it: runs/e292_a_s0 -> RUNS/e292_a_s0)
    BP.DATA     the problem's data: B123 (the GaussianCaR features and the old graph GT, the former ~/exp_gcar_b123)
    BP.NUSC     the raw nuScenes folder (the former ~/data/nuscenes_full)        BP.SCRATCH  scratch outputs of diagnostics (below RUNS, never a record)
    BP.file_sha256(path)   the sha256 of a file
    BP.resolve(p)   an absolute path as it is; "runs/x" -> RUNS/x (the form stored in old checkpoints and caches); else relative to ROOT
"""
import hashlib
import json
import os

ROOT = os.path.normpath(os.environ.get("BFLC_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))   # BFLC_ROOT: a test points a moved copy at the problem
REPO = os.path.dirname(os.path.dirname(ROOT))


def _machine():
    path = os.path.expanduser(os.environ.get("RESEARCH_MACHINE") or "~/.research/machine.json")
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


_M = _machine()
_DATA_ROOT = os.environ.get("RESEARCH_DATA") or _M.get("data") or os.path.join(REPO, "data")
RUNS = os.environ.get("RESEARCH_RUNS") or os.path.join(_M.get("runs") or os.path.join(REPO, "runs"), "lane_center")
DATA = os.path.join(_DATA_ROOT, "lane_center")
B123 = os.path.join(DATA, "b123")
NUSC = os.path.join(_DATA_ROOT, "nuscenes")
SCRATCH = os.path.join(RUNS, "_scratch")


def resolve(p):
    """a path of a record or an argument: absolute stays; 'runs/...' is below RUNS; anything else is relative to the problem folder"""
    p = str(p)
    if os.path.isabs(p):
        return p
    if p == "runs" or p.startswith("runs/"):
        return os.path.join(RUNS, p[5:]) if len(p) > 4 else RUNS
    return os.path.join(ROOT, p)


def file_sha256(path):
    """the sha256 of a file (hex): the identity of a checkpoint in a recipe, of the B' a token cache was made from, of a stored dump"""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


if __name__ == "__main__":
    assert os.path.isfile(os.path.join(ROOT, "problem.json")), f"ROOT is not the problem folder: {ROOT}"
    assert os.path.isdir(os.path.join(REPO, "autoresearch")), f"REPO has no autoresearch/: {REPO}"
    assert resolve("runs/e292_a_s0") == os.path.join(RUNS, "e292_a_s0") and resolve("/x/y") == "/x/y" and resolve("data/splits.json") == os.path.join(ROOT, "data", "splits.json")
    print("bflc_paths ok:", ROOT, RUNS, DATA)
