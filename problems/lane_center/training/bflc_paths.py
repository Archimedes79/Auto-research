"""Paths of the project, derived from where this checkout lives: the repository folder can be renamed or moved, no path is typed anywhere.

    import bflc_paths as BP            # flat import like every module (PYTHONPATH comes from tools/env/bflc_env.sh)
    BP.ROOT     the problem folder (contains project.json, runs/, models/, training/ ...)
    BP.REPO     its parent (contains autoresearch/ and this problem folder)
    BP.RUNS     runs/        BP.DATA   data/        BP.REPORTS   reports/

The environment variable BFLC_ROOT (exported by tools/env/_py14.sh) overrides the location of the file.
Self-test: `tools/env/_py14.sh bflc_paths.py`.
"""
import os

ROOT = os.path.normpath(os.environ.get("BFLC_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
REPO = os.path.dirname(ROOT)
RUNS = ROOT + "/runs"
DATA = ROOT + "/data"
REPORTS = ROOT + "/reports"

if __name__ == "__main__":
    assert os.path.isfile(ROOT + "/project.json"), f"ROOT is not the problem folder: {ROOT}"
    assert os.path.isdir(REPO + "/autoresearch"), f"REPO has no autoresearch/: {REPO}"
    assert os.path.abspath(__file__).startswith(ROOT), "bflc_paths.py is not inside ROOT"
    print("bflc_paths selftest ok:", ROOT)
