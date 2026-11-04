"""GT versions (decision D29, 2026-11-01): every change of the generator signature is a version vN.

The signature (10 hex chars over the six files grid.py, layers.py, flow.py, nuscenes_adapter.py, nuscenes_gt.py and the options) names the
folder of the stored GT; the version is its human key. A version is: a row in versions.csv (next to this file), a git tag `autogt/vN`
on the commit that holds that generator, and the key `gt_version` in the index.json of the stored GT. The rule when you edit one of the
six files: the signature changes, `python -m autogt selftest` fails until versions.csv has a row for the new signature, then commit
and tag (`git tag -a autogt/v6 -m "..."`), rebuild the GT, tell the others. An old GT is rebuilt from its tag:
`git worktree add temp/gt_v5 autogt/v5`, then run `build` of the package in that checkout (the old tags have no `generate`; exact
command and the check that it is the checkout's package that runs: README.md, section Versions).

    autogt.versions.table()          the rows (dicts)
    autogt.versions.version_of(sig)  "v5" or None; autogt.current_version() the GT version of THIS source with the default options
    autogt.versions.canonical(sig)   the signature of the store that `sig`'s code fills (`same_gt_as`: v9 makes the GT of v8, so it uses its store)
    autogt.versions.version_of_key(k) the same for a training cache key '0.5_4_<signature>' (what runs/<name>/gt.json and the cards record)
    python -m autogt versions        the table, the current version marked, which GT is stored
    autogt.versions.code_state()     {"commit", "dirty"} of the generator code, recorded in the manifest of a stored GT (index.json)
"""
import csv
import os

FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "versions.csv")

# the modules of this package that do NOT shape the GT (command line, storage, reports, pictures, checks, scene sets, the light devkit loader,
# this table). Every other module is hashed into the signature (nuscenes_gt.SIGNATURE_FILES); `python -m autogt selftest` fails for a module
# that is in neither list, and for a hashed module that nuscenes_gt.py does not import, so a change of the generator cannot slip past a version.
# the lane GT (lane_gt.py, lane_graph.py, road_graph.py, lane_lines.py) has its own version (lane_versions.csv), not the front GT's signature
NOT_GENERATOR = ("__init__.py", "__main__.py", "checks.py", "clean.py", "cli.py", "lane_gt.py", "lane_graph.py", "lane_graph_selftest.py", "lane_graph_view.py",
                 "lane_lines.py", "light_nusc.py", "report.py", "road_graph.py", "selftest.py", "sets.py", "suite.py", "versions.py", "video.py", "view.py")


def table():
    """the rows of versions.csv, oldest first"""
    with open(FILE, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def version_of(signature):
    """the version key of a generator signature (default options), or None when it is not a released version"""
    return next((r["version"] for r in table() if r["signature"] == signature), None)


def canonical(signature):
    """the signature of the stored GT that the code of `signature` makes: a version whose code makes the same GT byte for byte as an older one
    (faster or tidier code, column `same_gt_as` of versions.csv) shares that version's store; any other signature is its own"""
    row = next((r for r in table() if r["signature"] == signature), None)
    while row and row.get("same_gt_as"):
        row = row_of(row["same_gt_as"])
    return row["signature"] if row else signature


def version_of_key(key):
    """the GT version of a GT cache key whose last part is the generator signature, such as the training's '0.5_4_15efbde2d1' or
    '0.5_x0_40_y-20_20_4_15efbde2d1' (or a bare signature): "v5", or None when it is not a released version (or has other options). A
    version that makes the same GT as an older one gives the older key: the key names the GT, the signature the code."""
    return version_of(canonical(str(key).rsplit("_", 1)[-1]))


def row_of(version):
    return next((r for r in table() if r["version"] == version), None)


def code_state():
    """{"commit": short hash of HEAD, "dirty": True when anything in autogt/ differs from it (edited or new files)}; both None, with a warning,
    when git cannot tell. The manifest of a stored GT records it, so a GT made from a clean tree can be told from a scratch one. A worktree
    that Windows git made has a path that WSL git cannot read: the Windows git (git.exe) is tried then."""
    import shutil
    import subprocess
    here = os.path.dirname(os.path.abspath(__file__))
    attempts = [("git", here)]
    if shutil.which("git.exe"):
        win = subprocess.run(["wslpath", "-w", here], capture_output=True, text=True).stdout.strip()
        attempts.append(("git.exe", win or here))
    for exe, where in attempts:
        try:
            head = subprocess.run([exe, "-C", where, "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=30)
            status = subprocess.run([exe, "-C", where, "status", "--porcelain", "--", "."], capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.SubprocessError):
            continue
        if not (head.returncode or status.returncode):
            return {"commit": head.stdout.strip(), "dirty": bool(status.stdout.strip())}
    import sys
    print("WARNING: git cannot tell the commit and the state of autogt/ here: the manifest gets no commit and the store is no release; "
          "run `python -m autogt generate` again from a clean checkout to stamp it", file=sys.stderr, flush=True)
    return {"commit": None, "dirty": None}
