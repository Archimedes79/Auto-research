"""GT versions (decision D29, 2026-11-01): every change of the generator signature is a version vN.

The signature (10 hex chars over the five files grid.py, layers.py, flow.py, nuscenes_adapter.py, nuscenes_gt.py and the options) names the
folder of the stored GT; the version is its human key. A version is: a row in versions.csv (next to this file), a git tag `autogt/vN`
on the commit that holds that generator, and the key `gt_version` in the index.json of the stored GT. The rule when you edit one of the
five files: the signature changes, `python -m autogt selftest` fails until versions.csv has a row for the new signature, then commit
and tag (`git tag -a autogt/v6 -m "..."`), rebuild the GT, tell the others. An old GT is rebuilt from its tag:
`git worktree add temp/gt_v5 autogt/v5`, then run `build` of the package in that checkout (the old tags have no `generate`; exact
command and the check that it is the checkout's package that runs: README.md, section Versions).

    autogt.versions.table()          the rows (dicts)
    autogt.versions.version_of(sig)  "v5" or None; autogt.current_version() the version of THIS source with the default options
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
NOT_GENERATOR = ("__init__.py", "__main__.py", "checks.py", "clean.py", "cli.py", "light_nusc.py", "report.py",
                 "selftest.py", "sets.py", "suite.py", "versions.py", "video.py", "view.py")


def table():
    """the rows of versions.csv, oldest first"""
    with open(FILE, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def version_of(signature):
    """the version key of a generator signature (default options), or None when it is not a released version"""
    return next((r["version"] for r in table() if r["signature"] == signature), None)


def version_of_key(key):
    """the version of a GT cache key whose last part is the generator signature, such as the training's '0.5_4_15efbde2d1' or
    '0.5_x0_40_y-20_20_4_15efbde2d1' (or a bare signature): "v5", or None when it is not a released version (or has other options)"""
    return version_of(str(key).rsplit("_", 1)[-1])


def row_of(version):
    return next((r for r in table() if r["version"] == version), None)


def code_state():
    """{"commit": short hash of HEAD, "dirty": True when anything in autogt/ differs from it (edited or new files)}; both None when
    git is not available here. The manifest of a stored GT records it, so a GT made from a clean tree can be told from a scratch one."""
    import subprocess
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        head = subprocess.run(["git", "-C", here, "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=30)
        status = subprocess.run(["git", "-C", here, "status", "--porcelain", "--", "."], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}
    if head.returncode or status.returncode:
        return {"commit": None, "dirty": None}
    return {"commit": head.stdout.strip(), "dirty": bool(status.stdout.strip())}
