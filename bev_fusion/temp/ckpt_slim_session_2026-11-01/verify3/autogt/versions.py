"""GT versions (decision D29, 2026-11-01): every change of the generator signature is a version vN.

The signature (10 hex chars over the five files grid.py, layers.py, flow.py, nuscenes_adapter.py, nuscenes_gt.py and the options) names the
folder of the stored GT; the version is its human key. A version is: a row in versions.csv (next to this file), a git tag `autogt/vN`
on the commit that holds that generator, and the key `gt_version` in the index.json of the stored GT. The rule when you edit one of the
five files: the signature changes, `python -m autogt selftest` fails until versions.csv has a row for the new signature, then commit
and tag (`git tag -a autogt/v6 -m "..."`), rebuild the GT, tell the others. An old GT is rebuilt from its tag:
`git worktree add temp/gt_v3 autogt/v3`, then generate with the package of that checkout (docs: README.md, section Versions).

    autogt.versions.table()          the rows (dicts)
    autogt.versions.version_of(sig)  "v5" or None; autogt.current_version() the version of THIS source with the default options
    python -m autogt versions        the table, the current version marked, which GT is stored
"""
import csv
import os

FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "versions.csv")


def table():
    """the rows of versions.csv, oldest first"""
    with open(FILE, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def version_of(signature):
    """the version key of a generator signature (default options), or None when it is not a released version"""
    return next((r["version"] for r in table() if r["signature"] == signature), None)


def row_of(version):
    return next((r for r in table() if r["version"] == version), None)
