"""GT versions (decision D29, 2026-11-01): every change of the generator signature is a version vN.

The signature (10 hex chars over the six files grid.py, layers.py, flow.py, nuscenes_adapter.py, nuscenes_gt.py, coverage.py and the options) names the
folder of the stored GT; the version is its human key. A version is: a row in versions.csv (next to this file), a git tag `autogt/vN`
on the commit that holds that generator, and the key `gt_version` in the index.json of the stored GT. The rule when you edit one of the
six files: the signature changes, `python -m GtGenerator selftest` fails until versions.csv has a row for the new signature, then commit
and tag (`git tag -a autogt/v6 -m "..."`), rebuild the GT, tell the others. An old GT is rebuilt from its tag:
`git worktree add temp/gt_v5 autogt/v5`, then run `build` of the package in that checkout (the old tags have no `generate`; exact
command and the check that it is the checkout's package that runs: README.md, section Versions).

    GtGenerator.versions.table()          the rows (dicts)
    GtGenerator.versions.version_of(sig)  "v5" or None; GtGenerator.current_version() the GT version of THIS source with the default options
    GtGenerator.versions.canonical(sig)   the signature of the store that `sig`'s code fills (`same_gt_as`: v9 makes the GT of v8, so it uses its store)
    GtGenerator.versions.version_of_key(k) the same for a training cache key '0.5_4_<signature>' (what runs/<name>/gt.json and the cards record)
    GtGenerator.versions.releases()       the rows of releases.csv: ONE number for the whole suite (the front GT version and the lane GT version that belong together)
    GtGenerator.versions.current_release()  the release of THIS source ("v10"), or None when the pair (front version, lane version) has no row yet
    python -m GtGenerator versions        the table, the current version marked, which GT is stored
    GtGenerator.versions.code_state()     {"commit", "dirty"} of the generator code, recorded in the manifest of a stored GT (index.json)
"""
import csv
import os

FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "versions.csv")
RELEASES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "releases.csv")

# the modules of this package that do NOT shape the GT (command line, storage, reports, pictures, checks, scene sets, the light devkit loader,
# this table). Every other module is hashed into the signature (nuscenes_gt.SIGNATURE_FILES); `python -m GtGenerator selftest` fails for a module
# that is in neither list, and for a hashed module that nuscenes_gt.py does not import, so a change of the generator cannot slip past a version.
# the lane GT (lane_gt.py, lane_graph.py, road_graph.py, lane_lines.py) has its own version (lane_versions.csv), not the front GT's signature
NOT_GENERATOR = ("__init__.py", "__main__.py", "checks.py", "clean.py", "api.py", "cli.py", "pipeline.py", "store.py", "lane_gt.py", "lane_graph.py", "lane_graph_selftest.py", "lane_graph_view.py", "lane_tiers.py", "lane_tiers_selftest.py",
                 "cli_selftest.py", "generator_selftest.py", "jobs.py", "jobs_selftest.py", "gt_api.py", "gt_api_selftest.py", "lane_lines.py", "lane_raster.py", "store_audit.py", "light_nusc.py", "report.py", "rig_masks.py", "road_graph.py", "selftest.py", "sets.py", "suite.py", "versions.py", "video.py", "view.py")


_TABLES = {}


def _rows(path):
    """the rows of a csv table, read again only when the file changed (a signature lookup needs the table twice: 6 - 16 ms per GtGenerator.load otherwise)"""
    st = os.stat(path)
    key = (st.st_mtime_ns, st.st_size)
    if _TABLES.get(path, (None,))[0] != key:
        with open(path, newline="", encoding="utf-8") as f:
            _TABLES[path] = (key, list(csv.DictReader(f)))
    return _TABLES[path][1]


def table():
    """the rows of versions.csv, oldest first"""
    return [dict(r) for r in _rows(FILE)]


def version_of(signature):
    """the version key of a generator signature (default options), or None when it is not a released version"""
    return next((r["version"] for r in table() if r["signature"] == signature), None)


def canonical(signature):
    """the signature of the stored GT that the code of `signature` makes: a version whose code makes the same GT byte for byte as an older one
    (faster or tidier code, column `same_gt_as` of versions.csv) shares that version's store; any other signature is its own"""
    row, seen = next((r for r in table() if r["signature"] == signature), None), set()
    while row and row.get("same_gt_as"):
        if row["version"] in seen:
            raise ValueError(f"versions.csv: the rows with same_gt_as form a cycle at {row['version']}")
        seen.add(row["version"])
        nxt = row_of(row["same_gt_as"])
        if nxt is None:
            raise ValueError(f"versions.csv: {row['version']} is the same GT as {row['same_gt_as']}, which has no row")
        row = nxt
    return row["signature"] if row else signature


def check_table(rows=None, release_rows=None, lane_versions=None):
    """versions.csv and releases.csv are consistent: signatures are unique, every same_gt_as names an EARLIER row (no cycle, no dangling name), every release names a front version of
    versions.csv and a lane version of lane_versions.csv; raises ValueError (`GtGenerator selftest` calls it)"""
    rows = table() if rows is None else rows
    seen = []
    for r in rows:
        if r["signature"] in {x["signature"] for x in seen}:
            raise ValueError(f"versions.csv: the signature {r['signature']} is in more than one row")
        if r.get("same_gt_as") and r["same_gt_as"] not in {x["version"] for x in seen}:
            raise ValueError(f"versions.csv: {r['version']} says same_gt_as {r['same_gt_as']}, which is no EARLIER row")
        seen.append(r)
    if release_rows is None:
        release_rows = releases()
    if lane_versions is None:
        from . import lane_gt
        lane_versions = {x["version"] for x in lane_gt.table()}
    front = {r["version"] for r in rows}
    for r in release_rows:
        if r["front"] not in front or r["lane"] not in lane_versions:
            raise ValueError(f"releases.csv: {r['release']} names front {r['front']} / lane {r['lane']}, which is not a row of versions.csv / lane_versions.csv")
    if len({r["release"] for r in release_rows}) != len(release_rows):
        raise ValueError("releases.csv: a release number is used twice")


def version_of_key(key):
    """the GT version of a GT cache key whose last part is the generator signature, such as the training's '0.5_4_15efbde2d1' or
    '0.5_x0_40_y-20_20_4_15efbde2d1' (or a bare signature): "v5", or None when it is not a released version (or has other options). A
    version that makes the same GT as an older one gives the older key: the key names the GT, the signature the code."""
    return version_of(canonical(str(key).rsplit("_", 1)[-1]))


def releases():
    """the rows of releases.csv, oldest first: release, front (a version of versions.csv), lane (a version of lane_versions.csv), date, what_changed.
    A release is the suite as one: `python -m GtGenerator generate` makes the standard nuScenes GT (front) and the lane GT together, and the pair carries one
    number. The numbers are one running sequence with the front versions (v10 = front v9 + lane l3): every later change of the front GT or of the lane GT is the
    next number (v11, ...), so the next front row of versions.csv is v11, not v10."""
    return [dict(r) for r in _rows(RELEASES)]


def release_of(front, lane):
    """the release of a pair (front GT version, lane GT version), or None"""
    return next((r["release"] for r in releases() if r["front"] == front and r["lane"] == lane), None)


def current_release():
    """the release of THIS source: its front version (default options) and its lane version"""
    from . import lane_gt
    from .suite import code_signature
    return release_of(version_of(code_signature()), lane_gt.version_of(lane_gt.code_signature()))      # the version of the code (v9 / l5), not of its store (v8 / l4)


def row_of(version):
    return next((r for r in table() if r["version"] == version), None)


# ---------------------------------------------------------------------------------------------------------------- new rows, derived from git: nobody types the tables
def _number(key):
    """"v12" -> 12 (the front versions and the releases are one running sequence)"""
    return int(key[1:])


def last_change(files, cwd=None):
    """{"date", "subject"} of the last commit that touched `files` (git log), "(uncommitted: describe the change)" as the subject while they differ from HEAD; the date is today's then"""
    import datetime
    import subprocess
    here = cwd or os.path.dirname(os.path.abspath(__file__))
    today = datetime.date.today().isoformat()
    try:
        dirty = subprocess.run(["git", "-C", here, "status", "--porcelain", "--"] + list(files), capture_output=True, text=True, timeout=60).stdout.strip()
        log = subprocess.run(["git", "-C", here, "log", "-1", "--format=%cs%n%s", "--"] + list(files), capture_output=True, text=True, timeout=60).stdout.splitlines()
    except (OSError, subprocess.SubprocessError):
        return {"date": today, "subject": "(describe the change)"}
    if dirty or len(log) < 2:
        return {"date": today, "subject": "(uncommitted: describe the change)"}
    return {"date": log[0], "subject": log[1]}


def propose(signature, lane_version, change, versions_rows=None, release_rows=None):
    """the rows that a source with this front `signature` and this `lane_version` still needs: {"front": a row of versions.csv or None, "release": a row of releases.csv or None}. The number is the next of the
    running sequence (front versions and releases together), the text and the date are those of the last commit that touched the hashed files (`change`, see last_change). What stays by hand: `same_gt_as`
    (it needs the pinned digests, front_digests.json) and the lane versions (lane_versions.csv belongs to the lane chat)."""
    vrows = table() if versions_rows is None else versions_rows
    rrows = releases() if release_rows is None else release_rows
    nxt = max([_number(r["version"]) for r in vrows] + [_number(r["release"]) for r in rrows]) + 1
    out = {"front": None, "release": None}
    front = next((r["version"] for r in vrows if r["signature"] == signature), None)
    if front is None:
        front = f"v{nxt}"
        out["front"] = {"version": front, "signature": signature, "tag": f"autogt/{front}", "date": change["date"], "generator": f"GtGenerator/ (tag autogt/{front})",
                        "what_changed": change["subject"], "same_gt_as": ""}
        nxt += 1
    if lane_version and not any(r["front"] == front and r["lane"] == lane_version for r in rrows):
        out["release"] = {"release": f"v{nxt}", "front": front, "lane": lane_version, "date": change["date"], "what_changed": change["subject"]}
    return out


def write_rows(proposal, versions_file=None, releases_file=None):
    """append the proposed rows to the csv tables (the header of each file names the columns; the files are in git: review the diff, commit, tag)"""
    for key, path in (("front", versions_file or FILE), ("release", releases_file or RELEASES)):
        row = proposal.get(key)
        if not row:
            continue
        with open(path, newline="", encoding="utf-8") as f:
            text = f.read()
        fields = next(csv.reader([text.splitlines()[0]]))
        with open(path, "a", newline="", encoding="utf-8") as f:
            if not text.endswith("\n"):
                f.write("\n")
            csv.DictWriter(f, fieldnames=fields, lineterminator="\n").writerow(row)


_CODE_STATE = {}
DOCS = (".md", ".svg")                                    # a change of these cannot change a GT: the text, and the picture drawn from architecture.json


def _changes_the_code(line):
    """one line of `git status --porcelain`: does it name a file that could change the GT (not a document, not the drawing's spec)?"""
    path = line[3:].split(" -> ")[-1].strip().strip('"')
    name = os.path.basename(path)
    return not (name.endswith(DOCS) or name == "architecture.json")


def code_state(max_age=20.0):
    """{"commit": short hash of HEAD, "dirty": True when a file of GtGenerator/ other than a document (.md, .svg, architecture.json) differs from it (edited or new files)}; both None, with a warning,
    when git cannot tell. The manifest of a stored GT records it, so a GT made from a clean tree can be told from a scratch one. A worktree
    that Windows git made has a path that WSL git cannot read: the Windows git (git.exe) is tried then. The answer is kept `max_age` seconds: two git processes cost 0.5 - 1.5 s
    and a run writes its index several times within seconds (an hours-long run asks again at its end)."""
    import shutil
    import subprocess
    import time
    if _CODE_STATE and time.time() - _CODE_STATE["at"] < max_age:
        return dict(_CODE_STATE["value"])
    here = os.path.dirname(os.path.abspath(__file__))
    attempts = [("git", here)]
    if shutil.which("git.exe"):
        try:
            win = subprocess.run(["wslpath", "-w", here], capture_output=True, text=True).stdout.strip()
        except OSError:                                                          # native Windows python: no wslpath
            win = ""
        attempts.append(("git.exe", win or here))
    for exe, where in attempts:
        try:
            head = subprocess.run([exe, "-C", where, "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=30)
            status = subprocess.run([exe, "-C", where, "status", "--porcelain", "--", "."], capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.SubprocessError):
            continue
        if not (head.returncode or status.returncode):
            value = {"commit": head.stdout.strip(), "dirty": any(_changes_the_code(line) for line in status.stdout.splitlines())}
            _CODE_STATE.update(at=time.time(), value=value)
            return dict(value)
    import sys
    print("WARNING: git cannot tell the commit and the state of GtGenerator/ here: the manifest gets no commit and the store is no release; "
          "run `python -m GtGenerator generate` again from a clean checkout to stamp it", file=sys.stderr, flush=True)
    return {"commit": None, "dirty": None}
