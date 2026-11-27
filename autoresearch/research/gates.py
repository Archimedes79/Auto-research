"""The gates: the owner's decided decisions, the TEST gate, holds, protected files and grants.

Decisions: a problem's DECISIONS.md and the root DECISIONS.md, each a table of decided rows `| <date> | D<n>: ... |` above a heading `## Open`.
TEST: a run opens a holdout split only when its problem's TEST_LOG.md has a row `| <date> | <run id> | D<n> | <note> |` in the table under
`## Runs` and D<n> is decided. Holds and grants are machine state (research/paths.state_dir), never in a checkout.

Protected: the root list (the hook, settings, the gate code: this file, paths.py and Evaluation/defaults.py, where every holdout read is decided;
champions/; every problem's TEST_LOG.md and problem.json) and each problem's own `protected` list; matched without regard to case. DECISIONS.md
is not protected: agents record the owner's answers there, and a grant needs one."""
from __future__ import annotations

import fnmatch
import os
import re
import time
from typing import Any, Dict, List, Optional, Set

from research.paths import ResearchError, problem_dir, read_json, repo, state_dir, write_json

DECIDED_ROW = re.compile(r"^\|\s*\d{4}-\d{2}-\d{2}[^|]*\|\s*(D\d+)\s*:.*$", re.M)
ROOT_PROTECTED = (".claude/settings.json", ".claude/settings.local.json", "autoresearch/research/hook.py", "autoresearch/research/gates.py",
                  "autoresearch/research/paths.py", "autoresearch/research/__init__.py", "autoresearch/Evaluation/defaults.py", "champions/*",
                  "problems/*/TEST_LOG.md")


def decided_rows(problem: Optional[str] = None, root: Optional[str] = None) -> Dict[str, str]:
    """decided id -> the text of its row (of both rows when the root file and the problem's file both have it)"""
    root = root or repo()
    files = [os.path.join(root, "DECISIONS.md")] + ([os.path.join(problem_dir(problem, root), "DECISIONS.md")] if problem else [])
    rows: Dict[str, str] = {}
    for path in files:
        try:
            with open(path, encoding="utf-8") as f:
                text = f.read()
        except OSError:
            continue
        for m in DECIDED_ROW.finditer(text.split("\n## Open", 1)[0]):
            rows[m.group(1)] = (rows.get(m.group(1), "") + "\n" + m.group(0)).strip()
    return rows


def decided(problem: Optional[str] = None, root: Optional[str] = None) -> Set[str]:
    """the decided ids of the root file and, with a problem, of the problem's file"""
    return set(decided_rows(problem, root))


def _cells(line: str) -> List[str]:
    return [c.strip().strip("`") for c in line.strip().strip("|").split("|")]


def test_refusal(problem: str, rid: str, root: Optional[str] = None) -> str:
    """'' when the TEST log opens the holdout split for this run, else why not"""
    path = os.path.join(problem_dir(problem, root), "TEST_LOG.md")
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return f"there is no {path}"
    section = text.split("\n## Runs", 1)
    if len(section) != 2:
        return f"{path} has no section '## Runs'"
    rows = [c for c in (_cells(ln) for ln in section[1].splitlines() if ln.lstrip().startswith("|")) if len(c) >= 4 and c[1] == rid]
    if not rows:
        return f"{rid} is not named in the '## Runs' table of {path}"
    if len(rows) > 1:
        return f"{rid} is named twice in {path}: a TEST use is one row for one new run id"
    d = rows[0][2]
    if not re.fullmatch(r"D\d+", d):
        return f"the row of {rid} in {path} names no decision (D<n>) in its third cell"
    if d not in decided(problem, root):
        return f"{d}, named for {rid} in {path}, is not a decided decision"
    return ""


# ------------------------------------------------------------------------------------------------ holds
def _holds_file() -> str:
    return os.path.join(state_dir(), "holds.json")


def holds() -> Dict[str, Any]:
    try:
        return read_json(_holds_file())
    except (OSError, ValueError):
        return {}


def hold(problem: str, reason: str) -> None:
    h = holds()
    h[problem] = {"reason": reason, "since": time.strftime("%Y-%m-%d %H:%M")}
    write_json(_holds_file(), h)


def release(problem: str) -> bool:
    h = holds()
    if problem not in h:
        return False
    del h[problem]
    write_json(_holds_file(), h)
    return True


def hold_refusal(problem: str) -> str:
    h = holds().get(problem)
    return f"{problem} is on hold since {h['since']}: {h['reason']} (`research release {problem}`)" if h else ""


# ------------------------------------------------------------------------------------------------ protected files and grants
def protected_patterns(root: Optional[str] = None) -> List[str]:
    """repository-relative patterns: the root list and every problem's `protected` list (a folder ends with /)"""
    root = root or repo()
    pats = list(ROOT_PROTECTED)
    base = os.path.join(root, "problems")
    for name in sorted(os.listdir(base)) if os.path.isdir(base) else []:
        try:
            p = read_json(os.path.join(base, name, "problem.json"))
        except (OSError, ValueError):
            pats.append(f"problems/{name}/problem.json")          # an unreadable problem.json protects itself
            continue
        pats.append(f"problems/{name}/problem.json")
        for item in p.get("protected", []):
            item = str(item).replace("\\", "/").lstrip("/")
            pats.append(f"problems/{name}/{item}*" if item.endswith("/") else f"problems/{name}/{item}")
    return pats


def is_protected(relpath: str, root: Optional[str] = None) -> bool:
    """without regard to case: on Windows test_log.md opens TEST_LOG.md"""
    relpath = relpath.replace("\\", "/").lower()
    return any(fnmatch.fnmatchcase(relpath, pat.lower()) for pat in protected_patterns(root))


def _grants_file() -> str:
    return os.path.join(state_dir(), "grants.json")


def grants() -> List[Dict[str, Any]]:
    """the grants in force; a grants file that cannot be read, or an entry that is no grant, grants nothing"""
    try:
        data = read_json(_grants_file())
    except (OSError, ValueError):
        return []
    return [g for g in (data if isinstance(data, list) else []) if isinstance(g, dict) and isinstance(g.get("path"), str)
            and isinstance(g.get("until"), (int, float)) and g["until"] > time.time()]


def grant(relpath: str, decision: str, hours: float, note: str, problem: Optional[str] = None) -> Dict[str, Any]:
    """allow edits of one protected path for some hours under a decided decision"""
    if decision not in decided(problem):
        raise ResearchError(f"{decision} is not a decided decision{' of ' + problem if problem else ''} (root DECISIONS.md" + (", the problem's DECISIONS.md)" if problem else ")"))
    if not 0 < hours <= 24:
        raise ResearchError("a grant lasts more than 0 and at most 24 hours")
    g = {"path": relpath.replace("\\", "/"), "decision": decision, "until": time.time() + hours * 3600, "note": note,
         "given": time.strftime("%Y-%m-%d %H:%M")}
    write_json(_grants_file(), grants() + [g])
    return g


def granted(relpath: str, root: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """the grant that allows an edit of this path now: it names the path (or a folder above it) and its decision is decided NOW, not only
    when the grant was given (the problem's DECISIONS.md counts for a path of that problem)"""
    rp = relpath.replace("\\", "/").lower()
    parts = rp.split("/")
    of_problem = len(parts) > 2 and parts[0] == "problems" and os.path.isfile(os.path.join(root or repo(), "problems", parts[1], "problem.json"))
    problem = parts[1] if of_problem else None
    for g in grants():
        gp = str(g.get("path", "")).lower()
        if (rp == gp or (gp.endswith("/") and rp.startswith(gp))) and g.get("decision") in decided(problem, root):
            return g
    return None


def protected_refusal(relpath: str, root: Optional[str] = None) -> str:
    """'' when an edit of this repository-relative path may go ahead"""
    if not is_protected(relpath, root) or granted(relpath, root):
        return ""
    return (f"{relpath} is protected: an edit needs a grant under a decided decision (`research grant {relpath} --decision D<n> --hours 2 --note ...`); "
            "without one, write the question into the Open table of DECISIONS.md and wait for the owner")
