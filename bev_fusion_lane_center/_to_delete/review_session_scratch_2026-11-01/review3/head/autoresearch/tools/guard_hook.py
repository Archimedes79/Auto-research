"""Gate hook (ROADMAP P4): a Claude Code PreToolUse hook that refuses Write / Edit / NotebookEdit (and the file-writing MCP tools) on the protected
files of the problem unless a grant that names a decided D<n> covers them. Standard library only.

    (hook)  wsl.exe --cd "$CLAUDE_PROJECT_DIR" python3 autoresearch/tools/guard_hook.py || true
            reads the hook JSON on stdin; a refusal is a JSON "deny" on stdout (exit 0), so a missing script or a crash never blocks an edit
    python3 autoresearch/tools/guard_hook.py grant <path> --decision D8 [--hours 8] [--note "..."]    allow edits of <path> (relative to the
            problem folder, "../" reaches the repository; it must cover or lie inside a protected path)
    python3 autoresearch/tools/guard_hook.py grants                 the grants, active and expired
    python3 autoresearch/tools/guard_hook.py check <path>           would an edit of <path> be refused? (exit 2 = refused)
    python3 autoresearch/tools/guard_hook.py --selftest

Protected (project.json "protected", relative to the problem folder): whole files or folders (ending in /), and for a json file optionally only some
top-level entries ({"models/registry.json": ["champion", "control"]}: adding or changing other entries stays free, changing the champion is a
decision). The grant file (project.json "grants") is always protected: only `grant` writes it.
A grant is valid while `until` lies ahead and its decision has a row in the "Decided" table of DECISIONS.md (the decision cell starts with D<n>).
Every grant is a row of the grant file (a text record in git), so every deliberate edit of a protected file is visible.
This is a guard against accidental and stray edits (a subagent, a scout, a typo in a path), not against a determined agent: Bash can still write
the files.
"""
import csv
import datetime as dt
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import project as P  # noqa: E402

TOOLS = ("Write", "Edit", "MultiEdit", "NotebookEdit")
MCP_WRITERS = ("write_file", "edit_block", "move_file", "write_pdf")  # e.g. mcp__desktop-commander__write_file (input key "path")
GCOLS = ["granted", "until", "path", "decision", "note"]


def norm(p, cwd=None):
    """one spelling for a path from Windows (C:\\x\\y, \\\\?\\C:\\x, x.md., x.md::$DATA, relative to cwd) or WSL (/mnt/c/x/y): /mnt/c/x/y,
    lower case under /mnt/ (the Windows file system ignores case)"""
    p = p.replace("\\", "/")
    if p.startswith(("//?/", "//./")):
        p = p[4:]
    m = re.match(r"^([A-Za-z]):/(.*)$", p)
    if m:
        p = f"/mnt/{m.group(1).lower()}/{m.group(2)}"
    elif not p.startswith("/") and cwd:
        p = norm(cwd).rstrip("/") + "/" + p
    if p.startswith("/mnt/"):
        p = p.split(":", 1)[0]  # an NTFS stream (x.md::$DATA) is the file itself
        p = "/".join(s.rstrip(". ") or s for s in p.split("/"))  # Windows drops trailing dots and spaces of a name
    p = os.path.normpath(p).replace("\\", "/")
    return p.lower() if p.startswith("/mnt/") else p


def grant_file(root, cfg):
    return os.path.join(root, cfg.get("grants", "experiment_execution/grants.csv"))


def rules(root, cfg):
    """[(relative spelling, absolute path, is_folder, entries or None)] from project.json"""
    prot = cfg.get("protected", {})
    items = prot.items() if isinstance(prot, dict) else ((x, None) for x in prot)
    return [(rel, norm(os.path.join(root, rel)), rel.endswith("/"), entries) for rel, entries in items]


def decided(root, cfg):
    """the D<n> ids that have a row of their own in the Decided table of DECISIONS.md (a row "| date | D8: ... |"; an id only mentioned in
    another row's text, or listed under Open, does not count)"""
    p = os.path.join(root, cfg.get("decisions", "experiment_execution/DECISIONS.md"))
    if not os.path.exists(p):
        return set()
    text = open(p, encoding="utf-8").read()
    m = re.search(r"^## Decided\s*$(.*?)(?=^## |\Z)", text, re.M | re.S)
    return set(re.findall(r"^\|\s*[\d/-]+\s*\|\s*(D\d+)\b", m.group(1), re.M)) if m else set()


def grants(root, cfg):
    p = grant_file(root, cfg)
    if not os.path.exists(p):
        return []
    with open(p, newline="", encoding="utf-8") as f:
        return [g for g in csv.DictReader(f) if all(g.get(k) for k in ("until", "path", "decision"))]  # a broken row opens nothing


def covers(gp, path):
    return path == gp or path.startswith(gp.rstrip("/") + "/")


def granted(path, root, cfg, now=None):
    now = now or dt.datetime.now().isoformat(timespec="minutes")
    ok = decided(root, cfg)
    for g in grants(root, cfg):
        if covers(norm(os.path.join(root, g["path"])), path) and g["until"] > now and g["decision"] in ok:
            return g
    return None


def new_text(tool, inp, path):
    """the file text after the edit, where the hook can reproduce it (Write, Edit, MultiEdit); None when it cannot"""
    if tool == "Write":
        return inp.get("content")
    edits = inp.get("edits") if tool == "MultiEdit" else [inp] if tool == "Edit" else None
    if edits is None:
        return None
    try:
        text = open(path, encoding="utf-8").read()
    except OSError:
        return None
    for e in edits:
        o, n = e.get("old_string", ""), e.get("new_string", "")
        if not o or o not in text:
            return None  # e.g. a line-ending difference Claude Code normalises: the result is unknown
        text = text.replace(o, n) if e.get("replace_all") else text.replace(o, n, 1)
    return text


def entries_changed(path, after, entries):
    """which protected top-level entries of a json file the edit changes (all of them when the result is not a readable json object)"""
    try:
        before = json.load(open(path, encoding="utf-8"))
    except (OSError, ValueError):
        before = {}
    try:
        new = json.loads(after) if after is not None else None
    except ValueError:
        new = None
    if not isinstance(new, dict) or not isinstance(before, dict):
        return list(entries)
    return [k for k in entries if before.get(k) != new.get(k)]


def target(tool, inp):
    """the path a tool call writes, or None for a tool that writes no file"""
    if tool in TOOLS:
        return inp.get("file_path") or inp.get("notebook_path")
    if tool.startswith("mcp__") and tool.endswith(MCP_WRITERS):
        return inp.get("path") or inp.get("file_path") or inp.get("destination")
    return None


def verdict(tool, inp, root=None, cfg=None, now=None, cwd=None):
    """None = allowed, else the reason for the refusal"""
    root, cfg = root or P.root(), cfg or P.cfg()
    raw = target(tool, inp)
    if not raw:
        return None
    path = norm(raw, cwd)
    if path == norm(grant_file(root, cfg)):
        return "guard_hook: the grant file is written only by `guard_hook.py grant` (a decided D<n> behind each row)"
    for rel, prot, folder, entries in rules(root, cfg):
        if not (covers(prot, path) if folder else path == prot):
            continue
        what = rel
        if entries:
            changed = entries_changed(path, new_text(tool, inp, path), entries)
            if not changed:
                return None
            what += f" (entries {', '.join(changed)})"
        if granted(path, root, cfg, now):
            return None
        return (f"guard_hook: {what} is protected (a decision of the person, DECISIONS.md). If a decided entry covers this edit, record a grant first: "
                f"`python3 ../autoresearch/tools/guard_hook.py grant {rel} --decision D<n> --hours 8 --note \"...\"` (from the problem folder, in the WSL); "
                f"otherwise write the question into DECISIONS.md and wait for the answer. Do not go around this through the shell.")
    return None


def add_grant(path, decision, hours, note, root=None, cfg=None):
    root, cfg = root or P.root(), cfg or P.cfg()
    if decision not in decided(root, cfg):
        sys.exit(f"{decision} has no row in the Decided table of DECISIONS.md: no grant")
    full = norm(os.path.join(root, path))
    hit = [rel for rel, prot, folder, _ in rules(root, cfg) if covers(prot, full) or (full != norm(root) and covers(full, prot) and not covers(full, norm(root)))]
    if not hit:
        sys.exit(f"{path} ({full}) is no protected path and lies inside none; give it relative to the problem folder, e.g. ../autoresearch/tools/ "
                 f"(protected: {', '.join(r for r, *_ in rules(root, cfg))})")
    p = grant_file(root, cfg)
    now = dt.datetime.now()
    row = dict(granted=now.isoformat(timespec="minutes"), until=(now + dt.timedelta(hours=hours)).isoformat(timespec="minutes"), path=path,
               decision=decision, note=note)
    with P.locked(p):
        rows = grants(root, cfg) + [row]
        import io
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=GCOLS, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
        P.atomic_write(p, buf.getvalue())
    print(f"grant: {path} -> {full} until {row['until']} ({decision}; opens {', '.join(hit)})")


def deny(reason):
    """the refusal as Claude Code reads it: JSON on stdout with exit 0 (a crash, by contrast, prints nothing and lets the edit through)"""
    print(json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny", "permissionDecisionReason": reason}}))


def selftest():
    import shutil
    import tempfile
    assert norm("C:\\ML-Experiment\\x\\PROBLEM.md") == "/mnt/c/ml-experiment/x/problem.md"
    assert norm("\\\\?\\C:\\ML-Experiment\\x\\PROBLEM.md") == norm("C:/ml-experiment/X/problem.md")
    assert norm("C:\\ML-Experiment\\x\\PROBLEM.md.") == norm("C:\\ML-Experiment\\x\\PROBLEM.md::$DATA") == norm("/mnt/c/ML-Experiment/x/PROBLEM.md")
    assert norm("x\\PROBLEM.md", cwd="C:\\ML-Experiment") == norm("/mnt/c/ML-Experiment/x/PROBLEM.md")
    assert norm("/mnt/c/a/b/../c.md") == "/mnt/c/a/c.md"
    root = tempfile.mkdtemp()
    os.makedirs(os.path.join(root, "experiment_execution"))
    os.makedirs(os.path.join(root, "tools"))
    open(os.path.join(root, "experiment_execution/DECISIONS.md"), "w").write(
        "## Open\n| D9 | x |\n\n## Decided\n| date | decision |\n|---|---|\n| 2026-10-25 | D8: yes, see also D9 and D3 |\n\n## Notes\n| 2026-10-26 | D4: not a decision |\n")
    open(os.path.join(root, "reg.json"), "w").write(json.dumps({"champion": {"run": "a"}, "fast": {"run": "b"}}))
    cfg = dict(protected={"TEST_LOG.md": None, "tools/": None, "reg.json": ["champion"]}, grants="experiment_execution/grants.csv",
               decisions="experiment_execution/DECISIONS.md")
    assert decided(root, cfg) == {"D8"}, decided(root, cfg)
    v = lambda tool, **inp: verdict(tool, inp, root, cfg)  # noqa: E731
    assert v("Write", file_path=os.path.join(root, "TEST_LOG.md"), content="x"), "a protected file must be refused"
    assert v("Edit", file_path=os.path.join(root, "tools/run.py"), old_string="a", new_string="b"), "a file in a protected folder must be refused"
    assert v("mcp__desktop-commander__write_file", path=os.path.join(root, "TEST_LOG.md"), content="x"), "an MCP file writer must be refused too"
    assert verdict("Edit", dict(file_path="TEST_LOG.md"), root, cfg, cwd=root), "a relative path is read against the cwd of the event"
    assert v("Edit", file_path=os.path.join(root, "other.md"), old_string="a", new_string="b") is None, "other files pass"
    assert v("Read", file_path=os.path.join(root, "TEST_LOG.md")) is None, "reading passes"
    reg = os.path.join(root, "reg.json")
    assert v("Write", file_path=reg, content=json.dumps({"champion": {"run": "a"}, "fast": {"run": "c"}, "new": {}})) is None, "other entries are free"
    assert v("Write", file_path=reg, content=json.dumps({"champion": {"run": "z"}, "fast": {"run": "b"}})), "a champion change must be refused"
    assert v("Write", file_path=reg, content="[]"), "a registry that is no object must be refused"
    assert v("Edit", file_path=reg, old_string='"run": "a"', new_string='"run": "z"'), "an Edit of the champion must be refused"
    assert v("Edit", file_path=reg, old_string='"run": "b"', new_string='"run": "c"') is None, "an Edit of another entry passes"
    assert v("Edit", file_path=reg, old_string='"run":\r\n "b"', new_string='"run": "c"'), "an Edit the hook cannot reproduce must be refused"
    assert v("Write", file_path=os.path.join(root, "experiment_execution/grants.csv"), content="x"), "the grant file is never edited"
    for bad in (("TEST_LOG.md", "D9"), ("TEST_LOG.md", "D3"), ("TEST_LOG.md", "D4")):
        try:
            add_grant(bad[0], bad[1], 1, "not decided", root, cfg)
            raise AssertionError(f"a grant needs a decided entry: {bad}")
        except SystemExit:
            pass
    for bad in ("", ".", "..", "other.md"):
        try:
            add_grant(bad, "D8", 1, "too wide or not protected", root, cfg)
            raise AssertionError(f"a grant must cover or lie inside a protected path, not {bad!r}")
        except SystemExit:
            pass
    add_grant("tools/", "D8", 1, "selftest", root, cfg)
    assert v("Edit", file_path=os.path.join(root, "tools/run.py"), old_string="a", new_string="b") is None, "a grant opens the folder"
    assert v("Write", file_path=os.path.join(root, "TEST_LOG.md"), content="x"), "a grant opens only its path"
    later = (dt.datetime.now() + dt.timedelta(hours=2)).isoformat(timespec="minutes")
    assert verdict("Edit", dict(file_path=os.path.join(root, "tools/run.py")), root, cfg, now=later), "an expired grant opens nothing"
    with open(grant_file(root, cfg), "a") as f:
        f.write("2026-10-25T10:00,,TEST_LOG.md,D8\n")
    assert v("Write", file_path=os.path.join(root, "TEST_LOG.md"), content="x"), "a broken grant row opens nothing"
    shutil.rmtree(root)
    print("guard_hook selftest passed")


def main():
    a = sys.argv[1:]
    if "--selftest" in a:
        return selftest()
    if a and a[0] == "grant":
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd")
        ap.add_argument("path")
        ap.add_argument("--decision", required=True)
        ap.add_argument("--hours", type=float, default=8)
        ap.add_argument("--note", default="")
        g = ap.parse_args(a)
        return add_grant(g.path, g.decision, g.hours, g.note)
    if a and a[0] == "grants":
        for g in grants(P.root(), P.cfg()):
            print(f"{g['granted']} -> {g['until']}  {g['decision']:4} {g['path']}  {g['note']}")
        return 0
    if a and a[0] == "check":
        why = verdict("Write", dict(file_path=os.path.abspath(a[1])))  # unknown content: every protected entry counts as changed
        print(why or "allowed")
        sys.exit(2 if why else 0)
    try:
        ev = json.load(sys.stdin)
    except ValueError:
        return 0
    why = verdict(ev.get("tool_name", ""), ev.get("tool_input") or {}, cwd=ev.get("cwd"))
    if why:
        deny(why)
    return 0


if __name__ == "__main__":
    main()
