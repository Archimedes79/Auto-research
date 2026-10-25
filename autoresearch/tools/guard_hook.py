"""Gate hook (ROADMAP P4): a Claude Code PreToolUse hook that refuses Write / Edit / MultiEdit / NotebookEdit on the protected files of the problem
unless a grant that names a decided D<n> covers them. Standard library only.

    (hook)  wsl python3 autoresearch/tools/guard_hook.py           reads the hook JSON on stdin; exit 2 + reason = refused, exit 0 = allowed
    python3 autoresearch/tools/guard_hook.py grant <path> --decision D8 [--hours 8] [--note "..."]    allow edits of <path> (file or folder)
    python3 autoresearch/tools/guard_hook.py grants                 the grants, active and expired
    python3 autoresearch/tools/guard_hook.py check <path>           would an edit of <path> be refused? (exit 2 = refused)
    python3 autoresearch/tools/guard_hook.py --selftest

Protected (project.json "protected", relative to the problem folder, "../" reaches the repository): whole files or folders, and for a json file
optionally only some top-level entries ({"models/registry.json": ["champion", "control"]}: adding or changing other entries stays free, changing
the champion is a decision). The grant file (project.json "grants") is always protected: only `grant` writes it.
A grant is valid while `until` lies ahead and its decision appears in the "Decided" part of DECISIONS.md. Every grant is a row of the grant file
(a text record in git), so every deliberate edit of a protected file is visible.
This is a guard against accidental and stray edits (a subagent, a scout, a typo in a path), not against a determined agent: Bash can still write
the files. If the hook itself fails (no WSL, a crash) Claude Code lets the edit through (any exit code other than 2).
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
GCOLS = ["granted", "until", "path", "decision", "note"]


def norm(p):
    """one spelling for a path from Windows (C:\\x\\y) or WSL (/mnt/c/x/y): /mnt/c/x/y, lower case under /mnt/ (case-insensitive file system)"""
    p = p.replace("\\", "/")
    m = re.match(r"^([A-Za-z]):/(.*)$", p)
    if m:
        p = f"/mnt/{m.group(1).lower()}/{m.group(2)}"
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
    """the D<n> ids in the Decided part of DECISIONS.md"""
    p = os.path.join(root, cfg.get("decisions", "experiment_execution/DECISIONS.md"))
    if not os.path.exists(p):
        return set()
    text = open(p, encoding="utf-8").read()
    i = text.find("## Decided")
    return set(re.findall(r"\b(D\d+)\b", text[i:])) if i >= 0 else set()


def grants(root, cfg):
    p = grant_file(root, cfg)
    if not os.path.exists(p):
        return []
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def granted(path, root, cfg, now=None):
    now = now or dt.datetime.now().isoformat(timespec="minutes")
    ok = decided(root, cfg)
    for g in grants(root, cfg):
        gp = norm(os.path.join(root, g["path"]))
        if (path == gp or path.startswith(gp.rstrip("/") + "/")) and g["until"] > now and g["decision"] in ok:
            return g
    return None


def new_text(tool, inp, path):
    """the file text after the edit, where it can be known (Write, Edit, MultiEdit); None otherwise"""
    if tool == "Write":
        return inp.get("content", "")
    try:
        text = open(path, encoding="utf-8").read()
    except OSError:
        return None
    edits = inp.get("edits") if tool == "MultiEdit" else [inp] if tool == "Edit" else None
    if edits is None:
        return None
    for e in edits:
        o, n = e.get("old_string", ""), e.get("new_string", "")
        text = text.replace(o, n) if e.get("replace_all") else text.replace(o, n, 1)
    return text


def entries_changed(path, after, entries):
    """which protected top-level entries of a json file the edit changes (all of them when the result cannot be read)"""
    try:
        before = json.load(open(path, encoding="utf-8"))
    except (OSError, ValueError):
        before = {}
    try:
        new = json.loads(after) if after is not None else None
    except ValueError:
        new = None
    if new is None:
        return list(entries)
    return [k for k in entries if before.get(k) != new.get(k)]


def verdict(tool, inp, root=None, cfg=None, now=None):
    """None = allowed, else the reason for the refusal"""
    root, cfg = root or P.root(), cfg or P.cfg()
    if tool not in TOOLS:
        return None
    raw = inp.get("file_path") or inp.get("notebook_path") or ""
    if not raw:
        return None
    path = norm(raw)
    if path == norm(grant_file(root, cfg)):
        return "guard_hook: the grant file is written only by `guard_hook.py grant` (a decided D<n> behind each row)"
    for rel, prot, folder, entries in rules(root, cfg):
        if not (path.startswith(prot.rstrip("/") + "/") if folder else path == prot):
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
                f"otherwise write the question into DECISIONS.md and wait for the answer.")
    return None


def add_grant(path, decision, hours, note, root=None, cfg=None):
    root, cfg = root or P.root(), cfg or P.cfg()
    if decision not in decided(root, cfg):
        sys.exit(f"{decision} is not in the Decided part of DECISIONS.md: no grant")
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
    print(f"grant: {path} until {row['until']} ({decision})")


def selftest():
    import shutil
    import tempfile
    root = tempfile.mkdtemp()
    os.makedirs(os.path.join(root, "experiment_execution"))
    os.makedirs(os.path.join(root, "tools"))
    open(os.path.join(root, "experiment_execution/DECISIONS.md"), "w").write("## Open\n| D9 | x |\n\n## Decided\n| 2026 | D8: yes |\n")
    open(os.path.join(root, "reg.json"), "w").write(json.dumps({"champion": {"run": "a"}, "fast": {"run": "b"}}))
    cfg = dict(protected={"TEST_LOG.md": None, "tools/": None, "reg.json": ["champion"]}, grants="experiment_execution/grants.csv",
               decisions="experiment_execution/DECISIONS.md")
    win = lambda rel: re.sub(r"^/mnt/([a-z])/", lambda m: m.group(1).upper() + ":\\\\", os.path.join(root, rel)).replace("/", "\\")  # noqa: E731
    v = lambda tool, **inp: verdict(tool, inp, root, cfg)  # noqa: E731
    assert v("Write", file_path=os.path.join(root, "TEST_LOG.md"), content="x"), "a protected file must be refused"
    assert v("Edit", file_path=os.path.join(root, "tools/run.py"), old_string="a", new_string="b"), "a file in a protected folder must be refused"
    assert v("Edit", file_path=os.path.join(root, "other.md"), old_string="a", new_string="b") is None, "other files pass"
    assert v("Read", file_path=os.path.join(root, "TEST_LOG.md")) is None, "reading passes"
    if root.startswith("/mnt/"):
        assert v("Write", file_path=win("TEST_LOG.md"), content="x"), "a Windows spelling of the path must be refused too"
    reg = os.path.join(root, "reg.json")
    assert v("Write", file_path=reg, content=json.dumps({"champion": {"run": "a"}, "fast": {"run": "c"}, "new": {}})) is None, "other entries are free"
    assert v("Write", file_path=reg, content=json.dumps({"champion": {"run": "z"}, "fast": {"run": "b"}})), "a champion change must be refused"
    assert v("Edit", file_path=reg, old_string='"run": "a"', new_string='"run": "z"'), "an Edit of the champion must be refused"
    assert v("Edit", file_path=reg, old_string='"run": "b"', new_string='"run": "c"') is None, "an Edit of another entry passes"
    assert v("Write", file_path=os.path.join(root, "experiment_execution/grants.csv"), content="x"), "the grant file is never edited"
    try:
        add_grant("TEST_LOG.md", "D9", 1, "open decision", root, cfg)
        raise AssertionError("a grant needs a decided entry")
    except SystemExit:
        pass
    add_grant("tools/", "D8", 1, "selftest", root, cfg)
    assert v("Edit", file_path=os.path.join(root, "tools/run.py"), old_string="a", new_string="b") is None, "a grant opens the folder"
    assert v("Write", file_path=os.path.join(root, "TEST_LOG.md"), content="x"), "a grant opens only its path"
    later = (dt.datetime.now() + dt.timedelta(hours=2)).isoformat(timespec="minutes")
    assert verdict("Edit", dict(file_path=os.path.join(root, "tools/run.py")), root, cfg, now=later), "an expired grant opens nothing"
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
    why = verdict(ev.get("tool_name", ""), ev.get("tool_input") or {})
    if why:
        print(why, file=sys.stderr)
        sys.exit(2)
    return 0


if __name__ == "__main__":
    main()
