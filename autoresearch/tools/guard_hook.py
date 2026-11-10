"""Gate hook (ROADMAP P4): a Claude Code PreToolUse hook that refuses Write / Edit / NotebookEdit (and the file-writing MCP tools) on the protected
files of the problems unless a grant that names a decided D<n> covers them. Standard library only.

    (hook)  wsl.exe --cd "$CLAUDE_PROJECT_DIR" python3 autoresearch/tools/guard_hook.py || true
            reads the hook JSON on stdin; a refusal is a JSON "deny" on stdout (exit 0). A crash of the decision (an unreadable project.json, a bad
            grant file, ...) refuses every path that names a protected file and is logged to autoresearch/state/guard_hook_crash.log; a crash
            that happens before the script runs (a syntax error in this file, no python3, no wsl.exe) lets the edit through: `|| true`
    python3 autoresearch/tools/guard_hook.py grant <path> --decision D8 [--hours 8] [--note "..."]    allow edits of <path> (relative to the
            problem folder you are in, else the default problem; "../" reaches the repository; it must cover or lie inside a protected
            path; a path that several problems protect gets one row in each problem's grant file)
    python3 autoresearch/tools/guard_hook.py grants                 the grants of every problem, active and expired
    python3 autoresearch/tools/guard_hook.py check <path>           would an edit of <path> be refused? (exit 2 = refused)
    python3 autoresearch/tools/guard_hook.py --selftest

Problems (D75, 2026-11-10): EVERY sub-folder of the workspace with a project.json is a problem (the set commit_paths.py reads; a link to a problem folder is
asked under its own name too); a path is refused when the protected list of any of them covers it and that problem's grants do not open it. The hook reads
project.json itself and does not import project.py or read config.json's default_project for its decision, and the grant command does not need project.py
either (own lock and atomic write): a broken project.py or an edited default_project neither switches the hook off nor locks the repair out.
A move of a FOLDER that holds a protected path is refused like an edit of that path.
Protected (project.json "protected", relative to the problem folder): whole files or folders (ending in /), and for a json file optionally only some
top-level entries ({"models/registry.json": ["champion", "control"]}: adding or changing other entries stays free, changing the champion is a
decision). The grant file (project.json "grants") is always protected: only `grant` writes it.
A grant is valid while `until` lies ahead and its decision is decided: its file D<n>.md in the decisions folder has a "Decided: <date> ..." line (D69), or, with the older
file form, it has a row in the "Decided" table of DECISIONS.md (the decision cell starts with D<n>). The key "grant_decisions" names that record for the hook alone
(else "decisions" does): the bev_fusion problem points it at the lane problem's folder and leaves "decisions" unset, because tools/testgate.py reads "decisions"
and a decision of the lane must not open a TEST run of the bev problem.
Every grant is a row of the grant file (a text record in git), so every deliberate edit of a protected file is visible.
This is a guard against accidental and stray edits (a subagent, a scout, a typo in a path), not against a determined agent: Bash can still write
the files.
"""
import os
import sys

if __name__ == "__main__" and sys.path and os.path.realpath(sys.path[0] or ".") == os.path.dirname(os.path.realpath(__file__)):
    del sys.path[0]  # run as the hook: a file of this folder named like a standard module (json.py) must not shadow the real one and crash the hook

import contextlib  # noqa: E402
import csv  # noqa: E402
import datetime as dt  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
AR = os.path.dirname(HERE)       # autoresearch/
WS = os.path.dirname(AR)         # the workspace: its sub-folders that hold a project.json are the problem folders
CRASH_LOG = os.path.join(AR, "state", "guard_hook_crash.log")  # state/ is not in git

TOOLS = ("Write", "Edit", "MultiEdit", "NotebookEdit")
MCP_WRITERS = ("write_file", "edit_block", "move_file", "write_pdf")  # e.g. mcp__desktop-commander__write_file (input key "path")
GCOLS = ["granted", "until", "path", "decision", "note"]
GATE_KEYS = ["protected", "grants", "decisions", "grant_decisions"]  # the top-level keys of project.json that are protected whatever the problem's own list says
# what a crashed hook still refuses by NAME when no protected list can be read (the gate's own files and the truth files of the problems; lower case, /mnt/ paths are lowered by norm)
GATE_NAMES = re.compile(r"/(?:autoresearch/(?:tools/(?:guard_hook|project)\.py|config\.json)|\.claude/settings(?:\.local)?\.json"
                        r"|project\.json|grants\.csv|test_usage_log\.md|problem\.md|champion\.json|golden\.json|registry\.json|e123_d7_r3_f1\.json)$", re.I)  # selftest: covers every file rule of the live lists


def norm(p, cwd=None):
    """one spelling for a path from Windows (C:\\x\\y, \\\\?\\C:\\x, x.md., x.md::$DATA, relative to cwd, \\x on the drive of cwd), a share
    (\\\\wsl$\\<distro>\\x and \\\\wsl.localhost\\<distro>\\x -> /x, \\\\host\\C$\\x -> C:\\x) or WSL (/mnt/c/x/y, //mnt/c/x/y): /mnt/c/x/y, 8.3 short
    names of existing folders (AUTO-R~1) expanded, lower case under /mnt/ (the Windows file system ignores case)"""
    p = p.replace("\\", "/")
    if p.startswith(("//?/", "//./")):
        p = p[4:]
        if p.lower().startswith("unc/"):  # \\?\UNC\host\share\x
            p = "//" + p[4:]
    m = re.match(r"^//(?:wsl\$|wsl\.localhost)/[^/]+(/.*)?$", p, re.I)
    if m:
        p = m.group(1) or "/"
    m = re.match(r"^//[^/]+/([A-Za-z])\$(/.*)?$", p)
    if m:
        p = f"{m.group(1)}:{m.group(2) or '/'}"
    d = re.match(r"^([A-Za-z]):[\\/]", cwd or "")  # a Windows cwd: the event comes from Windows
    m = re.match(r"^([A-Za-z]):(?!/)(.*)$", p)
    if m and d and m.group(1).lower() == d.group(1).lower():  # C:x is relative to the cwd of that drive
        p = cwd.replace("\\", "/").rstrip("/") + "/" + m.group(2)
    elif m:  # C:x on another drive: its cwd is unknown, read it at the drive's root (verdict refuses this form anyway)
        p = f"{m.group(1)}:/{m.group(2)}"
    m = re.match(r"^([A-Za-z]):/(.*)$", p)
    if m:
        p = f"/mnt/{m.group(1).lower()}/{m.group(2)}"
    elif p.startswith("/"):
        p = "/" + p.lstrip("/")
        g = re.match(r"^/([A-Za-z])(/.*)?$", p)
        if d and g:  # from a Windows cwd, /c/x is the Git Bash spelling of C:\x
            p = f"/mnt/{g.group(1).lower()}{g.group(2) or '/'}"
        elif not p.startswith("/mnt/") and d:  # from a Windows cwd, \x is drive-relative: the drive of cwd (from a Linux cwd /x is a Linux path)
            p = f"/mnt/{d.group(1).lower()}{p}"
    elif cwd:
        p = norm(cwd).rstrip("/") + "/" + p
    if p.startswith("/mnt/"):
        p = p.split(":", 1)[0]  # an NTFS stream (x.md::$DATA) is the file itself
        p = "/".join(s.rstrip(". ") or s for s in p.split("/"))  # Windows drops trailing dots and spaces of a name
    p = os.path.normpath(p).replace("\\", "/")
    if p.startswith("/mnt/") and re.search(r"~\d", p):
        p = long_name(p)
    return p.lower() if p.startswith("/mnt/") else p


def long_name(p):
    """expand the 8.3 short names (AUTO-R~1) of the existing folders and files of a /mnt/ path: the entry of the parent that is the same file"""
    cur = ""
    for s in p.strip("/").split("/"):
        nxt = cur + "/" + s
        if re.search(r"~\d", s) and os.path.exists(nxt):
            try:
                s = next((e for e in os.listdir(cur or "/") if e != s and _same(os.path.join(cur or "/", e), nxt)), s)
            except OSError:
                pass
        cur += "/" + s
    return cur


def _same(a, b):
    try:
        return os.path.samefile(a, b)
    except OSError:  # e.g. pagefile.sys cannot be stat-ed
        return False


def grant_file(root, cfg):
    return os.path.join(root, cfg.get("grants", "experiment_execution/grants.csv"))


def load_cfg(root):
    """the project.json of a problem folder as written (no defaults: the hook reads only protected / grants / decisions); raises, naming the file, when it is
    unreadable or not a json object, so the decision crashes and the gate fails closed (decide_event)"""
    f = os.path.join(root, "project.json")
    try:
        with open(f, encoding="utf-8") as fh:
            cfg = json.load(fh)
    except (OSError, ValueError) as e:
        raise ValueError(f"{f}: {e}") from e
    if not isinstance(cfg, dict):
        raise ValueError(f"{f} is not a json object")
    return cfg


def problems(ws=None):
    """[(problem folder, its project.json)] of every sub-folder of the workspace that has a project.json (the set commit_paths.py reads; a link to a problem
    folder is a problem under its own name too, so an edit through either spelling is asked); an unreadable project.json raises (the gate fails closed instead
    of skipping a problem)"""
    ws = ws or WS
    return [(os.path.join(ws, n), load_cfg(os.path.join(ws, n))) for n in sorted(os.listdir(ws)) if os.path.isfile(os.path.join(ws, n, "project.json"))]


def default_root():
    """the default problem, as project.root() finds it: AR_PROJECT, else default_project of config.json (relative to autoresearch/)"""
    p = os.environ.get("AR_PROJECT")
    if not p:
        with open(os.path.join(AR, "config.json"), encoding="utf-8") as f:
            p = json.load(f)["default_project"]
    r = os.path.normpath(p if os.path.isabs(p) else os.path.join(AR, p))
    if not os.path.isfile(os.path.join(r, "project.json")) and not os.path.isabs(p) and os.path.isfile(os.path.join(os.path.abspath(p), "project.json")):
        r = os.path.normpath(os.path.abspath(p))  # a relative AR_PROJECT may also mean: relative to the current folder
    if not os.path.isfile(os.path.join(r, "project.json")):
        sys.exit(f"no project.json in {r} (AR_PROJECT={os.environ.get('AR_PROJECT')!r}, else default_project of autoresearch/config.json)")
    return r


@contextlib.contextmanager
def _locked(path, wait_s=60, stale_s=120):
    """exclusive lock on a grant file: <path>.lock made with O_EXCL (the same file project.locked uses); one older than stale_s is taken over. Own code, so the
    grant command works when project.py is broken"""
    lock, t0 = path + ".lock", time.time()
    while True:
        try:
            os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            break
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(lock) > stale_s:
                    os.remove(lock)
                    continue
            except OSError:  # gone meanwhile, or not removable: wait like for a live lock (and time out)
                pass
            if time.time() - t0 > wait_s:
                raise TimeoutError(f"{lock} held for more than {wait_s} s")
            time.sleep(0.2)
    try:
        yield
    finally:
        try:
            os.remove(lock)
        except OSError:
            pass


def _atomic_write(path, text):
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def rules(root, cfg):
    """[(relative spelling, absolute path, is_folder, entries or None)] from project.json"""
    prot = cfg.get("protected", {})
    if not isinstance(prot, (dict, list)):
        raise ValueError(f"{root}: project.json 'protected' must be an object or a list, not {type(prot).__name__}")
    out = []
    for rel, entries in prot.items() if isinstance(prot, dict) else ((x, None) for x in prot):
        if not isinstance(rel, str) or not (entries is None or (isinstance(entries, list) and all(isinstance(e, str) for e in entries))):
            raise ValueError(f"{root}: protected entry {rel!r}: {entries!r} is neither null nor a list of key names")  # a string would be read as its letters: nothing 'changed'
        out.append((rel, norm(os.path.join(root, rel)), rel.endswith("/"), entries))
    return out


def gate_rules(root, cfg):
    """rules() plus the baseline: whatever a problem's list says (or lacks), the keys of its project.json that name the gate's own records (GATE_KEYS: protected,
    grants, decisions, grant_decisions) are protected. Used by the hook and the grant command; rules() stays the plain list (commit_paths.py reads it)"""
    out = rules(root, cfg)
    own = norm(os.path.join(root, "project.json"))
    for i, (rel, p, folder, entries) in enumerate(out):
        if p == own and not folder:
            out[i] = (rel, p, folder, entries + [k for k in GATE_KEYS if k not in entries] if entries else entries)  # a null / empty entry list = the whole file
            return out
    return out + [("project.json", own, False, list(GATE_KEYS))]


def decided(root, cfg):
    """the D<n> ids that count as decided. project.json "decisions" names either a FOLDER with one file D<n>.md per decision (D69, 2026-11-10: the id is
    decided when its file has a line "Decided: <yyyy-mm-dd> ..."; a file without one is an open question) or the older file DECISIONS.md, where an id
    is decided when it has a row of its own in the Decided table (a row "| date | D8: ... |"; an id only mentioned in another row's text, or listed
    under Open, does not count)"""
    p = os.path.join(root, cfg.get("grant_decisions") or cfg.get("decisions", "experiment_execution/DECISIONS.md"))
    if os.path.isdir(p):
        out = set()
        for name in os.listdir(p):
            m = re.fullmatch(r"(D\d+)\.md", name)  # the id is written exactly (D05 is not D5)
            f = os.path.join(p, name)
            if not m or not os.path.isfile(f):
                continue
            try:
                with open(f, encoding="utf-8") as fh:
                    body = fh.read()
            except (OSError, ValueError):  # an unreadable file (bad bytes, no permission) decides nothing; it must not crash the gate, which fails open
                continue
            body = re.sub(r"^```.*?^```[ \t]*$", "", body, flags=re.M | re.S)  # a Decided line inside a code block (an example, a template) does not count
            d = re.search(r"^Decided:[ \t]*(\d{4}-\d{2}-\d{2})\b", body, re.M)  # the text after the date is free
            if d:
                try:
                    dt.date.fromisoformat(d.group(1))
                except ValueError:
                    continue
                out.add(m.group(1))
        return out
    if not os.path.exists(p):
        return set()
    text = open(p, encoding="utf-8").read()
    m = re.search(r"^## Decided\s*$(.*?)(?=^## |\Z)", text, re.M | re.S)
    return set(re.findall(r"^\|\s*[\d/-]+\s*\|\s*(D\d+)\b", m.group(1), re.M)) if m else set()


def grants(root, cfg, strict=False):
    """the grant rows of a problem. A broken row opens nothing; so does an unreadable file (bad bytes), unless strict (the grant command, which must not overwrite it)"""
    p = grant_file(root, cfg)
    if not os.path.exists(p):
        return []
    try:
        with open(p, newline="", encoding="utf-8") as f:
            return [g for g in csv.DictReader(f) if all(g.get(k) for k in ("until", "path", "decision"))]
    except (OSError, ValueError, csv.Error):  # UnicodeDecodeError is a ValueError
        if strict:
            raise
        return []


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
    """the file text after the edit, where the hook can reproduce it (Write, Edit, MultiEdit and the MCP write_file / edit_block); None when it
    cannot (e.g. move_file)"""
    mcp = tool.startswith("mcp__")
    if tool == "Write" or (mcp and tool.endswith("write_file") and inp.get("mode", "rewrite") == "rewrite"):
        return inp.get("content")
    if mcp and tool.endswith("write_file") and inp.get("mode") != "append":
        return None
    edits = (inp.get("edits") if tool == "MultiEdit" else [inp] if tool == "Edit" or (mcp and tool.endswith("edit_block"))
             else [] if mcp and tool.endswith("write_file") else None)
    if edits is None:
        return None
    try:
        text = open(path, encoding="utf-8").read()
    except OSError:
        return None
    if not edits:  # write_file in append mode
        return text + (inp.get("content") or "")
    for e in edits:
        o, n = e.get("old_string", ""), e.get("new_string", "")
        if not o or o not in text:
            return None  # e.g. a line-ending difference Claude Code normalises: the result is unknown
        k = -1 if e.get("replace_all") else int(e.get("expected_replacements", 1) or 1)
        text = text.replace(o, n, k)
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


def targets(tool, inp):
    """the paths a tool call writes (a move writes its source and its destination); empty for a tool that writes no file"""
    if tool in TOOLS:
        keys = ("file_path", "notebook_path")
    elif tool.startswith("mcp__") and tool.endswith(MCP_WRITERS):
        keys = ("path", "file_path", "source", "destination", "outputPath")
    else:
        return []
    return [inp[k] for k in keys if isinstance(inp.get(k), str) and inp[k]]


def verdict(tool, inp, root=None, cfg=None, now=None, cwd=None):
    """None = allowed, else the reason for the refusal. Every problem folder of the workspace is asked (D75): a path is refused when the protected list of ANY of them
    covers it and no grant of that problem opens it; root (and cfg) name ONE problem instead (tests, tools). A relative path is read against the event's cwd;
    without one against both the hook's own folder and the problem folders (an MCP server resolves against its own folder, which the hook cannot know)"""
    probs = problems() if root is None else [(root, cfg if cfg is not None else load_cfg(root))]
    for raw in targets(tool, inp):
        m = re.match(r"^([A-Za-z]):(?![\\/])", raw)
        d = re.match(r"^([A-Za-z]):[\\/]", cwd or "")
        if m and not (d and m.group(1).lower() == d.group(1).lower()):
            return f"guard_hook: {raw} is relative to the current folder of drive {m.group(1)}:, which the hook cannot know: write the full path"
        roots = [r for r, _ in probs]
        bases = [cwd] if cwd else [os.getcwd(), WS] + roots
        if cwd and tool.startswith("mcp__"):
            bases += [WS] + roots  # an MCP server resolves a relative path against ITS folder, which the event's cwd does not tell
        for c in bases:
            for r, g in probs:
                why = verdict_path(tool, inp, norm(raw, c), r, g, now)
                if why:
                    return why
    return None


def verdict_path(tool, inp, path, root, cfg, now):
    move = tool.endswith("move_file")  # a move of a FOLDER takes every protected path inside it along (the folder of the hook itself, a problem folder)
    grant = norm(grant_file(root, cfg))
    if path == grant or (move and covers(path, grant)):
        return "guard_hook: the grant file is written only by `guard_hook.py grant` (a decided D<n> behind each row)"
    for rel, prot, folder, entries in gate_rules(root, cfg):
        above = move and path != prot and covers(path, prot)
        if not above and not (covers(prot, path) if folder else path == prot):
            continue
        what = rel
        if above:
            what += " (inside the folder that is moved)"
        elif entries:
            changed = entries_changed(path, new_text(tool, inp, path), entries)
            if not changed:
                return None
            what += f" (entries {', '.join(changed)})"
        if granted(prot if above else path, root, cfg, now):
            if above:
                continue  # this protected path is open; the others inside the folder are still asked
            return None
        return (f"guard_hook: {what} is protected (a decision of the person, the decisions record). If a decided entry covers this edit, record a grant first: "
                f"`python3 ../autoresearch/tools/guard_hook.py grant {rel} --decision D<n> --hours 8 --note \"...\"` (from the folder {os.path.basename(os.path.normpath(root))}/, in the WSL); "
                f"otherwise write the question into the decisions record (a file D<n>.md in the decisions folder, or DECISIONS.md) and wait for the answer. Do not go around this through the shell.")
    return None


def cli_base():
    """the folder a relative path of the grant command is read against: AR_PROJECT when set, else the nearest folder above the cwd that holds a project.json
    (the problem you are in), else the default problem of config.json"""
    if os.environ.get("AR_PROJECT"):
        return default_root()
    d = os.getcwd()
    while d != os.path.dirname(d):
        if os.path.isfile(os.path.join(d, "project.json")):
            return d
        d = os.path.dirname(d)
    return default_root()


def add_grant(path, decision, hours, note, root=None, cfg=None):
    """one row per problem whose protected list the path covers or contains, each in that problem's own grant file and relative to its folder; the decision must be
    decided in each of them (the decisions folder may be shared). root (and cfg) name ONE problem (tests, tools)"""
    probs, base = (problems(), cli_base()) if root is None else ([(root, cfg if cfg is not None else load_cfg(root))], root)
    target = os.path.normpath(os.path.join(base, path))
    full = norm(os.path.join(base, path))
    hits = []
    for r, c in probs:
        h = [rel for rel, prot, folder, _ in gate_rules(r, c) if covers(prot, full) or (full != norm(r) and covers(full, prot) and not covers(full, norm(r)))]
        if h:
            hits.append((r, c, h))
    for r, c, _ in hits:
        if decision not in decided(r, c):
            sys.exit(f"{decision} is not decided for {os.path.basename(os.path.normpath(r))} (no 'Decided:' line in its file of the decisions folder, no row in the "
                     f"Decided table of DECISIONS.md): no grant")
    if not hits:
        sys.exit(f"{path} ({full}) is no protected path and lies inside none; give it relative to the problem folder, e.g. ../autoresearch/tools/ "
                 f"(protected: {', '.join(rel for r, c in probs for rel, *_ in gate_rules(r, c))})")
    now = dt.datetime.now()
    stamp = dict(granted=now.isoformat(timespec="minutes"), until=(now + dt.timedelta(hours=hours)).isoformat(timespec="minutes"), decision=decision, note=note)
    for r, c, h in hits:
        rel = path if os.path.normpath(r) == os.path.normpath(base) else os.path.relpath(target, r).replace(os.sep, "/") + ("/" if path.endswith(("/", os.sep)) else "")
        p = grant_file(r, c)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with _locked(p):
            buf = io.StringIO()
            w = csv.DictWriter(buf, fieldnames=GCOLS, extrasaction="ignore", lineterminator="\n")
            w.writeheader()
            w.writerows(grants(r, c, strict=True) + [dict(stamp, path=rel)])
            _atomic_write(p, buf.getvalue())
        print(f"grant: {rel} -> {full} until {stamp['until']} ({decision}; {os.path.basename(os.path.normpath(r))} opens {', '.join(h)})")


def deny(reason):
    """the refusal as Claude Code reads it: JSON on stdout with exit 0 (a crash that prints nothing lets the edit through: decide_event turns a crash into a refusal)"""
    print(json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny", "permissionDecisionReason": reason}}))


def crash_verdict(tool, inp, cwd=None):
    """what the hook decides after a crash of the decision: refuse a path that names a protected file, by what can still be read. First the protected lists of the
    problem folders that still load (every entry counts as changed, a grant is not looked at), then the fixed names of GATE_NAMES (the gate's own files and the
    truth files). A crash must not open the gate. None = the paths name nothing protected (or the event has no path)"""
    try:
        raws = targets(tool, inp)
    except Exception:  # noqa: BLE001  a malformed event: nothing to judge
        return None
    move = tool.endswith("move_file")
    try:
        names = sorted(n for n in os.listdir(WS) if os.path.isfile(os.path.join(WS, n, "project.json")))
    except OSError:
        names = []
    anchors = [norm(a) for a in [os.path.join(HERE, "guard_hook.py"), os.path.join(HERE, "project.py"), os.path.join(AR, "config.json"),
                                 os.path.join(WS, ".claude", "settings.json"), os.path.join(WS, ".claude", "settings.local.json")]
               + [os.path.join(WS, n, "project.json") for n in names]]  # a move of a folder above any of them takes the gate along
    for raw in raws:
        paths = {norm(raw, c) for c in ([cwd] if cwd else [os.getcwd(), WS])}
        if move and any(covers(p, a) for p in paths for a in anchors):
            return f"{raw} is a folder that holds the hook, a gate file or a project.json"
        for name in names:
            try:
                d = os.path.join(WS, name)
                for rel, prot, folder, _ in gate_rules(d, load_cfg(d)):
                    if any((covers(prot, p) if folder else p == prot) or (move and covers(p, prot)) for p in paths):
                        return f"{raw} is on the protected list of {name} ({rel})"
            except Exception:  # noqa: BLE001  this problem's list cannot be read: the fixed names below still apply
                continue
        if any(GATE_NAMES.search(p) for p in paths):
            return f"{raw} names a gate file or a truth file"
    return None


def log_crash(exc, tool, inp):
    """one entry in autoresearch/state/guard_hook_crash.log (not in git); a failing log changes nothing"""
    try:
        try:
            where = targets(tool, inp)
        except Exception:  # noqa: BLE001
            where = "?"
        os.makedirs(os.path.dirname(CRASH_LOG), exist_ok=True)
        with open(CRASH_LOG, "a", encoding="utf-8") as f:
            f.write(f"{dt.datetime.now().isoformat(timespec='seconds')} {tool} {where}\n{''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))}\n")
    except Exception:  # noqa: BLE001
        pass


def decide_event(ev):
    """the reason to refuse a hook event, or None. A crash of the decision (an unreadable project.json or grant file, a SystemExit, anything) refuses a path that
    names a protected file (crash_verdict) and is logged; a path that names none passes"""
    tool, inp, cwd = ev.get("tool_name", ""), ev.get("tool_input") or {}, ev.get("cwd")
    try:
        return verdict(tool, inp, cwd=cwd)
    except (Exception, SystemExit) as e:  # noqa: BLE001  SystemExit too: a missing project.json ends a tool with sys.exit, and that must not open the gate
        log_crash(e, tool, inp)
        try:
            why = crash_verdict(tool, inp, cwd)
        except Exception as e2:  # noqa: BLE001
            why = f"the crash handler failed too ({type(e2).__name__}: {e2})"
        return (f"guard_hook: the gate crashed ({type(e).__name__}: {e}) and refuses what it cannot check: {why}. The crash is logged in {CRASH_LOG}. Fix the cause "
                f"from a shell (a broken project.json, a bad grant file) or ask the person; do not go around this.") if why else None


def selftest():
    import shutil
    import tempfile
    assert norm("C:\\ML-Experiment\\x\\PROBLEM.md") == "/mnt/c/ml-experiment/x/problem.md"
    assert norm("\\\\?\\C:\\ML-Experiment\\x\\PROBLEM.md") == norm("C:/ml-experiment/X/problem.md")
    assert norm("C:\\ML-Experiment\\x\\PROBLEM.md.") == norm("C:\\ML-Experiment\\x\\PROBLEM.md::$DATA") == norm("/mnt/c/ML-Experiment/x/PROBLEM.md")
    assert norm("x\\PROBLEM.md", cwd="C:\\ML-Experiment") == norm("/mnt/c/ML-Experiment/x/PROBLEM.md")
    assert norm("/mnt/c/a/b/../c.md") == "/mnt/c/a/c.md"
    want = "/mnt/c/ml-experiment/x/problem.md"  # D33 (b): share, doubled-slash and drive-relative spellings
    for s in ("\\\\wsl$\\Ubuntu\\mnt\\c\\ML-Experiment\\x\\PROBLEM.md", "\\\\wsl.localhost\\Ubuntu-22.04\\mnt\\c\\ML-Experiment\\x\\PROBLEM.md",
              "\\\\localhost\\C$\\ML-Experiment\\x\\PROBLEM.md", "\\\\?\\UNC\\localhost\\c$\\ML-Experiment\\x\\PROBLEM.md",
              "//mnt/c/ML-Experiment/x/PROBLEM.md", "///mnt/c/ML-Experiment/x/PROBLEM.md"):
        assert norm(s) == want, (s, norm(s))
    assert norm("\\ML-Experiment\\x\\PROBLEM.md", cwd="C:\\other") == want, "a drive-relative path is on the drive of cwd"
    assert norm("\\ML-Experiment\\x\\PROBLEM.md", cwd="D:\\y") == "/mnt/d/ml-experiment/x/problem.md"
    assert norm("/tmp/x.md", cwd="/mnt/c/y") == "/tmp/x.md", "from a Linux cwd a rooted path is a Linux path"
    assert norm("/home/u/x.md", cwd="/home/u") == "/home/u/x.md", "a Linux path stays one"
    assert norm("/c/ML-Experiment/x/PROBLEM.md", cwd="C:\\other") == want, "the Git Bash spelling from a Windows cwd"
    assert norm("/c/x.md", cwd="/home/u") == "/c/x.md", "from a Linux cwd /c/x is a Linux path"
    assert norm("C:x\\PROBLEM.md", cwd="C:\\ML-Experiment") == want, "C:x is relative to the cwd of drive C:"
    calls = []
    real_long = globals()["long_name"]
    globals()["long_name"] = lambda q: calls.append(q) or q
    try:
        norm("C:\\AUTO-R~1\\x.md")
        norm("C:\\Auto-research\\x.md")
    finally:
        globals()["long_name"] = real_long
    assert calls == ["/mnt/c/AUTO-R~1/x.md"], "norm expands a short name of a /mnt/ path, and only then"
    root = tempfile.mkdtemp()
    os.makedirs(os.path.join(root, "LONGNAME"))
    os.symlink(os.path.join(root, "LONGNAME"), os.path.join(root, "LONGNA~1"))  # stands in for an 8.3 alias: another name of the same folder
    assert long_name(os.path.join(root, "LONGNA~1", "f.md")) == os.path.join(root, "LONGNAME", "f.md"), "a short name is expanded"
    assert long_name(os.path.join(root, "NOSUCH~1", "f.md")) == os.path.join(root, "NOSUCH~1", "f.md"), "an unknown short name stays"
    shutil.rmtree(root)
    root = tempfile.mkdtemp()
    os.makedirs(os.path.join(root, "experiment_execution"))
    os.makedirs(os.path.join(root, "tools"))
    open(os.path.join(root, "experiment_execution/DECISIONS.md"), "w").write(
        "## Open\n| D9 | x |\n\n## Decided\n| date | decision |\n|---|---|\n| 2026-10-25 | D8: yes, see also D9 and D3 |\n\n## Notes\n| 2026-10-26 | D4: not a decision |\n")
    open(os.path.join(root, "reg.json"), "w").write(json.dumps({"champion": {"run": "a"}, "fast": {"run": "b"}}))
    cfg = dict(protected={"TEST_LOG.md": None, "tools/": None, "reg.json": ["champion"]}, grants="experiment_execution/grants.csv",
               decisions="experiment_execution/DECISIONS.md")
    assert decided(root, cfg) == {"D8"}, decided(root, cfg)
    # D69: a folder of decision files D<n>.md; decided only by a line that STARTS with "Decided: <yyyy-mm-dd>", the file name gives the id
    dd = os.path.join(root, "decisions")
    os.makedirs(dd)
    for name, body in (("D7.md", "# D7\nQuestion: x\nDecided: 2026-11-10 yes\n"), ("D9.md", "# D9\nQuestion: still open\n"),
                       ("D5.md", "# D5\nnot Decided: 2026-11-10 in the middle of a line\n"), ("D6.md", "Decided: soon\n"),
                       ("notes.md", "Decided: 2026-11-10 D3\n"), ("D12.txt", "Decided: 2026-11-10\n")):
        open(os.path.join(dd, name), "w", encoding="utf-8").write(body)
    for name, body in (("D2.md", "# D2\r\nDecided:\t2026-11-10 tab and CRLF\r\n"), ("D3.md", "# D3\n```\nDecided: 2026-11-10 inside an example\n```\n"),
                       ("D4.md", "# D4\nDecided: 2026-99-99 no such date\n"), ("D1.md", "﻿Decided: 2026-11-10 a BOM before the first line is not read: undecided, the safe side\n"),
                       ("D10.md", "# D10\n```\nexample\n```\nDecided: 2026-11-10 after a code block\n")):
        open(os.path.join(dd, name), "w", encoding="utf-8", newline="").write(body)
    open(os.path.join(dd, "D8.md"), "wb").write(b"# D8\n\xff\xfe Decided: 2026-11-10\n")  # bad bytes: skipped, the gate stays up
    os.makedirs(os.path.join(dd, "D11.md"))  # a folder named like a decision file: skipped
    cfg_dir = dict(cfg, decisions="decisions")
    assert decided(root, cfg_dir) == {"D7", "D2", "D10"}, decided(root, cfg_dir)
    for bad in ("D9", "D5", "D6", "D3", "D12", "D4", "D1", "D8", "D11"):
        try:
            add_grant("TEST_LOG.md", bad, 1, "open, undated, misplaced or misnamed", root, cfg_dir)
            raise AssertionError(f"a grant needs a decided file: {bad}")
        except SystemExit:
            pass
    assert decided(root, dict(cfg, decisions="no_such_folder_or_file")) == set(), "a missing record decides nothing"
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
    dc = "mcp__desktop-commander__"  # D33 (a), (c), (d)
    assert v(dc + "move_file", source=os.path.join(root, "TEST_LOG.md"), destination=os.path.join(root, "x.md")), "moving a protected file away"
    assert v(dc + "move_file", source=os.path.join(root, "x.md"), destination=os.path.join(root, "TEST_LOG.md")), "moving onto a protected file"
    assert v(dc + "move_file", source=os.path.join(root, "x.md"), destination=os.path.join(root, "y.md")) is None, "other moves pass"
    assert v(dc + "move_file", source=reg, destination=os.path.join(root, "r2.json")), "moving the registry away changes its champion"
    here = os.getcwd()
    os.chdir(os.path.dirname(root))  # as in production: the hook runs in the workspace, the parent of the problem folder
    try:
        assert verdict("Edit", dict(file_path=os.path.join(os.path.basename(root), "TEST_LOG.md")), root, cfg), "relative to the hook's own folder"
        assert verdict("Edit", dict(file_path="TEST_LOG.md"), root, cfg), "relative to the problem folder (an MCP server's own cwd is unknown)"
        assert verdict("Edit", dict(file_path="other.md"), root, cfg) is None
    finally:
        os.chdir(here)
    assert verdict("Edit", dict(file_path="D:x.md"), root, cfg, cwd="C:\\w"), "C:x on another drive than cwd's is refused (its folder is unknown)"
    assert v(dc + "write_pdf", path=os.path.join(root, "in.md"), outputPath=os.path.join(root, "TEST_LOG.md")), "write_pdf writes its outputPath"
    assert v(dc + "write_file", path=reg, content=" ", mode="append") is None, "an append that keeps the json and its champion passes"
    assert v(dc + "write_file", path=reg, content="{}", mode="weird"), "an unknown write mode is refused"
    open(reg, "w").write(json.dumps({"fast": {"run": "b"}, "champion": {"run": "b"}}))
    assert v(dc + "edit_block", path=reg, old_string='"run": "b"', new_string='"run": "c"', expected_replacements=2), "the 2nd match is the champion"
    assert v(dc + "edit_block", path=reg, old_string='"run": "b"', new_string='"run": "c"') is None, "one replacement touches only the first entry"
    open(reg, "w").write(json.dumps({"champion": {"run": "a"}, "fast": {"run": "b"}}))
    assert v(dc + "edit_block", path=reg, old_string='"run": "b"', new_string='"run": "c"') is None, "an MCP edit of another entry passes"
    assert v(dc + "edit_block", path=reg, old_string='"run": "a"', new_string='"run": "z"'), "an MCP edit of the champion is refused"
    assert v(dc + "write_file", path=reg, content=json.dumps({"champion": {"run": "a"}, "fast": {"run": "c"}})) is None, "an MCP rewrite of another entry"
    assert v(dc + "write_file", path=reg, content=json.dumps({"champion": {"run": "z"}})), "an MCP rewrite of the champion is refused"
    assert v(dc + "write_file", path=reg, content="{}", mode="append"), "an MCP append that breaks the json is refused"
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
    selftest_problems()
    print("guard_hook selftest passed")


def selftest_problems():
    """D75: every problem folder of the workspace is guarded (hole 1: the second problem's list is enforced, its decisions folder may be shared, a grant lands in
    the grant file of the problem that protects the path) and a crash of the decision refuses protected names (hole 2: a half-written project.json, a SystemExit, a
    syntax error in project.py, an unreadable grant file)"""
    import shutil
    import subprocess
    import tempfile
    ws = tempfile.mkdtemp()
    here, saved = os.getcwd(), (globals()["WS"], globals()["CRASH_LOG"], globals()["problems"], globals()["AR"], globals()["HERE"])
    pj = lambda rel: os.path.join(ws, rel)  # noqa: E731
    try:
        saved_real = [(r, c) for r, c in problems(saved[0])]  # the lists of the real workspace (none when it cannot be read: the check below is skipped)
    except (OSError, ValueError):
        saved_real = []

    def w(rel, text="x\n"):
        os.makedirs(os.path.dirname(pj(rel)), exist_ok=True)
        with open(pj(rel), "w", encoding="utf-8") as f:
            f.write(text)

    lane = {"protected": {"PROBLEM.md": None, "../autoresearch/tools/guard_hook.py": None, "../shared.md": None, "../bev/records/extra.md": None, "models/special.cfg": None},
            "decisions": "decisions", "grants": "grants.csv"}
    bev = {"name": "bev", "splits": ["val", "test"], "holdout_split": "test",
           "protected": {"project.json": ["splits", "holdout_split", "protected", "grants", "decisions", "grant_decisions"], "records/champion.json": ["base", "promoted", "options"],
                         "records/TEST_USAGE_LOG.md": None, "../autoresearch/config.json": None, "../autoresearch/tools/project.py": None, "../shared.md": None,
                         "records/extra.md": None},
           "grant_decisions": "../lane/decisions", "grants": "records/grants.csv"}  # no "decisions": tools/testgate.py reads that key, a TEST run of bev must not open by a lane decision
    w("lane/project.json", json.dumps(lane))
    w("bev/project.json", json.dumps(bev, indent=1))
    w("lane/decisions/D1.md", "# D1\nDecided: 2026-11-10 yes\n")
    w("lane/decisions/D2.md", "# D2\nstill open\n")
    for rel in ("lane/PROBLEM.md", "shared.md", "bev/records/extra.md", "bev/readme.md", "lane/models/special.cfg"):
        w(rel)
    w("autoresearch/config.json", "{}")
    w("autoresearch/tools/project.py", "x = 1\n")
    champ = pj("bev/records/champion.json")
    w("bev/records/champion.json", json.dumps({"base": "t8", "promoted": None, "options": [], "open": ["a"]}))
    globals()["WS"], globals()["CRASH_LOG"] = ws, pj("autoresearch/state/guard_hook_crash.log")
    globals()["AR"], globals()["HERE"] = pj("autoresearch"), pj("autoresearch/tools")
    v = lambda tool, **inp: verdict(tool, inp)  # noqa: E731  every problem of the workspace
    ev = lambda tool, **inp: {"tool_name": tool, "tool_input": inp, "cwd": ws}  # noqa: E731
    try:
        assert [os.path.basename(r) for r, _ in problems()] == ["bev", "lane"], "every sub-folder with a project.json is a problem"
        assert decided(pj("bev"), bev) == {"D1"} and "decisions" not in bev, "the hook's decisions record of bev is the lane folder, through its own key"
        for r, c in saved_real:  # the fixed names of the crash handler cover every file rule of the real lists: a new protected file needs a new name there
            for rel, prot, folder, entries in rules(r, c):
                assert folder or GATE_NAMES.search(prot), f"GATE_NAMES must name {rel} of {r} (the crash handler refuses by name when it cannot read the lists)"
        for bad in ({"protected": "records/champion.json"}, {"protected": {"f.json": "promoted"}}, {"protected": {"f.json": [3]}}, {"protected": [1]}, {"protected": 5}):
            try:
                rules(ws, bad)
                raise AssertionError(f"a protected list of the wrong type must raise: {bad}")
            except ValueError:
                pass
        # hole 1: the second problem is guarded
        assert v("Write", file_path=pj("bev/project.json"), content=json.dumps(dict(bev, holdout_split="val"))), "an edit of holdout_split of the bev project.json is refused"
        assert v("Edit", file_path=pj("bev/project.json"), old_string='"holdout_split": "test"', new_string='"holdout_split": "val"'), "...also as an Edit"
        assert v("Write", file_path=pj("bev/project.json"), content=json.dumps(dict(bev, protected={}))), "emptying the protected list is refused"
        assert v("Write", file_path=pj("bev/project.json"), content=json.dumps(dict(bev, name="other"))) is None, "the other entries of project.json stay free"
        assert v("Write", file_path=champ, content=json.dumps({"base": "t8", "promoted": {"by": "x"}, "options": [], "open": ["a"]})), "promoting the champion is refused"
        assert v("Write", file_path=champ, content=json.dumps({"base": "t8", "promoted": None, "options": [], "open": ["a", "b"]})) is None, "the open list is free"
        assert v("Write", file_path=pj("bev/records/TEST_USAGE_LOG.md"), content="x"), "the TEST log of the bev problem (not there yet) is protected"
        assert v("Write", file_path=pj("autoresearch/config.json"), content="{}"), "config.json (default_project) is protected"
        assert v("Write", file_path=pj("autoresearch/tools/project.py"), content="x = 2"), "project.py is protected"
        assert v("Write", file_path=pj("autoresearch/tools/guard_hook.py"), content="x"), "a path of the lane list is refused whatever the default problem is"
        assert v("Write", file_path=pj("lane/PROBLEM.md"), content="x"), "...and the lane problem still is"
        assert v("Write", file_path=pj("shared.md"), content="x"), "a path of both lists is refused"
        assert v("Write", file_path=pj("bev/records/grants.csv"), content="x"), "the grant file of the bev problem is written only by the grant command"
        assert v("Write", file_path=pj("bev/readme.md"), content="x") is None and v("Write", file_path=pj("lane/other.md"), content="x") is None, "other files pass"
        mv = "mcp__desktop-commander__move_file"  # a move of a folder takes the protected paths inside it along
        for folder in ("autoresearch/tools", "autoresearch", "bev/records", "bev", "lane", ""):
            assert v(mv, source=pj(folder), destination=pj("moved")), f"moving the folder {folder!r} that holds protected paths is refused"
        assert v(mv, source=pj("bev/readme.md"), destination=pj("bev/readme2.md")) is None, "moving an unprotected file passes"
        os.makedirs(pj("bev/training"))
        assert v(mv, source=pj("bev/training"), destination=pj("bev/training2")) is None, "moving a folder with nothing protected inside passes"
        assert verdict("mcp__desktop-commander__write_file", dict(path="bev/records/champion.json", content="{}", mode="rewrite"), cwd=pj("bev")), \
            "an MCP relative path is also read against the workspace (the server's own folder is not the event's cwd)"
        w("third/project.json", json.dumps({"name": "t", "grants": "gr/grants.csv"}))  # a problem WITHOUT a protected list: its gate keys are protected all the same
        w("third/gr/grants.csv", "granted,until,path,decision,note\n")
        assert v("Write", file_path=pj("third/project.json"), content=json.dumps({"name": "t", "grants": "gr/grants.csv", "grant_decisions": "../lane/decisions"})), \
            "a problem without a list: an edit that adds grant_decisions is refused"
        assert v("Write", file_path=pj("third/project.json"), content=json.dumps({"name": "u", "grants": "gr/grants.csv"})) is None, "...its other keys are free"
        assert v(mv, source=pj("third/gr"), destination=pj("third/gr2")), "moving the folder that holds a grant file is refused even when nothing else is protected in it"
        shutil.rmtree(pj("third"))
        w("fourth/project.json", json.dumps({"grants": "g.csv", "grant_decisions": "../lane/decisions", "protected": {"a/x.md": None, "a/y.md": None}}))
        w("fourth/a/x.md")
        w("fourth/a/y.md")
        os.chdir(pj("fourth"))
        add_grant("a/x.md", "D1", 1, "one of two")
        os.chdir(here)
        assert v("Write", file_path=pj("fourth/a/x.md"), content="z") is None, "the granted file is open"
        assert v(mv, source=pj("fourth/a"), destination=pj("fourth/a2")), "one granted path of two does not open the move of the folder that holds both"
        shutil.rmtree(pj("fourth"))
        assert verdict("Write", dict(file_path=pj("bev/project.json"), content="{}"), pj("lane"), lane) is None, "root names ONE problem: the lane list does not protect the bev file"
        # the decisions folder is shared; a grant lands in the file of the problem that protects the path, relative to that problem
        os.chdir(pj("bev/records"))
        assert cli_base() == pj("bev"), "the grant command reads a relative path against the problem folder you are in"
        os.chdir(pj("bev"))
        os.environ["AR_PROJECT"] = "../lane"  # an explicit AR_PROJECT wins over the folder you are in (read against the cwd when autoresearch/ has no such folder)
        try:
            assert cli_base() == pj("lane")
        finally:
            del os.environ["AR_PROJECT"]
        lk = pj("lockfile")
        with _locked(lk):
            assert os.path.exists(lk + ".lock")
        assert not os.path.exists(lk + ".lock")
        os.mkdir(lk + ".lock")  # a lock that cannot be taken over (a folder) times out instead of spinning
        t0 = time.time()
        try:
            with _locked(lk, wait_s=0.6, stale_s=0):
                raise AssertionError("a held lock must not be entered")
        except TimeoutError:
            pass
        assert time.time() - t0 < 10
        os.rmdir(lk + ".lock")
        with open(lk + ".lock", "w", encoding="utf-8") as f:
            f.write("x")
        os.utime(lk + ".lock", (1, 1))
        with _locked(lk, wait_s=0.6, stale_s=60):  # a stale lock is taken over
            pass
        add_grant("records/champion.json", "D1", 1, "selftest")
        assert not os.path.exists(pj("lane/grants.csv")), "the lane list does not cover the champion: no row in its file"
        with open(pj("bev/records/grants.csv"), encoding="utf-8") as f:
            assert [r["path"] for r in csv.DictReader(f)] == ["records/champion.json"]
        assert v("Write", file_path=champ, content=json.dumps({"base": "t8", "promoted": {"by": "x"}, "options": [], "open": ["a"]})) is None, "the grant opens the champion"
        assert v("Write", file_path=pj("bev/records/TEST_USAGE_LOG.md"), content="x"), "...and only that path"
        assert v(mv, source=pj("bev/records"), destination=pj("bev/records_old")), "a grant of ONE protected path does not open the move of the folder around it"
        for bad in ("D2", "D404"):
            try:
                add_grant("project.json", bad, 1, "open or unknown decision")
                raise AssertionError(f"a grant needs a decided decision of the shared folder: {bad}")
            except SystemExit:
                pass
        try:
            add_grant("readme.md", "D1", 1, "not protected")
            raise AssertionError("an unprotected path gets no grant")
        except SystemExit:
            pass
        os.chdir(pj("lane"))
        add_grant("../bev/records/extra.md", "D1", 1, "selftest")  # protected by both problems: one row in each file, each relative to its own folder
        with open(pj("lane/grants.csv"), encoding="utf-8") as f:
            assert [r["path"] for r in csv.DictReader(f)] == ["../bev/records/extra.md"]
        with open(pj("bev/records/grants.csv"), encoding="utf-8") as f:
            assert [r["path"] for r in csv.DictReader(f)] == ["records/champion.json", "records/extra.md"]
        assert v("Write", file_path=pj("bev/records/extra.md"), content="y") is None, "a path of two lists is open when both opened it"
        os.remove(pj("lane/grants.csv"))
        assert v("Write", file_path=pj("bev/records/extra.md"), content="y"), "...and closed again when one problem has no grant"
        add_grant("../autoresearch/tools/", "D1", 1, "selftest")  # a folder grant opens the protected files inside it, in the file of each problem that protects one
        with open(pj("lane/grants.csv"), encoding="utf-8") as f:
            assert [r["path"] for r in csv.DictReader(f)] == ["../autoresearch/tools/"]
        assert v("Write", file_path=pj("autoresearch/tools/guard_hook.py"), content="x") is None and v("Write", file_path=pj("autoresearch/tools/project.py"), content="x = 3") is None
        assert v("Write", file_path=pj("autoresearch/config.json"), content="{}"), "...but not the file beside the folder"
        os.symlink(pj("bev"), pj("aa_bev"))  # a link to a problem folder is asked under its own name too and shares its grant file (one row)
        try:
            assert [os.path.basename(r) for r, _ in problems()] == ["aa_bev", "bev", "lane"]
            log = "records/TEST_USAGE_LOG.md"
            assert v("Write", file_path=pj("bev/" + log), content="x") and v("Write", file_path=pj("aa_bev/" + log), content="x"), "refused through either spelling"
            os.chdir(pj("bev"))
            add_grant(log, "D1", 1, "link")
            os.chdir(pj("lane"))
            with open(pj("bev/records/grants.csv"), encoding="utf-8") as f:
                rows = [r["path"] for r in csv.DictReader(f)]
            assert [x for x in rows if x.endswith(log)] == [log], ("one row for the real folder and its link", rows)
            assert v("Write", file_path=pj("bev/" + log), content="x") is None and v("Write", file_path=pj("aa_bev/" + log), content="x") is None, "open through both"
        finally:
            os.remove(pj("aa_bev"))
        argv, buf = sys.argv, io.StringIO()  # the two read-only commands
        try:
            sys.argv = ["guard_hook.py", "grants"]
            with contextlib.redirect_stdout(buf):
                main()
            assert "bev: " in buf.getvalue() and "lane: " in buf.getvalue(), buf.getvalue()
            for target, code in ((pj("lane/PROBLEM.md"), 2), (pj("bev/readme.md"), 0)):
                sys.argv = ["guard_hook.py", "check", target]
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        main()
                    raise AssertionError("check must exit")
                except SystemExit as e:
                    assert e.code == code, (target, e.code)
        finally:
            sys.argv = argv
        # a row of an undecided decision opens nothing; a grant file with bad bytes opens nothing, and the grant command does not overwrite it
        with open(pj("bev/records/grants.csv"), "w", encoding="utf-8") as f:  # a row written by hand for an OPEN decision (D2) or an unknown one opens nothing
            f.write("granted,until,path,decision,note\n2026-11-10T00:00,2999-01-01T00:00,records/champion.json,D2,open decision\n"
                    "2026-11-10T00:00,2999-01-01T00:00,records/champion.json,D404,unknown decision\n")
        assert v("Write", file_path=champ, content=json.dumps({"base": "t8", "promoted": {"by": "x"}, "options": [], "open": ["a"]})), "a grant of an undecided decision opens nothing"
        with open(pj("bev/records/grants.csv"), "w", encoding="utf-8") as f:
            f.write("granted,until,path,decision,note\n2026-11-10T00:00,2999-01-01T00:00,records/champion.json,D1,decided\n")
        assert v("Write", file_path=champ, content=json.dumps({"base": "t8", "promoted": {"by": "x"}, "options": [], "open": ["a"]})) is None, "...a decided one does"
        with open(pj("bev/records/grants.csv"), "wb") as f:
            f.write(b"granted,until,path,decision,note\n\xff\xfe\n")
        assert grants(pj("bev"), bev) == [], "an unreadable grant file opens nothing"
        assert v("Write", file_path=champ, content=json.dumps({"base": "t8", "promoted": {"by": "x"}, "options": [], "open": ["a"]})), "...so the champion is closed again"
        try:
            add_grant("../bev/records/champion.json", "D1", 1, "must not overwrite")
            raise AssertionError("the grant command must stop at an unreadable grant file")
        except ValueError:
            pass
        os.chdir(here)
        # hole 2: a crash of the decision refuses protected names, passes the rest, and is logged
        good = open(pj("bev/project.json"), encoding="utf-8").read()
        with open(pj("bev/project.json"), "w", encoding="utf-8") as f:
            f.write(good[:25])  # a half-written project.json
        assert not os.path.exists(globals()["CRASH_LOG"])
        for path in ("bev/project.json", "bev/records/champion.json", "lane/PROBLEM.md", "autoresearch/config.json", "autoresearch/tools/project.py", "bev/records/TEST_USAGE_LOG.md"):
            r = decide_event(ev("Write", file_path=pj(path), content="{}"))
            assert r and "crashed" in r, (path, r)  # by name (bev's own list cannot be read) or by the lane list that still loads
        r = decide_event(ev("Write", file_path=pj("lane/models/special.cfg"), content="x"))
        assert r and "crashed" in r and "protected list of lane" in r, ("a path that only a readable list names is refused by that list", r)
        assert decide_event(ev("Write", file_path=pj("bev/readme.md"), content="x")) is None, "a crash passes a path that names nothing protected"
        assert decide_event(ev("Read", file_path=pj("bev/project.json"))) is None and decide_event({"tool_name": "Write", "tool_input": ["x"]}) is None, "no path, nothing to refuse"
        r = decide_event({"tool_name": "Write", "tool_input": {"file_path": "lane/models/special.cfg", "content": "x"}})  # no cwd in the event: a relative path is read against the workspace too
        assert r and "protected list of lane" in r, ("a relative path without a cwd", r)
        for folder in ("autoresearch/tools", "autoresearch", "bev", "lane", "bev/records", ""):  # a move of a folder that holds the hook or a protected path, in crash mode
            r = decide_event(ev(mv, source=pj(folder), destination=pj("moved")))
            assert r and "crashed" in r, ("a folder move in crash mode", folder, r)
        assert decide_event(ev(mv, source=pj("bev/readme.md"), destination=pj("bev/readme2.md"))) is None, "an unrelated move passes in crash mode"
        lane_good = open(pj("lane/project.json"), encoding="utf-8").read()
        with open(pj("lane/project.json"), "w", encoding="utf-8") as f:
            f.write(lane_good[:25])  # now NO list can be read: the fixed anchors alone must keep the hook and the project.json files from being moved away
        try:
            for folder in ("autoresearch/tools", "autoresearch", "bev", "lane", ""):
                r = decide_event(ev(mv, source=pj(folder), destination=pj("moved")))
                assert r and "crashed" in r and "folder that holds" in r, ("no list readable: a folder move", folder, r)
            assert decide_event(ev(mv, source=pj("bev/readme.md"), destination=pj("bev/readme2.md"))) is None
        finally:
            with open(pj("lane/project.json"), "w", encoding="utf-8") as f:
                f.write(lane_good)
        assert "JSONDecodeError" in open(globals()["CRASH_LOG"], encoding="utf-8").read(), "the crash is logged"
        globals()["problems"] = lambda ws=None: sys.exit("no project.json")  # a SystemExit is a crash too
        assert "crashed" in (decide_event(ev("Write", file_path=pj("autoresearch/config.json"), content="{}")) or ""), "SystemExit refuses a protected name"
        assert decide_event(ev("Write", file_path=pj("bev/readme.md"), content="x")) is None
        globals()["problems"] = saved[2]
        with open(pj("bev/project.json"), "w", encoding="utf-8") as f:
            f.write(good)
        assert decide_event(ev("Write", file_path=pj("bev/project.json"), content=json.dumps(dict(bev, holdout_split="val")))), "repaired: the normal refusal is back"
        assert "crashed" not in decide_event(ev("Write", file_path=pj("bev/project.json"), content=json.dumps(dict(bev, holdout_split="val"))))
        # the real script as the hook runs it: a syntax error in project.py or a half-written project.json must not open the gate (exit 0, a JSON deny)
        tools = pj("autoresearch/tools")
        hookpy = os.path.join(tools, "guard_hook.py")
        shutil.copy(os.path.abspath(__file__), hookpy)
        w("autoresearch/tools/project.py", "def broken(:\n")

        def run_hook(event, raw=None):
            p = subprocess.run([sys.executable, hookpy], input=raw or json.dumps(event), capture_output=True, text=True, check=False)
            return p.returncode, p.stdout.strip()
        w("autoresearch/tools/json.py", "raise ImportError('a file named like a standard module')\n")  # beside the hook: must not shadow the real json
        rc, out = run_hook(ev("Write", file_path=pj("lane/PROBLEM.md"), content="x"))
        assert rc == 0 and '"deny"' in out, ("a json.py beside the hook does not switch it off", rc, out)
        os.remove(pj("autoresearch/tools/json.py"))
        rc, out = run_hook(None, '{"tool_name": "Write", "tool_input": {"file_path": "x", "y": ' + "[" * 5000 + "]" * 5000 + "}}")
        assert rc == 0 and '"deny"' in out, ("an event nested thousands deep is refused, not crashed on", rc, out)
        with open(pj("bev/records/grants.csv"), "w", encoding="utf-8") as f:
            f.write("granted,until,path,decision,note\n")
        rc, out = run_hook(ev("Write", file_path=champ, content=json.dumps({"base": "t9", "promoted": None, "options": [], "open": []})))
        assert rc == 0 and '"permissionDecision": "deny"' in out and "crashed" not in out, ("a broken project.py does not switch the hook off", rc, out)
        rc, out = run_hook(ev("Write", file_path=pj("bev/readme.md"), content="x"))
        assert (rc, out) == (0, ""), (rc, out)
        rc, out = run_hook(ev("Write", file_path=pj("autoresearch/tools/project.py"), content="x = 1"))
        assert rc == 0 and '"deny"' in out, "the broken project.py itself is protected"
        gp = subprocess.run([sys.executable, hookpy, "grant", "../autoresearch/tools/project.py", "--decision", "D1", "--hours", "1", "--note", "repair a broken project.py"],
                            cwd=pj("bev"), capture_output=True, text=True, check=False)
        assert gp.returncode == 0 and "grant: ../autoresearch/tools/project.py" in gp.stdout, ("the grant command needs no project.py", gp.returncode, gp.stdout, gp.stderr[-300:])
        rc, out = run_hook(ev("Write", file_path=pj("autoresearch/tools/project.py"), content="x = 1"))
        assert (rc, out) == (0, ""), ("...and the granted repair of the broken project.py goes through", rc, out)
        with open(pj("bev/project.json"), "w", encoding="utf-8") as f:
            f.write(good[:25])
        rc, out = run_hook(ev("Write", file_path=pj("bev/project.json"), content="{}"))
        assert rc == 0 and '"deny"' in out and "crashed" in out, ("a half-written project.json", rc, out)
        assert os.path.getsize(pj("autoresearch/state/guard_hook_crash.log")) > 0
    finally:
        globals()["WS"], globals()["CRASH_LOG"], globals()["problems"], globals()["AR"], globals()["HERE"] = saved
        os.chdir(here)
        shutil.rmtree(ws)


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
        for r, c in problems():
            for g in grants(r, c):
                print(f"{os.path.basename(os.path.normpath(r))}: {g['granted']} -> {g['until']}  {g['decision']:4} {g['path']}  {g['note']}")
        return 0
    if a and a[0] == "check":
        why = verdict("Write", dict(file_path=os.path.abspath(a[1])))  # unknown content: every protected entry counts as changed
        print(why or "allowed")
        sys.exit(2 if why else 0)
    try:
        ev = json.load(sys.stdin)
    except ValueError:
        return 0  # not an event
    except RecursionError:  # an event nested thousands deep is no edit of ours: refuse it (the crash would print nothing and let it through)
        deny("guard_hook: the hook event is nested too deeply to read, so it is refused (no tool call of Claude Code looks like this)")
        return 0
    why = decide_event(ev if isinstance(ev, dict) else {})
    if why:
        deny(why)
    return 0


if __name__ == "__main__":
    main()
