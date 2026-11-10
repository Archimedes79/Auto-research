"""Copy the agent definitions (autoresearch/agents/*.md) into the workspace's .claude/agents/ so Claude Code sessions started in the workspace
can spawn them by name (ar-planner, ar-critic, ...). Run again after editing an agent. Standard library only.

    python autoresearch/tools/install_agents.py [--workspace <folder>]   (default: the folder that contains autoresearch/)
    python autoresearch/tools/install_agents.py --check                  exit 1 when an installed copy differs, is missing or is not ours, or a definition fails the lint
    python autoresearch/tools/install_agents.py --selftest

The lint (also run before every copy, a failing definition is not installed) keeps what does not belong into an agent out of it: an agent is a short, stable,
English role text that other sessions load as an instruction. It checks: the frontmatter (name = file name, description, a known tools list), ASCII text only (no
pasted notes or chat words), no absolute or personal paths (C:\\, /mnt/, ~/, a mail address), no person's name (the role says "the person"), no merge or edit
markers or TODOs, no date or run id that goes stale, no removed role name, every `autoresearch/...` path and every tool script it names exists, and a size limit.
"""
import argparse
import filecmp
import os
import re
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import project as P  # noqa: E402

KNOWN_TOOLS = {"Read", "Write", "Edit", "Glob", "Grep", "Bash", "PowerShell", "WebSearch", "WebFetch"}
REMOVED = ("ar-scout", "ar-summarizer", "ar-hypothesis", "ar-referee", "ar-reviewer")
MAX_CHARS = 7000
BAD = [
    (re.compile(r"[A-Za-z]:\\|/mnt/[a-z]/|/home/|~/|\b[\w.-]+@[\w-]+\.\w+"), "an absolute or personal path or a mail address"),
    (re.compile(r"<<<<<<<|>>>>>>>|^=======$|\bTODO\b|\bFIXME\b|\bXXX\b", re.M), "a merge or edit marker, a TODO"),
    (re.compile(r"\b20\d\d-\d\d-\d\d\b|\b[tE]\d+\.[A-Za-z0-9_]+\b"), "a date or a run id (state goes stale: it belongs in a record)"),
    (re.compile(r"\bMirko\b"), "a person's name (write: the person)"),
]


def lint(path, ar_root=None):
    """list of problems of one agent definition (empty = fine)"""
    ar_root = ar_root or P.AR
    raw = open(path, "rb").read()
    problems = []
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as e:
        problems.append(f"non-ASCII text at byte {e.start}: agents are English, no pasted notes")
        text = raw.decode("utf-8", errors="replace")
    text = text.replace("\r\n", "\n")
    m = re.match(r"---\n(.*?)\n---\n", text, re.S)
    stem = os.path.splitext(os.path.basename(path))[0]
    if not m:
        problems.append("no frontmatter (--- name / description / tools ---)")
        body = text
    else:
        front = {}
        for line in m.group(1).split("\n"):
            k, _, v = line.partition(":")
            front[k.strip()] = v.strip().strip('"')
        body = text[m.end():]
        if front.get("name") != stem:
            problems.append(f"name '{front.get('name')}' is not the file name '{stem}'")
        if len(front.get("description", "")) < 20:
            problems.append("description missing or too short")
        tools = [t.strip() for t in front.get("tools", "").split(",") if t.strip()]
        if not tools or any(t not in KNOWN_TOOLS for t in tools):
            problems.append(f"tools list empty or unknown: {front.get('tools')}")
    for rx, why in BAD:
        hit = rx.search(text)
        if hit:
            problems.append(f"{why}: '{hit.group(0)}'")
    for name in REMOVED:
        if name in text:
            problems.append(f"names the removed role {name}")
    if len(text) > MAX_CHARS:
        problems.append(f"{len(text)} characters: an agent stays under {MAX_CHARS} (move detail into a template or a prompt file)")
    repo = os.path.dirname(ar_root)
    for ref in set(re.findall(r"`(autoresearch/[\w./-]+)`", text)):
        if not os.path.exists(os.path.join(repo, ref)):
            problems.append(f"names {ref}, which does not exist")
    scripts = {d: set(os.listdir(os.path.join(ar_root, d))) for d in ("tools", "Evaluation", "Training") if os.path.isdir(os.path.join(ar_root, d))}
    for py in set(re.findall(r"`([\w]+\.py)", text)):
        if not any(py in files for files in scripts.values()):
            problems.append(f"names the script {py}, which is in none of tools/, Evaluation/, Training/")
    return problems


def selftest():
    tmp = tempfile.mkdtemp()
    try:
        ar = os.path.join(tmp, "autoresearch")
        os.makedirs(os.path.join(ar, "tools"))
        open(os.path.join(ar, "tools", "run.py"), "w").write("")
        open(os.path.join(ar, "METHOD.md"), "w").write("")
        good = '---\nname: ar-x\ndescription: "A role that does one thing well enough."\ntools: Read, Grep\n---\nUse `run.py` and read `autoresearch/METHOD.md`. The person decides.\n'

        def put(text, name="ar-x.md", binary=False):
            p = os.path.join(tmp, name)
            open(p, "wb" if binary else "w", newline="").write(text)
            return p
        assert lint(put(good), ar) == [], lint(put(good), ar)
        assert lint(put(good.replace("\n", "\r\n")), ar) == [], "CRLF files are fine"
        for text, word in ((good.replace("name: ar-x", "name: ar-y"), "not the file name"), (good.replace("tools: Read, Grep", "tools: Read, Fly"), "unknown"),
                           (good.replace("Use", "Gr\u00fc\u00dfe: use"), "non-ASCII"), (good + "see C:\\Users\\x\\notes.txt\n", "absolute"), (good + "mail me@web.de\n", "absolute"),
                           (good + "Mirko wants this\n", "person's name"), (good + "TODO later\n", "marker"), (good + "since 2026-11-01\n", "date or a run id"), (good + "run t10.dual_s0\n", "date or a run id"),
                           (good + "ask ar-scout\n", "removed role"), (good + "see `autoresearch/gone.md`\n", "does not exist"), (good + "call `nothing.py`\n", "none of tools"),
                           (good + "x" * 7000, "characters"), ("no frontmatter here\n", "no frontmatter")):
            got = lint(put(text), ar)
            assert any(word in g for g in got), (word, got)
        open(os.path.join(tmp, "bad.md"), "wb").write("---\nname: bad\n---\n\xe4".encode("latin-1"))
        assert any("non-ASCII" in g for g in lint(os.path.join(tmp, "bad.md"), ar)), "a file that is not UTF-8 is reported, not raised"
        # the real definitions of this repository pass
        real = os.path.join(P.AR, "agents")
        for f in sorted(os.listdir(real)):
            if f.endswith(".md"):
                assert lint(os.path.join(real, f)) == [], (f, lint(os.path.join(real, f)))
        print("install_agents selftest passed")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workspace", default=os.path.dirname(P.AR))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    src = os.path.join(P.AR, "agents")
    dst = os.path.join(a.workspace, ".claude", "agents")
    ours = sorted(f for f in os.listdir(src) if f.endswith(".md"))
    found = {f: lint(os.path.join(src, f)) for f in ours}
    lint_bad = [f"{f}: {p}" for f, ps in found.items() for p in ps]
    if a.check:
        there = sorted(f for f in os.listdir(dst) if f.endswith(".md")) if os.path.isdir(dst) else []
        bad = lint_bad + [f"{f}: missing" for f in ours if f not in there] + [f"{f}: differs" for f in ours if f in there and not filecmp.cmp(os.path.join(src, f), os.path.join(dst, f), shallow=False)]
        bad += [f"{f}: installed but not in autoresearch/agents" for f in there if f not in ours]
        print("\n".join(bad) if bad else f"{len(ours)} agents installed, in step and clean")
        return 1 if bad else 0
    if lint_bad:
        print("not installed, fix the definitions first:\n  " + "\n  ".join(lint_bad))
        return 1
    os.makedirs(dst, exist_ok=True)
    for f in ours:
        shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
    print(f"{len(ours)} agents copied to {dst}")


if __name__ == "__main__":
    sys.exit(main())
