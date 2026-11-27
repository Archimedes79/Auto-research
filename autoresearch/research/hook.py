"""The Claude Code hook that guards the protected files (PreToolUse on Write, Edit, MultiEdit, NotebookEdit).

    .claude/settings.json: {"hooks": {"PreToolUse": [{"matcher": "Write|Edit|MultiEdit|NotebookEdit",
                             "hooks": [{"type": "command", "command": "python3 autoresearch/research/hook.py"}]}]}}

Reads the event from stdin; exit 2 with the reason on stderr refuses the edit, exit 0 lets it through. The protected paths and the grants are
research/gates.py; a path is judged without its NTFS stream name (x::$DATA writes x). Every path inside the machine's state folder (the
folder of the machine file: grants, holds, GPU slots) is refused: only the research commands write there. Any other file outside a research
checkout is not this hook's business. When the hook itself fails (a gate module that cannot be imported, a broken grants file), it judges by
a static list without research/ (`fallback`): the gate files, champions/, every problem's problem.json and TEST_LOG.md, the settings and the
default state folder are refused, everything else goes through (a broken hook must not stop all work, nor open the gates)."""
from __future__ import annotations

import fnmatch
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STATIC_PROTECTED = (".claude/settings.json", ".claude/settings.local.json", "autoresearch/research/hook.py", "autoresearch/research/gates.py",
                    "autoresearch/research/paths.py", "autoresearch/research/__init__.py", "autoresearch/evaluation/defaults.py", "champions/*",
                    "problems/*/test_log.md", "problems/*/problem.json")     # gates.ROOT_PROTECTED and more, lower case, for a broken gate module
DEFAULT_STATE = os.path.expanduser("~/.research")
TOOLS ={"Write": "file_path", "Edit": "file_path", "MultiEdit": "file_path", "NotebookEdit": "notebook_path"}
MCP_WRITERS = ("write_file", "edit_block", "move_file", "write_pdf")             # file tools of MCP servers (mcp__<server>__<tool>)
PATH_KEYS = ("file_path", "notebook_path", "path", "source", "destination", "outputPath")


def targets(event: dict) -> list:
    """the paths an edit event would write"""
    tool, inp = event.get("tool_name", ""), event.get("tool_input") or {}
    if tool in TOOLS:
        return [inp.get(TOOLS[tool], "")] if inp.get(TOOLS[tool]) else []
    if tool.startswith("mcp__") and tool.rsplit("__", 1)[-1] in MCP_WRITERS:
        return [inp[k] for k in PATH_KEYS if isinstance(inp.get(k), str) and inp[k]]
    return []


def to_local(path: str) -> str:
    """a Windows path seen from WSL and the other way round: the form this process can open (C:\\x -> /mnt/c/x,
    \\\\wsl.localhost\\<distro>\\home\\x -> /home/x)"""
    p = path.replace("\\", "/")
    if os.name != "nt" and len(p) > 2 and p[1] == ":":
        return f"/mnt/{p[0].lower()}{p[2:]}"
    if os.name != "nt" and p.lower().startswith(("//wsl.localhost/", "//wsl$/")):
        return "/" + (p.split("/", 4)[4] if p.count("/") >= 4 else "")
    if os.name == "nt" and p.startswith("/mnt/") and len(p) > 6 and p[6] == "/":
        return f"{p[5].upper()}:{p[6:]}"
    return path


def strip_stream(path: str) -> str:
    """the file an NTFS stream name writes: everything from a ':' in the last part of the path on is dropped (beyond a drive letter)"""
    p = path.replace("\\", "/")
    drive = 2 if len(p) > 1 and p[0].isalpha() and p[1] == ":" else 0
    head, sep, name = p[drive:].rpartition("/")
    return p[:drive] + head + sep + name.split(":", 1)[0]


def inside(path: str, folder: str) -> bool:
    """does `path` lie in `folder`? Without regard to case"""
    return (os.path.realpath(path).lower() + os.sep).startswith(os.path.realpath(folder).lower() + os.sep)


def state_folder() -> str:
    """the machine's state folder: the folder of the machine file"""
    from research.paths import machine_file
    return os.path.dirname(machine_file())


def resolve(raw: str, event: dict) -> tuple:
    """(the absolute local path a target names, without a stream name; its research checkout or ''; the path relative to that checkout)"""
    path = to_local(strip_stream(raw))
    if not os.path.isabs(path):
        path = os.path.join(to_local(event.get("cwd") or os.getcwd()), path)
    root = checkout_of(path)
    return path, root, os.path.relpath(os.path.abspath(path), root) if root else ""


def checkout_of(path: str) -> str:
    """the research checkout that holds `path` (a folder with autoresearch/research/ and problems/), '' if none"""
    d = os.path.dirname(os.path.abspath(path))
    while True:
        if os.path.isdir(os.path.join(d, "autoresearch", "research")) and os.path.isdir(os.path.join(d, "problems")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            return ""
        d = parent


def verdict(event: dict) -> str:
    """'' = allow, else the reason of the refusal"""
    from research import gates
    for raw in targets(event):
        path, root, rel = resolve(raw, event)
        if inside(path, state_folder()):
            return f"{raw} lies in the machine's state folder (grants, holds, GPU slots): only the research commands write there (`research grant`)"
        reason = gates.protected_refusal(rel, root) if root else ""
        if reason:
            return reason
    return ""


def fallback(event: dict, error: Exception) -> str:
    """after a failure of the hook: the static list, no import of research/ (a gate module may be what broke); everything else goes through"""
    for raw in targets(event):
        path, root, rel = resolve(str(raw), event)
        rel = rel.replace("\\", "/").lower()
        if inside(path, DEFAULT_STATE) or (root and any(fnmatch.fnmatchcase(rel, pat) for pat in STATIC_PROTECTED)):
            return (f"the gate hook failed ({type(error).__name__}: {error}); while it is broken {raw} is refused, as no grant can be checked. "
                    "A broken gate module is repaired through git (`git checkout -- <file>`)")
    return ""


def main() -> int:
    event: dict = {}
    try:
        event = json.load(sys.stdin)
        reason = verdict(event)
    except Exception as e:                                             # noqa: BLE001 - the hook must answer
        try:
            reason = fallback(event, e) if isinstance(event, dict) else ""
        except Exception:                                              # noqa: BLE001 - the event names no path that can be read
            reason = ""
    if reason:
        print(reason, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
