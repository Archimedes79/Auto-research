"""The Claude Code hook that guards the protected files (PreToolUse on Write, Edit, MultiEdit, NotebookEdit).

    .claude/settings.json: {"hooks": {"PreToolUse": [{"matcher": "Write|Edit|MultiEdit|NotebookEdit",
                             "hooks": [{"type": "command", "command": "python3 autoresearch/research/hook.py"}]}]}}

Reads the event from stdin; exit 2 with the reason on stderr refuses the edit, exit 0 lets it through. The protected paths and the grants are
research/gates.py. A file outside a research checkout is not this hook's business. When the hook itself fails, an edit of a file whose name
is one of the gate files is refused, every other edit goes through (a broken hook must not stop all work, nor open the gates)."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GATE_NAMES = {"problem.json", "TEST_LOG.md", "settings.json", "settings.local.json", "hook.py", "gates.py"}
TOOLS = {"Write": "file_path", "Edit": "file_path", "MultiEdit": "file_path", "NotebookEdit": "notebook_path"}


def to_local(path: str) -> str:
    """a Windows path seen from WSL and the other way round: the form this process can open"""
    p = path.replace("\\", "/")
    if os.name != "nt" and len(p) > 2 and p[1] == ":":
        return f"/mnt/{p[0].lower()}{p[2:]}"
    if os.name == "nt" and p.startswith("/mnt/") and len(p) > 6 and p[6] == "/":
        return f"{p[5].upper()}:{p[6:]}"
    return path


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
    key = TOOLS.get(event.get("tool_name", ""))
    raw = (event.get("tool_input") or {}).get(key or "", "") if key else ""
    if not raw:
        return ""
    path = to_local(raw)
    if not os.path.isabs(path):
        path = os.path.join(event.get("cwd") or os.getcwd(), path)
    root = checkout_of(path)
    if not root:
        return ""
    from research import gates
    return gates.protected_refusal(os.path.relpath(os.path.abspath(path), root), root)


def main() -> int:
    event: dict = {}
    try:
        event = json.load(sys.stdin)
        reason = verdict(event)
    except Exception as e:                                             # noqa: BLE001 - the hook must answer
        key = TOOLS.get(event.get("tool_name", "")) if isinstance(event, dict) else None
        name = os.path.basename(str(((event.get("tool_input") or {}) if isinstance(event, dict) else {}).get(key or "", "")).replace("\\", "/"))
        reason = f"the gate hook failed ({type(e).__name__}: {e}); edits of the gate files are refused until it works" if name in GATE_NAMES else ""
    if reason:
        print(reason, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
