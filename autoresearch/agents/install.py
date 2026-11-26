"""Copy the agent roles (ar-*.md) into the workspace's .claude/agents/, where Claude Code finds them; --check exits 1 when the copies differ.

    python autoresearch/agents/install.py [--check]
A role text stays short, stable and general: no paths of one machine, no names, dates or run ids (checked here)."""
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join(os.path.dirname(os.path.dirname(HERE)), ".claude", "agents")
LINT = re.compile(r"(/mnt/[a-z]/|[A-Z]:\|/home/|\b20\d\d-\d\d-\d\d\b|\bE\d{2,}\.[a-z])")


def main(argv):
    roles = sorted(f for f in os.listdir(HERE) if f.startswith("ar-") and f.endswith(".md"))
    bad = [f"{f}: {m.group(0)!r}" for f in roles for m in [LINT.search(open(os.path.join(HERE, f), encoding="utf-8").read())] if m]
    if bad:
        print("a role text names something of one machine, day or run:\n  " + "\n  ".join(bad), file=sys.stderr)
        return 1
    differ = [f for f in roles if not os.path.isfile(os.path.join(DEST, f)) or open(os.path.join(DEST, f), "rb").read() != open(os.path.join(HERE, f), "rb").read()]
    if "--check" in argv:
        print("copies current" if not differ else "differ: " + ", ".join(differ))
        return 1 if differ else 0
    os.makedirs(DEST, exist_ok=True)
    for f in differ:
        shutil.copyfile(os.path.join(HERE, f), os.path.join(DEST, f))
    print(f"installed {len(differ)} of {len(roles)} roles into {DEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
