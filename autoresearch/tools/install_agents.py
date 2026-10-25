"""Copy the agent definitions (autoresearch/agents/*.md) into the workspace's .claude/agents/ so Claude Code sessions started in the workspace
can spawn them by name (ar-scout, ar-referee, ...). Run again after editing an agent. Standard library only.

    python autoresearch/tools/install_agents.py [--workspace <folder>]   (default: the folder that contains autoresearch/)
"""
import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import project as P  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workspace", default=os.path.dirname(P.AR))
    a = ap.parse_args()
    dst = os.path.join(a.workspace, ".claude", "agents")
    os.makedirs(dst, exist_ok=True)
    n = 0
    for f in sorted(os.listdir(os.path.join(P.AR, "agents"))):
        if f.endswith(".md"):
            shutil.copy2(os.path.join(P.AR, "agents", f), os.path.join(dst, f))
            n += 1
    print(f"{n} agents copied to {dst}")


if __name__ == "__main__":
    main()
